import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glp_hyperparameter_optimization import glp_model as gm
from glp_hyperparameter_optimization.search_config import (
    GLPSearchConfig,
    GLPSearchConfigError,
)


def _fake_ctx(n=3, *, psi=True, sur=1, noc=1, ss=None):
    """A lightweight GLPContext stand-in (no covbayesvar required)."""
    if ss is None:
        ss = np.array([1.0, 2.0, 0.5, 1.5, 0.8, 1.2, 0.9][:n], dtype=float)
    ss = np.asarray(ss, dtype=float).reshape(n, 1)
    psi_lo = np.ravel(ss) * 0.01
    psi_hi = np.ravel(ss) * 100.0
    return SimpleNamespace(
        n=n,
        mn={"psi": 1 if psi else 0},
        sur=sur,
        noc=noc,
        SS=ss,
        MIN={"lambda": 1.0e-4, "theta": 1.0e-4, "miu": 1.0e-4, "psi": psi_lo},
        MAX={"lambda": 5.0, "theta": 50.0, "miu": 50.0, "psi": psi_hi},
    )


def _full_params(ctx):
    params = {"lam": 0.2, "theta": 1.0, "miu": 1.0}
    for i in range(ctx.n):
        params[f"psi_{i}"] = float(np.ravel(ctx.SS)[i])
    return params


# --------------------------------------------------------------------------- #
# Full search dimensions.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [3, 7])
def test_full_search_dimension(n):
    ctx = _fake_ctx(n=n)
    resolved = GLPSearchConfig.legacy_full().resolve(ctx)
    # lambda + n psi + theta + miu
    assert resolved.search_dimension == n + 3
    assert resolved.config.mode == "full"


def test_reduced_three_parameter_search():
    ctx = _fake_ctx(n=5)
    resolved = GLPSearchConfig.reduced_lambda_theta_miu().resolve(ctx)
    assert resolved.search_dimension == 3
    assert resolved.optimized_names == ("lam", "theta", "miu")
    assert resolved.config.mode == "reduced"
    # psi genuinely excluded from the optimizer space.
    space = resolved.mango_param_space()
    assert set(space) == {"lam", "theta", "miu"}
    assert not any(key.startswith("psi") for key in space)


# --------------------------------------------------------------------------- #
# Deterministic ordering.
# --------------------------------------------------------------------------- #
def test_deterministic_parameter_order():
    ctx = _fake_ctx(n=3)
    resolved = GLPSearchConfig.legacy_full().resolve(ctx)
    assert resolved.optimized_names == ("lam", "psi_0", "psi_1", "psi_2", "theta", "miu")


def test_deterministic_parameter_order_log_multiplier():
    ctx = _fake_ctx(n=2)
    resolved = GLPSearchConfig.legacy_full(psi_parameterization="ss_log_multiplier").resolve(ctx)
    assert resolved.optimized_names == (
        "lam",
        "psi_log_multiplier_0",
        "psi_log_multiplier_1",
        "theta",
        "miu",
    )


# --------------------------------------------------------------------------- #
# Fixed-psi conversion.
# --------------------------------------------------------------------------- #
def test_fixed_psi_conversion_context_ss():
    ctx = _fake_ctx(n=3)
    resolved = GLPSearchConfig.reduced_lambda_theta_miu().resolve(ctx)
    vec = resolved.to_natural({"lam": 0.2, "theta": 1.0, "miu": 1.0}, ctx)
    # vector = [lambda, psi_0..psi_2, theta, miu]
    assert vec.shape == (ctx.n + 3,)
    np.testing.assert_allclose(vec[1 : 1 + ctx.n], np.ravel(ctx.SS))
    assert vec[0] == pytest.approx(0.2)
    assert vec[-2] == pytest.approx(1.0)
    assert vec[-1] == pytest.approx(1.0)


def test_fixed_psi_conversion_supplied():
    ctx = _fake_ctx(n=3)
    supplied = [1.1, 1.9, 0.6]
    config = GLPSearchConfig.reduced_lambda_theta_miu(
        fixed_psi_source="supplied", fixed_psi_values=supplied
    )
    resolved = config.resolve(ctx)
    vec = resolved.to_natural({"lam": 0.3, "theta": 2.0, "miu": 1.5}, ctx)
    np.testing.assert_allclose(vec[1 : 1 + ctx.n], supplied)


# --------------------------------------------------------------------------- #
# Supplied-psi validation.
# --------------------------------------------------------------------------- #
def test_supplied_psi_requires_values():
    with pytest.raises(GLPSearchConfigError, match="requires fixed_psi_values"):
        GLPSearchConfig(optimize_psi=False, fixed_psi_source="supplied")


def test_supplied_psi_rejects_nonpositive():
    with pytest.raises(GLPSearchConfigError, match="strictly positive"):
        GLPSearchConfig(
            optimize_psi=False, fixed_psi_source="supplied", fixed_psi_values=[1.0, -1.0]
        )


def test_supplied_psi_rejects_nonfinite():
    with pytest.raises(GLPSearchConfigError, match="finite"):
        GLPSearchConfig(
            optimize_psi=False, fixed_psi_source="supplied", fixed_psi_values=[1.0, np.inf]
        )


def test_supplied_psi_wrong_length_fails_on_resolve():
    ctx = _fake_ctx(n=3)
    config = GLPSearchConfig.reduced_lambda_theta_miu(
        fixed_psi_source="supplied", fixed_psi_values=[1.0, 2.0]
    )
    with pytest.raises(GLPSearchConfigError, match="length 2 but the context has 3"):
        config.resolve(ctx)


def test_optimize_psi_with_fixed_source_conflicts():
    with pytest.raises(GLPSearchConfigError, match="fixed_psi_source must be None"):
        GLPSearchConfig(optimize_psi=True, fixed_psi_source="context_ss")


# --------------------------------------------------------------------------- #
# Inactive theta / miu behavior.
# --------------------------------------------------------------------------- #
def test_optimize_theta_inactive_prior_raises():
    ctx = _fake_ctx(n=3, sur=0)
    with pytest.raises(GLPSearchConfigError, match="theta.*inactive"):
        GLPSearchConfig.legacy_full().resolve(ctx)


def test_optimize_miu_inactive_prior_raises():
    ctx = _fake_ctx(n=3, noc=0)
    with pytest.raises(GLPSearchConfigError, match="miu.*inactive"):
        GLPSearchConfig.legacy_full().resolve(ctx)


def test_inactive_theta_uses_placeholder_and_is_excluded_from_search():
    ctx = _fake_ctx(n=3, sur=0)
    # Do not optimize theta (its prior is inactive); psi fixed to keep it simple.
    config = GLPSearchConfig(
        optimize_lambda=True,
        optimize_theta=False,
        optimize_miu=True,
        optimize_psi=False,
        fixed_psi_source="context_ss",
    )
    resolved = config.resolve(ctx)
    assert "theta" not in resolved.optimized_names
    vec = resolved.to_natural({"lam": 0.2, "miu": 1.0}, ctx)
    # theta placeholder is a valid in-bounds default.
    assert 1.0e-4 < vec[-2] < 50.0


def test_optimize_psi_inactive_context_raises():
    ctx = _fake_ctx(n=3, psi=False)
    with pytest.raises(GLPSearchConfigError, match="psi is not estimated"):
        GLPSearchConfig.legacy_full().resolve(ctx)


# --------------------------------------------------------------------------- #
# Empty search space, bounds validation.
# --------------------------------------------------------------------------- #
def test_empty_search_space_rejected():
    with pytest.raises(GLPSearchConfigError, match="at least one parameter"):
        GLPSearchConfig(
            optimize_lambda=False,
            optimize_theta=False,
            optimize_miu=False,
            optimize_psi=False,
            fixed_psi_source="context_ss",
        )


def test_invalid_bounds_rejected():
    with pytest.raises(GLPSearchConfigError, match="lower < upper"):
        GLPSearchConfig(bounds={"lambda": (5.0, 1.0), "theta": (1e-4, 50.0), "miu": (1e-4, 50.0)})


def test_non_optimized_lambda_requires_initial_value():
    ctx = _fake_ctx(n=3)
    config = GLPSearchConfig(
        optimize_lambda=False,
        optimize_theta=True,
        optimize_miu=True,
        optimize_psi=False,
        fixed_psi_source="context_ss",
    )
    with pytest.raises(GLPSearchConfigError, match="fixed lambda"):
        config.resolve(ctx)


# --------------------------------------------------------------------------- #
# Metadata round trip and provenance.
# --------------------------------------------------------------------------- #
def test_config_dict_round_trip():
    config = GLPSearchConfig.reduced_lambda_theta_miu(
        fixed_psi_source="supplied", fixed_psi_values=[1.0, 2.0, 0.5]
    )
    restored = GLPSearchConfig.from_dict(config.to_dict())
    assert restored == config


def test_full_config_dict_round_trip():
    config = GLPSearchConfig.legacy_full(psi_parameterization="ss_log_multiplier")
    restored = GLPSearchConfig.from_dict(config.to_dict())
    assert restored == config


def test_metadata_distinguishes_full_and_reduced_without_command_strings():
    ctx = _fake_ctx(n=3)
    full_meta = GLPSearchConfig.legacy_full().resolve(ctx).metadata()
    reduced_meta = GLPSearchConfig.reduced_lambda_theta_miu().resolve(ctx).metadata()

    assert full_meta["mode"] == "full"
    assert reduced_meta["mode"] == "reduced"
    assert full_meta["search_dimension"] == 6
    assert reduced_meta["search_dimension"] == 3
    assert "psi" not in reduced_meta["fixed_parameters"] or reduced_meta["fixed_parameters"]["psi"]
    # Reduced records the fixed psi and its source for provenance.
    assert reduced_meta["fixed_parameters"]["psi_source"] == "context_ss"
    assert "psi" in reduced_meta["fixed_parameters"]


def test_metadata_records_optimized_and_bounds():
    ctx = _fake_ctx(n=2)
    meta = GLPSearchConfig.legacy_full().resolve(ctx).metadata()
    assert meta["optimized_parameters"] == ["lam", "psi_0", "psi_1", "theta", "miu"]
    assert set(meta["natural_bounds"]) == set(meta["optimized_parameters"])
    assert set(meta["transformed_bounds"]) == set(meta["optimized_parameters"])


# --------------------------------------------------------------------------- #
# Legacy behavior equivalence.
# --------------------------------------------------------------------------- #
def test_legacy_full_matches_existing_params_to_natural():
    ctx = _fake_ctx(n=3)
    params = _full_params(ctx)
    resolved = GLPSearchConfig.legacy_full().resolve(ctx)
    from_config = resolved.to_natural(params, ctx)
    from_legacy = gm._params_to_natural(params, ctx)
    np.testing.assert_allclose(from_config, from_legacy)


def test_legacy_full_param_space_matches_existing_make_param_space_keys():
    ctx = _fake_ctx(n=3)
    resolved = GLPSearchConfig.legacy_full().resolve(ctx)
    config_space = resolved.mango_param_space()
    legacy_space = gm.make_param_space(ctx)
    assert set(config_space) == set(legacy_space)
