"""Shared reproducibility helpers for forecast experiments.

The experiment writers use this module to make every output self-describing:
the exact repository/dependency state, input-file hashes, runtime platform, and
deterministic child seeds are all derived in one place.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import subprocess
import sys
from typing import Iterable, Iterator

import numpy as np


AUDITED_MBFVAR_COMMIT = "5b06f93272cd6ebf370fbf2aac3b3573c7830493"
SCIENTIFIC_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "arm-mango",
    "bayesian-optimization",
    "covbayesvar",
    "MBFVAR",
)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 fingerprint for *path*."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(project_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip()


def repository_state(project_root: str | Path) -> dict[str, object]:
    """Describe the Git revision without failing outside a checkout."""
    root = Path(project_root).resolve()
    commit = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(root, "status", "--porcelain", "--untracked-files=no")
    return {
        "repository_commit": commit,
        "repository_dirty": None if status is None else bool(status),
    }


def _distribution_source(distribution: importlib.metadata.Distribution) -> dict[str, object] | None:
    text = distribution.read_text("direct_url.json")
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"direct_url_raw": text}
    return value


def dependency_state(
    names: Iterable[str] = SCIENTIFIC_DISTRIBUTIONS,
) -> tuple[dict[str, str | None], dict[str, object]]:
    """Return installed versions and PEP 610 source records."""
    versions: dict[str, str | None] = {}
    sources: dict[str, object] = {}
    for name in names:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
            continue
        versions[name] = distribution.version
        source = _distribution_source(distribution)
        if source is not None:
            sources[name] = source
    return versions, sources


def validate_mbfvar_revision(
    sources: dict[str, object],
    *,
    expected_commit: str = AUDITED_MBFVAR_COMMIT,
) -> None:
    """Fail unless installed MBFVAR provenance records the audited commit."""
    source = sources.get("MBFVAR") or sources.get("mbfvar")
    if not isinstance(source, dict):
        raise RuntimeError("Installed MBFVAR has no PEP 610 source record; install requirements.lock.")
    vcs_info = source.get("vcs_info")
    if not isinstance(vcs_info, dict):
        raise RuntimeError("Installed MBFVAR is not a verifiable VCS build; install requirements.lock.")
    commit = vcs_info.get("commit_id")
    if commit != expected_commit:
        raise RuntimeError(
            "Installed MBFVAR revision is incompatible with this experiment: "
            f"expected {expected_commit}, found {commit}. Install requirements.lock."
        )


def runtime_provenance(
    project_root: str | Path,
    *,
    data_paths: Iterable[str | Path] = (),
) -> dict[str, object]:
    """Build the common machine-readable provenance block."""
    versions, sources = dependency_state()
    if versions.get("MBFVAR") is not None:
        validate_mbfvar_revision(sources)
    fingerprints: dict[str, str] = {}
    for raw_path in data_paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Cannot fingerprint missing input data: {path}")
        fingerprints[str(path)] = sha256_file(path)
    return {
        **repository_state(project_root),
        "dependency_versions": versions,
        "dependency_sources": sources,
        "expected_mbfvar_commit": AUDITED_MBFVAR_COMMIT,
        "data_fingerprints_sha256": fingerprints,
        "platform": {
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }


def stable_child_seed(base_seed: int | None, *components: object) -> int | None:
    """Derive a stable uint32 child seed through ``SeedSequence``.

    Python's randomized ``hash`` is deliberately avoided. Components are
    encoded canonically, hashed, and supplied as SeedSequence entropy.
    """
    if base_seed is None:
        return None
    if not isinstance(base_seed, (int, np.integer)):
        raise TypeError("base_seed must be an integer or None.")
    payload = json.dumps(components, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    entropy = [int(base_seed), *np.frombuffer(digest, dtype=np.uint32).astype(int).tolist()]
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def stable_rng(base_seed: int | None, *components: object) -> np.random.Generator:
    """Return a Generator backed by a deterministic child stream."""
    seed = stable_child_seed(base_seed, *components)
    return np.random.default_rng(seed)


@contextmanager
def deterministic_rng_context(seed: int | None) -> Iterator[None]:
    """Temporarily make legacy global and fresh NumPy RNG use repeatable.

    Mango 1.6 draws candidates from the NumPy and Python global RNGs, while the
    audited MBFVAR revision creates fresh ``default_rng()`` instances without
    accepting a generator. This adapter makes both behaviors deterministic and
    restores all process-global state afterward. Callers must not overlap this
    context between threads; process workers are isolated and safe.

    Explicit ``default_rng(explicit_seed)`` calls keep their normal semantics.
    Unseeded calls receive successive child SeedSequences from *seed*.
    """
    if seed is None:
        yield
        return
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer or None.")

    numpy_state = np.random.get_state()
    python_state = random.getstate()
    original_default_rng = np.random.default_rng
    root_sequence = np.random.SeedSequence(int(seed))

    def seeded_default_rng(explicit_seed=None):
        if explicit_seed is not None:
            return original_default_rng(explicit_seed)
        return original_default_rng(root_sequence.spawn(1)[0])

    np.random.seed(int(seed) % (2**32))
    random.seed(int(seed))
    np.random.default_rng = seeded_default_rng
    try:
        yield
    finally:
        np.random.default_rng = original_default_rng
        np.random.set_state(numpy_state)
        random.setstate(python_state)
