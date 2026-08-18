# Forecast variables vs. objective variables (mixed-frequency target subset)

The mixed-frequency BVAR (MBFVAR) hyperparameter objective previously conflated
two distinct ideas into a single `optimization_variables` list. That forced the
RMSE objective to require the full quarterly block and made a GDP-only objective
collapse the forecast state, which surfaced as a generic numerical penalty rather
than a real evaluation. The workflow now separates two concepts.

## The two-variable-set contract

- **`forecast_variables`** — the complete set required to construct a
  dimensionally valid mixed-frequency **state forecast**. For the current
  quarterly block this is `GDP, INVFIX, GOV`. The audited MBFVAR revision cannot
  forecast a reduced block safely, so the forecast state is always fit with the
  full block via `var_of_interest=forecast_variables`.
- **`objective_variables`** — the subset whose forecast errors enter the
  optimization **objective (loss)**. It may contain only `GDP` or any other valid
  subset of `forecast_variables`.

### Guarantees

- The forecast state is always built from `forecast_variables`; a GDP-only
  objective never reduces the state forecast to a single quarterly series.
- Loss extraction uses only `objective_variables`; errors from the other
  forecast-block variables never enter the objective.
- `objective_variables` must be a subset of `forecast_variables` (or otherwise
  resolvable in the generated forecast panel). Invalid names and non-subset
  requests fail descriptively **before** any expensive optimization begins.
- Horizon-specific scoring selects the correct quarterly forecast date.

## API and backward compatibility

Resolution is centralized in
[`resolve_forecast_objective_variables`](../src/paper_hyperparameter_optimization/forecasting.py):

- Existing calls using the legacy single `optimization_variables` argument remain
  valid and map to both concepts when dimensionally safe: the objective is the
  supplied subset while the forecast block expands to the full quarterly block.
  A legacy full-block call therefore behaves exactly as before.
- New calls may pass `forecast_variables` and/or `objective_variables`
  explicitly. Passing the legacy argument together with either explicit argument
  is rejected.
- For the marginal-data-density (`mango_mdd`) objective, `var_of_interest`
  reduces the fitted system, so the forecast block and objective subset coincide.

### CLI

`build_optimizer_parser` exposes:

- `--optimization-variables` — legacy objective subset (mapped as above).
- `--forecast-variables` — explicit forecast-state block (RMSE requires the full
  quarterly block).
- `--objective-variables` — explicit objective subset (e.g. `GDP`).

## Dependency pinning

The fix is repository-local: it fits the full forecast block and scores a subset
from the already-available quarterly posterior draws, so **no upstream MBFVAR
change is required**. The dependency is reproducibly pinned to commit
`5b06f93272cd6ebf370fbf2aac3b3573c7830493` in
[`requirements.txt`](../requirements.txt) and
[`requirements.lock`](../requirements.lock). Dimension errors are raised as
descriptive `ValueError`s and are therefore distinguishable from genuine
numerical failures (`np.linalg.LinAlgError`, `FloatingPointError`, etc.), which
are the only exceptions mapped to the optimizer penalty.

## Tests

See [`tests/test_forecast_objective_variables.py`](../tests/test_forecast_objective_variables.py),
which proves, using a lightweight fake MBFVAR model, that:

- `forecast_variables` contains the full quarterly block and `objective_variables`
  may contain only `GDP`;
- the objective fits the full forecast block while only GDP errors enter the loss;
- a GDP-only objective returns a finite score rather than the generic penalty;
- the legacy full-block call is unchanged;
- invalid/non-subset requests fail before fold building.
