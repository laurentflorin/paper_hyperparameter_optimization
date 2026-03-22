# Explosive VAR Fix Documentation

## Problem Summary

When running `run_paper_hyperparameters.py` with the paper's hyperparameters `[0.09, 4.30, 1.0, 2.70, 4.30]`, the MBFVAR package frequently encounters explosive VAR draws during MCMC sampling, causing the estimation to fail.

## Root Cause

The issue stems from weak Minnesota prior shrinkage (λ1=0.09) combined with the MBFVAR package's explosive VAR handling:

1. **What are explosive VARs?**: A VAR model is "explosive" (unstable) when eigenvalues of its companion matrix have modulus ≥ 1, leading to nonstationary, divergent forecasts.

2. **MBFVAR's detection**: During Gibbs sampling, the package checks each drawn parameter for stability using `is_explosive()` in `/MBFVAR/_estimation.py`.

3. **Original bug**: When all `max_it_stable=1000` attempts fail to find non-explosive coefficients:
   - The code printed a warning and tried to `continue` the loop
   - At j=0, this caused an IndexError when accessing empty deques
   - At j>0, iterations were silently skipped, producing invalid results

4. **Why λ1=0.09 causes problems**:
   - λ1 controls overall tightness (shrinkage toward zero/random walk)
   - Small λ1 = weak prior = parameters can wander far from stable region
   - Makes explosive draws much more likely during MCMC sampling

## Solution

### 1. MBFVAR Package Fix

**File**: `/tmp/MBFVAR/MBFVAR/_estimation.py` (lines 853-868)

Changed the explosive VAR handling from:
```python
if attempts == max_it_stable:
    explosive_counter += 1
    print(f"Explosive VAR detected {explosive_counter} times.")
    m = 0
    if j == 0:
        j -= 1
    continue  # BUG: causes IndexError or silent failures
```

To:
```python
if attempts == max_it_stable:
    explosive_counter += 1
    print(f"Explosive VAR detected {explosive_counter} times.")
    raise ValueError(
        f"Failed to draw non-explosive VAR coefficients after {max_it_stable} attempts "
        f"at iteration j={j}, frequency block m={m}. "
        f"This typically indicates that the hyperparameters are too weak (lambda1 too small) "
        f"or the data characteristics make stable draws very unlikely. "
        f"Consider: (1) increasing lambda1 for stronger shrinkage, "
        f"(2) increasing max_it_stable, or (3) checking data for issues. "
        f"Total explosive draws so far: {explosive_counter}"
    )
```

**Benefits**:
- Clear error message explaining the problem
- Suggests concrete solutions
- Prevents silent failures and data corruption

### 2. Repository Configuration

**File**: `src/paper_hyperparameter_optimization/config.py`

Added `MAX_IT_STABLE = 10_000` configuration with documentation:
```python
# Explosive VAR handling
# The paper hyperparameters (especially λ1=0.09) can produce explosive VAR draws
# during MCMC sampling. The MBFVAR package attempts up to max_it_stable draws
# to find non-explosive coefficients. If all attempts fail, it raises an error.
#
# The default max_it_stable=1000 may be insufficient for weak priors like λ1=0.09.
# We increase this to 10000 to give more attempts at finding stable draws.
MAX_IT_STABLE = 10_000  # Increased from default 1000
```

**File**: `src/paper_hyperparameter_optimization/forecasting.py`

- Import `MAX_IT_STABLE` from config
- Add `max_it_stable` parameter to `run_recursive_experiment()`
- Pass it through to `model.fit(..., max_it_stable=...)`
- Include in task template and metadata

## Recommendations

### For Researchers Reproducing the Paper

1. **Use increased max_it_stable**: The default 10,000 should work for most origins
2. **Monitor explosive_counter**: Check model.explosive_counter after fitting
3. **Accept some failures**: With λ1=0.09, some origins may still fail
4. **Check failed_origins.csv**: Review any failed origins for patterns

### For Production Use

Consider these alternatives to the fixed paper hyperparameters:

1. **Stronger shrinkage**: Increase λ1 from 0.09 to 0.2-0.5
   - Reduces explosive draws
   - Still allows reasonable flexibility
   - May improve out-of-sample forecasts

2. **MDD-optimized hyperparameters**: Use `run_mango_mdd.py`
   - Schorfheide & Song (2015) recommend data-driven selection
   - Automatically balances fit vs. explosiveness
   - Different optima for different forecast origins

3. **RMSE-optimized hyperparameters**: Use `run_mango_rmse.py`
   - Directly optimizes forecast accuracy
   - Horizon-specific tuning available
   - More robust to explosive regions

## Testing the Fix

Run a quick test with paper hyperparameters:

```bash
# Test single origin (fast)
python scripts/run_paper_hyperparameters.py \
  --output-dir /tmp/test_explosive_fix \
  --start 1997-07-31 \
  --end 1997-07-31 \
  --fit-nsim 1000

# Check results
cat /tmp/test_explosive_fix/run_metadata.json
cat /tmp/test_explosive_fix/failed_origins.csv
```

Expected behavior:
- Either successful completion (explosive draws found but < 10,000 attempts)
- OR clear ValueError with actionable suggestions
- NO IndexError or silent failures

## References

- Schorfheide, F. & Song, D. (2015). *Real-Time Forecasting With a Mixed-Frequency VAR*. JBES 33(3), 366-380
- MBFVAR package: https://github.com/laurentflorin/MBFVAR
- Companion matrix stability: eigenvalues must have modulus < 1 for stationarity
