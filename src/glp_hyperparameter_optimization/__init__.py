"""Giannone-Lenza-Primiceri (2015) prior-selection vs. Mango hyperparameter study.

A workflow that is intentionally separate from the Schorfheide-Song MF-VAR
workflow in ``paper_hyperparameter_optimization``. It compares GLP's native
marginal-likelihood prior selection against three Bayesian-optimization (Mango)
strategies ported from ``MBFVAR`` and adapted to the ``covbayesvar`` GLP BVAR.
"""

from .config import GLP_PARAM_SPACE_BOUNDS, MODEL_SIZE_CODES, SERIES_SPECS, model_series

__all__ = ["SERIES_SPECS", "MODEL_SIZE_CODES", "GLP_PARAM_SPACE_BOUNDS", "model_series"]
