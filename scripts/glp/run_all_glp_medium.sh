# Predictive-mean RMSE objective runs (beta-averaged RMSE selection).
#
# Final forecast density: paper-style 20,000 retained draws with 5,000 burn-in
# via --mcmc-draws / --mcmc-discard.
#
# Inner RMSE selection objective: --optimization-n-obj-draws controls only the
# beta-averaged point forecast used while tuning hyperparameters. It is kept far
# below 20,000 because it is evaluated repeatedly across horizons, origins, and
# Mango candidates; using 20,000 there would make the run prohibitively slow.

nohup python3 -u run_glp_all.py \
	--stages paper,mango_mdd,mango_rmse,mango_rmse_random,compare \
	--output-root outputs/glp/all_small_predictive_mean \
	--model-size small \
	--variables GDP \
	--mcmc-draws 20000 \
	--mcmc-discard 5000 \
	--optimization-eval-horizons-quarters 1,2,4,8 \
	--optimization-n-eval 12 \
	--optimization-n-obj-draws 2000 \
	--optimization-random-seed 123 \
	--optimization-init-points 10 \
	--optimization-iterations 14 \
	--optimization-njobs 4 \
	--n-workers 15 \
	--per-origin-selection \
	> output_small_predictive_mean.log 2>&1 &

nohup python3 -u run_glp_all.py \
	--stages paper,mango_mdd,mango_rmse,mango_rmse_random,compare \
	--output-root outputs/glp/all_medium_predictive_mean \
	--model-size medium \
	--variables GDP \
	--mcmc-draws 20000 \
	--mcmc-discard 5000 \
	--optimization-eval-horizons-quarters 1,2,4,8 \
	--optimization-n-eval 10 \
	--optimization-n-obj-draws 2000 \
	--optimization-random-seed 123 \
	--optimization-init-points 9 \
	--optimization-iterations 12 \
	--optimization-njobs 4 \
	--n-workers 15 \
	--per-origin-selection \
	> output_medium_predictive_mean.log 2>&1 &

nohup python3 -u run_glp_all.py \
	--stages paper,mango_mdd,mango_rmse,mango_rmse_random,compare \
	--output-root outputs/glp/all_large_predictive_mean \
	--model-size large \
	--variables GDP \
	--mcmc-draws 20000 \
	--mcmc-discard 5000 \
	--optimization-eval-horizons-quarters 1,2,4,8 \
	--optimization-n-eval 8 \
	--optimization-n-obj-draws 2000 \
	--optimization-random-seed 123 \
	--optimization-init-points 8 \
	--optimization-iterations 10 \
	--optimization-njobs 4 \
	--n-workers 15 \
	--per-origin-selection \
	> output_large_predictive_mean.log 2>&1 &
