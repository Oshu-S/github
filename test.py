from vertical_weighting_sensitivity import load_trial_from_csv, local_sensitivity, weight_and_metrics

# 1) Load one trial (uses fs=100 Hz by default, matching your dataset)
accel, fs = load_trial_from_csv("results_500mc_trial_matrix.csv", trial_index=0)

# 2) Define your baseline (your current ISO2631 straight-line approx)
base = dict(low_gain=0.4, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0)

# 3) Generate metrics and plots
metrics = weight_and_metrics(accel, fs=fs, **base, do_plots=True)


# 4) Run local (±5%) derivative-based sensitivity — choose the outputs you care about
df = local_sensitivity(
    acceleration=accel,
    fs=fs,
    base_params=base,
    rel_step=0.05,  # 5% symmetric finite-difference step
    metrics_to_track=["Peak_weighted","RMS_weighted", "MTVV_weighted", "MTVV_sqrt2_weighted", "CF_weighted", "VDV_weighted", "R_weighted"]
)

print(df.round(4))

