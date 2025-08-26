from Function_VerticalFrequencyWeighting import (
    analyze_vibration,
    local_sensitivity_from_file,
    tornado_grid_elasticity,
    plot_sweep_multi_metric,  
)

# ====================================================
# A) Apply vertical frequency weighting + analyze weighted and unweighted metrics
# CHOOSE WEIGHTING CURVE PARAMETERS HERE
analyze_vibration(
    file_path="results_500mc_trial_matrix.csv",
    low_gain=0.4,
    f_low=0.5,
    f_mid_start=2.0,
    f_mid_end=5.0,
    f_flat_end=16.0
)

# ====================================================
# B) Local derivative-based sensitivity using central finite differences & elasticities at baseline
# CHOOSE BASELINE WEIGHTING CURVE PARAMETERS FOR SENSITIVITY ANALYSIS HERE
base = dict(low_gain=0.4, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0)

# Elasticity & Partial derivatives
# CHOOSE % CHANGE IN PARAMETER FOR FINITE DIFFERENCE SENSITIVITY ANALYSIS HERE
df_elast, df_dydp = local_sensitivity_from_file(
    file_path="results_500mc_trial_matrix.csv",
    trial_index=0,
    fs=100,
    base_params=base,
    rel_step=0.05 # 0.05 = 5% change in parameter
    # metrics_to_track optional — default is all 7 weighted metrics
)

# Rename columns for pretty labels before plotting
pretty_map = {
    "Peak_weighted": "Peak",
    "RMS_weighted": "RMS",
    "MTVV_weighted": "MTVV",
    "MTVV_sqrt2_weighted": "MTVV*sqrt(2)",
    "CF_weighted": "CF",
    "VDV_weighted": "VDV",
    "R_weighted": "R",
}
df_elast_pretty = df_elast.rename(columns=pretty_map)

# Tornado grid (one figure with subplots)
tornado_grid_elasticity(
    df_elasticity=df_elast_pretty,
    metrics_to_plot=["Peak","RMS","MTVV","MTVV*sqrt(2)","CF","VDV","R"],
    ncols=3,
    sharex=True,
    annotate=True,
    title="Local Sensitivity (Elasticity) — All Metrics"
)

# ====================================================
# C) Range sweep (one-at-a-time) → % change vs baseline
# - show how metrics change when you move a parameter across a user-defined range with others fixed
# Plot sweep to see % change in metrics as one weighting curve parameter is swept
# CHOOSE PARAMETER, RANGE, AND METRICS TO PLOT HERE
plot_sweep_multi_metric(
    file_path="results_500mc_trial_matrix.csv",
    trial_index=0,
    fs=100,
    base_params=base,
    param_name="f_mid_end", # CHOOSE PARAMETER TO SWEEP FOR PLOTTING
    param_range=(4.0, 6.0), # CHOOSE RANGE TO SWEEP FOR PLOTTING
    samples=11, # Controls how many points are taken in sweep (how fine the resolution is) --> number of equally spaced points in the range
    metrics=["RMS_weighted", "VDV_weighted", "R_weighted"], # CHOOSE METRICS TO PLOT
    # metrics=None, # Default is all 7 weighted metrics
    title="% Change in metrics w.r.t. f_mid_end",
)