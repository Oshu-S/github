import matplotlib.pyplot as plt

from Function_VerticalFrequencyWeighting import (
    analyze_vibration,
    local_sensitivity_from_file,
    tornado_grid_elasticity,
    plot_sweep_multi_metric,  
)
# ====================================================
# A) Apply vertical frequency weighting + analyze weighted and unweighted metrics
# CHOOSE WEIGHTING CURVE PARAMETERS HERE
[t, hist_w_base] = analyze_vibration(
    file_path="results_500mc_trial_matrix.csv",
    low_gain=0.40,
    f_low=0.5,
    f_mid_start=2.0,
    f_mid_end=5.0,
    f_flat_end=16.0
)

[t, hist_w_pert] = analyze_vibration(
    file_path="results_500mc_trial_matrix.csv",
    low_gain=0.4,
    f_low=0.5,
    f_mid_start=2.0,
    f_mid_end=5.0,
    f_flat_end=7.0
)

plt.plot(t, hist_w_base, label='Weighted Signal (Base)')
plt.plot(t, hist_w_pert, label='Weighted Signal (Perturbed)')
plt.legend()
plt.show()

# ====================================================
# B) Local derivative-based sensitivity using central finite differences & elasticities at baseline
# CHOOSE BASELINE WEIGHTING CURVE PARAMETERS FOR SENSITIVITY ANALYSIS HERE
base = dict(low_gain=0.4, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0)

# Elasticity & Partial derivatives
# CHOOSE % CHANGE IN PARAMETER FOR ELASTICITIES HERE
df_elast, df_dydp = local_sensitivity_from_file(
    file_path="results_500mc_trial_matrix.csv",
    trial_index=0,
    fs=100,
    base_params=base,
    rel_step=0.05 # ±0.05 = ±5% change in parameter
    # metrics_to_track optional — default is all 7 weighted metrics
)

# Tornado grid (one figure with subplots)
rel_step=0.05 # CHOOSE % CHANGE IN PARAMETER FOR ELASTICITIES HERE (should match above, just for title purposes)
tornado_grid_elasticity(
    df_elasticity=df_elast,
    metrics_to_plot=None, # ["Peak","RMS","MTVV","MTVV*√2","CF","VDV","R"],
    ncols=3,
    sharex=True,
    annotate=True,
    title="Local Sensitivity (Elasticity) - When %ΔParameter = ±" + str(rel_step*100)  +  "%"
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
    param_name="low_gain", # CHOOSE PARAMETER TO SWEEP
    param_range=(0.1, 1.0), # CHOOSE RANGE TO SWEEP (Units: Amplitude (m/s²) or Hz)
    samples=11, # Controls how many points are taken in sweep (how fine the resolution is) --> number of equally spaced points in the range
    # CHOOSE METRICS TO PLOT. metrics=None, # Default is all 7 weighted metrics
    metrics=["RMS"], # ["Peak","RMS","MTVV","MTVV*√2","CF","VDV","R"],
    title="% Change in metrics w.r.t. low_gain",
)
