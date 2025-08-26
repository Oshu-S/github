from Function_VerticalFrequencyWeighting import (
    analyze_vibration,
    local_sensitivity_from_file,
    range_sweep_percent,
    tornado_grid_elasticity
)

# A) Show your three plots + printed table
analyze_vibration(
    file_path="results_500mc_trial_matrix.csv",
    low_gain=0.4,
    f_low=0.5,
    f_mid_start=2.0,
    f_mid_end=5.0,
    f_flat_end=16.0
)

# B) Local derivatives & elasticities at your baseline (ALL 7 metrics)
base = dict(low_gain=0.4, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0)

df_elast, df_dydp = local_sensitivity_from_file(
    file_path="results_500mc_trial_matrix.csv",
    trial_index=0,
    fs=100,
    base_params=base,
    rel_step=0.05,
    metrics_to_track=(
        "Peak_weighted","RMS_weighted","MTVV_weighted",
        "MTVV_sqrt2_weighted","CF_weighted","VDV_weighted","R_weighted"
    )
)

print("Elasticities (out/in):")
print(df_elast.round(4))
print("\nRaw partial derivatives (metric units per param unit):")
print(df_dydp.round(6))

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

# C) Range sweep (one-at-a-time) → % change vs baseline
param_ranges = {
    "low_gain":   (0.2, 0.6),
    "f_low":      (0.3, 1.0),
    "f_mid_start":(1.5, 3.0),
    "f_mid_end":  (4.0, 6.0),
    "f_flat_end": (12.0, 20.0),
}

sweep = range_sweep_percent(
    file_path="results_500mc_trial_matrix.csv",
    trial_index=0,
    fs=100,
    base_params=base,
    param_ranges=param_ranges,
    samples_per_param=7,
    metrics_to_track=(
        "Peak_weighted","RMS_weighted","MTVV_weighted",
        "MTVV_sqrt2_weighted","CF_weighted","VDV_weighted","R_weighted"
    )
)

# Example: plot % change for RMS_weighted vs parameter value for 'low_gain' 
import matplotlib.pyplot as plt 
ax = sweep["low_gain"]["RMS_weighted"].plot(marker="o") 
ax.set_title("RMS_weighted: % change vs low_gain") 
ax.set_xlabel("low_gain") # ax.set_ylabel("% change relative to baseline") 
ax.grid(True) 
plt.show()
