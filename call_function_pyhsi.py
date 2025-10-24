from analyze_vibration import analyze_vibration
from local_sensitivity import local_sensitivity, tornado_grid_elasticity
from plot_sweep_multi_metric import plot_sweep_multi_metric
from matplotlib import pyplot as plt
import numpy as np

# ====================================================
# A) Apply vertical frequency weighting + analyze weighted and unweighted metrics
# CHOOSE WEIGHTING CURVE PARAMETERS HERE
[t, acceleration, weighted_signal, 
    t_rms, rms_running_unweighted, rms_running_weighted,
    df_metrics,
    f_fft, fft_unw, fft_w,
    f_psd, PSD_unw, PSD_w] = analyze_vibration(
    file_path="pyhsi_results.csv", # og_data --> keep the same trial="Acceleration With HSI (m/s²)" below
    trial="Acceleration With HSI (m/s²)", #"Acceleration Without HSI (m/s²)"
    low_gain=0.40,
    f_low=0.5,
    f_mid_start=2.0,
    f_mid_end=5.0,
    f_flat_end=16.0,
    tail_exponent=1.0,
    show_knots=True,
    plot=True
)

# ====================================================
# B) Local derivative-based sensitivity using central finite differences & elasticities at baseline
# CHOOSE BASELINE WEIGHTING CURVE PARAMETERS FOR SENSITIVITY ANALYSIS HERE
base = dict(low_gain=0.4, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0)

# Elasticity & Partial derivatives
# CHOOSE % CHANGE IN PARAMETER FOR ELASTICITIES HERE
df_elasticity, df_dydp, df_pct_change = local_sensitivity(
    file_path="pyhsi_results.csv",
    base_params=base,
    rel_step=0.05 # ±0.05 = ±5% change in parameter
    # metrics_to_track optional — default is all 7 weighted metrics
)

# Tornado grid (one figure with subplots)
rel_step=0.05 # CHOOSE % CHANGE IN PARAMETER FOR ELASTICITIES HERE (should match above, just for title purposes)
tornado_grid_elasticity(
    df_elasticity=df_elasticity,
    metrics_to_plot= ["RMS"], # None, # ["Peak","RMS","MTVV","MTVV*√2","CF","VDV","R"],
    ncols=1, # number of columns in subplot grid
    sharex=True,
    annotate=True,
    title="Local Sensitivity (Elasticity) - When %ΔParameter = ±" + str(rel_step*100)  +  "%"
)

# ====================================================
# C) Range sweep (one-at-a-time) → % change vs baseline
# - show how metrics change when you move a parameter across a user-defined range with others fixed
# Plot sweep to see % change in metrics as one weighting curve parameter is swept
# CHOOSE PARAMETER, RANGE, AND METRICS TO PLOT HERE
file = "pyhsi_results.csv"
plot_sweep_multi_metric(
    file_path=file,
    base_params=base,
    param_name="low_gain", # CHOOSE PARAMETER TO SWEEP
    param_range=(0.1, 1.0), # CHOOSE RANGE TO SWEEP (Units: Magnitude (m/s²) or Hz)
    samples=16, # Controls how many points are taken in sweep (how fine the resolution is) --> number of equally spaced points in the range
    # CHOOSE METRICS TO PLOT. metrics=None, # Default is all 7 weighted metrics
    metrics=["RMS_weighted"], # ["Peak acceleration","RMS acceleration","MTVV","MTVV*√2","CF","VDV","R"],
    title=False,
    show_legend=False,
)

plot_sweep_multi_metric(
    file_path=file,
    base_params=base,
    param_name="f_low", # CHOOSE PARAMETER TO SWEEP
    param_range=(0.016, 0.5), # CHOOSE RANGE TO SWEEP (Units: Magnitude (m/s²) or Hz)
    samples=100, # Controls how many points are taken in sweep (how fine the resolution is) --> number of equally spaced points in the range
    # CHOOSE METRICS TO PLOT. metrics=None, # Default is all 7 weighted metrics
    metrics=["RMS acceleration"], # ["Peak acceleration","RMS acceleration","MTVV","MTVV*√2","CF","VDV","R"],
    show_legend=False,
)

plot_sweep_multi_metric(
    file_path=file,
    base_params=base,
    param_name="f_mid_start", # CHOOSE PARAMETER TO SWEEP
    param_range=(0.5, 5.0), # CHOOSE RANGE TO SWEEP (Units: Magnitude (m/s²) or Hz)
    samples=40, # Controls how many points are taken in sweep (how fine the resolution is) --> number of equally spaced points in the range
    # CHOOSE METRICS TO PLOT. metrics=None, # Default is all 7 weighted metrics
    metrics=["RMS_weighted"], # ["Peak acceleration","RMS acceleration","MTVV","MTVV*√2","CF","VDV","R"],
    title=False,
    show_legend=False,
)

plot_sweep_multi_metric(
    file_path=file,
    base_params=base,
    param_name="f_mid_end", # CHOOSE PARAMETER TO SWEEP
    param_range=(4.0, 16.0), # CHOOSE RANGE TO SWEEP (Units: Magnitude (m/s²) or Hz)
    samples=100, # Controls how many points are taken in sweep (how fine the resolution is) --> number of equally spaced points in the range
    # CHOOSE METRICS TO PLOT. metrics=None, # Default is all 7 weighted metrics
    metrics=["RMS acceleration"], # ["Peak acceleration","RMS acceleration","MTVV","MTVV*√2","CF","VDV","R"],
    show_legend=False,
)

plot_sweep_multi_metric(
    file_path=file,
    base_params=base,
    param_name="f_flat_end", # CHOOSE PARAMETER TO SWEEP
    param_range=(16.0, 60.0), # CHOOSE RANGE TO SWEEP (Units: Magnitude (m/s²) or Hz)
    samples=100, # Controls how many points are taken in sweep (how fine the resolution is) --> number of equally spaced points in the range
    # CHOOSE METRICS TO PLOT. metrics=None, # Default is all 7 weighted metrics
    metrics=["RMS acceleration"], # ["Peak acceleration","RMS acceleration","MTVV","MTVV*√2","CF","VDV","R"],
    show_legend=False,
)

