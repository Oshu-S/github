from analyze_vibration import analyze_vibration
from local_sensitivity import local_sensitivity, tornado_grid_elasticity
from plot_sweep_multi_metric import plot_sweep_multi_metric
from plot_frequency_weighting import plot_frequency_weighting
from matplotlib import pyplot as plt
import numpy as np

# f1, W1 = plot_frequency_weighting(
#     low_gain=0.4,
#     f_low=0.5,
#     f_mid_start=2.0,
#     f_mid_end=5.0,
#     f_flat_end=16.0,
#     # plotting domain
#     fmin=0.016, fmax=63.0, n_points=2000,
#     # shape controls
#     log_ramp=True,             # ramp (f_mid_start → f_mid_end) straight on log–log
#     tail_exponent=1.0,         # high-freq slope: W ∝ f^{-beta} (beta=1 ⇒ −6 dB/oct)
#     tail_db_per_oct=None,      # optional: override beta using desired |dB/oct| (e.g., 6, 9, 12)
#     # visuals
#     show_knots=False, plot=False,
#     title="Asymptotic Approximation of Vertical Frequency Weighting"
# )

# f2, W2 = plot_frequency_weighting(
#     low_gain=0.4,
#     f_low=0.5,
#     f_mid_start=3.0,
#     f_mid_end=5.0,
#     f_flat_end=16.0,
#     # plotting domain
#     fmin=0.016, fmax=63.0, n_points=2000,
#     # shape controls
#     log_ramp=True,             # ramp (f_mid_start → f_mid_end) straight on log–log
#     tail_exponent=1.0,         # high-freq slope: [-dB/octave] --> W ∝ f^{-beta} (beta=1 ⇒ −6 dB/oct)
#     tail_db_per_oct=None,      # optional: override beta using desired |dB/oct| (e.g., 6, 9, 12)
#     # visuals
#     show_knots=False, plot=False,
#     title="Asymptotic Approximation of Vertical Frequency Weighting"
# )

# plt.figure(figsize=(12, 5))
# plt.loglog(f1, W1, label="Curve 1")
# plt.loglog(f2, W2, label="Curve 2")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Frequency Weighting")
# plt.title("Asymptotic Approximation of Vertical Frequency Weighting")
# octave_centers = np.array([0.016, 0.0315, 0.063, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 31.5, 63])
# plt.xticks(octave_centers, [str(f) for f in octave_centers])
# # Comment out if you want to hover over y-axis values
# plt.yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2], ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"])
# plt.xlim(0.016, 63)
# plt.ylim(0.005, 1.5)
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.legend()
# plt.tight_layout()
# plt.show()

# ====================================================
# A) Apply vertical frequency weighting + analyze weighted and unweighted metrics
# CHOOSE WEIGHTING CURVE PARAMETERS HERE
[t, acc_unw, acc_w, t_rms, rms_run_unw, rms_run_w, df_metrics] = analyze_vibration(
    file_path="pyhsi_results.csv", # og_data --> keep the same trial="Acceleration With HSI (m/s²)" below
    trial="Acceleration With HSI (m/s²)", #"Acceleration Without HSI (m/s²)"
    low_gain=0.40,
    f_low=0.5,
    f_mid_start=2.0,
    f_mid_end=5.0,
    f_flat_end=16.0,
    tail_exponent=1.0
)

# # ====================================================
# # B) Local derivative-based sensitivity using central finite differences & elasticities at baseline
# # CHOOSE BASELINE WEIGHTING CURVE PARAMETERS FOR SENSITIVITY ANALYSIS HERE
# base = dict(low_gain=0.4, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0)

# # Elasticity & Partial derivatives
# # CHOOSE % CHANGE IN PARAMETER FOR ELASTICITIES HERE
# df_elast, df_dydp = local_sensitivity(
#     file_path="pyhsi_results.csv",
#     base_params=base,
#     rel_step=0.05 # ±0.05 = ±5% change in parameter
#     # metrics_to_track optional — default is all 7 weighted metrics
# )

# # Tornado grid (one figure with subplots)
# rel_step=0.05 # CHOOSE % CHANGE IN PARAMETER FOR ELASTICITIES HERE (should match above, just for title purposes)
# tornado_grid_elasticity(
#     df_elasticity=df_elast,
#     metrics_to_plot= ["RMS"], # None, # ["Peak","RMS","MTVV","MTVV*√2","CF","VDV","R"],
#     ncols=1, # number of columns in subplot grid
#     sharex=True,
#     annotate=True,
#     title="Local Sensitivity (Elasticity) - When %ΔParameter = ±" + str(rel_step*100)  +  "%"
# )

# # ====================================================
# # C) Range sweep (one-at-a-time) → % change vs baseline
# # - show how metrics change when you move a parameter across a user-defined range with others fixed
# # Plot sweep to see % change in metrics as one weighting curve parameter is swept
# # CHOOSE PARAMETER, RANGE, AND METRICS TO PLOT HERE
# file = "og_data.csv"
# plot_sweep_multi_metric(
#     file_path=file,
#     base_params=base,
#     param_name="low_gain", # CHOOSE PARAMETER TO SWEEP
#     param_range=(0.1, 1.0), # CHOOSE RANGE TO SWEEP (Units: Magnitude (m/s²) or Hz)
#     samples=16, # Controls how many points are taken in sweep (how fine the resolution is) --> number of equally spaced points in the range
#     # CHOOSE METRICS TO PLOT. metrics=None, # Default is all 7 weighted metrics
#     metrics=["RMS acceleration"], # ["Peak acceleration","RMS acceleration","MTVV","MTVV*√2","CF","VDV","R"],
#     show_legend=False,
# )

# plot_sweep_multi_metric(
#     file_path=file,
#     base_params=base,
#     param_name="f_low", # CHOOSE PARAMETER TO SWEEP
#     param_range=(0.016, 0.5), # CHOOSE RANGE TO SWEEP (Units: Magnitude (m/s²) or Hz)
#     samples=100, # Controls how many points are taken in sweep (how fine the resolution is) --> number of equally spaced points in the range
#     # CHOOSE METRICS TO PLOT. metrics=None, # Default is all 7 weighted metrics
#     metrics=["RMS acceleration"], # ["Peak acceleration","RMS acceleration","MTVV","MTVV*√2","CF","VDV","R"],
#     show_legend=False,
# )

# plot_sweep_multi_metric(
#     file_path=file,
#     base_params=base,
#     param_name="f_mid_start", # CHOOSE PARAMETER TO SWEEP
#     param_range=(1.9, 2.1), # CHOOSE RANGE TO SWEEP (Units: Magnitude (m/s²) or Hz)
#     samples=100, # Controls how many points are taken in sweep (how fine the resolution is) --> number of equally spaced points in the range
#     # CHOOSE METRICS TO PLOT. metrics=None, # Default is all 7 weighted metrics
#     metrics=["RMS acceleration"], # ["Peak acceleration","RMS acceleration","MTVV","MTVV*√2","CF","VDV","R"],
#     show_legend=False,
# )

# plot_sweep_multi_metric(
#     file_path=file,
#     base_params=base,
#     param_name="f_mid_end", # CHOOSE PARAMETER TO SWEEP
#     param_range=(4.0, 16.0), # CHOOSE RANGE TO SWEEP (Units: Magnitude (m/s²) or Hz)
#     samples=100, # Controls how many points are taken in sweep (how fine the resolution is) --> number of equally spaced points in the range
#     # CHOOSE METRICS TO PLOT. metrics=None, # Default is all 7 weighted metrics
#     metrics=["RMS acceleration"], # ["Peak acceleration","RMS acceleration","MTVV","MTVV*√2","CF","VDV","R"],
#     show_legend=False,
# )

# plot_sweep_multi_metric(
#     file_path=file,
#     base_params=base,
#     param_name="f_flat_end", # CHOOSE PARAMETER TO SWEEP
#     param_range=(16.0, 60.0), # CHOOSE RANGE TO SWEEP (Units: Magnitude (m/s²) or Hz)
#     samples=100, # Controls how many points are taken in sweep (how fine the resolution is) --> number of equally spaced points in the range
#     # CHOOSE METRICS TO PLOT. metrics=None, # Default is all 7 weighted metrics
#     metrics=["RMS acceleration"], # ["Peak acceleration","RMS acceleration","MTVV","MTVV*√2","CF","VDV","R"],
#     show_legend=False,
# )

