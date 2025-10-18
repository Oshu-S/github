from analyze_vibration import analyze_vibration
from plot_frequency_weighting import plot_frequency_weighting
from matplotlib import pyplot as plt
import numpy as np
import math

# WALKING ====================================================
# f1, W1 = plot_frequency_weighting(
#     low_gain=0.4, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0,
#     # plotting domain
#     fmin=0.016, fmax=63.0, n_points=2000,
#     # shape controls
#     log_ramp=True,             # ramp (f_mid_start → f_mid_end) straight on log–log
#     tail_exponent=1.0,         # high-freq slope: W ∝ f^{-beta} (beta=1 ⇒ −6 dB/oct)
#     tail_db_per_oct=None,      # optional: override beta using desired |dB/oct| (e.g., 6, 9, 12)
#     # visuals
#     show_knots=True, plot=False,
#     title="Asymptotic Approximation of Vertical Frequency Weighting"
# )

# # Lower Bound (faster pacing, less sensitive)
# f2, W2 = plot_frequency_weighting(
#     low_gain=0.1, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0,
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

# # Upper Bound (slower pacing, more sensitive)
# f3, W3 = plot_frequency_weighting(
#     low_gain=0.2, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0,
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

# plt.figure(figsize=(10, 5))
# plt.loglog(f1, W1, color="C0", ls="-.", linewidth=2, label="ISO 2631 ($W_k$)") # Baseline
# plt.loglog(f2, W2, color="C1", ls="--", linewidth=2,  label="Lower Bound (faster pacing)") # Lower Bound
# plt.loglog(f3, W3, color="C2", ls=":", linewidth=2, label="Upper Bound (slower pacing)") # Upper Bound
# plt.xlabel("$f$ (Hz)", fontsize=16)
# plt.ylabel("Frequency Weighting", fontsize=16)
# octave_centers = np.array([0.016, 0.0315, 0.063, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 31.5, 63])
# plt.xticks(octave_centers, [str(f) for f in octave_centers], fontsize=12)
# # Comment out if you want to hover over y-axis values
# plt.yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2], ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"], fontsize=12)
# plt.xlim(0.016, 63)
# plt.ylim(0.005, 1.5)
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.legend(loc='lower left', bbox_to_anchor=(0.4, 0.05), fontsize=15, framealpha=1.0)
# plt.tight_layout()
# plt.show()


# # ====================================================
# # A) Apply vertical frequency weighting + analyze weighted and unweighted metrics
# # CHOOSE WEIGHTING CURVE PARAMETERS HERE
# [t, acceleration, weighted_signal, 
#     t_rms, rms_running_unweighted, rms_running_weighted,
#     df_metrics,
#     f_fft, fft_unw, fft_w,
#     f_psd, PSD_unw, PSD_w] = analyze_vibration(
#     file_path="pyhsi_results.csv", # og_data --> keep the same trial="Acceleration With HSI (m/s²)" below
#     trial="Acceleration With HSI (m/s²)", #"Acceleration Without HSI (m/s²)"
#     low_gain=0.40,
#     f_low=0.5,
#     f_mid_start=2.0,
#     f_mid_end=5.0,
#     f_flat_end=16.0,
#     tail_exponent=1.0,
#     show_knots=True,
#     plot=False
# )
    
# # fMidStart (f2) ====================================================
# # Baseline --> fMidStart = 2.0 Hz
# f1, W1 = plot_frequency_weighting(
#     low_gain=0.4, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0,
#     # plotting domain
#     fmin=0.02, fmax=63.0, n_points=2000,
#     # shape controls
#     log_ramp=True,             # ramp (f_mid_start → f_mid_end) straight on log–log
#     tail_exponent=1.0,         # high-freq slope: W ∝ f^{-beta} (beta=1 ⇒ −6 dB/oct)
#     tail_db_per_oct=None,      # optional: override beta using desired |dB/oct| (e.g., 6, 9, 12)
#     # visuals
#     show_knots=True, plot=False,
#     title="Asymptotic Approximation of Vertical Frequency Weighting"
# )

# # fMidStart = 0.1 Hz
# f2, W2 = plot_frequency_weighting(
#     low_gain=0.4, f_low=0.5, f_mid_start=0.1, f_mid_end=5.0, f_flat_end=16.0,
#     # plotting domain
#     fmin=0.02, fmax=63.0, n_points=2000,
#     # shape controls
#     log_ramp=True,             # ramp (f_mid_start → f_mid_end) straight on log–log
#     tail_exponent=1.0,         # high-freq slope: [-dB/octave] --> W ∝ f^{-beta} (beta=1 ⇒ −6 dB/oct)
#     tail_db_per_oct=None,      # optional: override beta using desired |dB/oct| (e.g., 6, 9, 12)
#     # visuals
#     show_knots=False, plot=False,
#     title="Asymptotic Approximation of Vertical Frequency Weighting"
# )

# # fMidStart = 5.0 Hz
# f3, W3 = plot_frequency_weighting(
#     low_gain=0.4, f_low=0.5, f_mid_start=5.0, f_mid_end=5.0, f_flat_end=16.0,
#     # plotting domain
#     fmin=0.02, fmax=63.0, n_points=2000,
#     # shape controls
#     log_ramp=True,             # ramp (f_mid_start → f_mid_end) straight on log–log
#     tail_exponent=1.0,         # high-freq slope: W ∝ f^{-beta} (beta=1 ⇒ −6 dB/oct)
#     tail_db_per_oct=None,      # optional: override beta using desired |dB/oct| (e.g., 6, 9, 12)
#     # visuals
#     show_knots=False, plot=False,
#     title="Asymptotic Approximation of Vertical Frequency Weighting"
# )

# plt.figure(figsize=(8, 4.5))
# plt.loglog(f1, W1, color="C0", ls="-.", linewidth=2, label="$f_{2,0}=2.0$ Hz (baseline)") # Baseline
# plt.loglog(f2, W2, color="C1", ls="--", linewidth=2,  label="$f_2=0.1$ Hz")
# plt.loglog(f3, W3, color="C2", ls=":", linewidth=2, label="$f_2=5.0$ Hz")
# plt.plot(f_fft, fft_unw, color="C3", linewidth=1.5, label="Footbridge response (m/s²)")
# plt.xlabel("$f$ (Hz)", fontsize=15)
# plt.ylabel("Frequency Weighting", fontsize=15)
# octave_centers = np.array([0.016, 0.063, 0.25, 0.5, 1, 2, 4, 8, 16, 31.5, 63])
# plt.xticks(octave_centers, [str(f) for f in octave_centers], fontsize=12)
# # Comment out if you want to hover over y-axis values
# plt.yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2], ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"], fontsize=12)
# plt.xlim(0.016, 63)
# plt.ylim(0.005, 1.5)
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.legend(loc='lower left', bbox_to_anchor=(0.6, 0.05), fontsize=13, framealpha=1.0)
# plt.tight_layout()
# plt.show()


# ISO 2631 W_k vs Final Experimental Weighting Curve Hypothesised
f_wk, W_wk = plot_frequency_weighting(
    low_gain=0.4, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0,
    # plotting domain
    fmin=0.016, fmax=63.0, n_points=2000,
    # shape controls
    log_ramp=True,             # ramp (f_mid_start → f_mid_end) straight on log–log
    tail_exponent=1.0,         # high-freq slope: [-dB/octave] --> W ∝ f^{-beta} (beta=1 ⇒ −6 dB/oct)
    tail_db_per_oct=None,      # optional: override beta using desired |dB/oct| (e.g., 6, 9, 12)
    # visuals
    show_knots=False, plot=False,
    title="Asymptotic Approximation of Vertical Frequency Weighting"
)

f_hyp, W_hyp = plot_frequency_weighting(
    low_gain=0.2, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0,
    # plotting domain
    fmin=0.016, fmax=63.0, n_points=2000,
    # shape controls
    log_ramp=True,             # ramp (f_mid_start → f_mid_end) straight on log–log
    tail_exponent=1.0,         # high-freq slope: W ∝ f^{-beta} (beta=1 ⇒ −6 dB/oct)
    tail_db_per_oct=None,      # optional: override beta using desired |dB/oct| (e.g., 6, 9, 12)
    # visuals
    show_knots=False, plot=False,
    title="Asymptotic Approximation of Vertical Frequency Weighting"
)

plt.figure(figsize=(8, 4.5))
plt.loglog(f_wk, W_wk, color="C0", ls="-.", linewidth=2, label="ISO 2631 $W_k$") # Baseline
plt.loglog(f_hyp, W_hyp, color="C1", ls="--", linewidth=2,  label="Final experimental weighting (hypothesised)") # Lower Bound
plt.xlabel("$f$ (Hz)", fontsize=15)
plt.ylabel("Frequency Weighting", fontsize=15)
octave_centers = np.array([0.016, 0.063, 0.25, 0.5, 1, 2, 4, 8, 16, 31.5, 63])
plt.xticks(octave_centers, [str(f) for f in octave_centers], fontsize=12)
# Comment out if you want to hover over y-axis values
plt.yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2], ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"], fontsize=12)
plt.xlim(0.016, 63)
plt.ylim(0.005, 1.5)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend(fontsize=13, framealpha=1.0)
plt.tight_layout()
plt.show()