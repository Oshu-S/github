from plot_contour import plot_equivalent_comfort_contour
from plot_frequency_weighting import plot_frequency_weighting
from matplotlib import pyplot as plt
import numpy as np

legsize = 17
labsize = 19
ticksize = 17

# Experimental contour plots for different reference points
fc1, c1 = plot_equivalent_comfort_contour(
    low_gain=0.2, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0,
    # plotting domain
    fmin=0.5, fmax=20, n_points=2000,
    # shape controls
    log_ramp=True,           # ramp (f_mid_start → f_mid_end) straight on log–log
    tail_exponent=1.0,       # high-freq tail ~ 1 / f^tail_exponent (6 dB/oct ≈ 1.0)
    # amplitude/anchoring (in acceleration units)
    base_level=1,          # m/s² when W ≈ 1 (flat region)
    target_point=(2,0.75),       # (f_target [Hz], a_target [m/s²]) → force contour through this point
    # visuals
    show_knots=True, plot=False,
    title="Equivalent Comfort Contour (acceleration)"
)

fc2, c2 = plot_equivalent_comfort_contour(
    low_gain=0.2, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0,
    # plotting domain
    fmin=0.5, fmax=20, n_points=2000,
    # shape controls
    log_ramp=True,           # ramp (f_mid_start → f_mid_end) straight on log–log
    tail_exponent=1.0,       # high-freq tail ~ 1 / f^tail_exponent (6 dB/oct ≈ 1.0)
    # amplitude/anchoring (in acceleration units)
    base_level=1,          # m/s² when W ≈ 1 (flat region)
    target_point=(10,0.5),       # (f_target [Hz], a_target [m/s²]) → force contour through this point
    # visuals
    show_knots=True, plot=False,
    title="Equivalent Comfort Contour (acceleration)"
)

# fig 1
plt.figure(figsize=(7, 7))
plt.loglog(fc1, c1, color="C0", linewidth=2,  label="2 Hz, 0.75 m/s² \"reference\"") # Lower Bound
plt.loglog(fc2, c2, color="C1", linewidth=2,  label="10 Hz, 0.5 m/s² \"reference\"") # Lower Bound
plt.loglog(2, 0.75, marker="o", color="C0", markersize=8)
plt.loglog(10, 0.5, marker="o", color="C1", markersize=8)
plt.xlabel("Frequency (Hz)", fontsize=labsize)
plt.ylabel("$a$ (m/s$^2$ RMS)", fontsize=labsize)
octave_centers = np.array([0.5, 0.8, 1.25, 2.0, 3.15, 5.0, 8.0, 12.5, 20.0])
plt.xticks(octave_centers, [str(f) for f in octave_centers], fontsize=ticksize)
magnitude_ticks = [0.10, 0.16, 0.25, 0.40, 0.63, 1.0, 1.6, 2.5]
plt.yticks(magnitude_ticks, [str(f) for f in magnitude_ticks], fontsize=ticksize)
plt.ylim(0.09, 3)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend(loc='lower left', bbox_to_anchor=(0, -0.01), fontsize=legsize, framealpha=1.0) # loc='lower left', bbox_to_anchor=(0.4, 0.05)
plt.tight_layout()

# Final experimental contour vs ISO 2631 
fc_wk, c_wk = plot_equivalent_comfort_contour(
    low_gain=0.4, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0,
    # plotting domain
    fmin=0.016, fmax=63.0, n_points=2000,
    # shape controls
    log_ramp=True,           # ramp (f_mid_start → f_mid_end) straight on log–log
    tail_exponent=1.0,       # high-freq tail ~ 1 / f^tail_exponent (6 dB/oct ≈ 1.0)
    # amplitude/anchoring (in acceleration units)
    base_level=1.0,          # m/s² when W ≈ 1 (flat region)
    target_point=(10,1),       # (f_target [Hz], a_target [m/s²]) → force contour through this point
    # visuals
    show_knots=True, plot=False,
    title="Equivalent Comfort Contour (acceleration)"
)

fc3, c3 = plot_equivalent_comfort_contour(
    low_gain=0.2, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0,
    # plotting domain
    fmin=0.5, fmax=20, n_points=2000,
    # shape controls
    log_ramp=True,           # ramp (f_mid_start → f_mid_end) straight on log–log
    tail_exponent=1.0,       # high-freq tail ~ 1 / f^tail_exponent (6 dB/oct ≈ 1.0)
    # amplitude/anchoring (in acceleration units)
    base_level=1.0,          # m/s² when W ≈ 1 (flat region)
    target_point=(10,1),       # (f_target [Hz], a_target [m/s²]) → force contour through this point
    # visuals
    show_knots=True, plot=False,
    title="Equivalent Comfort Contour (acceleration)"
)

# fig 2
plt.figure(figsize=(7, 7))
plt.loglog(fc_wk, c_wk, color="C3", ls="--", linewidth=2, label="ISO 2631 $W_k$ equivalent contour") # ISO 2631
plt.loglog(fc3, c3, color="C2", linewidth=2,  label="Experimental contour") # Lower Bound
plt.xlabel("Frequency (Hz)", fontsize=labsize)
plt.ylabel("$a$ (m/s$^2$ RMS)", fontsize=labsize)
plt.xticks(octave_centers, [str(f) for f in octave_centers], fontsize=ticksize)
contour_ticks = [0.5, 0.75, 1, 1.5, 2, 3, 4, 5]
plt.yticks(contour_ticks, [str(f) for f in contour_ticks], fontsize=ticksize)
plt.xlim(0.4, 25)
plt.ylim(0.75, 5.5)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend(loc='lower left', bbox_to_anchor=(0, -0.015), fontsize=legsize, framealpha=1.0) # loc='lower left', bbox_to_anchor=(0.4, 0.05)
plt.tight_layout()

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

# fig 3
plt.figure(figsize=(7, 7))
plt.loglog(f_wk, W_wk, color="C3", ls="--", linewidth=2, label="ISO 2631 $W_k$") # Baseline
plt.loglog(f_hyp, W_hyp, color="C2", linewidth=2,  label="Experimental weighting") # Lower Bound
plt.xlabel("$f$ (Hz)", fontsize=labsize)
plt.ylabel("Frequency Weighting", fontsize=labsize)
octave_centers = np.array([0.016, 0.063, 0.25, 1, 4, 16, 63])
plt.xticks(octave_centers, [str(f) for f in octave_centers], fontsize=ticksize)
# Comment out if you want to hover over y-axis values
fig3ticks = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
plt.yticks(fig3ticks, [str(f) for f in fig3ticks], fontsize=ticksize)
plt.xlim(0.016, 63)
# plt.ylim(0.005, 1.5)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend(loc='lower right', fontsize=legsize, framealpha=1.0)
plt.tight_layout()

plt.show()