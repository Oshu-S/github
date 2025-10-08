import numpy as np
import matplotlib.pyplot as plt

def plot_frequency_weighting(
    low_gain=0.4,
    f_low=0.5,
    f_mid_start=2.0,
    f_mid_end=5.0,
    f_flat_end=16.0,
    *,
    # plotting domain
    fmin=0.016, fmax=63.0, n_points=2000,
    # shape controls
    log_ramp=True,             # ramp (f_mid_start → f_mid_end) straight on log–log
    tail_exponent=1.0,         # high-freq slope: W ∝ f^{-beta} (beta=1 ⇒ −6 dB/oct)
    tail_db_per_oct=None,      # optional: override beta using desired |dB/oct| (e.g., 6, 9, 12)
    # visuals
    show_knots=True, plot=True,
    title="Asymptotic Approximation of Vertical Frequency Weighting"
):
    """
    Build and plot a frequency weighting W(f).

    Tail steepness:
      - Set `tail_exponent=beta` for W ~ (f_flat_end/f)^beta (straight line on log–log).
      - Or pass `tail_db_per_oct` (positive) to target a slope in dB/oct; e.g. 6 → beta≈1.
    """
    # --- validate / sanitize ---
    eps = 1e-9
    f_low       = max(eps, float(f_low))
    f_mid_start = max(f_low + eps, float(f_mid_start))
    f_mid_end   = max(f_mid_start + eps, float(f_mid_end))
    f_flat_end  = max(f_mid_end + eps, float(f_flat_end))
    low_gain    = float(low_gain)

    # convert desired dB/oct to exponent if provided
    if tail_db_per_oct is not None:
        # |dB/oct| = 20*log10(2^beta) = 6.0206*beta  →  beta = |dB/oct| / 6.0206
        tail_exponent = float(tail_db_per_oct) / (20.0 * np.log10(2.0))
    tail_exponent = max(tail_exponent, eps)  # keep positive

    # frequency grid (log-spaced for smooth log–log plots)
    f = np.logspace(np.log10(fmin), np.log10(fmax), n_points)

    # piecewise weighting
    W = np.zeros_like(f, dtype=float)
    val_low  = low_gain
    val_flat = 1.0
    f_flat_start = f_mid_end

    # seg 0: 0..f_low (linear rise up to low plateau)
    m0 = (f > 0.0) & (f < f_low)
    W[m0] = val_low * (f[m0] / f_low)

    # seg 1: f_low..f_mid_start (low plateau)
    m1 = (f >= f_low) & (f <= f_mid_start)
    W[m1] = val_low

    # seg 2: f_mid_start..f_mid_end (ramp to 1.0)
    m2 = (f > f_mid_start) & (f <= f_mid_end)
    if np.any(m2):
        if log_ramp:
            # power-law ramp: straight line on log–log, exact at endpoints
            alpha = np.log(val_flat / val_low) / np.log(f_mid_end / f_mid_start)
            W[m2] = val_low * (f[m2] / f_mid_start) ** alpha
        else:
            # linear in frequency (appears curved on log–log)
            W[m2] = val_low + (val_flat - val_low) * (
                (f[m2] - f_mid_start) / (f_mid_end - f_mid_start)
            )

    # seg 3: f_flat_start..f_flat_end (flat at 1.0)
    m3 = (f > f_flat_start) & (f <= f_flat_end)
    W[m3] = val_flat

    # seg 4: > f_flat_end (power-law tail; straight on log–log)
    m4 = (f > f_flat_end)
    # continuity at f_flat_end is automatic: W(f_flat_end) = val_flat
    W[m4] = val_flat * (f_flat_end / f[m4]) ** tail_exponent

    # --- plot ---
    if plot:
        plt.figure(figsize=(12, 5))
        plt.loglog(f, W)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Frequency Weighting")
        octave_centers = np.array([0.016, 0.0315, 0.063, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 31.5, 63])
        plt.xticks(octave_centers, [str(f) for f in octave_centers])
        # Comment out if you want to hover over y-axis values
        # plt.yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2], ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"])
        plt.title(title)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        if show_knots:
            for x, lab in [(f_low, "fLow"), (f_mid_start, "fMidStart"),
                           (f_mid_end, "fMidEnd"), (f_flat_end, "fFlatEnd")]:
                plt.axvline(x, ls="--", lw=1.0, color="r", alpha=0.6)
                # put labels just above the lowest visible weighting to keep tidy
                ylab = max(np.min(W[W>0]), 1e-3)
                plt.text(x, 1.1*ylab, lab, color="r",
                         rotation=90, va="bottom", ha="right", fontsize=9)

        plt.tight_layout()
        plt.show()

    return f, W
