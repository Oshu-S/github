import numpy as np
import matplotlib.pyplot as plt

def plot_equivalent_comfort_contour(
    low_gain=0.4,
    f_low=0.5,
    f_mid_start=2.0,
    f_mid_end=5.0,
    f_flat_end=16.0,
    *,
    # plotting domain
    fmin=0.016, fmax=63.0, n_points=2000,
    # shape controls
    log_ramp=True,           # ramp (f_mid_start → f_mid_end) straight on log–log
    tail_exponent=1.0,       # high-freq tail ~ 1 / f^tail_exponent (6 dB/oct ≈ 1.0)
    # amplitude/anchoring (in acceleration units)
    base_level=1.0,          # m/s² when W ≈ 1 (flat region)
    target_point=None,       # (f_target [Hz], a_target [m/s²]) → force contour through this point
    # visuals
    show_knots=True,
    plot=True,
    title="Equivalent Comfort Contour (acceleration)"
):
    """
    Build and (optionally) plot an equivalent comfort contour C(f) in m/s².

    Steps:
      1) Construct weighting W(f) from curve parameters.
      2) C_raw(f) = base_level / W(f).
      3) If target_point=(f*, a*): translate vertically so C(f*) = a*.

    Notes:
      • If the translation makes any values ≤ 0, they’re clipped for plotting on log–log axes
        (returned data C is not clipped).
    """
    # --- validate / sanitize ---
    eps = 1e-12
    f_low       = max(eps, float(f_low))
    f_mid_start = max(f_low + eps, float(f_mid_start))
    f_mid_end   = max(f_mid_start + eps, float(f_mid_end))
    f_flat_end  = max(f_mid_end + eps, float(f_flat_end))
    low_gain    = float(low_gain)
    tail_exponent = max(float(tail_exponent), eps)
    base_level    = float(base_level)

    # frequency grid (log-spaced for smooth log–log plots)
    f = np.logspace(np.log10(fmin), np.log10(fmax), int(n_points))

    # --- build weighting W(f) ---
    W = np.zeros_like(f, dtype=float)
    val_low  = low_gain
    val_flat = 1.0
    f_flat_start = f_mid_end

    # seg 0: 0..f_low (linear rise to low plateau)
    m0 = (f > 0.0) & (f < f_low)
    W[m0] = val_low * (f[m0] / f_low)

    # seg 1: f_low..f_mid_start (low plateau)
    m1 = (f >= f_low) & (f <= f_mid_start)
    W[m1] = val_low

    # seg 2: f_mid_start..f_mid_end (ramp to 1.0)
    m2 = (f > f_mid_start) & (f <= f_mid_end)
    if np.any(m2):
        if log_ramp:
            # power-law ramp: straight on log–log, exact at endpoints
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
    W[m4] = val_flat * (f_flat_end / f[m4]) ** tail_exponent

    W = np.maximum(W, eps)  # for safety

    # baseline contour (no translation)
    C_raw = base_level / W

    # --- compute translation if target point is given ---
    translate_by = 0.0
    if target_point is not None:
        f_t, a_t = float(target_point[0]), float(target_point[1])
        # guard: keep f_t inside domain for interpolation
        f_t = np.clip(f_t, fmin, fmax)
        # interpolate W at f_t on log-frequency axis (smooth for power-law segments)
        W_t = np.interp(np.log10(f_t), np.log10(f), W)
        translate_by = a_t - (base_level / max(W_t, eps))

    # translated contour
    C = C_raw + translate_by

    # clip for plotting on log axis (data C is not clipped)
    C_plot = np.clip(C, 1e-9, None)

    if plot:
        plt.figure(figsize=(10, 4.5))
        plt.loglog(f, C_plot)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Equivalent comfort level, C(f) [m/s²]")
        plt.title(title)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        if show_knots:
            for x, lab in [(f_low,  r"$f_{\mathrm{low}}$"),
                           (f_mid_start, r"$f_{\mathrm{midStart}}$"),
                           (f_mid_end,   r"$f_{\mathrm{midEnd}}$"),
                           (f_flat_end,  r"$f_{\mathrm{flatEnd}}$")]:
                plt.axvline(x, ls="--", lw=1.0, color="r", alpha=0.6)
                ylab = max(np.min(C_plot), 1e-9)
                plt.text(x, 1.15*ylab, lab, color="r",
                         rotation=90, va="bottom", ha="right", fontsize=10, fontweight="bold")

        # mark target point if supplied (and inside domain)
        if target_point is not None and (fmin <= f_t <= fmax) and (a_t > 0):
            plt.scatter([f_t], [a_t], s=40, c="k", zorder=3)
            plt.text(f_t, a_t, "  target", va="center", ha="left", fontsize=9)

        plt.tight_layout()
        plt.show()

    return f, C
