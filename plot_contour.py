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
    eps = 1e-12
    f_low       = max(eps, float(f_low))
    f_mid_start = max(f_low + eps, float(f_mid_start))
    f_mid_end   = max(f_mid_start + eps, float(f_mid_end))
    f_flat_end  = max(f_mid_end + eps, float(f_flat_end))
    low_gain    = float(low_gain)
    tail_exponent = max(float(tail_exponent), eps)
    base_level    = float(base_level)

    # frequency grid (log-spaced)
    f = np.logspace(np.log10(fmin), np.log10(fmax), int(n_points))

    # --- Weighting W(f) ---
    W = np.zeros_like(f, dtype=float)
    val_low  = low_gain
    val_flat = 1.0
    f_flat_start = f_mid_end

    # seg 0: 0..f_low
    m0 = (f > 0.0) & (f < f_low)
    W[m0] = val_low * (f[m0] / f_low)

    # seg 1: f_low..f_mid_start
    m1 = (f >= f_low) & (f <= f_mid_start)
    W[m1] = val_low

    # seg 2: f_mid_start..f_mid_end
    m2 = (f > f_mid_start) & (f <= f_mid_end)
    if np.any(m2):
        if log_ramp:
            alpha = np.log(val_flat / val_low) / np.log(f_mid_end / f_mid_start)
            W[m2] = val_low * (f[m2] / f_mid_start) ** alpha
        else:
            W[m2] = val_low + (val_flat - val_low) * (
                (f[m2] - f_mid_start) / (f_mid_end - f_mid_start)
            )

    # seg 3: f_flat_start..f_flat_end
    m3 = (f > f_flat_start) & (f <= f_flat_end)
    W[m3] = val_flat

    # seg 4: > f_flat_end
    m4 = (f > f_flat_end)
    W[m4] = val_flat * (f_flat_end / f[m4]) ** tail_exponent

    W = np.maximum(W, eps)

    # Baseline contour (no scaling)
    C_raw = base_level / W

    # --- Multiplicative scaling to hit target point (keeps shape) ---
    scale = 1.0
    if target_point is not None:
        f_t, a_t = float(target_point[0]), float(target_point[1])
        if (fmin <= f_t <= fmax) and (a_t > 0):
            # interpolate W at f_t on log-f axis for smoothness
            W_t = np.interp(np.log10(f_t), np.log10(f), W)
            C_raw_t = base_level / max(W_t, eps)
            scale = a_t / C_raw_t

    C = scale * C_raw  # shape-preserving vertical shift on log axis

    # Clip for plotting on log axes
    C_plot = np.clip(C, 1e-12, None)

    if plot:
        plt.figure(figsize=(10, 4.5))
        plt.loglog(f, C_plot)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Equivalent comfort level, C(f) [m/s²]")
        plt.title(title)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        if show_knots:
            for x, lab in [(f_low,  r"$f_1$"),
                           (f_mid_start, r"$f_2$"),
                           (f_mid_end,   r"$f_3$"),
                           (f_flat_end,  r"$f_4$")]:
                plt.axvline(x, ls="--", lw=1.0, color="r", alpha=0.6)
                ylab = np.nanmin(C_plot[np.isfinite(C_plot)])
                plt.text(x, 1.15*ylab, lab, color="r",
                         rotation=90, va="bottom", ha="right", fontsize=10, fontweight="bold")

        if target_point is not None and (fmin <= f_t <= fmax) and (a_t > 0):
            plt.scatter([f_t], [a_t], s=40, c="k", zorder=3)
            plt.text(f_t, a_t, "  target", va="center", ha="left", fontsize=9)

        plt.tight_layout()
        plt.show()

    return f, C
