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
    Build and (optionally) plot a frequency weighting W(f).
    """
    # --- validate / sanitize ---
    eps = 1e-9
    f_low       = max(eps, float(f_low))
    f_mid_start = max(f_low + eps, float(f_mid_start))
    f_mid_end   = max(f_mid_start + eps, float(f_mid_end))
    f_flat_end  = max(f_mid_end + eps, float(f_flat_end))
    low_gain    = float(low_gain)

    # optional: map desired |dB/oct| to exponent
    if tail_db_per_oct is not None:
        tail_exponent = float(tail_db_per_oct) / (20.0 * np.log10(2.0))
    tail_exponent = max(tail_exponent, eps)

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
            alpha = np.log(val_flat / val_low) / np.log(f_mid_end / f_mid_start)
            W[m2] = val_low * (f[m2] / f_mid_start) ** alpha
        else:
            W[m2] = val_low + (val_flat - val_low) * (
                (f[m2] - f_mid_start) / (f_mid_end - f_mid_start)
            )

    # seg 3: f_flat_start..f_flat_end (flat at 1.0)
    m3 = (f > f_flat_start) & (f <= f_flat_end)
    W[m3] = val_flat

    # seg 4: > f_flat_end (power-law tail; straight on log–log)
    m4 = (f > f_flat_end)
    W[m4] = val_flat * (f_flat_end / f[m4]) ** tail_exponent

    # --- plot (match appearance from analyze_vibration) ---
    if plot:
        mask = f > 0  # avoid log(0)
        plt.figure(figsize=(8, 4.5))
        plt.loglog(f[mask], W[mask])

        # X ticks and labels
        octave_centers = np.array([0.016, 0.0315, 0.063, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 31.5, 63])
        plt.xticks(octave_centers, [str(x) for x in octave_centers])

        # Y ticks/limits to match your other plot
        weightticks = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
        plt.yticks(weightticks, [str(y) for y in weightticks])
        plt.ylim([0.005, 1.5])
        plt.xlim(0.016, 63)

        # Axis labels (same naming)
        plt.xlabel("$f$ (Hz)", fontsize=12)
        plt.ylabel("Weighting Factor", fontsize=12)

        # Grid + layout
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()

        # Knot overlays (match style)
        if show_knots:
            ax = plt.gca()

            # vertical knot lines & labels
            x_knots = [
                (f_low,        r"$f_1$", "r"),
                (f_mid_start,  r"$f_2$", "r"),
                (f_mid_end,    r"$f_3$", "r"),
                (f_flat_end,   r"$f_4$", "r"),
            ]
            # horizontal knot (w)
            y_knot = (low_gain, r"$w$", "g")

            # draw lines + text
            for x, lab, color in x_knots:
                ax.axvline(x, ls="--", lw=1.0, color=color, alpha=0.7)
                ylab = max(np.min(W[W > 0]), 1e-3)
                ax.text(x, 1.1 * ylab, lab, color=color,
                        rotation=90, va="bottom", ha="right",
                        fontsize=15, fontweight="bold")

            ax.axhline(y_knot[0], ls="--", lw=1.0, color=y_knot[2], alpha=0.7)
            ax.text(0.018, y_knot[0]*1.05, y_knot[1], color=y_knot[2],
                    va="bottom", ha="left", fontsize=15, fontweight="bold")

            # put numeric values on axes (same trick with transforms)
            trans_x = ax.get_xaxis_transform()
            trans_y = ax.get_yaxis_transform()
            y_offset_xaxis = -0.025
            x_offset_yaxis = -0.048

            for x, _, color in x_knots:
                ax.text(
                    x, y_offset_xaxis, f"{x:.1f}",
                    color=color, fontsize=10, fontweight="bold",
                    ha="center", va="top", transform=trans_x,
                    clip_on=False,
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.2)
                )

            ax.text(
                x_offset_yaxis, y_knot[0], f"{y_knot[0]:g}",
                color=y_knot[2], fontsize=10, fontweight="bold",
                ha="left", va="center", transform=trans_y,
                clip_on=False,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.2)
            )

        plt.show()

    return f, W
