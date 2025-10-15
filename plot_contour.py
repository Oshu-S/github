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
    # shape controls for W(f)
    log_ramp=True,             # ramp (f_mid_start → f_mid_end) straight on log–log
    tail_exponent=1.0,         # high-freq slope: W ∝ f^{-beta} (beta=1 ⇒ −6 dB/oct)
    tail_db_per_oct=None,      # optional dB/oct override (e.g., 6, 9, 12) → sets beta
    # vertical translation (applied to C = scale / W)
    scale=1.0,                 # linear multiplier for entire contour
    offset_db=None,            # alternative: add this many dB to C (overrides scale)
    normalize_at=None,         # optional: set C(f_ref)=target_at_ref (sets scale)
    target_at_ref=1.0,
    # visuals
    show_knots=True, plot=True,
    title="Equivalent Comfort Contour (1 / Weighting)"
):
    """
    Build and (optionally) plot an equivalent comfort contour C(f) = scale / W(f),
    where W(f) is the asymptotic frequency weighting defined by the user parameters.

    Vertical translation options (choose one):
      - scale (linear): C ← scale * (1 / W)
      - offset_db: C_dB ← 20*log10(C) + offset_db  (overrides 'scale')
      - normalize_at: choose a reference frequency f_ref and set C(f_ref)=target_at_ref

    Tail steepness:
      - Set tail_exponent=beta for W ~ (f_flat_end/f)^beta (straight line on log–log).
      - Or pass tail_db_per_oct>0 to target a slope in dB/oct; e.g., 6 → beta≈1.
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
        # |dB/oct| = 20*log10(2^beta) → beta = |dB/oct| / (20*log10(2))
        tail_exponent = float(tail_db_per_oct) / (20.0 * np.log10(2.0))
    tail_exponent = max(tail_exponent, eps)

    # frequency grid (log-spaced for smooth log–log plots)
    f = np.logspace(np.log10(fmin), np.log10(fmax), n_points)

    # ---------- build weighting W(f) ----------
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
    W[m4] = val_flat * (f_flat_end / f[m4]) ** tail_exponent

    # Avoid divide-by-zero
    W = np.clip(W, eps, None)

    # ---------- build contour C(f) = scale / W(f) ----------
    C = 1.0 / W

    # apply vertical translation
    if normalize_at is not None:
        # set C(f_ref) = target_at_ref
        f_ref = float(normalize_at)
        # find nearest grid point
        idx = np.argmin(np.abs(f - f_ref))
        current = C[idx]
        if current > 0:
            scale = float(target_at_ref) / current

    if offset_db is not None:
        # convert linear C to dB, offset, back to linear
        C_db = 20.0 * np.log10(C) + float(offset_db)
        C = 10.0 ** (C_db / 20.0)
    else:
        C = float(scale) * C

    # ---------- plot ----------
    if plot:
        plt.figure(figsize=(12, 5))
        plt.loglog(f, C, label="Equivalent comfort contour (1/W)")

        # axis labels/ticks
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Equivalent comfort contour")
        octave_centers = np.array([0.016, 0.0315, 0.063, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 31.5, 63])
        plt.xticks(octave_centers, [str(x) for x in octave_centers])
        plt.title(title)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        if show_knots:
            # vertical knots for frequency breakpoints
            for x, lab in [(f_low, r"$f_{\mathrm{low}}$"),
                           (f_mid_start, r"$f_{\mathrm{midStart}}$"),
                           (f_mid_end, r"$f_{\mathrm{midEnd}}$"),
                           (f_flat_end, r"$f_{\mathrm{flatEnd}}$")]:
                plt.axvline(x, ls="--", lw=1.0, color="r", alpha=0.6)
                # label just above the lowest visible value to keep tidy
                ylab = max(np.min(C[C > 0]), 1e-3)
                plt.text(x, 1.1 * ylab, lab, color="r",
                         rotation=90, va="bottom", ha="right", fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.show()

    return f, C
