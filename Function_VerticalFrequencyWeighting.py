import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.fft import fft, ifft, fftfreq
from tabulate import tabulate
from typing import List, Optional

# ---- Function Parameters ----
# file_path: Path to the CSV file containing acceleration data
# trial_index: Index of the trial to analyze
# low_gain: Gain for the flat, low frequency region (m/s²)
# f_low: Position of lower edge of flat, low frequency region (Hz)
# f_mid_start: Position of start of ramp up (segment 2) (Hz)
# f_mid_end: Position of end of ramp up (segment 2) (Hz)
# f_flat_start: Position of start of flat, most sensitive region (segment 3) (Hz)
# f_flat_end: Position of end of flat, most sensitive region (Hz)

# If when function is called, the user does not specify the parameters, the default values will be used.
def analyze_vibration(file_path, trial_index=0, fs=100, low_gain=0.4, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0):
    def _validate_breakpoints(low_gain, f_low, f_mid_start, f_mid_end, f_flat_end):
        # Enforce monotonic increasing breakpoints and sensible ranges
        eps = 1e-6
        f_low       = max(eps, f_low)
        f_mid_start = max(f_low + eps, f_mid_start)
        f_mid_end   = max(f_mid_start + eps, f_mid_end)
        f_flat_end  = max(f_mid_end + eps, f_flat_end)
        low_gain    = float(low_gain)
        return low_gain, f_low, f_mid_start, f_mid_end, f_flat_end
    
    low_gain, f_low, f_mid_start, f_mid_end, f_flat_end = _validate_breakpoints(
        low_gain, f_low, f_mid_start, f_mid_end, f_flat_end
    )

    # Load data
    data = pd.read_csv(file_path)
    accel_data = data.iloc[trial_index, 1:].values
    # The above code snippet in Python is setting the variable `fs` to a value of 100, which
    # represents the sampling rate in Hertz. This means that the data is being sampled at a rate of
    # 100 samples per second, with a time step of 0.01 seconds between each sample.
    accel_mean = np.mean(accel_data)
    acceleration =  accel_data - accel_mean  # Remove mean to center (spike) around zero

    # 1. Setup time and sampling
    dt = 1 / fs
    n = len(acceleration)
    t = np.arange(n) * dt

    # 2. Frequency vector for FFT
    freq_vector = fftfreq(n, d=dt)
    acc_fft = fft(acceleration)
    magnitude = np.abs(acc_fft) / n  # Normalize

    # 3. Create weighting vector (same length as FFT)
    weights = np.ones_like(freq_vector)

    # Apply weighting only for positive frequencies (mirror for negative)
    positive_freqs = freq_vector[:n//2 + 1]
    W = np.zeros_like(positive_freqs)
    positive_magnitude = magnitude[:n//2 + 1]

    # Create weighting parameters
    f_flat_start = f_mid_end  # Start of flat 1.0 region (segment 3)

    # Value at the end of Segment 1 (flat) and start of Segment 2 (ramp)
    val_low = low_gain
    val_flat = 1.0 # Most sensitive region always at 0dB or 1m/s²

    # Segment 0: 0.0–f_low — linear ramp to val_low
    mask_0 = (positive_freqs > 0.0) & (positive_freqs < f_low)
    W[mask_0] = val_low * (positive_freqs[mask_0] / f_low)

    # Segment 1: f_low–f_mid_start — constant at val_low
    mask_1 = (positive_freqs >= f_low) & (positive_freqs <= f_mid_start)
    W[mask_1] = val_low

    # Segment 2: f_mid_start–f_mid_end — linear ramp to val_flat
    mask_2 = (positive_freqs > f_mid_start) & (positive_freqs <= f_mid_end)
    if f_mid_end > f_mid_start:
        W[mask_2] = val_low + (val_flat - val_low) * (
            (positive_freqs[mask_2] - f_mid_start) / (f_mid_end - f_mid_start)
        )
    else:
        # degenerate safeguard: collapse to low plateau
        W[mask_2] = val_low

    # Segment 3: f_flat_start–f_flat_end — constant at val_flat
    mask_3 = (positive_freqs > f_flat_start) & (positive_freqs <= f_flat_end)
    W[mask_3] = val_flat

    # Segment 4: > f_flat_end — decaying 6 dB/oct slope (W = val_flat * f_flat_end / f)
    mask_4 = (positive_freqs > f_flat_end)
    W[mask_4] = val_flat * f_flat_end / positive_freqs[mask_4]

    # Apply to full spectrum (positive + symmetric negative side)
    weights[:n//2 + 1] = W

    # Assign weights (mirror)
    weights[:n//2 + 1] = W
    if n % 2 == 0:
        weights[n//2 + 1:] = W[1:n//2][::-1]
    else:
        weights[n//2 + 1:] = W[1:(n//2)+1][::-1]

    # 4. Apply weighting to FFT
    weighted_fft = acc_fft * weights

    # 5. Inverse FFT to get time-weighted signal
    weighted_signal = np.real(ifft(weighted_fft)) # Weighted acceleration time-history

    # 6. Compute magnitude spectrum of weighted FFT
    weighted_magnitude = np.abs(weighted_fft) / n  # Normalized magnitude

    # 7. Extract positive frequencies only
    weighted_magnitude_pos = weighted_magnitude[:n//2 + 1]

    # ----------------- Plot Frequency Weighting -----------------
    plt.figure(figsize=(10, 5))
    plt.loglog(positive_freqs, W, label="Apply to acceleration data in units of m/s²") # Plot in m/s²
    # plt.semilogx(positive_freqs, 20 * np.log10(W), label="Apply to acceleration data in units of dB") # Plot in dB

    # Define 1/3-octave frequency axis ticks --> Comment out if you want to see frequency position when you hover cursor over the plot
    octave_centers = np.array([0.016, 0.0315, 0.063, 0.125, 0.25, 0.5,
                            1, 2, 4, 8, 16, 31.5, 63])
    plt.xticks(octave_centers, [str(f) for f in octave_centers]) # Apply custom ticks and labels to x-axis

    # Apply custom ticks and labels to y-axis - Acceleration, modulus, gain
    # yaxis = np.array([0.01, 0.1, 1.0])
    # plt.yticks(yaxis, [str(f) for f in yaxis])
    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Frequency Weighting")
    plt.title("Asymptotic Approximation of Vertical Frequency Weighting")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # ----------------- Compare Unweighted and Weighted Acceleration Time-History -----------------
    plt.figure(figsize=(10, 5))
    plt.plot(t, acceleration, label="Unweighted", color='blue')
    plt.plot(t, weighted_signal, label="Weighted", color='orange')
    plt.title("Acceleration Time-History")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    # ----------------- Compare Unweighted and Weighted Spectra on the Same Plot -----------------
    plt.figure(figsize=(10, 5))
    plt.plot(positive_freqs, positive_magnitude, label="Unweighted", color='blue')
    plt.plot(positive_freqs, weighted_magnitude_pos, label="Weighted", color='orange')
    plt.title("FFT - Frequency Spectra")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()
    plt.xlim(0, 20)
    plt.ylim(0, 0.031)
    plt.grid(True)
    plt.tight_layout()

    # ----------------- Vibration Serviceability Metrics -----------------
    # Peak Acceleration 
    peak_unw = np.max(np.abs(acceleration))
    peak_w = np.max(np.abs(weighted_signal))

    # RMS Acceleration
    rms_unw = np.sqrt(np.mean(acceleration ** 2))
    rms_w = np.sqrt(np.mean(weighted_signal ** 2))

    # MTVV (Running RMS, 1s)
    # Define window parameters
    window_duration = 1.0  # seconds
    window_size = int(fs * window_duration)  # number of samples in 1 second
    # Function to calculate running RMS
    def running_rms(signal, window_size):
        return np.sqrt(np.convolve(signal**2, np.ones(window_size)/window_size, mode='valid'))
    # Calculate running RMS for unweighted and weighted signals
    rms_running_unweighted = running_rms(acceleration, window_size)
    rms_running_weighted = running_rms(weighted_signal, window_size)
    # Find the maximum RMS in those running windows
    mtvv_unw = np.max(rms_running_unweighted)
    mtvv_w = np.max(rms_running_weighted)

    # MTVV*sqrt(2) = Peak Acceleration
    mtvv_root2_unw = np.sqrt(2)*mtvv_unw
    mtvv_root2_w = np.sqrt(2)*mtvv_w

    # Crest Factor (CF)
    cf_unw = peak_unw / rms_unw
    cf_w = peak_w / rms_w

    # Vibration Dose Value (VDV)
    vdv_unw = (np.sum(acceleration**4) * dt) ** 0.25
    vdv_w = (np.sum(weighted_signal**4) * dt) ** 0.25

    # Response Factor (R)
    R_unw = mtvv_unw/0.005
    R_w = mtvv_w/0.005

    # Make Table of Vibration Serviceability Metrics
    metrics = {
        "Peak (m/s^2)": [peak_unw, peak_w],
        "RMS (m/s^2)": [rms_unw, rms_w],
        "MTVV (m/s^2)": [mtvv_unw, mtvv_w],
        "MTVV*sqrt(2) (m/s^2)": [mtvv_root2_unw, mtvv_root2_w],
        "CF": [cf_unw, cf_w],
        "VDV (m/s^1.75)": [vdv_unw, vdv_w],
        "R": [R_unw, R_w]
    }
    # Create DataFrame
    df_metrics = pd.DataFrame(metrics, index=["Unweighted", "Weighted"])
    # Display table
    print(tabulate(df_metrics.round(6), headers='keys', tablefmt='grid', numalign="center", stralign="center"))
    
    plt.show()
    return df_metrics   

# df = analyze_vibration(
#     file_path="results_500mc_trial_matrix.csv",
#     low_gain=0.4,
#     f_low=0.5,
#     f_mid_start=2.0,
#     f_mid_end=5.0,
#     f_flat_end=16.0
# )

# Calling function from another file:
# - Comment out the function call above (line 198)
# from Function_VerticalFrequencyWeighting import analyze_vibration

# ===========================
# Core helpers (no plotting)
# ===========================
def _build_weighting_array(positive_freqs, low_gain, f_low, f_mid_start, f_mid_end, f_flat_end):
    W = np.zeros_like(positive_freqs, dtype=float)
    val_low = low_gain
    val_flat = 1.0
    f_flat_start = f_mid_end

    # seg 0: 0..f_low (linear up to low plateau)
    mask_0 = (positive_freqs > 0.0) & (positive_freqs < f_low)
    W[mask_0] = val_low * (positive_freqs[mask_0] / f_low)

    # seg 1: f_low..f_mid_start (low plateau)
    mask_1 = (positive_freqs >= f_low) & (positive_freqs <= f_mid_start)
    W[mask_1] = val_low

    # seg 2: f_mid_start..f_mid_end (ramp up to 1.0)
    mask_2 = (positive_freqs > f_mid_start) & (positive_freqs <= f_mid_end)
    if f_mid_end > f_mid_start:
        W[mask_2] = val_low + (val_flat - val_low) * (
            (positive_freqs[mask_2] - f_mid_start) / (f_mid_end - f_mid_start)
        )
    else:
        W[mask_2] = val_low  # degenerate safeguard

    # seg 3: f_flat_start..f_flat_end (flat at 1.0)
    mask_3 = (positive_freqs > f_flat_start) & (positive_freqs <= f_flat_end)
    W[mask_3] = val_flat

    # seg 4: > f_flat_end (6 dB/oct ≈ 1/f)
    mask_4 = (positive_freqs > f_flat_end)
    W[mask_4] = np.where(positive_freqs[mask_4] > 0.0, val_flat * f_flat_end / positive_freqs[mask_4], 0.0)

    # DC exactly zero
    W[positive_freqs == 0.0] = 0.0
    return W


def _metrics_no_plots(acceleration, fs, low_gain, f_low, f_mid_start, f_mid_end, f_flat_end):
    """Compute metrics without plotting; returns dict with _unweighted and _weighted values."""
    acceleration = np.asarray(acceleration, dtype=float)
    acceleration = acceleration - np.mean(acceleration)

    n = len(acceleration)
    dt = 1.0 / fs

    acc_fft = fft(acceleration)
    freq_vector = fftfreq(n, d=dt)

    # positive side
    posN = n // 2 + 1
    positive_freqs = freq_vector[:posN]

    # build weights (positive) and mirror to full spectrum
    Wpos = _build_weighting_array(positive_freqs, low_gain, f_low, f_mid_start, f_mid_end, f_flat_end)
    weights = np.empty_like(freq_vector, dtype=float)
    weights[:posN] = Wpos
    if n % 2 == 0:
        weights[posN:] = Wpos[1:posN-1][::-1]
    else:
        weights[posN:] = Wpos[1:posN][::-1]

    weighted_fft = acc_fft * weights
    weighted_signal = np.real(ifft(weighted_fft))

    # --- metrics ---
    peak_unw = float(np.max(np.abs(acceleration)))
    peak_w   = float(np.max(np.abs(weighted_signal)))

    rms_unw = float(np.sqrt(np.mean(acceleration**2)))
    rms_w   = float(np.sqrt(np.mean(weighted_signal**2)))

    # MTVV: running RMS with 1 s window
    window = int(fs * 1.0)
    def running_rms(sig, w):
        return np.sqrt(np.convolve(sig**2, np.ones(w)/w, mode='valid'))
    mtvv_unw = float(np.max(running_rms(acceleration, window)))
    mtvv_w   = float(np.max(running_rms(weighted_signal, window)))

    mtvv_root2_unw = float(np.sqrt(2.0) * mtvv_unw)
    mtvv_root2_w   = float(np.sqrt(2.0) * mtvv_w)

    cf_unw = float(peak_unw / rms_unw) if rms_unw > 0 else np.inf
    cf_w   = float(peak_w   / rms_w)   if rms_w   > 0 else np.inf

    vdv_unw = float((np.sum(acceleration**4) * dt) ** 0.25)
    vdv_w   = float((np.sum(weighted_signal**4) * dt) ** 0.25)

    R_unw = float(mtvv_unw / 0.005)
    R_w   = float(mtvv_w   / 0.005)

    return {
        "Peak_unweighted": peak_unw,  "Peak_weighted": peak_w,
        "RMS_unweighted": rms_unw,    "RMS_weighted": rms_w,
        "MTVV_unweighted": mtvv_unw,  "MTVV_weighted": mtvv_w,
        "MTVV_sqrt2_unweighted": mtvv_root2_unw, "MTVV_sqrt2_weighted": mtvv_root2_w,
        "CF_unweighted": cf_unw,      "CF_weighted": cf_w,
        "VDV_unweighted": vdv_unw,    "VDV_weighted": vdv_w,
        "R_unweighted": R_unw,        "R_weighted": R_w
    }


def _load_trial_series(file_path, trial_index=0):
    """Helper: load one trial row (col0 label, others are samples) and return acceleration array."""
    data = pd.read_csv(file_path)
    accel_data = data.iloc[trial_index, 1:].values.astype(float)
    return accel_data

# ====================================================
# Local derivative-based sensitivity at a baseline
# one-at-a-time (OAT) local, derivative-based sensitivity using central finite differences
# ====================================================
def local_sensitivity_from_file(
    file_path,
    trial_index=0,
    fs=100,
    base_params=None,
    rel_step=0.05,
    metrics_to_track=None  # ← default None => all 7 weighted metrics
):
    """
    Computes one-at-a-time sensitivities at baseline using symmetric finite differences.
    Prints tabulated Elasticity and dY/dP tables (CF & R unitless; parameters shown without units).
    Returns:
      - df_elasticity (programmatic columns)
      - df_dydp       (programmatic columns)
    """
    if base_params is None:
        base_params = dict(low_gain=0.4, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0)

    # Default to ALL seven weighted metrics
    if metrics_to_track is None:
        metrics_to_track = (
            "Peak_weighted",
            "RMS_weighted",
            "MTVV_weighted",
            "MTVV_sqrt2_weighted",
            "CF_weighted",
            "VDV_weighted",
            "R_weighted",
        )

    # ---- pretty metric names & units (CF, R unitless) ----
    metric_pretty = {
        "Peak_weighted": "Peak",
        "RMS_weighted": "RMS",
        "MTVV_weighted": "MTVV",
        "MTVV_sqrt2_weighted": "MTVV*sqrt(2)",
        "CF_weighted": "CF",
        "VDV_weighted": "VDV",
        "R_weighted": "R",
    }
    metric_units = {
        "Peak_weighted": "m/s^2",
        "RMS_weighted": "m/s^2",
        "MTVV_weighted": "m/s^2",
        "MTVV_sqrt2_weighted": "m/s^2",
        "CF_weighted": "",          # unitless
        "VDV_weighted": "m/s^1.75",
        "R_weighted": "",           # unitless
    }

    # ---- load and baseline ----
    accel = _load_trial_series(file_path, trial_index=trial_index)
    m0 = _metrics_no_plots(accel, fs, **base_params)
    y0 = np.array([m0[k] for k in metrics_to_track], dtype=float)

    param_order = ["low_gain", "f_low", "f_mid_start", "f_mid_end", "f_flat_end"]
    dydp_mat = np.zeros((len(param_order), len(metrics_to_track)), dtype=float)
    elast_mat = np.zeros_like(dydp_mat)

    for i, pname in enumerate(param_order):
        p0 = float(base_params[pname])
        step = rel_step * max(abs(p0), 1e-12)

        # guard: keep frequencies positive on the down step
        p_down = p0 - step
        if pname.startswith("f_") and p_down <= 0.0:
            p_down = p0

        # evaluate down
        par_down = dict(base_params); par_down[pname] = p_down
        y_down = np.array([_metrics_no_plots(accel, fs, **par_down)[k] for k in metrics_to_track], dtype=float)

        # evaluate up
        p_up = p0 + step
        par_up = dict(base_params); par_up[pname] = p_up
        y_up = np.array([_metrics_no_plots(accel, fs, **par_up)[k] for k in metrics_to_track], dtype=float)

        dy = y_up - y_down
        dp = (p_up - p_down) if (p_up - p_down) != 0 else step
        dydp = dy / dp
        dydp_mat[i, :] = dydp

        # elasticity = (dy/y0) / (dp/p0) = dydp * (p0 / y0)
        with np.errstate(divide='ignore', invalid='ignore'):
            elast = dydp * (p0 / y0)
        elast[~np.isfinite(elast)] = np.nan
        elast_mat[i, :] = elast

    # Programmatic DataFrames (for code use)
    df_elasticity = pd.DataFrame(elast_mat, index=param_order, columns=metrics_to_track)
    df_dydp       = pd.DataFrame(dydp_mat,   index=param_order, columns=metrics_to_track)

    # ---------- Pretty printing ----------
    # 1) Elasticity (dimensionless) – pretty metric names
    df_elast_pretty = df_elasticity.rename(columns=metric_pretty)
    df_elast_print  = df_elast_pretty.round(6).replace({np.nan: ""})
    print("\nElasticity (% change output / % change input):")
    print(tabulate(df_elast_print, headers="keys", tablefmt="grid",
                   numalign="center", stralign="center"))

    # 2) Partial derivatives – metric units only in headers; parameters shown without units
    col_with_units = {
        k: f"{metric_pretty[k]}{f' ({metric_units[k]})' if metric_units[k] else ''}"
        for k in df_dydp.columns
    }
    df_dydp_u = df_dydp.rename(columns=col_with_units).copy()
    df_dydp_u.index = list(df_dydp.index)  # raw param names only (no units)

    df_dydp_print = df_dydp_u.round(6).replace({np.nan: ""})
    print("\nPartial derivatives dY/dP")
    print(tabulate(df_dydp_print, headers="keys", tablefmt="grid",
                   numalign="center", stralign="center"))

    # Return programmatic frames (without prettified labels) for plotting/saving
    return df_elasticity, df_dydp

# ====================================================
# Tornado plots for local elasticities
# ====================================================
def tornado_grid_elasticity(
    df_elasticity,
    metrics_to_plot: Optional[List[str]] = None,
    ncols: int = 2,
    sharex: bool = True,
    annotate: bool = True,
    figsize_per: tuple = (6, 3.8),  # width, height per subplot
    title: str = "Local Sensitivity (Elasticity)",
    tight_layout: bool = True,
    save_path: Optional[str] = None
):
    """
    Render all tornado charts (elasticity) as subplots in a single figure.

    Parameters
    ----------
    df_elasticity : pd.DataFrame
        Rows = parameters, Columns = metrics; entries = elasticity (%Δout / %Δin).
    metrics_to_plot : list[str] or None
        Which columns to plot. Default = all columns in df.
    ncols : int
        Number of subplot columns.
    sharex : bool
        If True, all subplots share the same x-axis limits.
    annotate : bool
        If True, writes numeric elasticity on the bar ends.
    figsize_per : (w, h)
        Size per subplot; final figure size scales by number of subplots.
    title : str
        Overall figure title.
    tight_layout : bool
        Apply tight_layout at the end.
    save_path : str or None
        If provided, saves the whole figure to this path (e.g., "figs/tornado_grid.png").
    """
    if metrics_to_plot is None:
        metrics_to_plot = list(df_elasticity.columns)

    nplots = len(metrics_to_plot)
    nrows = math.ceil(nplots / ncols)
    fig_w = figsize_per[0] * ncols
    fig_h = figsize_per[1] * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=sharex)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Determine a common x-limit if sharex, based on all metrics
    if sharex:
        all_vals = []
        for metric in metrics_to_plot:
            s = df_elasticity[metric].dropna()
            all_vals.extend(s.values.tolist())
        xabs = np.nanmax(np.abs(all_vals)) if len(all_vals) else 1.0
        # Pad a little for labels
        xlim = (-1.05 * xabs, 1.05 * xabs)
    else:
        xlim = None

    # Draw each metric
    for idx, metric in enumerate(metrics_to_plot):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        s = df_elasticity[metric].copy()
        order = np.argsort(np.abs(s.values))[::-1]
        s_sorted = s.iloc[order]

        y = np.arange(len(s_sorted))
        ax.barh(y, s_sorted.values)
        ax.set_yticks(y)
        ax.set_yticklabels(s_sorted.index)
        ax.axvline(0.0, linewidth=1.0)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.5)
        ax.set_title(metric)

        if sharex:
            ax.set_xlim(*xlim)
        else:
            # individual padding
            local_max = np.nanmax(np.abs(s_sorted.values)) if len(s_sorted) else 1.0
            ax.set_xlim(-1.05 * local_max, 1.05 * local_max)

        if annotate:
            # Skip NaNs and place text slightly beyond the bar
            xmin, xmax = ax.get_xlim()
            span = xmax - xmin
            for yi, val in enumerate(s_sorted.values):
                if np.isnan(val):
                    continue
                xoff = 0.02 * (1 if val >= 0 else -1) * span
                ax.text(val + xoff, yi, f"{val:.2f}", va="center")


        # x-label only on bottom row
        if r == nrows - 1:
            ax.set_xlabel("Elasticity  (%Δoutput / %Δinput)")

    # If there are extra axes (when nrows*ncols > nplots), hide them
    for extra in range(nplots, nrows * ncols):
        r = extra // ncols
        c = extra % ncols
        axes[r, c].axis("off")

    # Overall title
    fig.suptitle(title, y=0.995)

    if tight_layout:
        plt.tight_layout(rect=(0, 0, 1, 0.97))  # leave room for suptitle

    if save_path:
        plt.savefig(save_path, dpi=200)

    plt.show()

# ====================================================
# Range sweep: user-defined ranges → % change w.r.t. baseline
# Plot sweep to see how metrics change as one weighting curve parameter is swept
# - one-at-a-time global sweep (but not derivative-based)
# - Show how metrics change when you move a parameter across a user-defined range with others fixed
# - 1.Set that parameter to the grid value (others at baseline). 2.Recompute metrics. 3.Report % change relative to baseline:
# ====================================================
def plot_sweep_multi_metric(
    file_path,
    trial_index=0,
    fs=100,
    base_params=None,
    param_name="low_gain",
    param_range=(0.2, 0.6),
    samples=9,
    metrics=None,
    title=None,
    show_legend=True,
    legend_outside=True,
):
    """
    Plot % change (relative to baseline) for multiple metrics as the chosen parameter is swept.

    Parameters
    ----------
    file_path : str
        CSV path.
    trial_index : int
        Which row to load (col0 label, others are samples).
    fs : float
        Sampling rate (Hz).
    base_params : dict or None
        Baseline parameters {low_gain, f_low, f_mid_start, f_mid_end, f_flat_end}.
        Defaults to (0.4, 0.5, 2.0, 5.0, 16.0) if None.
    param_name : str
        Parameter to sweep: "low_gain", "f_low", "f_mid_start", "f_mid_end", or "f_flat_end".
    param_range : (float, float)
        Sweep range (inclusive endpoints).
    samples : int
        Number of points in the sweep (>= 2).
    metrics : list[str] or None
        Metrics to plot. Defaults to all seven weighted metrics.
        Valid keys: "Peak_weighted","RMS_weighted","MTVV_weighted",
                    "MTVV_sqrt2_weighted","CF_weighted","VDV_weighted","R_weighted"
    title : str or None
        Optional plot title. If None, a sensible default is used.
    show_legend : bool
        Whether to show a legend.
    legend_outside : bool
        If True, places legend outside the plot (top-right).

    Returns
    -------
    df : pd.DataFrame
        A DataFrame indexed by the swept parameter values with columns = metrics (% change vs baseline).
    """
    if base_params is None:
        base_params = dict(low_gain=0.4, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0)
    if param_name not in base_params:
        raise ValueError(f"param_name '{param_name}' not in base_params: {list(base_params.keys())}")
    if samples < 2:
        raise ValueError("samples must be >= 2")

    # Pretty names & units for legend (CF & R unitless)
    metric_pretty = {
        "Peak_weighted": "Peak",
        "RMS_weighted": "RMS",
        "MTVV_weighted": "MTVV",
        "MTVV_sqrt2_weighted": "MTVV*sqrt(2)",
        "CF_weighted": "CF",
        "VDV_weighted": "VDV",
        "R_weighted": "R",
    }
    metric_units = {
        "Peak_weighted": "m/s^2",
        "RMS_weighted": "m/s^2",
        "MTVV_weighted": "m/s^2",
        "MTVV_sqrt2_weighted": "m/s^2",
        "CF_weighted": "",        # unitless
        "VDV_weighted": "m/s^1.75",
        "R_weighted": "",         # unitless
    }
    def _pretty_with_units(k):
        u = metric_units.get(k, "")
        return f"{metric_pretty.get(k,k)} ({u})" if u else metric_pretty.get(k,k)

    # Default metrics = all 7 weighted
    if metrics is None:
        metrics = [
            "Peak_weighted","RMS_weighted","MTVV_weighted",
            "MTVV_sqrt2_weighted","CF_weighted","VDV_weighted","R_weighted"
        ]

    # Load once and compute baseline
    accel = _load_trial_series(file_path, trial_index=trial_index)
    baseline = _metrics_no_plots(accel, fs, **base_params)
    y0 = {m: baseline[m] for m in metrics}

    # Build sweep grid
    lo, hi = float(param_range[0]), float(param_range[1])
    grid = np.linspace(lo, hi, samples)

    rows = []
    for val in grid:
        params = dict(base_params)
        # keep frequencies > 0
        if param_name.startswith("f_") and val <= 0.0:
            val = max(1e-6, lo)
        params[param_name] = float(val)

        m = _metrics_no_plots(accel, fs, **params)
        row = {"param_value": val}
        for k in metrics:
            base = y0[k]
            row[k] = np.nan if base == 0 else 100.0 * (m[k] - base) / base
        rows.append(row)

    df = pd.DataFrame(rows).set_index("param_value")

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in metrics:
        if k not in df.columns:
            continue
        ax.plot(df.index.values, df[k].values, marker="o", label=_pretty_with_units(k))

    ax.set_xlabel(param_name)  # parameters shown without units
    ax.set_ylabel("% change of metrics relative to baseline")
    if title is None:
        ax.set_title(f"% change of selected metrics vs {param_name}")
    else:
        ax.set_title(title)
    ax.grid(True)

    if show_legend:
        if legend_outside:
            fig.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98))
        else:
            ax.legend()

    plt.tight_layout(rect=(0, 0, 0.96, 0.96) if legend_outside else None)
    plt.show()

    return df
