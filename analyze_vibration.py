import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from tabulate import tabulate

# ---- Function Parameters ----
# file_path: Path to the CSV file containing acceleration data
# low_gain: Gain for the flat, low frequency region (m/s²)
# f_low: Position of lower edge of flat, low frequency region (Hz)
# f_mid_start: Position of start of ramp up (segment 2) (Hz)
# f_mid_end: Position of end of ramp up (segment 2) (Hz)
# f_flat_start: Position of start of flat, most sensitive region (segment 3) (Hz)
# f_flat_end: Position of end of flat, most sensitive region (Hz)

# If when function is called, the user does not specify the parameters, the default values will be used.
def analyze_vibration(file_path, trial, low_gain=0.4, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0, tail_exponent=1.0, log_ramp=True, show_knots=True, plot=True):
    def _validate_breakpoints(low_gain, f_low, f_mid_start, f_mid_end, f_flat_end):
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
    # Read CSV (use utf-8 to be safe with the superscript ²)
    df = pd.read_csv(file_path, encoding="utf-8")
    accel_data = df[trial].to_numpy()
    # t = df["Time (s)"].to_numpy()

    accel_mean = np.mean(accel_data)
    acceleration = accel_data - accel_mean  # center

    # 1) Time & FFT prep
    dt = 0.01  # actual average time step from the CSV
    fs = 1.0 / dt
    n = len(acceleration)
    t = np.arange(n) * dt
    
    freq_vector = fftfreq(n, d=dt)
    acc_fft = fft(acceleration)
    magnitude = np.abs(acc_fft) / n
    
    # 2) Weighting on positive freqs
    positive_freqs = freq_vector[:n//2 + 1]
    positive_freqs = np.maximum(positive_freqs, 0)  # clamp any small negatives to 0
    positive_magnitude = magnitude[:n//2 + 1]
    W = np.zeros_like(positive_freqs)
    val_low = low_gain
    val_flat = 1.0
    f_flat_start = f_mid_end
    
    # seg 0: 0..f_low (linear rise to low plateau)
    mask_0 = (positive_freqs > 0.0) & (positive_freqs < f_low)
    W[mask_0] = val_low * (positive_freqs[mask_0] / f_low)
    
    # seg 1: f_low..f_mid_start (low plateau)
    mask_1 = (positive_freqs >= f_low) & (positive_freqs <= f_mid_start)
    W[mask_1] = val_low

    # seg 2: f_mid_start..f_mid_end (ramp up to 1.0)
    mask_2 = (positive_freqs > f_mid_start) & (positive_freqs <= f_mid_end)
    if np.any(mask_2):
        if log_ramp:
            # power-law ramp so it is straight on a log–log plot:
            # W = val_low * (f / f_mid_start) ** alpha
            alpha = np.log(val_flat / val_low) / np.log(f_mid_end / f_mid_start)
            W[mask_2] = val_low * (positive_freqs[mask_2] / f_mid_start) ** alpha
        else:
            # linear-in-frequency ramp (will look slightly curved on log–log)
            W[mask_2] = val_low + (val_flat - val_low) * (
                (positive_freqs[mask_2] - f_mid_start) / (f_mid_end - f_mid_start)
            )

    # seg 3: f_flat_start..f_flat_end (flat at 1.0)
    mask_3 = (positive_freqs > f_flat_start) & (positive_freqs <= f_flat_end)
    W[mask_3] = val_flat

    # seg 4: > f_flat_end (≈ 1/f roll-off, 6 dB/oct)
    mask_4 = (positive_freqs > f_flat_end)
    W[mask_4] = val_flat * (f_flat_end / positive_freqs[mask_4]) ** tail_exponent

    # Mirror weights to full spectrum --> magnitude spectrum is symmetric --> |X(f)| = |X(-f)| --> Mirror the positive half of the weighting function to the negative side of the FFT before applying it.
    weights = np.ones_like(freq_vector) # same size as the FFT frequency vector (positive + negative frequencies)
    weights[:n//2 + 1] = W # first half of frequencies --> positive frequencies (0 Hz to Nyquist) → Fill the first half (DC to Nyquist) with the computed weighting curve W(f)
    if n % 2 == 0: # if n is even, the FFT has a unique Nyquist frequency at n/2 → Copy everything except DC (index 0) and Nyquist (n//2), then reverse it ([::-1]) to mirror the weighting.
        weights[n//2 + 1:] = W[1:n//2][::-1] 
    else: # if n is odd, there is no unique Nyquist frequency → Copy everything except DC (index 0), then reverse it to mirror the weighting.
        weights[n//2 + 1:] = W[1:(n//2)+1][::-1]
    # This ensures the weighting function is symmetric about 0 Hz: W(-f) = W(f)

    # 3) Apply weighting in freq domain
    weighted_fft = acc_fft * weights # weighted magnitude spectrum in freq domain but is complex-valued (negative frequencies too)
    weighted_signal = np.real(ifft(weighted_fft)) # weighted acceleration in time domain

    weighted_magnitude = np.abs(weighted_fft) / n # weighted magnitude spectrum (positive + negative frequencies)
    weighted_magnitude_pos = weighted_magnitude[:n//2 + 1]

    # --------- METRICS ----------
    peak_unw = np.max(np.abs(acceleration))
    peak_w   = np.max(np.abs(weighted_signal))

    rms_unw = np.sqrt(np.mean(acceleration**2))
    rms_w   = np.sqrt(np.mean(weighted_signal**2))

    # Running RMS (1 s window)
    window_duration = 1.0
    window_size = int(fs * window_duration)

    def running_rms(signal, win):
        return np.sqrt(np.convolve(signal**2, np.ones(win)/win, mode='valid'))

    rms_running_unweighted = running_rms(acceleration, window_size)
    rms_running_weighted   = running_rms(weighted_signal, window_size)

    # MTVV (max running RMS)
    mtvv_unw = np.max(rms_running_unweighted)
    mtvv_w   = np.max(rms_running_weighted)

    mtvv_root2_unw = np.sqrt(2) * mtvv_unw
    mtvv_root2_w   = np.sqrt(2) * mtvv_w

    cf_unw = peak_unw / rms_unw
    cf_w   = peak_w   / rms_w

    vdv_unw = (np.sum(acceleration**4) * dt) ** 0.25
    vdv_w   = (np.sum(weighted_signal**4) * dt) ** 0.25

    R_unw = mtvv_unw / 0.005
    R_w   = mtvv_w   / 0.005

    metrics = {
        "Peak (m/s^2)":      [peak_unw, peak_w],
        "RMS (m/s^2)":       [rms_unw, rms_w],
        "MTVV (m/s^2)":      [mtvv_unw, mtvv_w],
        "MTVV*√2 (m/s^2)":   [mtvv_root2_unw, mtvv_root2_w],
        "CF":                [cf_unw, cf_w],
        "VDV (m/s^1.75)":    [vdv_unw, vdv_w],
        "R":                 [R_unw, R_w]
    }

    # --------- PLOTS ----------
    # Frequency weighting
    mask = positive_freqs > 0 # avoid plotting the zero frequency point (0 Hz) on log-log scale since log(0)→−∞
    if plot:
        plt.figure(figsize=(8, 4.5))
        plt.loglog(positive_freqs[mask], W[mask]) 
        octave_centers = np.array([0.016, 0.0315, 0.063, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 31.5, 63])
        plt.xticks(octave_centers, [str(f) for f in octave_centers])
        # Comment out if you want to hover over y-axis values
        plt.yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.5], ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "1.5"])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Frequency Weighting")
        # plt.title("Asymptotic Approximation of Vertical Frequency Weighting")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
    if plot:
        if show_knots:
            # vertical knots for frequency breakpoints
            for x, lab in [(f_low, r"$\it{f_{low}}$"), (f_mid_start, r"$\it{f_{midStart}}$"),
                        (f_mid_end, r"$\it{f_{midEnd}}$"), (f_flat_end, r"$\it{f_{flatEnd}}$")]:
                plt.axvline(x, ls="--", lw=1.0, color="r", alpha=0.6)
                ylab = max(np.min(W[W > 0]), 1e-3)
                plt.text(x, 1.1 * ylab, lab, color="r",
                        rotation=90, va="bottom", ha="right", fontsize=14)

        # horizontal knot for lowGain (place label near the left x-limit actually used)
        x_left = float(np.min(positive_freqs[mask])) if np.any(mask) else f_low
        plt.axhline(low_gain, ls="--", lw=1.0, color="g", alpha=0.6)
        plt.text(x_left * 1.1, low_gain * 1.05, r"$\it{lowGain}$", color="g",
                va="bottom", ha="left", fontsize=10)

    # --- PLOT: Running 1 s RMS vs time (centered timestamps) ---
    # For mode='valid', the i-th value corresponds to samples [i, i+win-1].
    # Use the midpoint time: t_center[i] = (i + (win-1)/2) / fs
    Lr = len(rms_running_weighted)
    t_rms = np.arange(Lr) / fs + (window_size - 1) / (2.0 * fs)
    
    # Time histories/Running 1 s RMS
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(t, acceleration, label="Acceleration (unweighted)")
        plt.plot(t_rms, rms_running_unweighted, label="Running RMS (1 s) (unweighted)")
        plt.plot(t, weighted_signal, label="Acceleration (weighted)")
        plt.plot(t_rms, rms_running_weighted, label="Running RMS (1 s) (weighted)")
        # plt.title("Time History of Unweighted Acceleration")
        plt.xlabel("Time (s)", fontsize=15)
        plt.ylabel("Acceleration (m/s²)", fontsize=15)
        plt.yticks(np.arange(-1.2, 1.2, step=0.2))
        plt.ylim(-1.1, 1.1)
        plt.legend(loc='lower left', fontsize=13)
        plt.grid(True)
        plt.tight_layout()
    
    # # Time histories/Running 1 s RMS (weighted)
    # plt.figure(figsize=(10, 5))
    # plt.plot(t, weighted_signal, label="Acceleration")
    # plt.plot(t_rms, rms_running_weighted, label="Running RMS (1 s)")
    # # plt.title("Time History of Weighted Acceleration")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Acceleration (m/s²)")
    # plt.yticks(np.arange(-1.2, 1.2, step=0.2))
    # plt.ylim(-1.1, 1.1)
    # plt.legend(loc='upper left')
    # plt.grid(True)
    # plt.tight_layout()

    # Spectra FFT (unweighted vs weighted)
    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(positive_freqs, positive_magnitude, label="Unweighted")
        plt.plot(positive_freqs, weighted_magnitude_pos, label="Weighted")
        # plt.title("FFT - Frequency Spectra")
        plt.xlabel("Frequency (Hz)", fontsize=12)
        plt.ylabel("Amplitude (m/s²)", fontsize=12)
        plt.legend(fontsize=12)
        plt.xticks(np.arange(0, 20, step=2))
        plt.xlim(0, 20)
        # plt.ylim(0, 0.031)
        plt.grid(True)
        plt.tight_layout()
    
    df_metrics = pd.DataFrame(metrics, index=["Unweighted", "Weighted"])
    print(tabulate(df_metrics.round(6), headers='keys', tablefmt='grid', numalign="center", stralign="center"))
    if plot:
        plt.show()

    return t, acceleration, weighted_signal, t_rms, rms_running_unweighted, rms_running_weighted, df_metrics, positive_freqs, positive_magnitude, weighted_magnitude_pos