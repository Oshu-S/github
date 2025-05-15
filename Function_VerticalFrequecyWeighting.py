import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import freqs
from scipy.fft import fft, ifft
from tabulate import tabulate

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
    # Load data
    data = pd.read_csv(file_path)
    acceleration = data.iloc[trial_index, 1:].values

    # 1. Setup time and sampling
    fs = 100  # Hz - sampling rate (time step = 0.01 s)
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
    W[mask_2] = val_low + (val_flat - val_low) * (
        (positive_freqs[mask_2] - f_mid_start) / (f_mid_end - f_mid_start)
    )


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
    plt.loglog(positive_freqs, W, label="Weighting (m/s²)") # Plot in m/s²
    # plt.semilogx(positive_freqs, 20 * np.log10(W), label="Weighting (dB)") # Plot in dB

    # Define 1/3-octave frequency axis ticks --> Comment out if you want to see frequency position when you hover cursor over the plot
    octave_centers = np.array([0.016, 0.0315, 0.063, 0.125, 0.25, 0.5,
                            1, 2, 4, 8, 16, 31.5, 63])
    plt.xticks(octave_centers, [str(f) for f in octave_centers]) # Apply custom ticks and labels to x-axis

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Frequency Weighting (m/s²)")
    plt.title("Asymptotic Approximation of Vertical Weighting")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlim(positive_freqs[1], positive_freqs[-1])  # Avoid log(0)
    plt.legend()
    plt.tight_layout()

    # ----------------- Compare Unweighted and Weighted Acceleration Time-History -----------------
    plt.figure(figsize=(10, 5))
    plt.plot(t, acceleration, label="Unweighted", color='blue')
    plt.plot(t, weighted_signal, label="Weighted", color='orange')
    plt.title("Unweighted Acceleration Time-History")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()
    plt.grid(True)
    
    # ----------------- Plot Unweighted Frequency Spectrum (positive frequencies only) -----------------
    plt.figure(figsize=(10, 5))
    plt.plot(positive_freqs, positive_magnitude)
    plt.title("Unweighted Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Unweighted Acceleration (m/s²)")
    plt.xlim(0, 20)
    plt.ylim(0, 0.031)
    plt.grid(True)
    plt.tight_layout()

    # ----------------- Plot Weighted Frequency Spectrum -----------------
    plt.figure(figsize=(10, 5))
    plt.plot(positive_freqs, weighted_magnitude_pos, color='orange')
    plt.title("Weighted Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Weighted Acceleration (m/s²)")
    plt.xlim(0, 20)
    plt.ylim(0, 0.031)
    plt.grid(True)
    plt.tight_layout()

    # ----------------- Compare Unweighted and Weighted Spectra on the Same Plot -----------------
    plt.figure(figsize=(10, 5))
    plt.plot(positive_freqs, positive_magnitude, label="Unweighted", color='blue')
    plt.plot(positive_freqs, weighted_magnitude_pos, label="Weighted", color='orange')
    plt.title("Comparison of Frequency Spectra")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()
    plt.xlim(0, 20)
    plt.ylim(0, 0.031)
    plt.grid(True)
    plt.tight_layout()

    # ----------------- Vibration Serviceability Metrics -----------------
    # Peak Acceleration 
    peak_unweighted = np.max(np.abs(acceleration))
    peak_weighted = np.max(np.abs(weighted_signal))

    # RMS Acceleration
    rms_unweighted = np.sqrt(np.mean(acceleration ** 2))
    rms_weighted = np.sqrt(np.mean(weighted_signal ** 2))

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
    MTVV_unweighted = np.max(rms_running_unweighted)
    MTVV_weighted = np.max(rms_running_weighted)

    # MTVV*sqrt(2) = Peak Acceleration
    MTVV_root2_unweighted = np.sqrt(2)*MTVV_unweighted
    MTVV_root2_weighted = np.sqrt(2)*MTVV_weighted

    # Crest Factor (CF)
    CF_unweighted = peak_unweighted / rms_unweighted
    CF_weighted = peak_weighted / rms_weighted

    # Vibration Dose Value (VDV)
    vdv_unweighted = (np.sum(acceleration**4) * dt) ** 0.25
    vdv_weighted = (np.sum(weighted_signal**4) * dt) ** 0.25

    # Response Factor (R)
    R_unweighted = MTVV_unweighted/0.005
    R_weighted = MTVV_weighted/0.005

    # Make Table of Vibration Serviceability Metrics
    metrics = {
        "Peak (m/s^2)": [peak_unweighted, peak_weighted],
        "RMS (m/s^2)": [rms_unweighted, rms_weighted],
        "MTVV (m/s^2)": [MTVV_unweighted, MTVV_weighted],
        "MTVV*sqrt(2) (m/s^2)": [MTVV_root2_unweighted, MTVV_root2_weighted],
        "CF": [CF_unweighted, CF_weighted],
        "VDV (m/s^1.75)": [vdv_unweighted, vdv_weighted],
        "R": [R_unweighted, R_weighted]
    }
    # Create DataFrame
    df_metrics = pd.DataFrame(metrics, index=["Unweighted", "Weighted"])
    # Display table
    print(tabulate(df_metrics.round(6), headers='keys', tablefmt='grid', numalign="center", stralign="center"))


    plt.show() 

df = analyze_vibration(
    file_path="results_500mc_trial_matrix.csv",
    low_gain=0.4,
    f_low=0.5,
    f_mid_start=2.0,
    f_mid_end=5.0,
    f_flat_end=16.0
)

# Calling function from another file:
# from Function_VerticalFrequecyWeighting import analyze_vibration