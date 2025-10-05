import numpy as np
from scipy.fft import fft, ifft, fftfreq

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
