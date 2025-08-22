
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

def _build_weighting(positive_freqs: np.ndarray,
                     low_gain: float,
                     f_low: float,
                     f_mid_start: float,
                     f_mid_end: float,
                     f_flat_end: float) -> np.ndarray:
    W = np.zeros_like(positive_freqs, dtype=float)
    val_low = low_gain
    val_flat = 1.0
    f_flat_start = f_mid_end

    mask_0 = (positive_freqs > 0.0) & (positive_freqs < f_low)
    W[mask_0] = val_low * (positive_freqs[mask_0] / f_low)

    mask_1 = (positive_freqs >= f_low) & (positive_freqs <= f_mid_start)
    W[mask_1] = val_low

    mask_2 = (positive_freqs > f_mid_start) & (positive_freqs <= f_mid_end)
    if f_mid_end > f_mid_start:
        W[mask_2] = val_low + (val_flat - val_low) * (
            (positive_freqs[mask_2] - f_mid_start) / (f_mid_end - f_mid_start)
        )
    else:
        W[mask_2] = val_low

    mask_3 = (positive_freqs > f_flat_start) & (positive_freqs <= f_flat_end)
    W[mask_3] = val_flat

    mask_4 = (positive_freqs > f_flat_end)
    W[mask_4] = np.where(positive_freqs[mask_4] > 0.0, val_flat * f_flat_end / positive_freqs[mask_4], 0.0)

    W[positive_freqs == 0.0] = 0.0
    return W

def weight_and_metrics(acceleration: np.ndarray,
                       fs: int = 100,
                       low_gain: float = 0.4,
                       f_low: float = 0.5,
                       f_mid_start: float = 2.0,
                       f_mid_end: float = 5.0,
                       f_flat_end: float = 16.0,
                       do_plots: bool = False) -> Dict[str, float]:
    acceleration = np.asarray(acceleration).astype(float)
    acceleration = acceleration - np.mean(acceleration)

    n = acceleration.size
    dt = 1.0 / fs
    t = np.arange(n) * dt

    acc_fft = fft(acceleration)
    freq_vector = fftfreq(n, d=dt)

    pos_count = n // 2 + 1
    positive_freqs = freq_vector[:pos_count]

    Wpos = _build_weighting(positive_freqs, low_gain, f_low, f_mid_start, f_mid_end, f_flat_end)

    weights = np.empty_like(freq_vector, dtype=float)
    weights[:pos_count] = Wpos
    if n % 2 == 0:
        weights[pos_count:] = Wpos[1:pos_count-1][::-1]
    else:
        weights[pos_count:] = Wpos[1:pos_count][::-1]

    weighted_fft = acc_fft * weights
    weighted_signal = np.real(ifft(weighted_fft))

    peak_unw = np.max(np.abs(acceleration))
    peak_w = np.max(np.abs(weighted_signal))

    rms_unw = np.sqrt(np.mean(acceleration ** 2))
    rms_w = np.sqrt(np.mean(weighted_signal ** 2))

    window_size = int(fs * 1.0)
    def running_rms(sig, w):
        kernel = np.ones(w) / w
        return np.sqrt(np.convolve(sig**2, kernel, mode='valid'))
    mtvv_unw = float(np.max(running_rms(acceleration, window_size)))
    mtvv_w = float(np.max(running_rms(weighted_signal, window_size)))

    mtvv_root2_unw = float(np.sqrt(2.0) * mtvv_unw)
    mtvv_root2_w = float(np.sqrt(2.0) * mtvv_w)

    cf_unw = float(peak_unw / rms_unw) if rms_unw > 0 else np.inf
    cf_w = float(peak_w / rms_w) if rms_w > 0 else np.inf

    vdv_unw = float((np.sum(acceleration**4) * dt) ** 0.25)
    vdv_w = float((np.sum(weighted_signal**4) * dt) ** 0.25)

    R_unw = float(mtvv_unw / 0.005)
    R_w = float(mtvv_w / 0.005)

    metrics = {
        "Peak_unweighted": peak_unw, "Peak_weighted": peak_w,
        "RMS_unweighted": rms_unw, "RMS_weighted": rms_w,
        "MTVV_unweighted": mtvv_unw, "MTVV_weighted": mtvv_w,
        "MTVV_sqrt2_unweighted": mtvv_root2_unw, "MTVV_sqrt2_weighted": mtvv_root2_w,
        "CF_unweighted": cf_unw, "CF_weighted": cf_w,
        "VDV_unweighted": vdv_unw, "VDV_weighted": vdv_w,
        "R_unweighted": R_unw, "R_weighted": R_w
    }

    if do_plots:
        plt.figure(figsize=(8, 4.5))
        plt.loglog(positive_freqs[positive_freqs > 0], Wpos[positive_freqs > 0])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Weighting")
        plt.title("Asymptotic Approximation of Vertical Frequency Weighting")
        plt.grid(True, which="both")

        plt.figure(figsize=(8, 4.5))
        plt.plot(t, acceleration, label="Unweighted")
        plt.plot(t, weighted_signal, label="Weighted")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (m/s²)")
        plt.title("Acceleration Time-History")
        plt.legend()
        plt.grid(True)

        magnitude_unw = np.abs(fft(acceleration)) / n
        magnitude_w = np.abs(weighted_fft) / n
        plt.figure(figsize=(8, 4.5))
        plt.plot(positive_freqs, magnitude_unw[:pos_count], label="Unweighted")
        plt.plot(positive_freqs, magnitude_w[:pos_count], label="Weighted")
        plt.xlim(0, fs/2)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (m/s²)")
        plt.title("FFT - Frequency Spectra")
        plt.legend()
        plt.grid(True)

    return metrics

def local_sensitivity(acceleration: np.ndarray,
                      fs: int,
                      base_params: Dict[str, float],
                      rel_step: float = 0.05,
                      metrics_to_track: List[str] = None) -> pd.DataFrame:
    m0 = weight_and_metrics(acceleration, fs=fs, **base_params, do_plots=False)
    if metrics_to_track is None:
        metrics_to_track = [k for k in m0.keys() if k.endswith("_weighted")]

    def extract(metrics_dict):
        return np.array([metrics_dict[k] for k in metrics_to_track], dtype=float)

    y0 = extract(m0)
    param_order = ["low_gain", "f_low", "f_mid_start", "f_mid_end", "f_flat_end"]
    S = np.zeros((len(param_order), len(metrics_to_track)), dtype=float)

    for i, pname in enumerate(param_order):
        pval = float(base_params[pname])
        step = rel_step * max(abs(pval), 1e-12)

        p_down = pval - step
        if "f_" in pname and p_down <= 0.0:
            p_down = pval

        params_down = dict(base_params)
        params_down[pname] = p_down
        y_down = extract(weight_and_metrics(acceleration, fs=fs, **params_down, do_plots=False))

        p_up = pval + step
        params_up = dict(base_params)
        params_up[pname] = p_up
        y_up = extract(weight_and_metrics(acceleration, fs=fs, **params_up, do_plots=False))

        dy = y_up - y_down
        dp = p_up - p_down if (p_up - p_down) != 0 else step
        dydp = dy / dp

        with np.errstate(divide='ignore', invalid='ignore'):
            sens = dydp * (pval / y0)
        sens[~np.isfinite(sens)] = np.nan
        S[i, :] = sens

    df = pd.DataFrame(S, index=param_order, columns=metrics_to_track)
    return df

def load_trial_from_csv(file_path: str, trial_index: int = 0) -> Tuple[np.ndarray, int]:
    df = pd.read_csv(file_path)
    accel_data = df.iloc[trial_index, 1:].values.astype(float)
    fs = 100
    return accel_data, fs

def generate_test_signal(fs: int = 100, duration: float = 60.0) -> np.ndarray:
    t = np.arange(int(fs * duration)) / fs
    sig = 0.15*np.sin(2*np.pi*2.0*t) + 0.06*np.sin(2*np.pi*4.0*t) + 0.02*np.random.randn(t.size)
    return sig
