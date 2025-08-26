import numpy as np

# synthetic time series around walking harmonics
fs = 100
t = np.arange(fs*60)/fs
accel = 0.12*np.sin(2*np.pi*2.0*t) + 0.05*np.sin(2*np.pi*4.0*t) + 0.02*np.random.randn(t.size)

# Baseline params
base = dict(low_gain=0.4, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0)

# Use the internal function to bypass file I/O:
m = _metrics_no_plots(accel, fs, **base)
print({k: round(v, 6) for k, v in m.items() if k.endswith("_weighted")})

# Sensitivity over all 7
df_elast, df_dydp = local_sensitivity_from_file(
    file_path="results_500mc_trial_matrix.csv",  # not used in this quick test
    trial_index=0, fs=fs, base_params=base, rel_step=0.05,
    metrics_to_track=None  # <- all 7 weighted by default
)
