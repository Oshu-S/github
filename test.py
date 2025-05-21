from Function_VerticalFrequencyWeighting import analyze_vibration

df = analyze_vibration(
    file_path="results_500mc_trial_matrix.csv",
    low_gain=1.0,
    f_low=0.5,
    f_mid_start=2.0,
    f_mid_end=5.0,
    f_flat_end=16.0
)


