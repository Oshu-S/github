from Function_VerticalFrequecyWeighting import analyze_vibration

df = analyze_vibration(
    file_path="results_500mc_trial_matrix.csv",
    # scale=0.5,
    f_low=0.2,
    f_mid_start=8.0,
    f_mid_end=12.0,
    f_flat_end=16.0
)
