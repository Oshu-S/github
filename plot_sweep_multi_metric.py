import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _metrics_no_plots import _metrics_no_plots

# ====================================================
# Range sweep: user-defined ranges → % change w.r.t. baseline
# Plot sweep to see how metrics change as one weighting curve parameter is swept
# - one-at-a-time global sweep (but not derivative-based)
# - Show how metrics change when you move a parameter across a user-defined range with others fixed
# - 1.Set that parameter to the grid value (others at baseline). 2.Recompute metrics. 3.Report % change relative to baseline:
# ====================================================
def plot_sweep_multi_metric(
    file_path,
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
    Accepts 'metrics' as raw keys or pretty names (or a mix).
    """

    if base_params is None:
        base_params = dict(low_gain=0.4, f_low=0.5, f_mid_start=2.0, f_mid_end=5.0, f_flat_end=16.0)
    if param_name not in base_params:
        raise ValueError(f"param_name '{param_name}' not in base_params: {list(base_params.keys())}")
    if samples < 2:
        raise ValueError("samples must be >= 2")

    # --- mapping between programmatic and pretty ---
    metric_pretty = {
        "Peak_weighted": "Peak acceleration",
        "RMS_weighted": "RMS acceleration",
        "MTVV_weighted": "MTVV",
        "MTVV_sqrt2_weighted": "MTVV*√2",
        "CF_weighted": "CF",
        "VDV_weighted": "VDV",
        "R_weighted": "R",
    }
    pretty_to_raw = {v: k for k, v in metric_pretty.items()}

    # Parameter pretty labels (with units where relevant)
    param_pretty = {
        "low_gain": "low_gain (dB)",
        "f_low": "f_low (Hz)",
        "f_mid_start": "f_mid_start (Hz)",
        "f_mid_end": "f_mid_end (Hz)",
        "f_flat_end": "f_flat_end (Hz)",
    }

    # Accept a few common aliases for the sqrt(2) metric name
    alias_to_raw = {
        "MTVV*sqrt(2)": "MTVV_sqrt2_weighted",
        "MTVV_sqrt2": "MTVV_sqrt2_weighted",
        "MTVV×√2": "MTVV_sqrt2_weighted",
    }

    def resolve_to_raw(name: str) -> str:
        """Return the internal raw metric key for any given name/alias."""
        if name in metric_pretty:              # already a raw key
            return name
        if name in pretty_to_raw:              # pretty → raw
            return pretty_to_raw[name]
        if name in alias_to_raw:               # alias → raw
            return alias_to_raw[name]
        # As a last resort, accept exact string if user passed another valid raw key
        return name

    def pretty_metric(raw_key: str) -> str:
        return metric_pretty.get(raw_key, raw_key)

    # Default metrics = all 7 weighted (raw keys)
    if metrics is None:
        metrics_raw = list(metric_pretty.keys())
    else:
        metrics_raw = [resolve_to_raw(m) for m in metrics]

    # Load once and compute baseline
    fs = 100
    df = pd.read_csv(file_path, encoding="utf-8")
    accel = df["Acceleration With HSI (m/s²)"].to_numpy()
    baseline = _metrics_no_plots(accel, fs, **base_params)

    # Make sure all requested raw metrics exist in baseline
    missing = [m for m in metrics_raw if m not in baseline]
    if missing:
        raise KeyError(f"Requested metrics not found: {missing}. "
                       f"Valid keys include: {list(baseline.keys())}")

    y0 = {m: baseline[m] for m in metrics_raw}

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
        for k in metrics_raw:
            base_val = y0[k]
            row[k] = np.nan if base_val == 0 else 100.0 * (m[k] - base_val) / base_val
        rows.append(row)

    # DataFrame columns will be raw keys; index = swept parameter values
    df = pd.DataFrame(rows).set_index("param_value")
    
    # ----- Plot ----- 
    fig, ax = plt.subplots(figsize=(10, 5))
    for raw_key in metrics_raw:
        if raw_key not in df.columns:
            continue
        ax.plot(df.index.values, df[raw_key].values, marker="o", label=pretty_metric(raw_key))

    # Pretty param label
    xlab = param_pretty.get(param_name, param_name)
    ax.set_xlabel(xlab)

    # If only one metric → use its pretty name
    if len(metrics_raw) == 1:
        pretty_name = pretty_metric(metrics_raw[0])
        ax.set_ylabel(f"% change in {pretty_name} relative to baseline")
        if title is None:
            ax.set_title(f"% change in {pretty_name} vs {param_name}")
        else:
            ax.set_title(title)
    else:
        ax.set_ylabel("% change in metrics relative to baseline")
        if title is None:
            ax.set_title(f"% change in selected metrics vs {xlab}")
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
