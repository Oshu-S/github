import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from tabulate import tabulate
from typing import List, Optional
from _metrics_no_plots import _metrics_no_plots

def local_sensitivity(
    file_path,
    base_params=None,
    rel_step=0.05,
    metrics_to_track=None, # ← default None => all 7 weighted metrics
    print_elast=None,
    print_dydp=None,
    print_pct=None
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
        "MTVV_sqrt2_weighted": "MTVV*√2",
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
    fs = 100
    df = pd.read_csv(file_path, encoding="utf-8")
    accel = df["Acceleration With HSI (m/s²)"].to_numpy()

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
    if print_elast is None:
        df_elast_pretty = df_elasticity.rename(columns=metric_pretty)
        df_elast_print  = df_elast_pretty.round(5).replace({np.nan: ""})
        print("\nElasticity (%ΔMetric / %ΔParameter) when curve parameter is perturbed by ±" + str(rel_step*100)  +  "%:")
        print(tabulate(df_elast_print, headers="keys", tablefmt="grid", numalign="center", stralign="center"))

    # 2) Partial derivatives – metric units only in headers; parameters shown without units
    if print_dydp is None:
        col_with_units = {
            k: f"{metric_pretty[k]}{f' ({metric_units[k]})' if metric_units[k] else ''}"
            for k in df_dydp.columns
        }
        df_dydp_u = df_dydp.rename(columns=col_with_units).copy()
        df_dydp_u.index = list(df_dydp.index)  # raw param names only (no units)

        df_dydp_print = df_dydp_u.round(5).replace({np.nan: ""})
        # print("\nPartial derivatives (∂M/∂P) when curve parameter is perturbed by ±" + str(rel_step*100)  +  "%:")
        # print(tabulate(df_dydp_print, headers="keys", tablefmt="grid",
        #             numalign="center", stralign="center"))
    
    # --- Approximate % change in metrics (using elasticity * rel_step * 100)
    pct_change_mat = elast_mat * (rel_step * 100)
    df_pct_change = pd.DataFrame(pct_change_mat, index=param_order, columns=metrics_to_track)

    if print_pct is None:
        df_pct_pretty = df_pct_change.rename(columns=metric_pretty)
        print(f"\nApproximate % change in metrics for ±{rel_step*100:.1f}% parameter perturbation (Elasticity × {rel_step*100:.1f}%):")
        print(tabulate(df_pct_pretty.round(5).replace({np.nan: ""}),
                    headers="keys", tablefmt="grid",
                    numalign="center", stralign="center"))

    # Return programmatic frames (without prettified labels) for plotting/saving
    return df_elasticity, df_dydp, df_pct_change

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
    Accepts DataFrames with either raw metric keys (*_weighted) or pretty keys,
    and always displays pretty titles.
    """

    # --- mapping between programmatic and pretty ---
    metric_pretty = {
        "Peak_weighted": "Peak",
        "RMS_weighted": "RMS",
        "MTVV_weighted": "MTVV",
        "MTVV_sqrt2_weighted": "MTVV*√2",
        "CF_weighted": "CF",
        "VDV_weighted": "VDV",
        "R_weighted": "R",
    }
    pretty_to_raw = {v: k for k, v in metric_pretty.items()}

    # Helper: given a requested name, find the column key and display name
    def resolve_metric(name):
        # If the exact name is in the DF, use it; choose display accordingly
        if name in df_elasticity.columns:
            # If it's a raw key, map to pretty for display; else keep as is
            disp = metric_pretty.get(name, name)
            return name, disp
        # If a pretty name was passed, see if its raw exists
        if name in pretty_to_raw and pretty_to_raw[name] in df_elasticity.columns:
            return pretty_to_raw[name], name
        # Last chance: try common alias for sqrt2 variations
        aliases = {
            "MTVV*sqrt(2)": "MTVV_sqrt2_weighted",
            "MTVV_sqrt2": "MTVV_sqrt2_weighted",
            "MTVV×√2": "MTVV_sqrt2_weighted",
        }
        if name in aliases and aliases[name] in df_elasticity.columns:
            return aliases[name], metric_pretty.get(aliases[name], name)
        raise KeyError(f"Metric '{name}' not found in df_elasticity columns {list(df_elasticity.columns)}")

    # If no list provided, use whatever is in the DF, but derive pretty display names
    if metrics_to_plot is None:
        metrics_to_plot = list(df_elasticity.columns)

    # Build the list of (col_key, display_name) in requested order
    resolved = []
    for m in metrics_to_plot:
        col_key, disp = resolve_metric(m)
        resolved.append((col_key, disp))

    # Layout
    nplots = len(resolved)
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

    # Common x-limits if requested
    if sharex:
        all_vals = []
        for col_key, _disp in resolved:
            s = df_elasticity[col_key].dropna()
            all_vals.extend(s.values.tolist())
        xabs = np.nanmax(np.abs(all_vals)) if len(all_vals) else 1.0
        xlim = (-1.05 * xabs, 1.05 * xabs)
    else:
        xlim = None

    desired_param_order = ["low_gain", "f_low", "f_mid_start", "f_mid_end", "f_flat_end"]

    # Draw each subplot
    for idx, (col_key, disp) in enumerate(resolved):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        # Reindex to a common order (no sorting by magnitude)
        s = df_elasticity[col_key].reindex(desired_param_order)

        y = np.arange(len(desired_param_order))
        ax.barh(y, s.values)
        ax.set_yticks(y)
        ax.set_yticklabels(desired_param_order)  # bottom→top in this order
        ax.axvline(0.0, linewidth=1.0)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.5)
        ax.set_title(disp)  # pretty title

        if sharex:
            ax.set_xlim(*xlim)
        else:
            local_max = np.nanmax(np.abs(s.values)) if len(s) else 1.0
            ax.set_xlim(-1.05 * local_max, 1.05 * local_max)

        if annotate:
            xmin, xmax = ax.get_xlim()
            span = xmax - xmin
            for yi, val in enumerate(s.values):
                if np.isnan(val):
                    continue
                xoff = 0.02 * (1 if val >= 0 else -1) * span
                ax.text(val + xoff, yi, f"{val:.2f}", va="center")

        ax.set_xlabel("Elasticity (%ΔMetric / %ΔParameter)")


    # Hide any extra axes (when grid > number of plots)
    for extra in range(nplots, nrows * ncols):
        r = extra // ncols
        c = extra % ncols
        axes[r, c].axis("off")

    fig.suptitle(title, y=0.995)
    if tight_layout:
        plt.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()
