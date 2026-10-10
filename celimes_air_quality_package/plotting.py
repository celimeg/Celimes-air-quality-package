"""Figures. Every function returns the matplotlib Figure and saves only when save_path is given."""

import numpy as np
import pandas as pd

from .footprints import FootprintGrid, load_footprint
from .validation import compute_stats, stats_by_aod_bin

_STYLE = dict(FONT_LABEL=22, FONT_TICK=18, FONT_ANNO=17, FONT_LEGEND=18,
              DOT_ALPHA=0.45, DOT_SIZE=18, DOT_COLOR="#2166ac")


def _save(fig, save_path, dpi):
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {save_path}")
    return fig


def _stats_text(s):
    return (f"R² = {s['r2']:.3f}\nRMSE = {s['rmse']:.3f}\nBias = {s['bias']:.3f}\nN = {s['n']:,}\n"
            f"GOES mean±std = {s['mean_goes']:.3f} ± {s['std_goes']:.3f}\n"
            f"AERONET mean±std = {s['mean_aeronet']:.3f} ± {s['std_aeronet']:.3f}\n"
            f"Within EE = {s['pct_ee']:.1f}%")


def _scatter_panel(ax, merged, stats, xlabel, aod_max, full_ee=False):
    st = _STYLE
    ae = merged["AOD_AERONET"].to_numpy()
    goes = merged["AOD_GOES"].to_numpy()
    xs = np.linspace(0, aod_max, 300) if full_ee else np.sort(ae)
    ax.scatter(goes, ae, color=st["DOT_COLOR"], alpha=st["DOT_ALPHA"], s=st["DOT_SIZE"], zorder=2)
    l1, = ax.plot(xs, xs + (0.05 + 0.15 * xs), "r-", lw=2, zorder=3)
    l2, = ax.plot(xs, xs - (0.05 + 0.15 * xs), "r-", lw=2, zorder=3)
    x1 = np.linspace(0, aod_max, 300)
    l3, = ax.plot(x1, x1, "k--", lw=1.5, alpha=0.6, zorder=1)
    ax.set_xlim(0, aod_max)
    ax.set_ylim(0, aod_max)
    ax.set_xlabel(xlabel, fontsize=st["FONT_LABEL"], fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", labelsize=st["FONT_TICK"])
    ax.text(0.03, 0.97, _stats_text(stats), transform=ax.transAxes, fontsize=st["FONT_ANNO"], va="top",
            ha="left", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.85))
    return l1, l2, l3


def plot_validation_scatter(raw_merged, filled_merged, titles=None, aod_max=4.0,
                            save_path=None, dpi=900):
    """Raw (left) and gap-filled (right) GOES vs AERONET with EE envelope and stats."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.subplots_adjust(wspace=0.08)
    panels = [(axes[0], raw_merged, "Raw GOES-R ABI AOD"), (axes[1], filled_merged, "Gap-Filled GOES-R ABI AOD")]
    for k, (ax, m, xl) in enumerate(panels):
        handles = _scatter_panel(ax, m, compute_stats(m), xl, aod_max)
        if titles:
            ax.set_title(titles[k], fontsize=_STYLE["FONT_LABEL"], fontweight="bold", pad=12)
    axes[1].legend(handles=handles, labels=["EE Upper", "EE Lower", "1:1 Line"], loc="lower right",
                   fontsize=_STYLE["FONT_LEGEND"], framealpha=0.9)
    axes[0].set_ylabel("AERONET AOD", fontsize=_STYLE["FONT_LABEL"], fontweight="bold", labelpad=10)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.tight_layout()
    return _save(fig, save_path, dpi)


def season_from_doy(dt):
    """Season by day of year, as in the original seasonal figures (DJF = DOY >= 335 or <= 59)."""
    d = pd.to_datetime(dt).dt.dayofyear
    return np.select([(d >= 335) | (d <= 59), d <= 151, d <= 243], ["DJF", "MAM", "JJA"], "SON")


def plot_seasonal_scatter(merged, label="Gap-Filled GOES-R ABI AOD", aod_max=4.0, save_dir=None, dpi=900):
    """One scatter per season; returns {season: Figure}. Saves PNGs only if save_dir is given."""
    import os

    import matplotlib.pyplot as plt

    seasons = season_from_doy(merged["Datetime"])
    figs = {}
    for s in ["DJF", "MAM", "JJA", "SON"]:
        sub = merged[seasons == s]
        if len(sub) < 2:
            print(f"{s}: {len(sub)} points, skipping")
            continue
        fig, ax = plt.subplots(figsize=(8, 7))
        h = _scatter_panel(ax, sub, compute_stats(sub), f"{label} ({s})", aod_max, full_ee=True)
        ax.set_ylabel("AERONET AOD", fontsize=_STYLE["FONT_LABEL"], fontweight="bold", labelpad=10)
        ax.legend(handles=h, labels=["EE Upper", "EE Lower", "1:1 Line"], loc="lower right",
                  fontsize=_STYLE["FONT_LEGEND"], framealpha=0.9)
        fig.tight_layout()
        figs[s] = _save(fig, os.path.join(save_dir, f"GOES_AERONET_{s}.png") if save_dir else None, dpi)
    return figs


def plot_metric_by_aod_bin(merged, metric="r2", bins=(0, 1, 2, 3, 5), ymax=None, save_path=None, dpi=900):
    """Bar chart of R2 or RMSE within AERONET AOD bins, with N in each bar."""
    import matplotlib.pyplot as plt

    res = stats_by_aod_bin(merged, bins)
    vals, ns = res[metric].to_numpy(float), res["n"].to_numpy()
    top = ymax if ymax is not None else (1.0 if metric == "r2" else np.nanmax(vals) * 1.15)
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(res))
    ax.bar(x, vals, width=1.0, color="#2166ac", edgecolor="white", linewidth=1.5)
    for xi, v, n in zip(x, vals, ns):
        if np.isfinite(v):
            ax.text(xi, v + top * 0.02, f"{v:.{2 if metric == 'r2' else 3}f}", ha="center", fontsize=13,
                    fontweight="bold")
            ax.text(xi, top * 0.02, f"N={n:,}", ha="center", va="bottom", fontsize=11, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} AOD" for b in res["Bin"]], fontsize=14)
    ax.set_xlim(-0.5, len(res) - 0.5)
    ax.set_ylim(0, top)
    ax.set_ylabel({"r2": "R²", "rmse": "RMSE"}.get(metric, metric), fontsize=18, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, save_path, dpi)


def plot_footprint(footprint, highlight_pixels=None, target=None, radii_km=(7.5, 27.5),
                   label_first=10, title=None, save_path=None, dpi=300):
    """Footprint rectangles; optional highlighted pixels, site star and radius circles."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    fp = load_footprint(footprint)
    grid = FootprintGrid.from_footprint(fp)
    hl = set(highlight_pixels or [])
    fig, ax = plt.subplots(figsize=(10, 8))
    rects, colors = [], []
    for _, r in fp.iterrows():
        rects.append(Rectangle((r["Longitude"] - grid.lon_step / 2, r["Latitude"] - grid.lat_step / 2),
                               grid.lon_step, grid.lat_step))
        colors.append("red" if r["Pixel Number"] in hl else "steelblue")
    ax.add_collection(PatchCollection(rects, facecolor=colors, edgecolor="black", linewidth=0.3))
    for _, r in fp[fp["Pixel Number"] <= label_first].iterrows():
        ax.text(r["Longitude"], r["Latitude"], int(r["Pixel Number"]), ha="center", va="center",
                color="white", fontsize=9, weight="bold")
    if target is not None:
        lat, lon = target
        ax.plot(lon, lat, marker="*", color="black", markersize=14, zorder=8)
        t = np.linspace(0, 2 * np.pi, 361)
        for rk, col in zip(radii_km, ["white", "yellow", "orange"]):
            ax.plot(lon + rk / (111.0 * np.cos(np.radians(lat))) * np.cos(t), lat + rk / 111.0 * np.sin(t),
                    color=col, lw=2, label=f"{rk} km radius", zorder=7)
        ax.legend(loc="lower right")
    le, oe = grid.lat_edges, grid.lon_edges
    ax.set_xlim(oe[0], oe[-1])
    ax.set_ylim(le[-1], le[0])
    ax.set_aspect(1 / np.cos(np.radians(fp["Latitude"].mean())))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title or f"Footprint: {len(fp)} pixels")
    fig.tight_layout()
    return _save(fig, save_path, dpi)


def plot_hour(df, column, datetime, title=None, cmap="viridis", vmin=None, vmax=None, save_path=None, dpi=300):
    """Map one column of a gridded table (e.g. filled_AOD) for one hour."""
    import matplotlib.pyplot as plt

    from .utils import parse_datetime
    d = df.copy()
    d["Datetime"] = parse_datetime(d["Datetime"])
    h = d[d["Datetime"] == pd.Timestamp(datetime)]
    if h.empty:
        raise ValueError(f"No rows at {datetime}")
    grid = FootprintGrid.from_footprint(h[["Pixel Number", "Latitude", "Longitude"]])
    g = np.full(grid.shape, np.nan)
    r, c = grid.index_of(h["Pixel Number"])
    g[r, c] = h[column].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 7))
    m = ax.pcolormesh(grid.lon_edges, grid.lat_edges, np.ma.masked_invalid(g), cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(m, ax=ax, label=column)
    ax.set_aspect(1 / np.cos(np.radians(h["Latitude"].mean())))
    ax.set_title(title or f"{column}  {pd.Timestamp(datetime):%Y-%m-%d %H}:00 UTC")
    fig.tight_layout()
    return _save(fig, save_path, dpi)
