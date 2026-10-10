"""Grid-level steps shared by the AOD and NO2 gap fillers.

These reproduce the notebook functions (two_pass_interp,
edge_direction_update, weighted_update_all, met_cosine_similarity) step for
step. Steps where every pixel reads only the previous grid are vectorized;
the edge-preserving pass updates the grid in place, so later pixels see
earlier updates, and it stays a sequential loop to give identical results.

Grids are (n_lat, n_lon) with latitude ASCENDING by row (row 0 = south),
the orientation the notebooks used (np.unique of the latitudes). The
sequential edge pass depends on this order.
"""

import math
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import median_filter, prewitt

# Neighbour offsets in the notebooks' loop order (ii outer, jj inner, centre skipped)
_OFFSETS = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if not (di == 0 and dj == 0)]
_DISTS = np.array([np.hypot(di, dj) for di, dj in _OFFSETS])


def _shift_stack(grid, fill=np.nan):
    """(8, H, W) stack of the 8 neighbours of every cell, NaN beyond the edge."""
    H, W = grid.shape
    p = np.pad(grid.astype(float), 1, constant_values=fill)
    return np.stack([p[1 + di:1 + di + H, 1 + dj:1 + dj + W] for di, dj in _OFFSETS])


# ------------------------------------------------------------------
# Step 2: two-pass adaptive neighbour fill
# ------------------------------------------------------------------

def _frontier(grid):
    """NaN cells with at least one non-NaN cell among their 8 neighbours."""
    has_val_nb = np.any(~np.isnan(_shift_stack(grid)), axis=0)
    return np.isnan(grid) & has_val_nb


def _adaptive_fill(data, targets, min_neighbors, max_window):
    """Median of the non-NaN neighbours in the smallest odd window (3..max_window)
    holding at least min_neighbors values; NaN if none does."""
    out = np.full(data.shape, np.nan)
    todo = targets.copy()
    half_max = max_window // 2
    padded = np.pad(data.astype(float), half_max, constant_values=np.nan)
    H, W = data.shape
    for size in range(3, max_window + 1, 2):
        if not todo.any():
            break
        h = size // 2
        sub = padded[half_max - h:half_max - h + H + 2 * h, half_max - h:half_max - h + W + 2 * h]
        win = sliding_window_view(sub, (size, size))           # (H, W, size, size)
        win = win[todo].reshape(-1, size * size).copy()
        win[:, (size * size) // 2] = np.nan                    # exclude the centre
        cnt = np.sum(~np.isnan(win), axis=1)
        ok = cnt >= min_neighbors
        if ok.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(win[ok], axis=1)
            idx = np.argwhere(todo)[ok]
            out[idx[:, 0], idx[:, 1]] = med
            done = np.zeros(data.shape, bool)
            done[idx[:, 0], idx[:, 1]] = True
            todo &= ~done
    return out


def two_pass_interp(grid, min_neighbors=2, max_window=9):
    """Fill the one-pixel frontier around valid data, twice."""
    g1 = grid.copy()
    m1 = _frontier(grid)
    g1[m1] = _adaptive_fill(grid, m1, min_neighbors, max_window)[m1]
    g2 = g1.copy()
    m2 = _frontier(g1)
    g2[m2] = _adaptive_fill(g1, m2, min_neighbors, max_window)[m2]
    return g2


# ------------------------------------------------------------------
# Steps 3/5: edge-preserving directional interpolation (sequential)
# ------------------------------------------------------------------

def edge_direction_update(i, j, grid, theta):
    """Weighted mean of the 3x3 neighbours, down-weighting those along the gradient."""
    vals, weights = [], []
    for ii in range(max(0, i - 1), min(grid.shape[0], i + 2)):
        for jj in range(max(0, j - 1), min(grid.shape[1], j + 2)):
            if ii == i and jj == j:
                continue
            if np.isnan(grid[ii, jj]):
                continue
            vx, vy = jj - j, ii - i
            d = np.hypot(vx, vy) + 1e-12
            gvec = np.array([np.cos(theta[ii, jj]), np.sin(theta[ii, jj])])
            align = abs(np.dot([vx / d, vy / d], gvec))
            w = (1 - align) / d
            vals.append(grid[ii, jj])
            weights.append(w)
    if not vals:
        return grid[i, j]
    weights = np.nan_to_num(weights)
    return (np.sum(np.array(vals) * weights) / np.sum(weights)
            if np.sum(weights) else np.median(vals))


def edge_pass(grid, adjustable_mask):
    """Gradient field from the current grid, then an in-place sequential edge update.

    Returns (updated_grid, grad_mag, grad_thresh) where grad_thresh is the
    80th percentile of the gradient magnitude before the update.
    """
    g = grid.copy()
    gx = prewitt(g, axis=1)
    gy = prewitt(g, axis=0)
    grad_mag = np.hypot(gx, gy)
    theta = np.arctan2(gy, gx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        grad_thresh = np.nanpercentile(grad_mag, 80)
    _edge_pass_inplace(g, np.cos(theta).tolist(), np.sin(theta).tolist(), adjustable_mask)
    return g, grad_mag, grad_thresh


def _edge_pass_inplace(g, cos_t, sin_t, adjustable_mask):
    """Same arithmetic as edge_direction_update, in plain Python floats (about 10x faster).

    Row-major order over adjustable pixels, reading the grid as it is
    being updated, exactly like the notebook loop.
    """
    H, W = g.shape
    rows = g.tolist()
    isnan = math.isnan
    for i, j in np.argwhere(adjustable_mask).tolist():
        vals, ws = [], []
        for ii in range(max(0, i - 1), min(H, i + 2)):
            row = rows[ii]
            ct, st = cos_t[ii], sin_t[ii]
            for jj in range(max(0, j - 1), min(W, j + 2)):
                if ii == i and jj == j:
                    continue
                v = row[jj]
                if isnan(v):
                    continue
                vx, vy = jj - j, ii - i
                d = math.hypot(vx, vy) + 1e-12
                align = abs((vx / d) * ct[jj] + (vy / d) * st[jj])
                w = (1 - align) / d
                vals.append(v)
                ws.append(0.0 if isnan(w) else w)
        if not vals:
            continue
        s = sum(ws)
        if s:
            rows[i][j] = sum(v * w for v, w in zip(vals, ws)) / s
        else:
            rows[i][j] = float(np.median(vals))
    g[:, :] = np.array(rows)


# ------------------------------------------------------------------
# Step 4: meteorologically weighted update with temporal priors
# ------------------------------------------------------------------

def weighted_update_all(grid, ugrid, vgrid, blhgrid, grad, grad_scale, wind_scale, blh_scale,
                        wind_exp, blh_exp, grad_exp, val_weight_exp, adjustable_mask,
                        temporal_priors, abs_values=False):
    """Vectorized weighted_update_all from the notebooks.

    Spatial weight of each of the 8 neighbours:
        1/(d + 0.3) * exp(-(|wind - median wind| / wind_scale)^wind_exp
                          -(|pbl - median pbl| / blh_scale)^blh_exp      (if blhgrid)
                          -(grad / grad_scale)^grad_exp) * value^val_weight_exp
    (|value| when abs_values=True, as in the NO2 version). Spatial weights
    share 1 - sum(temporal weights); temporal_priors is a list of
    (weight_grid, value_grid).
    """
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        vals = _shift_stack(grid)
        us, vs = _shift_stack(ugrid), _shift_stack(vgrid)
        grads = _shift_stack(grad)
        d = _DISTS[:, None, None]

        wind_term = (np.hypot(us - np.nanmedian(us, axis=0), vs - np.nanmedian(vs, axis=0)) / wind_scale) ** wind_exp
        expo = wind_term + (grads / grad_scale) ** grad_exp
        if blhgrid is not None:
            blhs = _shift_stack(blhgrid)
            expo = wind_term + (np.abs(blhs - np.nanmedian(blhs, axis=0)) / blh_scale) ** blh_exp \
                + (grads / grad_scale) ** grad_exp
        base = np.abs(vals) if abs_values else vals
        w_sp = (1 / (d + 0.3)) * np.exp(-expo) * (base ** val_weight_exp)

        t_vals, t_ws = [], []
        total_t = np.zeros(grid.shape)
        for w_t, ctx in temporal_priors:
            use = ~np.isnan(ctx) & ~np.isnan(w_t) & (w_t > 0)
            t_vals.append(np.where(use, ctx, np.nan))
            t_ws.append(np.where(use, w_t, np.nan))
            total_t += np.where(use, w_t, 0.0)

        share = np.maximum(1.0 - total_t, 0.0)
        w_sum_sp = np.nansum(w_sp, axis=0)
        w_sp = np.where(w_sum_sp > 0, w_sp * (share / np.where(w_sum_sp > 0, w_sum_sp, 1)), w_sp)

        all_v = np.concatenate([vals] + [t[None] for t in t_vals]) if t_vals else vals
        all_w = np.concatenate([w_sp] + [t[None] for t in t_ws]) if t_ws else w_sp
        valid = ~np.isnan(all_v) & ~np.isnan(all_w)
        w_sum = np.sum(np.where(valid, all_w, 0.0), axis=0)
        num = np.sum(np.where(valid, all_w * all_v, 0.0), axis=0)
        ok = adjustable_mask & valid.any(axis=0) & (w_sum != 0) & np.isfinite(w_sum)
        out = grid.copy()
        out[ok] = num[ok] / w_sum[ok]
    return out


# ------------------------------------------------------------------
# Meteorological similarity between the current and a context hour
# ------------------------------------------------------------------

def cosine_similarity(cur, ctx, weights):
    """Per-pixel weighted cosine similarity of two (F, H, W) feature stacks, clipped to [0, 1].

    Pixels with any NaN feature (in either hour) or a near-zero vector get 0.
    """
    w = np.asarray(weights, dtype=float)[:, None, None]
    a, b = cur * w, ctx * w
    bad = np.any(np.isnan(a), axis=0) | np.any(np.isnan(b), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        na = np.sqrt(np.sum(a * a, axis=0))
        nb = np.sqrt(np.sum(b * b, axis=0))
        sim = np.sum((a / na) * (b / nb), axis=0)
    sim = np.where(bad | (na < 1e-12) | (nb < 1e-12), 0.0, sim)
    return np.clip(sim, 0, 1)


# ------------------------------------------------------------------
# The full spatial pipeline for one hour
# ------------------------------------------------------------------

def fill_hour(grid_raw, adjustable_mask, ugrid, vgrid, blhgrid, temporal_priors,
              min_neighbors=2, max_window=9, wind_scale=5.0, blh_scale=2.0,
              wind_exp=1.0, blh_exp=1.0, grad_exp=1.0, val_weight_exp=4.0, abs_values=False):
    """Steps 2-5 of the method for one hour; returns the filled grid.

    2. two-pass adaptive neighbour fill, then 3x3 median filter with the
       original valid pixels restored
    3. edge-preserving directional interpolation on adjustable pixels
    4. wind/PBL/gradient weighted update blended with temporal priors
    5. edge-preserving interpolation again
    """
    raw_mask = ~np.isnan(grid_raw)
    g = two_pass_interp(grid_raw, min_neighbors, max_window)
    g = median_filter(g, size=3, mode="nearest")
    g[raw_mask] = grid_raw[raw_mask]
    g, grad_mag, grad_thresh = edge_pass(g, adjustable_mask)
    g = weighted_update_all(g, ugrid, vgrid, blhgrid, grad_mag, grad_thresh, wind_scale, blh_scale,
                            wind_exp, blh_exp, grad_exp, val_weight_exp, adjustable_mask,
                            temporal_priors, abs_values)
    g, _, _ = edge_pass(g, adjustable_mask)
    return g
