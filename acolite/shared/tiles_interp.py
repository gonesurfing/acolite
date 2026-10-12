## def tiles_interp
## interpolates tiled dataset to full scene extent
## written by Quinten Vanhellemont, RBINS for the PONDER project
## 2017-12-11
## modifications: 2020-11-17 (QV) added target mask for limiting interpolation extent and processing time
##                2020-11-18 (QV) added dtype to convert from griddata float64, by default float32
##                                this improves peak memory use when several datasets are kept in memory
##                2021-02-11 (QV) added smooth keyword,  default to nearest
##                2024-04-18 (QV) new version using interpn, added option to use RGI
##                2026-10-12 separable evaluation for linear and nearest methods, same result as interpn
##                           but interpolates along x on the tile rows first, then along y,
##                           avoiding per pixel index searches and full scene float64 temporaries

def tiles_interp(data, xnew, ynew, smooth = False, kern_size=2, method='nearest', mask = None,
                 target_mask = None, target_mask_full = False, fill_nan = True, dtype = 'float32', use_rgi = False,
                 separable = True):

    import numpy as np
    from scipy.interpolate import interpn, RegularGridInterpolator
    from scipy.ndimage import uniform_filter
    import acolite as ac

    if mask is not None: data[mask] = np.nan

    ## fill nans with closest value
    if fill_nan:
        cur_data = ac.shared.fillnan(data)
    else:
        cur_data = data*1.0

    dim = cur_data.shape
    if smooth: cur_data = uniform_filter(cur_data, size = kern_size)

    ## grid positions
    x = np.arange(0., dim[1], 1)
    y = np.arange(0., dim[0], 1)

    ## separable evaluation on the regular tile grid
    ## NaNs left in the tiles fall back to interpn, which handles them differently per method
    if separable & (method in ['linear', 'nearest']) & (len(dim) == 2) and (min(dim) > 1) and np.isfinite(cur_data).all():
        znew = _tiles_interp_separable(cur_data, x, y, xnew, ynew, method = method,
                                       target_mask = target_mask, target_mask_full = target_mask_full,
                                       dtype = dtype)
    ## set up interpolator
    elif use_rgi: ## use RGI
        rgi = RegularGridInterpolator((y,x), cur_data, bounds_error = False, fill_value = np.nan, method = method)
        if target_mask is None:
            znew = rgi((ynew[:,None], xnew[None, :]))
        else: ## limit to target mask
            vd = np.where(target_mask)
            if target_mask_full:
                znew = np.zeros((len(ynew), len(xnew))).astype(dtype)+np.nan
                znew[vd] = rgi((ynew[vd[0]], xnew[vd[1]]))
            else:
                znew = rgi((ynew[vd[0]], xnew[vd[1]]))
    else:
        if target_mask is None:
            znew = interpn((y,x), cur_data, (ynew[:,None], xnew[None, :]), method = method, bounds_error = False)
        else: ## limit to target mask
            vd = np.where(target_mask)
            if target_mask_full:
                znew = np.zeros((len(ynew), len(xnew))).astype(dtype)+np.nan
                znew[vd] = interpn((y,x), cur_data, (ynew[vd[0]], xnew[vd[1]]), method = method, bounds_error = False)
            else:
                znew = interpn((y,x), cur_data, (ynew[vd[0]], xnew[vd[1]]), method = method, bounds_error = False)

    ## to convert data type - scipy always returns float64
    if dtype is not None: znew = znew.astype(np.dtype(dtype), copy = False)
    return(znew)

## indices and weights along one axis, following the interval search used by interpn
## points outside the grid (or NaN) are flagged for NaN fill as in interpn with bounds_error=False
def _axis_weights(grid, q, method):
    import numpy as np
    q = np.asarray(q, dtype = np.float64)
    i0 = np.clip(np.searchsorted(grid, q) - 1, 0, len(grid) - 2)
    w = (q - grid[i0]) / (grid[i0 + 1] - grid[i0])
    oob = ~((q >= grid[0]) & (q <= grid[-1]))
    if method == 'nearest':
        i0 = np.where(w <= .5, i0, i0 + 1)
        return(i0, i0, None, oob)
    return(i0, i0 + 1, w, oob)

def _tiles_interp_separable(cur_data, x, y, xnew, ynew, method = 'linear',
                            target_mask = None, target_mask_full = False, dtype = 'float32', block = 512):
    import numpy as np

    ix0, ix1, wx, oobx = _axis_weights(x, xnew, method)
    iy0, iy1, wy, ooby = _axis_weights(y, ynew, method)

    ## interpolate the tile rows along x: (n tile rows, len(xnew))
    if method == 'nearest':
        a = cur_data[:, ix0]
    else:
        a = cur_data[:, ix0] * (1 - wx) + cur_data[:, ix1] * wx
    a[:, oobx] = np.nan

    odt = a.dtype if dtype is None else np.dtype(dtype)
    if target_mask is None:
        ## interpolate along y in row blocks to limit float64 temporaries
        znew = np.empty((len(ynew), len(xnew)), dtype = odt)
        for s in range(0, len(ynew), block):
            e = min(s + block, len(ynew))
            if method == 'nearest':
                znew[s:e] = a[iy0[s:e]]
            else:
                znew[s:e] = a[iy0[s:e]] * (1 - wy[s:e, None]) + a[iy1[s:e]] * wy[s:e, None]
        znew[ooby] = np.nan
    else:
        ## evaluate only at the target mask pixels
        r, c = np.where(target_mask)
        if method == 'nearest':
            zsub = a[iy0[r], c]
        else:
            zsub = a[iy0[r], c] * (1 - wy[r]) + a[iy1[r], c] * wy[r]
        zsub[ooby[r]] = np.nan
        if target_mask_full:
            znew = np.full((len(ynew), len(xnew)), np.nan, dtype = np.float64 if dtype is None else odt)
            znew[r, c] = zsub
        else:
            znew = zsub
    return(znew)
