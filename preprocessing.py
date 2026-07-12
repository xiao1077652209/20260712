"""MSC and Competitive Adaptive Reweighted Sampling (CARS)."""
from __future__ import annotations
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold


def msc(spectra, reference=None):
    x = np.asarray(spectra, dtype=np.float64); ref = x.mean(0) if reference is None else np.asarray(reference)
    corrected = np.empty_like(x)
    for i, spectrum in enumerate(x):
        slope, intercept = np.polyfit(ref, spectrum, 1); corrected[i] = (spectrum-intercept)/slope
    return corrected, ref


def _rmsecv(x, y, indices, components, folds):
    errors = []
    for train, valid in folds.split(x):
        ncomp = max(1, min(components, len(indices), len(train)-1))
        pls = PLSRegression(n_components=ncomp).fit(x[train][:, indices], y[train])
        errors.append(np.mean((pls.predict(x[valid][:, indices])-y[valid])**2))
    return float(np.sqrt(np.mean(errors)))


def _variable_importance(pls, variable_count):
    coef = np.asarray(pls.coef_)
    variable_axis = 0 if coef.shape[0] == variable_count else 1
    return np.linalg.norm(coef, axis=1-variable_axis)


def cars_select(x, y, n_features=114, iterations=50, components=10, seed=42, return_history=False):
    """CARS using MC calibration, EDF elimination, ARS sampling and RMSECV."""
    x, y = np.asarray(x), np.asarray(y)
    if y.ndim == 1:
        classes = np.unique(y); y = np.eye(len(classes))[np.searchsorted(classes, y)]
    rng = np.random.default_rng(seed); active = np.arange(x.shape[1]); folds = KFold(5, shuffle=True, random_state=seed)
    candidates = []; history = []
    for step in range(iterations):
        sample = rng.choice(len(x), max(2, int(.8*len(x))), replace=False)
        pls = PLSRegression(n_components=max(1, min(components, len(active), len(sample)-1))).fit(x[sample][:, active], y[sample])
        weights = _variable_importance(pls, len(active)); weights = weights / max(weights.sum(), np.finfo(float).eps)
        target = max(n_features, round(x.shape[1]*(n_features/x.shape[1])**((step+1)/iterations)))
        chosen = rng.choice(len(active), size=min(target, len(active)), replace=False, p=weights)
        active = active[chosen]
        rmse = _rmsecv(x, y, active, components, folds); history.append((step+1, len(active), rmse)); candidates.append((rmse, active.copy()))
    eligible = [item for item in candidates if len(item[1]) >= n_features]
    _, best = min(eligible, key=lambda z: z[0])
    if len(best) > n_features:
        pls = PLSRegression(n_components=min(components, len(best))).fit(x[:, best], y)
        importance = _variable_importance(pls, len(best)); best = best[np.argsort(importance)[-n_features:]]
    result = np.sort(best)
    return (result, history) if return_history else result
