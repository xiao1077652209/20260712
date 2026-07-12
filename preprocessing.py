"""MSC preprocessing and a reproducible CARS-style wavelength selector."""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold


def msc(spectra: np.ndarray, reference: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(spectra, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("spectra must be a 2-D [samples, wavenumbers] array")
    ref = x.mean(axis=0) if reference is None else np.asarray(reference)
    corrected = np.empty_like(x)
    for i, spectrum in enumerate(x):
        slope, intercept = np.polyfit(ref, spectrum, 1)
        corrected[i] = (spectrum - intercept) / slope
    return corrected, ref


def cars_select(x: np.ndarray, y: np.ndarray, n_features: int = 114,
                iterations: int = 50, components: int = 10, seed: int = 42) -> np.ndarray:
    """Select variables by MC sampling, EDF elimination and PLS coefficients.

    The paper does not publish its selected indices; fitting on training data is required.
    """
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(seed)
    active = np.arange(x.shape[1])
    best, best_rmse = active.copy(), np.inf
    folds = KFold(5, shuffle=True, random_state=seed)
    for step in range(iterations):
        sample = rng.choice(len(x), max(2, int(0.8 * len(x))), replace=False)
        pls = PLSRegression(n_components=min(components, len(active), len(sample) - 1))
        pls.fit(x[sample][:, active], y[sample])
        importance = np.linalg.norm(np.atleast_2d(pls.coef_), axis=1)
        target = max(n_features, int(x.shape[1] * (n_features / x.shape[1]) ** ((step + 1) / iterations)))
        active = active[np.argsort(importance)[-min(target, len(active)):]]
        errors = []
        for train, valid in folds.split(x):
            p = PLSRegression(n_components=min(components, len(active), len(train) - 1))
            p.fit(x[train][:, active], y[train])
            errors.append(np.mean((p.predict(x[valid][:, active]).ravel() - y[valid]) ** 2))
        rmse = float(np.sqrt(np.mean(errors)))
        if len(active) == n_features and rmse < best_rmse:
            best, best_rmse = active.copy(), rmse
    return np.sort(best if len(best) == n_features else active[:n_features])
