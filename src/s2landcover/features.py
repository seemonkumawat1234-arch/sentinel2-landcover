"""Feature construction from Sentinel-2 bands.

The classifier is fed raw band reflectance plus a handful of spectral indices.
Indices are not redundant with the bands they are computed from: a tree
ensemble can only split on axis-aligned thresholds, so a ratio like NDVI that
would need a diagonal boundary in (NIR, red) space is genuinely new information
to it, and adding indices measurably improves class separation for little cost.

Band references are Sentinel-2 MSI:

    B2   blue    490 nm
    B3   green   560 nm
    B4   red     665 nm
    B8   NIR     842 nm
    B11  SWIR-1  1610 nm
    B12  SWIR-2  2190 nm
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = ["BANDS", "INDEX_NAMES", "normalised_difference", "compute_indices",
           "build_feature_stack", "feature_names"]

BANDS: Tuple[str, ...] = ("B2", "B3", "B4", "B8", "B11", "B12")

INDEX_NAMES: Tuple[str, ...] = ("NDVI", "NDWI", "NDBI", "NBR", "BSI")

_EPS = 1e-10


def normalised_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), NaN where the denominator is zero.

    NaN rather than 0, because a zero index value is a real measurement and
    substituting one for no-data would let masked pixels pass as valid.
    """
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    if a.shape != b.shape:
        raise ValueError("shapes differ: {} vs {}".format(a.shape, b.shape))
    denom = a + b
    out = np.full(a.shape, np.nan, dtype="float64")
    ok = np.abs(denom) > _EPS
    out[ok] = (a[ok] - b[ok]) / denom[ok]
    return out


def compute_indices(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Five indices chosen to separate the classes this package targets.

    NDVI  vegetation vigour, separates green cover from everything else
    NDWI  open water, the one class NDVI alone confuses with dense shadow
    NDBI  built and bare surfaces, which are SWIR-bright
    NBR   burnt ground, low NIR and high SWIR2, common across savanna
    BSI   bare soil, which NDBI alone does not distinguish from built
    """
    missing = [b for b in BANDS if b not in bands]
    if missing:
        raise KeyError("missing band(s) {}; have {}".format(
            missing, sorted(bands)))

    b2, b3, b4 = bands["B2"], bands["B3"], bands["B4"]
    b8, b11, b12 = bands["B8"], bands["B11"], bands["B12"]

    bsi_num = (b11 + b4) - (b8 + b2)
    bsi_den = (b11 + b4) + (b8 + b2)
    bsi = np.full(np.asarray(b4).shape, np.nan, dtype="float64")
    ok = np.abs(bsi_den) > _EPS
    bsi[ok] = bsi_num[ok] / bsi_den[ok]

    return {
        "NDVI": normalised_difference(b8, b4),
        "NDWI": normalised_difference(b3, b8),
        "NDBI": normalised_difference(b11, b8),
        "NBR": normalised_difference(b8, b12),
        "BSI": bsi,
    }


def feature_names(use_indices: bool = True) -> List[str]:
    """Column order of the feature matrix. Stable, so a saved model matches."""
    names = list(BANDS)
    if use_indices:
        names += list(INDEX_NAMES)
    return names


def build_feature_stack(bands: Dict[str, np.ndarray],
                        use_indices: bool = True) -> Tuple[np.ndarray, List[str]]:
    """Stack bands and indices into ``(n_pixels, n_features)`` plus names.

    Rows are pixels in row-major order, so the result reshapes straight back to
    the raster grid. Returns float32 to halve memory against float64, which
    matters on a laptop once a scene passes a few megapixels.
    """
    missing = [b for b in BANDS if b not in bands]
    if missing:
        raise KeyError("missing band(s) {}; have {}".format(
            missing, sorted(bands)))

    shape = np.asarray(bands[BANDS[0]]).shape
    for name in BANDS:
        if np.asarray(bands[name]).shape != shape:
            raise ValueError("band {} has shape {} but {} has {}".format(
                name, np.asarray(bands[name]).shape, BANDS[0], shape))

    layers = [np.asarray(bands[b], dtype="float64") for b in BANDS]
    if use_indices:
        idx = compute_indices(bands)
        layers += [idx[n] for n in INDEX_NAMES]

    flat = np.stack([layer.ravel() for layer in layers], axis=1)
    return flat.astype("float32"), feature_names(use_indices)
