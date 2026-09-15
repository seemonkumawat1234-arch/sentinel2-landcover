"""Scene and label input, output, and a synthetic scene for testing.

``load_bands``      reads Sentinel-2 band GeoTIFFs.
``load_labels``     reads a rasterised training-class raster.
``synthetic_scene`` generates a labelled multi-class scene with no download.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import BANDS

__all__ = ["LandcoverScene", "CLASSES", "class_names", "synthetic_scene",
           "load_bands", "load_labels", "write_geotiff"]

DARWIN_UTM = "EPSG:32752"
DEFAULT_PIXEL_M = 10.0          # Sentinel-2 B2/B3/B4/B8 native resolution

# Class 0 is always no-data / unlabelled, in every array this package touches.
CLASSES: Tuple[Tuple[int, str], ...] = (
    (1, "Water"),
    (2, "Dense woodland"),
    (3, "Open savanna"),
    (4, "Grassland"),
    (5, "Bare / soil"),
    (6, "Built"),
    (7, "Burnt"),
)


def class_names() -> Dict[int, str]:
    return {cid: name for cid, name in CLASSES}


@dataclass
class LandcoverScene:
    """Bands on a grid, with optional reference labels."""

    bands: Dict[str, np.ndarray]
    labels: Optional[np.ndarray] = None       # int16, 0 = unlabelled
    truth: Optional[np.ndarray] = None        # full-coverage truth, synthetic only
    crs: str = DARWIN_UTM
    transform: Optional[tuple] = None
    pixel_size_m: float = DEFAULT_PIXEL_M
    synthetic: bool = False

    @property
    def shape(self) -> Tuple[int, int]:
        return np.asarray(self.bands[BANDS[0]]).shape


def _upsample(small: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    n = small.shape[0]
    cy = np.linspace(0.0, n - 1.0, rows)
    cx = np.linspace(0.0, n - 1.0, cols)
    y0, x0 = np.floor(cy).astype(int), np.floor(cx).astype(int)
    y1, x1 = np.minimum(y0 + 1, n - 1), np.minimum(x0 + 1, n - 1)
    fy, fx = (cy - y0)[:, None], (cx - x0)[None, :]
    top = small[y0][:, x0] * (1 - fx) + small[y0][:, x1] * fx
    bot = small[y1][:, x0] * (1 - fx) + small[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


def _coherent_field(shape: Tuple[int, int], cells: int,
                    rng: np.random.Generator, octaves: int = 3) -> np.ndarray:
    """Multi-octave value noise in 0..1 with landscape-scale structure.

    Blurred white noise has a correlation length of a few pixels, so land cover
    patches cut from it would be speckle. Generating coarse and upsampling sets
    the dominant patch size to roughly the scene size over ``cells``, which is
    what makes a spatial block split meaningful: with speckle, every block
    contains every class and the split proves nothing.
    """
    total = np.zeros(shape, dtype="float64")
    amp, norm, c = 1.0, 0.0, max(2, cells)
    for _ in range(max(1, octaves)):
        total += amp * _upsample(rng.random((c, c)), shape)
        norm += amp
        amp *= 0.5
        c = min(c * 2, min(shape))
    out = total / norm
    return (out - out.min()) / (out.max() - out.min() + 1e-12)


# Mean reflectance per class, ordered as features.BANDS. These are plausible
# rather than measured: the point is that the classes are separable in a way a
# real classifier finds non-trivial, with overlapping pairs (open savanna
# against grassland, bare against built) that force the indices to do work.
_SIGNATURES: Dict[int, Tuple[float, ...]] = {
    1: (0.040, 0.055, 0.035, 0.020, 0.012, 0.010),   # Water
    2: (0.030, 0.055, 0.035, 0.330, 0.150, 0.070),   # Dense woodland
    3: (0.055, 0.085, 0.080, 0.250, 0.220, 0.140),   # Open savanna
    4: (0.070, 0.105, 0.115, 0.215, 0.255, 0.180),   # Grassland
    5: (0.130, 0.165, 0.205, 0.255, 0.330, 0.290),   # Bare / soil
    6: (0.150, 0.170, 0.185, 0.215, 0.290, 0.265),   # Built
    7: (0.045, 0.060, 0.070, 0.090, 0.210, 0.260),   # Burnt
}


def synthetic_scene(shape: Tuple[int, int] = (256, 256),
                    seed: int = 20260915,
                    label_fraction: float = 0.04,
                    noise: float = 0.032) -> LandcoverScene:
    """Generate a labelled multi-class scene.

    Returns a scene carrying both ``truth`` (every pixel's true class, used for
    honest evaluation) and ``labels`` (a sparse subset, standing in for
    hand-drawn training data).

    ``noise`` is per-band reflectance standard deviation. The default is set
    so the class signatures genuinely overlap, because the pairs that matter
    are separated by only 0.02 to 0.04 reflectance: open savanna against
    grassland, and bare soil against built. A synthetic scene that classifies
    at 99 percent measures nothing, and published Sentinel-2 land cover
    accuracies sit well below that.
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape

    # Independent fields decide the class, which produces contiguous regions
    # with realistic adjacency rather than stripes.
    wetness = _coherent_field(shape, cells=5, rng=rng, octaves=4)
    cover = _coherent_field(shape, cells=7, rng=rng, octaves=4)
    disturbance = _coherent_field(shape, cells=11, rng=rng, octaves=3)
    urban = _coherent_field(shape, cells=13, rng=rng, octaves=2)

    # Thresholds are quantiles of each field, not fixed values, so every class
    # gets a controlled share of the scene. Intersecting the tails of two
    # independent fields, which an earlier version did for the built class,
    # produces an area that collapses to a handful of pixels once the fields
    # happen to be spatially anti-correlated. A class with five pixels cannot
    # be trained or scored, and its accuracy row comes back as NaN.
    def q(field, p):
        return float(np.quantile(field, p))

    truth = np.full(shape, 3, dtype="int16")
    truth[cover >= q(cover, 0.72)] = 2                              # woodland
    truth[(cover < q(cover, 0.72)) & (cover >= q(cover, 0.40))] = 3  # savanna
    truth[(cover < q(cover, 0.40)) & (cover >= q(cover, 0.14))] = 4  # grassland
    truth[cover < q(cover, 0.14)] = 5                               # bare
    # These overwrite the cover classes, in increasing priority.
    truth[disturbance >= q(disturbance, 0.87)] = 7                  # burnt
    truth[urban >= q(urban, 0.955)] = 6                             # built
    truth[wetness >= q(wetness, 0.955)] = 1                         # water

    # Reflectance from the class signatures plus noise.
    bands: Dict[str, np.ndarray] = {}
    for i, band in enumerate(BANDS):
        arr = np.zeros(shape, dtype="float64")
        for cid, sig in _SIGNATURES.items():
            sel = truth == cid
            if sel.any():
                arr[sel] = sig[i]
        # A broad illumination gradient, standing in for terrain and view-angle
        # effects. Spatially correlated error is what breaks naive classifiers.
        yy = np.linspace(-1.0, 1.0, rows)[:, None]
        xx = np.linspace(-1.0, 1.0, cols)[None, :]
        arr = arr * (1.0 + 0.10 * (0.6 * yy + 0.4 * xx))
        arr = arr + rng.normal(0.0, noise, shape)
        bands[band] = np.clip(arr, 0.0, 1.0)

    # Sparse labels, in patches rather than scattered pixels, because that is
    # what hand-drawn training polygons actually look like.
    patch_field = _coherent_field(shape, cells=max(8, min(shape) // 24),
                                  rng=rng, octaves=2)
    label_mask = patch_field > np.quantile(patch_field, 1.0 - label_fraction)
    labels = np.where(label_mask, truth, 0).astype("int16")

    # Any class missing from the labels cannot be learned, so seed a few
    # pixels for it. Otherwise a rare class simply vanishes from the model and
    # the test would be exercising a different problem than intended.
    for cid, _name in CLASSES:
        if np.count_nonzero(labels == cid) < 40:
            where = np.nonzero(truth == cid)
            if where[0].size:
                take = min(120, where[0].size)
                pick = rng.choice(where[0].size, take, replace=False)
                labels[where[0][pick], where[1][pick]] = cid

    transform = (DEFAULT_PIXEL_M, 0.0, 700000.0,
                 0.0, -DEFAULT_PIXEL_M, 8620000.0)
    return LandcoverScene(bands=bands, labels=labels, truth=truth,
                          crs=DARWIN_UTM, transform=transform,
                          pixel_size_m=DEFAULT_PIXEL_M, synthetic=True)


def load_bands(paths: Dict[str, str], scale: float = 10000.0) -> LandcoverScene:
    """Load Sentinel-2 bands from GeoTIFFs onto a common grid.

    Refuses to run when bands disagree on CRS, transform or shape, rather than
    resampling silently. A band resampled without the caller knowing produces a
    classification that looks fine and is wrong along every edge.
    """
    import rasterio

    missing = [b for b in BANDS if b not in paths]
    if missing:
        raise KeyError("missing path(s) for band(s) {}".format(missing))

    bands, meta = {}, None
    for name in BANDS:
        path = paths[name]
        if not os.path.isfile(path):
            raise FileNotFoundError("band {}: {}".format(name, path))
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype("float64")
            if ds.nodata is not None:
                arr[arr == ds.nodata] = np.nan
            bands[name] = arr / scale
            this = (ds.crs.to_string() if ds.crs else None,
                    tuple(ds.transform)[:6], ds.shape)
        if meta is None:
            meta = this
        elif this != meta:
            raise ValueError(
                "band {} is not on the same grid as {}. Got {} but expected "
                "{}. Resample to a common grid first.".format(
                    name, BANDS[0], this, meta))

    crs, transform, _shape = meta
    return LandcoverScene(bands=bands, crs=crs or DARWIN_UTM,
                          transform=transform, pixel_size_m=abs(transform[0]),
                          synthetic=False)


def load_labels(path: str, shape: Tuple[int, int]) -> np.ndarray:
    """Load a rasterised training-class raster, 0 for unlabelled."""
    import rasterio

    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        if ds.nodata is not None:
            arr = np.where(arr == ds.nodata, 0, arr)
        if ds.shape != shape:
            raise ValueError("labels are {} but the scene is {}".format(
                ds.shape, shape))
    out = np.asarray(arr).astype("int16")
    if (out < 0).any():
        raise ValueError("label raster holds negative values; mask no-data to 0")
    return out


def write_geotiff(path: str, array: np.ndarray, scene: LandcoverScene,
                  dtype: str | None = None, nodata=None) -> str:
    """Write one array as a GeoTIFF carrying the scene's geometry."""
    import rasterio
    from rasterio.transform import Affine

    arr = np.asarray(array)
    dtype = dtype or ("int16" if arr.dtype.kind in "iu" else "float32")
    arr = arr.astype(dtype)

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    transform = Affine(*scene.transform) if scene.transform else Affine.identity()
    profile = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                   count=1, dtype=dtype, crs=scene.crs, transform=transform,
                   compress="deflate", tiled=True)
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr, 1)
        tags = {"generator": "sentinel2-landcover"}
        if scene.synthetic:
            tags["SYNTHETIC_INPUT"] = "1"
            tags["WARNING"] = ("derived from a synthetic scene, "
                               "not real satellite imagery")
        ds.update_tags(**tags)
    return path
