"""End-to-end classification, evaluation and figures."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .accuracy import ConfusionResult, assess
from .classify import (ClassifierResult, kmeans_segment, predict_raster,
                       train_random_forest)
from .features import build_feature_stack
from .scene import CLASSES, LandcoverScene, class_names, write_geotiff

__all__ = ["LandcoverResult", "run_supervised", "run_unsupervised", "plot_result"]


@dataclass
class LandcoverResult:
    """One classification run."""

    classified: np.ndarray
    mode: str                                    # "supervised" or "unsupervised"
    class_areas_ha: Dict[str, float]
    classifier: Optional[ClassifierResult] = None
    accuracy_test: Optional[ConfusionResult] = None
    accuracy_full: Optional[ConfusionResult] = None
    n_clusters: Optional[int] = None
    synthetic: bool = False
    written: Dict[str, str] = field(default_factory=dict)

    def text_summary(self) -> str:
        names = class_names()
        lines = []
        if self.synthetic:
            lines += ["*** SYNTHETIC SCENE, NOT REAL IMAGERY ***", ""]
        lines.append("Mode: {}".format(self.mode))
        if self.classifier is not None:
            lines.append("Training pixels: {:,}   held-out pixels: {:,}".format(
                self.classifier.n_train, self.classifier.n_test))
        lines.append("")
        lines.append("{:<18} {:>14}".format("class", "hectares"))
        for label, ha in self.class_areas_ha.items():
            lines.append("{:<18} {:>14,.1f}".format(label, ha))

        if self.accuracy_test is not None:
            lines += ["", "Accuracy on held-out spatial blocks", "-" * 44,
                      self.accuracy_test.report(names)]
        if self.accuracy_full is not None:
            lines += ["", "Accuracy against full-scene truth", "-" * 44,
                      self.accuracy_full.report(names)]
            if self.accuracy_test is not None:
                gap = (self.accuracy_test.overall_accuracy
                       - self.accuracy_full.overall_accuracy) * 100
                if gap > 3.0:
                    note = ("The labelled patches are easier than the scene as "
                            "a whole, which is normal for hand-drawn training "
                            "data and is why the held-out figure flatters the "
                            "map.")
                elif gap < -3.0:
                    note = ("The held-out blocks are harder than the scene "
                            "average, so the reported figure is conservative "
                            "here. Check whether the labels concentrate on "
                            "boundaries or on confusable classes.")
                else:
                    note = ("The two agree closely, which means the labelled "
                            "patches are representative of the scene.")
                lines += ["", "Held-out blocks read {:+.1f} points against "
                              "full truth. {}".format(gap, note)]
        if self.classifier is not None and self.classifier.importances:
            lines += ["", "Feature importance", "-" * 44,
                      self.classifier.importance_report()]
        return "\n".join(lines)


def _areas(classified: np.ndarray, pixel_size_m: float,
           mode: str, n_clusters: Optional[int]) -> Dict[str, float]:
    px_ha = (pixel_size_m ** 2) / 10_000.0
    out: Dict[str, float] = {}
    if mode == "supervised":
        for cid, label in CLASSES:
            out[label] = float(np.count_nonzero(classified == cid) * px_ha)
    else:
        for cid in range(1, (n_clusters or 0) + 1):
            out["Cluster {}".format(cid)] = float(
                np.count_nonzero(classified == cid) * px_ha)
    unclassified = float(np.count_nonzero(classified == 0) * px_ha)
    if unclassified:
        out["Unclassified"] = unclassified
    return out


def run_supervised(scene: LandcoverScene, use_indices: bool = True,
                   block: int = 16, test_fraction: float = 0.35,
                   n_estimators: int = 300, seed: int = 0,
                   out_dir: Optional[str] = None) -> LandcoverResult:
    """Train a random forest on the scene's labels and classify every pixel.

    Two accuracy figures are produced when the scene carries full truth, and
    the difference between them is the point. The first is measured on
    held-out spatial blocks of the sparse labels, which is what you can compute
    for real data. The second is against every pixel's true class, which you
    almost never have. Reporting both shows how far the first overstates the
    map, because hand-drawn training patches sit in the clearest, most typical
    parts of each class and skip the ambiguous boundaries.
    """
    if scene.labels is None:
        raise ValueError("supervised mode needs scene.labels; got None")
    if not np.any(scene.labels > 0):
        raise ValueError("scene.labels has no labelled pixels")

    features, names = build_feature_stack(scene.bands, use_indices=use_indices)
    clf = train_random_forest(features, scene.labels, names, block=block,
                              test_fraction=test_fraction,
                              n_estimators=n_estimators, seed=seed)
    classified = predict_raster(clf, features, scene.shape)

    flat_labels = np.asarray(scene.labels).ravel()
    flat_pred = classified.ravel()
    acc_test = assess(flat_labels[clf.test_index], flat_pred[clf.test_index],
                      labels=clf.classes)

    acc_full = None
    if scene.truth is not None:
        acc_full = assess(np.asarray(scene.truth).ravel(), flat_pred,
                          labels=clf.classes)

    result = LandcoverResult(
        classified=classified, mode="supervised",
        class_areas_ha=_areas(classified, scene.pixel_size_m, "supervised", None),
        classifier=clf, accuracy_test=acc_test, accuracy_full=acc_full,
        synthetic=scene.synthetic)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        result.written["classified"] = write_geotiff(
            os.path.join(out_dir, "landcover_supervised.tif"), classified,
            scene, "int16", nodata=0)
    return result


def run_unsupervised(scene: LandcoverScene, n_clusters: int = 6,
                     use_indices: bool = True, seed: int = 0,
                     out_dir: Optional[str] = None) -> LandcoverResult:
    """k-means clustering, no labels required.

    Cluster ids are arbitrary and carry no ordering or meaning. Turning them
    into land cover classes needs a human comparing clusters against imagery,
    so no accuracy figure is reported here: there is nothing to compare against
    without that interpretation step, and inventing one would be dishonest.
    """
    features, _names = build_feature_stack(scene.bands, use_indices=use_indices)
    clustered = kmeans_segment(features, scene.shape,
                               n_clusters=n_clusters, seed=seed)

    result = LandcoverResult(
        classified=clustered, mode="unsupervised",
        class_areas_ha=_areas(clustered, scene.pixel_size_m,
                              "unsupervised", n_clusters),
        n_clusters=n_clusters, synthetic=scene.synthetic)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        result.written["clusters"] = write_geotiff(
            os.path.join(out_dir, "landcover_clusters.tif"), clustered,
            scene, "int16", nodata=0)
    return result


def _stretch(arr: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Percentile stretch to 0..1 for display."""
    a = np.asarray(arr, dtype="float64")
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


# Index 0 is no-data, then one colour per class id in CLASSES order. Chosen so
# the map reads without the legend: water blue, woody cover dark green through
# to grass yellow-green, bare tan, built grey, burnt near-black.
_PALETTE = [
    "#f4f4f2",   # 0  no-data
    "#2b6cb0",   # 1  Water
    "#14532d",   # 2  Dense woodland
    "#4f8f4a",   # 3  Open savanna
    "#a8c256",   # 4  Grassland
    "#c49a5a",   # 5  Bare / soil
    "#8e8e8e",   # 6  Built
    "#2f2f2f",   # 7  Burnt
]


def plot_result(result: LandcoverResult, scene: LandcoverScene, path: str,
                title: str = "Land cover classification") -> str:
    """Four panels: true colour, training labels or NDVI, the map, and either
    the confusion matrix or cluster areas."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.patch.set_facecolor("white")

    rgb = np.dstack([_stretch(scene.bands["B4"]),
                     _stretch(scene.bands["B3"]),
                     _stretch(scene.bands["B2"])])
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("True colour (B4, B3, B2)")

    supervised = result.mode == "supervised"
    n_ids = len(CLASSES) if supervised else (result.n_clusters or 1)
    ids = list(range(1, n_ids + 1))
    labels = ([name for _cid, name in CLASSES] if supervised
              else ["Cluster {}".format(i) for i in ids])
    # One colour per value 0..n_ids, so no-data gets its own bin. Boundaries
    # must number one more than the colours, or the first bin swallows both 0
    # and 1 and every class after it renders one colour off.
    palette = ListedColormap((_PALETTE * 4)[:n_ids + 1])
    norm = BoundaryNorm([i - 0.5 for i in range(n_ids + 2)], palette.N)

    if supervised and scene.labels is not None:
        axes[0, 1].imshow(scene.labels, cmap=palette, norm=norm,
                          interpolation="nearest")
        axes[0, 1].set_title("Training labels ({:,} px)".format(
            int(np.count_nonzero(scene.labels))))
    else:
        from .features import compute_indices
        axes[0, 1].imshow(compute_indices(scene.bands)["NDVI"],
                          cmap="RdYlGn", vmin=-0.3, vmax=0.9)
        axes[0, 1].set_title("NDVI")

    im2 = axes[1, 0].imshow(result.classified, cmap=palette, norm=norm,
                            interpolation="nearest")
    axes[1, 0].set_title("Classified" if supervised else "k-means clusters")
    cb = fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, ticks=ids)
    cb.ax.set_yticklabels(labels, fontsize=8)

    ax = axes[1, 1]
    if result.accuracy_test is not None:
        cm = result.accuracy_test.matrix.astype("float64")
        rows = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            norm_cm = np.where(rows > 0, cm / rows, 0.0)
        im3 = ax.imshow(norm_cm, cmap="Blues", vmin=0, vmax=1)
        names = class_names()
        ticks = [names.get(c, str(c)) for c in result.accuracy_test.labels]
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(ticks)))
        ax.set_yticklabels(ticks, fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_ylabel("reference")
        ax.set_title("Confusion matrix, row-normalised")
        for i in range(norm_cm.shape[0]):
            for j in range(norm_cm.shape[1]):
                if norm_cm[i, j] >= 0.005:
                    ax.text(j, i, "{:.0f}".format(norm_cm[i, j] * 100),
                            ha="center", va="center", fontsize=6.5,
                            color="white" if norm_cm[i, j] > 0.55 else "#222222")
        fig.colorbar(im3, ax=ax, fraction=0.046)
    else:
        areas = [v for k, v in result.class_areas_ha.items()
                 if k != "Unclassified"]
        ax.bar(range(1, len(areas) + 1), areas, color="#2b6cb0")
        ax.set_title("Cluster areas")
        ax.set_xlabel("cluster")
        ax.set_ylabel("hectares")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    for a in (axes[0, 0], axes[0, 1], axes[1, 0]):
        a.set_xticks([])
        a.set_yticks([])

    bits = []
    if result.accuracy_test is not None:
        bits.append("held-out blocks: OA {:.1%}, kappa {:.3f}".format(
            result.accuracy_test.overall_accuracy, result.accuracy_test.kappa))
    if result.accuracy_full is not None:
        bits.append("full truth: OA {:.1%}, kappa {:.3f}".format(
            result.accuracy_full.overall_accuracy, result.accuracy_full.kappa))
    if result.synthetic:
        bits.append("SYNTHETIC SCENE, NOT REAL IMAGERY")
    subtitle = "   |   ".join(bits)

    fig.suptitle(title, fontsize=15, y=0.977)
    if subtitle:
        fig.text(0.5, 0.944, subtitle, ha="center", fontsize=9,
                 color="#9e2b25" if result.synthetic else "#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.935))

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
