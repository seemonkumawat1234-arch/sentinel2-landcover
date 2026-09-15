"""Supervised and unsupervised land cover classification.

Supervised uses a random forest. Unsupervised uses k-means. Both are CPU only
and both are in scikit-learn, so there is no GPU path to miss.

The one design decision worth stating: sampling is split by **spatial block**,
not by random pixel. See ``train_test_split_blocks``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["ClassifierResult", "train_test_split_blocks",
           "train_random_forest", "predict_raster", "kmeans_segment"]


@dataclass
class ClassifierResult:
    """A fitted supervised model and what it was trained on."""

    model: object
    feature_names: List[str]
    classes: List[int]
    n_train: int
    n_test: int
    train_index: np.ndarray
    test_index: np.ndarray
    importances: Dict[str, float] = field(default_factory=dict)

    def importance_report(self, top: int = 10) -> str:
        rows = sorted(self.importances.items(), key=lambda kv: -kv[1])[:top]
        width = max((len(k) for k, _ in rows), default=8)
        lines = ["{:<{w}} {:>8}".format("feature", "weight", w=width)]
        for name, val in rows:
            lines.append("{:<{w}} {:>7.3f}".format(name, val, w=width))
        return "\n".join(lines)


def train_test_split_blocks(labels: np.ndarray, block: int = 16,
                            test_fraction: float = 0.35,
                            seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Split labelled pixels into train and test by spatial block.

    ``labels`` is a 2D array of class ids with 0 for unlabelled. Returns two
    flat index arrays into ``labels.ravel()``.

    Why blocks rather than random pixels
    ------------------------------------
    Neighbouring pixels in satellite imagery are strongly autocorrelated, so a
    random pixel split puts near-duplicates of the same ground on both sides.
    The model then scores on test pixels it has effectively already seen, and
    the reported accuracy comes out far above what the map achieves anywhere
    new. Overall accuracies in the high nineties from a random pixel split over
    hand-drawn training polygons are usually this artefact, not a good model.

    Splitting by contiguous block keeps each patch of ground wholly on one side
    and gives a number that means something. It is still optimistic, because
    blocks from the same scene share illumination and phenology, but it is much
    closer to honest.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("labels must be 2D, got shape {}".format(labels.shape))
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if block < 1:
        raise ValueError("block must be at least 1 pixel")

    rows, cols = labels.shape
    by, bx = (rows + block - 1) // block, (cols + block - 1) // block
    rng = np.random.default_rng(seed)
    assign = rng.random((by, bx)) < test_fraction

    block_id_y = np.arange(rows) // block
    block_id_x = np.arange(cols) // block
    is_test = assign[block_id_y][:, block_id_x]

    labelled = labels > 0
    flat_labelled = labelled.ravel()
    flat_test = is_test.ravel()

    all_idx = np.nonzero(flat_labelled)[0]
    test_idx = all_idx[flat_test[all_idx]]
    train_idx = all_idx[~flat_test[all_idx]]

    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError(
            "block split left one side empty ({} train, {} test). The block "
            "size is probably larger than the labelled area; reduce block="
            .format(train_idx.size, test_idx.size))
    return train_idx, test_idx


def train_random_forest(features: np.ndarray, labels: np.ndarray,
                        feature_names: Sequence[str],
                        block: int = 16, test_fraction: float = 0.35,
                        n_estimators: int = 300, max_depth: Optional[int] = None,
                        seed: int = 0,
                        class_weight: Optional[str] = "balanced_subsample"
                        ) -> ClassifierResult:
    """Fit a random forest on block-split labelled pixels.

    ``class_weight="balanced_subsample"`` by default because land cover classes
    are almost never balanced. Without it the forest optimises for the majority
    class and the minority classes, which are usually the interesting ones,
    come out with poor recall while overall accuracy still looks fine.
    """
    from sklearn.ensemble import RandomForestClassifier

    features = np.asarray(features)
    labels2d = np.asarray(labels)
    if features.shape[0] != labels2d.size:
        raise ValueError(
            "features has {} rows but labels has {} pixels".format(
                features.shape[0], labels2d.size))

    train_idx, test_idx = train_test_split_blocks(
        labels2d, block=block, test_fraction=test_fraction, seed=seed)
    flat_labels = labels2d.ravel()

    x_train = np.nan_to_num(features[train_idx], nan=0.0)
    y_train = flat_labels[train_idx]

    if len(np.unique(y_train)) < 2:
        raise ValueError("training split has only one class; the block split "
                         "or the labels need adjusting")

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight=class_weight, random_state=seed, n_jobs=-1)
    model.fit(x_train, y_train)

    importances = {name: float(w) for name, w
                   in zip(feature_names, model.feature_importances_)}

    return ClassifierResult(
        model=model, feature_names=list(feature_names),
        classes=[int(c) for c in model.classes_],
        n_train=int(train_idx.size), n_test=int(test_idx.size),
        train_index=train_idx, test_index=test_idx, importances=importances)


def predict_raster(result: ClassifierResult, features: np.ndarray,
                   shape: Tuple[int, int],
                   valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Classify every pixel and reshape to the raster grid.

    ``valid`` is an optional 2D boolean mask. Pixels outside it come back as 0,
    which every scheme here reserves for no-data. Passing a cloud or scene
    mask here keeps invented classes out of areas with no usable observation.
    """
    features = np.asarray(features)
    n = shape[0] * shape[1]
    if features.shape[0] != n:
        raise ValueError("features has {} rows but shape implies {}".format(
            features.shape[0], n))

    out = np.zeros(n, dtype="int16")
    if valid is None:
        keep = np.ones(n, dtype=bool)
    else:
        valid = np.asarray(valid)
        if valid.shape != shape:
            raise ValueError("valid mask shape {} does not match {}".format(
                valid.shape, shape))
        keep = valid.ravel()

    if keep.any():
        x = np.nan_to_num(features[keep], nan=0.0)
        out[keep] = result.model.predict(x).astype("int16")
    return out.reshape(shape)


def kmeans_segment(features: np.ndarray, shape: Tuple[int, int],
                   n_clusters: int = 6, seed: int = 0,
                   sample: int = 60_000) -> np.ndarray:
    """Unsupervised k-means clustering, returning cluster ids starting at 1.

    Fitted on a random subsample and then applied to every pixel, which is what
    makes this tractable on a laptop: fitting on 60,000 sampled pixels rather
    than all of them cuts the cost by orders of magnitude on a large scene and
    changes the cluster centres very little.

    Cluster ids are arbitrary. They are not land cover classes and carry no
    ordering. Interpreting them requires a human looking at the result, which
    is the fundamental limit of unsupervised classification, not a gap here.
    """
    from sklearn.cluster import KMeans

    features = np.asarray(features)
    n = shape[0] * shape[1]
    if features.shape[0] != n:
        raise ValueError("features has {} rows but shape implies {}".format(
            features.shape[0], n))
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")

    x = np.nan_to_num(features, nan=0.0)
    rng = np.random.default_rng(seed)
    if x.shape[0] > sample:
        fit_on = x[rng.choice(x.shape[0], sample, replace=False)]
    else:
        fit_on = x

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(fit_on)
    return (km.predict(x).astype("int16") + 1).reshape(shape)
