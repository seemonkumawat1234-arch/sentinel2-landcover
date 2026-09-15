"""sentinel2-landcover: supervised and unsupervised land cover classification.

Random forest and k-means over Sentinel-2 bands plus spectral indices, with a
spatial-block train/test split and full accuracy assessment. CPU only.
"""

from .accuracy import ConfusionResult, assess, confusion_matrix
from .classify import (ClassifierResult, kmeans_segment, predict_raster,
                       train_random_forest, train_test_split_blocks)
from .features import (BANDS, INDEX_NAMES, build_feature_stack,
                       compute_indices, feature_names, normalised_difference)
from .pipeline import LandcoverResult, plot_result, run_supervised, run_unsupervised
from .scene import (CLASSES, LandcoverScene, class_names, load_bands,
                    load_labels, synthetic_scene, write_geotiff)

__version__ = "0.1.0"

__all__ = [
    "BANDS", "INDEX_NAMES", "normalised_difference", "compute_indices",
    "build_feature_stack", "feature_names",
    "CLASSES", "class_names", "LandcoverScene", "synthetic_scene",
    "load_bands", "load_labels", "write_geotiff",
    "ClassifierResult", "train_test_split_blocks", "train_random_forest",
    "predict_raster", "kmeans_segment",
    "ConfusionResult", "assess", "confusion_matrix",
    "LandcoverResult", "run_supervised", "run_unsupervised", "plot_result",
    "__version__",
]
