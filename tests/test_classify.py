"""Tests for the split, the classifiers and the accuracy assessment."""

import numpy as np
import pytest

from s2landcover import accuracy
from s2landcover.classify import (kmeans_segment, predict_raster,
                                  train_random_forest, train_test_split_blocks)
from s2landcover.features import build_feature_stack
from s2landcover.scene import synthetic_scene


# ------------------------------------------------------- block split

def test_split_returns_disjoint_index_sets():
    labels = np.ones((64, 64), dtype="int16")
    train, test = train_test_split_blocks(labels, block=8, seed=1)
    assert set(train.tolist()).isdisjoint(test.tolist())
    assert train.size + test.size == labels.size


def test_split_only_returns_labelled_pixels():
    labels = np.zeros((32, 32), dtype="int16")
    labels[:8, :8] = 3
    train, test = train_test_split_blocks(labels, block=4, seed=0)
    flat = labels.ravel()
    assert np.all(flat[train] > 0)
    assert np.all(flat[test] > 0)
    assert train.size + test.size == 64


def test_split_keeps_whole_blocks_on_one_side():
    # This is the property the whole function exists for: a block must never
    # be divided, or autocorrelated neighbours land on both sides and the test
    # score becomes meaningless.
    labels = np.ones((32, 32), dtype="int16")
    block = 8
    train, test = train_test_split_blocks(labels, block=block, seed=3)
    cols = labels.shape[1]
    side = np.full(labels.size, -1, dtype="int8")
    side[train] = 0
    side[test] = 1
    grid = side.reshape(labels.shape)
    for by in range(0, labels.shape[0], block):
        for bx in range(0, cols, block):
            patch = grid[by:by + block, bx:bx + block]
            assert len(np.unique(patch)) == 1, "block was split across sides"


def test_split_is_reproducible_for_a_seed():
    labels = np.ones((40, 40), dtype="int16")
    a = train_test_split_blocks(labels, block=8, seed=7)
    b = train_test_split_blocks(labels, block=8, seed=7)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_split_rejects_bad_parameters():
    labels = np.ones((16, 16), dtype="int16")
    with pytest.raises(ValueError):
        train_test_split_blocks(labels, test_fraction=0.0)
    with pytest.raises(ValueError):
        train_test_split_blocks(labels, test_fraction=1.0)
    with pytest.raises(ValueError):
        train_test_split_blocks(labels, block=0)
    with pytest.raises(ValueError):
        train_test_split_blocks(np.ones(16, dtype="int16"))


def test_split_raises_when_one_side_ends_up_empty():
    # A block larger than the labelled area cannot be divided.
    labels = np.zeros((32, 32), dtype="int16")
    labels[0, 0] = 1
    with pytest.raises(ValueError) as e:
        train_test_split_blocks(labels, block=64, test_fraction=0.5, seed=0)
    assert "block" in str(e.value)


# --------------------------------------------------- random forest

@pytest.fixture(scope="module")
def scene():
    return synthetic_scene(shape=(128, 128), seed=4)


def test_forest_trains_and_reports_importances(scene):
    pytest.importorskip("sklearn")
    features, names = build_feature_stack(scene.bands)
    clf = train_random_forest(features, scene.labels, names, n_estimators=60, seed=0)
    assert clf.n_train > 0 and clf.n_test > 0
    assert set(clf.importances) == set(names)
    assert sum(clf.importances.values()) == pytest.approx(1.0, abs=1e-6)


def test_forest_rejects_a_feature_label_size_mismatch(scene):
    pytest.importorskip("sklearn")
    features, names = build_feature_stack(scene.bands)
    with pytest.raises(ValueError):
        train_random_forest(features[:10], scene.labels, names)


def test_predict_raster_covers_every_pixel(scene):
    pytest.importorskip("sklearn")
    features, names = build_feature_stack(scene.bands)
    clf = train_random_forest(features, scene.labels, names, n_estimators=40, seed=0)
    out = predict_raster(clf, features, scene.shape)
    assert out.shape == scene.shape
    assert np.count_nonzero(out == 0) == 0


def test_predict_raster_honours_a_validity_mask(scene):
    pytest.importorskip("sklearn")
    features, names = build_feature_stack(scene.bands)
    clf = train_random_forest(features, scene.labels, names, n_estimators=40, seed=0)
    valid = np.ones(scene.shape, dtype=bool)
    valid[:20, :] = False
    out = predict_raster(clf, features, scene.shape, valid=valid)
    # Masked pixels must be no-data, not an invented class.
    assert np.all(out[:20, :] == 0)
    assert np.all(out[20:, :] > 0)


def test_predict_raster_rejects_a_wrong_shaped_mask(scene):
    pytest.importorskip("sklearn")
    features, names = build_feature_stack(scene.bands)
    clf = train_random_forest(features, scene.labels, names, n_estimators=20, seed=0)
    with pytest.raises(ValueError):
        predict_raster(clf, features, scene.shape,
                       valid=np.ones((4, 4), dtype=bool))


# ---------------------------------------------------------- k-means

def test_kmeans_returns_ids_starting_at_one(scene):
    pytest.importorskip("sklearn")
    features, _ = build_feature_stack(scene.bands)
    out = kmeans_segment(features, scene.shape, n_clusters=5, seed=0)
    assert out.shape == scene.shape
    assert out.min() >= 1
    assert out.max() <= 5
    # 0 stays reserved for no-data, so no cluster may claim it.
    assert np.count_nonzero(out == 0) == 0


def test_kmeans_finds_the_requested_number_of_clusters(scene):
    pytest.importorskip("sklearn")
    features, _ = build_feature_stack(scene.bands)
    out = kmeans_segment(features, scene.shape, n_clusters=4, seed=0)
    assert len(np.unique(out)) == 4


def test_kmeans_rejects_one_cluster(scene):
    features, _ = build_feature_stack(scene.bands)
    with pytest.raises(ValueError):
        kmeans_segment(features, scene.shape, n_clusters=1)


# --------------------------------------------------------- accuracy

def test_kappa_is_one_for_perfect_agreement():
    ref = np.array([1, 2, 3, 3, 1])
    res = accuracy.assess(ref, ref.copy())
    assert res.overall_accuracy == pytest.approx(1.0)
    assert res.kappa == pytest.approx(1.0)


def test_majority_class_guessing_scores_high_oa_and_zero_kappa():
    # 90 percent one class. Predicting it everywhere looks good on OA and is
    # useless, which is exactly why kappa and per-class recall are reported.
    ref = np.array([3] * 90 + [1] * 10)
    pred = np.array([3] * 100)
    res = accuracy.assess(ref, pred, labels=[1, 3])
    assert res.overall_accuracy == pytest.approx(0.90)
    assert res.kappa == pytest.approx(0.0, abs=1e-12)
    assert res.producers_accuracy[1] == pytest.approx(0.0)


def test_confusion_matrix_is_reference_rows_by_predicted_columns():
    labels, m = accuracy.confusion_matrix(np.array([2]), np.array([1]),
                                          labels=[1, 2])
    assert m[1, 0] == 1 and m[0, 1] == 0


def test_nodata_is_excluded_from_the_sample():
    res = accuracy.assess(np.array([0, 2, 3]), np.array([1, 2, 3]))
    assert res.n == 2
