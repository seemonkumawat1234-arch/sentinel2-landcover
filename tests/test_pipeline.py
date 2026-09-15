"""End-to-end tests on the synthetic scene."""

import os

import numpy as np
import pytest

from s2landcover import (CLASSES, plot_result, run_supervised,
                         run_unsupervised, synthetic_scene)
from s2landcover.scene import LandcoverScene, load_bands, write_geotiff

pytest.importorskip("sklearn")


@pytest.fixture(scope="module")
def scene():
    return synthetic_scene(shape=(160, 160), seed=6)


@pytest.fixture(scope="module")
def supervised(scene):
    return run_supervised(scene, n_estimators=120, seed=0)


# ------------------------------------------------------- synthetic scene

def test_scene_is_flagged_synthetic(scene):
    assert scene.synthetic is True


def test_scene_reflectance_is_physical(scene):
    for band, arr in scene.bands.items():
        assert arr.min() >= 0.0, band
        assert arr.max() <= 1.0, band


def test_scene_is_reproducible():
    a = synthetic_scene(shape=(64, 64), seed=11)
    b = synthetic_scene(shape=(64, 64), seed=11)
    np.testing.assert_allclose(a.bands["B8"], b.bands["B8"])
    np.testing.assert_array_equal(a.truth, b.truth)


def test_scene_contains_every_class(scene):
    # A class missing from truth cannot be learned or scored, so the test
    # would silently be exercising a smaller problem than intended.
    present = set(np.unique(scene.truth).tolist())
    for cid, name in CLASSES:
        assert cid in present, "class {} ({}) absent from truth".format(cid, name)


def test_labels_cover_every_class_and_are_sparse(scene):
    labelled = set(np.unique(scene.labels).tolist()) - {0}
    for cid, name in CLASSES:
        assert cid in labelled, "class {} ({}) has no labels".format(cid, name)
    fraction = np.count_nonzero(scene.labels) / scene.labels.size
    assert 0.0 < fraction < 0.25, "labels should be sparse, got {:.1%}".format(fraction)


def test_labels_never_disagree_with_truth(scene):
    sel = scene.labels > 0
    np.testing.assert_array_equal(scene.labels[sel], scene.truth[sel])


def test_classes_are_not_trivially_separable(scene):
    # Two of the hardest pairs must actually overlap in reflectance, or the
    # scene is too easy and any accuracy figure from it means nothing.
    for a, b in ((3, 4), (5, 6)):        # savanna/grassland, bare/built
        band = scene.bands["B11"]
        lo, hi = band[scene.truth == a], band[scene.truth == b]
        assert lo.min() < hi.max() and hi.min() < lo.max(), \
            "classes {} and {} do not overlap at all".format(a, b)


# ------------------------------------------------------------ supervised

def test_supervised_classifies_every_pixel(supervised, scene):
    assert supervised.classified.shape == scene.shape
    assert np.count_nonzero(supervised.classified == 0) == 0


def test_supervised_beats_chance_by_a_wide_margin(supervised):
    assert supervised.accuracy_test.overall_accuracy > 0.70
    assert supervised.accuracy_test.kappa > 0.60


def test_supervised_reports_both_accuracy_figures(supervised):
    # Held-out blocks of the sparse labels, and the full-scene truth. The
    # difference between them is the honest measure of how much hand-drawn
    # training data flatters a map.
    assert supervised.accuracy_test is not None
    assert supervised.accuracy_full is not None
    assert supervised.accuracy_full.n > supervised.accuracy_test.n


def test_label_accuracy_does_not_understate_full_truth(supervised):
    # Labelled patches sit in the clearest parts of each class, so the
    # held-out figure should be at least as high as full-scene truth. If it
    # ever came out materially lower, something is wrong with the split.
    assert (supervised.accuracy_test.overall_accuracy
            >= supervised.accuracy_full.overall_accuracy - 0.05)


def test_every_class_gets_some_recall(supervised):
    # A class with zero producer's accuracy means the model never finds it,
    # which overall accuracy would happily hide.
    for cid, name in CLASSES:
        pa = supervised.accuracy_full.producers_accuracy.get(cid)
        if pa is not None and not np.isnan(pa):
            assert pa > 0.10, "class {} ({}) has recall {:.2f}".format(cid, name, pa)


def test_indices_are_used_by_the_model(supervised):
    from s2landcover.features import INDEX_NAMES
    total = sum(supervised.classifier.importances.get(n, 0.0)
                for n in INDEX_NAMES)
    # If the indices contributed nothing the forest would be ignoring them,
    # and computing them would be wasted work.
    assert total > 0.05


def test_class_areas_sum_to_the_scene(supervised, scene):
    px_ha = (scene.pixel_size_m ** 2) / 10_000.0
    total = scene.shape[0] * scene.shape[1] * px_ha
    assert sum(supervised.class_areas_ha.values()) == pytest.approx(total)


def test_summary_marks_synthetic_and_reports_the_gap(supervised):
    text = supervised.text_summary()
    assert "SYNTHETIC" in text
    assert "held-out" in text.lower()
    assert "points against" in text


def test_supervised_requires_labels(scene):
    bare = LandcoverScene(bands=scene.bands, labels=None)
    with pytest.raises(ValueError):
        run_supervised(bare)


def test_supervised_rejects_empty_labels(scene):
    bare = LandcoverScene(bands=scene.bands,
                          labels=np.zeros(scene.shape, dtype="int16"))
    with pytest.raises(ValueError):
        run_supervised(bare)


def test_dropping_indices_still_runs(scene):
    result = run_supervised(scene, use_indices=False, n_estimators=60, seed=0)
    assert result.accuracy_test.overall_accuracy > 0.5


# ---------------------------------------------------------- unsupervised

def test_unsupervised_runs_without_labels(scene):
    bare = LandcoverScene(bands=scene.bands, labels=None,
                          pixel_size_m=scene.pixel_size_m,
                          transform=scene.transform, synthetic=True)
    result = run_unsupervised(bare, n_clusters=5, seed=0)
    assert result.mode == "unsupervised"
    assert result.n_clusters == 5
    assert len(np.unique(result.classified)) == 5
    # No accuracy is reported, because cluster ids are not classes and
    # inventing a score would be dishonest.
    assert result.accuracy_test is None
    assert result.accuracy_full is None


def test_unsupervised_areas_sum_to_the_scene(scene):
    result = run_unsupervised(scene, n_clusters=4, seed=0)
    px_ha = (scene.pixel_size_m ** 2) / 10_000.0
    total = scene.shape[0] * scene.shape[1] * px_ha
    assert sum(result.class_areas_ha.values()) == pytest.approx(total)


# --------------------------------------------------------------- output

def test_geotiff_written_with_geometry_and_synthetic_tag(tmp_path, scene):
    rasterio = pytest.importorskip("rasterio")
    result = run_supervised(scene, n_estimators=60, seed=0, out_dir=str(tmp_path))
    path = result.written["classified"]
    assert os.path.isfile(path)
    with rasterio.open(path) as ds:
        assert ds.crs.to_string() == scene.crs
        assert tuple(ds.transform)[:6] == pytest.approx(scene.transform)
        assert ds.tags().get("SYNTHETIC_INPUT") == "1"
        assert ds.nodata == pytest.approx(0.0)


def test_load_bands_rejects_mismatched_grids(tmp_path, scene):
    pytest.importorskip("rasterio")
    from s2landcover.features import BANDS
    small = synthetic_scene(shape=(32, 32), seed=1)
    paths = {}
    for i, band in enumerate(BANDS):
        src = small if i == 0 else scene
        p = str(tmp_path / "{}.tif".format(band))
        write_geotiff(p, (src.bands[band] * 10000).astype("int16"), src, "int16")
        paths[band] = p
    with pytest.raises(ValueError) as e:
        load_bands(paths)
    assert "grid" in str(e.value).lower()


def test_figure_written_for_both_modes(tmp_path, scene, supervised):
    pytest.importorskip("matplotlib")
    a = plot_result(supervised, scene, str(tmp_path / "sup.png"))
    assert os.path.getsize(a) > 20_000
    unsup = run_unsupervised(scene, n_clusters=5, seed=0)
    b = plot_result(unsup, scene, str(tmp_path / "uns.png"))
    assert os.path.getsize(b) > 20_000
