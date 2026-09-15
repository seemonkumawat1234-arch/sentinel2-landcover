"""Tests for feature construction."""

import numpy as np
import pytest

from s2landcover import features


def make_bands(value=0.2, shape=(4, 4)):
    return {b: np.full(shape, value) for b in features.BANDS}


def test_normalised_difference_matches_hand_calculation():
    out = features.normalised_difference(np.array([0.4]), np.array([0.2]))
    assert out[0] == pytest.approx(0.2 / 0.6)


def test_normalised_difference_is_bounded():
    rng = np.random.default_rng(0)
    out = features.normalised_difference(rng.random((40, 40)), rng.random((40, 40)))
    assert np.nanmin(out) >= -1.0 - 1e-12
    assert np.nanmax(out) <= 1.0 + 1e-12


def test_zero_denominator_is_nan_not_zero():
    # A zero index is a real measurement, so no-data must not collapse into it.
    assert np.isnan(features.normalised_difference(np.array([0.0]), np.array([0.0]))[0])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        features.normalised_difference(np.zeros((3, 3)), np.zeros((3, 4)))


def test_compute_indices_returns_all_five():
    idx = features.compute_indices(make_bands())
    assert set(idx) == set(features.INDEX_NAMES)
    for name, arr in idx.items():
        assert arr.shape == (4, 4), name


def test_missing_band_raises_naming_the_band():
    bands = make_bands()
    del bands["B11"]
    with pytest.raises(KeyError) as e:
        features.compute_indices(bands)
    assert "B11" in str(e.value)


def test_ndvi_is_high_for_vegetation_and_low_for_water():
    veg = {b: np.array([[0.05]]) for b in features.BANDS}
    veg["B8"] = np.array([[0.33]])
    veg["B4"] = np.array([[0.04]])
    water = {b: np.array([[0.04]]) for b in features.BANDS}
    water["B8"] = np.array([[0.02]])
    water["B4"] = np.array([[0.035]])
    assert features.compute_indices(veg)["NDVI"][0, 0] > 0.5
    assert features.compute_indices(water)["NDVI"][0, 0] < 0.0


def test_ndwi_separates_water_from_vegetation():
    # NDWI is the index that stops dense shadow being called water, which NDVI
    # alone cannot do, so its sign must be right.
    water = {b: np.array([[0.03]]) for b in features.BANDS}
    water["B3"] = np.array([[0.055]])
    water["B8"] = np.array([[0.02]])
    veg = {b: np.array([[0.05]]) for b in features.BANDS}
    veg["B3"] = np.array([[0.055]])
    veg["B8"] = np.array([[0.33]])
    assert features.compute_indices(water)["NDWI"][0, 0] > 0
    assert features.compute_indices(veg)["NDWI"][0, 0] < 0


def test_nbr_is_low_for_burnt_ground():
    burnt = {b: np.array([[0.06]]) for b in features.BANDS}
    burnt["B8"] = np.array([[0.09]])
    burnt["B12"] = np.array([[0.26]])
    assert features.compute_indices(burnt)["NBR"][0, 0] < 0


def test_feature_names_order_is_stable_and_matches_the_stack():
    bands = make_bands()
    stack, names = features.build_feature_stack(bands, use_indices=True)
    assert names == features.feature_names(True)
    assert names[:len(features.BANDS)] == list(features.BANDS)
    assert stack.shape[1] == len(names)


def test_stack_without_indices_has_only_bands():
    stack, names = features.build_feature_stack(make_bands(), use_indices=False)
    assert names == list(features.BANDS)
    assert stack.shape[1] == len(features.BANDS)


def test_stack_rows_are_pixels_in_row_major_order():
    shape = (3, 5)
    bands = {b: np.arange(15, dtype="float64").reshape(shape) for b in features.BANDS}
    stack, _ = features.build_feature_stack(bands, use_indices=False)
    assert stack.shape[0] == 15
    # Column 0 is B2, which must come back as the original raster raveled.
    np.testing.assert_allclose(stack[:, 0], np.arange(15))


def test_stack_reshapes_back_to_the_grid():
    shape = (6, 7)
    bands = {b: np.random.default_rng(1).random(shape) for b in features.BANDS}
    stack, _ = features.build_feature_stack(bands, use_indices=True)
    assert stack[:, 0].reshape(shape).shape == shape


def test_stack_is_float32_to_halve_memory():
    stack, _ = features.build_feature_stack(make_bands())
    assert stack.dtype == np.float32


def test_stack_rejects_mismatched_band_shapes():
    bands = make_bands()
    bands["B12"] = np.zeros((5, 5))
    with pytest.raises(ValueError) as e:
        features.build_feature_stack(bands)
    assert "B12" in str(e.value)
