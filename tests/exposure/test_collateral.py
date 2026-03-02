"""
tests/exposure/test_collateral.py
==================================
100% coverage for src/xva_engine/exposure/collateral.py

Tests calculate_collateral() for every CSA mode:
- None / csa=None  →  zeros
- mode="none"      →  zeros
- mode="perfect_vm"→  copy of V_ns
- mode="threshold" →  threshold clip (all three sub-branches:
                       V_ns > threshold, V_ns < -threshold, in between)
"""
import numpy as np
import pytest

from xva_engine.io.schemas import CSA
from xva_engine.exposure.collateral import calculate_collateral


RNG = np.random.default_rng(0)
V = RNG.standard_normal((10, 500)) * 100_000  # (steps, paths)


class TestCollateralNone:
    def test_csa_none_object_returns_zeros(self):
        C = calculate_collateral(V, None)
        np.testing.assert_array_equal(C, 0.0)

    def test_csa_mode_none_returns_zeros(self):
        csa = CSA(csa_id="C1", mode="none")
        C = calculate_collateral(V, csa)
        np.testing.assert_array_equal(C, 0.0)

    def test_output_shape_matches_input(self):
        C = calculate_collateral(V, None)
        assert C.shape == V.shape


class TestCollateralPerfectVM:
    def test_equals_v_ns(self):
        csa = CSA(csa_id="C2", mode="perfect_vm")
        C = calculate_collateral(V, csa)
        np.testing.assert_array_equal(C, V)

    def test_is_copy_not_view(self):
        csa = CSA(csa_id="C2", mode="perfect_vm")
        C = calculate_collateral(V, csa)
        assert C is not V

    def test_shape_preserved(self):
        csa = CSA(csa_id="C2", mode="perfect_vm")
        C = calculate_collateral(V, csa)
        assert C.shape == V.shape


class TestCollateralThreshold:
    THRESHOLD = 50_000.0

    @pytest.fixture
    def csa(self):
        return CSA(csa_id="C3", mode="threshold", threshold=self.THRESHOLD)

    def test_above_threshold_clips(self, csa):
        V_test = np.array([[100_000.0]])  # > threshold
        C = calculate_collateral(V_test, csa)
        assert abs(C[0, 0] - (100_000.0 - self.THRESHOLD)) < 1e-9

    def test_below_neg_threshold_clips(self, csa):
        V_test = np.array([[-100_000.0]])  # < -threshold
        C = calculate_collateral(V_test, csa)
        assert abs(C[0, 0] - (-100_000.0 + self.THRESHOLD)) < 1e-9

    def test_within_band_is_zero(self, csa):
        V_test = np.array([[10_000.0], [-10_000.0]])  # within band
        C = calculate_collateral(V_test, csa)
        np.testing.assert_array_equal(C, 0.0)

    def test_shape_preserved(self, csa):
        C = calculate_collateral(V, csa)
        assert C.shape == V.shape

    def test_three_regions_exercised(self, csa):
        # Construct a V with values in all three zones
        V_test = np.array([[200_000.0, -200_000.0, 10_000.0]])
        C = calculate_collateral(V_test, csa)
        assert C[0, 0] > 0        # above threshold
        assert C[0, 1] < 0        # below -threshold
        assert C[0, 2] == 0.0     # within band

    def test_zero_threshold(self):
        csa = CSA(csa_id="C4", mode="threshold", threshold=0.0)
        V_test = np.array([[50.0, -50.0, 0.0]])
        C = calculate_collateral(V_test, csa)
        assert C[0, 0] > 0
        assert C[0, 1] < 0
        assert C[0, 2] == 0.0
