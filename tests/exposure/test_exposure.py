"""
tests/exposure/test_exposure.py
================================
100% coverage for src/xva_engine/exposure/exposure.py

Tests calculate_exposures() for output shapes, mathematical
correctness of E/NE definitions, perfect-VM special case,
and no NaN/Inf guarantees.
"""
import numpy as np
import pytest

from xva_engine.exposure.exposure import calculate_exposures


STEPS, PATHS = 15, 5_000
RNG = np.random.default_rng(99)


class TestCalculateExposures:
    def test_output_keys_present(self):
        V = RNG.standard_normal((STEPS, PATHS)) * 1e5
        C = np.zeros((STEPS, PATHS))
        result = calculate_exposures(V, C)
        assert "E_paths" in result
        assert "NE_paths" in result
        assert "EE" in result
        assert "ENE" in result

    def test_shapes(self):
        V = np.random.standard_normal((STEPS, PATHS)) * 1e4
        C = np.zeros((STEPS, PATHS))
        r = calculate_exposures(V, C)
        assert r["E_paths"].shape == (STEPS, PATHS)
        assert r["NE_paths"].shape == (STEPS, PATHS)
        assert r["EE"].shape == (STEPS,)
        assert r["ENE"].shape == (STEPS,)

    def test_E_paths_non_negative(self):
        V = RNG.standard_normal((STEPS, PATHS)) * 1e5
        C = np.zeros((STEPS, PATHS))
        r = calculate_exposures(V, C)
        assert np.all(r["E_paths"] >= 0.0)

    def test_NE_paths_non_negative(self):
        V = RNG.standard_normal((STEPS, PATHS)) * 1e5
        C = np.zeros((STEPS, PATHS))
        r = calculate_exposures(V, C)
        assert np.all(r["NE_paths"] >= 0.0)

    def test_EE_non_negative(self):
        V = np.abs(RNG.standard_normal((STEPS, PATHS))) * 1e5
        C = np.zeros((STEPS, PATHS))
        r = calculate_exposures(V, C)
        assert np.all(r["EE"] >= 0.0)

    def test_ENE_non_negative(self):
        V = -np.abs(RNG.standard_normal((STEPS, PATHS))) * 1e5
        C = np.zeros((STEPS, PATHS))
        r = calculate_exposures(V, C)
        assert np.all(r["ENE"] >= 0.0)

    def test_perfect_vm_EE_is_zero(self):
        V = RNG.standard_normal((STEPS, PATHS)) * 1e5
        C = V.copy()  # perfect VM
        r = calculate_exposures(V, C)
        np.testing.assert_allclose(r["EE"], 0.0, atol=1e-8)

    def test_perfect_vm_ENE_is_zero(self):
        V = RNG.standard_normal((STEPS, PATHS)) * 1e5
        C = V.copy()
        r = calculate_exposures(V, C)
        np.testing.assert_allclose(r["ENE"], 0.0, atol=1e-8)

    def test_no_collateral_ENE_zero_for_positive_V(self):
        V = np.abs(RNG.standard_normal((STEPS, PATHS))) * 1e4
        C = np.zeros((STEPS, PATHS))
        r = calculate_exposures(V, C)
        np.testing.assert_allclose(r["ENE"], 0.0, atol=1e-8)

    def test_E_definition_pointwise(self):
        V = np.array([[10.0, -5.0]])
        C = np.zeros((1, 2))
        r = calculate_exposures(V, C)
        np.testing.assert_allclose(r["E_paths"], [[10.0, 0.0]])
        np.testing.assert_allclose(r["NE_paths"], [[0.0, 5.0]])

    def test_no_nan_or_inf(self):
        V = RNG.standard_normal((STEPS, PATHS)) * 1e6
        C = RNG.standard_normal((STEPS, PATHS)) * 5e5
        r = calculate_exposures(V, C)
        assert np.all(np.isfinite(r["EE"]))
        assert np.all(np.isfinite(r["ENE"]))
