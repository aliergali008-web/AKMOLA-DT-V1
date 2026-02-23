"""
AKMOLA-DT-V1 — Unit Tests: Feature Engineering
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from config import TERRAIN_CHANNELS, DTYPE, CHANNEL_NAMES
from pipeline.features import GeoFeatureCalculator
from pipeline.ingest import generate_synthetic_dem


class TestSlopeComputation:
    """Test slope vector calculation."""

    def setup_method(self):
        self.calc = GeoFeatureCalculator(pixel_size=30.0)

    def test_flat_surface(self):
        """Flat surface should have near-zero slope."""
        flat = np.full((64, 64), 300.0, dtype=DTYPE)
        slope_mag, _, _ = self.calc.compute_slope(flat)
        assert np.allclose(slope_mag, 0.0, atol=1e-6)

    def test_tilted_plane(self):
        """Tilted plane should have constant non-zero slope."""
        y = np.arange(64).reshape(-1, 1).astype(DTYPE)
        tilted = np.broadcast_to(y * 10.0, (64, 64)).copy()  # 10m rise per pixel
        slope_mag, _, _ = self.calc.compute_slope(tilted, method="zevenbergen")
        # Interior should be relatively uniform
        interior = slope_mag[5:-5, 5:-5]
        assert interior.mean() > 0.01

    def test_output_shapes(self):
        elev = np.random.rand(64, 64).astype(DTYPE) * 100
        slope_mag, aspect_cos, aspect_sin = self.calc.compute_slope(elev)
        assert slope_mag.shape == (64, 64)
        assert aspect_cos.shape == (64, 64)
        assert aspect_sin.shape == (64, 64)

    def test_both_methods(self):
        elev = np.random.rand(64, 64).astype(DTYPE) * 100
        s1, _, _ = self.calc.compute_slope(elev, method="sobel")
        s2, _, _ = self.calc.compute_slope(elev, method="zevenbergen")
        assert s1.shape == s2.shape
        # Both should detect slope, but may differ in magnitude
        assert s1.mean() > 0
        assert s2.mean() > 0


class TestTRI:
    """Test Terrain Ruggedness Index."""

    def setup_method(self):
        self.calc = GeoFeatureCalculator()

    def test_flat_surface_low_tri(self):
        flat = np.full((64, 64), 300.0, dtype=DTYPE)
        tri = self.calc.compute_tri(flat)
        assert np.allclose(tri, 0.0, atol=1e-6)

    def test_rough_surface_high_tri(self):
        rough = np.random.rand(64, 64).astype(DTYPE) * 100
        tri = self.calc.compute_tri(rough)
        assert tri.mean() > 1.0

    def test_output_shape(self):
        elev = np.random.rand(128, 128).astype(DTYPE) * 100
        tri = self.calc.compute_tri(elev)
        assert tri.shape == (128, 128)


class TestCurvature:
    """Test Laplacian curvature."""

    def setup_method(self):
        self.calc = GeoFeatureCalculator()

    def test_flat_surface(self):
        flat = np.full((64, 64), 300.0, dtype=DTYPE)
        curv = self.calc.compute_curvature(flat)
        assert np.allclose(curv, 0.0, atol=1e-6)

    def test_concave_surface(self):
        """Bowl shape should have positive curvature in center."""
        y, x = np.ogrid[:64, :64]
        bowl = ((x - 32.0) ** 2 + (y - 32.0) ** 2).astype(DTYPE)
        curv = self.calc.compute_curvature(bowl)
        # Center should be positive (concave)
        assert curv[32, 32] > 0

    def test_output_shape(self):
        elev = np.random.rand(128, 128).astype(DTYPE)
        curv = self.calc.compute_curvature(elev)
        assert curv.shape == (128, 128)


class TestFlowAccumulation:
    """Test flow accumulation computation."""

    def setup_method(self):
        self.calc = GeoFeatureCalculator()

    def test_output_shape(self):
        """Small DEM for quick test."""
        elev = np.random.rand(32, 32).astype(DTYPE) * 100 + 200
        log_flow = self.calc.compute_flow_accumulation(elev)
        assert log_flow.shape == (32, 32)

    def test_positive_values(self):
        """log(1 + accum) should always be >= 0."""
        elev = np.random.rand(32, 32).astype(DTYPE) * 50 + 200
        log_flow = self.calc.compute_flow_accumulation(elev)
        assert np.all(log_flow >= 0)

    def test_valley_accumulates_more(self):
        """A V-shaped valley should accumulate more flow than ridges."""
        y, x = np.ogrid[:32, :32]
        valley = np.abs(x - 16.0).astype(DTYPE) * 10 + 200
        log_flow = self.calc.compute_flow_accumulation(valley)
        # Center column (valley bottom) should have higher accumulation
        center_accum = log_flow[:, 16].mean()
        edge_accum = log_flow[:, 0].mean()
        assert center_accum >= edge_accum


class TestTensorBuilding:
    """Test the full tensor stacking pipeline."""

    def setup_method(self):
        self.calc = GeoFeatureCalculator()

    def test_tensor_shape_small(self):
        elev = np.random.rand(32, 32).astype(DTYPE) * 100 + 200
        tensor = self.calc.build_tensor(elev, normalize=True)
        assert tensor.shape == (TERRAIN_CHANNELS, 32, 32)

    def test_tensor_channels_count(self):
        elev = np.random.rand(32, 32).astype(DTYPE) * 100
        tensor = self.calc.build_tensor(elev, normalize=False)
        assert tensor.shape[0] == 6
        assert tensor.shape[0] == len(CHANNEL_NAMES)

    def test_normalized_range(self):
        elev = np.random.rand(32, 32).astype(DTYPE) * 100 + 200
        tensor = self.calc.build_tensor(elev, normalize=True)
        for c in range(tensor.shape[0]):
            assert tensor[c].min() >= -0.01  # small tolerance
            assert tensor[c].max() <= 1.01

    def test_dtype(self):
        elev = np.random.rand(32, 32).astype(DTYPE) * 100
        tensor = self.calc.build_tensor(elev)
        assert tensor.dtype == DTYPE

    def test_synthetic_dem_pipeline(self):
        """Full integration: synthetic DEM → tensor."""
        elev, _ = generate_synthetic_dem(height=64, width=64)
        tensor = self.calc.build_tensor(elev)
        assert tensor.shape == (6, 64, 64)
