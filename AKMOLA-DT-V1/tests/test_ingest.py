"""
AKMOLA-DT-V1 — Unit Tests: Data Ingestion
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from config import CHIP_SIZE, STRIDE, PIXEL_SIZE, DTYPE
from pipeline.ingest import RasterProcessor, generate_synthetic_dem


class TestSyntheticDEM:
    """Test synthetic DEM generation."""

    def test_shape(self):
        elev, transform = generate_synthetic_dem(height=128, width=128)
        assert elev.shape == (128, 128)

    def test_dtype(self):
        elev, _ = generate_synthetic_dem(height=64, width=64)
        assert elev.dtype == DTYPE

    def test_has_variability(self):
        elev, _ = generate_synthetic_dem(height=128, width=128)
        assert elev.std() > 0.1, "DEM should have elevation variability"

    def test_has_depressions(self):
        """Steppe DEM should contain micro-depressions."""
        elev, _ = generate_synthetic_dem(height=256, width=256, seed=42)
        # Check that not all gradient is monotonic (depressions exist)
        # The Laplacian should be positive in depression centers
        from scipy.ndimage import laplace
        lap = laplace(elev)
        assert np.sum(lap > 0) > 100, "Should contain micro-depressions"

    def test_reproducible_with_seed(self):
        e1, _ = generate_synthetic_dem(height=64, width=64, seed=123)
        e2, _ = generate_synthetic_dem(height=64, width=64, seed=123)
        np.testing.assert_array_equal(e1, e2)


class TestRasterProcessorTiling:
    """Test the tile_generator method."""

    def setup_method(self):
        self.processor = RasterProcessor(chip_size=64, stride=32)
        self.elev, self.transform = generate_synthetic_dem(
            height=128, width=128, pixel_size=PIXEL_SIZE
        )

    def test_tiles_generated(self):
        tiles = list(self.processor.tile_generator(self.elev, self.transform,
                                                    chip_size=64, stride=32))
        assert len(tiles) > 0

    def test_tile_shape(self):
        tiles = list(self.processor.tile_generator(self.elev, self.transform,
                                                    chip_size=64, stride=32))
        for tile in tiles:
            assert tile["data"].shape == (64, 64)

    def test_tile_has_metadata(self):
        tiles = list(self.processor.tile_generator(self.elev, self.transform,
                                                    chip_size=64, stride=32))
        for tile in tiles:
            assert "tile_id" in tile
            assert "bounds" in tile
            assert "affine_transform" in tile
            assert len(tile["bounds"]) == 4
            assert len(tile["affine_transform"]) == 6

    def test_tile_dtype(self):
        tiles = list(self.processor.tile_generator(self.elev, self.transform,
                                                    chip_size=64, stride=32))
        for tile in tiles:
            assert tile["data"].dtype == DTYPE

    def test_small_array_padding(self):
        """Array smaller than chip_size should still produce tiles via padding."""
        small = np.random.rand(32, 32).astype(DTYPE)
        from pipeline.ingest import Affine
        tr = Affine(30.0, 0, 0, 0, -30.0, 0)
        tiles = list(self.processor.tile_generator(small, tr, chip_size=64, stride=32))
        assert len(tiles) > 0
        for tile in tiles:
            assert tile["data"].shape == (64, 64)


class TestRasterProcessorConfig:
    """Test configuration and initialization."""

    def test_default_config(self):
        proc = RasterProcessor()
        assert proc.pixel_size == PIXEL_SIZE
        assert proc.chip_size == CHIP_SIZE
        assert proc.stride == STRIDE

    def test_custom_config(self):
        proc = RasterProcessor(pixel_size=10.0, chip_size=128, stride=64)
        assert proc.pixel_size == 10.0
        assert proc.chip_size == 128
        assert proc.stride == 64
