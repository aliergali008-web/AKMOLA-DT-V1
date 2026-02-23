"""
AKMOLA-DT-V1 — Unit Tests: EL-FNO Model Architecture
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from model.core import LagrangianGating, SpectralBlock2d, EL_FNO_Model, model_summary


class TestLagrangianGating:
    """Test the terrain-guided deformation module."""

    def test_output_shape(self):
        module = LagrangianGating(terrain_channels=6, hidden_dim=32)
        x = torch.randn(2, 6, 64, 64)
        out = module(x)
        assert out.shape == (2, 2, 64, 64)

    def test_output_channels_are_offsets(self):
        """Should output exactly 2 channels (delta_x, delta_y)."""
        module = LagrangianGating(terrain_channels=6)
        x = torch.randn(1, 6, 32, 32)
        out = module(x)
        assert out.shape[1] == 2

    def test_batch_independence(self):
        """Each batch element should be processed independently."""
        module = LagrangianGating(terrain_channels=6)
        x = torch.randn(4, 6, 32, 32)
        out = module(x)
        assert out.shape[0] == 4


class TestSpectralBlock2d:
    """Test the FNO spectral block."""

    def test_output_shape(self):
        block = SpectralBlock2d(in_channels=64, out_channels=64, modes1=8, modes2=8)
        x = torch.randn(2, 64, 64, 64)
        out = block(x)
        assert out.shape == (2, 64, 64, 64)

    def test_different_channels(self):
        block = SpectralBlock2d(in_channels=32, out_channels=64, modes1=8, modes2=8)
        x = torch.randn(1, 32, 64, 64)
        out = block(x)
        assert out.shape == (1, 64, 64, 64)

    def test_preserves_spatial_dims(self):
        """FFT → filter → iFFT should preserve spatial dimensions."""
        block = SpectralBlock2d(64, 64, modes1=16, modes2=16)
        for size in [32, 64, 128]:
            x = torch.randn(1, 64, size, size)
            out = block(x)
            assert out.shape[-2:] == (size, size)


class TestEL_FNO_Model:
    """Test the complete EL-FNO model."""

    def setup_method(self):
        self.model = EL_FNO_Model(
            state_ch=2, terrain_ch=6, width=32, modes=8, n_layers=2
        )

    def test_forward_pass(self):
        state = torch.randn(2, 2, 64, 64)
        terrain = torch.randn(2, 6, 64, 64)
        out = self.model(state, terrain)
        assert out.shape == (2, 1, 64, 64)

    def test_output_shape_256(self):
        """ТЗ requirement: [Batch, 1, 256, 256]."""
        model = EL_FNO_Model(width=16, modes=8, n_layers=2)
        state = torch.randn(1, 2, 256, 256)
        terrain = torch.randn(1, 6, 256, 256)
        with torch.no_grad():
            out = model(state, terrain)
        assert out.shape == (1, 1, 256, 256)

    def test_single_sample(self):
        state = torch.randn(1, 2, 64, 64)
        terrain = torch.randn(1, 6, 64, 64)
        out = self.model(state, terrain)
        assert out.shape == (1, 1, 64, 64)

    def test_gradient_flow(self):
        """Verify gradients can flow through the model."""
        state = torch.randn(1, 2, 32, 32, requires_grad=True)
        terrain = torch.randn(1, 6, 32, 32, requires_grad=True)
        out = self.model(state, terrain)
        loss = out.sum()
        loss.backward()
        assert state.grad is not None
        assert terrain.grad is not None

    def test_deterministic_eval(self):
        """Model should produce same output in eval mode with same input."""
        self.model.eval()
        state = torch.randn(1, 2, 32, 32)
        terrain = torch.randn(1, 6, 32, 32)
        with torch.no_grad():
            out1 = self.model(state, terrain)
            out2 = self.model(state, terrain)
        torch.testing.assert_close(out1, out2)


class TestModelSummary:
    """Test model summary utility."""

    def test_summary_keys(self):
        model = EL_FNO_Model(width=16, modes=4, n_layers=2)
        info = model_summary(model)
        assert "total_params" in info
        assert "trainable_params" in info
        assert "total_params_M" in info

    def test_params_positive(self):
        model = EL_FNO_Model(width=16, modes=4, n_layers=2)
        info = model_summary(model)
        assert info["total_params"] > 0
        assert info["trainable_params"] > 0
