"""
AKMOLA-DT-V1 — EL-FNO Model Architecture
==========================================
Eulerian-Lagrangian Adaptive Fourier Neural Operator.

Architecture:
    Lagrangian Guide (terrain → grid deformation offsets)
    + Eulerian Solver (FNO spectral blocks → PDE solution)

Ref: Mukhitov et al. (2025), Li et al. (2021), Chen et al. (SGNet)
Ref ТЗ §4 (model.core)
"""

import torch
import torch.nn as nn
import torch.fft


class LagrangianGating(nn.Module):
    """
    Spatial Information Guided module (Ref: SGNet/DKN).
    DEM features act as a 'Guide' to predict deformation offsets.

    In steppe context, the guide learns to emphasize offsets near
    micro-depressions and water accumulation pathways.
    """

    def __init__(self, terrain_channels: int = 6, hidden_dim: int = 32):
        super().__init__()
        # Maps terrain complexity → deformation offsets (delta_x, delta_y)
        self.guide_net = nn.Sequential(
            nn.Conv2d(terrain_channels, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1),
        )

    def forward(self, terrain: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        terrain : torch.Tensor [B, terrain_channels, H, W]

        Returns
        -------
        offsets : torch.Tensor [B, 2, H, W]
            Predicted spatial deformation offsets (delta_x, delta_y).
        """
        offsets = self.guide_net(terrain)
        return offsets


class SpectralBlock2d(nn.Module):
    """
    Standard Eulerian FNO Block (Ref: Li et al., 2021).
    Solves global PDEs in frequency domain (Fourier Space).

    Pipeline: x → FFT → spectral_filter(low-pass) → iFFT → output
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 16,
        modes2: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1.0 / (in_channels * out_channels)

        # Complex weights for the lowest frequency modes
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2,
                dtype=torch.cfloat,
            )
        )

    def compl_mul2d(
        self, input: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Complex multiplication:
        (batch, in, x, y) × (in, out, x, y) → (batch, out, x, y)
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor [B, C_in, H, W]

        Returns
        -------
        out : torch.Tensor [B, C_out, H, W]
        """
        batchsize = x.shape[0]

        # 1. FFT (real → complex)
        x_ft = torch.fft.rfft2(x)

        # 2. Spectral Filter (Low-pass): keep only low-frequency modes
        out_ft = torch.zeros(
            batchsize, self.out_channels,
            x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        # Low frequencies (top-left corner of spectrum)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )

        # Low frequencies (bottom-left corner, for negative freq)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # 3. Inverse FFT (complex → real)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class EL_FNO_Model(nn.Module):
    """
    The Master Class: Eulerian-Lagrangian Fourier Neural Operator.

    Architecture:
        A. Lagrangian Adapter (terrain guide → grid offsets)
        B. Feature Lifter (project to hidden width)
        C. Eulerian Solver (4× FNO spectral blocks + skip connections)
        D. Decoder (project to water depth prediction)

    Input:
        state   [B, state_ch, H, W]   — hydrodynamic state (water depth, velocity)
        terrain [B, terrain_ch, H, W]  — terrain features (6-channel tensor)

    Output:
        prediction [B, 1, H, W] — predicted water depth
    """

    def __init__(
        self,
        state_ch: int = 2,
        terrain_ch: int = 6,
        width: int = 64,
        modes: int = 16,
        n_layers: int = 4,
    ):
        super().__init__()
        self.state_ch = state_ch
        self.terrain_ch = terrain_ch
        self.width = width

        # A. Lagrangian Adapter (The Guide)
        self.grid_guide = LagrangianGating(terrain_channels=terrain_ch)

        # B. Feature Lifter: state + terrain + offsets → width
        #    Channels: state_ch + terrain_ch + 2 (offsets)
        self.fc0 = nn.Linear(state_ch + terrain_ch + 2, width)

        # C. Eulerian Solver (FNO Layers)
        self.fno_layers = nn.ModuleList([
            SpectralBlock2d(width, width, modes, modes)
            for _ in range(n_layers)
        ])

        # D. Skip Connections (Residual, 1×1 conv)
        self.ws = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])

        # E. Decoder: width → 128 → 1 (water depth)
        self.fc_out = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        terrain: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor [B, state_ch, H, W]
        terrain : torch.Tensor [B, terrain_ch, H, W]

        Returns
        -------
        prediction : torch.Tensor [B, 1, H, W]
        """
        # 1. Compute Lagrangian Offsets
        offsets = self.grid_guide(terrain)  # [B, 2, H, W]

        # 2. Concatenate state + terrain + offsets
        # In full implementation, offsets would drive grid_sample().
        # For MVP: concatenate so FNO can learn from "warped" geometry context.
        x = torch.cat([state, terrain, offsets], dim=1)  # [B, state+terrain+2, H, W]

        # 3. Lift to hidden dimension
        x = x.permute(0, 2, 3, 1)     # [B, H, W, C_in]
        x = self.fc0(x)                # [B, H, W, width]
        x = x.permute(0, 3, 1, 2)     # [B, width, H, W]

        # 4. Eulerian Solver: 4× (SpectralBlock + skip + GELU)
        for i in range(len(self.fno_layers)):
            x = self.fno_layers[i](x) + self.ws[i](x)
            x = torch.nn.functional.gelu(x)

        # 5. Decode to water depth
        x = x.permute(0, 2, 3, 1)     # [B, H, W, width]
        x = self.fc_out(x)            # [B, H, W, 1]
        x = x.permute(0, 3, 1, 2)     # [B, 1, H, W]

        return x


# ─── Model Summary Utility ──────────────────────────────────────

def model_summary(model: nn.Module) -> dict:
    """Quick summary of model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "total_params_M": f"{total / 1e6:.2f}M",
    }


# ─── CLI Entry Point ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AKMOLA-DT-V1 EL-FNO Model Test")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--size", type=int, default=256, help="Spatial size (H=W)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"], help="Device")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print("=" * 60)
    print("  AKMOLA-DT-V1 | EL-FNO Model Architecture Test")
    print("=" * 60)

    model = EL_FNO_Model().to(device)
    info = model_summary(model)
    print(f"\nModel parameters: {info['total_params_M']}")
    print(f"  Total:     {info['total_params']:,}")
    print(f"  Trainable: {info['trainable_params']:,}")

    # Test forward pass
    B, H, W = args.batch, args.size, args.size
    state = torch.randn(B, 2, H, W, device=device)
    terrain = torch.randn(B, 6, H, W, device=device)

    print(f"\nInput shapes:")
    print(f"  state:   {list(state.shape)}")
    print(f"  terrain: {list(terrain.shape)}")

    with torch.no_grad():
        output = model(state, terrain)

    print(f"\nOutput shape: {list(output.shape)}")
    print(f"Expected:     [{B}, 1, {H}, {W}]")

    assert output.shape == (B, 1, H, W), \
        f"Shape mismatch! Got {output.shape}, expected ({B}, 1, {H}, {W})"

    print("\n✓ Forward pass successful!")
    print("=" * 60)
