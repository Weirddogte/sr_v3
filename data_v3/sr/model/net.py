from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double-conv block with optional stride and dropout."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SupportResistanceHeatmapNetV3(nn.Module):
    """
    Input:  [B, 3, H, W]  RGB chart image
    Output: [B, 5, H]     five-channel heatmap logits (sigmoid applied in loss)

    Channels:
        0 – support
        1 – resistance
        2 – active zone
        3 – historical zone
        4 – proximity
    """

    def __init__(self, dropout: float = 0.05) -> None:
        super().__init__()

        # --- Encoder ---
        self.block1 = ConvBlock(3,   48,  stride=2, dropout=0.0)     # → H/2,  W/2
        self.block2 = ConvBlock(48,  96,  stride=2, dropout=0.0)     # → H/4,  W/4
        self.block3 = ConvBlock(96,  192, stride=2, dropout=0.0)     # → H/8,  W/8
        self.block4 = ConvBlock(192, 288, stride=2, dropout=0.0)     # → H/16, W/16

        # --- Skip branch from block1 (2-D → 1-D) ---
        # AdaptiveAvgPool2d((None, 1)) collapses width dimension
        self.skip_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.skip_conv  = nn.Conv1d(48, 16, kernel_size=1)
        self.skip_bn    = nn.BatchNorm1d(16)
        self.skip_act   = nn.SiLU(inplace=True)

        # --- Width attention pooling ---
        self.width_attn_conv = nn.Conv2d(288, 1, kernel_size=1)

        # --- Row head (1-D convolutions over H/16) ---
        self.row_conv1  = nn.Conv1d(288, 288, kernel_size=5, padding=2)
        self.row_bn1    = nn.BatchNorm1d(288)
        self.row_act1   = nn.SiLU(inplace=True)
        self.row_conv2  = nn.Conv1d(288, 144, kernel_size=5, padding=2)
        self.row_bn2    = nn.BatchNorm1d(144)
        self.row_act2   = nn.SiLU(inplace=True)
        self.row_drop   = nn.Dropout(dropout)
        self.row_out    = nn.Conv1d(144, 5, kernel_size=1)

        # --- 8× upsample (H/16 → H/2) ---
        # output_size = (L_in - 1)*stride - 2*padding + kernel_size
        #             = (H/16 - 1)*8    - 2*4         + 16
        #             = H/2  - 8 + 8 = H/2  ✓
        self.upsample_8x = nn.ConvTranspose1d(5, 5, kernel_size=16, stride=8, padding=4)

        # --- Skip blend (concatenated 5 + 16 = 21 channels → 5) ---
        self.blend_conv = nn.Conv1d(21, 5, kernel_size=3, padding=1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_skip(self, x1: torch.Tensor) -> torch.Tensor:
        """Build skip tensor from block1 output.

        Args:
            x1: [B, 48, H/2, W/2]

        Returns:
            skip: [B, 16, H/2]
        """
        # Collapse width → [B, 48, H/2, 1]
        pooled = self.skip_pool(x1)
        # Squeeze spatial width → [B, 48, H/2]
        squeezed = pooled.squeeze(3)
        # 1-D conv + BN + activation
        out = self.skip_conv(squeezed)   # [B, 16, H/2]
        out = self.skip_bn(out)
        out = self.skip_act(out)
        return out

    def _width_attn_pool(self, x4: torch.Tensor) -> torch.Tensor:
        """Attention-weighted pooling over the width dimension.

        Args:
            x4: [B, 288, H/16, W/16]

        Returns:
            pooled: [B, 288, H/16]
        """
        attn_logits  = self.width_attn_conv(x4)           # [B, 1, H/16, W/16]
        attn_weights = F.softmax(attn_logits, dim=3)       # softmax over width
        pooled       = (x4 * attn_weights).sum(dim=3)      # [B, 288, H/16]
        return pooled

    def _row_head(self, pooled: torch.Tensor) -> torch.Tensor:
        """1-D convolutional head over rows.

        Args:
            pooled: [B, 288, H/16]

        Returns:
            logits: [B, 5, H/16]
        """
        x = self.row_act1(self.row_bn1(self.row_conv1(pooled)))
        x = self.row_act2(self.row_bn2(self.row_conv2(x)))
        x = self.row_drop(x)
        x = self.row_out(x)
        return x

    def _upsample_with_skip(
        self,
        logits_small: torch.Tensor,
        skip: torch.Tensor,
        H: int,
    ) -> torch.Tensor:
        """8× upsample, skip fusion, final interpolation to H.

        Args:
            logits_small: [B, 5, H/16]
            skip:         [B, 16, H/2]
            H:            original image height

        Returns:
            out: [B, 5, H]
        """
        H2 = H // 2

        # Step 1: 8× upsample → [B, 5, H/2]
        x = self.upsample_8x(logits_small)          # [B, 5, ~H/2]

        # Trim or pad to exactly H/2 in case of off-by-one
        if x.shape[2] > H2:
            x = x[:, :, :H2]
        elif x.shape[2] < H2:
            x = F.pad(x, (0, H2 - x.shape[2]))

        # Step 2: concatenate with skip
        x = torch.cat([x, skip], dim=1)             # [B, 21, H/2]

        # Step 3: blend
        x = self.blend_conv(x)                      # [B, 5, H/2]

        # Step 4: interpolate to full H
        x = F.interpolate(x, size=H, mode='linear', align_corners=False)  # [B, 5, H]

        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            logits: [B, 5, H]  (raw logits, no sigmoid)
        """
        B, C, H, W = x.shape  # noqa: F841

        # Encoder
        x1 = self.block1(x)          # [B, 48,  H/2,  W/2 ]
        x2 = self.block2(x1)         # [B, 96,  H/4,  W/4 ]
        x3 = self.block3(x2)         # [B, 192, H/8,  W/8 ]
        x4 = self.block4(x3)         # [B, 288, H/16, W/16]

        # Skip connection from block1
        skip = self._make_skip(x1)              # [B, 16, H/2]

        # Width-attention pooling over block4
        pooled = self._width_attn_pool(x4)      # [B, 288, H/16]

        # Row-wise 1-D convolutions → logits at H/16 resolution
        logits_small = self._row_head(pooled)   # [B, 5, H/16]

        # Upsample, fuse skip, interpolate to H
        out = self._upsample_with_skip(logits_small, skip, H)  # [B, 5, H]

        return out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
