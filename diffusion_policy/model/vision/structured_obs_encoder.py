"""
StructuredObsEncoder — branched encoder for the GROVE 4-channel input.

Splits the 4-channel image into a 3-channel seg branch and a 1-channel depth
branch, runs each through its own small CNN, applies SpatialSoftmax per branch
to extract keypoints, then concatenates with low-dim obs (eef_pose, centroids).

Implements section B of the structured-input refactor:
  #5  random-init small CNN backbone (4 conv layers + GroupNorm), no ImageNet.
  #6  branched architecture: separate trunks for mask and depth modalities.
  #7  per-channel-group SpatialSoftmax (keypoints, not avg-pool).
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


class SmallCNN(nn.Module):
    """4-layer CNN with GroupNorm; preserves enough spatial extent for SpatialSoftmax."""

    def __init__(self, in_channels: int, base_channels: int = 32,
                 out_channels: int = 64, n_groups: int = 8):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(min(n_groups, c1), c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(n_groups, c2), c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(n_groups, c2), c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(n_groups, c3), c3),
            nn.ReLU(inplace=True),
        )
        self.out_channels = c3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialSoftmax(nn.Module):
    """Per-channel expected-coordinate pooling. (B,C,H,W) -> (B, 2*C)."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device
        # softmax over spatial dims
        flat = x.view(B, C, H * W) / self.temperature
        attn = F.softmax(flat, dim=-1)              # (B,C,H*W)
        # coordinate grid in [-1, 1]
        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_x = grid_x.reshape(1, 1, H * W)
        grid_y = grid_y.reshape(1, 1, H * W)
        exp_x = (attn * grid_x).sum(-1)             # (B,C)
        exp_y = (attn * grid_y).sum(-1)
        return torch.cat([exp_x, exp_y], dim=-1)    # (B, 2*C)


class StructuredObsEncoder(ModuleAttrMixin):
    """
    Drop-in obs_encoder for the seg4ch workspace.

    Expects shape_meta.obs to contain:
      cam_high          : type=rgb,    shape=[4, H, W]   (channels: bg, veg, tom, depth)
      <any low_dim>     : type=low_dim                   (e.g. eef_pose, target_centroid, obstacle_centroid)

    The rgb tensor is split:
      seg branch   : channels 0..2  (3 one-hot masks)
      depth branch : channel  3     (normalized depth)

    Each branch -> SmallCNN -> SpatialSoftmax -> keypoint vector.
    Concatenate keypoints + low-dim obs.
    """

    OUTPUT_KEYS = ("seg_keypoints", "depth_keypoints")

    def __init__(self,
                 shape_meta: dict,
                 seg_channels: Tuple[int, ...] = (0, 1, 2),
                 depth_channels: Tuple[int, ...] = (3,),
                 seg_base_channels: int = 32,
                 seg_out_channels: int = 32,
                 depth_base_channels: int = 16,
                 depth_out_channels: int = 16,
                 softmax_temperature: float = 1.0,
                 n_groups: int = 8):
        super().__init__()
        obs_meta = shape_meta["obs"]

        rgb_keys = [k for k, v in obs_meta.items() if v.get("type", "low_dim") == "rgb"]
        if len(rgb_keys) != 1:
            raise ValueError(f"StructuredObsEncoder expects exactly 1 rgb key, got {rgb_keys}")
        self.rgb_key = rgb_keys[0]
        c, h, w = tuple(obs_meta[self.rgb_key]["shape"])
        if c != len(seg_channels) + len(depth_channels):
            raise ValueError(
                f"channel split mismatch: rgb shape c={c} vs "
                f"seg={seg_channels} + depth={depth_channels}")
        self._spatial_hw = (h, w)
        self.seg_channels = tuple(seg_channels)
        self.depth_channels = tuple(depth_channels)

        self.seg_backbone = SmallCNN(
            in_channels=len(seg_channels),
            base_channels=seg_base_channels,
            out_channels=seg_out_channels,
            n_groups=n_groups,
        )
        self.depth_backbone = SmallCNN(
            in_channels=len(depth_channels),
            base_channels=depth_base_channels,
            out_channels=depth_out_channels,
            n_groups=n_groups,
        )
        self.seg_pool = SpatialSoftmax(temperature=softmax_temperature)
        self.depth_pool = SpatialSoftmax(temperature=softmax_temperature)

        self.lowdim_keys: List[str] = [
            k for k, v in obs_meta.items()
            if v.get("type", "low_dim") == "low_dim"
        ]
        self.lowdim_shapes = {k: tuple(obs_meta[k]["shape"]) for k in self.lowdim_keys}

        self.shape_meta = shape_meta

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-step encoding (B has the time axis folded in upstream)."""
        cam = obs_dict[self.rgb_key]
        if cam.dim() != 4:
            raise ValueError(f"{self.rgb_key} must be (B, C, H, W), got {tuple(cam.shape)}")

        seg = cam[:, list(self.seg_channels)]
        depth = cam[:, list(self.depth_channels)]

        seg_feat = self.seg_backbone(seg)
        depth_feat = self.depth_backbone(depth)

        seg_kp = self.seg_pool(seg_feat)        # (B, 2*seg_out_channels)
        depth_kp = self.depth_pool(depth_feat)  # (B, 2*depth_out_channels)

        parts = [seg_kp, depth_kp]
        for k in self.lowdim_keys:
            v = obs_dict[k]
            assert v.dim() == 2, f"lowdim '{k}' must be (B, D), got {tuple(v.shape)}"
            parts.append(v)
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def output_shape(self):
        """Shape per obs step (no batch). Used by the policy to size conditioning."""
        h, w = self._spatial_hw
        c = len(self.seg_channels) + len(self.depth_channels)
        dummy = {self.rgb_key: torch.zeros(1, c, h, w, dtype=self.dtype, device=self.device)}
        for k, shape in self.lowdim_shapes.items():
            dummy[k] = torch.zeros((1,) + tuple(shape), dtype=self.dtype, device=self.device)
        return tuple(self.forward(dummy).shape[1:])


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    shape_meta = {
        "obs": {
            "cam_high":          {"shape": [4, 240, 320], "type": "rgb"},
            "eef_pose":          {"shape": [7],           "type": "low_dim"},
            "target_centroid":   {"shape": [3],           "type": "low_dim"},
            "obstacle_centroid": {"shape": [3],           "type": "low_dim"},
        },
        "action": {"shape": [7]},
    }
    enc = StructuredObsEncoder(shape_meta)
    enc.eval()

    B = 2
    obs = {
        "cam_high":          torch.randn(B, 4, 240, 320),
        "eef_pose":          torch.randn(B, 7),
        "target_centroid":   torch.randn(B, 3),
        "obstacle_centroid": torch.randn(B, 3),
    }
    with torch.no_grad():
        out = enc(obs)
    expected = 2 * 32 + 2 * 16 + 7 + 3 + 3  # 64 + 32 + 13 = 109
    assert out.shape == (B, expected), f"got {tuple(out.shape)}, expected (B={B}, {expected})"
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"StructuredObsEncoder OK. Output (B, {expected}). Params: {n_params:,}")
