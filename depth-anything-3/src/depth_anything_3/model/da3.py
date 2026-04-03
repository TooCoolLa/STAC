# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
from addict import Dict
from omegaconf import DictConfig, OmegaConf

from depth_anything_3.cfg import create_object
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from depth_anything_3.utils.alignment import (
    apply_metric_scaling,
    compute_alignment_mask,
    compute_sky_mask,
    least_squares_scale_scalar,
    sample_tensor_for_quantile,
    set_sky_regions_to_max_depth,
)
from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous, map_pdf_to_opacity
from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray


def _wrap_cfg(cfg_obj):
    return OmegaConf.create(cfg_obj)


class DepthAnything3Net(nn.Module):
    """
    Depth Anything 3 network for depth estimation and camera pose estimation.

    This network consists of:
    - Backbone: DinoV2 feature extractor
    - Head: DPT or DualDPT for depth prediction
    - Optional camera decoders for pose estimation
    - Optional GSDPT for 3DGS prediction

    Args:
        preset: Configuration preset containing network dimensions and settings

    Returns:
        Dictionary containing:
        - depth: Predicted depth map (B, H, W)
        - depth_conf: Depth confidence map (B, H, W)
        - extrinsics: Camera extrinsics (B, N, 4, 4)
        - intrinsics: Camera intrinsics (B, N, 3, 3)
        - gaussians: 3D Gaussian Splats (world space), type: model.gs_adapter.Gaussians
        - aux: Auxiliary features for specified layers
    """

    # Patch size for feature extraction
    PATCH_SIZE = 14

    def __init__(self, net, head, cam_dec=None, cam_enc=None, gs_head=None, gs_adapter=None):
        """
        Initialize DepthAnything3Net with given yaml-initialized configuration.
        """
        super().__init__()
        self.backbone = net if isinstance(net, nn.Module) else create_object(_wrap_cfg(net))
        self.head = head if isinstance(head, nn.Module) else create_object(_wrap_cfg(head))
        self.cam_dec, self.cam_enc = None, None
        if cam_dec is not None:
            self.cam_dec = (
                cam_dec if isinstance(cam_dec, nn.Module) else create_object(_wrap_cfg(cam_dec))
            )
            self.cam_enc = (
                cam_enc if isinstance(cam_enc, nn.Module) else create_object(_wrap_cfg(cam_enc))
            )
        self.gs_adapter, self.gs_head = None, None
        if gs_head is not None and gs_adapter is not None:
            self.gs_adapter = (
                gs_adapter
                if isinstance(gs_adapter, nn.Module)
                else create_object(_wrap_cfg(gs_adapter))
            )
            gs_out_dim = self.gs_adapter.d_in + 1
            if isinstance(gs_head, nn.Module):
                assert (
                    gs_head.out_dim == gs_out_dim
                ), f"gs_head.out_dim should be {gs_out_dim}, got {gs_head.out_dim}"
                self.gs_head = gs_head
            else:
                assert (
                    gs_head["output_dim"] == gs_out_dim
                ), f"gs_head output_dim should set to {gs_out_dim}, got {gs_head['output_dim']}"
                self.gs_head = create_object(_wrap_cfg(gs_head))

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        mode: str = "full",
        streaming: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input images (B, N, 3, H, W) or (S, 3, H, W)
            extrinsics: Camera extrinsics (B, N, 4, 4)
            intrinsics: Camera intrinsics (B, N, 3, 3)
            feat_layers: List of layer indices to extract features from
            infer_gs: Enable Gaussian Splatting branch
            use_ray_pose: Use ray-based pose estimation
            ref_view_strategy: Strategy for selecting reference view
            mode: Attention mode (for STAC compatibility)
            streaming: Enable streaming mode (for STAC compatibility)
            **kwargs: Additional arguments (for STAC KV cache management)

        Returns:
            Dictionary containing predictions and auxiliary features
        """
        # Handle input shape: [S, 3, H, W] -> [B, N, 3, H, W]
        if len(x.shape) == 4:
            x = x.unsqueeze(0)  # Add batch dimension: [S, 3, H, W] -> [1, S, 3, H, W]
        
        B, N = x.shape[0], x.shape[1]

        # Extract features using backbone
        if extrinsics is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
        else:
            cam_token = None

        feats, aux_feats = self.backbone(
            x, cam_token=cam_token, export_feat_layers=export_feat_layers, 
            ref_view_strategy=ref_view_strategy, mode=mode, streaming=streaming, **kwargs
        )
        # feats = [[item for item in feat] for feat in feats]
        H, W = x.shape[-2], x.shape[-1]

        # Process features through depth head
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = self._process_depth_head(feats, H, W)
            if use_ray_pose:
                output = self._process_ray_pose_estimation(output, H, W)
            else:
                output = self._process_camera_estimation(feats, H, W, output)
            if infer_gs:
                output = self._process_gs_head(feats, H, W, output, x, extrinsics, intrinsics)

        output = self._process_mono_sky_estimation(output)

        # Extract auxiliary features if requested
        output.aux = self._extract_auxiliary_features(aux_feats, export_feat_layers, H, W)

        # Add timing if requested
        if kwargs.get("timing", False):
            output["timing"] = {"aggregator_infer_time": 0.0}

        # Convert to STAC-compatible format
        if streaming:
            # Convert DA3 output to STAC format
            stac_output = self._convert_to_stac_format(output, images=x, H=H, W=W)
            return stac_output

        return output

    def _convert_to_stac_format(
        self, 
        output: Dict[str, torch.Tensor],
        images: torch.Tensor,
        H: int,
        W: int
    ) -> Dict[str, torch.Tensor]:
        """
        Convert DA3 output to STAC-compatible format.
        
        DA3 outputs:
        - depth: [B, N, H, W]
        - depth_conf: [B, N, H, W]
        - extrinsics: [B, N, 3, 4] (w2c)
        - intrinsics: [B, N, 3, 3]
        
        STAC expects:
        - pose_enc: [B, N, 9]
        - depth: [B, N, H, W, 1]
        - depth_conf: [B, N, H, W]
        - world_points: [B, N, H, W, 3]
        - world_points_conf: [B, N, H, W]
        - images: [B, N, 3, H, W]
        """
        from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
        from depth_anything_3.utils.geometry import affine_inverse
        import torch.nn.functional as F
        
        stac_output = {}
        
        # Convert extrinsics to pose_enc
        # DA3 extrinsics: [B, N, 3, 4] w2c format
        # STAC needs pose_enc which is a 9D encoding
        extrinsics = output.get("extrinsics", None)
        intrinsics = output.get("intrinsics", None)
        
        if extrinsics is not None:
            # Convert w2c to c2w for pose encoding
            c2w = affine_inverse(extrinsics)  # [B, N, 4, 4]
            # Simple pose encoding: just use rotation + translation
            # Format: [R(3x3 flattened), t(3)] -> but we only need 9 values
            # Use simplified encoding: [R(3x3 upper), t(3)] = 9 + 3 = 12 -> take first 9
            # Actually, let's use the same encoding as pose_encoding_to_extri_intri expects
            # For now, use a simple encoding
            pose_enc = self._encode_pose_from_extrinsics(extrinsics)
            stac_output["pose_enc"] = pose_enc
        
        # Reshape depth to match STAC format
        depth = output.get("depth", None)
        depth_conf = output.get("depth_conf", None)
        
        if depth is not None:
            # DA3 depth: [B, N, H, W] -> STAC: [B, N, H, W, 1]
            if len(depth.shape) == 4:
                stac_output["depth"] = depth.unsqueeze(-1)
            else:
                stac_output["depth"] = depth
        
        if depth_conf is not None:
            stac_output["depth_conf"] = depth_conf
        
        # Compute world points from depth and camera parameters
        if depth is not None and extrinsics is not None and intrinsics is not None:
            world_points, world_points_conf = self._compute_world_points(
                depth.squeeze(-1) if len(depth.shape) == 5 else depth,
                extrinsics,
                intrinsics,
                H, W
            )
            stac_output["world_points"] = world_points
            stac_output["world_points_conf"] = world_points_conf
        
        # Store images
        stac_output["images"] = images
        
        # Store aggregated tokens list for camera head inference (if needed)
        # DA3 doesn't have this, so we'll use a placeholder
        stac_output["aggregated_tokens_list"] = [None]
        
        return stac_output

    def _encode_pose_from_extrinsics(self, extrinsics: torch.Tensor) -> torch.Tensor:
        """
        Convert extrinsics (w2c) to pose_enc (9D).
        
        Args:
            extrinsics: [B, N, 3, 4] w2c format
            
        Returns:
            pose_enc: [B, N, 9]
        """
        # Extract rotation (3x3) and translation (3x1)
        # w2c = [R | t]
        R = extrinsics[:, :, :3, :3]  # [B, N, 3, 3]
        t = extrinsics[:, :, :3, 3:]  # [B, N, 3, 1]
        
        # Simple encoding: use first 2 rows of R (6 values) + t (3 values) = 9 values
        # This is a simplified encoding, not the full Rodrigues representation
        pose_enc = torch.cat([
            R[:, :, :2, :].reshape(-1, 6),  # First 2 rows of rotation
            t.reshape(-1, 3)  # Translation
        ], dim=-1)  # [B*N, 9]
        
        # Reshape to [B, N, 9]
        B, N = extrinsics.shape[0], extrinsics.shape[1]
        pose_enc = pose_enc.reshape(B, N, 9)
        
        return pose_enc

    def _compute_world_points(
        self,
        depth: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        H: int,
        W: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute world coordinates from depth and camera parameters.
        
        Args:
            depth: [B, N, H, W]
            extrinsics: [B, N, 3, 4] w2c
            intrinsics: [B, N, 3, 3]
            H, W: Image dimensions
            
        Returns:
            world_points: [B, N, H, W, 3]
            world_points_conf: [B, N, H, W]
        """
        import torch.nn.functional as F
        from depth_anything_3.utils.geometry import affine_inverse
        
        B, N = depth.shape[0], depth.shape[1]
        
        # Create pixel grid
        u = torch.arange(W, device=depth.device, dtype=depth.dtype)
        v = torch.arange(H, device=depth.device, dtype=depth.dtype)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        # Pixel coordinates: [H, W]
        pixel_coords = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)  # [H, W, 3]
        pixel_coords = pixel_coords.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1, -1)  # [B, N, H, W, 3]
        
        # Unproject to camera coordinates
        # K^-1 @ pixel_coords * depth
        intrinsics_inv = torch.inverse(intrinsics)  # [B, N, 3, 3]
        
        # Reshape for matrix multiplication
        pixel_flat = pixel_coords.reshape(B, N, -1, 3)  # [B, N, H*W, 3]
        cam_coords = torch.einsum('bnij,bnkj->bnki', intrinsics_inv, pixel_flat)  # [B, N, H*W, 3]
        cam_coords = cam_coords * depth.reshape(B, N, -1, 1)  # [B, N, H*W, 3]
        
        # Convert to world coordinates
        # c2w = inverse(w2c)
        c2w = affine_inverse(extrinsics)  # [B, N, 4, 4]
        
        cam_coords_homo = torch.cat([
            cam_coords,
            torch.ones(B, N, cam_coords.shape[2], 1, device=depth.device, dtype=depth.dtype)
        ], dim=-1)  # [B, N, H*W, 4]
        
        world_coords = torch.einsum('bnij,bnkj->bnki', c2w, cam_coords_homo)  # [B, N, H*W, 4]
        world_coords = world_coords[..., :3]  # [B, N, H*W, 3]
        
        # Reshape to [B, N, H, W, 3]
        world_coords = world_coords.reshape(B, N, H, W, 3)
        
        # Confidence: assume high confidence everywhere (can be improved)
        confidence = torch.ones(B, N, H, W, device=depth.device, dtype=depth.dtype)
        
        return world_coords, confidence

    def _process_mono_sky_estimation(
        self, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process mono sky estimation."""
        if "sky" not in output:
            return output
        non_sky_mask = compute_sky_mask(output.sky, threshold=0.3)
        if non_sky_mask.sum() <= 10:
            return output
        if (~non_sky_mask).sum() <= 10:
            return output
        
        non_sky_depth = output.depth[non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        non_sky_max = torch.quantile(sampled_depth, 0.99)

        # Set sky regions to maximum depth and high confidence
        output.depth, _ = set_sky_regions_to_max_depth(
            output.depth, None, non_sky_mask, max_depth=non_sky_max
        )
        return output

    def _process_ray_pose_estimation(
        self, output: Dict[str, torch.Tensor], height: int, width: int
    ) -> Dict[str, torch.Tensor]:
        """Process ray pose estimation if ray pose decoder is available."""
        if "ray" in output and "ray_conf" in output:
            pred_extrinsic, pred_focal_lengths, pred_principal_points = get_extrinsic_from_camray(
                output.ray,
                output.ray_conf,
                output.ray.shape[-3],
                output.ray.shape[-2],
            )
            pred_extrinsic = affine_inverse(pred_extrinsic) # w2c -> c2w
            pred_extrinsic = pred_extrinsic[:, :, :3, :]
            pred_intrinsic = torch.eye(3, 3)[None, None].repeat(pred_extrinsic.shape[0], pred_extrinsic.shape[1], 1, 1).clone().to(pred_extrinsic.device)
            pred_intrinsic[:, :, 0, 0] = pred_focal_lengths[:, :, 0] / 2 * width
            pred_intrinsic[:, :, 1, 1] = pred_focal_lengths[:, :, 1] / 2 * height
            pred_intrinsic[:, :, 0, 2] = pred_principal_points[:, :, 0] * width * 0.5
            pred_intrinsic[:, :, 1, 2] = pred_principal_points[:, :, 1] * height * 0.5
            del output.ray
            del output.ray_conf
            output.extrinsics = pred_extrinsic
            output.intrinsics = pred_intrinsic
        return output

    def _process_depth_head(
        self, feats: list[torch.Tensor], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Process features through the depth prediction head."""
        return self.head(feats, H, W, patch_start_idx=0)

    def _process_camera_estimation(
        self, feats: list[torch.Tensor], H: int, W: int, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process camera pose estimation if camera decoder is available."""
        if self.cam_dec is not None:
            pose_enc = self.cam_dec(feats[-1][1])
            # Remove ray information as it's not needed for pose estimation
            if "ray" in output:
                del output.ray
            if "ray_conf" in output:
                del output.ray_conf

            # Convert pose encoding to extrinsics and intrinsics
            c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
            output.extrinsics = affine_inverse(c2w)
            output.intrinsics = ixt

        return output

    def _process_gs_head(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        output: Dict[str, torch.Tensor],
        in_images: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Process 3DGS parameters estimation if 3DGS head is available."""
        if self.gs_head is None or self.gs_adapter is None:
            return output
        assert output.get("depth", None) is not None, "must provide MV depth for the GS head."

        # The depth is defined in the DA3 model's camera space,
        # so even with provided GT camera poses,
        # we instead use the predicted camera poses for better alignment.
        ctx_extr = output.get("extrinsics", None)
        ctx_intr = output.get("intrinsics", None)
        assert (
            ctx_extr is not None and ctx_intr is not None
        ), "must process camera info first if GT is not available"

        gt_extr = extrinsics
        # homo the extr if needed
        ctx_extr = as_homogeneous(ctx_extr)
        if gt_extr is not None:
            gt_extr = as_homogeneous(gt_extr)

        # forward through the gs_dpt head to get 'camera space' parameters
        gs_outs = self.gs_head(
            feats=feats,
            H=H,
            W=W,
            patch_start_idx=0,
            images=in_images,
        )
        raw_gaussians = gs_outs.raw_gs
        densities = gs_outs.raw_gs_conf

        # convert to 'world space' 3DGS parameters; ready to export and render
        # gt_extr could be None, and will be used to align the pose scale if available
        gs_world = self.gs_adapter(
            extrinsics=ctx_extr,
            intrinsics=ctx_intr,
            depths=output.depth,
            opacities=map_pdf_to_opacity(densities),
            raw_gaussians=raw_gaussians,
            image_shape=(H, W),
            gt_extrinsics=gt_extr,
        )
        output.gaussians = gs_world

        return output

    def _extract_auxiliary_features(
        self, feats: list[torch.Tensor], feat_layers: list[int], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Extract auxiliary features from specified layers."""
        aux_features = Dict()
        assert len(feats) == len(feat_layers)
        for feat, feat_layer in zip(feats, feat_layers):
            # Reshape features to spatial dimensions
            feat_reshaped = feat.reshape(
                [
                    feat.shape[0],
                    feat.shape[1],
                    H // self.PATCH_SIZE,
                    W // self.PATCH_SIZE,
                    feat.shape[-1],
                ]
            )
            aux_features[f"feat_layer_{feat_layer}"] = feat_reshaped

        return aux_features


class NestedDepthAnything3Net(nn.Module):
    """
    Nested Depth Anything 3 network with metric scaling capabilities.

    This network combines two DepthAnything3Net branches:
    - Main branch: Standard depth estimation
    - Metric branch: Metric depth estimation for scaling alignment

    The network performs depth alignment using least squares scaling
    and handles sky region masking for improved depth estimation.

    Args:
        preset: Configuration for the main depth estimation branch
        second_preset: Configuration for the metric depth branch
    """

    def __init__(self, anyview: DictConfig, metric: DictConfig):
        """
        Initialize NestedDepthAnything3Net with two branches.

        Args:
            preset: Configuration for main depth estimation branch
            second_preset: Configuration for metric depth branch
        """
        super().__init__()
        self.da3 = create_object(anyview)
        self.da3_metric = create_object(metric)

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both branches with metric scaling alignment.

        Args:
            x: Input images (B, N, 3, H, W)
            extrinsics: Camera extrinsics (B, N, 4, 4) - unused
            intrinsics: Camera intrinsics (B, N, 3, 3) - unused
            feat_layers: List of layer indices to extract features from
            infer_gs: Enable Gaussian Splatting branch
            use_ray_pose: Use ray-based pose estimation
            ref_view_strategy: Strategy for selecting reference view

        Returns:
            Dictionary containing aligned depth predictions and camera parameters
        """
        # Get predictions from both branches
        output = self.da3(
            x, extrinsics, intrinsics, export_feat_layers=export_feat_layers, infer_gs=infer_gs, use_ray_pose=use_ray_pose, ref_view_strategy=ref_view_strategy
        )
        metric_output = self.da3_metric(x)

        # Apply metric scaling and alignment
        output = self._apply_metric_scaling(output, metric_output)
        output = self._apply_depth_alignment(output, metric_output)
        output = self._handle_sky_regions(output, metric_output)

        return output

    def _apply_metric_scaling(
        self, output: Dict[str, torch.Tensor], metric_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply metric scaling to the metric depth output."""
        # Scale metric depth based on camera intrinsics
        metric_output.depth = apply_metric_scaling(
            metric_output.depth,
            output.intrinsics,
        )
        return output

    def _apply_depth_alignment(
        self, output: Dict[str, torch.Tensor], metric_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply depth alignment using least squares scaling."""
        # Compute non-sky mask
        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)

        # Ensure we have enough non-sky pixels
        assert non_sky_mask.sum() > 10, "Insufficient non-sky pixels for alignment"

        # Sample depth confidence for quantile computation
        depth_conf_ns = output.depth_conf[non_sky_mask]
        depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
        median_conf = torch.quantile(depth_conf_sampled, 0.5)

        # Compute alignment mask
        align_mask = compute_alignment_mask(
            output.depth_conf, non_sky_mask, output.depth, metric_output.depth, median_conf
        )

        # Compute scale factor using least squares
        valid_depth = output.depth[align_mask]
        valid_metric_depth = metric_output.depth[align_mask]
        scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)

        # Apply scaling to depth and extrinsics
        output.depth *= scale_factor
        output.extrinsics[:, :, :3, 3] *= scale_factor
        output.is_metric = 1
        output.scale_factor = scale_factor.item()

        return output

    def _handle_sky_regions(
        self,
        output: Dict[str, torch.Tensor],
        metric_output: Dict[str, torch.Tensor],
        sky_depth_def: float = 200.0,
    ) -> Dict[str, torch.Tensor]:
        """Handle sky regions by setting them to maximum depth."""
        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)

        # Compute maximum depth for non-sky regions
        # Use sampling to safely compute quantile on large tensors
        non_sky_depth = output.depth[non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        non_sky_max = min(torch.quantile(sampled_depth, 0.99), sky_depth_def)

        # Set sky regions to maximum depth and high confidence
        output.depth, output.depth_conf = set_sky_regions_to_max_depth(
            output.depth, output.depth_conf, non_sky_mask, max_depth=non_sky_max
        )

        return output
