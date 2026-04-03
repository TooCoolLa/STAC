import torch
from causalvggt.models.vggt import CausalVGGT
from stac.kv_manager import KVManager
from stac.stac_voxel import STACVoxelKV
from causalvggt.utils.geometry import unproject_depth_map_to_point_map
from causalvggt.utils.pose_enc import pose_encoding_to_extri_intri

import os
import logging
from copy import deepcopy

from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console, Group
from rich.logging import RichHandler

_console = Console(stderr=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=_console, show_path=False, rich_tracebacks=True)],
)
logger = logging.getLogger("StreamSession")

VERBOSE = os.environ.get("VERBOSE", "0").strip().lower() in ("1", "true", "yes")


def _make_progress(description: str, total: int) -> tuple:
    """Create a (Progress, task_id) pair with consistent styling."""
    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )
    task_id = progress.add_task(description, total=total)
    return progress, task_id


def _stats_table(rows: list[tuple[str, str]]) -> Table:
    """Build a compact stats Table from (label, value) pairs."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold cyan", width=10)
    table.add_column()
    for label, value in rows:
        table.add_row(label, value)
    return table


def _gpu_mem_mb():
    """Return (allocated_MB, reserved_MB) for the current CUDA device."""
    if torch.cuda.is_available():
        return (torch.cuda.memory_allocated() / 1024**2,
                torch.cuda.memory_reserved() / 1024**2)
    return 0.0, 0.0


class StreamSession:
    """
    A causal streaming inference session with KV cache management for CausalVGGT or DA3.
    """

    def __init__(
        self,
        model,
        cam_cache_update: bool = False,
        device: torch.device = torch.device("cuda"),
        model_type: str = "causalvggt",
    ):
        self.model_type = model_type
        self.model = model.to(device)
        self.device = device

        if model_type == "da3":
            # DA3 uses backbone as the aggregator
            # NestedDepthAnything3Net has da3.da3.backbone.pretrained
            if hasattr(model.model, 'da3'):
                self.aggregator = model.model.da3.backbone.pretrained
            elif hasattr(model.model, 'backbone'):
                self.aggregator = model.model.backbone.pretrained
            else:
                raise AttributeError(f"Cannot find backbone in DA3 model: {type(model.model).__name__}")
        else:
            # CausalVGGT uses dedicated aggregator
            self.aggregator = model.aggregator

        self.aggregator_kv_cache_depth = self.aggregator.depth if hasattr(self.aggregator, 'depth') else 24
        self.camera_head_kv_cache_depth = model.camera_head.trunk_depth if hasattr(model, 'camera_head') and model.camera_head is not None else 0
        self.camera_head_iterations = 4 if hasattr(model, 'camera_head') and model.camera_head is not None else 0
        self.cam_cache_update = cam_cache_update
        self.pose_tokens_list = []
        # DA3 pose accumulation: track last global w2c transformation
        self._da3_last_c2w = None  # [1, 1, 4, 4] c2w matrix for pose accumulation
        # Prediction keys to track, where the element of prediction shape like [B, S, ...]
        self.predictions_keys = ["pose_enc", "world_points", "world_points_conf", "depth", "depth_conf", "images"]

        self._processed_frames = 0
        self._global_scale_factor = None  # Pre-computed global metric scale factor
        self.init()

    def init(self):
        self._processed_frames = 0
        self._global_scale_factor = None
        self.predictions = {k: [] for k in self.predictions_keys}
        self.pose_tokens_list = []
        self.benchmark_metrics = {}
        self.stats = {}
        self._da3_last_c2w = None

    def clear(self):
        self._clear_predictions()
        if hasattr(self.model, 'aggregator') and self.model.aggregator is not None:
            self.model.aggregator.clear_kv_mgr()
        elif hasattr(self.aggregator, 'clear_kv_mgr'):
            self.aggregator.clear_kv_mgr()
        torch.cuda.empty_cache()
        self.pose_tokens_list = []
        self._processed_frames = 0
        self.benchmark_metrics = {}
        self.stats = {}

    # ======== Prediction management methods ========
    def _clear_predictions(self):
        for k in self.predictions:
            for i in reversed(range(len(self.predictions[k]))):
                tensor = self.predictions[k][i]
                if isinstance(tensor, torch.Tensor):
                    del tensor
                elif isinstance(tensor, list):
                    for j in reversed(range(len(tensor))):
                        if isinstance(tensor[j], torch.Tensor):
                            del tensor[j]
                    del tensor
        self.predictions = {k: [] for k in self.predictions_keys}

    def _update_predictions(self, predictions: dict, device: str = 'cpu'):
        for k in predictions:
            if k in self.predictions:
                if predictions[k] is None:
                    continue
                B,S = predictions[k].shape[0], predictions[k].shape[1]
                for i in range(B):
                    for j in range(S):
                        self.predictions[k].append(predictions[k][i:i+1, j:j+1].to(device=device))

    def get_all_predictions(self, device='cpu'):
        # return self.predictions
        all_predictions = dict()
        for key in self.predictions_keys:
            if key in self.predictions:
                if isinstance(self.predictions[key], torch.Tensor):
                    all_predictions[key] = self.predictions[key].to(device=device)
                    continue
                if len(self.predictions[key]) == 0:
                    continue
                if self._processed_frames != len(self.predictions[key]):
                    raise ValueError(f"Processed frames {self._processed_frames} != stored predictions {len(self.predictions[key])} for key {key}")
                if isinstance(self.predictions[key][0], torch.Tensor):
                    all_predictions[key] = torch.cat(self.predictions[key], dim=1).to(device=device)
                elif isinstance(self.predictions[key][0], list):
                    prediction_list = []
                    for layer_idx in range(len(self.predictions[key][0])):
                        layer_predictions = []
                        for frame_idx in range(len(self.predictions[key])):
                            layer_predictions.append(self.predictions[key][frame_idx][layer_idx].to(device=device))
                        prediction_list.append(torch.cat(layer_predictions, dim=1))
                    all_predictions[key] = prediction_list # list of tensors
                else:
                    raise ValueError(f"Unsupported prediction type for key {key}: {type(self.predictions[key][0])}")
        # Include auxiliary keys not in predictions_keys
        for key in ("is_metric", "scale_factor"):
            if key in self.predictions and len(self.predictions[key]) > 0:
                val = self.predictions[key][0]
                if isinstance(val, torch.Tensor):
                    all_predictions[key] = val.item() if val.numel() == 1 else val.to(device=device)
        return all_predictions

    def get_last_prediction(self):
        last_predictions = dict()

        for k in self.predictions_keys:
            if k in self.predictions:
                last_predictions[k] = self.predictions[k][-1]
        return last_predictions

    def pop_first_prediction(self):
        first_predictions = dict()
        for k in self.predictions_keys:
            if k in self.predictions and len(self.predictions[k]) > 0:
                first_predictions[k] = self.predictions[k].pop(0)
        return first_predictions

    def pushback_prediction(self, predictions, device='cpu'):
        self._update_predictions(predictions, device=device)
    
    def _update_benchmark(self, metrics: dict):
        if not self.benchmark_metrics:
            self.benchmark_metrics = metrics
        else:
            for k in metrics:
                if k in self.benchmark_metrics:
                    self.benchmark_metrics[k] += metrics[k]
                else:
                    self.benchmark_metrics[k] = metrics[k]
            
    def get_benchmark(self):
        return self.benchmark_metrics
    
    def get_stats(self):
        return self.stats

    def apply_global_metric_alignment(self, sample_stride=8):
        """
        Apply post-hoc global metric alignment to accumulated predictions.

        For DA3 nested models in streaming mode, the per-chunk metric alignment
        is skipped. This method applies a pre-computed global scale factor
        (passed via kwargs) to all depth/world_points/extrinsics.

        If no pre-computed scale is available, falls back to computing one
        from a subset of frames (for backward compatibility).

        Args:
            sample_stride: Sample every N frames for metric branch inference (fallback only)
        """
        if self.model_type != "da3":
            return

        # Check if model is NestedDepthAnything3Net (has da3_metric branch)
        model_net = self.model.model if hasattr(self.model, 'model') else self.model
        if not hasattr(model_net, 'da3_metric'):
            return

        # Check for pre-computed global scale factor (computed in run_model before pipeline)
        global_scale = getattr(self, '_global_scale_factor', None)

        if global_scale is not None and global_scale > 0:
            logger.info("Applying pre-computed global metric scale factor: %.4f", global_scale)
            self._apply_scale_to_predictions(global_scale)
            device = self.device
            self.predictions["is_metric"] = [torch.tensor(1, device=device)]
            self.predictions["scale_factor"] = [torch.tensor(global_scale, device=device)]
            return

        # Fallback: compute scale from sampled frames (original behavior)
        depth_list = self.predictions.get("depth", [])
        images_list = self.predictions.get("images", [])
        if not depth_list or not images_list:
            return

        num_frames = len(depth_list)
        sample_indices = list(range(0, num_frames, sample_stride))
        if not sample_indices:
            sample_indices = [0]

        device = self.device
        sampled_depths = []
        sampled_images = []
        for idx in sample_indices:
            sampled_depths.append(depth_list[idx].to(device))
            sampled_images.append(images_list[idx].to(device))

        depth_sampled = torch.cat(sampled_depths, dim=1)
        images_sampled = torch.cat(sampled_images, dim=1)
        del sampled_depths, sampled_images
        torch.cuda.empty_cache()

        if depth_sampled is None or images_sampled is None:
            return

        logger.info("Computing metric alignment from %d/%d frames (fallback)...",
                     len(sample_indices), num_frames)

        import torch.nn.functional as F
        from depth_anything_3.utils.alignment import (
            compute_sky_mask, compute_alignment_mask,
            least_squares_scale_scalar, sample_tensor_for_quantile,
        )

        metric_depths = []
        rel_depths = []
        with torch.no_grad():
            for idx in range(len(sample_indices)):
                frame = images_sampled[:, idx:idx+1]
                with torch.autocast(device_type=frame.device.type, dtype=torch.bfloat16):
                    metric_out = model_net.da3_metric(frame)
                    metric_depths.append(metric_out.depth.float())
                    rel_out = model_net.da3(frame, streaming=False)
                    rel_depths.append(rel_out.depth.float())

        metric_depth_all = torch.cat(metric_depths, dim=1)
        rel_depth_all = torch.cat(rel_depths, dim=1)

        if rel_depth_all.shape[-2:] != metric_depth_all.shape[-2:]:
            rel_depth_all = F.interpolate(
                rel_depth_all, size=metric_depth_all.shape[-2:],
                mode='bilinear', align_corners=False
            )

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                sky_out = model_net.da3_metric(images_sampled)
        non_sky_mask = compute_sky_mask(sky_out.sky, threshold=0.3)

        depth_conf_list = self.predictions.get("depth_conf", [])
        if depth_conf_list and len(depth_conf_list) > 0:
            conf_sampled = torch.cat([depth_conf_list[i].to(device) for i in sample_indices], dim=1)
            if len(conf_sampled.shape) == 5:
                conf_sampled = conf_sampled.squeeze(-1)
            conf_sampled_flat = conf_sampled[non_sky_mask]
            conf_sampled_flat = sample_tensor_for_quantile(conf_sampled_flat, max_samples=100000)
            median_conf = torch.quantile(conf_sampled_flat, 0.5)
        else:
            median_conf = 0.5

        align_mask = compute_alignment_mask(
            depth_conf=conf_sampled if depth_conf_list else torch.ones_like(rel_depth_all),
            non_sky_mask=non_sky_mask,
            depth=rel_depth_all,
            metric_depth=metric_depth_all,
            median_conf=median_conf,
        )

        valid_rel = rel_depth_all[align_mask]
        valid_metric = metric_depth_all[align_mask]

        del metric_depths, rel_depths, metric_depth_all, rel_depth_all
        del depth_sampled, images_sampled
        torch.cuda.empty_cache()

        if valid_rel.numel() < 10:
            logger.info("Global metric alignment skipped (insufficient valid pixels).")
            self.predictions["is_metric"] = [torch.tensor(0, device=device)]
            return

        scale_factor = least_squares_scale_scalar(valid_metric, valid_rel).item()
        logger.info("Fallback global metric alignment applied. scale_factor=%.4f", scale_factor)
        self._apply_scale_to_predictions(scale_factor)
        self.predictions["is_metric"] = [torch.tensor(1, device=device)]
        self.predictions["scale_factor"] = [torch.tensor(scale_factor, device=device)]

    def _apply_scale_to_predictions(self, scale_factor):
        """Apply a scale factor to all depth/world_points/extrinsics predictions."""
        device = self.device
        for key in self.predictions:
            if key not in ("depth", "world_points", "extrinsics"):
                continue
            pred_list = self.predictions[key]
            if not pred_list:
                continue
            for i in range(len(pred_list)):
                if isinstance(pred_list[i], torch.Tensor):
                    pred_list[i] = (pred_list[i].to(device) * scale_factor).cpu()


    # ======== Inference methods ========

    def camera_head_inference(
            self,
            agg_token_lists,
    ):
        time_start = torch.cuda.Event(enable_timing=True)
        time_end = torch.cuda.Event(enable_timing=True)
        time_start.record()
        pose_tokens = agg_token_lists[-1][:, :, 0].detach()
        self.pose_tokens_list.append(pose_tokens)
        pose_token_cache = torch.cat(self.pose_tokens_list, dim=1)
        with torch.amp.autocast("cuda", enabled=False):
            pose_enc_list, _ = self.model.camera_head.inference(                                            
                    aggregated_tokens_list=None,
                    pose_token_cache=pose_token_cache,
                    mode="full",
                    kv_cache_list=None,
                )
            self.predictions["pose_enc"] = pose_enc_list[-1]
            outputs = {"pose_enc": pose_enc_list[-1]}
        time_end.record()
        torch.cuda.synchronize()
        elapsed = time_start.elapsed_time(time_end)
        self._update_benchmark({"camera_head_time": elapsed})
        return outputs
    
    def get_pointmap(self, outputs, conf_threshold=1.0, special_tokens_size=0, 
                     pose_enc = None, images=None,
                     prediction_mode="pointmap"):
        """ 
        Process the point cloud from model outputs and update KV cache positions.
        """
        # Update KV cache positions
        if (prediction_mode == "pointmap"):
            pts3d = outputs.get("world_points", None) # [B,S,H,W,3]
            pts3d_conf = outputs.get("world_points_conf", None) # [B,S,H,W]
        else:
            depth_map = outputs.get("depth", None) # [B,S,H,W,1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                    pose_enc, images.shape[-2:]
                )
            pts3d_conf = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
            depths_conf = outputs.get("depth_conf", None) # [B,S,H,W]
            
        assert pts3d is not None and pts3d_conf is not None, "World points and confidence must be provided outputs."
        B, S, H, W, C = pts3d.shape
        assert B==1, "Batch size must be 1."
        pts3d_conf = pts3d_conf.unsqueeze(-1) # [B,S,H,W,1]
        pts3d = pts3d.permute(0, 1, 4, 2, 3).view(-1, C, H, W) # [S, 3, H, W]
        pts3d_conf = pts3d_conf.permute(0, 1, 4, 2, 3).view(-1, 1, H, W) # [S, 1, H, W]
        # downsample to patch level
        if self.model_type == "da3":
            ds_patch = 14  # DA3 ViT patch size
        else:
            ds_patch = self.model.point_head.patch_size
        ds_size = (max(1, H // ds_patch), max(1, W // ds_patch))
        pts3d = torch.nn.functional.interpolate(
            pts3d, size=ds_size, mode='bilinear', align_corners=False
        )
        pts3d_conf = torch.nn.functional.interpolate(
            pts3d_conf, size=ds_size, mode='bilinear', align_corners=False
        ) # [S, 1, H', W']
        # reshape to [S, H'*W', 3]
        H2, W2 = pts3d.shape[-2:]
        pts3d = pts3d.permute(0, 2, 3, 1).reshape(S, -1, C) # [S, H'*W', 3]
        pts_special = pts3d.new_zeros((S, special_tokens_size, C))
        pts3d = torch.cat((pts_special, pts3d), dim=1) # [S, special+H'*W', 3]
        
        pts3d_conf = pts3d_conf.permute(0, 2, 3, 1).reshape(S,-1) # [S, H'*W']
        valid_mask = pts3d_conf > conf_threshold # [S, H'*W']
        valid_special = valid_mask.new_zeros((S, special_tokens_size)).bool()
        valid_mask = torch.cat((valid_special, valid_mask), dim=1) # [S, special+H'*W']
        return pts3d, valid_mask

    def _accumulate_da3_poses(self, pose_enc, outputs, frame_idx, chunk_size):
        """Accumulate DA3 poses globally across chunks.
        
        DA3 outputs relative extrinsics per chunk (each chunk is in its own coordinate system).
        We need to convert to global coordinates by composing with the previous chunk's last pose.
        
        Args:
            pose_enc: [B, N, 9] pose encoding
            outputs: model output dict containing extrinsics [B, N, 3, 4] w2c
            frame_idx: current global frame index
            chunk_size: number of frames in this chunk
            
        Returns:
            Updated pose_enc and outputs with globally accumulated poses
        """
        import torch
        
        extrinsics_w2c = outputs.get("extrinsics", None)
        if extrinsics_w2c is None:
            extrinsics_w2c = getattr(outputs, "extrinsics", None)
        if extrinsics_w2c is None:
            return pose_enc, outputs
        
        B, N = extrinsics_w2c.shape[0], extrinsics_w2c.shape[1]
        
        # Convert w2c [B, N, 3, 4] to homogeneous [B, N, 4, 4]
        ones = torch.zeros(B, N, 1, 4, device=extrinsics_w2c.device, dtype=extrinsics_w2c.dtype)
        ones[:, :, 0, 3] = 1.0
        w2c_homo = torch.cat([extrinsics_w2c, ones], dim=2)  # [B, N, 4, 4]
        
        # Invert to get c2w
        c2w_homo = self._invert_pose(w2c_homo)  # [B, N, 4, 4]
        
        if self._da3_last_c2w is None:
            # First chunk: poses are already global (relative to first frame)
            self._da3_last_c2w = c2w_homo[:, -1:, :, :].detach()  # Save last c2w
        else:
            # Subsequent chunks: compose with previous chunk's last pose
            # Current chunk's first frame c2w (relative)
            first_c2w_rel = c2w_homo[:, :1, :, :]  # [B, 1, 4, 4]
            # Previous chunk's last frame c2w (global)
            last_c2w_global = self._da3_last_c2w  # [B, 1, 4, 4]
            
            # Compute transformation: T_global = T_prev_last @ T_rel @ T_rel_first_inv
            # Simpler: compute relative transform from chunk's first frame to each frame,
            # then apply to global first frame
            
            # Relative transform from frame 0 to frame i in current chunk:
            # T_rel_0_to_i = c2w_rel[i] @ inv(c2w_rel[0])
            first_c2w_inv = self._invert_pose(first_c2w_rel)  # [B, 1, 4, 4]
            
            # Global first frame c2w: use last_c2w_global as reference
            # Actually, we need the global pose of the first frame in this chunk
            # which should be the continuation from the previous chunk's last frame
            # But DA3 doesn't give us that directly.
            
            # Approach: compute relative motion within chunk, then accumulate
            # For frame i in chunk: T_global[i] = T_global[first-1] @ T_rel[0->i]
            # where T_rel[0->i] = c2w[i] @ inv(c2w[0])
            
            # Get global pose of frame just before this chunk
            prev_global_c2w = self._da3_last_c2w  # [B, 1, 4, 4]
            
            # Compute relative transforms within chunk (relative to first frame)
            rel_transforms = torch.matmul(c2w_homo, first_c2w_inv.expand(-1, N, -1, -1))  # [B, N, 4, 4]
            
            # Apply to previous global pose
            global_c2w = torch.matmul(prev_global_c2w.expand(-1, N, -1, -1), rel_transforms)  # [B, N, 4, 4]
            
            # Convert back to w2c
            global_w2c = self._invert_pose(global_c2w)  # [B, N, 4, 4]
            global_w2c_3x4 = global_w2c[:, :, :3, :]  # [B, N, 3, 4]
            
            # Update outputs
            outputs["extrinsics"] = global_w2c_3x4
            
            # Re-encode pose
            pose_enc = self._encode_pose_from_extrinsics(global_w2c_3x4)
            outputs["pose_enc"] = pose_enc  # Also update outputs so pushback saves correct pose
            
            # Update last global c2w
            self._da3_last_c2w = global_c2w[:, -1:, :, :].detach()
        
        return pose_enc, outputs
    
    def _invert_pose(self, w2c):
        """Invert w2c [B, N, 4, 4] to c2w [B, N, 4, 4]."""
        import torch
        R = w2c[:, :, :3, :3]  # [B, N, 3, 3]
        t = w2c[:, :, :3, 3:]  # [B, N, 3, 1]
        R_inv = R.transpose(-2, -1)  # [B, N, 3, 3]
        t_inv = -torch.matmul(R_inv, t)  # [B, N, 3, 1]
        c2w = torch.eye(4, device=w2c.device, dtype=w2c.dtype).unsqueeze(0).unsqueeze(0).expand(
            w2c.shape[0], w2c.shape[1], -1, -1
        ).clone()
        c2w[:, :, :3, :3] = R_inv
        c2w[:, :, :3, 3:] = t_inv
        return c2w
    
    def _encode_pose_from_extrinsics(self, extrinsics):
        """Encode extrinsics [B, N, 3, 4] w2c to pose_enc [B, N, 9]."""
        import torch
        B, N = extrinsics.shape[0], extrinsics.shape[1]
        R = extrinsics[:, :, :3, :3]  # [B, N, 3, 3]
        t = extrinsics[:, :, :3, 3:]  # [B, N, 3, 1]
        pose_enc = torch.cat([
            R[:, :, :2, :].reshape(B, N, 6),
            t.reshape(B, N, 3)
        ], dim=-1)
        return pose_enc

    def pipeline(self, 
                 images: torch.Tensor, 
                 mode="causal", 
                 **kwargs
                    ) -> dict:
        self.clear()
        # [S, 3, H, W]
        num_frames = images.shape[0]
        device = kwargs.get("device", self.device)
        dtype = kwargs.get("dtype", torch.float16)

        # Accept pre-computed global scale factor from run_model
        self._global_scale_factor = kwargs.get("global_scale_factor", None)
        if self._global_scale_factor is not None:
            logger.info("Using pre-computed global scale factor: %.4f", self._global_scale_factor)

        logger.info("Streaming Pipeline Warming up the model...")
        for _ in range(1):
            if self.model_type == "da3":
                self.model(
                    image=images[0:1].to(device=device, dtype=dtype),
                    mode="full",
                    streaming=True,
                )
            else:
                self.model(
                    images=images[0:1].to(device=device, dtype=dtype),
                    mode="full",
                    camera_head_kv_cache_list=None,
                    streaming=True,
                    is_anchor_exist=True,
                )  # warmup

        if mode in ["window_kv","causal"]:
            # Use H2O attention to maintain a heavy-hitter + recent KV cache for the aggregator.
            window_size = kwargs.get("window_size", 0)
            chunk_size = kwargs.get("chunk_size", 1)
            transfer_chunk_size = kwargs.get("transfer_chunk_size", max(chunk_size, 16))
            if mode == "causal":
                window_size = num_frames
            if window_size < 0:
                logger.warning("Switching to causal attention mode.")
                window_size = num_frames  # effectively causal
            if chunk_size < 1:
                raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
            if transfer_chunk_size < chunk_size:
                logger.warning(
                    "transfer_chunk_size (%d) < chunk_size (%d), clamping to chunk_size.",
                    transfer_chunk_size, chunk_size,
                )
                transfer_chunk_size = chunk_size
            if window_size > 0 and chunk_size > window_size:
                logger.warning(
                    f"chunk_size ({chunk_size}) > window_size ({window_size}): "
                    "prune may trigger every chunk."
                )
            kv_kwargs = deepcopy(kwargs)
            kv_kwargs.update({
                "register_layers": None,
                "window_size": window_size,
                "chunk_size": chunk_size,
            })
            kv_kwargs = self.register_kv_mgr(mode, images, KVManager, **kv_kwargs)
            if self.model_type == "da3":
                self.aggregator.set_camhead(self.cam_cache_update)
            else:
                self.model.set_camhead(self.cam_cache_update)
            debug_timing = kwargs.get("timing", True)
            logger.info("Window mode: chunk=%d, transfer_chunk=%d, window=%d",
                        chunk_size, transfer_chunk_size, window_size)
            progress, task = _make_progress("Window Mode", num_frames)
            with Live(progress, console=_console, refresh_per_second=8) as live:
                for transfer_start in range(0, num_frames, transfer_chunk_size):
                    transfer_end = min(transfer_start + transfer_chunk_size, num_frames)
                    transfer_chunk = images[transfer_start:transfer_end].to(
                        device=self.device, non_blocking=True
                    )
                    transfer_count = transfer_chunk.shape[0]

                    for local_offset in range(0, transfer_count, chunk_size):
                        frame_idx = transfer_start + local_offset
                        local_end = min(local_offset + chunk_size, transfer_count)
                        frame_buffer = transfer_chunk[local_offset:local_end]
                        frame_buffer_size = frame_buffer.shape[0]
                        outputs = self.model(
                            image=frame_buffer,
                            mode="full",
                            streaming=True,
                            timing=debug_timing,
                        )
                        timing = outputs.get("timing", {})
                        if not self.cam_cache_update and self.model.camera_head is not None:
                            self.camera_head_inference(outputs["aggregated_tokens_list"])
                        
                        prune_time = self.model.aggregator.prune_kv_mgr(timing=debug_timing)
                        timing["kv_pruning_time"] = prune_time

                        self.pushback_prediction(outputs)
                        self._update_benchmark(outputs.get("timing", {}))

                        kvcache_info = (self.model.aggregator.get_kv_mgr_info()
                                        if hasattr(self.model, 'aggregator')
                                        else self.aggregator.get_kv_mgr_info())
                        kvcache_size = kvcache_info["kvcache_size"][0]
                        kvcache_mem = kvcache_info["kvcache_used"]
                        # When CPU offload is active, show total (gpu+cpu) so stats reflect full context
                        if "kvcache_size_total" in kvcache_info:
                            total_tok = kvcache_info["kvcache_size_total"][0]
                            total_mem = kvcache_info["kvcache_used_total"]
                            cache_str = f"tokens={total_tok} (gpu {kvcache_size})  mem={total_mem:.0f}MB (gpu {kvcache_mem:.0f}MB)"
                        else:
                            cache_str = f"tokens={kvcache_size}  mem={kvcache_mem:.0f}MB"

                        agg_time = timing.get("aggregator_infer_time", 0) / frame_buffer_size
                        prune_time = timing.get("kv_pruning_time", 0) / frame_buffer_size
                        allocated, reserved = _gpu_mem_mb()

                        progress.update(task, advance=frame_buffer_size)
                        live.update(Group(
                            progress,
                            _stats_table([
                                ("Time(ms)", f"agg={agg_time:.1f}  prune={prune_time:.1f}"),
                                ("KV Cache", cache_str),
                                ("GPU(MB)", f"alloc={allocated:.0f}  reserved={reserved:.0f}"),
                            ]),
                        ))
                        self._processed_frames += frame_buffer_size

                    del transfer_chunk
            if VERBOSE:
                logger.info("Window mode done.")
            # Token stats are not meaningful for our kv_manager; keep Token empty. Persist Memory(MB) for eval/compare.
            if self.model_type == "da3":
                kv_mgr = self.aggregator.kv_manager
            else:
                kv_mgr = self.model.aggregator.kv_manager
            if kv_mgr is not None:
                metrics = {"Token": {}}
                if hasattr(kv_mgr, "get_memory_details"):
                    metrics["Memory(MB)"] = kv_mgr.get_memory_details()
                self.stats = metrics

        elif mode in ["window_chunk_merge"]:
            # Use Voxel attention to maintain a voxel + recent KV cache for the aggregator.
            voxel_size = kwargs.get("voxel_size", 0.05)
            dist_thres = 2.0 * voxel_size
            kv_kwargs = deepcopy(kwargs)
            chunk_size = kwargs.get("chunk_size", 1)
            window_size = kwargs.get("window_size", 0)
            if chunk_size < 1:
                raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
            if window_size > 0 and chunk_size > window_size:
                logger.warning(
                    f"chunk_size ({chunk_size}) > window_size ({window_size}): "
                    "retrieval and print will trigger every chunk."
                )
            debug_timing = kwargs.get("timing", True)

            merge_layers = None

            sim_threshold = kwargs.get("sim_threshold", 0.8)
            merger_kwargs = {
                        "voxel_size": voxel_size,
                        "voxelize_layers": merge_layers,
                        "init_voxels": kwargs.get("voxel_num", 4096),
                        "voxel_buf_cap": kwargs.get("voxel_buf_cap", 8),
                        "voxel_piv_cap": kwargs.get("voxel_piv_cap", 4),
                        "voxel_backend": kwargs.get("voxel_backend", "python"),
                        "sim_threshold": sim_threshold,
                        "replace_threshold": sim_threshold,
                        "score_threshold": 0.2,
                        "slab_growth": 1024,
                        "slab_cap": 10000,
                        "seg_size": 1,
                        "retrieval_size": kwargs.get("retrieval_size", -1),
                        "allocator": kwargs.get("allocator", "slab"),
                        # CPU offload parameters
                        "enable_alloc_cpu": kwargs.get("enable_alloc_cpu", False),
                        "gpu_threshold_gb": kwargs.get("gpu_threshold_gb", 10.0),
                        "cold_frame_threshold": kwargs.get("cold_frame_threshold", 5),
            }
            kv_kwargs.update(merger_kwargs)
            kv_kwargs = self.register_kv_mgr(mode, images, STACVoxelKV, **kv_kwargs)
            if self.model_type == "da3":
                kv_manager = self.aggregator.kv_manager
            else:
                kv_manager = self.model.aggregator.kv_manager

            window_size = kv_kwargs.get("recent_size", 0)
            ret_size = kv_kwargs.get("retrieval_size", -1)
            buffer_size = kv_kwargs.get("buffer_size", 16)

            if self.model_type == "da3":
                pass  # DA3 doesn't have set_camhead
            else:
                self.model.set_camhead(self.cam_cache_update)

            conf_threshold = kwargs.get("conf_threshold", 2.0)
            transfer_chunk_size = kwargs.get("transfer_chunk_size", max(chunk_size, 16))
            if transfer_chunk_size < chunk_size:
                logger.warning(
                    "transfer_chunk_size (%d) < chunk_size (%d), clamping to chunk_size.",
                    transfer_chunk_size, chunk_size,
                )
                transfer_chunk_size = chunk_size
            logger.info("STAC chunk-merge: chunk=%d, transfer_chunk=%d, window=%d, conf_threshold=%.1f",
                        chunk_size, transfer_chunk_size, window_size, conf_threshold)
            special_tokens_size = (self.model.aggregator.patch_start_idx
                                   if hasattr(self.model, 'aggregator')
                                   else self.aggregator.patch_start_idx)
            
            progress, task = _make_progress("STAC Mode", num_frames)
            with Live(progress, console=_console, refresh_per_second=8) as live:
                for transfer_start in range(0, num_frames, transfer_chunk_size):
                    transfer_end = min(transfer_start + transfer_chunk_size, num_frames)
                    transfer_chunk = images[transfer_start:transfer_end].to(
                        device=self.device, non_blocking=True
                    )
                    transfer_count = transfer_chunk.shape[0]

                    for local_offset in range(0, transfer_count, chunk_size):
                        frame_idx = transfer_start + local_offset
                        local_end = min(local_offset + chunk_size, transfer_count)
                        frame_buffer = transfer_chunk[local_offset:local_end]
                        frame_buffer_size = frame_buffer.shape[0]
                        outputs = self.model(
                            image=frame_buffer,
                            mode="full",
                            streaming=True,
                            timing=debug_timing,
                        )
                        timing = outputs.get("timing", {})
                        if self.model_type == "da3":
                            pose_enc = outputs["pose_enc"]
                            # Accumulate DA3 poses globally across chunks
                            # DA3 outputs relative extrinsics per chunk; we need global poses
                            pose_enc, outputs = self._accumulate_da3_poses(
                                pose_enc, outputs, frame_idx, frame_buffer_size
                            )
                        elif not self.cam_cache_update and self.model.camera_head is not None:
                            cam_output = self.camera_head_inference(outputs["aggregated_tokens_list"])
                            pose_enc = cam_output["pose_enc"]
                        else:
                            pose_enc = outputs["pose_enc"]

                        pts3d, valid_mask = self.get_pointmap(outputs, conf_threshold=conf_threshold,
                                                              special_tokens_size=special_tokens_size,
                                                              pose_enc = pose_enc, images=frame_buffer
                                                              )
                        if self.model_type == "da3":
                            kv_pos_time = self.aggregator.update_kv_mgr_pos(pts3d, valid_mask, timing=debug_timing)
                        else:
                            kv_pos_time = self.model.aggregator.update_kv_mgr_pos(pts3d, valid_mask, timing=debug_timing)
                        timing["kv_position_time"] = kv_pos_time

                        retrieval_time = 0.0
                        if frame_idx > max(buffer_size, 16):
                            if ret_size > 0:
                                chunks_per_window = max(1, window_size // chunk_size)
                                if (frame_idx // chunk_size + 1) % chunks_per_window == 0:
                                    if self.model_type == "da3":
                                        retrieval_time = self.aggregator.retrieve_kv_mgr(timing=debug_timing, verbose=False,
                                                                                           dist_thres=dist_thres,
                                                                                           return_buf=kwargs.get("return_buf", False))
                                    else:
                                        retrieval_time = self.model.aggregator.retrieve_kv_mgr(timing=debug_timing, verbose=False,
                                                                                           dist_thres=dist_thres,
                                                                                           return_buf=kwargs.get("return_buf", False))
                            elif ret_size == -1:
                                if self.model_type == "da3":
                                    retrieval_time = self.aggregator.retrieve_kv_mgr(timing=debug_timing, verbose=False,
                                                                                       dist_thres=dist_thres,
                                                                                       return_buf=kwargs.get("return_buf", False))
                                else:
                                    retrieval_time = self.model.aggregator.retrieve_kv_mgr(timing=debug_timing, verbose=False,
                                                                                       dist_thres=dist_thres,
                                                                                       return_buf=kwargs.get("return_buf", False))

                        timing["kv_retrieval_time"] = retrieval_time

                        evict_merge_time = 0.0
                        if self.model_type == "da3":
                            evict_merge_time = self.aggregator.prune_kv_mgr(timing=debug_timing)
                        else:
                            evict_merge_time = self.model.aggregator.prune_kv_mgr(timing=debug_timing)
                        timing["kv_evict_merge_time"] = evict_merge_time

                        if frame_idx % (chunk_size * 4) == 0 or frame_idx >= num_frames - chunk_size:
                            _mem_profile = os.environ.get("MERGER_MEM_PROFILE", "0") == "1"
                            if _mem_profile:
                                torch.cuda.synchronize()
                                a_before = torch.cuda.memory_allocated() / (1024**2)
                                r_before = torch.cuda.memory_reserved() / (1024**2)
                                frag_before = r_before - a_before
                            torch.cuda.empty_cache()
                            if _mem_profile:
                                r_after = torch.cuda.memory_reserved() / (1024**2)
                                frag_after = r_after - a_before
                                freed = r_before - r_after
                                logger.debug(
                                    "  [MEM-FRAG] frame=%d | alloc=%.0fMB, "
                                    "res_before=%.0fMB, res_after=%.0fMB, "
                                    "frag_before=%.0fMB, frag_after=%.0fMB, "
                                    "freed_by_empty_cache=%.0fMB",
                                    frame_idx, a_before, r_before, r_after,
                                    frag_before, frag_after, freed,
                                )

                        self.pushback_prediction(outputs)
                        self._update_benchmark(timing)

                        kvcache_info = (self.model.aggregator.get_kv_mgr_info()
                                        if hasattr(self.model, 'aggregator')
                                        else self.aggregator.get_kv_mgr_info())
                        merger_stat = kv_manager.get_merger_info()
                        merger_stat["frame_idx"] = frame_idx
                        total_time = 0.0
                        for key, value in timing.items():
                            merger_stat[key] = value / frame_buffer_size
                            total_time += value
                        merger_stat["total_time"] = total_time / frame_buffer_size

                        allocated, reserved = _gpu_mem_mb()

                        agg_t = timing.get("aggregator_infer_time", 0) / frame_buffer_size
                        pos_t = kv_pos_time / frame_buffer_size
                        mrg_t = evict_merge_time / frame_buffer_size
                        ret_t = retrieval_time / frame_buffer_size

                        mem_details = kv_manager.get_memory_details()
                        temporal_mem   = mem_details.get("temporal_cache_usage", 0)
                        vox_used  = (mem_details.get("voxel_buffer_usage", 0)
                                     + mem_details.get("voxel_pivot_usage", 0))
                        vox_alloc = (mem_details.get("voxel_buffer_alloc", 0)
                                     + mem_details.get("voxel_pivot_alloc", 0))
                        spatial_mem   = mem_details.get("spatial_cache_usage", 0)

                        progress.update(task, advance=frame_buffer_size)
                        live.update(Group(
                            progress,
                            _stats_table([
                                ("Time(ms)", f"agg={agg_t:.1f} | ret={ret_t:.1f} | pos={pos_t:.1f} | evict&merge={mrg_t:.1f}"),
                                ("Cache(MB)", f"temporal={temporal_mem:.0f} | spatial(retrieval)={spatial_mem:.0f} |  voxel(used/alloc)={vox_used:.0f}/{vox_alloc:.0f}  "),
                                ("GPU(MB)", f"allocated={allocated:.0f} | reserved={reserved:.0f}"),
                            ]),
                        ))
                        self._processed_frames += frame_buffer_size

                    del transfer_chunk
            if VERBOSE:
                logger.info("STAC chunk-merge mode done.")

            # Token stats are not meaningful for our kv_manager; keep Token empty. Persist Memory(MB) for eval/compare.
            if self.model_type == "da3":
                kv_mgr = self.aggregator.kv_manager
            else:
                kv_mgr = self.model.aggregator.kv_manager
            if kv_mgr is not None:
                metrics = {"hyperparameters": merger_kwargs, "Token": {}}
                if hasattr(kv_mgr, "get_memory_details"):
                    metrics["Memory(MB)"] = kv_mgr.get_memory_details()
                self.stats = metrics

        # Post-hoc global metric alignment for DA3 nested models
        if self.model_type == "da3":
            self.apply_global_metric_alignment()

    def register_kv_mgr(self, mode,
                            images, 
                            kv_manager,
                            **kwargs):
            
            default_kwargs = {
                "chunk_size": kwargs.get("chunk_size", 1),
                "recent_size": kwargs.get("window_size", 2),
                "pinned_idx": kwargs.get("pinned_frame_indices", [0]),
                "hh_size": 0,
                "persist_size": 0,
                "temperature": 0.9,
                "device": self.device,
                "dtype": kwargs.get("dtype", torch.float16),
            }

            kwargs_kv = default_kwargs.copy()
            kwargs_kv.update(kwargs)

            recent_size = kwargs_kv["recent_size"]
            assert recent_size >= 1, "window_size must be at least 1."
            pinned_frame_indices = kwargs_kv["pinned_frame_indices"]
            hh_size = kwargs["hh_size"]
            chunk_size = kwargs["chunk_size"]
            pinned_size = len(pinned_frame_indices)
            buffer_size = chunk_size + pinned_size + recent_size + hh_size
            if buffer_size > 300:
                logger.warning(f"Buffer size {buffer_size} is large, may cause OOM issues; part memory offload to CPU device.")
            logger.info(f"Using {mode} mode: processing frames in windows of size {recent_size} with {kv_manager.__name__}-manager")

            if len(images.shape) == 4:
                S, C, H, W = images.shape
            else:
                B, S, C, H, W = images.shape
                assert B == 1, "Batch size must be 1 when input is 5D."
            if self.model_type == "da3":
                vit_patch_size = self.aggregator.patch_embed.patch_size
                cam_tokens = self.aggregator.patch_start_idx
            else:
                vit_patch_size = self.model.aggregator.patch_embed.patch_size
                cam_tokens = self.model.aggregator.patch_start_idx
            if isinstance(vit_patch_size, (list, tuple)):
                vit_patch_size = vit_patch_size[0]
            img_tokens = (H // vit_patch_size) * (W // vit_patch_size)
            token_per_frame = img_tokens + cam_tokens

            kwargs_kv.update({
                "token_per_frame": token_per_frame,
                "buffer_size": buffer_size,
            })
            if self.model_type == "da3":
                self.aggregator.register_kv_mgr(kv_manager, **kwargs_kv)
            else:
                self.model.aggregator.register_kv_mgr(kv_manager, **kwargs_kv)

            return kwargs_kv
