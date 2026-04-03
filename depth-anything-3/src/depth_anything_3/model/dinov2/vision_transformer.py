# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import math
from typing import Callable, List, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask

from depth_anything_3.utils.logger import logger

from .layers import LayerScale  # noqa: F401
from .layers import Mlp  # noqa: F401
from .layers import (  # noqa: F401
    Block,
    PatchEmbed,
    PositionGetter,
    RotaryPositionEmbedding2D,
    SwiGLUFFNFused,
)
from depth_anything_3.model.reference_view_selector import (
    RefViewStrategy,
    select_reference_view,
    reorder_by_reference,
    restore_original_order,
)
from depth_anything_3.utils.constants import THRESH_FOR_REF_SELECTION
from depth_anything_3.model.dinov2.layers.attention import SparseAttention

# logger = logging.getLogger("dinov2")


def create_attn_mask(S: int, P: int, mode: str,
                     dtype: torch.dtype, device: torch.device,
                     look_ahead_size: int = 0,
                     window_size: int = 2,
                     ) -> torch.Tensor:
    """
    Create attention mask for block-wise attention.
    Adapted from STAC's causalvggt/layers/block.py for DA3 integration.

    Args:
        S (int): Number of frames in the sequence.
        P (int): Number of patches per frame.
        mode (str): Attention mode: "causal", "window", "full", "test".
        dtype (torch.dtype): Data type of the mask.
        device (torch.device): Device to place the mask on.
        look_ahead_size (int): Number of future frames to attend to.
        window_size (int): Window size for sliding window attention.

    Returns:
        torch.Tensor or None: Attention block mask for flex_attention.
    """
    total_length = S * P
    mask = None

    if mode == "causal" or "causal" in mode:
        ends = torch.zeros(total_length, device=device, dtype=torch.long)
        frame_indices = torch.arange(start=0, end=total_length, step=P, device=device)

        for frame_idx, frame_start in enumerate(frame_indices):
            frame_end = frame_start + P
            context_end = min((frame_idx + 1 + look_ahead_size) * P, total_length)
            ends[frame_start:frame_end] = context_end

        def block_causal_mask(b, h, q_idx, kv_idx):
            _ = b, h
            return (kv_idx < ends[q_idx])

        mask = torch.compile(create_block_mask)(
            block_causal_mask,
            B=None, H=None,
            Q_LEN=total_length,
            KV_LEN=total_length,
            device=device,
        )
    elif mode == "window":
        ends = torch.zeros(total_length, device=device, dtype=torch.long)
        starts = torch.zeros(total_length, device=device, dtype=torch.long)
        frame_indices = torch.arange(start=0, end=total_length, step=P, device=device)

        for frame_idx, frame_start in enumerate(frame_indices):
            frame_end = frame_start + P
            context_end = min((frame_idx + 1 + look_ahead_size) * P, total_length)
            ends[frame_start:frame_end] = context_end
            start_frame = max(0, frame_idx - window_size)
            context_start = start_frame * P
            starts[frame_start:frame_end] = context_start

        def block_window_mask(b, h, q_idx, kv_idx):
            _ = b, h
            return ((starts[q_idx] <= kv_idx) & (kv_idx < ends[q_idx])) | (kv_idx < P)

        mask = torch.compile(create_block_mask)(
            block_window_mask,
            B=None, H=None,
            Q_LEN=total_length,
            KV_LEN=total_length,
            device=device,
        )
    elif mode == "test":
        ends = torch.zeros(total_length, device=device, dtype=torch.long)
        starts = torch.zeros(total_length, device=device, dtype=torch.long)
        frame_indices = torch.arange(start=0, end=total_length, step=P, device=device)
        window_start = window_size * P
        for frame_idx, frame_start in enumerate(frame_indices):
            frame_end = frame_start + P
            context_end = min((frame_idx + 1) * P, total_length) + window_start
            ends[frame_start:frame_end] = context_end
            context_start = frame_idx * P + window_start
            starts[frame_start:frame_end] = context_start

        def block_window_mask(b, h, q_idx, kv_idx):
            _ = b, h
            return ((starts[q_idx] <= kv_idx) & (kv_idx < ends[q_idx])) | (kv_idx < window_start)

        mask = torch.compile(create_block_mask)(
            block_window_mask,
            B=None, H=None,
            Q_LEN=total_length,
            KV_LEN=total_length + window_start,
            device=device,
        )
    elif mode == "full":
        mask = None
    else:
        raise NotImplementedError(f"Unknown attention mode: {mode}")
    return mask


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=1.0,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        alt_start=-1,
        qknorm_start=-1,
        rope_start=-1,
        rope_freq=100,
        plus_cam_token=False,
        cat_token=True,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating
                positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating
                positional embeddings
        """
        super().__init__()
        self.patch_start_idx = 1
        norm_layer = nn.LayerNorm
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.alt_start != -1:
            self.camera_token = nn.Parameter(torch.randn(1, 2, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule
        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        if self.rope_start != -1:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
            self.position_getter = PositionGetter() if self.rope is not None else None
        else:
            self.rope = None
        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                qk_norm=i >= qknorm_start if qknorm_start != -1 else False,
                rope=self.rope if i >= rope_start and rope_start != -1 else None,
            )
            for i in range(depth)
        ]

        # === STAC Integration: Replace attention with SparseAttention for global layers ===
        # Global layers are those at odd indices after alt_start
        self._global_layer_indices = []
        for i in range(depth):
            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                # This is a global layer, replace with SparseAttention
                old_attn = blocks_list[i].attn
                new_attn = SparseAttention(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    attn_drop=old_attn.attn_drop.p,
                    proj_drop=old_attn.proj_drop.p,
                    norm_layer=type(old_attn.q_norm) if not isinstance(old_attn.q_norm, nn.Identity) else nn.LayerNorm,
                    qk_norm=i >= qknorm_start if qknorm_start != -1 else False,
                    fused_attn=old_attn.fused_attn,
                    rope=self.rope if i >= rope_start and rope_start != -1 else None,
                )
                # Copy weights from old attention to new SparseAttention
                new_attn.qkv.load_state_dict(old_attn.qkv.state_dict())
                new_attn.proj.load_state_dict(old_attn.proj.state_dict())
                if not isinstance(old_attn.q_norm, nn.Identity):
                    new_attn.q_norm.load_state_dict(old_attn.q_norm.state_dict())
                if not isinstance(old_attn.k_norm, nn.Identity):
                    new_attn.k_norm.load_state_dict(old_attn.k_norm.state_dict())

                blocks_list[i].attn = new_attn
                blocks_list[i].attn.set_layer_idx(len(self._global_layer_indices))
                self._global_layer_indices.append(i)

        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)

        # KV manager for streaming inference
        self.kv_manager = None

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the
            # interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using
            # both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_cls_token(self, B, S):
        cls_token = self.cls_token.expand(B, S, -1)
        cls_token = cls_token.reshape(B * S, -1, self.embed_dim)
        return cls_token

    def prepare_tokens_with_masks(self, x, masks=None, cls_token=None, **kwargs):
        B, S, nc, w, h = x.shape
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        cls_token = self.prepare_cls_token(B, S)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        x = rearrange(x, "(b s) n c -> b s n c", b=B, s=S)
        return x

    def _prepare_rope(self, B, S, H, W, device):
        pos = None
        pos_nodiff = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=device
            )
            pos = rearrange(pos, "(b s) n c -> b s n c", b=B)
            pos_nodiff = torch.zeros_like(pos).to(pos.dtype)
            if self.patch_start_idx > 0:
                pos = pos + 1
                pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(device).to(pos.dtype)
                pos_special = rearrange(pos_special, "(b s) n c -> b s n c", b=B)
                pos = torch.cat([pos_special, pos], dim=2)
                pos_nodiff = pos_nodiff + 1
                pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=2)
        return pos, pos_nodiff

    def _get_intermediate_layers_not_chunked(self, x, n=1, export_feat_layers=[], **kwargs):
        B, S, _, H, W = x.shape
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len, aux_output = [], len(self.blocks), []
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)

        for i, blk in enumerate(self.blocks):
            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos

            if self.alt_start != -1 and (i == self.alt_start - 1) and x.shape[1] >= THRESH_FOR_REF_SELECTION and kwargs.get("cam_token", None) is None:
                # Select reference view using configured strategy
                strategy = kwargs.get("ref_view_strategy", "saddle_balanced")
                logger.info(f"Selecting reference view using strategy: {strategy}")
                b_idx = select_reference_view(x, strategy=strategy)
                # Reorder views to place reference view first
                x = reorder_by_reference(x, b_idx)
                local_x = reorder_by_reference(local_x, b_idx)

            if self.alt_start != -1 and i == self.alt_start:
                if kwargs.get("cam_token", None) is not None:
                    logger.info("Using camera conditions provided by the user")
                    cam_token = kwargs.get("cam_token")
                else:
                    ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                    src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                    cam_token = torch.cat([ref_token, src_token], dim=1)
                x[:, :, 0] = cam_token

            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                x = self.process_attention(
                    x, blk, "global", pos=g_pos, attn_mask=kwargs.get("attn_mask", None),
                    kv_manager=self.kv_manager
                )
            else:
                x = self.process_attention(x, blk, "local", pos=l_pos, kv_manager=None)
                local_x = x

            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                # Restore original view order if reordering was applied
                if x.shape[1] >= THRESH_FOR_REF_SELECTION and self.alt_start != -1 and 'b_idx' in locals():
                    out_x = restore_original_order(out_x, b_idx)
                output.append((out_x[:, :, 0], out_x))
            if i in export_feat_layers:
                aux_output.append(x)
        return output, aux_output

    def process_attention(self, x, block, attn_type="global", pos=None, attn_mask=None, kv_manager=None):
        b, s, n = x.shape[:3]
        if attn_type == "local":
            x = rearrange(x, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")
        elif attn_type == "global":
            x = rearrange(x, "b s n c -> b (s n) c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> b (s n) c")
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")

        x = block(x, pos=pos, attn_mask=attn_mask, kv_manager=kv_manager)

        if attn_type == "local":
            x = rearrange(x, "(b s) n c -> b s n c", b=b, s=s)
        elif attn_type == "global":
            x = rearrange(x, "b (s n) c -> b s n c", b=b, s=s)
        return x

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        export_feat_layers: List[int] = [],
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        outputs, aux_outputs = self._get_intermediate_layers_not_chunked(
            x, n, export_feat_layers=export_feat_layers, **kwargs
        )
        camera_tokens = [out[0] for out in outputs]
        if outputs[0][1].shape[-1] == self.embed_dim:
            outputs = [self.norm(out[1]) for out in outputs]
        elif outputs[0][1].shape[-1] == (self.embed_dim * 2):
            outputs = [
                torch.cat(
                    [out[1][..., : self.embed_dim], self.norm(out[1][..., self.embed_dim :])],
                    dim=-1,
                )
                for out in outputs
            ]
        else:
            raise ValueError(f"Invalid output shape: {outputs[0][1].shape}")
        aux_outputs = [self.norm(out) for out in aux_outputs]
        outputs = [out[..., 1 + self.num_register_tokens :, :] for out in outputs]
        aux_outputs = [out[..., 1 + self.num_register_tokens :, :] for out in aux_outputs]
        return tuple(zip(outputs, camera_tokens)), aux_outputs


def vit_small(patch_size=16, num_register_tokens=0, depth=12, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=depth,
        num_heads=6,
        mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, depth=12, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=depth,
        num_heads=12,
        mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, depth=24, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=depth,
        num_heads=16,
        mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, depth=40, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=depth,
        num_heads=24,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


# ======== STAC KV Manager Methods ========

def register_kv_mgr(self, kv_manager_cls, **kwargs):
    """
    Register a KV manager for streaming inference.
    Similar to CausalAggregator.register_kv_mgr() in STAC.
    """
    if self.kv_manager is None:
        # Set layer indices for all SparseAttention layers
        for idx, block_idx in enumerate(self._global_layer_indices):
            self.blocks[block_idx].attn.set_layer_idx(idx)

        num_global_layers = len(self._global_layer_indices)
        self.kv_manager = kv_manager_cls(
            num_layers=num_global_layers,
            num_heads=self.num_heads,
            head_dim=self.embed_dim // self.num_heads,
            **kwargs
        )
        self.kv_manager.reset()
        logger.info(f"Registered KV manager with {num_global_layers} global layers")
    return kwargs

DinoVisionTransformer.register_kv_mgr = register_kv_mgr


def clear_kv_mgr(self):
    """Clear the KV manager cache."""
    if self.kv_manager is not None:
        self.kv_manager.reset()
        self.kv_manager.free()
        self.kv_manager = None

DinoVisionTransformer.clear_kv_mgr = clear_kv_mgr


def prune_kv_mgr(self, timing=False):
    """Prune the KV manager cache."""
    if self.kv_manager is not None:
        return self.kv_manager.prune_kv(timing=timing)
    return 0.0

DinoVisionTransformer.prune_kv_mgr = prune_kv_mgr


def retrieve_kv_mgr(self, timing=False, verbose=False, dist_thres=0.1, return_buf=False):
    """Retrieve relevant pivots from the voxel pool."""
    if self.kv_manager is not None and hasattr(self.kv_manager, 'retrieve_kv'):
        return self.kv_manager.retrieve_kv(
            timing=timing, verbose=verbose, dist_thres=dist_thres, return_buf=return_buf
        )
    return 0.0

DinoVisionTransformer.retrieve_kv_mgr = retrieve_kv_mgr


def update_kv_mgr_pos(self, pts3d, valid_mask, timing=False):
    """Update KV manager positions with 3D points."""
    if self.kv_manager is not None and hasattr(self.kv_manager, 'append_positions'):
        return self.kv_manager.append_positions(
            pts_3d=pts3d, valid_mask=valid_mask, timing=timing
        )
    return 0.0

DinoVisionTransformer.update_kv_mgr_pos = update_kv_mgr_pos


def get_kv_mgr_info(self):
    """Get KV manager information."""
    if self.kv_manager is not None:
        return self.kv_manager.get_info()
    return {}

DinoVisionTransformer.get_kv_mgr_info = get_kv_mgr_info


def set_camhead(self, status: bool):
    """
    Set camera head status (no-op for DA3, kept for API compatibility with STAC).
    DA3 predicts camera poses directly through the backbone, not via a separate head.
    """
    pass  # DA3 doesn't have a separate camera head

DinoVisionTransformer.set_camhead = set_camhead
