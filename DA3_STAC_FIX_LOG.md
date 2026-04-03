# DA3 STAC 集成修复记录

> 日期: 2026-04-03
> 目标: 使 DA3 (Depth Anything 3) 模型在 STAC 流式推理框架下正常工作

---

## 问题清单

### 1. 检查点路径不匹配
**现象**: DA3 检查点在 `ckpt/model.safetensors`，但代码期望 `ckpt/da3/DA3NESTED-GIANT-LARGE-1.1/model.safetensors`

**修复** (方案 A):
```bash
mkdir -p ckpt/da3/DA3NESTED-GIANT-LARGE-1.1
mv ckpt/model.safetensors ckpt/da3/DA3NESTED-GIANT-LARGE-1.1/
mv ckpt/config.json ckpt/da3/DA3NESTED-GIANT-LARGE-1.1/
```

---

### 2. `--base_model` 缺少 `da3nested-giant-large` 选项
**文件**: `run_streaming_inference.py`

**修复**:
```python
# 添加 "da3nested-giant-large" 到 choices 列表
choices=["stream3r", "streamvggt", "da3-small", "da3-base",
         "da3-large", "da3-large-1.1", "da3-giant", "da3nested-giant-large"],
```

---

### 3. DA3 backbone 访问路径错误
**现象**: `stream_session.py` 使用 `model.model.backbone.pretrained`，但 `NestedDepthAnything3Net` 的路径是 `model.model.da3.backbone.pretrained`

**文件**: `stream_session.py`

**修复**:
```python
if model_type == "da3":
    # NestedDepthAnything3Net has da3.da3.backbone.pretrained
    if hasattr(model.model, 'da3'):
        self.aggregator = model.model.da3.backbone.pretrained
    elif hasattr(model.model, 'backbone'):
        self.aggregator = model.model.backbone.pretrained
    else:
        raise AttributeError(...)
```

---

### 4. DA3 forward 参数不兼容
**现象**: DA3 的 `forward` 接受 `image=` 而非 `images=`，且不接受 `camera_head_kv_cache_list` 和 `is_anchor_exist`

**文件**: `stream_session.py`

**修复**: 在 warmup 和 pipeline 中，DA3 路径使用不同的参数:
```python
if self.model_type == "da3":
    self.model(image=frame_buffer, mode="full", streaming=True, timing=debug_timing)
else:
    self.model(images=frame_buffer, mode="full", camera_head_kv_cache_list=None,
               streaming=True, is_anchor_exist=True, timing=debug_timing)
```

---

### 5. DA3 forward 不接受 `mode`/`streaming` 参数
**文件**: `depth-anything-3/src/depth_anything_3/api.py`

**修复**: 添加参数透传:
```python
def forward(self, image, extrinsics=None, intrinsics=None, ...,
            mode: str = "full", streaming: bool = False, **kwargs):
    return self.model(image, extrinsics, intrinsics, ...,
                      mode=mode, streaming=streaming, **kwargs)
```

---

### 6. `NestedDepthAnything3Net.forward` 不接受 `mode`/`streaming` 参数
**文件**: `depth-anything-3/src/depth_anything_3/model/da3.py`

**修复**: 添加参数并透传给内部调用:
```python
def forward(self, x, ..., mode: str = "full", streaming: bool = False, **kwargs):
    output = self.da3(x, ..., mode=mode, streaming=streaming, **kwargs)
    if streaming:
        return output  # streaming 模式已返回 STAC 格式
    # ... 非 streaming 的 metric scaling 等
```

---

### 7. xformers SwiGLU CUDA 算子版本不匹配
**现象**: `libswiglu.so was compiled against CUDA 12.4 but this is CUDA 12.6`

**修复**: 在 `swiglu_ffn.py` 中添加运行时检测，失败时回退到纯 PyTorch:
```python
try:
    from xformers.ops import SwiGLU
    _test_swiglu = SwiGLU(8, 16, 8, bias=False)
    _test_swiglu(torch.randn(1, 8))
    XFORMERS_AVAILABLE = True
except Exception:
    SwiGLU = SwiGLUFFN
    XFORMERS_AVAILABLE = False
```

---

### 8. `export_feat_layers=None` 导致 `in` 操作报错
**文件**: `depth-anything-3/src/depth_anything_3/model/dinov2/vision_transformer.py`

**修复**: 先检查 None:
```python
if export_feat_layers is not None and i in export_feat_layers:
```

---

### 9. `_extract_auxiliary_features` 未处理 None 输入
**文件**: `depth-anything-3/src/depth_anything_3/model/da3.py`

**修复**:
```python
def _extract_auxiliary_features(self, feats, feat_layers, H, W):
    aux_features = Dict()
    if feats is None or feat_layers is None:
        return aux_features
    # ...
```

---

### 10. `patch_size` 是 tuple 不能直接整除
**文件**: `stream_session.py`

**修复**:
```python
vit_patch_size = self.aggregator.patch_embed.patch_size
if isinstance(vit_patch_size, (list, tuple)):
    vit_patch_size = vit_patch_size[0]
```

---

### 11. `register_kv_mgr` 参数名不一致
**现象**: CausalAggregator 用 `kv_manager=`, DA3 用位置参数

**修复**: 统一使用位置参数:
```python
if self.model_type == "da3":
    self.aggregator.register_kv_mgr(kv_manager, **kwargs_kv)
else:
    self.model.aggregator.register_kv_mgr(kv_manager, **kwargs_kv)
```

---

### 12. `get_pointmap` 中 `point_head.patch_size` 不存在
**文件**: `stream_session.py`

**修复**:
```python
if self.model_type == "da3":
    ds_patch = 14  # DA3 ViT patch size
else:
    ds_patch = self.model.point_head.patch_size
```

---

### 13. `update_kv_mgr_pos` 参数名不匹配
**现象**: DA3 传 `pts_3d=`, 但 `append_positions` 期望 `new_positions=`

**文件**: `depth-anything-3/src/depth_anything_3/model/dinov2/vision_transformer.py`

**修复**:
```python
def update_kv_mgr_pos(self, pts3d, valid_mask, timing=False):
    if self.kv_manager is not None and hasattr(self.kv_manager, 'append_positions'):
        import time
        start = time.time()
        # Flatten: [S, T_per_frame, 3] -> [S*T_per_frame, 3]
        pts3d = pts3d.view(-1, 3)
        valid_mask = valid_mask.view(-1).bool()
        self.kv_manager.append_positions(new_positions=pts3d, new_pos_mask=valid_mask)
        return (time.time() - start) * 1000 if timing else 0.0
    return 0.0
```

---

### 14. `prune_kv` / `retrieve_kv` 不接受 `timing` 参数
**文件**: `depth-anything-3/src/depth_anything_3/model/dinov2/vision_transformer.py`

**修复**: 在 wrapper 中手动计时:
```python
def prune_kv_mgr(self, timing=False):
    if self.kv_manager is not None:
        import time
        start = time.time()
        self.kv_manager.prune_kv()
        return (time.time() - start) * 1000 if timing else 0.0
    return 0.0

def retrieve_kv_mgr(self, timing=False, verbose=False, dist_thres=0.1, return_buf=False):
    if self.kv_manager is not None and hasattr(self.kv_manager, 'retrieve_kv'):
        import time
        start = time.time()
        self.kv_manager.retrieve_kv_parallel(verbose=verbose, dist_thres=dist_thres, return_buf=return_buf)
        return (time.time() - start) * 1000 if timing else 0.0
    return 0.0
```

---

### 15. `get_kv_mgr_info` 调用了不存在的 `get_info()`
**文件**: `depth-anything-3/src/depth_anything_3/model/dinov2/vision_transformer.py`

**修复**:
```python
def get_kv_mgr_info(self):
    if self.kv_manager is not None:
        info = {}
        if hasattr(self.kv_manager, 'get_memory_details'):
            info.update(self.kv_manager.get_memory_details())
        if hasattr(self.kv_manager, '_offset_hot'):
            info['kvcache_size'] = [t for t in self.kv_manager._offset_hot]
        return info
    return {}
```

---

### 16. bfloat16 无法直接转 numpy
**文件**: `run_streaming_inference.py`

**修复**: 所有 `.cpu().numpy()` 改为 `.cpu().float().numpy()`

---

### 17. DA3 `_convert_to_stac_format` 使用 `.get()` 无法访问属性
**现象**: DA3 的 output 是 `addict.Dict`，键值通过属性访问而非 `.get()`

**文件**: `depth-anything-3/src/depth_anything_3/model/da3.py`

**修复**:
```python
# 所有 output.get("key", None) 改为 getattr(output, "key", None)
extrinsics = getattr(output, "extrinsics", None)
intrinsics = getattr(output, "intrinsics", None)
depth = getattr(output, "depth", None)
depth_conf = getattr(output, "depth_conf", None)
```

同时保留 extrinsics/intrinsics 供外部位姿累积使用:
```python
if extrinsics is not None:
    pose_enc = self._encode_pose_from_extrinsics(extrinsics)
    stac_output["pose_enc"] = pose_enc
    stac_output["extrinsics"] = extrinsics  # 保留供 stream_session 累积
    stac_output["intrinsics"] = intrinsics
```

---

### 18. DA3 位姿不累积（每个 chunk 独立坐标系）
**现象**: 
- 每个 chunk 的 4 帧在自身坐标系内推理
- chunk 之间位姿不连续，边界跳变是内部的 2.9 倍
- 3840 帧总位移仅 0.065（实际应约 1700 米）

**文件**: `stream_session.py`

**修复**: 添加 `_accumulate_da3_poses` 方法，跨 chunk 累积位姿:

```python
# 初始化
self._da3_last_c2w = None  # 上一个 chunk 最后一帧的 c2w

def _accumulate_da3_poses(self, pose_enc, outputs, frame_idx, chunk_size):
    extrinsics_w2c = outputs.get("extrinsics", None)
    if extrinsics_w2c is None:
        extrinsics_w2c = getattr(outputs, "extrinsics", None)
    if extrinsics_w2c is None:
        return pose_enc, outputs
    
    B, N = extrinsics_w2c.shape[0], extrinsics_w2c.shape[1]
    # w2c -> c2w
    ones = torch.zeros(B, N, 1, 4, device=extrinsics_w2c.device, dtype=extrinsics_w2c.dtype)
    ones[:, :, 0, 3] = 1.0
    w2c_homo = torch.cat([extrinsics_w2c, ones], dim=2)
    c2w_homo = self._invert_pose(w2c_homo)
    
    if self._da3_last_c2w is None:
        # 第一个 chunk: 直接使用
        self._da3_last_c2w = c2w_homo[:, -1:, :, :].detach()
    else:
        # 后续 chunk: 计算相对变换并累积
        first_c2w_inv = self._invert_pose(c2w_homo[:, :1, :, :])
        rel_transforms = torch.matmul(c2w_homo, first_c2w_inv.expand(-1, N, -1, -1))
        global_c2w = torch.matmul(self._da3_last_c2w.expand(-1, N, -1, -1), rel_transforms)
        global_w2c = self._invert_pose(global_c2w)
        global_w2c_3x4 = global_w2c[:, :, :3, :]
        
        outputs["extrinsics"] = global_w2c_3x4
        pose_enc = self._encode_pose_from_extrinsics(global_w2c_3x4)
        outputs["pose_enc"] = pose_enc  # 重要: 更新 outputs 以便 pushback 保存
        self._da3_last_c2w = global_c2w[:, -1:, :, :].detach()
    
    return pose_enc, outputs
```

在 pipeline 中调用:
```python
if self.model_type == "da3":
    pose_enc = outputs["pose_enc"]
    pose_enc, outputs = self._accumulate_da3_poses(pose_enc, outputs, frame_idx, frame_buffer_size)
```

---

### 19. FPS 计时虚高（1224 vs 实际 25）
**现象**: `infer_fps` 只统计后处理时间（kv_position + retrieval + evict_merge ≈ 1ms），不含模型前向（≈ 39ms）

**修复**:
1. 在 `depth-anything-3/src/depth_anything_3/model/da3.py` 的 forward 中添加 CUDA 计时:
```python
debug_timing = kwargs.get("timing", False)
if debug_timing:
    time_start = torch.cuda.Event(enable_timing=True)
    time_end = torch.cuda.Event(enable_timing=True)
    time_start.record()
# ... forward logic ...
if debug_timing:
    time_end.record()
    torch.cuda.synchronize()
    aggregator_time = time_start.elapsed_time(time_end)
    output["timing"]["aggregator_infer_time"] = aggregator_time
```

2. 在 `_convert_to_stac_format` 中传递 timing:
```python
if "timing" in output:
    stac_output["timing"] = output["timing"]
```

**修复后**: FPS 1224 → **25.2**（aggregator_infer_time ≈ 38.8ms/帧）

---

### 20. `pinned_frame_indices` 缺失
**文件**: `run_streaming_inference.py`

**修复**:
```python
stac_kwargs = {
    ...
    "pinned_frame_indices": [0],
}
```

---

## 修复效果对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| **FPS** | 1224（虚高） | 25.2（真实） |
| **总位移** | 0.065 | 19.87（归一化尺度） |
| **Chunk 边界跳变比** | 2.90x | 0.11x |
| **帧重复** | 无 | 无 |
| **Roll 扭麻花** | 无 | 无 |
| **3840 帧推理时间** | ~176s | ~175s |

## 修改文件清单

| 文件 | 修改数 |
|------|--------|
| `stream_session.py` | +80 行（位姿累积、DA3 路径适配、参数兼容） |
| `run_streaming_inference.py` | ~10 行（base_model 选项、bfloat16 转换、pinned_frame） |
| `depth-anything-3/src/depth_anything_3/api.py` | ~5 行（mode/streaming 参数透传） |
| `depth-anything-3/src/depth_anything_3/model/da3.py` | ~30 行（forward 参数、timing、getattr 修复、extrinsics 保留） |
| `depth-anything-3/src/depth_anything_3/model/dinov2/vision_transformer.py` | ~20 行（KV manager wrapper 方法） |
| `depth-anything-3/src/depth_anything_3/model/dinov2/layers/swiglu_ffn.py` | ~3 行（运行时回退检测） |
