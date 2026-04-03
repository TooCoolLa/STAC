# DA3 + STAC KV-Cache集成使用指南

## 概述

本指南介绍如何使用Depth Anything 3 (DA3)骨干网络与STAC的KV-cache压缩功能进行长视频流式3D重建。

## 架构集成点

### 1. DA3 ViT改造

DA3的DinoV2骨干已被改造支持KV-cache管理:

**文件**: `depth-anything-3/src/depth_anything_3/model/dinov2/`
- `vision_transformer.py` - 添加了KV manager接口
- `layers/attention.py` - 添加了`SparseAttention`类
- `layers/block.py` - forward方法支持`**kwargs`透传

**关键改造**:
```python
# SparseAttention层替换 (仅在alt_start之后的奇数层)
self._global_layer_indices = []
for i in range(depth):
    if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
        # 替换为SparseAttention
        blocks_list[i].attn = SparseAttention(...)
        self._global_layer_indices.append(i)
```

### 2. KV Manager接口

DA3 ViT现在支持以下KV管理方法:

```python
# 注册KV管理器
model.model.backbone.pretrained.register_kv_mgr(
    kv_manager_cls=KVManager,  # 或 STACVoxelKV
    num_layers=num_global_layers,
    num_heads=self.num_heads,
    head_dim=self.embed_dim // self.num_heads,
    **kwargs
)

# 清除KV缓存
model.model.backbone.pretrained.clear_kv_mgr()

# 剪枝KV缓存
model.model.backbone.pretrained.prune_kv_mgr(timing=True)

# 检索KV缓存 (用于STAC voxel模式)
model.model.backbone.pretrained.retrieve_kv_mgr(
    timing=True, 
    dist_thres=0.1, 
    return_buf=False
)

# 更新3D点位置 (用于voxel缓存)
model.model.backbone.pretrained.update_kv_mgr_pos(
    pts3d=pts3d,  # [B, S, H'*W', 3]
    valid_mask=valid_mask,  # [B, S, H'*W']
    timing=True
)

# 获取KV缓存信息
info = model.model.backbone.pretrained.get_kv_mgr_info()
# 返回: {"kvcache_size": [...], "kvcache_used": float}
```

### 3. STAC格式适配

DA3的forward方法已适配STAC输出格式:

```python
# streaming=True时,输出格式与CausalVGGT一致
output = model(
    images,  # [S, 3, H, W] 或 [B, S, 3, H, W]
    mode="full",
    streaming=True,
    **kwargs
)

# 输出包含:
# - pose_enc: [B, S, 9]
# - depth: [B, S, H, W, 1]
# - depth_conf: [B, S, H, W]
# - world_points: [B, S, H, W, 3]
# - world_points_conf: [B, S, H, W]
# - images: [B, S, 3, H, W]
# - aggregated_tokens_list: [None] (占位符,DA3无独立aggregator)
```

## 使用方法

### 方法1: 通过model_wrapper使用 (推荐)

```python
from model_wrapper import load_model, run_model
import torch

# 1. 加载DA3模型
model = load_model("da3", base_model="da3-large", device="cuda")

# 2. 准备图像 [S, 3, H, W]
# images = load_scene_images(scene_dir, size=504).to("cuda")

# 3. STAC流式推理
predictions = run_model(
    model, images, "da3",
    mode="stac",  # 自动扩展为window_chunk_merge
    streaming=True,
    dtype=torch.bfloat16,
    device="cuda",
    # STAC参数
    window_size=4,
    chunk_size=2,
    hh_size=2,
    retrieval_size=2,
    voxel_size=0.05,
    voxel_num=4096,
    conf_threshold=2.0,
)

# 4. 访问结果
print(predictions["depth"].shape)        # [1, S, H, W, 1]
print(predictions["pose_enc"].shape)     # [1, S, 9]
print(predictions["world_points"].shape) # [1, S, H, W, 3]
print(predictions["timing"])             # 性能统计
print(predictions["merger"])             # KV缓存统计
```

### 方法2: 直接使用StreamSession

```python
from stream_session import StreamSession
import torch

# 加载模型
model = load_model("da3", base_model="da3-large", device="cuda")

# 创建StreamSession (model_type="da3"是关键)
session = StreamSession(
    model, 
    device="cuda", 
    cam_cache_update=False,
    model_type="da3"  # 重要:指定使用DA3模式
)

# 运行pipeline
session.pipeline(
    images,
    mode="window_chunk_merge",  # 或 "window_kv"
    dtype=torch.bfloat16,
    device="cuda",
    window_size=4,
    chunk_size=2,
    hh_size=2,
    retrieval_size=2,
    return_buf=False,
    voxel_size=0.05,
    conf_threshold=2.0,
)

# 获取结果
predictions = session.get_all_predictions()
benchmark = session.get_benchmark()
stats = session.get_stats()

# 清理
session.clear()
```

### 方法3: CLI命令行使用

修改`main.py`以支持DA3:

```python
# 在parse_args中添加
parser.add_argument("--model_name", type=str, default="causalvggt",
                   choices=["causalvggt", "da3"])
parser.add_argument("--base_model", type=str, default="stream3r",
                   choices=["stream3r", "streamvggt", 
                            "da3-small", "da3-base", "da3-large", 
                            "da3-giant", "da3nested-giant-large"])

# 在main函数中
model = load_model(args.model_name, base_model=args.base_model, device=device)
images = load_scene_images(args.scene_dir, size=518 if args.model_name == "causalvggt" else 504).to(device)

with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
    predictions = run_model(
        model, images, args.model_name,
        mode=args.mode,
        streaming=args.streaming,
        dtype=dtype, device=device,
        **model_kwargs,
    )
```

运行命令:
```bash
# STAC模式
python main.py --scene_dir /path/to/scene \
    --model_name da3 \
    --base_model da3-large \
    --mode stac \
    --streaming

# Window KV模式 (H2O剪枝)
python main.py --scene_dir /path/to/scene \
    --model_name da3 \
    --base_model da3-large \
    --mode window_kv \
    --streaming \
    --window_size 6 \
    --chunk_size 2 \
    --hh_size 2
```

## 可用模型

| 模型名称 | 参数量 | 推荐用途 | 显存需求 |
|---------|--------|---------|---------|
| `da3-small` | 0.08B | 快速测试 | ~2GB |
| `da3-base` | 0.12B | 轻量应用 | ~3GB |
| `da3-large` | 0.35B | 标准应用 | ~6GB |
| `da3-large-1.1` | 0.35B | 标准应用(改进) | ~6GB |
| `da3-giant-1.1` | 1.15B | 高质量 | ~16GB |
| `da3nested-giant-large` | 1.40B | 度量深度 | ~20GB |

## STAC模式对比

### 1. Full模式 (全注意力)
```python
mode="full", streaming=False
```
- 简单直接,适合短序列 (< 10帧)
- 显存随帧数线性增长
- 不使用KV-cache

### 2. Window KV模式 (H2O剪枝)
```python
mode="window_kv", streaming=True,
window_size=6, chunk_size=2, hh_size=2
```
- 维护Heavy-hitter + 最近帧的KV缓存
- 显存有界增长,适合长序列
- 自动剪枝不重要的token

### 3. Window Chunk Merge模式 (STAC Voxel)
```python
mode="stac", streaming=True,  # 自动扩展
window_size=4, chunk_size=2,
voxel_size=0.05, voxel_num=4096,
conf_threshold=2.0
```
- 使用体素化压缩3D点云
- 空间缓存,支持检索
- 最适合超长序列 (100+帧)

## 关键配置参数

### Window KV模式参数
- `window_size`: 窗口大小 (帧数)
- `chunk_size`: 每步处理的帧数
- `hh_size`: Heavy-hitter缓存大小
- `temperature`: 注意力温度 (默认0.9)

### Window Chunk Merge模式参数
- `window_size`: 窗口大小
- `chunk_size`: 块大小
- `voxel_size`: 体素大小 (米),默认0.05
- `voxel_num`: 初始体素数量,默认4096
- `conf_threshold`: 置信度阈值,默认2.0
- `retrieval_size`: 检索缓存大小 (-1=始终检索)
- `return_buf`: 返回缓冲区内容
- `voxel_backend`: "cuda" 或 "python"
- `allocator`: "segment", "slab", 或 "static"

## 性能优化建议

### 1. 显存优化
```python
# 启用CPU offload (适合超长序列)
kv_kwargs = {
    "enable_alloc_cpu": True,
    "gpu_threshold_gb": 10.0,  # GPU显存上限
    "cold_frame_threshold": 5,  # 冷帧阈值
}
```

### 2. 速度优化
```python
# 使用CUDA后端 (如果可用)
voxel_backend = "cuda"

# 增大chunk_size (如果显存允许)
chunk_size = 4  # 默认1

# 禁用不必要的输出
cam_cache_update = False  # 不更新camera head缓存
```

### 3. 精度优化
```python
# 降低voxel_size提高精度
voxel_size = 0.02  # 默认0.05

# 增加voxel_num
voxel_num = 8192  # 默认4096

# 降低置信度阈值保留更多点
conf_threshold = 1.0  # 默认2.0
```

## 输出格式说明

### predictions字典
```python
{
    "pose_enc": torch.Tensor,        # [B, S, 9] 姿态编码
    "depth": torch.Tensor,           # [B, S, H, W, 1] 深度图
    "depth_conf": torch.Tensor,      # [B, S, H, W] 深度置信度
    "world_points": torch.Tensor,    # [B, S, H, W, 3] 世界坐标
    "world_points_conf": torch.Tensor,  # [B, S, H, W] 点置信度
    "images": torch.Tensor,          # [B, S, 3, H, W] 输入图像
    "timing": {                      # 性能统计
        "aggregator_infer_time": float,  # 推理时间 (ms)
        "kv_pruning_time": float,        # 剪枝时间 (ms)
        "kv_position_time": float,       # 位置更新时间 (ms)
        "kv_retrieval_time": float,      # 检索时间 (ms)
        "infer_fps": float,              # FPS
    },
    "merger": {                      # KV缓存统计
        "Memory(MB)": {...},          # 显存使用详情
        "hyperparameters": {...},      # 使用的超参数
    },
    "mode": str,                     # 使用的模式
    "streaming": bool,               # 是否流式
    "effective_config": {...},        # 实际使用的配置
}
```

## 已知限制

1. **Camera Head**: DA3没有独立的camera head,pose通过backbone直接预测
2. **Aggregated Tokens**: DA3使用占位符`[None]`,不维护token缓存
3. **Pose Encoding**: 使用简化编码 (旋转矩阵前2行 + 平移)
4. **World Points**: 通过深度反投影计算,可能与CausalVGGT的point head有差异

## 调试技巧

### 1. 启用详细日志
```bash
export VERBOSE=1
python main.py --scene_dir /path/to/scene ...
```

### 2. 检查KV缓存状态
```python
kv_mgr = model.model.backbone.pretrained.kv_manager
if kv_mgr:
    info = kv_mgr.get_info()
    print(f"KV cache size: {info['kvcache_size']}")
    print(f"KV cache used: {info['kvcache_used']:.0f} MB")
```

### 3. 可视化缓存统计
StreamSession会自动显示Rich表格:
```
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Time(ms)   ┃ agg=15.2 | ret=3.1 | pos=1.5 | ...   ┃
┃ Cache(MB)  ┃ temporal=120 | spatial=45 | ...       ┃
┃ GPU(MB)    ┃ allocated=2048 | reserved=2560        ┃
┗━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

## 故障排除

### 问题1: OOM (显存不足)
**解决方案**:
- 减小`window_size`和`chunk_size`
- 启用CPU offload
- 使用更小的模型 (da3-small/base)
- 降低输入分辨率

### 问题2: 输出格式不匹配
**解决方案**:
- 检查是否使用`streaming=True`
- 确认`model_type="da3"`已正确设置
- 查看`_convert_to_stac_format`方法的输出

### 问题3: KV缓存未更新
**解决方案**:
- 确认`register_kv_mgr`已调用
- 检查`SparseAttention`层是否正确替换
- 验证`layer_idx`是否正确设置

## 后续扩展

### 1. 添加Camera Head支持 (可选)
```python
# 为DA3添加独立的camera head (类似CausalVGGT)
from causalvggt.heads.camera_head import CameraHead

model.camera_head = CameraHead(...)
model.enable_camera = True
```

### 2. 集成CUDA加速
```python
# 使用attn-cuda和merger-cuda
# 需要编译CUDA扩展
voxel_backend = "cuda"
attn_backend = "cuda"
```

### 3. 训练支持
```python
# DA3+STAC端到端微调
# 需要修改训练循环支持流式模式
model.train()
for batch in dataloader:
    output = model(batch["images"], streaming=True, ...)
    loss = compute_loss(output, batch["targets"])
    loss.backward()
```

## 参考文档

- [DA3_INTEGRATION_PLAN.md](./DA3_INTEGRATION_PLAN.md) - 详细集成计划
- [STAC README.md](./README.md) - STAC框架文档
- [DA3 README.md](./depth-anything-3/README.md) - DA3原始文档

## 联系与支持

如有问题或建议,请提交Issue或PR。
