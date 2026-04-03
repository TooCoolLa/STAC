# STAC — Project Context

## 项目概述

**STAC** (Plug-and-Play Spatio-Temporal Aware Cache Compression) 是一个用于长视频流式 3D 重建的 KV-cache 压缩框架。它将已驱逐的 KV-cache token 压缩到时空体素内存中，并按需检索相关 pivot，实现有界内存的长程空间推理。

- **论文**: [arXiv 2603.20284](https://arxiv.org/abs/2603.20284)
- **支持的骨干网络**: STream3R, StreamVGGT, Depth Anything 3 (DA3)
- **核心技术**: KV-cache 管理 + 3D 体素池 + Heavy-Hitter 评分 + CUDA 加速注意力/合并

## 项目结构

```
STAC/
├── main.py                    # 最小推理入口
├── model_wrapper.py           # load_model() / run_model() 公共 API
├── stream_session.py          # 逐块流式推理会话 (StreamSession)
├── requirements.txt           # Python 依赖
├── QWEN.md                    # 本项目上下文文档
├── DA3_INTEGRATION_PLAN.md    # DA3 接入改造计划
│
├── stac/                      # STAC KV-cache 压缩核心 (plug-and-play)
│   ├── kv_manager.py          #   滑动窗口 KV 缓存管理
│   ├── h2o.py                 #   Heavy-Hitter 评分机制
│   ├── stac_voxel.py          #   体素池: 驱逐 -> 合并 -> 检索
│   ├── merger.py              #   Token 合并操作
│   ├── voxel.py               #   体素网格量化
│   ├── allocator.py           #   内存分配器 (static/slab/segment)
│   └── flash_attn_triton.py   #   Triton 注意力内核
│
├── causalvggt/                # Causal-VGGT 适配器
│   ├── models/                #   CausalVGGT 主模型 + CausalAggregator
│   ├── layers/                #   SparseAttention 等层
│   ├── heads/                 #   CameraHead, DPTHead
│   └── utils/                 #   姿态编码、几何工具
│
├── depth-anything-3/          # DA3 源码 (已集成改造)
│   └── src/depth_anything_3/
│       ├── model/dinov2/
│       │   ├── layers/
│       │   │   ├── attention.py   ← +SparseAttention 类
│       │   │   └── block.py       ← forward(**kwargs) 透传
│       │   └── vision_transformer.py  ← +KV manager 接口
│       ├── api.py
│       └── configs/
│
├── eval/                      # 评估脚本
│   ├── long_recon/            #   3D 重建评估 (launch.py + run.sh)
│   ├── cam_pose/              #   相机姿态估计评估
│   ├── video_depth/           #   视频深度估计评估
│   └── utils/                 #   通用工具 (图像加载、指标等)
│
├── attn-cuda/                 # 自定义 CUDA 注意力扩展 (可选)
├── merger-cuda/               # 自定义 CUDA 体素合并扩展 (可选)
├── assets/                    # 论文图片
└── ckpt/                      # 模型权重目录 (需自行下载)
    ├── stream3r/
    ├── streamvggt/
    └── da3/
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.11, CUDA/C++ |
| 深度学习 | PyTorch 2.7+, FlashAttention, Triton |
| 框架 | Lightning, Transformers, Safetensors |
| 评估 | Open3D, evo, scikit-image, scipy |
| CUDA 扩展 | 自定义 attention kernel + merger kernel |

## 环境搭建

```bash
conda create -n stac python=3.11 cmake=3.14.0 -y
conda activate stac

# 安装 PyTorch (以 CUDA 12.8 为例)
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# 安装依赖
pip install -r requirements.txt

# 可选: CUDA 扩展
pip install -e merger-cuda --no-build-isolation
pip install -e attn-cuda --no-build-isolation
```

## 模型准备

将骨干网络权重放在 `ckpt/{stream3r|streamvggt|da3}/` 目录下:

```bash
# STream3R / StreamVGGT
mkdir -p ckpt/stream3r && hf download yslan/STream3R --local-dir ckpt/stream3r
mkdir -p ckpt/streamvggt && hf download lch01/StreamVGGT --local-dir ckpt/streamvggt

# DA3 全系列
mkdir -p ckpt/da3 && hf download depth-anything/DA3-GIANT-1.1 --local-dir ckpt/da3/DA3-GIANT-1.1
```

支持的文件格式: `.safetensors`, `.pt`, `.pth`

## 运行与推理

### 快速推理

```bash
# STAC 模式 (推荐, 有界内存, CausalVGGT 骨干)
python main.py --scene_dir /path/to/scene --mode stac

# STAC 模式 + DA3 骨干 (最强深度估计)
python main.py --scene_dir /path/to/scene --model_name da3 --base_model da3-giant --mode stac

# 全注意力基线 (无流式, 高内存)
python main.py --scene_dir /path/to/scene --mode full

# 自定义 STAC 配置
python main.py --scene_dir /path/to/scene \
  --base_model stream3r --streaming \
  --mode window_chunk_merge \
  -win 4 -ck 4 -hh 2 -ret_sz 2 -ret_buf
```

### Python API

```python
import torch
from model_wrapper import load_model, run_model
from eval.utils.image import load_scene_images

device = "cuda"
dtype = torch.bfloat16

images = load_scene_images(Path("data/scene/images"), size=518)

# CausalVGGT 骨干
model = load_model("causalvggt", base_model="stream3r", device=device)

# 或 DA3 骨干 (最强深度估计)
model = load_model("da3", base_model="da3-giant", device=device)

with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    predictions = run_model(
        model=model, images=images, model_name="causalvggt",  # 或 "da3"
        mode="stac", streaming=True, dtype=dtype, device=device, pinned=[0],
    )
# predictions 包含: extrinsic, intrinsic, depth, depth_conf,
#                   world_points, world_points_conf, timing, merger
```

### 评估任务

```bash
# 3D 重建
python eval/long_recon/launch.py --dataset_type NRGBD --scene_name complete_kitchen \
  --model_name causalvggt --base_model stream3r --mode stac --streaming

# 相机姿态
python eval/cam_pose/launch.py --dataset_type tum \
  --model_name causalvggt --base_model stream3r --mode stac --streaming

# 视频深度 (两步)
python eval/video_depth/launch.py --eval_dataset sintel \
  --model_name causalvggt --base_model stream3r --mode stac --streaming
python eval/video_depth/eval_depth.py --align scale
```

批量运行脚本: `eval/*/run.sh`

## 支持的骨干模型

| 骨干 | 来源 | 参数量 | STAC 支持 | 特点 |
|------|------|--------|-----------|------|
| **STream3R** | 中科大 | ~350M | ✅ 原生 | 默认骨干, 3D 重建优化 |
| **StreamVGGT** | Meta→社区 | ~350M | ✅ 原生 | 几何一致性好 |
| **DA3-Small** | 字节跳动 | 80M | ✅ 已集成 | 最小 DA3 模型 |
| **DA3-Base** | 字节跳动 | 120M | ✅ 已集成 | 轻量级 DA3 |
| **DA3-Large** | 字节跳动 | 350M | ✅ 已集成 | 平衡性能/速度 |
| **DA3-Giant** | 字节跳动 | 1.15B | ✅ 已集成 | 最强单模型 (any-view) |
| **DA3Nested-Giant-Large** | 字节跳动 | 1.40B | ✅ 已集成 | 最强 (any-view + metric 深度) |

### DA3 改造要点

DA3 通过**方案 A (最小侵入)** 集成:
- `SparseAttention` 类注入到 DA3 的 global 注意力层
- `Block.forward` 添加 `**kwargs` 透传 `kv_manager`
- `DinoVisionTransformer` 新增 `register_kv_mgr()`, `prune_kv_mgr()`, `retrieve_kv_mgr()` 等方法
- `stream_session.py` 适配 DA3 backbone (`model_type` 分发)

## 核心架构与数据流

### 注意力模式

| 模式 | 描述 | KVManager |
|------|------|-----------|
| `full` | 标准全量注意力 (无 KV 管理) | — |
| `causal` | 因果全序列 KV 缓存 | KVManager |
| `window_kv` | 滑动窗口 KV 缓存 | KVManager |
| `window_chunk_merge` | 滑动窗口 + 体素池 | STACVoxelKV |
| `stac` | `window_chunk_merge` 的别名, 带推荐默认值 | STACVoxelKV |

### 数据流

```
图像 [S, 3, H, W] → load_model() → run_model()
    ├── 非流式: model(images, mode=mode)
    └── 流式: StreamSession.pipeline()
        ├── 逐块前向传播 → model(chunk)
        ├── CameraHead 推理 (姿态编码, CausalVGGT 专属)
        ├── 获取点云图 → get_pointmap()
        ├── 更新 KV 位置 → update_kv_mgr_pos()
        ├── 检索 pivot → retrieve_kv_mgr()
        └── 驱逐/合并 → prune_kv_mgr() → 写入体素池
```

### 关键组件

| 组件 | 文件 | 职责 |
|------|------|------|
| `load_model()` / `run_model()` | `model_wrapper.py` | 公共 API, 加载模型, 分发到流式/非流式路径 |
| `StreamSession` | `stream_session.py` | 逐块流式推理, 管理 KV 缓存生命周期 |
| `CausalVGGT` | `causalvggt/models/vggt.py` | 顶层模型: CausalAggregator + CameraHead + DPTHead |
| `CausalAggregator` | `causalvggt/models/aggregator.py` | 24层 ViT-L 骨干, 交替注意力, 承载 kv_manager |
| `SparseAttention` | `causalvggt/layers/attention.py` | 注意力层, 挂钩 KVManager 进行 sparse decode |
| `KVManager` | `stac/kv_manager.py` | 滑动窗口 KV 缓存, 预分配 GPU 张量 |
| `HeavyHittersKV` | `stac/h2o.py` | 扩展 KVManager, 添加 H2O 评分剪枝 |
| `STACVoxelKV` | `stac/stac_voxel.py` | 扩展 H2O, 驱逐 token 到体素池, 按需检索 |
| `DA3 DinoVisionTransformer` | `depth-anything-3/.../vision_transformer.py` | DA3 骨干, 已改造支持 KV manager |
| `DA3 SparseAttention` | `depth-anything-3/.../attention.py` | DA3 版 SparseAttention, 与 STAC 兼容 |

### `--mode stac` 默认参数

`run_model()` 中 `stac` 模式会自动展开为以下默认配置:

```python
window_size=4, chunk_size=4, hh_size=2, retrieval_size=2,
return_buf=True, voxel_backend="cuda", allocator="segment"
```

## 环境变量

| 变量 | 作用 |
|------|------|
| `VERBOSE=1` | 打印每帧 KV 统计信息 |
| `MERGER_MEM_PROFILE=1` | 报告体素清理期间的 CUDA 内存碎片 |

## 数据集布局

场景目录需包含 `images/` 子文件夹, 内含 `.png` 或 `.jpg` 文件:

```
data/<dataset>/<scene>/images/*.png
# 例如: data/7scenes/chess/images/
#       data/neural_rgbd/whiteroom/images/
```

支持的数据集: `7scenes`, `neural_rgbd`, `DTU`, `tum`, `scannet`, `sintel`, `bonn`, `kitti`

## 编码约定

- **日志**: 使用 `logging` 模块, 通过 `RichHandler` 格式化输出
- **进度显示**: 使用 Rich 的 `Progress` + `Live` 组件显示实时统计
- **类型提示**: 关键函数有类型标注 (如 `torch.Tensor`, `dict`)
- **断言检查**: 关键路径使用 `assert` 验证 batch size=1 等约束
- **上下文管理**: 推理使用 `torch.no_grad()` + `torch.amp.autocast()` 包裹
- **配置传递**: 通过 `**kwargs` 传递 STAC 参数, 在 `run_model()` 中做默认值填充
- **多骨干适配**: 使用 `model_type` 参数在 `StreamSession` 中分发到不同代码路径

## DA3 集成关键要点 (2026-04-03 修复记录)

### DA3 模型结构差异
- `NestedDepthAnything3Net` 的 backbone 路径: `model.model.da3.backbone.pretrained`（非 `model.model.backbone.pretrained`）
- DA3 forward 参数: `image=` 而非 `images=`，不接受 `camera_head_kv_cache_list` / `is_anchor_exist`
- DA3 输出 Dict 是 `addict.Dict`，键值通过属性访问（`output.extrinsics`），`.get()` 可能返回 None
- DA3 没有 `camera_head` 和 `point_head`，pose 由 `cam_dec` 直接输出

### DA3 位姿累积（关键修复）
DA3 每个 chunk 独立推理，输出相对位姿。必须跨 chunk 累积：
- 使用 `_accumulate_da3_poses()` 方法，通过 c2w 矩阵复合实现
- 第一个 chunk 直接使用，后续 chunk 计算相对变换后与上一个 chunk 末帧 c2w 复合
- **必须同时更新 `outputs["pose_enc"]`**，否则 `pushback_prediction` 保存的是旧值
- 需要在 `_convert_to_stac_format` 中保留 `extrinsics` 和 `intrinsics` 供外部使用

### DA3 计时修复
- DA3 forward 需要手动添加 CUDA Event 计时（`aggregator_infer_time`）
- `_convert_to_stac_format` 必须复制 `timing` dict 到 `stac_output`
- 修复前 FPS 报告 1224（虚高），修复后 25.2（真实值）

### 常见陷阱
1. **`.get()` vs `getattr()`**: DA3 的 `addict.Dict` 输出需要用 `getattr(output, "key", None)` 访问
2. **bfloat16 → numpy**: 必须 `.cpu().float().numpy()`，不能直接 `.cpu().numpy()`
3. **patch_size tuple**: DA3 的 `patch_size` 是 `(14, 14)`，需要取 `[0]`
4. **KV manager 方法签名**: DA3 的 `update_kv_mgr_pos` 需要 flatten `[S, T_per_frame, 3]` → `[S*T_per_frame, 3]`
5. **`prune_kv`/`retrieve_kv` 不接受 `timing`**: 需要在 wrapper 中手动计时
6. **xformers SwiGLU 版本不匹配**: 添加运行时检测，失败时回退到纯 PyTorch 实现
