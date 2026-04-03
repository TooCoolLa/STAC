# STAC — Project Context

## 项目概述

**STAC** (Plug-and-Play Spatio-Temporal Aware Cache Compression) 是一个用于长视频流式 3D 重建的 KV-cache 压缩框架。它将已驱逐的 KV-cache token 压缩到时空体素内存中，并按需检索相关 pivot，实现有界内存的长程空间推理。

- **论文**: [arXiv 2603.20284](https://arxiv.org/abs/2603.20284)
- **支持的后端骨干网络**: STream3R (`stream3r`) 和 StreamVGGT (`streamvggt`)
- **核心技术**: KV-cache 管理 + 3D 体素池 + Heavy-Hitter 评分 + CUDA 加速注意力/合并

## 项目结构

```
STAC/
├── main.py                    # 最小推理入口
├── model_wrapper.py           # load_model() / run_model() 公共 API
├── stream_session.py          # 逐块流式推理会话 (StreamSession)
├── requirements.txt           # Python 依赖
├── QWEN.md                    # 本项目上下文文档
├── CLAUDE.md                  # Claude Code 指引文档
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
    └── streamvggt/
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

将骨干网络权重放在 `ckpt/{stream3r|streamvggt}/` 目录下:

```bash
mkdir -p ckpt/stream3r && hf download yslan/STream3R --local-dir ckpt/stream3r
mkdir -p ckpt/streamvggt && hf download lch01/StreamVGGT --local-dir ckpt/streamvggt
```

支持的文件格式: `.safetensors`, `.pt`, `.pth`

## 运行与推理

### 快速推理

```bash
# STAC 模式 (推荐, 有界内存)
python main.py --scene_dir /path/to/scene --mode stac

# 全注意力基线 (无流式, 高内存)
python main.py --scene_dir /path/to/scene --mode full

# 使用 StreamVGGT 骨干
python main.py --scene_dir /path/to/scene --base_model streamvggt --mode stac

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
model = load_model("causalvggt", base_model="stream3r", device=device)

with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    predictions = run_model(
        model=model, images=images, model_name="causalvggt",
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
        ├── CameraHead 推理 (姿态编码)
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
| `VoxelKVMerger` | `stac/merger.py` | 长期体素存储, 合并 buffer token 到 pivot |
| `BinaryVoxel` | `stac/voxel.py` | 3D 世界坐标量化到体素 ID |
| 分配器 | `stac/allocator.py` | static / slab / segment 内存分配 |

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
