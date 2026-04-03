# STAC流式推理使用指南

## 📋 快速开始

### 前置准备

确保图片按以下结构组织：

```
/path/to/your/scene/
├── images/              # 可选：放在images子目录
│   ├── frame_001.png
│   ├── frame_002.png
│   └── ...
```

或直接放置图片：

```
/path/to/your/scene/
├── frame_001.png
├── frame_002.png
└── ...
```

支持格式：`.png`, `.jpg`, `.jpeg`

---

## 🚀 基础用法

### 最简单命令

```bash
python run_streaming_inference.py \
    --scene_dir /path/to/scene \
    --output_dir ./output
```

---

## 🎯 针对不同场景的推荐配置

### 场景1: 60-100帧（常用）

**适用：短视频、小规模场景**

```bash
# 使用DA3模型（推荐）
python run_streaming_inference.py \
    --scene_dir /path/to/scene \
    --output_dir ./output \
    --model_name da3 \
    --base_model da3-large \
    --window_size 8 \
    --chunk_size 4 \
    --hh_size 4 \
    --voxel_size 0.03 \
    --voxel_num 8192 \
    --save_depth_vis

# 使用CausalVGGT模型
python run_streaming_inference.py \
    --scene_dir /path/to/scene \
    --output_dir ./output \
    --model_name causalvggt \
    --base_model stream3r \
    --window_size 8 \
    --chunk_size 4 \
    --hh_size 4 \
    --voxel_size 0.03 \
    --voxel_num 8192
```

**预估：**
- ⏱️ 时间：2-5秒
- 💾 显存：10-14GB
- 📁 输出：~150MB（depth.npy）

---

### 场景2: 100-500帧（中等）

**适用：中等长度视频**

```bash
python run_streaming_inference.py \
    --scene_dir /path/to/scene \
    --output_dir ./output \
    --window_size 6 \
    --chunk_size 4 \
    --hh_size 3 \
    --retrieval_size 2 \
    --voxel_size 0.04 \
    --voxel_num 4096
```

**预估：**
- ⏱️ 时间：15-60秒
- 💾 显存：12-16GB
- 📁 输出：~750MB（500帧）

---

### 场景3: 500-3000帧（大规模）

**适用：长视频、完整场景重建**

```bash
python run_streaming_inference.py \
    --scene_dir /path/to/scene \
    --output_dir ./output_3000 \
    --window_size 4 \
    --chunk_size 4 \
    --hh_size 2 \
    --retrieval_size 2 \
    --voxel_size 0.05 \
    --voxel_num 4096 \
    --dtype bf16 \
    --voxel_backend cuda
```

**预估（3000帧）：**
- ⏱️ 时间：3-5分钟
- 💾 显存：10-15GB
- 📁 输出：~4.7GB（depth.npy）

---

### 场景4: 3000+帧（超大规模）

**适用：超长视频、长序列SLAM**

#### 方案A：直接流式（3000-5000帧）

```bash
python run_streaming_inference.py \
    --scene_dir /path/to/scene \
    --output_dir ./output \
    --window_size 4 \
    --chunk_size 2 \
    --hh_size 2 \
    --voxel_size 0.05 \
    --voxel_num 4096
```

#### 方案B：分块处理（5000+帧，推荐）

```python
# process_ultra_long.py
from run_streaming_inference import process_very_long_sequence

process_very_long_sequence(
    scene_dir="/path/to/very_long_scene",
    output_dir="./output_chunked",
    chunk_frames=1000,      # 每块1000帧
    overlap_frames=50,      # 块间重叠50帧
    window_size=4,
    chunk_size=2,
    voxel_size=0.05,
)
```

---

## 📖 核心参数详解

### 流式推理参数

| 参数 | 缩写 | 含义 | 60-100帧 | 3000帧 |
|------|------|------|----------|--------|
| `--window_size` | `-win` | 上下文窗口大小（帧数） | 8 | 4 |
| `--chunk_size` | `-ck` | 每步处理帧数 | 4 | 2-4 |
| `--hh_size` | `-hh` | Heavy-hitter缓存大小 | 4 | 2 |
| `--retrieval_size` | `-ret_sz` | 检索缓存大小（-1=始终检索） | 0 | 2 |

### 体素化参数

| 参数 | 含义 | 高精度 | 标准 | 快速 |
|------|------|--------|------|------|
| `--voxel_size` | 体素大小（米） | 0.02 | 0.05 | 0.08 |
| `--voxel_num` | 初始体素数量 | 8192 | 4096 | 2048 |
| `--conf_threshold` | 置信度阈值 | 1.0 | 2.0 | 3.0 |

### 其他参数

| 参数 | 含义 | 推荐值 |
|------|------|--------|
| `--dtype` | 数据类型 | `bf16`（4090）/ `fp16`（其他GPU） |
| `--voxel_backend` | 体素后端 | `cuda`（快）/ `python` |
| `--max_frames` | 最大处理帧数 | `None`（全部） |
| `--save_depth_vis` | 保存深度可视化 | 按需启用 |

---

## 🎨 输出结果

### 目录结构

```
output/
├── results/
│   ├── depth.npy              # 深度图 [1, S, H, W, 1]
│   ├── depth_conf.npy         # 深度置信度 [1, S, H, W]
│   ├── pose_enc.npy           # 姿态编码 [1, S, 9]
│   ├── extrinsic.npy          # 相机外参 [1, S, 3, 4]
│   ├── intrinsic.npy          # 相机内参 [1, S, 3, 3]
│   └── world_points.npy       # 3D世界坐标 [1, S, H, W, 3]（可选）
├── depth_visualization/       # 深度可视化（如果启用--save_depth_vis）
│   ├── depth_00000.png
│   ├── depth_00010.png
│   └── ...
└── statistics.json            # 性能统计
```

### 读取结果示例

```python
import numpy as np

# 加载深度
depth = np.load("output/results/depth.npy")  # shape: [1, S, H, W, 1]

# 加载相机位姿
extrinsic = np.load("output/results/extrinsic.npy")  # shape: [1, S, 3, 4]
intrinsic = np.load("output/results/intrinsic.npy")  # shape: [1, S, 3, 3]

# 查看统计
import json
with open("output/statistics.json") as f:
    stats = json.load(f)
    print(f"FPS: {stats['performance']['fps']}")
```

---

## 🔧 可用模型

### DA3系列

| 模型 | 参数量 | 适用场景 | 显存需求 |
|------|--------|---------|---------|
| `da3-small` | 0.08B | 快速测试 | ~2GB |
| `da3-base` | 0.12B | 轻量应用 | ~3GB |
| `da3-large` | 0.35B | 标准应用 | ~6GB |
| `da3-large-1.1` | 0.35B | 标准应用（改进） | ~6GB |
| `da3-giant-1.1` | 1.15B | 高质量 | ~16GB |

### CausalVGGT系列

| 模型 | 参数量 | 适用场景 | 显存需求 |
|------|--------|---------|---------|
| `stream3r` | ~0.3B | 标准应用 | ~8GB |
| `streamvggt` | ~0.6B | 高质量 | ~12GB |

---

## ⚡ 性能优化

### 提高速度

```bash
# 1. 增大chunk_size
--chunk_size 6

# 2. 使用CUDA后端
--voxel_backend cuda --attn_backend cuda

# 3. 减少检索频率
--retrieval_size 0
```

### 降低显存

```bash
# 1. 减小窗口
--window_size 4 --chunk_size 2

# 2. 降低分辨率
--size 224

# 3. 启用CPU offload（在代码中）
stac_kwargs = {
    "enable_alloc_cpu": True,
    "gpu_threshold_gb": 16.0,
}
```

### 提高精度

```bash
# 1. 减小体素
--voxel_size 0.02 --voxel_num 8192

# 2. 增大窗口
--window_size 8 --hh_size 4

# 3. 降低置信度阈值
--conf_threshold 1.0
```

---

## 🎛️ 监控推理过程

运行时会自动显示实时统计：

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Window Mode  • 50/100  • 0:00:15 • 0:00:10            ┃
┣━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Time(ms)   ┃ agg=15.2  ret=3.1  pos=1.5  merge=2.8    ┃
┃ Cache(MB)  ┃ temporal=120  spatial=45  voxel=256/512  ┃
┃ GPU(MB)    ┃ allocated=8192  reserved=10240           ┃
┗━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

**含义：**
- **Time**: 各阶段耗时（聚合/检索/位置更新/合并）
- **Cache**: 缓存使用情况
- **GPU**: 显存占用

另开终端监控显存：
```bash
watch -n 1 nvidia-smi
```

---

## ❓ 常见问题

### Q1: 显存不足怎么办？

```bash
# 减小配置
--window_size 4 --chunk_size 2 --voxel_num 2048 --size 224
```

### Q2: 如何提高推理速度？

```bash
# 增大chunk_size
--chunk_size 6 --voxel_backend cuda
```

### Q3: 输出文件太大？

只保存必要文件，编辑 `run_streaming_inference.py` 注释掉不需要的保存逻辑。

### Q4: 图片顺序不对？

确保图片文件名可以正确排序：
```
frame_001.png  ✓
frame_1.png    ✗ (会导致1, 10, 11, ..., 2, 20...的顺序)
```

---

## 📝 完整示例

### 示例1: 处理60帧视频（日常使用）

```bash
python run_streaming_inference.py \
    --scene_dir ./data/video_60frames \
    --output_dir ./results/video_60 \
    --model_name da3 \
    --base_model da3-large \
    --window_size 8 \
    --chunk_size 4 \
    --hh_size 4 \
    --voxel_size 0.03 \
    --voxel_num 8192 \
    --save_depth_vis
```

### 示例2: 处理3000帧长视频

```bash
python run_streaming_inference.py \
    --scene_dir ./data/long_video \
    --output_dir ./results/long_3000 \
    --window_size 4 \
    --chunk_size 4 \
    --hh_size 2 \
    --retrieval_size 2 \
    --voxel_size 0.05 \
    --voxel_num 4096 \
    --dtype bf16 \
    --voxel_backend cuda
```

### 示例3: 快速测试

```bash
python run_streaming_inference.py \
    --scene_dir ./data/test \
    --output_dir ./results/test \
    --max_frames 30 \
    --window_size 6 \
    --chunk_size 4
```

---

## 🎓 进阶使用

### Python API调用

```python
from run_streaming_inference import run_inference
import argparse

# 创建配置
args = argparse.Namespace(
    scene_dir="/path/to/scene",
    output_dir="./output",
    model_name="da3",
    base_model="da3-large",
    window_size=8,
    chunk_size=4,
    hh_size=4,
    retrieval_size=0,
    voxel_size=0.03,
    voxel_num=8192,
    conf_threshold=2.0,
    voxel_backend="cuda",
    dtype="bf16",
    max_frames=None,
    save_depth_vis=True,
)

# 运行推理
predictions = run_inference(args, device="cuda", dtype=torch.bfloat16)

# 访问结果
print(f"Depth shape: {predictions['depth'].shape}")
print(f"FPS: {predictions['timing']['infer_fps']}")
```

---

## 📚 更多资源

- [DA3_STAC_INTEGRATION_GUIDE.md](./DA3_STAC_INTEGRATION_GUIDE.md) - DA3集成详细指南
- [DA3_STAC_DEVELOPMENT_SUMMARY.md](./DA3_STAC_DEVELOPMENT_SUMMARY.md) - 开发总结
- [main.py](./main.py) - 原始STAC推理脚本

---

## 💡 提示

1. **首次使用**：先用 `--max_frames 30` 测试，确认流程正常
2. **4090用户**：直接用 `--dtype bf16` 和 `--voxel_backend cuda` 获得最佳性能
3. **磁盘空间**：长序列输出可能很大，确保有足够空间
4. **监控显存**：运行时用 `nvidia-smi` 监控

---

**有问题？查看日志输出或查阅详细文档！** 🚀
