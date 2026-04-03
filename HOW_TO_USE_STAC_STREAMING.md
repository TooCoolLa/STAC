# STAC 流式推理长序列图片完整指南

## 📋 目录结构准备

你的图片目录应该这样组织:

```
/path/to/your/scene/
└── images/
    ├── frame_0001.png
    ├── frame_0002.png
    ├── frame_0003.png
    ├── ...
    └── frame_1000.png
```

或者直接在目录下放置图片文件(支持 `.png`, `.jpg`, `.jpeg`)。

## 🚀 方法一: 命令行使用 (最简单)

### 1. 使用 CausalVGGT 骨干 (原始STAC)

```bash
# 基础STAC流式推理 (适合100+帧长序列)
python main.py --scene_dir /path/to/scene \
    --base_model stream3r \
    --mode stac \
    --streaming \
    --size 518

# 高级配置: 调整缓存参数
python main.py --scene_dir /path/to/scene \
    --base_model streamvggt \
    --mode stac \
    --streaming \
    --window_size 4 \
    --chunk_size 4 \
    --hh_size 2 \
    --retrieval_size 2 \
    --voxel_size 0.05 \
    --voxel_num 4096 \
    --conf_threshold 2.0
```

### 2. 使用 DA3 骨干 (新集成)

```bash
# DA3流式推理 (需要先修改main.py支持da3)
python main.py --scene_dir /path/to/scene \
    --model_name da3 \
    --base_model da3-large \
    --mode stac \
    --streaming \
    --size 504
```

### 参数说明

| 参数 | 默认值 | 说明 | 长序列推荐值 |
|------|--------|------|-------------|
| `--mode` | stac | 推理模式 | `stac` (自动扩展为window_chunk_merge) |
| `--streaming` | False | 启用流式推理 | ✅ 必须启用 |
| `--window_size` | 0 | 窗口大小(帧数) | 4-8 |
| `--chunk_size` | 1 | 每步处理帧数 | 2-4 |
| `--hh_size` | 0 | Heavy-hitter缓存 | 2-4 |
| `--retrieval_size` | 0 | 检索缓存大小 | 2-4 或 -1(始终检索) |
| `--voxel_size` | 0.05 | 体素大小(米) | 0.03-0.08 |
| `--voxel_num` | 4096 | 初始体素数 | 4096-8192 |
| `--conf_threshold` | 2.0 | 置信度阈值 | 1.5-3.0 |
| `--voxel_backend` | cuda | 体素后端 | `cuda`(快) 或 `python` |

## 💻 方法二: Python API使用 (更灵活)

### 完整示例代码

```python
"""
流式推理长序列图片示例
支持CausalVGGT和DA3两种骨干网络
"""

import os
import sys
import torch
import numpy as np
from model_wrapper import load_model, run_model
from eval.utils.image import load_scene_images
from causalvggt.utils.pose_enc import pose_encoding_to_extri_intri
from causalvggt.utils.geometry import unproject_depth_map_to_point_map

def inference_long_sequence(
    scene_dir,
    model_name="causalvggt",  # 或 "da3"
    base_model="stream3r",    # 或 "da3-large"
    output_dir=None,
    **kwargs
):
    """
    对长序列图片进行流式推理
    
    Args:
        scene_dir: 场景目录路径
        model_name: 模型名称 ("causalvggt" 或 "da3")
        base_model: 骨干模型名称
        output_dir: 输出目录
        **kwargs: 其他配置参数
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    print(f"🚀 Loading model: {model_name} ({base_model})")
    print(f"📍 Device: {device}, dtype: {dtype}")
    
    # 1. 加载模型
    model = load_model(model_name, base_model=base_model, device=device)
    model.eval()
    
    # 2. 加载图片
    print(f"\n📸 Loading images from: {scene_dir}")
    image_size = 518 if model_name == "causalvggt" else 504
    images = load_scene_images(scene_dir, size=image_size).to(device)
    print(f"✅ Loaded {images.shape[0]} frames, shape: {images.shape}")
    
    # 3. 配置STAC参数
    stac_kwargs = {
        "window_size": kwargs.get("window_size", 4),
        "chunk_size": kwargs.get("chunk_size", 2),
        "hh_size": kwargs.get("hh_size", 2),
        "retrieval_size": kwargs.get("retrieval_size", 2),
        "return_buf": kwargs.get("return_buf", False),
        "voxel_size": kwargs.get("voxel_size", 0.05),
        "voxel_num": kwargs.get("voxel_num", 4096),
        "conf_threshold": kwargs.get("conf_threshold", 2.0),
        "temperature": kwargs.get("temperature", 0.9),
        "voxel_backend": kwargs.get("voxel_backend", "cuda"),
        "allocator": kwargs.get("allocator", "segment"),
        "timing": True,
    }
    
    print(f"\n⚙️  STAC Config:")
    print(f"   window_size={stac_kwargs['window_size']}")
    print(f"   chunk_size={stac_kwargs['chunk_size']}")
    print(f"   voxel_size={stac_kwargs['voxel_size']}m")
    print(f"   voxel_num={stac_kwargs['voxel_num']}")
    
    # 4. 运行流式推理
    print(f"\n🎬 Starting streaming inference...")
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
        predictions = run_model(
            model, images, model_name,
            mode="stac",
            streaming=True,
            dtype=dtype,
            device=device,
            **stac_kwargs,
        )
    
    # 5. 处理结果
    print(f"\n✅ Inference complete!")
    print(f"📊 Results:")
    print(f"   Depth shape: {predictions['depth'].shape}")
    print(f"   FPS: {predictions['timing']['infer_fps']:.1f}")
    
    # 解码位姿
    if model_name == "causalvggt":
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        
        # 计算世界坐标点
        depth_map = predictions["depth"]
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy().squeeze(0)
            extrinsic_np = extrinsic.cpu().numpy().squeeze(0)
            intrinsic_np = intrinsic.cpu().numpy().squeeze(0)
        
        world_points = unproject_depth_map_to_point_map(
            depth_map, extrinsic_np, intrinsic_np
        )
        print(f"   World points shape: {world_points.shape}")
    
    # 6. 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n💾 Saving results to: {output_dir}")
        
        # 保存深度图
        if isinstance(predictions["depth"], torch.Tensor):
            depth_np = predictions["depth"].cpu().numpy()
        else:
            depth_np = predictions["depth"]
        np.save(os.path.join(output_dir, "depth.npy"), depth_np)
        
        # 保存位姿
        if "pose_enc" in predictions:
            np.save(os.path.join(output_dir, "pose_enc.npy"), 
                   predictions["pose_enc"].cpu().numpy())
        
        # 保存外参和内参
        if "extrinsic" in predictions:
            np.save(os.path.join(output_dir, "extrinsic.npy"), 
                   predictions["extrinsic"].cpu().numpy() if isinstance(predictions["extrinsic"], torch.Tensor) else predictions["extrinsic"])
        if "intrinsic" in predictions:
            np.save(os.path.join(output_dir, "intrinsic.npy"), 
                   predictions["intrinsic"].cpu().numpy() if isinstance(predictions["intrinsic"], torch.Tensor) else predictions["intrinsic"])
        
        # 保存统计信息
        import json
        stats = {
            "num_frames": images.shape[0],
            "fps": predictions["timing"]["infer_fps"],
            "voxel_stats": predictions.get("merger", {}),
        }
        with open(os.path.join(output_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"✅ Results saved!")
    
    return predictions


def main():
    # 示例1: 使用CausalVGGT处理长序列
    predictions = inference_long_sequence(
        scene_dir="/path/to/your/long_video_scene",
        model_name="causalvggt",
        base_model="stream3r",
        output_dir="./output/causalvggt_results",
        window_size=4,
        chunk_size=2,
        voxel_size=0.05,
        voxel_num=4096,
    )
    
    # 示例2: 使用DA3处理长序列 (需要DA3模型)
    # predictions = inference_long_sequence(
    #     scene_dir="/path/to/your/long_video_scene",
    #     model_name="da3",
    #     base_model="da3-large",
    #     output_dir="./output/da3_results",
    #     window_size=6,
    #     chunk_size=3,
    #     voxel_size=0.04,
    #     voxel_num=8192,
    # )


if __name__ == "__main__":
    main()
```

## 🎯 方法三: 自定义长序列处理脚本

```python
"""
处理超长序列 (>500帧) 的优化脚本
支持帧采样、分块处理、断点续传等功能
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
from model_wrapper import load_model, run_model

def load_custom_images(scene_dir, size=518, max_frames=None, 
                       frame_skip=1, start_idx=0):
    """
    自定义图片加载,支持更多控制选项
    
    Args:
        scene_dir: 场景目录
        size: 图像大小
        max_frames: 最大帧数 (None=全部)
        frame_skip: 帧跳过间隔
        start_idx: 起始帧索引
    """
    # 查找所有图片
    img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']
    img_files = []
    for ext in img_extensions:
        img_files.extend(glob.glob(os.path.join(scene_dir, ext)))
        img_files.extend(glob.glob(os.path.join(scene_dir, 'images', ext)))
    
    img_files = sorted(list(set(img_files)))
    
    if not img_files:
        raise ValueError(f"No images found in {scene_dir}")
    
    # 应用帧过滤
    img_files = img_files[start_idx:]
    if frame_skip > 1:
        img_files = img_files[::frame_skip]
    if max_frames:
        img_files = img_files[:max_frames]
    
    print(f"Found {len(img_files)} images")
    
    # 加载并预处理
    images = []
    for img_path in img_files:
        img = Image.open(img_path).convert('RGB')
        # Resize
        w, h = img.size
        scale = size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Pad到固定大小
        new_img = Image.new('RGB', (size, size), (0, 0, 0))
        paste_x = (size - new_w) // 2
        paste_y = (size - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        # 转tensor
        img_tensor = torch.from_numpy(np.array(new_img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # H,W,C -> C,H,W
        images.append(img_tensor)
    
    images = torch.stack(images)  # [S, C, H, W]
    return images


def process_very_long_sequence(
    scene_dir,
    output_dir,
    chunk_frames=100,  # 每块处理100帧
    overlap_frames=10,  # 块间重叠10帧
    **stac_kwargs
):
    """
    处理超长序列的分块策略
    """
    device = "cuda"
    dtype = torch.bfloat16
    
    # 加载模型(只加载一次)
    model = load_model("causalvggt", base_model="stream3r", device=device)
    model.eval()
    
    # 加载所有图片
    images = load_custom_images(scene_dir, size=518)
    total_frames = images.shape[0]
    
    print(f"\n📊 Total frames: {total_frames}")
    print(f"🧩 Chunk size: {chunk_frames}, overlap: {overlap_frames}")
    
    # 分块处理
    all_predictions = []
    chunk_start = 0
    chunk_idx = 0
    
    while chunk_start < total_frames:
        chunk_end = min(chunk_start + chunk_frames, total_frames)
        
        print(f"\n{'='*60}")
        print(f"🎬 Processing chunk {chunk_idx}: frames {chunk_start}-{chunk_end}")
        print(f"{'='*60}")
        
        # 提取当前块
        chunk_images = images[chunk_start:chunk_end]
        
        # 流式推理
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            predictions = run_model(
                model, chunk_images, "causalvggt",
                mode="stac",
                streaming=True,
                dtype=dtype,
                device=device,
                **stac_kwargs,
            )
        
        # 保存当前块结果
        chunk_output_dir = os.path.join(output_dir, f"chunk_{chunk_idx:04d}")
        os.makedirs(chunk_output_dir, exist_ok=True)
        
        # 保存预测
        torch.save(predictions, os.path.join(chunk_output_dir, "predictions.pt"))
        
        # 保存深度图可视化
        if "depth" in predictions:
            depth = predictions["depth"]
            if isinstance(depth, torch.Tensor):
                depth = depth.cpu().numpy()
            
            # 保存为npy
            np.save(os.path.join(chunk_output_dir, "depth.npy"), depth)
            
            # 可选: 保存为可视化图片
            import matplotlib.pyplot as plt
            os.makedirs(os.path.join(chunk_output_dir, "depth_vis"), exist_ok=True)
            for i in range(depth.shape[1]):
                plt.imsave(
                    os.path.join(chunk_output_dir, "depth_vis", f"frame_{chunk_start+i:05d}.png"),
                    depth[0, i, :, :, 0] if len(depth.shape) == 5 else depth[0, i],
                    cmap='turbo'
                )
        
        all_predictions.append(predictions)
        
        # 移动到下一个块(考虑重叠)
        chunk_start = chunk_end - overlap_frames
        chunk_idx += 1
        
        # 清理显存
        torch.cuda.empty_cache()
    
    print(f"\n✅ All {chunk_idx} chunks processed!")
    print(f"💾 Results saved to: {output_dir}")
    
    return all_predictions


# 使用示例
if __name__ == "__main__":
    # 处理超长视频
    predictions = process_very_long_sequence(
        scene_dir="/path/to/very_long_video",
        output_dir="./output/chunked_results",
        chunk_frames=100,
        overlap_frames=10,
        window_size=4,
        chunk_size=2,
        voxel_size=0.05,
        conf_threshold=2.0,
    )
```

## 🔧 常见问题和解决方案

### Q1: 显存不足怎么办?

```python
# 方案1: 减小窗口和块大小
stac_kwargs = {
    "window_size": 2,  # 从4减到2
    "chunk_size": 1,   # 从2减到1
    "voxel_num": 2048, # 从4096减到2048
}

# 方案2: 启用CPU offload
stac_kwargs = {
    "enable_alloc_cpu": True,
    "gpu_threshold_gb": 8.0,
}

# 方案3: 降低输入分辨率
images = load_scene_images(scene_dir, size=224)  # 从518减到224

# 方案4: 使用更小的模型
model = load_model("causalvggt", base_model="stream3r")  # 而不是streamvggt
```

### Q2: 如何提高推理速度?

```python
# 方案1: 增大chunk_size (如果显存允许)
stac_kwargs = {
    "chunk_size": 4,  # 从2增到4
}

# 方案2: 使用CUDA后端
stac_kwargs = {
    "voxel_backend": "cuda",  # 而不是"python"
    "attn_backend": "cuda",
}

# 方案3: 减少检索频率
stac_kwargs = {
    "retrieval_size": -1,  # -1表示始终检索,改为0禁用或更大减少频率
}
```

### Q3: 如何提高精度?

```python
# 方案1: 减小体素大小
stac_kwargs = {
    "voxel_size": 0.02,  # 从0.05减到0.02 (更精细)
    "voxel_num": 8192,   # 增加体素数
}

# 方案2: 降低置信度阈值
stac_kwargs = {
    "conf_threshold": 1.0,  # 从2.0减到1.0 (保留更多点)
}

# 方案3: 使用更大的窗口
stac_kwargs = {
    "window_size": 8,  # 从4增到8
    "hh_size": 4,      # 增加heavy-hitter缓存
}
```

## 📊 监控推理过程

流式推理时会显示实时统计:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Window Mode  • 8/100  • 0:02:15 • 0:12:30                      ┃
┣━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Time(ms)   ┃ agg=15.2  ret=3.1  pos=1.5  evict&merge=2.8       ┃
┃ KV Cache   ┃ tokens=4096  mem=1024MB                            ┃
┃ GPU(MB)    ┃ alloc=2048  reserved=2560                          ┃
┗━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

- **Time**: 各阶段耗时(推理/检索/位置更新/剪枝合并)
- **KV Cache**: 缓存token数和显存占用
- **GPU**: 当前显存使用情况

## 💡 最佳实践建议

### 对于100帧以内的序列:
```bash
python main.py --scene_dir /path/to/scene \
    --mode stac --streaming \
    --window_size 4 --chunk_size 2
```

### 对于100-500帧的序列:
```bash
python main.py --scene_dir /path/to/scene \
    --mode stac --streaming \
    --window_size 6 --chunk_size 3 \
    --hh_size 3 --retrieval_size 2 \
    --voxel_size 0.05 --voxel_num 4096
```

### 对于500+帧的超长序列:
```bash
# 使用分块策略,每100帧一个块,重叠10帧
python process_long_sequence.py \
    --chunk_frames 100 \
    --overlap_frames 10 \
    --window_size 4 --chunk_size 2 \
    --enable_cpu_offload
```

希望这份指南能帮助你成功使用STAC进行长序列流式推理! 🚀
