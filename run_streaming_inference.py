#!/usr/bin/env python3
"""
STAC流式推理长序列图片 - 完整可运行示例

使用方法:
    python run_streaming_inference.py --scene_dir /path/to/scene --output_dir ./output
    
支持的场景目录结构:
    /path/to/scene/
    ├── images/
    │   ├── frame_001.png
    │   ├── frame_002.png
    │   └── ...
    或者直接放图片:
    /path/to/scene/
    ├── frame_001.png
    ├── frame_002.png
    └── ...
"""

import argparse
import os
import sys
import json
import logging
import time
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from model_wrapper import load_model, run_model
from eval.utils.image import load_scene_images
from causalvggt.utils.pose_enc import pose_encoding_to_extri_intri
from causalvggt.utils.geometry import unproject_depth_map_to_point_map

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("streaming_inference")


def parse_args():
    parser = argparse.ArgumentParser(
        description="STAC流式推理长序列图片",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument(
        "--scene_dir",
        type=str,
        required=True,
        help="场景目录，包含images/子目录或直接包含图片"
    )
    
    # 输出参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="输出目录"
    )
    
    # 模型参数
    parser.add_argument(
        "--model_name",
        type=str,
        default="causalvggt",
        choices=["causalvggt", "da3"],
        help="模型类型"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="stream3r",
        choices=["stream3r", "streamvggt", "da3-small", "da3-base",
                 "da3-large", "da3-large-1.1", "da3-giant", "da3nested-giant-large"],
        help="骨干模型"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        choices=[224, 512, 518, 504],
        help="输入分辨率(自动根据模型选择)"
    )
    
    # STAC参数
    parser.add_argument(
        "--mode",
        type=str,
        default="stac",
        choices=["stac", "window_kv", "full"],
        help="推理模式"
    )
    parser.add_argument(
        "--window_size", "-win",
        type=int,
        default=4,
        help="窗口大小(帧数)"
    )
    parser.add_argument(
        "--chunk_size", "-ck",
        type=int,
        default=2,
        help="每步处理帧数"
    )
    parser.add_argument(
        "--hh_size", "-hh",
        type=int,
        default=2,
        help="Heavy-hitter缓存大小"
    )
    parser.add_argument(
        "--retrieval_size", "-ret_sz",
        type=int,
        default=2,
        help="检索缓存大小(-1=始终检索)"
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.05,
        help="体素大小(米)"
    )
    parser.add_argument(
        "--voxel_num",
        type=int,
        default=4096,
        help="初始体素数量"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=2.0,
        help="置信度阈值"
    )
    parser.add_argument(
        "--voxel_backend",
        type=str,
        default="cuda",
        choices=["cuda", "python"],
        help="体素后端"
    )
    
    # 其他参数
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
        help="数据类型"
    )
    parser.add_argument(
        "--kf_every",
        type=int,
        default=1,
        help="每隔K帧采样一次(>1时跳过帧)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="最大处理帧数(None=全部)"
    )
    parser.add_argument(
        "--save_depth_vis",
        action="store_true",
        help="保存深度可视化图片"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出"
    )
    
    return parser.parse_args()


def setup_config(args):
    """配置设置"""
    # 自动选择输入大小
    if args.size is None:
        if args.model_name == "da3":
            args.size = 504
        else:
            args.size = 518
    
    # 自动选择数据类型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "auto":
        if device.type == "cuda":
            if torch.cuda.get_device_capability()[0] >= 8:
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        else:
            dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    return device, dtype


def run_inference(args, device, dtype):
    """运行流式推理"""
    
    print("\n" + "="*70)
    print("🚀 STAC Streaming Inference")
    print("="*70)
    
    # 1. 加载模型
    print(f"\n📦 Loading model: {args.model_name} ({args.base_model})")
    print(f"   Device: {device}, dtype: {dtype}")
    
    model_start = time.time()
    model = load_model(
        args.model_name,
        base_model=args.base_model,
        device=device
    )
    model.eval()
    model_time = time.time() - model_start
    print(f"✅ Model loaded in {model_time:.2f}s")
    
    # 2. 加载图片
    print(f"\n📸 Loading images from: {args.scene_dir}")
    img_start = time.time()
    images = load_scene_images(args.scene_dir, size=args.size).to(device)
    
    # 应用帧采样
    if args.kf_every > 1:
        images = images[::args.kf_every]
        print(f"   ⚡ Frame sampling: every {args.kf_every} frame")
    
    if args.max_frames:
        images = images[:args.max_frames]
        print(f"   ⚡ Limit to {args.max_frames} frames")
    
    img_time = time.time() - img_start
    print(f"✅ Loaded {images.shape[0]} frames in {img_time:.2f}s")
    print(f"   Image shape: {images.shape}")
    
    # 3. 配置STAC参数
    stac_kwargs = {
        "window_size": args.window_size,
        "chunk_size": args.chunk_size,
        "hh_size": args.hh_size,
        "retrieval_size": args.retrieval_size,
        "return_buf": False,
        "voxel_size": args.voxel_size,
        "voxel_num": args.voxel_num,
        "conf_threshold": args.conf_threshold,
        "temperature": 0.9,
        "voxel_backend": args.voxel_backend,
        "allocator": "segment",
        "timing": True,
        "pinned_frame_indices": [0],
    }
    
    print(f"\n⚙️  STAC Configuration:")
    print(f"   Mode: {args.mode}")
    print(f"   Window size: {stac_kwargs['window_size']}")
    print(f"   Chunk size: {stac_kwargs['chunk_size']}")
    print(f"   HH size: {stac_kwargs['hh_size']}")
    print(f"   Voxel size: {stac_kwargs['voxel_size']}m")
    print(f"   Voxel num: {stac_kwargs['voxel_num']}")
    
    # 4. 运行推理
    print(f"\n🎬 Starting streaming inference...")
    inference_start = time.time()
    
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    
    with torch.no_grad(), autocast_ctx:
        predictions = run_model(
            model,
            images,
            args.model_name,
            mode=args.mode,
            streaming=True,
            dtype=dtype,
            device=device,
            **stac_kwargs,
        )
    
    inference_time = time.time() - inference_start
    print(f"\n✅ Inference completed in {inference_time:.2f}s")
    
    # 5. 输出统计
    print(f"\n📊 Performance Statistics:")
    print(f"   Total frames: {images.shape[0]}")
    print(f"   Total time: {inference_time:.2f}s")
    print(f"   FPS: {predictions['timing']['infer_fps']:.1f}")
    
    if 'timing' in predictions:
        print(f"\n   Timing breakdown:")
        for key, value in predictions['timing'].items():
            if key != 'infer_fps':
                print(f"     {key}: {value:.2f}ms")
    
    # 6. 解码位姿 (仅CausalVGGT)
    if args.model_name == "causalvggt" and "pose_enc" in predictions:
        print(f"\n🎯 Decoding camera poses...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            images.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        
        print(f"   Extrinsic shape: {extrinsic.shape}")
        print(f"   Intrinsic shape: {intrinsic.shape}")
        
        # 计算世界坐标点
        depth_map = predictions["depth"]
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy().squeeze(0)
            extrinsic_np = extrinsic.cpu().numpy().squeeze(0)
            intrinsic_np = intrinsic.cpu().numpy().squeeze(0)
        else:
            extrinsic_np = extrinsic
            intrinsic_np = intrinsic
        
        print(f"   Computing world points...")
        world_points = unproject_depth_map_to_point_map(
            depth_map, extrinsic_np, intrinsic_np
        )
        predictions["world_points_np"] = world_points
        print(f"   World points shape: {world_points.shape}")
    
    # 7. 保存结果
    if args.output_dir:
        print(f"\n💾 Saving results to: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 保存为numpy格式
        save_numpy_results(predictions, args, args.output_dir)
        
        # 保存深度可视化
        if args.save_depth_vis and "depth" in predictions:
            save_depth_visualization(predictions["depth"], args.output_dir)
        
        # 保存统计信息
        save_statistics(predictions, images.shape[0], inference_time, 
                       args.output_dir, args)
        
        print(f"✅ All results saved!")
    
    return predictions


def save_numpy_results(predictions, args, output_dir):
    """保存numpy格式的结果"""
    
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 深度图
    if "depth" in predictions:
        depth = predictions["depth"]
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().float().numpy()
        np.save(os.path.join(results_dir, "depth.npy"), depth)
        print(f"   ✓ depth.npy ({depth.shape})")
    
    # 深度置信度
    if "depth_conf" in predictions:
        depth_conf = predictions["depth_conf"]
        if isinstance(depth_conf, torch.Tensor):
            depth_conf = depth_conf.cpu().float().numpy()
        np.save(os.path.join(results_dir, "depth_conf.npy"), depth_conf)
    
    # 位姿编码
    if "pose_enc" in predictions:
        pose_enc = predictions["pose_enc"]
        if isinstance(pose_enc, torch.Tensor):
            pose_enc = pose_enc.cpu().float().numpy()
        np.save(os.path.join(results_dir, "pose_enc.npy"), pose_enc)
    
    # 外参
    if "extrinsic" in predictions:
        extrinsic = predictions["extrinsic"]
        if isinstance(extrinsic, torch.Tensor):
            extrinsic = extrinsic.cpu().float().numpy()
        np.save(os.path.join(results_dir, "extrinsic.npy"), extrinsic)
    
    # 内参
    if "intrinsic" in predictions:
        intrinsic = predictions["intrinsic"]
        if isinstance(intrinsic, torch.Tensor):
            intrinsic = intrinsic.cpu().float().numpy()
        np.save(os.path.join(results_dir, "intrinsic.npy"), intrinsic)
    
    # 世界坐标点
    if "world_points" in predictions:
        world_points = predictions["world_points"]
        if isinstance(world_points, torch.Tensor):
            world_points = world_points.cpu().float().numpy()
        np.save(os.path.join(results_dir, "world_points.npy"), world_points)


def save_depth_visualization(depth, output_dir):
    """保存深度可视化图片"""
    try:
        import matplotlib.pyplot as plt
        
        vis_dir = os.path.join(output_dir, "depth_visualization")
        os.makedirs(vis_dir, exist_ok=True)
        
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        
        # 假设depth shape: [1, S, H, W, 1] 或 [1, S, H, W]
        if len(depth.shape) == 5:
            depth = depth[0, :, :, :, 0]  # [S, H, W]
        elif len(depth.shape) == 4:
            depth = depth[0]  # [S, H, W]
        
        num_frames = depth.shape[0]
        print(f"   Generating {num_frames} depth visualizations...")
        
        for i in range(num_frames):
            if i % 10 == 0 or i == num_frames - 1:  # 每10帧保存一次
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                im = ax.imshow(depth[i], cmap='turbo')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f'Frame {i}')
                ax.axis('off')
                plt.tight_layout()
                plt.savefig(
                    os.path.join(vis_dir, f"depth_{i:05d}.png"),
                    dpi=150,
                    bbox_inches='tight'
                )
                plt.close()
        
        print(f"   ✓ Depth visualizations saved to {vis_dir}")
        
    except Exception as e:
        print(f"   ⚠ Failed to save depth visualization: {e}")


def save_statistics(predictions, num_frames, inference_time, output_dir, args):
    """保存统计信息"""
    stats = {
        "config": {
            "model_name": args.model_name,
            "base_model": args.base_model,
            "input_size": args.size,
            "mode": args.mode,
            "window_size": args.window_size,
            "chunk_size": args.chunk_size,
            "voxel_size": args.voxel_size,
            "voxel_num": args.voxel_num,
        },
        "performance": {
            "num_frames": num_frames,
            "total_time_seconds": inference_time,
            "fps": predictions["timing"]["infer_fps"],
            "timing_ms": {
                k: v for k, v in predictions["timing"].items()
                if k != "infer_fps"
            }
        },
        "output_shapes": {}
    }
    
    # 记录输出形状
    for key in ["depth", "pose_enc", "extrinsic", "intrinsic", "world_points"]:
        if key in predictions:
            val = predictions[key]
            if isinstance(val, torch.Tensor):
                stats["output_shapes"][key] = list(val.shape)
            elif isinstance(val, np.ndarray):
                stats["output_shapes"][key] = list(val.shape)
    
    # 保存merger统计
    if "merger" in predictions:
        merger = predictions["merger"]
        if hasattr(merger, '__dict__'):
            stats["merger"] = str(merger)
        else:
            stats["merger"] = merger
    
    # 写入文件
    stats_file = os.path.join(output_dir, "statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"   ✓ statistics.json")


def main():
    args = parse_args()
    
    # 验证场景目录
    if not os.path.exists(args.scene_dir):
        logger.error(f"Scene directory not found: {args.scene_dir}")
        sys.exit(1)
    
    # 设置配置
    device, dtype = setup_config(args)
    
    # 运行推理
    try:
        predictions = run_inference(args, device, dtype)
        
        print("\n" + "="*70)
        print("🎉 Inference completed successfully!")
        print("="*70)
        print(f"\nOutput directory: {args.output_dir}")
        print(f"Results: {os.path.join(args.output_dir, 'results/')}")
        if args.save_depth_vis:
            print(f"Visualizations: {os.path.join(args.output_dir, 'depth_visualization/')}")
        print(f"Statistics: {os.path.join(args.output_dir, 'statistics.json')}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
