# DA3 + STAC KV-Cache集成开发总结

## 📋 项目概述

本次开发完成了Depth Anything 3 (DA3)骨干网络与STAC框架的KV-cache压缩功能集成,使DA3能够利用STAC的流式推理能力进行长视频3D重建。

## ✅ 完成的工作

### 1. DA3 ViT骨干网络改造

**文件**: `depth-anything-3/src/depth_anything_3/model/dinov2/vision_transformer.py`

#### 已完成的改造:

1. **SparseAttention层替换** (Phase 0 & 1)
   - 在`alt_start`之后的奇数层替换为标准`SparseAttention`
   - 设置`layer_idx`用于KV缓存管理
   - 记录`_global_layer_indices`用于管理器注册

2. **KV Manager接口** (Phase 1)
   ```python
   # 已实现的方法:
   - register_kv_mgr()      # 注册KV管理器
   - clear_kv_mgr()         # 清除缓存
   - prune_kv_mgr()         # 剪枝缓存 (H2O模式)
   - retrieve_kv_mgr()      # 检索缓存 (Voxel模式)
   - update_kv_mgr_pos()    # 更新3D点位置
   - get_kv_mgr_info()      # 获取缓存信息
   - set_camhead()          # API兼容性(no-op)
   ```

3. **Forward方法扩展** (Phase 1 & 2)
   - 添加`mode`和`streaming`参数
   - 支持STAC格式的输入输出
   - 实现`_convert_to_stac_format()`适配器
   - 实现`_encode_pose_from_extrinsics()`姿态编码
   - 实现`_compute_world_points()`世界坐标计算

### 2. Attention层改造

**文件**: `depth-anything-3/src/depth_anything_3/model/dinov2/layers/attention.py`

#### 已完成:

1. **SparseAttention类**
   - 继承自标准`Attention`
   - 支持`kv_manager`参数
   - 实现三种注意力分发:
     * KV缓存模式: `decode_sparse_attn()`
     * Flex attention模式: `flex_attention()`
     * 标准SDPA模式: `scaled_dot_product_attention()`

2. **Attention.forward()**
   - 添加`**kwargs`支持
   - 透传`kv_manager`等参数

### 3. Block层改造

**文件**: `depth-anything-3/src/depth_anything_3/model/dinov2/layers/block.py`

#### 已完成:

1. **Block.forward()**
   - 签名改为`forward(x, pos=None, attn_mask=None, **kwargs)`
   - `attn_residual_func`内部透传`**kwargs`给attention
   - 支持drop_path训练模式下的参数传递

### 4. StreamSession适配

**文件**: `/Users/donger/Documents/Git/STAC/stream_session.py`

#### 已完成:

代码中已包含DA3支持(第78-83行):
```python
if model_type == "da3":
    # DA3 uses backbone as the aggregator
    self.aggregator = model.model.backbone.pretrained
else:
    # CausalVGGT uses dedicated aggregator
    self.aggregator = model.aggregator
```

以及在pipeline中的多处条件判断:
- 第314行: `set_camhead`调用适配
- 第451行: STAC模式下`set_camhead`适配
- 第505-512行: `update_kv_mgr_pos`调用
- 第517-527行: `retrieve_kv_mgr`调用
- 第530-533行: `prune_kv_mgr`调用

### 5. Model Wrapper注册

**文件**: `/Users/donger/Documents/Git/STAC/model_wrapper.py`

#### 已完成:

1. **DA3模型注册** (第33行)
   ```python
   model_wrappers = {
       "causalvggt": CausalVGGT,
       "da3": DepthAnything3,  # 已注册
   }
   ```

2. **StreamSession注册** (第38-40行)
   ```python
   stream_sessions = {
       "causalvggt": StreamSession,
       "da3": StreamSession,  # 已注册
   }
   ```

3. **模型路径配置** (第22-28行)
   ```python
   model_paths = {
       # ... 原有模型 ...
       "da3-small": ...,
       "da3-base": ...,
       "da3-large": ...,
       "da3-giant": ...,
       "da3nested-giant-large": ...,
   }
   ```

4. **load_model函数** (第85-103行)
   - DA3使用HuggingFace风格加载
   - 支持`from_pretrained()`和本地checkpoint

### 6. DA3模型输出适配

**文件**: `depth-anything-3/src/depth_anything_3/model/da3.py`

#### 已完成:

1. **DepthAnything3Net.forward()扩展**
   ```python
   def forward(
       self,
       x,
       extrinsics=None,
       intrinsics=None,
       export_feat_layers=[],
       infer_gs=False,
       use_ray_pose=False,
       ref_view_strategy="saddle_balanced",
       mode="full",          # 新增: STAC兼容
       streaming=False,      # 新增: 流式模式
       **kwargs              # 新增: KV缓存参数
   )
   ```

2. **STAC格式转换器**
   - `_convert_to_stac_format()`: 输出格式统一
   - `_encode_pose_from_extrinsics()`: 姿态编码
   - `_compute_world_points()`: 3D点反投影

## 📁 修改的文件清单

### 核心改造文件

| 文件路径 | 改动类型 | 改动量 | 风险等级 |
|---------|---------|--------|---------|
| `depth-anything-3/src/depth_anything_3/model/dinov2/vision_transformer.py` | 修改+新增 | +250行 | 🔴 高 |
| `depth-anything-3/src/depth_anything_3/model/dinov2/layers/attention.py` | 新增类 | +80行 | 🟡 中 |
| `depth-anything-3/src/depth_anything_3/model/dinov2/layers/block.py` | 修改签名 | ~15行 | 🟢 低 |
| `depth-anything-3/src/depth_anything_3/model/da3.py` | 修改+新增 | +280行 | 🔴 高 |
| `model_wrapper.py` | 修改注册 | +5行 | 🟢 低 |
| `stream_session.py` | 已包含DA3支持 | 无需改动 | - |

### 文档和测试文件

| 文件路径 | 类型 | 说明 |
|---------|------|------|
| `DA3_STAC_INTEGRATION_GUIDE.md` | 新增 | 用户使用指南 |
| `DA3_STAC_DEVELOPMENT_SUMMARY.md` | 新增 | 本文档 |
| `test_da3_stac.py` | 新增 | 集成测试脚本 |

## 🏗️ 架构设计

### 数据流图

```
┌─────────────────────────────────────────────────────────────┐
│                        Input Images                          │
│                      [S, 3, H, W]                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   DepthAnything3.forward()                   │
│  - 处理输入形状 [S,3,H,W] -> [1,S,3,H,W]                    │
│  - 调用backbone提取特征                                      │
│  - 通过depth head预测深度                                    │
│  - 通过cam_dec预测姿态                                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              DinoVisionTransformer (backbone)                │
│  ┌──────────────────────────────────────────────────┐       │
│  │ Local Layers (alt_start之前或偶数层)              │       │
│  │ - 标准Attention                                  │       │
│  │ - kv_manager=None                                │       │
│  └──────────────────────────────────────────────────┘       │
│  ┌──────────────────────────────────────────────────┐       │
│  │ Global Layers (alt_start之后奇数层) ⭐            │       │
│  │ - SparseAttention                                │       │
│  │ - kv_manager=自管理器                            │       │
│  │ - 支持KV append/prune/retrieve                   │       │
│  └──────────────────────────────────────────────────┘       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           _convert_to_stac_format() 适配器                   │
│  - extrinsics -> pose_enc (9D编码)                          │
│  - depth -> [B,S,H,W,1]                                     │
│  - depth + extrinsics + intrinsics -> world_points          │
│  - 输出STAC兼容的字典格式                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    StreamSession.pipeline()                  │
│  - 分块处理图像序列                                          │
│  - 调用KV manager进行缓存管理                                │
│  - 聚合预测结果                                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      Final Output                            │
│  - pose_enc: [1, S, 9]                                      │
│  - depth: [1, S, H, W, 1]                                   │
│  - world_points: [1, S, H, W, 3]                            │
│  - timing/merger统计信息                                     │
└─────────────────────────────────────────────────────────────┘
```

### KV Cache生命周期

```
1. 注册阶段 (register_kv_mgr)
   ├─ 设置SparseAttention层的layer_idx
   ├─ 创建KVManager或STACVoxelKV实例
   └─ 初始化缓存数据结构

2. 推理阶段 (forward per chunk)
   ├─ SparseAttention.append_kv() - 添加新KV
   ├─ decode_sparse_attn() - 使用缓存计算注意力
   └─ 输出预测结果

3. 管理阶段 (每N个chunk)
   ├─ update_kv_mgr_pos() - 更新3D点位置 (Voxel模式)
   ├─ retrieve_kv_mgr() - 检索相关token (Voxel模式)
   ├─ prune_kv_mgr() - 剪枝不重要token (H2O模式)
   └─ get_kv_mgr_info() - 获取统计信息

4. 清理阶段 (clear_kv_mgr)
   ├─ reset() - 重置缓存
   ├─ free() - 释放显存
   └─ kv_manager = None
```

## 🔧 关键技术点

### 1. SparseAttention层识别

```python
# 在DinoVisionTransformer.__init__()中
self._global_layer_indices = []
for i in range(depth):
    if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
        # 这是全局层,替换为SparseAttention
        old_attn = blocks_list[i].attn
        new_attn = SparseAttention(...)
        blocks_list[i].attn = new_attn
        blocks_list[i].attn.set_layer_idx(len(self._global_layer_indices))
        self._global_layer_indices.append(i)
```

**关键点**:
- 只在`alt_start`之后生效
- 只替换奇数层 (偶数层是局部层)
- `layer_idx`从0开始连续编号

### 2. Pose编码简化

```python
def _encode_pose_from_extrinsics(self, extrinsics):
    # extrinsics: [B, N, 3, 4] w2c格式
    # w2c = [R | t], R是3x3旋转,t是3x1平移
    
    R = extrinsics[:, :, :3, :3]  # [B, N, 3, 3]
    t = extrinsics[:, :, :3, 3:]  # [B, N, 3, 1]
    
    # 简化编码: R的前2行(6值) + t(3值) = 9值
    pose_enc = torch.cat([
        R[:, :, :2, :].reshape(-1, 6),
        t.reshape(-1, 3)
    ], dim=-1).reshape(B, N, 9)
    
    return pose_enc
```

**注意**: 这是简化编码,与CausalVGGT的完整Rodrigues表示不同。如需完全兼容,需实现完整的姿态编码/解码器。

### 3. World Points反投影

```python
def _compute_world_points(self, depth, extrinsics, intrinsics, H, W):
    # 1. 创建像素网格 [H, W, 3]
    pixel_coords = stack([u, v, 1])
    
    # 2. 反投影到相机坐标
    cam_coords = K^-1 @ pixel_coords * depth
    
    # 3. 转换到世界坐标
    c2w = inverse(extrinsics)  # w2c -> c2w
    world_coords = c2w @ cam_coords_homo
    
    return world_coords, confidence
```

### 4. StreamSession中的DA3适配

```python
# 在StreamSession.__init__中
if model_type == "da3":
    self.aggregator = model.model.backbone.pretrained
else:
    self.aggregator = model.aggregator

# 在pipeline中多处使用条件判断
if self.model_type == "da3":
    kv_pos_time = self.model.model.backbone.pretrained.update_kv_mgr_pos(...)
    retrieval_time = self.model.model.backbone.pretrained.retrieve_kv_mgr(...)
    evict_merge_time = self.model.model.backbone.pretrained.prune_kv_mgr(...)
else:
    kv_pos_time = self.model.aggregator.update_kv_mgr_pos(...)
    ...
```

## 📊 与原始计划的对比

### 原计划 (DA3_INTEGRATION_PLAN.md)

| 阶段 | 预估工时 | 实际状态 | 说明 |
|------|---------|---------|------|
| Phase 0: 基础对接 | 5h | ✅ 完成 | SparseAttention类、**kwargs透传 |
| Phase 1: ViT核心 | 24h | ✅ 完成 | KV manager接口、流式推理 |
| Phase 2: 上层适配 | 18h | ✅ 完成 | StreamSession、model_wrapper |
| Phase 3: 集成调试 | 20h | ⏸️ 待定 | 需要实际运行测试 |

**实际开发时间**: ~6-8小时 (代码编写) + 测试调试时间

### 偏离计划的部分

1. **未实现的部分**:
   - `inference()`方法 (逐帧推理) - 当前使用`forward(streaming=True)`替代
   - 独立的`_forward_streaming()`方法 - 集成到主`forward()`中

2. **简化的部分**:
   - Pose编码使用简化版本 (9D而非完整表示)
   - World Points通过反投影计算而非point head
   - aggregated_tokens_list使用占位符`[None]`

3. **额外的改进**:
   - 添加了完整的用户使用文档
   - 创建了测试脚本框架
   - 添加了详细的代码注释

## 🚀 使用方法

### 快速开始

```python
from model_wrapper import load_model, run_model
import torch

# 1. 加载模型
model = load_model("da3", base_model="da3-large", device="cuda")

# 2. 准备图像 [S, 3, H, W]
# images = ...

# 3. STAC流式推理
predictions = run_model(
    model, images, "da3",
    mode="stac",
    streaming=True,
    dtype=torch.bfloat16,
    window_size=4,
    chunk_size=2,
    voxel_size=0.05,
)

# 4. 访问结果
print(predictions["depth"].shape)
print(predictions["timing"])
```

详细用法请参考: **[DA3_STAC_INTEGRATION_GUIDE.md](./DA3_STAC_INTEGRATION_GUIDE.md)**

## 🧪 测试建议

### 1. 单元测试

```python
# test_da3_unit.py
def test_sparse_attention_replacement():
    model = load_model("da3", base_model="da3-small")
    backbone = model.model.backbone.pretrained
    
    # 检查SparseAttention层
    global_layers = backbone._global_layer_indices
    assert len(global_layers) > 0, "No SparseAttention layers found"
    
    # 检查layer_idx设置
    for idx in global_layers:
        assert backbone.blocks[idx].attn.layer_idx is not None

def test_kv_manager_registration():
    model = load_model("da3", base_model="da3-small")
    backbone = model.model.backbone.pretrained
    
    # 注册KV manager
    backbone.register_kv_mgr(KVManager, recent_size=4, chunk_size=2)
    assert backbone.kv_manager is not None
    
    # 清理
    backbone.clear_kv_mgr()
```

### 2. 集成测试

```bash
# 运行测试脚本
python test_da3_stac.py --test all

# 或单独测试某个模式
python test_da3_stac.py --test standard
python test_da3_stac.py --test streaming
python test_da3_stac.py --test window_kv
```

### 3. 性能测试

```python
# benchmark_da3.py
import time

# 测试不同序列长度的性能
for num_frames in [4, 8, 16, 32, 64]:
    images = create_test_images(num_frames)
    
    start = time.time()
    predictions = run_model(model, images, "da3", mode="stac", streaming=True)
    elapsed = time.time() - start
    
    print(f"Frames: {num_frames}, Time: {elapsed:.2f}s, "
          f"FPS: {predictions['timing']['infer_fps']:.1f}")
```

## ⚠️ 已知限制和改进建议

### 当前限制

1. **Pose编码简化**
   - 当前使用9D简化编码 (R的前2行 + t)
   - 建议: 实现完整的Rodrigues编码以兼容CausalVGGT

2. **无Aggregated Tokens**
   - 使用占位符`[None]`
   - Camera head inference无法使用
   - 建议: 实现token缓存机制

3. **World Points计算**
   - 通过深度反投影,可能有精度损失
   - 建议: 添加point head或改进反投影算法

4. **无逐帧推理接口**
   - 当前只支持chunk模式
   - 建议: 实现`inference()`方法支持逐帧推理

### 性能优化建议

1. **CUDA加速**
   ```python
   # 编译CUDA扩展
   voxel_backend = "cuda"
   attn_backend = "cuda"
   ```

2. **CPU Offload**
   ```python
   # 适合超长序列
   enable_alloc_cpu = True
   gpu_threshold_gb = 10.0
   ```

3. **显存优化**
   - 减小`window_size`和`chunk_size`
   - 降低输入分辨率
   - 使用更小的模型

### 功能扩展建议

1. **添加Camera Head**
   ```python
   from causalvggt.heads.camera_head import CameraHead
   model.camera_head = CameraHead(...)
   ```

2. **训练支持**
   - 修改训练循环支持streaming模式
   - 实现梯度累积

3. **多场景批处理**
   - 支持同时处理多个视频序列

## 📚 参考资源

- [DA3_INTEGRATION_PLAN.md](./DA3_INTEGRATION_PLAN.md) - 原始集成计划
- [DA3_STAC_INTEGRATION_GUIDE.md](./DA3_STAC_INTEGRATION_GUIDE.md) - 用户使用指南
- [STAC README.md](./README.md) - STAC框架文档
- [DA3 README.md](./depth-anything-3/README.md) - DA3原始文档
- [CausalVGGT代码](./causalvggt/) - 参考实现

## 📝 变更日志

### 2026-04-03 (初始开发)

- ✅ 完成DA3 ViT骨干网络的KV manager接口
- ✅ 完成SparseAttention层替换
- ✅ 完成Block forward的**kwargs透传
- ✅ 完成DA3 forward的STAC格式适配
- ✅ 完成model_wrapper.py的DA3注册
- ✅ 完成StreamSession的DA3适配
- ✅ 创建用户使用文档
- ✅ 创建测试脚本框架

## 🎯 下一步工作

1. **安装依赖并运行测试**
   ```bash
   pip install omegaconf addict einops
   pip install xformers torch>=2
   python test_da3_stac.py
   ```

2. **修复发现的问题**
   - 查看测试输出
   - 调试错误
   - 修正代码

3. **性能基准测试**
   - 对比DA3+STAC vs DA3全注意力
   - 测量显存、FPS、精度

4. **文档完善**
   - 添加实际运行截图
   - 补充故障排除案例
   - 更新性能数据

---

**开发者**: Qwen Code Assistant  
**日期**: 2026-04-03  
**状态**: ✅ 代码开发完成,待测试验证
