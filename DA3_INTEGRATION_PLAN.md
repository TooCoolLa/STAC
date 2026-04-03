# DA3 接入 STAC 改造计划

> **目标**: 将 Depth Anything 3 (DA3) 作为可插拔骨干模型接入 STAC 框架，使其能够利用 STAC 的 KV-cache 压缩能力进行长视频流式 3D 重建。

---

## 一、项目概览

### 1.1 架构对比

| 维度 | STAC (CausalVGGT) | DA3 (Depth Anything 3) |
|------|-------------------|------------------------|
| **骨干架构** | CausalAggregator (24层 ViT-L) | DinoV2 (vitb/vitl/vitg) |
| **注意力结构** | 双流: `frame_blocks` + `global_blocks` | 单流: `blocks`，通过 `alt_start` 交替 |
| **注意力层** | `SparseAttention` (支持 `kv_manager`) | 标准 `Attention` (无 KV 管理) |
| **Block 透传** | `forward(x, pos, attn_mask, **kwargs)` | `forward(x, pos, attn_mask)` |
| **KV Cache** | `register_kv_mgr` + `HeavyHittersKV` + prune/retrieve | 无 |
| **流式推理** | `StreamSession` 逐块 + KV 缓存管理 | `DA3_Streaming` 分块 + 后对齐 (Sim3) |
| **输出** | pose_enc + depth + pointmap | depth + extrinsics + intrinsics |
| **Patch Size** | 14 | 14 |

### 1.2 可行性结论

✅ **可行**。DA3 的 DinoV2 骨干本质上是纯 ViT，与 STAC 的 CausalAggregator 同源。接入需要的主要改造集中在：
- DA3 的 Attention/Block 层添加 KV 管理接口
- DA3 ViT 添加 `register_kv_mgr` 等流式推理方法
- `stream_session.py` 适配 DA3 的输入输出格式

---

## 二、推荐策略：方案 A (最小侵入)

**核心思路**: 在 DA3 ViT 内部，将 `alt_start` 之后的**奇数层 block** 的注意力替换为 `SparseAttention`，通过 `**kwargs` 透传 `kv_manager`。无需拆分 blocks 列表。

**优势**:
- 改动集中在 DA3 代码内部，不破坏原有功能
- 无需重写 ViT forward 逻辑
- 总工时从 ~77h 降至 **~60h**

**改动范围**:

```
depth-anything-3/src/depth_anything_3/
├── model/dinov2/layers/
│   ├── attention.py    ← 新增 SparseAttention 类 (+60行)
│   └── block.py        ← forward 加 **kwargs 透传 (~15行改)
├── model/dinov2/
│   └── vision_transformer.py  ← 核心改造 (+200行)
└── model/da3.py        ← forward 加 mode 参数 (+40行)

STAC/
├── stream_session.py   ← 适配 DA3 的 backbone (~70行改)
└── model_wrapper.py    ← 新增 da3 注册 (+20行)
```

---

## 三、阶段划分与任务清单

### Phase 0: 基础对接 (预估: 3h 开发 + 2h 调试 = 5h)

> **目标**: 让 DA3 的注意力层具备接收 `kv_manager` 的能力

| ID | 任务 | 文件 | 改动量 | 风险 | 说明 |
|----|------|------|--------|------|------|
| P0.1 | 新增 `SparseAttention` 类 | `attention.py` | +60行 | 低 | 从 `causalvggt/layers/attention.py` 复制，调整 import |
| P0.2 | `Attention.forward` 加 `**kwargs` | `attention.py` | +1行 | 低 | 签名改为 `forward(x, pos, attn_mask, **kwargs)` |
| P0.3 | `Block.forward` 加 `**kwargs` 透传 | `block.py` | ~15行改 | 低 | `attn_residual_func` 内部透传给 `self.attn()` |
| P0.4 | 移植 `create_attn_mask` 函数 | `vision_transformer.py` | +70行 | 低 | 从 `causalvggt/layers/block.py` 复制 |

**验收标准**: DA3 模型能正常加载并 forward，`SparseAttention` 层能接收 `kv_manager=None` 并回退到标准注意力。

---

### Phase 1: ViT 核心改造 (预估: 12h 开发 + 12h 调试 = 24h) ⚠️ 关键路径

> **目标**: 让 DA3 ViT 支持 KV 缓存管理的全套接口

| ID | 任务 | 文件 | 改动量 | 风险 | 说明 |
|----|------|------|--------|------|------|
| P1.1 | 标记 SparseAttention 层 + 设置 layer_idx | `vision_transformer.py` | ~30行改 | **高** | 遍历 blocks，将 `alt_start` 后奇数层的 `block.attn` 设为 `SparseAttention`，调用 `set_layer_idx()` |
| P1.2 | 新增 `register_kv_mgr()` 方法 | `vision_transformer.py` | +40行 | 中 | 参照 `CausalAggregator.register_kv_mgr()`，实例化 `KVManager`/`STACVoxelKV` |
| P1.3 | 新增 `clear_kv_mgr()` / `prune_kv_mgr()` / `retrieve_kv_mgr()` | `vision_transformer.py` | +50行 | 中 | 委托给 `self.kv_manager` 的对应方法 |
| P1.4 | 新增 `update_kv_mgr_pos()` / `get_kv_mgr_info()` | `vision_transformer.py` | +30行 | 中 | 适配 DA3 的 token 格式 |
| P1.5 | `_get_intermediate_layers` 改造传递 kv_manager | `vision_transformer.py` | ~40行改 | **高** | 每层 forward 时传入 `kv_manager=self.kv_manager`，处理 prune/retrieval 时机 |
| P1.6 | 新增 `inference()` 流式推理方法 | `vision_transformer.py` | +50行 | **高** | 对应 `CausalAggregator.inference()`，逐帧/逐块推理 |
| P1.7 | `forward` 增加 `mode` 参数 | `vision_transformer.py` | +10行 | 中 | 支持 `"full"`/`"causal"`/`"window_kv"`/`"window_chunk_merge"` |

**验收标准**:
- `register_kv_mgr()` 能成功注册 KVManager
- 模型能按 chunk 调用 forward，KV cache 正确增长/剪枝
- `inference()` 能逐帧推理，输出格式与 DPT head 兼容

---

### Phase 2: 上层适配 (预估: 8h 开发 + 10h 调试 = 18h)

> **目标**: 让 STAC 的 StreamSession 和 model_wrapper 能正确调用 DA3

| ID | 任务 | 文件 | 改动量 | 风险 | 说明 |
|----|------|------|--------|------|------|
| P2.1 | `model_wrapper.py` 注册 DA3 | `model_wrapper.py` | +20行 | 低 | 新增 `da3` 到 `model_wrappers`/`stream_sessions` 字典 |
| P2.2 | `stream_session.py` 适配 DA3 backbone | `stream_session.py` | ~70行改 | **高** | `model.aggregator.*` → `model.backbone.*`，共 ~20 处调用点 |
| P2.3 | 适配 DA3 输出格式 | `stream_session.py` | ~15行改 | 中 | DA3 输出是 dict，需解析 depth/extrinsics/intrinsics |
| P2.4 | `camera_head_inference` 适配 | `stream_session.py` | ~15行改 | 中 | DA3 无独立 camera_head，pose 直接从 backbone 输出 |
| P2.5 | `get_pointmap` 适配 patch_size | `stream_session.py` | ~5行改 | 低 | DA3 patch_size=14，与 CausalVGGT 一致 |
| P2.6 | DA3 `da3.py` 增加 streaming 支持 | `da3.py` | +40行 | 中 | `_forward_streaming()` 调用 backbone.inference |

**验收标准**:
- `load_model("da3", base_model="da3-large")` 能正确加载
- `run_model()` 在 `mode="stac"` 下能端到端跑通
- 输出 predictions 包含 depth, extrinsics, intrinsics, world_points

---

### Phase 3: 集成调试 (预估: 0h 开发 + 20h 调试 = 20h)

> **目标**: 验证功能正确性、精度和性能

| ID | 任务 | 预估工时 | 风险 | 说明 |
|----|------|----------|------|------|
| P3.1 | 端到端流式推理验证 | 6h | **高** | 单场景跑通 STAC 模式，检查 KV cache 生命周期 |
| P3.2 | 精度回归测试 | 4h | 中 | 对比 DA3+STAC vs DA3 全注意力模式的深度/姿态精度 |
| P3.3 | 显存 profiling | 4h | 中 | 验证长视频 (100+ 帧) 内存有界增长 |
| P3.4 | 性能 benchmark | 2h | 低 | FPS、显存占用 vs 原始 STAC (CausalVGGT) |
| P3.5 | 多场景评估 | 4h | 中 | 在 7scenes/NRGBD 上跑完整评估流程 |

**验收标准**:
- DA3+STAC 在 100+ 帧视频上不 OOM
- 深度/姿态精度损失 < 5% vs 全注意力基线
- 推理 FPS 与 CausalVGGT+STAC 相当

---

## 四、工时汇总

| 阶段 | 开发 (h) | 调试 (h) | 合计 (h) | 人日 |
|------|:--------:|:--------:|:--------:|:----:|
| Phase 0 基础对接 | 3 | 2 | 5 | 0.6 |
| Phase 1 ViT 核心 | 12 | 12 | 24 | 3.0 |
| Phase 2 上层适配 | 8 | 10 | 18 | 2.3 |
| Phase 3 集成调试 | 0 | 20 | 20 | 2.5 |
| **总计** | **23** | **44** | **67** | **~8.4 天** |

**建议预留 2 天缓冲**（处理意外兼容性），**总计约 10-11 个工作日 (~2 周)**。

---

## 五、关键风险与缓解措施

### 🔴 高风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **P1.1 架构差异** | DA3 的 `alt_start` 交替机制与 STAC 双流设计不同，layer_idx 映射可能出错 | 先写单元测试验证 layer_idx 映射；打印每层注意力类型确认 |
| **P1.5 输出格式** | `_get_intermediate_layers` 输出必须与 DPT head 兼容 | 对比 DA3 原始输出与改造后输出，确保 shape 一致 |
| **P2.2 stream_session** | ~20 处 `model.aggregator.*` 调用需改写，容易遗漏 | 全局 grep 搜索，逐处验证；写适配器层统一接口 |

### 🟡 中等风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **P1.6 inference 方法** | 流式推理的 KV cache 生命周期管理复杂 | 参照 CausalAggregator.inference() 逐行对照实现 |
| **DA3 依赖兼容性** | DA3 的 xformers/torch.compile 可能与 STAC 冲突 | 隔离测试环境，确认 DA3 能在 STAC 环境下正常加载 |

---

## 六、实施顺序与依赖关系

```
Phase 0 (5h)
  ├── P0.1 SparseAttention 类
  ├── P0.2 Attention forward **kwargs
  ├── P0.3 Block forward **kwargs
  └── P0.4 create_attn_mask 移植
       │
       ▼
Phase 1 (24h)  [依赖 Phase 0]
  ├── P1.1 标记 SparseAttention 层          ← 必须先完成
  ├── P1.2 register_kv_mgr()               ← 依赖 P1.1
  ├── P1.3 clear/prune/retrieve_kv_mgr()   ← 依赖 P1.2
  ├── P1.4 update_kv_mgr_pos()             ← 依赖 P1.2
  ├── P1.5 _get_intermediate_layers 改造    ← 依赖 P1.1
  ├── P1.6 inference() 方法                ← 依赖 P1.2, P1.3
  └── P1.7 forward mode 参数               ← 依赖 P0.4
       │
       ▼
Phase 2 (18h)  [依赖 Phase 1]
  ├── P2.1 model_wrapper.py 注册 DA3
  ├── P2.2 stream_session.py 适配          ← 依赖 P1.2~P1.6
  ├── P2.3 DA3 输出格式适配                ← 依赖 P2.2
  ├── P2.4 camera_head 适配                ← 依赖 P2.2
  └── P2.6 da3.py streaming 支持           ← 依赖 P1.6
       │
       ▼
Phase 3 (20h)  [依赖 Phase 2]
  ├── P3.1 端到端流式推理
  ├── P3.2 精度回归
  ├── P3.3 显存 profiling
  ├── P3.4 性能 benchmark
  └── P3.5 多场景评估
```

---

## 七、代码结构预览 (改造后)

```
STAC/
├── depth-anything-3/              # DA3 子模块 (已改造)
│   └── src/depth_anything_3/
│       ├── model/
│       │   ├── dinov2/
│       │   │   ├── layers/
│       │   │   │   ├── attention.py   ← +SparseAttention
│       │   │   │   └── block.py       ← forward(**kwargs)
│       │   │   └── vision_transformer.py  ← +KV manager 接口
│       │   └── da3.py             ← +streaming 支持
│       └── api.py                 ← 不变
│
├── stac/                          # STAC 核心 (基本不变)
│   ├── kv_manager.py
│   ├── h2o.py
│   └── stac_voxel.py
│
├── causalvggt/                    # 原始骨干 (保留)
│
├── model_wrapper.py               ← +DA3 注册
├── stream_session.py              ← ~70 行适配
└── da3_session.py                 ← [可选] DA3 专用 StreamSession 子类
```

---

## 八、验收清单

### 功能验收
- [ ] `load_model("da3", base_model="da3-large")` 成功加载
- [ ] `run_model(mode="stac", streaming=True)` 端到端跑通
- [ ] 输出 predictions 包含 depth, extrinsics, intrinsics, world_points
- [ ] `--mode full` (全注意力) 仍能正常工作 (回退验证)
- [ ] `--mode window_kv` (H2O 剪枝) 正常工作

### 精度验收
- [ ] DA3+STAC vs DA3 全注意力: 深度 RMSE 差异 < 5%
- [ ] DA3+STAC vs DA3 全注意力: 姿态 ATE 差异 < 5%
- [ ] 在至少 3 个场景上验证 (7scenes, NRGBD, DTU)

### 性能验收
- [ ] 100 帧视频推理显存 < 12GB (对比 DA3_Streaming 官方声称)
- [ ] 推理 FPS >= CausalVGGT+STAC 的 80%
- [ ] KV cache 内存增长有界 (不随帧数线性增长)

### 代码质量
- [ ] DA3 原有功能不受影响 (单图/多图深度估计仍正常)
- [ ] 添加必要的日志输出 (KV cache 大小、剪枝率等)
- [ ] 添加 README 说明 DA3 接入的使用方法

---

## 九、后续扩展 (可选)

| 扩展项 | 说明 | 优先级 |
|--------|------|--------|
| CUDA 加速 | 为 DA3 的 SparseAttention 添加 attn-cuda / merger-cuda 支持 | 中 |
| 训练支持 | 支持 DA3+STAC 的端到端微调 | 低 |
| 嵌套模型 | 支持 DA3Nested (Giant + Metric) 的 STAC 推理 | 低 |
| 自动模型选择 | 根据视频长度/场景自动选择 STAC 参数 | 低 |

---

*文档创建时间: 2026-04-03*
*预估总工时: ~67h (约 10-11 工作日)*
*推荐策略: 方案 A (最小侵入)*
