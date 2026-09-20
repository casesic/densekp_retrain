# Synthetic render P0 计划

## Phase 5.75：最小外观基线

状态：`PASS`。已在真实 motion 的 16 帧闭环上加入程序化肤色材质、简单地面、非透明世界背景与三点光；几何、相机、DenseKP 投影、人体 depth/mask/visibility 与原 P0 完全一致。当前机器没有可直接使用的 HDRI、皮肤纹理或动作兼容服装缓存。详见 `APPEARANCE_ASSET_AUDIT.md` 和 `APPEARANCE_BASELINE.md`。

## 目标与非目标

P0 只证明一件事：在远程 Linux 上，用同一组 SMPL-X 顶点生成 RGB/depth/mask 与严格对应的 DenseKP 138 二维真值。Unreal、真实衣服、头发、复杂场景、鱼眼、ERP 和大规模训练均不是 P0 前置条件。

## 推荐数据流

```text
SMPL-X model + motion parameters
                |
                v
       Python smplx forward
                |
       vertices [10475,3]  + fixed faces
                |
        +-------+-------------------+
        |                           |
        v                           v
Blender 主 renderer          densekp_regressor_R
RGB/depth/mask                 @ vertices
                                    |
                             DenseKP [138,3]
                                    |
                       同一个 world/camera/lens model
                                    |
                             DenseKP [138,2]
```

pyrender 对同一输入运行低成本交叉检查，不另算一套 pose。动画源优先级：

1. 原生 SMPL-X `motion_seq.npz`；
2. 可可靠转换为相同 SMPL-X 参数约定的 AMASS/自定义 motion；
3. Alembic 仅用于已有 GeometryCache/DCC 互操作，不作为 P0 必需格式。

## 单一几何源契约

每帧 manifest 至少保存：

```yaml
geometry:
  model_type: smplx
  gender: neutral
  num_vertices: 10475
  betas: []
  global_orient: []
  body_pose: []
  hand_pose: []
  expression: []
  transl: []
  motion_file: null
  motion_frame: 0
transform:
  model_to_world: [[...], [...], [...], [...]]
camera:
  model: perspective
  world_to_camera: [[...], [...], [...], [...]]
  intrinsics: {fx: 460.8, fy: 460.8, cx: 256.0, cy: 256.0}
  resolution: [512, 512]
  near: 0.05
  far: 100.0
render:
  renderer: blender
  engine: cycles
  depth_semantics: camera_z
  mask_ids: {body: 1}
```

坐标契约必须显式说明：米或厘米、右/左手系、world axis、camera forward、像素中心、图像 v 轴、主点、near/far、depth 语义和 clipping。Blender/OpenGL 与 OpenCV camera axes 的转换只允许在一个经过测试的函数内实现。

## 实施步骤

### P0.1 已完成的环境门槛

- Blender 5.1.1 headless Eevee：PASS。
- Blender Cycles OptiX + RTX 3090：PASS。
- pyrender EGL + RTX 3090：PASS。
- static neutral SMPL-X render：PASS。
- 同顶点 DenseKP 138 overlay：PASS。

### P0.2 最小动画

1. 建立独立 renderer Python 环境，锁定 Blender 与 Python 依赖，不修改 CameraHMR 环境。
2. 读取一段已授权的 `motion_seq.npz`，严格记录 fps、frame index、betas、pose、trans 和 gender。
3. 一次创建 Blender scene 与 mesh；逐帧用同一 `vertices` 更新 mesh，不重启 Blender。
4. 输出 8--32 个分散帧的 RGB、linear depth、body mask、camera/geometry manifest。
5. 同帧计算 DenseKP 138 并画 overlay。

### P0.3 数值验证

- 顶点固定为 `(10475,3)` 且 finite；R 固定为 `(138,10475)`。
- renderer mesh 顶点与 GT 分支输入应是同一数组或其记录明确的刚体变换结果。
- 选取可辨认的表面点，比较 Blender render、pyrender render 与解析投影。
- 验证 near/far clipping、画外点、相机后点、mask 与 depth background 值。
- 用 depth test 生成 provisional visibility，并在 T-pose、自遮挡 pose、近景截断各人工检查一帧。
- 只有 overlay 与 visibility 都通过，才允许扩大样本数量。

## Renderer 分工

- Blender：正式 RGB/depth/mask，先 Cycles low samples；后续可按速度/质量切 Eevee。
- pyrender：单元测试、相机投影 oracle、快速可视化、简单 baseline 与 Blender 输出交叉检查。
- Unreal：P0 不用；P2 若要直接复用 BEDLAM2 Starter Pack/Unreal-only scenes，再建立独立高保真分支。

## Fisheye / ERP 的进入条件

透视 P0 通过后才进入 camera domain：

- equidistant/equisolid：先用 Blender Cycles 原生 panoramic fisheye，并以同公式投影 DenseKP。
- 其他项目相机：目标 ray renderer 或同光心 cubemap -> target projection；depth-aware reprojection 仅作过渡。
- ERP：Blender 原生 equirectangular 为主，cubemap -> ERP 为独立交叉验证；DenseKP 直接 spherical projection。

## Phase 5 完成状态

Perspective P0 closed loop 已于 2026-09-05 完成：16 帧 procedural SMPL-X、Blender 单进程 RGB/depth/mask、DenseKP 138、解析投影对比、visibility、overlay、manifest 与 pyrender cross-check 全部通过。实测数据见 `P0_PERSPECTIVE_CLOSED_LOOP.md`、坐标矩阵见 `P0_COORDINATE_CONVENTION.md`、depth/visibility 见 `P0_DEPTH_VISIBILITY.md`。

## 后续 blocker

procedural animation P0 已无硬阻塞。进入真实 motion 验证还需要：

1. 指定一段可用、已授权且参数定义明确的 SMPL-X motion NPZ；
2. 将当前共享基础包的 `densekp_render` venv 收敛为完全独立、可锁定的正式环境；
3. 用真实 motion 重跑既有 projection/depth/visibility 回归测试；
4. 本阶段仍不修改训练 loss。

## 下一步

选择现有 tiny manifest 中需求最少的一段 `motion_seq.npz`；获得该单个 motion 后，在 Blender 常驻进程内渲染 8--32 帧 perspective RGB/depth/mask，并沿用本轮已通过的同顶点 DenseKP overlay 验证。无需 Unreal。
