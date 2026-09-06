# Headless synthetic renderer 选型

## 决策原则

评价优先级为：几何真值一致、camera model 可控、headless/offscreen、Linux 批量运行、SMPL-X motion、depth/mask/visibility、鱼眼/ERP、效率、外观，最后才是电影级画质。

结论不是三选一：

- **Blender：正式 synthetic dataset 主 renderer。**
- **pyrender：几何、投影、annotation 与快速回归测试 renderer。**
- **Unreal：仅在确实需要 BEDLAM2 高保真衣服、Groom、鞋或 Unreal-only scene 时启用，不是 P0 前置条件。**

## 直接比较

| 能力 | BEDLAM2 + Unreal | Blender headless | pyrender / trimesh / EGL |
|---|---|---|---|
| Headless Linux | 官方主流程要求 Windows 11 + UE 5.3.2，不符合当前首选部署 | `-b` 在本机无 DISPLAY 实测通过 | EGL 在本机无 DISPLAY、RTX 3090 实测通过 |
| Offscreen | MRQ 可自动渲染，但官方工作流依赖完整 Unreal 工程/插件 | Cycles/Eevee 后台渲染 | `OffscreenRenderer` 原生返回 RGB/depth |
| RTX 3090 | Windows 环境理论可用，当前未测 | Cycles OptiX 实测通过 | NVIDIA EGL/OpenGL 实测通过 |
| SMPL-X | 通过 Blender 预处理成 ABC/FBX/GeometryCache | 可直接用 Python 生成 mesh，或复用官方 NPZ->ABC 脚本 | 可直接接收逐帧 10475 顶点和固定 faces |
| Motion | LevelSequence + GeometryCache；资产制作链较长 | shape keys、mesh update、ABC 均可；脚本化完整 | 每帧更新/替换 vertex buffer；简单直接但 Python/OpenGL 管理需优化 |
| Texture/PBR | 视觉能力最强，BEDLAM2 材质资产可用 | Principled BSDF、UV、texture、节点材质完整 | glTF 2.0 metallic-roughness、纹理/normal map 可用，但材质能力有限 |
| Clothing | 可直接利用 BEDLAM2 预模拟 GeometryCache | 可导入 clothing ABC/mesh；新衣服仍需模拟或现成 cache | 可渲染现成 cloth mesh；不负责 cloth simulation |
| Hair / shoes | BEDLAM2 Groom 与 toeless shoe 资产是优势 | 可用 mesh/curve/particle，但不能直接等价复用所有 Unreal asset | 简单 hair/clothing mesh 可以；高保真 Groom 不合适 |
| Scene / HDRI | Unreal level、HDRI、生态最丰富 | HDRI、简单/复杂 scene、glTF/FBX/OBJ/USD 导入均成熟 | 多 mesh 可用，适合中小型场景，不适合大型内容生态 |
| Depth | EXR pass | Z pass / compositor / multilayer EXR | 线性 depth 直接返回，已实测 |
| Mask / object ID | cryptomatte/layer | object/material index、Cryptomatte | `SEG` + node map 已实测；多类需维护颜色映射 |
| Triangle / vertex ID | 需自定义 pass | 可定制材质/AOV；实现成本中等 | 默认无 triangle/vertex ID，需要编码材质或自定义 shader/render pass |
| Visibility GT | depth/mask 可做，但坐标桥接复杂 | 同一 mesh、相机与 depth，容易保持一致 | 同一 NumPy vertices 和 depth，最容易调试 |
| Camera control | perspective CineCamera 强；BEDLAM2 JSON 完整 | camera matrix、intrinsics、projection、panorama 可完全脚本化 | perspective intrinsics 完全可设；非线性 lens 不原生 |
| Custom fisheye | BEDLAM 未集成；需插件/postprocess/cubemap | Cycles 原生有 equidistant、equisolid、polynomial fisheye；其他模型可用 cubemap/custom camera | 不原生；应 cubemap/multi-view + ray reprojection，或写 shader |
| ERP | BEDLAM 未集成 | Cycles 原生 `EQUIRECTANGULAR` panoramic camera | 不原生；六面透视/cubemap 后映射 ERP |
| Batch generation | MRQ 强，但 setup 重且 Windows-only official route | 很适合；常驻/一次加载多帧可摊薄启动 | 小中型 geometry batch 很快；复杂 scene 与资源管理上限较低 |
| GT exactness | 可做，但 NPZ->ABC->UE 与 temporal subframe 桥接风险最高 | 直接以同一 Python vertices 构建 mesh，风险低 | smoke test 已证明同一 vertices 分流，风险最低 |
| Setup complexity | 高 | 中；本机现有版本已经通过 GPU test | 低；当前环境已通过，但正式项目应独立环境 |
| Runtime complexity | 高，插件、LevelSequence、MRQ、postprocess | 中，可集中在一个 Python pipeline | 低到中；高级 pass/custom lens 会快速增加维护成本 |
| Visual realism | 最高，尤其 BEDLAM2 clothing/hair/scenes | 高，可逐步增加材质、衣服、HDRI、scene | 中低，更适合几何与 baseline |
| 推荐角色 | P2 可选高保真分支 | **主 renderer** | **debug/annotation oracle 与快速 baseline** |

## Unreal 是否不可替代

Unreal 不提供 DenseKP baseline 的不可替代能力。离开 Unreal 后仍能做到：SMPL-X + motion + 基础材质/现成衣服 mesh + scene/background + perspective/fisheye/ERP + RGB/depth/mask + DenseKP GT。

Unreal 的独特价值主要是 BEDLAM2 Starter Pack 中已准备的 Unreal-only GeometryCache、Groom、shoe/material、HDR/level 资产及 MRQ 生态，它们提高 appearance realism 和复杂场景生产能力。P0 所需的裸体/基础材质 SMPL-X、可控相机、RGB/depth/mask 和 138 点并不依赖这些资源。

## Blender 能力与复用边界

- Linux 无 GUI可运行，当前机器已实测，无需 X server。
- Cycles CUDA/OptiX 可直接使用 RTX 3090；Eevee 后台也通过。
- Python 可创建/导入 mesh、animation、camera、lighting、material，并输出 RGB、depth、mask、normal、object/material ID。
- 可逐帧读取 `matrix_world`、相机矩阵和 depsgraph evaluated mesh 顶点；也可直接让 SMPL-X Python 顶点成为渲染 mesh，从源头消除近似 pose。
- 数万帧适合一次加载多帧渲染；进程级并发要按 GPU memory 限流。
- 可复用 BEDLAM v1/v2 的 `blender/smplx_anim_to_alembic/`，但 P0 更推荐跳过 Alembic，直接用 SMPL-X 参数生成逐帧 vertices。只有需要把动画交给 DCC/Unreal、复用预模拟 cache 或降低 Python SMPL-X 计算耦合时才需要 ABC。

## pyrender 角色与限制

pyrender 支持多个 mesh、显式内参、PBR texture、RGB、linear depth 和 node segmentation；当前简单场景约 2.2 ms/帧。动画 SMPL-X 可每帧构建新 `pyrender.Mesh`，但这样会反复上传顶点；正式优化应复用 scene/context 并研究更新 GPU primitive buffer，或者按短 chunk 构建。默认 API 不提供布料、复杂 DCC 节点、triangle ID、vertex ID、鱼眼或 ERP。

因此它非常适合：

1. 验证 SMPL-X/相机/坐标轴；
2. 生成最简单的 P0 baseline；
3. 作为 Blender 输出的 annotation oracle；
4. 运行低成本 regression test。

它不应直接承担未来高保真大型场景的全部责任。

## Fisheye 方案

当前 Blender 5.1.1 实际枚举出 `FISHEYE_EQUIDISTANT`、`FISHEYE_EQUISOLID`、`FISHEYE_LENS_POLYNOMIAL`，因此前两类可优先用 Cycles 原生 ray camera。Kannala-Brandt、Unified/Omni、Double Sphere 和项目自定义模型不应强塞进近似 polynomial；推荐从同光心 cubemap 或 Blender custom camera/ray shader生成 RGB，同时让 DenseKP 3D 用完全相同公式投影。

简单二维 warp 只重新采样已有透视图：在大 FOV、近景、出视锥和 disocclusion 区域没有源像素，不能恢复正确遮挡。基于 depth 的 reprojection 能按每个源像素反投影到 3D 并 z-buffer 到目标相机，能处理一部分重遮挡，但单视图仍无法补回原本不可见表面。可靠路径是直接投射目标 rays，或从同光心、同一时刻、覆盖完整方向的 cubemap 重采样。

## ERP 方案

Blender Cycles 原生 `EQUIRECTANGULAR` camera 是主路线，RGB rays 和场景遮挡由 renderer 直接求解；DenseKP 从同一 camera-space 3D 直接用 longitude/latitude 投影。作为交叉验证，可另做同光心六面 cubemap -> ERP。pyrender 若参与，只应渲染六面 perspective，必须共享同一光心和帧，再做方向采样；不能把普通 perspective 2D 点直接 warp 成 ERP。

## Visibility / occlusion 设计

每个 DenseKP 点先分类：

1. `outside`：投影无效、位于相机模型不可用域、落出图像/鱼眼有效圆；ERP 横向是周期坐标，不按普通左右越界处理。
2. `candidate`：投影有效，计算该点沿相机 ray 的深度 `d_point`。
3. 在同一像素邻域采样 renderer depth `d_surface`，并采用随深度/像素足迹设定的容差 `epsilon`。
4. `visible`：`|d_point - d_surface| <= epsilon`；`occluded`：`d_point > d_surface + epsilon`。
5. 借助 body instance/object/triangle ID，将 `occluded` 再分成 self-occluded、被其他人遮挡、被场景遮挡。

非透视相机必须比较同一种深度。ERP/鱼眼推荐比较 radial distance；若 renderer 输出 camera-Z，必须先转换，不能直接混用。对由三角面重心定义的 DenseKP 点，还应考虑采样像素、抗锯齿和点不一定恰落在可见三角形中心的问题。

## Scene / SimWorld 接口

P0 只需 SMPL-X 基础材质、世界背景、一个灯和可选地面。Blender 后续可渐进加入 HDRI、背景图、textured plane、简单 indoor/outdoor geometry、随机灯光、skin texture、hair mesh 与现成 clothing mesh。

不假设 SimWorld 内部实现。如果它能导出 mesh、glTF、FBX、OBJ、USD 或 Blender-compatible scene，Blender 通常容易接入；需额外统一单位、坐标轴、材质、实例和碰撞。若只能导出 Unreal level，且场景转换成本高于其价值，或必须复用 Unreal 专有材质/蓝图/交互，才值得回到 Unreal 分支。
