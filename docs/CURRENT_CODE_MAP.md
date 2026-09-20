# 当前代码地图（CURRENT CODE MAP）

> 更新日期：2026-09-20。本文档基于对仓库实际文件与调用链的检查，不是按文件名猜测。
> Phase 4 工作树基线 commit：`0b0e307`。

## 0. 一句话总结

本仓库 = **从 CameraHMR 摘出的 DenseKP 训练代码副本**（`densekp/`、`configs/hydra/`）
+ **一套围绕 Blender 渲染的 synthetic DenseKP GT 生成/验证工具**（`tools/`）
+ **少量第三方渲染/重定向工具**（`third_party/`，gitignored，各自带独立 .git）。

## 1. 顶层目录树

```text
densekp_retrain/
├── densekp/                    # 训练代码（源自 CameraHMR，包路径改为 densekp.core）
│   ├── train_keypoints.py      # 训练入口（Hydra + Lightning）
│   └── core/
│       ├── densekp_trainer.py  # LightningModule：DenseKP（backbone+head+loss）
│       ├── constants.py        # 常量：路径、138 点数、翻转置换、均值方差
│       ├── configs/__init__.py # DATASET_FOLDERS / DATASET_FILES 注册表 + 默认 yacs 配置
│       ├── datasets/           # DatasetKeypoints / DataModule / 裁剪与增强 utils
│       ├── backbones/vit.py    # ViTPose-H ViT backbone（vit() / create_backbone()）
│       ├── heads/smpl_head_keypoints.py   # KeypointsHead：单 token Transformer → 138×3
│       ├── components/         # pose_transformer.py（TransformerDecoder）、t_cond_mlp.py
│       └── utils/              # pylogger / rich_utils / train_utils / misc
├── configs/
│   ├── hydra/                  # 训练 Hydra 配置（train.yaml、data/、experiment/、trainer/…）
│   ├── datasets.example.json   # 数据路径示例（当前无代码读取，见 POTENTIAL_ISSUES）
│   ├── fisheye_scene_p0.json   # Phase 18 场景资产 spec（场景 .blend 的摆放约定）
│   └── perspective_camera_stress.json  # 8 个确定性相机压力配置
├── tools/                      # 全部 synthetic pipeline 脚本（本项目新增，2026-09-19 整理后）
│   ├── generate/               # SMPL-X motion → 几何制品（vertices/faces/densekp_3d）
│   ├── render/                 # Perspective Blender 渲染 + 后处理 + 相机压力
│   ├── fisheye/                # Omni-Radtan 相机模型 + ERP 源渲染 + fisheye pipeline + 单测
│   ├── annotation/             # DenseKP GT / 标注生成与检查（BEDLAM tiny 链）
│   ├── scene/                  # .blend 场景 build / validate / inspect 脚本
│   ├── validate/               # 验证与冒烟测试
│   ├── visualize/              # 对照图 / review 生成
│   └── experiments/            # 一次性审计、运动数据审计、sweep/验证性生成
├── scripts/                    # 工作区审计 + BEDLAM 纹理下载 shell 脚本
├── third_party/                # BEDLAM 官方 render / retargeting / clothing 工具（gitignored）
├── assets/                     # densekp_regressor_R.npy（138×10475 算子）、场景资产、示例
├── datasets/                   # 生成数据放置说明 + densekp_bedlam_tiny 产物（gitignored 内容）
├── outputs/                    # 所有渲染/验证产物（gitignored，6.7 GB）
├── docs/                       # 既有审计文档 + 本批新增文档
├── scripts/audit_workspace.sh  # 快速检查：表面算子 + 全仓库语法 + >100MB 文件
├── pyproject.toml / requirements.upstream.txt
└── README.md
```

## 2. 哪些代码属于 Original CameraHMR，哪些是本项目新增

| 范围 | 内容 | 来源 |
|---|---|---|
| **Original CameraHMR（DenseKP 训练子集）** | `densekp/` 全部、`configs/hydra/` 全部、`pyproject.toml`、`requirements.upstream.txt` | CameraHMR 上游 commit `65d11a0`（训练代码最早见 `0ca750f`），仅做包路径清理（见 README「与上游代码的差异」） |
| **本项目新增（synthetic data / retraining 准备）** | `tools/` 全部、`scripts/`、`docs/` 既有审计文档、`configs/*.json`、`assets/`、`datasets/README.md`、`outputs/`、`docs/TODO/TODO.md` | 本项目 |
| **第三方（不修改）** | `third_party/bedlam2_render`、`bedlam2_retargeting`、`bedlam_clothing`、`bedlam_clothing_download`、`bedlam_render`、`clothing_data_scripts.zip` | BEDLAM 官方仓库镜像，各有独立 .git |

要点：**`tools/` 下没有任何一个文件属于 CameraHMR 原始代码**；**`densekp/` 下没有任何一个文件是本项目新写的数据生成代码**（仅两处已记录的清理，见 README）。

## 2.1 Original CameraHMR 代码明细（densekp/）

| 文件 | 作用 | 关键入口 |
|---|---|---|
| `densekp/train_keypoints.py` | 训练入口：pyrootutils 定位根目录 → Hydra 读取 `configs/hydra/train.yaml` → `dataset_config()` → `DataModule` + `DenseKP` → Lightning `trainer.fit` | `main()` / `train()` |
| `densekp/core/densekp_trainer.py` | LightningModule。`forward_step`：图像 `[:,:,:,:,32:-32]` 宽度裁剪到 192 → backbone → head 输出 `[B,138,3]`。`compute_loss`：异方差 xy 损失（`with_var`）或纯 MSE，整体 ×60。`validation_step`：用 `batch['_trans']` 把原图坐标 `proj_verts` 变换到裁剪坐标再 `/256-0.5` | `DenseKP` |
| `densekp/core/datasets/dataset_keypoints.py` | `DatasetKeypoints`：读 NPZ（`imgname/scale/center/part|gtkps/proj_verts`），与图像共享同一仿射裁剪 | `DatasetKeypoints.__getitem__` |
| `densekp/core/datasets/datamodule_keypoints.py` | `DataModule`：按 `DATASETS_AND_RATIOS`（下划线切分）拼接训练集；`VAL_DATASETS` 建多个 val loader | `DataModule` |
| `densekp/core/datasets/utils.py` | **裁剪/增强核心**：`expand_to_aspect_ratio`、`do_augmentation`、`gen_trans_from_patch_cv`、`generate_image_patch_cv2`、`get_example_projverts`（图像与 proj_verts 共享同一 trans，然后 `xy/256-0.5`）、`extreme_cropping*`、`get_bbox` | `get_example_projverts` |
| `densekp/core/configs/__init__.py` | 数据集注册表（硬编码路径 `data/...`）+ yacs 默认配置 | `DATASET_FOLDERS`、`DATASET_FILES`、`default_config()` |
| `densekp/core/heads/smpl_head_keypoints.py` | `KeypointsHead`：token 图展平 → 6 层 TransformerDecoder → `nn.Linear(dim, 138*3)` | `build_keypoints_head()` |
| `densekp/core/backbones/__init__.py` + `vit.py` | `create_backbone()` = `vit()`（ViTPose-H 结构）；权重由 `densekp_trainer` 从 `VITPOSE_BACKBONE` 加载 | `create_backbone` |
| `densekp/core/components/pose_transformer.py` | `TransformerDecoder`（head 使用的交叉注意力解码器） | — |
| `densekp/core/constants.py` | `NUM_DENSEKP_SMPL=138`、`FLIP_KEYPOINT_PERMUTATION`、`IMAGE_SIZE=256` 等；模型/权重路径常量 | — |
| `densekp/core/utils/` | `train_utils.trans_points2d_parallel`（val 用）、`misc.task_wrapper`、日志工具 | — |

**缺失但被 import 的模块**：`densekp/core/losses/`（`densekp_trainer.py:12` `from .losses import Keypoint2DLoss`）——本仓库没有该文件，import 即失败。见 `POTENTIAL_ISSUES.md` #1。

**已知上游死代码**：`datasets/utils.py::generate_image_patch_skimage`、`get_transform/transform`（PARE 风格）只有 skimage 路径在用；`DatasetKeypoints.__init__` 的 `use_augmentation` 参数实际未被使用（增强开关实际是 `is_train`）。

## 2.2 本项目新增代码明细（tools/ 等）

### A. 几何生成（SMPL-X → vertices / faces / densekp_3d）

| 文件 | 用途 | 状态 |
|---|---|---|
| `tools/generate/generate_p0_toy_motion.py` | 确定性 16 帧 toy motion → SMPL-X 前向 → `vertices.npy/faces.npy/densekp_3d.npy/motion_params.npz/geometry_manifest.json`（几何制品的标准目录布局） | **在用**（P0 基线几何来源） |
| `tools/generate/load_real_smplx_motion.py` | 真实 SMPL-X motion（`poses[T,165]`）→ canonical 参数 NPZ + 帧映射 | **在用** |
| `tools/generate/generate_real_motion_geometry.py` | canonical motion → SMPL-X 前向 → 几何制品；`densekp_3d = einsum('kv,tvc->tkc', R, V)` | **在用**（当前主路径） |
| `src/densekp_retrain/body/geometry.py` | 两个 geometry CLI 共用的 SMPL-X 创建、shape validation、原始 `einsum` 和数组写入 | **在用（库）** |
| `tools/experiments/generate_cyclic_validation_motion.py` | 验证用确定性 cyclic gait（非真实 mocap） | 实验/验证 |
| `tools/generate/prepare_p0_dressed_geometry.py` | BEDLAM 6fps clothing NPZ → `clothing_vertices/faces/uv` + manifest（固定映射 clothing_id*5, source=101+…） | **在用** |

### B. Blender 渲染（RGB / Depth / Mask）

| 文件 | 用途 | 状态 |
|---|---|---|
| `tools/render/render_p0_blender.py` | **核心渲染器**。必须在 Blender 内运行（`import bpy`）。加载几何，应用 `T_smpl_to_blender` + Z 抬升，渲染 RGB PNG 与 Depth/Mask multilayer EXR；同时解析计算 `densekp_2d` 并与 `world_to_camera_view` 交叉验证；写 `manifest.jsonl`（相机/内参/外参/物体矩阵）与 `annotations/densekp_2d.npy`、`annotations/densekp_camera_xyz.npy`；含 depth 语义平面测试与 `--camera-placement` | **在用，入口** |
| `tools/render/eye_background_render.py` | bpy 侧库：procedural 眼睛材质 + 线性差分阴影合成（被 render_p0_blender import） | **在用（库）** |
| `tools/render/finalize_p0_outputs.py` | 渲染后处理：EXR→NPY depth、mask PNG、自动判定 depth 语义（CAMERA_SPACE_Z / EUCLIDEAN）、visibility 状态 0-3、容差、overlay/contact sheet、`annotations/densekp_visible.npy` 等 | **在用** |
| `tools/render/run_eye_background_fix.py` | 包装器：用冻结的外观参数调 `render_p0_blender.py`；Blender 和服装纹理路径由 CLI/环境配置 | **在用** |
| `tools/render/run_temporal_case.py` / `tools/render/finalize_temporal_case.py` | 渲染一个时间序列 case / 生成视频与时间统计 | **在用** |
| `tools/validate/p0_pyrender_crosscheck.py` | pyrender EGL 独立渲染 0/8/15 帧 silhouette，与 Blender mask 求 IoU | 验证 |
| `tools/validate/blender_headless_smoke_test.py` / `pyrender_headless_smoke_test.py` | 无头渲染环境冒烟测试 | 环境验证 |

### C. 相机压力测试（Perspective：近距/边缘/截断）

| 文件 | 用途 | 状态 |
|---|---|---|
| `tools/render/solve_camera_stress_placement.py` | 由 `configs/perspective_camera_stress.json` 用 `least_squares` 求每帧相机平移（内参固定） | **在用** |
| `tools/render/run_camera_stress.py` | 8 个配置逐个调用 `run_eye_background_fix.py --camera-placement` | **在用** |
| `tools/render/finalize_camera_stress.py` | 汇总 128 样本注记（densekp_3d/2d/depth/valid/in_frame/visible/visibility_state + manifests）与验收 gate | **在用** |

### D. Fisheye / ERP / 场景

| 文件 | 用途 | 状态 |
|---|---|---|
| `src/densekp_retrain/camera/omni_radtan.py` | Omni-Radtan 相机模型（NumPy）：`project_points_omni_radtan` / `unproject_pixels_omni_radtan`，CV 相机坐标约定 | **在用（核心）** |
| `tests/test_omni_radtan.py` | 单元测试：光轴、对称性、roundtrip（`python -m unittest tests.test_omni_radtan`） | **在用** |
| `tools/fisheye/render_equirectangular_source.py` | Blender 内：HDRI-only ERP 源渲染（Combined/Depth/IndexOB） | **在用** |
| `tools/fisheye/render_equirectangular_scene_source.py` | Blender 内：带静态 .blend 场景的 ERP 源渲染（Phase 18），import `tools/scene/blender_scene_bridge.py` | **在用** |
| `tools/fisheye/run_fisheye_camera_p0.py` | ERP→fisheye warp（`warp_case`：remap、mask/depth、Omni-Radtan 投影 DenseKP、4cm 深度可见性、注记 npy、report） | **在用（核心）** |
| `src/densekp_retrain/camera/extrinsics.py` | `pose()` 相机位姿构造，供 fisheye CLI 共用 | **在用（库）** |
| `tools/fisheye/run_fisheye_scene_p0.py` | Phase 18：场景 spec 校验（缺失则报 SCENE_ASSET_MISSING）→ 场景渲染 → 复用 warp_case | **在用** |
| `tools/fisheye/run_fisheye_seam_sweep.py` / `run_human_scale_sweep.py` | 相机 yaw / 人像尺度扫描（复用 Phase 17 pipeline） | 实验 |
| `src/densekp_retrain/scene/spec.py` | 场景资产 spec（dataclass + 校验，不含资产本体） | **在用（库）** |
| `tools/scene/blender_scene_bridge.py` | bpy 侧：链接 .blend、可见性/地面控制、场景包围盒 | **在用（库）** |
| `tools/scene/build_cycling_forest.py` / `build_reference_riverfront.py` / `prepare_hidden_alley.py` | 在 Blender 中构建骑行森林 / 河岸参考 / hidden alley 场景（需在 Blender 内或带 .blend 运行） | 场景资产制作 |
| `configs/fisheye_scene_p0.json` | 场景资产 spec（body 放置、ground_mode 等） | **在用** |

### E. DenseKP 标注生成与检查（喂给训练）

| 文件 | 用途 | 状态 |
|---|---|---|
| `tools/annotation/generate_bedlam_densekp.py` | 从少量 BEDLAM 几何记录生成 `DatasetKeypoints` 可读 NPZ（imgname/center/scale/proj_verts）；第三通道 = BEDLAM 官方齐次常数 1 | **在用（唯一"标注→训练格式"生成器）** |
| `tools/validate/validate_bedlam_densekp_tiny.py` | 用真实 `DatasetKeypoints` 读取 tiny NPZ、执行裁剪、可视化验证 | **在用（GT↔训练闭环验证）** |
| `tools/annotation/inspect_annotation.py` | 只读检查任意 DenseKP NPZ（结构/数值/第三通道分布） | **在用** |
| `tools/annotation/visualize_annotation.py` | 把 NPZ 的 proj_verts 画回原图与无增强裁剪图 | **在用** |
| `tools/annotation/inspect_surface_operator.py` | 导出/打印 138 行算子的顶点 ID 与权重 | **在用** |
| `tools/annotation/build_bedlam_tiny_frame_manifest.py` | BEDLAM v1 官方状态机 frame→person 映射 | 数据准备 |
| `tools/validate/validate_p0_textured_regression.py` | 纹理接入前后的几何/监督回归比对 | 验证 |

### F. 外观 / 眼睛 / 纹理审计（历史 phase，冻结使用）

`tools/experiments/audit_body_uv.py`、`audit_clothing_uv.py`、`audit_skin_texture.py`、`audit_eye_geometry.py`、`prepare_procedural_eyes.py`、`make_appearance_comparison.py`、`make_environment_comparison.py`、`make_eye_fix_comparison.py`、`make_skin_texture_review.py`、`review_eye_background_fix.py` —— 保留的审计/对照图生成器。`prepare_procedural_eyes.py` 仍是当前 eye asset 的生成来源；`audit_eye_geometry.py` 仍有代码调用方。

### G. 运动数据审计（骑行/BABEL/BVH）

`audit_cyclist_bvh.py`、`find_cyclist_clip.py`、`search_babel_cycling.py`、`review_cyclist_audit.py` —— 可复现当前骑行候选检查的只读工具，不渲染。旧 local-inventory audit 的结论已固化在 `CYCLING_MOTION_AUDIT.md`，脚本已删除。

### H. 外部脚本与资产

| 路径 | 用途 |
|---|---|
| `scripts/audit_workspace.sh` | 表面算子检查 + 全仓 Python 语法检查 + >100MB 文件清单（无需数据/权重即可跑） |
| `scripts/download_bedlam_*.sh` | BEDLAM 纹理/衣物下载脚本 |
| `assets/densekp_regressor_R.npy` | **DenseKP 138 点线性顶点回归算子 [138,10475] float32**（gitignored 内容，但被 5 个工具引用） |
| `assets/downsample_mat_smplx.pkl` | 上游遗留下采样矩阵，本仓库训练路径未使用 |
| `assets/scenes/` | 河岸场景 .blend1 / photoreal hidden_alley 分片 |
| `third_party/bedlam2_render` 等 | BEDLAM 官方 Unreal/Blender 渲染与重定向工具，未被本项目 Python 直接 import，仅作参考与离线流程 |

## 3. 当前真正的 entry point

| 目的 | 命令 |
|---|---|
| 训练 DenseKP | `python densekp/train_keypoints.py data=densekp_train experiment=densekp exp_name=...`（当前不可运行：缺 `densekp/core/losses`、`VITPOSE_BACKBONE` 权重、`data/` 标注） |
| 生成 toy 几何 | `python tools/generate/generate_p0_toy_motion.py --model-dir <SMPLX_NPZ dir>` |
| 真实 motion → 几何 | `python tools/generate/load_real_smplx_motion.py <npz> --output ...` → `python tools/generate/generate_real_motion_geometry.py --motion ... --model-dir ...` |
| Perspective 渲染 | Blender 内跑 `tools/render/render_p0_blender.py`（建议经 `tools/render/run_eye_background_fix.py` 封装） |
| 渲染后处理 + visibility | `python tools/render/finalize_p0_outputs.py --output-dir <render dir>` |
| 相机压力集 | `python tools/render/solve_camera_stress_placement.py` → `python tools/render/run_camera_stress.py` → `python tools/render/finalize_p0_outputs.py`（每渲染目录）→ `python tools/render/finalize_camera_stress.py` |
| Fisheye P0 | `python tools/fisheye/run_fisheye_camera_p0.py`（内部再调 Blender ERP 源渲染） |
| Fisheye + 场景 | `python tools/fisheye/run_fisheye_scene_p0.py`（需 `configs/fisheye_scene_p0.json` 指向的 .blend） |
| 训练格式标注生成（BEDLAM tiny） | `python tools/annotation/generate_bedlam_densekp.py <input.npz> --image-root ...` |
| GT 验证 | `python tools/validate/validate_bedlam_densekp_tiny.py` / `inspect_annotation.py` / `visualize_annotation.py` |
| 快速检查 | `bash scripts/audit_workspace.sh`；`python -m unittest discover -s tests` |

## 4. debug / legacy / experimental 分类

- **核心可复用库**：`src/densekp_retrain/{body,runtime,io,camera,annotation,rendering,scene}`；尚未迁移的高风险实现为 `tools/fisheye/run_fisheye_camera_p0.py::warp_case`、`tools/render/eye_background_render.py` 和 `tools/scene/blender_scene_bridge.py`。
- **一次性审计（历史 phase 产物，保留不删）**：第 2.2 F/G 类全部。
- **实验脚本（验证性 sweep）**：`run_fisheye_seam_sweep.py`、`run_human_scale_sweep.py`、`generate_cyclic_validation_motion.py`。
- **没有发现** `old_xxx / xxx_final / xxx_v2` 命名的可删脚本；`eye_fix_v2` 是输出目录名而非脚本。
- **遗留无效代码（上游原样，保留）**：`densekp/core/datasets/utils.py::generate_image_patch_skimage`（无调用方）、`DatasetKeypoints.use_augmentation` 形参（未使用）、`densekp/core/utils/rich_utils.py` 等（未验证是否被引用）。

## 5. 数据生成 pipeline（一页流）

```text
[BEDLAM 6fps clothing NPZ] ─ prepare_p0_dressed_geometry.py ─┐
[SMPL-X model NPZ] ─ generate_p0_toy_motion.py 或             │
  load_real_smplx_motion.py → generate_real_motion_geometry.py┤→ geometry/ (vertices, faces,
      │  densekp_3d = R @ V (assets/densekp_regressor_R.npy)  │   densekp_3d, motion_params, manifest)
      ▼                                                       ▼
      └──────────────→ tools/render/render_p0_blender.py (Blender headless)
                              │  T_smpl_to_blender + Z-lift；RGB PNG / Depth+Mask EXR
                              │  解析投影 densekp_2d ↔ world_to_camera_view 交叉验证
                              ▼
                       tools/render/finalize_p0_outputs.py
                              │  EXR→NPY depth、mask、visibility 状态 0-3
                              ▼
              RGB / Depth / Mask / annotations（可训练样本的原料）
```

## 6. training pipeline（一页流）

```text
configs/hydra/train.yaml ─ densekp/train_keypoints.py
        │ dataset_config()（空配置壳）+ cfg.DATASETS.DATASETS_AND_RATIOS
        ▼
densekp/core/datasets/datamodule_keypoints.py::DataModule
        ▼
densekp/core/datasets/dataset_keypoints.py::DatasetKeypoints
   读 NPZ: imgname/center/scale/proj_verts(+part|gtkps)
        ▼
densekp/core/datasets/utils.py::get_example_projverts
   共享仿射 crop + 归一化（xy/256-0.5）
        ▼
densekp/core/densekp_trainer.py::DenseKP
   backbones/vit.py (ViTPose-H, 输入 256×192)
   heads/smpl_head_keypoints.py (TransformerDecoder → [B,138,3])
        ▼
compute_loss（异方差 xy，×60）→ AdamW
```

数据集注册表在 `densekp/core/configs/__init__.py`（硬编码，`configs/datasets.example.json` 目前没有代码消费它，见 POTENTIAL_ISSUES #2）。

## 7. 已验证的回归基线（整理时必须保持一致）

`outputs/`（gitignored）内含已验证样本，报告链见既有文档：

- `outputs/baselines/p0_perspective/` — 16 帧 perspective 闭环（`rendering/P0_PERSPECTIVE_CLOSED_LOOP.md`）；
- `outputs/baselines/p0_dressed_eye_background_fixed/` — 当前外观基线，被 `run_eye_background_fix.py`、`run_fisheye_*`、`run_temporal_case.py` 引用为 manifest 来源；
- `outputs/baselines/fisheye_camera_p0/regression_report.json` — Phase 17 fisheye 基线（`densekp_3d_fingerprint` 被 Phase 18 比对）；
- `outputs/experiments/perspective_camera_stress/` — 128 样本相机压力集；
- `datasets/densekp_bedlam_tiny/` — 训练格式闭环产物。

任何"整理"都不应改变这些产物的数值链路（渲染器、投影、crop、visibility）。
