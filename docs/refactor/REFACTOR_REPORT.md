# REFACTOR_REPORT（readability / 整理报告）

日期：2026-09-19　基线 commit：`d12b963`（master，工作树干净，无未提交修改）

## 1. 阅读了哪些文件

**训练侧（逐行）**：`densekp/train_keypoints.py`、`densekp/core/densekp_trainer.py`、
`densekp/core/datasets/{dataset_keypoints,datamodule_keypoints,utils}.py`（utils 1100 行全读）、
`densekp/core/configs/__init__.py`、`densekp/core/{constants,backbones/__init__,heads/smpl_head_keypoints}.py`、
`configs/hydra/{train.yaml,data/densekp_train.yaml,experiment/densekp.yaml,trainer/default.yaml}`。

**数据生成侧（逐行）**：`tools/render/render_p0_blender.py`、`tools/generate/generate_real_motion_geometry.py`、
`tools/generate/generate_p0_toy_motion.py`（头 50 行）、`tools/generate/load_real_smplx_motion.py`、
`tools/generate/prepare_p0_dressed_geometry.py`、`tools/render/finalize_p0_outputs.py`、`tools/render/eye_background_render.py`（头部）、
`tools/fisheye/{omni_radtan,test_omni_radtan,render_equirectangular_source,render_equirectangular_scene_source}.py`、
`tools/fisheye/run_fisheye_camera_p0.py`、`tools/fisheye/run_fisheye_scene_p0.py`、`tools/fisheye/run_fisheye_seam_sweep.py`、
`tools/fisheye/run_human_scale_sweep.py`、`tools/scene/{scene_loader,blender_scene_bridge}.py`、
`tools/annotation/generate_bedlam_densekp.py`、`tools/{inspect_annotation,inspect_surface_operator,solve_camera_stress_placement,run_camera_stress,run_eye_background_fix,run_temporal_case,finalize_camera_stress,finalize_p0_outputs}.py`。

**粗读（docstring + 结构）**：tools/ 其余 40+ 脚本（见 `CURRENT_CODE_MAP.md` §2.2 分类）、
`scripts/audit_workspace.sh`、`pyproject.toml`、`README.md`、`docs/TODO/TODO.md`。

**既有文档（用于交叉核对，不新写结论）**：`TRAINING_AUDIT.md`、`ANNOTATION_AUDIT.md`、
`DATA_FORMAT.md`、`P0_COORDINATE_CONVENTION.md`、`P0_DEPTH_VISIBILITY.md`。

## 2. 当前项目架构

见 `docs/CURRENT_CODE_MAP.md`（目录/文件级）与 `docs/ARCHITECTURE.md`（箭头级数据流）。
一句话：`densekp/`+`configs/hydra/` = Original CameraHMR DenseKP 训练子集；
`tools/` = 本项目新增的 synthetic pipeline（几何生成 / Blender 渲染 / 视角与可见性 / 训练格式生成）；
`third_party/` = BEDLAM 官方工具镜像（不修改）。

## 3. 移动了哪些文件

**没有移动任何文件。**

理由：
1. `tools/` 内不存在 `debug_/test_/old_` 命名的废弃脚本（`test_omni_radtan.py` 是唯一真实单测，且被 `run_fisheye_scene_p0.py:62` 以 `python -m unittest tools.fisheye.test_omni_radtan` 子进程引用，移动必然破坏 entry point）；
2. 多个脚本依赖 `sys.path.insert(0, Path(__file__).parent)` 的相对导入（如 `run_fisheye_scene_p0` ← `run_fisheye_camera_p0`、`render_equirectangular_*` ← `tools/scene/`），移动需连锁改 import，违反"最小改动"；
3. 分类价值已通过 `CURRENT_CODE_MAP.md` 的分类表（核心库 / entry / 一次性审计 / 实验）实现，读者按表查阅即可。

**结论**：文件组织保持原样，只新增文档。

## 4. 修改了哪些 import / 代码

**零代码修改。** `git status` 确认：除新增 docs 文件外无任何改动（见下）。

```text
$ git status --porcelain
?? docs/ARCHITECTURE.md
?? docs/CURRENT_CODE_MAP.md
?? docs/COORDINATE_SYSTEMS.md
?? docs/DENSEKP_GT_PIPELINE.md
?? docs/POTENTIAL_ISSUES.md
?? docs/TRAINING_DATA_FORMAT.md
?? REFACTOR_REPORT.md
```

`git diff` 为空（没有任何 tracked 文件被修改）。

## 5. 哪些代码完全没有碰

全部代码。特别按约定只分析未改动的部分：SMPL-X local→world（`render_p0_blender.py:133`）、
world→camera、Blender 坐标转换、Perspective/ERP/Omni-Radtan 投影（`omni_radtan.py` 原样）、
fisheye remap、DenseKP 3D→2D、bbox/center/scale、crop transform（`datasets/utils.py` 原样）、
256×256 训练坐标变换、depth-based visibility（`finalize_p0_outputs.py`、`warp_case`）、
`DenseKP.compute_loss`、DatasetKeypoints、网络结构。所有已有输出与数据集未触碰。

## 6. 发现但未修改的问题

见 `docs/POTENTIAL_ISSUES.md`（12 条）。最重要的三条：

1. `densekp/core/losses` 缺失 → 训练代码 import 即失败（**本次确认的硬事实**）；
2. `configs/datasets.example.json` 无代码消费且 `bedlam-mini` 不在注册表；
3. 数据集名含 `closeup`/`portrait` 会让 RGB 旋转 90° 而标注不跟随 —— 对未来自造数据集是陷阱。

另有 flip↔138 点置换风险、`densekp_valid` 全 1 写法、硬编码路径等，详见该文档。

## 7. 做了哪些验证

| 验证 | 命令 | 结果 |
|---|---|---|
| 全仓库 Python 语法 | `bash scripts/audit_workspace.sh` | `python syntax: PASS` |
| 表面算子完整性 | 同上 | `shape=(138,10475) dtype=float32 nnz=204 row_sum≈1` |
| Omni-Radtan 单测（仓库唯一现成测试） | `python -m unittest tools.fisheye.test_omni_radtan` | 3 tests OK |
| 工具 `--help` / import 冒烟 | inspect_annotation / inspect_surface_operator / generate_real_motion_geometry / generate_bedlam_densekp / visualize_annotation / fisheye+scene 模块 import | 全部 OK |
| **Fisheye 投影数值回归**（已验证样本 `outputs/baselines/fisheye_camera_p0`，frame 0） | 用 `omni_radtan.project_points_omni_radtan` + camera_config + geometry 重投影，与落盘 `densekp_fisheye_2d.npy` 比对 | 3 相机 max diff < **1.3e-4 px**（float32 存储精度） |
| **Perspective 投影数值回归**（基线 `outputs/baselines/p0_dressed_eye_background_fixed`，frames 0/8/15） | 用 manifest `object_matrix_world`/`camera_matrix_world`/intrinsics 重算 densekp_2d、camera_xyz | 2D max diff < **4e-5 px**；camera xyz max diff < **1.7e-7 m** |

由于本次**零代码改动**，"整理前后行为一致"由构造保证；上表进一步验证了当前机器上的静态一致性。

## 8. 哪些验证无法完成

| 项 | 原因 | 状态 |
|---|---|---|
| 训练启动（哪怕 dry-run） | `densekp/core/losses` 缺失（POTENTIAL_ISSUES #1）；`VITPOSE_BACKBONE` 权重缺失（`assets_external/` 不存在）；无 `data/` 标注 | NOT TESTED |
| `import densekp.core.densekp_trainer` | 同上（预期 ModuleNotFoundError，已实跑确认） | 失败原因已定位，未修复 |
| Blender 渲染回归（RGB/Depth/Mask 逐像素） | 渲染一帧需纹理/HDRI/几何齐全且耗时；本次未改渲染代码，重跑 16 帧无信息增益。Blender 二进制确认存在（`/mnt/data1/khb/app/blender-5.1.1-linux-x64/blender`） | NOT RUN（零改动故不声称需要） |
| `run_camera_stress` 全链 | 依赖纹理目录 `/home/yy/datasets/BEDLAM/...` 未验证存在；且渲染耗时长 | NOT RUN |
| `validate_bedlam_densekp_tiny.py` / `visualize_annotation.py` | `datasets/densekp_bedlam_tiny` 当前不存在（无生成产物）；无历史真实标注 NPZ | NOT TESTED |
| `run_fisheye_scene_p0.py` | 依赖 `assets/scenes/outdoor_courtyard.blend`（spec 指向，仓库无此文件，设计上会报 SCENE_ASSET_MISSING） | NOT RUN |
| OpenEXR 依赖 | 当前 conda env（camerahmr）无 `OpenEXR`，`run_fisheye_camera_p0.py` 等 import 即失败（文档记录渲染后处理需独立 Python 环境） | 记录 |

## 9. 代码阅读路线（按此顺序读，不要按目录读）

### Step 1　训练入口（先看消费端，知道 GT 长什么样）
- `densekp/train_keypoints.py::main` → `configs/hydra/train.yaml` → `configs/hydra/data/densekp_train.yaml`
- `densekp/core/datasets/datamodule_keypoints.py::DataModule`

### Step 2　Dataset
- `densekp/core/datasets/dataset_keypoints.py::DatasetKeypoints.__getitem__`
- `densekp/core/datasets/utils.py::get_example_projverts`（整条 crop/增强链的核心，~520-623 行）
- 辅助：`gen_trans_from_patch_cv`、`generate_image_patch_cv2`、`trans_points2d_parallel`

### Step 3　一个 sample 的 annotation（数值层面）
- `tools/annotation/inspect_annotation.py`（拿任意 NPZ 跑一遍）
- `tools/annotation/visualize_annotation.py`（把 proj_verts 画回原图）
- 参考 `docs/DATA_FORMAT.md`

### Step 4　annotation generator（训练格式怎么造）
- `tools/annotation/generate_bedlam_densekp.py::official_bbox / project_official / main`
- `tools/validate/validate_bedlam_densekp_tiny.py`（用真实 DatasetKeypoints 闭环）
- 参考 `docs/TRAINING_DATA_FORMAT.md`

### Step 5　DenseKP 138 correspondence
- `tools/annotation/inspect_surface_operator.py` + `assets/densekp_regressor_R.npy`（[138,10475] 线性顶点回归）
- `tools/generate/generate_real_motion_geometry.py:15`（`R @ V`）
- 参考 `docs/DENSEKP_GT_PIPELINE.md`

### Step 6　camera projection（先 perspective，后 fisheye）
- `tools/render/render_p0_blender.py:165-224`（解析投影 + world_to_camera_view 交叉验证 + manifest）
- `tools/fisheye/omni_radtan.py`（投影/反投影）+ `tools/fisheye/test_omni_radtan.py`
- 参考 `docs/COORDINATE_SYSTEMS.md`（边读边对照公式）

### Step 7　SMPL-X geometry
- `tools/generate/load_real_smplx_motion.py`（poses[165] → canonical 参数）
- `tools/generate/generate_real_motion_geometry.py`（smplx 前向 → vertices/faces/densekp_3d）
- `tools/generate/generate_p0_toy_motion.py`（程序化最小例）

### Step 8　Blender renderer
- `tools/render/render_p0_blender.py::main`（mesh 写入、conv 矩阵、灯光/材质分支、EXR passes、plane_test）
- `tools/render/finalize_p0_outputs.py`（EXR→NPY、depth 语义、visibility 0-3、容差）
- `tools/render/eye_background_render.py`（眼睛/阴影库）
- 参考 `docs/P0_COORDINATE_CONVENTION.md`、`docs/P0_DEPTH_VISIBILITY.md`

### Step 9　Perspective → ERP → Omni-Radtan
- `tools/fisheye/run_fisheye_camera_p0.py::pose / read_exr / warp_case`（ERP→fisheye warp 与注记）
- `tools/fisheye/render_equirectangular_source.py` → `render_equirectangular_scene_source.py`（ERP 源渲染、Blender↔CV 朝向转换）
- `tools/fisheye/run_fisheye_scene_p0.py` + `tools/scene/scene_loader.py`（3D 场景接入点）
- 扫展：`run_fisheye_seam_sweep.py`、`run_human_scale_sweep.py`

### 补充（按需）
- 相机压力/截断/近距：`solve_camera_stress_placement.py` → `run_camera_stress.py` → `finalize_camera_stress.py`
- 场景资产制作：`tools/scene/build_*.py`、`prepare_hidden_alley.py`
- 运动审计（骑行准备）：`audit_cycling_motion.py` 等 G 类脚本
- 已知问题清单：`docs/POTENTIAL_ISSUES.md`

## 10. 遗留建议（未实施，供决策）

1. 从上游 CameraHMR 取回 `core/losses`（或明确本仓库训练入口不可用的事实并记录）；
2. 未来的 synthetic 训练集生成器沿用 `generate_bedlam_densekp.py` 的 schema 与校验风格，勿复用 'closeup'/'portrait' 命名；
3. 多人阶段需重设计 mask/instance 通道（POTENTIAL_ISSUES #12）；
4. 若要启用 flip 增强，必须先定义并测试 138 点镜像置换（POTENTIAL_ISSUES #4）。

---

# 第二轮：代码目录整理（2026-09-19，Phase A/B/C）

> 基线 commit：`cfdd3e9`（"骑行树林场景"，已包含第一轮文档与本轮前的场景资产）。
> 本次为 code organization refactor：移动/重命名文件 + 修复 import 与 subprocess 路径，**零算法改动**。

## A1. 整理前结构概括

`tools/` 平铺 47 个脚本 + 2 个包（fisheye/、scene/）。训练代码 `densekp/` 与 CameraHMR 边界不变。

## A2. 整理后目录树

```text
tools/
├── generate/     (4)  SMPL-X motion → 几何制品
│     generate_p0_toy_motion.py, load_real_smplx_motion.py,
│     generate_real_motion_geometry.py, prepare_p0_dressed_geometry.py
├── render/       (9)  Perspective 域：渲染、后处理、外观包装、相机压力
│     render_p0_blender.py, eye_background_render.py, run_eye_background_fix.py,
│     run_temporal_case.py, finalize_temporal_case.py, finalize_p0_outputs.py,
│     solve_camera_stress_placement.py, run_camera_stress.py, finalize_camera_stress.py
├── fisheye/      (8)  相机模型 + ERP 源渲染 + fisheye pipeline + 单测
│     omni_radtan.py, test_omni_radtan.py,
│     render_equirectangular_source.py, render_equirectangular_scene_source.py,
│     run_fisheye_camera_p0.py, run_fisheye_scene_p0.py,
│     run_fisheye_seam_sweep.py, run_human_scale_sweep.py
├── annotation/   (5)  DenseKP GT / 标注
│     build_bedlam_tiny_frame_manifest.py, generate_bedlam_densekp.py,
│     inspect_annotation.py, inspect_surface_operator.py, visualize_annotation.py
├── scene/        (9)  场景 build / validate / inspect（位置不变）
├── validate/     (6)  validate_bedlam_densekp_tiny.py, validate_p0_textured_regression.py,
│     p0_pyrender_crosscheck.py, blender_headless_smoke_test.py,
│     pyrender_headless_smoke_test.py, smplx_densekp_render_smoke_test.py
├── visualize/    (5)  make_appearance_comparison.py, make_environment_comparison.py,
│     make_eye_fix_comparison.py, make_skin_texture_review.py, review_eye_background_fix.py
└── experiments/  (12) 外观/眼睛/纹理一次性审计、运动数据审计、验证性生成
      audit_body_uv.py, audit_clothing_uv.py, audit_skin_texture.py,
      audit_eye_geometry.py, audit_eye_realism.py, prepare_procedural_eyes.py,
      audit_cycling_motion.py, audit_cyclist_bvh.py, find_cyclist_clip.py,
      search_babel_cycling.py, review_cyclist_audit.py,
      generate_cyclic_validation_motion.py
```

`tools/` 根目录不再有任何平铺脚本。Phase D（`core/` 包）未执行：`densekp/` 是 Original CameraHMR 边界，不动。

## A3. 移动的文件（git mv，全部保留历史）

- generate/ 4 个、annotation/ 5 个、render/ 9 个、fisheye/ +4（驱动脚本并入既有包）、
  validate/ 6 个、visualize/ 5 个、experiments/ 12 个。共 39 个文件移动，0 个删除。

## A4. 重命名

无重命名（只有目录移动）。

## A5. 修改的 import / 路径

| 文件 | 修改 |
|---|---|
| `tools/fisheye/run_fisheye_camera_p0.py` | `sys.path` parent→parents[1]；`root` parents[1]→[2] |
| `tools/fisheye/run_fisheye_scene_p0.py` | 同上；`from run_fisheye_camera_p0 import` → `from fisheye.run_fisheye_camera_p0 import` |
| `tools/fisheye/run_fisheye_seam_sweep.py`、`run_human_scale_sweep.py` | 同上 |
| `tools/render/run_eye_background_fix.py`、`run_temporal_case.py` | subprocess 路径 `tools/render_p0_blender.py`→`tools/render/render_p0_blender.py`；root parents[1]→[2] |
| `tools/render/run_camera_stress.py` | subprocess `tools/run_eye_background_fix.py`→`tools/render/run_eye_background_fix.py`；root→[2] |
| `tools/render/{finalize_camera_stress,solve_camera_stress_placement}.py` | ROOT parents[1]→[2] |
| `tools/annotation/{generate_bedlam_densekp,visualize_annotation}.py` | ROOT→[2]；生成器 README 自引路径 |
| `tools/validate/validate_bedlam_densekp_tiny.py`、`tools/visualize/{make_skin_texture_review,review_eye_background_fix}.py`、`tools/experiments/{audit_cycling_motion,audit_eye_realism,audit_skin_texture,review_cyclist_audit,search_babel_cycling,prepare_procedural_eyes}.py` | ROOT parents[1]→[2] |
| `scripts/audit_workspace.sh` | `tools/inspect_surface_operator.py`→`tools/annotation/inspect_surface_operator.py` |
| `README.md` | 快速检查命令 + 目录概览 |
| `tools/scene/scene_loader.py`、`configs/fisheye_scene_p0.json` | 报错文案中的 entry 路径 |
| `tools/annotation/generate_bedlam_densekp.py`、`tools/validate/blender_headless_smoke_test.py` | docstring 自引路径 |
| docs/* 与 REFACTOR_REPORT.md | 全部文件路径引用（数学内容未动） |

**未修改**：所有数学函数、投影、crop、visibility、loss、Dataset、网络；`tools/render/render_p0_blender.py` 与 `tools/render/eye_background_render.py` 的兄弟 import（同目录移动，无需改）；`tools/fisheye/omni_radtan.py`、`test_omni_radtan.py`、`tools/scene/*` 零改动；`densekp/`、`configs/hydra/` 零改动。

## A6. 故意不移动的文件

| 文件 | 原因 |
|---|---|
| `densekp/**`、`configs/hydra/**` | Original CameraHMR 边界（用户指令 §4） |
| `tools/fisheye/test_omni_radtan.py` | 被 `run_fisheye_scene_p0.py` 以 `python -m unittest tools.fisheye.test_omni_radtan` 子进程引用；名字像 test 但不是废弃脚本 |
| `tools/scene/**`、`tools/fisheye/omni_radtan.py`、两个 ERP 渲染脚本 | 已在合理位置；包内相对引用（`from scene import ...`、`from fisheye.omni_radtan import ...`）依赖现有层级 |
| `assets/scenes/photoreal/cycling_forest/` 等 | 场景资产已在 build/README 同位组织（README + .blend + validate 脚本成组）；.blend 巨大，移动无收益 |
| `third_party/**` | 各自带 .git 的上游镜像 |
| 旧 phase 历史审计文档中的命令行 | 属当时记录，不重写历史（现行入口以 CURRENT_CODE_MAP §3 为准） |

## A7. git diff --stat

见最终输出（由 `git diff --stat HEAD` 给出；全部为 rename + 上述小修改，无算法文件内容变更）。

## A8. 验证结果

1. `bash scripts/audit_workspace.sh`：全仓语法 PASS + 表面算子 PASS（每阶段后均执行）。
2. `python -m unittest tools.fisheye.test_omni_radtan`：3 tests OK。
3. Phase A/B/C 移动脚本 `--help` 冒烟：28/30 OK；2 个 FAIL 见 A9。
4. **数值回归（与整理前逐位一致）**：
   - Fisheye：`densekp_fisheye_2d.npy` 复算 max diff = 6.09e-5 / 6.10e-5 / 1.22e-4 px（center/cam_left/cam_right）——与整理前完全相同；
   - Perspective：`p0_dressed_eye_background_fixed` frames 0/8/15 重投影 max diff = 3.42e-5 / 3.85e-5 / 3.08e-5 px——与整理前完全相同。
5. Phase B：`blender -b assets/scenes/photoreal/cycling_forest/cycling_forest.blend --python tools/scene/validate_cycling_forest.py` → `passed: true`（455 ray samples，无 intrusions）。
6. 无旧路径残留：全仓（除 docs 历史文档与 outputs 历史记录）grep 旧路径为空。

## A9. NOT TESTED / 预期失败项

| 项 | 状态 |
|---|---|
| `tools/fisheye/run_fisheye_camera_p0.py --help`、`tools/render/finalize_p0_outputs.py --help` | NOT TESTED（当前 env 无 OpenEXR，整理前同样失败；渲染后处理需独立 Python env） |
| `tools/experiments/audit_cycling_motion.py` | NOT TESTED——无 argparse 的模块级一次性审计；依赖现位于 `outputs/audits/cycling_motion_audit/local_inventory.txt` 的陈旧绝对路径（含已删除的 /mnt/hdd 与 .vscode-server 路径）。与移动无关 |
| `tools/experiments/audit_eye_realism.py` | NOT TESTED（pre-existing）：模块级执行，SMPL-X replay 断言失败；已用 HEAD 原版对照确认失败与路径无关 |
| Blender 渲染全链（RGB/Depth/Mask 逐像素回归） | NOT RUN（渲染代码零改动，无需重跑 16 帧） |
| 训练入口 | 维持 NOT TESTED（`densekp/core/losses` 缺失，见 POTENTIAL_ISSUES #1，本次按指令不处理） |

## A10. 推荐代码阅读路线（更新）

按第一轮的 Step 1–9 顺序，路径对应关系：

| Step | 内容 | 新路径 |
|---|---|---|
| 1 训练入口 | 不变 | `densekp/train_keypoints.py` |
| 2 Dataset | 不变 | `densekp/core/datasets/` |
| 3 一个 sample 的 annotation | `tools/annotation/inspect_annotation.py`、`tools/annotation/visualize_annotation.py` | |
| 4 annotation generator | `tools/annotation/generate_bedlam_densekp.py`、`tools/validate/validate_bedlam_densekp_tiny.py` | |
| 5 DenseKP 138 correspondence | `tools/annotation/inspect_surface_operator.py` + `tools/generate/generate_real_motion_geometry.py` | |
| 6 camera projection | `tools/render/render_p0_blender.py`、`tools/fisheye/omni_radtan.py` + `test_omni_radtan.py` | |
| 7 SMPL-X geometry | `tools/generate/load_real_smplx_motion.py` → `tools/generate/generate_real_motion_geometry.py` | |
| 8 Blender renderer | `tools/render/render_p0_blender.py` + `tools/render/finalize_p0_outputs.py` + `tools/render/eye_background_render.py` | |
| 9 Perspective → ERP → Omni-Radtan | `tools/fisheye/run_fisheye_camera_p0.py`（warp_case/pose）→ `tools/fisheye/render_equirectangular_*.py` → `tools/fisheye/run_fisheye_scene_p0.py` + `tools/scene/` | |
| 补充：相机压力 | `tools/render/solve_camera_stress_placement.py` → `run_camera_stress.py` → `finalize_camera_stress.py` | |
| 补充：场景 | `tools/scene/build_cycling_forest.py` / `validate_cycling_forest.py` + `assets/scenes/photoreal/cycling_forest/` | |

常见问题速查：
- 生成 perspective 数据 → `tools/generate/`（几何）→ `tools/render/`（RGB/Depth/Mask + 注记）
- 生成 fisheye 数据 → `tools/fisheye/run_fisheye_camera_p0.py`（+ `run_fisheye_scene_p0.py` 场景版）
- 构建场景 → `tools/scene/build_*.py`；校验 → `tools/scene/validate_*.py`；资产 → `assets/scenes/`
- 验证 Omni-Radtan → `python -m unittest tools.fisheye.test_omni_radtan`
- DenseKP GT 计算 → `tools/generate/*`（R@V）+ `tools/annotation/`（训练格式）
- 训练入口 → `densekp/train_keypoints.py`（当前不可运行，见 POTENTIAL_ISSUES #1）
