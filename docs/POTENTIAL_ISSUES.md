# POTENTIAL ISSUES（当前状态与历史问题）

> 本文记录疑似问题及其处理状态；已解决的历史问题保留原因和验证限制。

## #1 已解决：训练代码缺少 `densekp/core/losses.py`

- 原因：暂存仓库曾遗漏上游 `tools/CameraHMR/core/losses.py`，而 `densekp/core/densekp_trainer.py` 在模块顶层导入 `Keypoint2DLoss`。
- 处理：已从上游补入 `densekp/core/losses.py`。`compute_loss` 仍使用原有内联损失实现，行为未改。
- 验证：使用 `/home/yy/conda_envs/camerahmr/bin/python` 成功导入 `Keypoint2DLoss` 和 `DenseKP`，并通过损失类数值检查、现有 7 个单元测试及训练入口的 Hydra 配置解析；尚未执行实际训练。历史审计记录保留在 `docs/refactor/REFACTOR_REPORT.md`。

## #2 `configs/datasets.example.json` 未被任何代码消费，且注册表缺 `bedlam-mini`

- 文件：`configs/datasets.example.json`；`densekp/core/configs/__init__.py`
- 函数：`DATASET_FOLDERS` / `DATASET_FILES`；`DatasetKeypoints.__init__`
- 位置：json 全文；configs/__init__.py:6-319；dataset_keypoints.py:33-34
- 怀疑原因：全仓库 grep `bedlam-mini` 只命中该 json；没有任何 Python 读 `datasets*.json`。数据集名实际硬编码在 `DATASET_FOLDERS`/`DATASET_FILES`，训练配置 `DATASETS_AND_RATIOS` 只能引用注册表中已有的名字。示例 json 给人"改 json 即可加数据集"的误导。
- 可能影响：新同学按示例 json 配置数据路径后，训练仍会 KeyError（'bedlam-mini' 不在注册表）或读错路径。建议后续要么实现 json 加载，要么在 README/文档中明确注册表是唯一入口（本次仅记录）。

## #3 数据集名包含 'closeup'/'portrait' 时图像被旋转而标注坐标不跟随

- 文件：`densekp/core/datasets/utils.py`
- 函数：`get_example_projverts`（同逻辑也在 `get_example`）
- 位置：utils.py:540-541
  `if dataset and ('closeup' in dataset or 'portrait' in dataset): cvimg = cv2.rotate(cvimg, cv2.ROTATE_90_CLOCKWISE)`
- 怀疑原因：旋转只作用于读入的 RGB；`center/scale/proj_verts` 原样用于旋转后图像。对 BEDLAM 官方 portrait 数据这是正确约定（官方标签按旋转后图存储，BEDLAM_TINY_FRAME_MAPPING.md），但对本项目自造数据集是隐患：只要数据集名含这两个子串，RGB 会被静默旋转 90°，与未旋转的 GT 错位（视觉上表现为整体 90° 偏移，训练损失仍然"有数值"）。
- 可能影响：生成 synthetic 训练集时若取名如 `synthetic-closeup-xxx` 将造成 GT 与图像系统性错位。生成新数据集时应避开这些子串或预先旋转图像（DATA_FORMAT/生成规范已在 TRAINING_DATA_FORMAT.md §5.4 提示）。

## #4 水平 flip 会破坏 proj_verts 的 138 点身份（上游已知，配置层面规避）

- 文件：`densekp/core/datasets/utils.py`
- 函数：`get_example_projverts` / `generate_image_patch_cv2`
- 位置：utils.py:266-268（flip 只翻图并镜像 c_x）；`flip_kp_permutation` 形参（utils.py:523）在函数体内从未使用（grep 证实只有签名处出现）
- 怀疑原因：flip 时图像水平翻转，`trans` 随翻转后的 c_x 构建，因此 xy 坐标数学上自洽；但 138 个点是否需要左右镜像置换（FLIP_KEYPOINT_PERMUTATION 已定义却未被应用）从未验证。训练 GT 若被翻转，第 0..137 行将对应错误的身体部位。
- 可能影响：yacs 默认 `_C.DATASETS.CONFIG.DO_FLIP=True, FLIP_AUG_RATE=0.5`（configs/__init__.py:368-369）。`configs/hydra` 训练链是否真正触发取决于最终 cfg；TRAINING_AUDIT.md 记录旧配置 `DO_FLIP=False, FLIP_AUG_RATE=0.0`。**任何重训练前必须确认运行时配置不启用 flip**，否则 138 点监督被静默镜像。

## #5 `DatasetKeypoints.__getitem__` 中 `bbox_size<1` 时进入 `breakpoint()`

- 文件：`densekp/core/datasets/dataset_keypoints.py`
- 位置：:60-61 `if bbox_size < 1: breakpoint()`
- 怀疑原因：DataLoader worker 中触发 `breakpoint()` 会挂起进程而非报错；同样模式还有 utils.py:54、297、338、559。
- 可能影响：脏样本（scale 过小/负值）会让训练卡死在 worker 而非报错。上游原样保留。

## #6 `validation_step` 的 GT 第三通道写法依赖通道顺序

- 文件：`densekp/core/densekp_trainer.py`
- 函数：`validation_step`
- 位置：:159-165 —— `gt_keypoints_2d_cropped[:,:,:2] = ...`（只变换 xy）随后 `proj_verts_loss` 用 `[:, :, :-1]`（前两通道）。
- 怀疑原因：`[:,:,:2]` 与 `[:,:,:-1]` 在 `[B,138,3]` 下等价（均前两通道），当前无错；但两处写法不一致，若未来数组变成 4 列（如 x,y,z,conf）两者将含义分裂。
- 可能影响：仅可读性风险，无当前行为差异。保持原样，只提示。

## #7 `finalize_p0_outputs.py` 将所有点的 `densekp_valid` 写为全 1

- 文件：`tools/render/finalize_p0_outputs.py`
- 位置：`np.save(out/'annotations/densekp_valid.npy', np.ones_like(vis, dtype=np.uint8))`
- 怀疑原因：`valid` 字段按 P0_DEPTH_VISIBILITY.md 语义应为"有效候选"，但这里无条件写 1（OUT_OF_FRAME/BEHIND_CAMERA 点也标 valid=1，只有 visible 状态区分）。camera-stress 分支（finalize_camera_stress.py:41-49）则按 z<0 与图内过滤重算了 valid/in_frame。两处语义不一致。
- 可能影响：下游若直接把 p0 渲染目录的 `densekp_valid.npy` 当"在画面内"使用会得到错误结论。当前训练不消费这些数组，暂无实际影响。

## #8 机器特定路径（runtime wrapper 已配置化，历史/数据路径仍保留）

- 当前状态：正式 render/fisheye wrapper 已统一按 `--blender-bin` → `BLENDER_BIN` → `PATH` 解析 Blender；frozen appearance manifest 可用 `--appearance-manifest` 覆盖；BEDLAM 服装纹理目录可用 CLI 或 `BEDLAM_CLOTHING_TEXTURE_DIR` 提供。
- 故意保留：`tools/experiments/` 的一次性审计数据路径、`configs/perspective_camera_stress.json` 和 `finalize_camera_stress.py` 对仓库内 frozen baseline 的数据引用，以及 `densekp/core/constants.py` 的项目相对权重路径。
- 下载脚本默认使用 `$HOME/datasets/BEDLAM`，并支持 `BEDLAM_ROOT`；历史实验仍需用户按本机数据布局调整，训练权重仍需按既有相对目录准备。

## #9 `densekp/core/configs/__init__.py` 中部分注册项图像根路径为相对路径

- 位置：configs/__init__.py:58-163（BEDLAM1 SMPLX 与 BEDLAM2 段 `data/training-images/...` 无 `os.path.join(base_dir, ...)` 前缀）
- 怀疑原因：同一 dict 里前者（:22-55）用绝对拼接、后者用裸相对路径；相对路径依赖进程 CWD 恰为仓库根（pyrootutils 会设置，风险被缓解）。混用易在其它入口（如 validate_bedlam_densekp_tiny 自建 cfg）出错。
- 可能影响：低；记录差异以便排查。

## #10 `training_step` 里未使用的变量与 `criterion_keypoints`

- 文件：`densekp/core/densekp_trainer.py`
- 位置：:38 `self.criterion_keypoints = nn.MSELoss(reduction='none')`（从未使用）；:137 `pred_smpl_params = output['pred_keypoints']`（未使用）
- 怀疑原因：上游遗留，无行为影响，仅噪声。
- 可能影响：无。保留原样。

## #11 `warp_case` 对 `densekp_2d.npy` 与 `densekp_fisheye_2d.npy` 双写

- 文件：`tools/fisheye/run_fisheye_camera_p0.py`（:64）；`tools/fisheye/run_fisheye_scene_p0.py`（:67 又把 fisheye_2d 复制为 densekp_2d.npy）
- 怀疑原因：同一数组两个文件名，历史兼容；读代码时容易以为两者语义不同。
- 可能影响：仅混淆；未修改。

## #12 Blender 5.1 中 `IndexOB` pass 的可用性分支

- 文件：`tools/render/render_p0_blender.py`（outputs() 用 Alpha 当 mask）与 `tools/fisheye/render_equirectangular_source.py`（使用 IndexOB，且 :39-40 有回退检测）
- 怀疑原因：P0_DEPTH_VISIBILITY.md 记录 Blender 5.1 Eevee 不再暴露 IndexOB，故 perspective 用 Alpha；fisheye 路径用 CYCLES + IndexOB。两域 mask 语义来源不同（Alpha vs object index）。若未来perspective 渲染加入场景物体，Alpha-mask 将无法区分人体与场景。
- 可能影响：多人/场景遮挡阶段的 mask 需要重新设计（P0_DEPTH_VISIBILITY.md §5 已预警）。当前单人体无影响。
