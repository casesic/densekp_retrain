# MAMMA 生成主流程结构整理 A3（2026-10-22）

本批交付代码、CPU 准备计划和现有标签的只读导出；未运行 Blender、GPU、模型、分割器或 RGB 渲染。旧小试、上游 DenseKP/MAMMA 源码和历史产物没有改写。正常生产的 31 帧路径仍待实际渲染验收。

## 主入口与合同

- 编辑器冻结由 `tools/scene_editor/cli.py freeze` 生成 `scene_episode_rig_list_v1`。场景编辑器与旧 rig 准备脚本共用 `data/scene_geometry.py` 的人物 body/cloth/groom 加载和变换；`runtime/multirig_prepare.py` 是多 rig 文件/配置准备，原 `run_multirig_roi_a1.py:prepare` 是兼容包装。服务的常驻预览从公共运行工具取文件辅助函数，不再为这些准备逻辑导入实验运行器。旧 CLI 参数、一次场景装配、逐帧全体人物状态更新、多相机消费和临时缓冲清理顺序均保留。
- 正常完整 episode 显式使用 `--mamma-512 --qualification-registry <reviewed-A1-reader-config>`；`--prepare-only` 只做文件和 CPU 计划，不查 GPU、不启动 Blender。随后用 `--run-prepared` 消费已冻结配置。启动服务、编辑器普通预览或无该开关的旧生产入口不会自行计算 512 标签。旧 `--label-probe` 仍限单 rig、最多三帧，且保留七框诊断。
- 新 policy `mask_native_v2` 在每相机写有序 `mamma_512.npz` 和 `label_policy.json`，不要求 `mamma_box_support.json`。消费者对每个有 mask 的人物用存储二值 mask 的 `cv2.boundingRect` 求包含右下末像素的完整 ERP 框，按 A1 的 1.2 扩张、384:512 宽高比、Gaussian halo 与插值边距核对 ROI 支持；超出直接拒绝。旧标签无 `sidecar_policy` 时仍按 `seven_box_probe_v1` 校验七框文件。序列打包对新 policy 校验真实标签、人物顺序、相机/帧、rig/episode、整数时间及来源帧；旧缺件仍拒绝。
- `export_mamma_observations_a3.py` 是只读相机产物的通用入口。它按冻结 run 中显式帧、rig 内相机和 recipe 中**全部人物**导出 `mamma_observation_v1`；无 mask/框人物保留 absent。`frame_scope` 明确区分完整 31 帧与选定稀疏帧。观察 ID 由 episode、rig、帧、相机、人物编号组成，重复或缺失引用拒绝。未知 body 缓存不按 person_id、性别或顶点数放行，必须在经过审核的 `qualified_assets` 注册表中。输出的 `files` 指向原 RGB、目标人物 mask、physical mask、annotations 与 512 NPZ；`mamma_512` 指向有序 XY、世界点和域/coverage 审计字段；`point_labels.visibility` 给 A1 三态值及 known。原框仅作来源对照，`mask_derived_crop_support` 才是新输入支持结果。接触任务无新 GT。保守 visibility 算法和数值均未改。

## 最短使用方式

下面的准备示例已在本批以 `PLAN_ONLY` 标识执行，仅写独立计划目录；其中 e03 配方是现有两 rig 参考，不代表整段 MAMMA RGB 已生成。将来实际渲染须由协调方另行安排 GPU、预算和新目录，不能直接把此计划当已完成数据。

```bash
python tools/scene_editor/cli.py freeze --recipe <frozen-editor-recipe> --output <new-prepared-dir>
python tools/experiments/run_multirig_roi_a1.py --output <new-run> --gpu-uuid <actual-gpu-uuid> --budget-seconds 1800 --rig-list-manifest <prepared-rig-list.json> --mamma-512 --qualification-registry <reviewed-reader-config.json> --prepare-only
# 将来获准渲染时，对该未启动计划运行：
python tools/experiments/run_multirig_roi_a1.py --output <new-run> --gpu-uuid <actual-gpu-uuid> --budget-seconds 1800 --run-prepared
python tools/experiments/export_mamma_observations_a3.py --run <new-run> --output <new-manifest.json> --qualification-registry <reviewed-reader-config.json>
```

A1 读取时复制其 `mamma_a3_mask_reader_20261022.json` 到自己的新配置，将 `manifest` 指向导出的 JSON，沿用 `MammaMaskDerivedDataset`、`mask_bbox_inclusive_v1` 和已审核资产注册。旧 `probe_manifest.json`、七框诊断与旧 native 框基线保持原含义；旧框右下端点比 mask 派生框多 1 像素，本批未追改。

本批现有产物只读导出：

```bash
python tools/experiments/export_mamma_observations_a3.py --run outputs/experiments/mamma_512_label_pilot_a3_20261015/run --output outputs/experiments/mamma_generation_refactor_a3_20261022/observations.json --qualification-registry outputs/experiments/mamma_512_label_pilot_a3_20261015/a1_reader_config.json
```

新清单 12 槽、10 observed、2 absent；每人均可为 target。导出没有复制 RGB/NPZ。A1 当前 v2 Dataset 实读其中 2 个 observed 槽，图像 3×512×384，支持均为 true；这只是契约回读，不是新训练资格验收。新清单和配置位于 `outputs/experiments/mamma_generation_refactor_a3_20261022/`。

## 验证和限制

小型合成元数据 fixture 验证两 rig、两相机、人物 7/9、未知资产拒绝、稀疏 scope、新无七框与旧带七框读取、缺标签、错人物顺序、错相机/时间和支持区域拒绝；fixture 资产仅用于测试，不能证明真实资产资格。e03 新旧 manifest 的身份时间、XY/visibility 引用与图像/mask 引用逐槽一致。共享准备逻辑用 e03 第 15 帧两个人的世界 512 点与已存 NPZ 对齐，R138 两项最小回归通过；`--prepare-only` 对现有 e03 两 rig 给出 124 次相机调用计划，实际调用 0。语法、help/import 路径通过；未发现启动副作用。定向 CPU 验证累计低于 2 分钟，包含一次缺 `PYTHONPATH` 的导入失败及一次 fixture 拒绝算例修正，低于 20 分钟上限。

此批没有验证正常整段 MAMMA 渲染、消费者完整收尾或新 policy 下的真实相机文件；计划目录没有 RGB。现有 512 小试仍只有 1137 known positive、0 个可信 negative、5007 unknown，代表性遮挡负例和官方 visibility 定义待另批核证。未知资产必须先另行审查登记。编辑器若已由用户运行，后端 Python 改动需用户稍后重启服务才会加载；本批没有操作服务进程。
