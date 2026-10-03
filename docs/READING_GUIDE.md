# 代码阅读指南

从 [样本旅程](SAMPLE_WALKTHROUGH.md) 进入；本页指向当前实现，不要求先读历史报告。业务和数值约定在 [CONTRACTS](CONTRACTS.md)，具体执行看 [COMMANDS](COMMANDS.md)。

## 两个包与配方

| 责任 | 当前定义与建议阅读顺序 |
| --- | --- |
| 输入与标签 | [keypoint_contract.py](../densekp/core/keypoint_contract.py) 的validate_annotations → [loading.py](../densekp/core/datasets/loading.py) 的路径解析 |
| 裁剪主线 | [DatasetKeypoints.__getitem__](../densekp/core/datasets/dataset_keypoints.py) → [crop_pipeline.py](../densekp/core/datasets/crop_pipeline.py) 的CropBox/CropPlan/CropResult、determine_crop、transform_crop |
| 图像与监督 | [image_processing.py](../densekp/core/datasets/image_processing.py) 的read_image/normalize_image → [keypoint_supervision.py](../densekp/core/keypoint_supervision.py) 的supervision_for_crop |
| 网络与目标 | [DenseKP.forward_step](../densekp/core/densekp_trainer.py) → [KeypointsHead.forward](../densekp/core/heads/smpl_head_keypoints.py) → [keypoint_loss](../densekp/core/keypoint_objective.py) |
| 训练组织 | [train_keypoints.train](../densekp/train_keypoints.py)、[DataModule](../densekp/core/datasets/datamodule_keypoints.py)、[ValidationCheckpoint](../densekp/core/checkpointing.py)；有界实验配方在tools/experiments/train_source_isolated_trial.py |
| 加载与单批预测 | [inference.py](../densekp/core/inference.py)：load_checkpoint_state_dict / load_lightning_checkpoint、predict_batch / predict_item；两加载/模式路线不强行合并 |
| 基础指标 | [evaluation.py](../densekp/core/evaluation.py)：evaluation_masks、point_errors、summarize；[evaluation/candidates.py](../src/densekp_retrain/evaluation/candidates.py) 保留另一套汇总失败/分母规则 |
| 人体与衣物 | [body/parameters.py](../src/densekp_retrain/body/parameters.py) 的evaluate/parameters_from_pose → [body/geometry.py](../src/densekp_retrain/body/geometry.py) 的compute_densekp_3d；[clothing/preparation.py](../src/densekp_retrain/clothing/preparation.py)、[groom_transfer.py](../src/densekp_retrain/clothing/groom_transfer.py) |
| 世界与相机 | [body/transforms.py](../src/densekp_retrain/body/transforms.py) 的world_points → [camera/rig.py](../src/densekp_retrain/camera/rig.py) 的rig_poses → [erp.py](../src/densekp_retrain/camera/erp.py)、[domain.py](../src/densekp_retrain/camera/domain.py)、[visibility.py](../src/densekp_retrain/camera/visibility.py) |
| Blender与成像 | [assembly.py](../src/densekp_retrain/rendering/blender/assembly.py)、materials/groom等属Blender边界；[imaging/ds.py](../src/densekp_retrain/imaging/ds.py) 的process_scene、[omni.py](../src/densekp_retrain/imaging/omni.py) 的warp_case：读取→重映射→有效域→标签/框→保存 |
| 来源→样本→衍生框 | [data/source_split.py](../src/densekp_retrain/data/source_split.py) 的validate_split → [packing.py](../src/densekp_retrain/data/packing.py) 的pack_index_rows → [boxes.py](../src/densekp_retrain/data/boxes.py) 的change_box |
| 应用评价与报告 | [candidate_runner.py](../src/densekp_retrain/evaluation/candidate_runner.py) 的输入、预测、配对、报告阶段；visualization与io各管图片/JSON，不经历史evaluator借公共功能 |
| 实验回调 | [trial_callbacks.FixedValidation](../src/densekp_retrain/training/trial_callbacks.py)、[training_audit.py](../densekp/core/training_audit.py) 的CheckUpdates/探针；保持模式、RNG、optimizer顺序与state_key |

`densekp`负责模型和直接输入/训练目标；`densekp_retrain`负责生成、来源、应用评价与实验组织。公共模块不反向import tools。不要预先将全部制作配方合成通用引擎。

## 哪些入口日常用，哪些按需用

- 日常：audit_workspace、inspect_annotation；配置打印。准备好明确输入后用candidate evaluator及DS固定框/遮挡入口。
- 实验组织：run_regression_suite的两个具名builder只组织任务，不统一评价公式；默认展示。source有paired汇总/compare，box compare保持独立。
- 按需资产：prepare_source_isolated_assets、prepare_adult_candidates，STYLES在adult_candidate_recipe；SOURCES/mesh_panel、固定场景布局仍是专用配方。完整生产与消费链见 [场景索引](rendering/SCENE_RECIPE_INDEX.md)。
- 按需诊断：姿态/衣物/groom验证、camera-stress、透视时序、BEDLAM适配仍保留，不把所有tools当公共API或日常必跑项。
- 按需真实检查：check_inference_runtime与check_fixed_validation_runtime先读build_plan，再读execute；默认无计算。回调复用源码tests中的run_case，不能从其存在推断独立wheel可运行。

现存专用关系包括candidate配方的layout复用、source准备的SOURCES/mesh_panel、DS box evaluator→DS occlusion evaluator。它们没有随精简迁移，不能把“公共模块无tools依赖”误读为脚本之间完全无调用。

旧loss/tuple裁剪/评分CLI、固定mixed制作、失败衣物/BVH支线和旧迁移驱动已明确退役；数据/保存预测消费保留。注册表key、to_lower及特定工具暂缓项见 [KNOWN_BEHAVIOR](KNOWN_BEHAVIOR.md)。验证证据与历史代码定位统一从 [VALIDATION](VALIDATION.md) 进入。
