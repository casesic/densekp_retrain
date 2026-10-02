# 兼容代码、专用配方与清理依据

2026-10-02。本表是当前归属索引，不要求读者先查历史重构报告。
“无活动调用”不等于“可以删除旧接口”；本批只删除语法上必不可达的实现，保留未能排除的外部调用/序列化用途。

## 核心与兼容代码

| 原位置/候选 | 现在的替代或保留位置 | 用途核查与决定 | 验证结果 |
| --- | --- | --- | --- |
| `core/datasets/utils.py`：resize_image、get_transform/transform、crop_img、generate_image_patch_skimage、fliplr_params/keypoints、keypoint_3d_processing、rot_aa、get_example | `densekp/legacy/crop.py`；旧 utils 显式重导出 | 运行源码/配置/脚本、文档符号与动态字符串核查没有发现当前 Dataset 调用；保留旧函数名、签名及独特插值、翻转、原地修改语义，不把 get_example 替换为 138 点流程 | 迁移函数 AST 精确（除下一行死分支）；34 项独立行为记录中覆盖裁剪、增强、RNG、失败分支 |
| generate_image_patch_skimage 中 `if False:` 的旧数组切片裁剪块 | 删除该不可达块；保留实际 crop_img 路径 | 条件为布尔字面量 False，无动态选择；没有运行输出或唯一复现用途，原代码在独立快照与 Git 历史 | 去除此块后 AST 与原函数一致；两种旋转及非方框拒绝行为精确 |
| utils.do_augmentation、get_example_projverts | 前者导出旧 tuple 适配；后者仍在 utils 薄适配到 crop_pipeline | Dataset 直接消费具名阶段，旧 API 只服务兼容和测试 | 原裁剪合约及固定旧 tuple 测试通过；未修正翻转/portrait/mask 既有规则 |
| core/utils/train_utils.denormalize_images | `core/image_normalization.py`；trainer 直接引用，旧模块重导出 | 唯一活动 train_utils 用途；不再让当前日志路径依赖历史 camera/checkpoint helpers | 原函数 AST、数值精确；trainer 仅换 import，未更新模型参数或运行训练 |
| train_utils 的 trans_points2d_parallel、strip_prefix_if_present、load_valid、perspective_projection、convert_to_full_img_cam | `densekp/legacy/train_utils.py`；旧模块重导出 | 无当前业务消费者；保留外部调用和旧相机复现用途。load_valid 的 skip_list 无效和多余 key 失败行为不修正 | 数值、前缀处理、微型权重加载/失败与 before 精确；没有用它替代当前严格 checkpoint loader |
| `core/losses.py` 六个旧 loss 类 | **原类路径原位保留**，模块说明明确为历史兼容；当前目标在 keypoint_objective.py | 当前 trainer 不用这些类；原 densekp.ckpt 的 pickle 元数据未引用它们，但无法排除外部完整对象 pickle，因此不迁类、不改 __module__ | 六类两 loss 形式共 11 个迁移前小型 pickle 恢复；值/梯度精确。只读取原权重 274260 字节 data.pkl，未读取 tensor storage |
| core/utils/misc 的 task_wrapper/extras/save_file/close_loggers/log_hyperparameters | 原位保留 | 有当前训练调用，尤其异常 finally 与 logger 收尾；不得按辅助函数外形删除 | 代码未变，调用证据记录；本批不执行训练生命周期 |
| misc.instantiate_callbacks/instantiate_loggers/get_metric_value | 原位保留，文档标为继承兼容 | 未找到活动调用，但含 Hydra `_target_` 动态构造且有旧训练工具接口用途；删除收益小，外部用途不足以排除 | 原函数未变；不声称验证所有 Hydra 插件 |
| constants 中非 DenseKP 常量 | 原位保留 | 继承的模型/数据路径字符串不是已验证资产，也不属于本批目标模型参数；不借清理启用或统一它们 | 原值未变 |
| 旧 Hydra 配置与 datasets.example.json | 原位保留；[配置索引](../../configs/README.md) 区分当前/历史/示例 | Hydra 组名可动态选择，示例有说明用途；未发现 example JSON 的当前加载器，不把它推广成新接口 | 内容/默认 data=unselected 不变；无资产配置解析检查通过 |

上表所有迁移的旧 import 路径继续存在。当前损失、裁剪、推理不能从 legacy 选“看起来相似”的函数替换。

## 导入与执行分离的 12 个专用脚本

下表脚本现在导入时只定义 main；直接执行才导入其可选依赖并按原顺序执行。
每个原始执行体（包括 import 顺序、随机调用、参数默认值）均与 main.body AST 精确一致。
纯 Python 隔离导入有 I/O/进程/目录扫描拦截，12 个均通过；不以此替代完整资产生成或 Blender 验收。

| 原位置（保留原路径） | 独有用途/输出 | 实际验证 |
| --- | --- | --- |
| tools/experiments/audit_skin_texture.py | 官方纹理、眼部 UV、body_uv 与审计 JSON；含历史 ZIP/摘要步骤 | main AST + 无副作用导入；未读纹理 ZIP 或执行摘要 |
| tools/experiments/prepare_procedural_eyes.py | procedural_eye_coordinates 的现有独有生成器 | main AST + 导入；未实例化 SMPL-X 或新增资产 |
| tools/experiments/review_cyclist_audit.py | BVH 映射 CSV、关节层级和运动学图 | main AST + 导入；保留 audit_cyclist_bvh 私有配方依赖 |
| tools/experiments/extract_commuter_union.py | 3 mm garment envelope 专用提取 | main AST + 导入 + 独立旧 parser help 对照 |
| tools/experiments/smooth_commuter_weights.py | 固定 24 步衣物图扩散 | main AST + 导入 + help 对照；未变随机/数值配方 |
| tools/experiments/trim_commuter_neck.py | 经肩高检查的历史领管裁切 | main AST + 导入 + help 对照；继续直引公共 clip_surface |
| tools/experiments/inspect_smplx_rigs.py | Blender 标准/locked-head rig 比较 | main AST + 导入；无 bpy 环境也可导入，未执行 rig 比较 |
| tools/validate/audit_commuter_component_contacts.py | 特定 collar/shirt 穿透检查 | main AST + 导入；直接 CLI 仍使用 `--`，执行需要 mathutils/BVH |
| tools/validate/audit_commuter_pose_sources.py | 特定源/retarget 姿态网格核查 | main AST + 导入；执行仍需要 Blender mathutils 与既有几何 |
| tools/validate/audit_commuter_repair_bodies.py | 修复产物的人体参数与 138 点证据 | main AST + 导入 + help 对照；未重算真实人体 |
| tools/validate/blender_headless_smoke_test.py | 独立渲染环境 smoke，具有写图和 renderer 调用 | main AST + 导入；本批不运行 smoke 或 Blender |
| tools/visualize/make_skin_texture_review.py | 旧肤色/衣物与眼部 ROI 对照图 | main AST + 导入；保留历史 before/after 目录用途 |

没有发现仓库调用者依赖上述脚本导入后产生的中间全局变量，也没有 globals/locals/eval/exec 动态查找；因此允许将其封装为 main。
完整工具 AST 扫描后剩余顶层动作主要是路径/常量、正则、绘图库后端设置与专用 EGL 环境选择；没有发现剩余顶层资产读取/生成/渲染调用。
普通 bpy 支持库本来要求 Blender 环境，这种依赖不等同于导入即装配或渲染，未改其执行语义。

## 证据与保留界限

本批原位置、行号、符号全文匹配、checkpoint 元数据、独立 before/after 和删除块位置保存在
[验收清单](../../outputs/acceptance/refactor_cleanup_20261002/execution_manifest.json)。原方案及指定历史报告的结论均作为线索重新核查，不把旧“无调用”直接当作当前删除依据。
核查范围为当前源码、配置、脚本、测试、文档、明确引用的原 checkpoint 和选定历史产物；没有递归扫描整个 outputs 或所有外部仓库。
未证明的外部调用与资产复现用途通过保留兼容路径处理，而不是推断不存在。
历史输出、失败版本、源场景和 TODO 均保留。P4 安装与本批修改尚未提交，不能将未跟踪文件写成已经进入 HEAD。
