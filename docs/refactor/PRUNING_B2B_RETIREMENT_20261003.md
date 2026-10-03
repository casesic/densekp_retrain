# B2b 第二批：旧裁剪与评分接口退役

日期：2026-10-03。状态：**实现及有界 CPU 验收完成，未提交，停在本批验收点。**

[执行清单及完整测试 ID](../../outputs/acceptance/pruning_b2b_retirement_20261003/execution_manifest.json) · [精确对照](../../outputs/acceptance/pruning_b2b_retirement_20261003/comparison.json) · [依赖核查](../../outputs/acceptance/pruning_b2b_retirement_20261003/dependency_check.json) · [无资产源码交付](../../outputs/acceptance/pruning_b2b_retirement_20261003/delivery.json)

## 实际基线与修改集合

实际起始分支 master，HEAD `8db7ca5ec7583d1749db2e5f33e96abf3eff5a1e`。B2a 两批及 B2b 首批已有未提交成果，另有用户 `docs/TODO.md` 修改；起始完整 status 保存在 starting_state.json。本批没有重置、暂存或提交，也没有重复实施此前迁移。

本批新增删除精确为七项，与 B2a 已有九项删除分别记录：

| 删除文件 | 前置依赖处理与停止支持内容 |
| --- | --- |
| densekp/core/datasets/utils.py | 已迁消费者直引定义模块；旧 get_example/get_example_projverts 及全部旧 re-export 导入、tuple 返回停止支持 |
| densekp/legacy/crop.py | 停止旧 SMPL/HMR、skimage/PARE 裁剪、SMPL 翻转和 3D 关节处理接口；不伪装为当前算法的别名 |
| densekp/legacy/__init__.py | 最后实现已退役，包发现不再列 densekp.legacy；未清理缓存 |
| tools/validate/evaluate_omni_densekp.py | 唯一保留动态消费者 erp_points 改公共路由；停止旧自动造框、ERP 居中和三相机评分 CLI |
| tools/validate/evaluate_densekp_npz.py | 停止旧注册 ID 评分 CLI；不删除 registry key 或旧 NPZ 读取能力 |
| tests/test_legacy_cleanup.py | 仅服务已退役旧裁剪的断言删除；当前归一化独立测试继续消费原 fixture |
| tests/legacy_cleanup_fixture.py | 最后旧裁剪专用消费者退役；没有删除任何原 fixture |

修改九个现存文件：

- `densekp/core/datasets/crop_pipeline.py`：仅删除 CropResult.legacy_tuple；移除此方法后全文件 AST 与 before 相同。
- `tests/geometry_camera_fixture.py`：TARGETS 改公共模块→有序符号表，api(original=True) 第一分支拒绝；synthetic 与 Capture 编码保持。
- `tests/test_geometry_camera_contract.py`：只删除旧导出身份测试及不用的 TARGETS 导入；四项有效方法 AST 完全相同。
- `tools/validate/verify_geometry_camera.py`：capture 提前拒绝，计算依赖延迟到 verify；保留比较、真实回放计算及报告逻辑。
- `tools/validate/check_workspace.py`：去掉旧裁剪测试，加入当前 geometry 契约和 geometry/Dataset capture 拒绝检查。扫描边界不变。
- `docs/READING_GUIDE.md`、`docs/COMMANDS.md`、`docs/environment/INSTALLATION.md`、`tests/fixtures/README.md`：更新受影响的接口、依赖和验收边界。

新增 `tests/test_geometry_capture_retirement.py` 三项长期检查及本报告。一次性驱动、源码副本和所有运行产物仅写新验收目录，没有增加公共测试框架。

## 保留能力及依赖闭包

Dataset、具名 crop_example/CropBox/CropResult、可视化和 BEDLAM validator 保持上一批实现；完整裁剪坐标、内部 192 列、138 点、标签第三值、旧 NPZ 权重、遮挡监督和 mask/portrait/closeup 行为不变。Omni 生成、ERP/相机公共计算，以及当前 candidate、严重遮挡、框偏移和历史固定样例评价路径保留。

旧 Omni 评分用 visible & in_crop & finite(error)，旧 NPZ CLI 有二值权重与非有限值拒绝策略；**当前 candidate evaluator 不被宣称为这两种分母、过滤或报告的等价替代。** 权重和模型代码不改，未加载真实权重，不扩大为模型验收。

对 314 个 Python 源文件的 2210 处 import 作 AST 核查（包括相对 import），另记录 40 处动态导入/子进程调用并核对可疑目标、配置与脚本命令。重点确认 geometry TARGETS、日常测试列表、固定验证/训练审计工具的动态模块选择、安装检查模块列表和活动评价命令。剩余字符串不以同名普通字段作为依赖；原 fixture 的历史键及历史 Markdown 是记录，不是导入。未发现范围外真实消费者，不存在需暂停的删除项。

## 独立 before 与执行保护

新目录 `outputs/acceptance/pruning_b2b_retirement_20261003/` 在代码编辑前创建。保存 487 份源码、配置、说明和原 fixture 字节快照，另保存 110 份具名历史输入/基线及上一批证据快照；不扫描全部 outputs。原 geometry/dataset/legacy/训练审计 fixture 均保持原字节，不重新录期望。

采集本轮 before 使用当时公共路由及现行 Dataset/低层调用；没有调用任何失效 capture。Dataset verify 同时读取 P2 独立 before，核实七样本、规格及引用图片存在。before_gate 通过后才迁移和删除。

使用已有 `/home/yy/conda_envs/camerahmr/bin/python`，禁用 bytecode 和 CUDA 可见设备。驱动阻断 torch.nn.Module 构造、Trainer 构造/fit、Optimizer 构造/step、torch.load、CUDA 初始化、SMPL 系列构造、smplx.create 和 load_smplx_model；写出限制在新验收目录。

首次驱动误将衣物 binding 导入 smplx.lbs 的纯公式视为禁止计算，导致 capture 与一项前测被保护钩子拒绝。失败制品保留在 before_guard_rejection/，原驱动与原因保留在 acceptance_initial_guard.py、guard_adjustment.json。审查后允许公式导入、继续阻断所有模型构造和加载，再完成 before。没有为通过验收修改算法、期望或容差，没有参数更新。

## 分项验收及限制

| 路径 | 本批结果 |
| --- | --- |
| 当前几何路由 | 同一小型合成输入前后 79 数组、结果编码与异常精确一致；符号次序不变。原 geometry_camera_v1 冻结期望、共同变换、轴向、距离、ERP 周期四项继续通过 |
| 当前裁剪 | 31 合成 Dataset 用例 + 七真实样本各关闭/启用增强；旧 ID/显式路径共90路线、2814数组；11低层用例156数组。两次均通过原 P2 before verify，本轮独立 before/after 精确相同 |
| 数值比较规则 | 合计3049数组，比较键序、dtype、shape、精确值和数组原字节；103份组内 JSON 逐字节相同，符号表另比。输入变异、增强参数、异常及 Python/NumPy/Torch CPU RNG 包含在原裁剪编码中；无忽略字段或新增容差 |
| Omni 历史数值 | 原三相机输入存在，FrozenNumericRegressionTest.test_fisheye_densekp 前后通过、无 skip；仍用原 1.3e-4 容差。仅公式对保存标签检查，无渲染/推理 |
| capture 拒绝 | geometry CLI 无 --before 时 argparse 退出2，在计算导入、资产读、mkdir/write 前拒绝；api/run original=True 在第一分支 RuntimeError。Dataset 和 B1 metrics 原拒绝检查仍通过 |
| geometry verify | 用显式中断 sentinel 证明 --before 选择 original=False 的 verify 分支；比较体、线程/种子设置至真实回放结尾、报告块做 before/after AST 相等检查。**未执行真实资产回放，分支检查不是完整回放通过** |
| CPU 方法白名单 | 前测21、后测24，均成功且无 skip；没有 unittest discover、Objective/Lifecycle 更新测试或其它模型模块批量测试。全部逐方法 ID 在清单中 |
| 无资产源码交付 | 新副本无 assets/outputs/权重，双包均从副本导入；同一明确白名单排除需要保存输出的 Omni 项，23项通过，无 skip。自动包发现无 densekp.legacy，已删模块在副本不可解析 |

无资产交付是在**现有依赖环境**运行源码副本，不是空环境依赖安装；本批未重新执行 editable 安装。日常测试不依赖本机验收目录；新增几何检查需要已有 geometry 依赖，但不读取人物资产。真实 SMPL-X、Blender、模型、训练生命周期和实际设备路径均未运行。上一批受控 BEDLAM 的限制仍在，本批没有追加真实 BEDLAM 验收。

## 保全与停止点

471 个未授权修改的保护文件逐字节相同，含 TODO、全部原 fixture、B2a 配置和上一批运行代码；110 份具名输入/证据字节相同。获准更新的阅读/fixture 说明保留原正文信息并更新现行状态。七项新增删除准确匹配授权；构建定义、constants、core/configs、注册表、misc、模型及公共几何源码未改。新测试和报告不被忽略。

没有未解决依赖或超出预期的业务数值差异。未覆盖范围如上明确保留。constants、注册表、misc 以及 B3–B5 留待另行评审；本批不继续清理。HEAD 不变，所有本批成果尚未提交。
