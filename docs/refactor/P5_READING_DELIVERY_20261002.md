# P5 最终阅读与交付核查

日期：2026-10-02。P1–P3 已提交于 `aaf08bd`，P4 安装、入口与历史清理已通过用户整体评审。
本轮完成 P5 文档与只读交付核查，**待评审决定整个重构能否收尾**；没有自动提交。
独立记录在 [execution_manifest.json](../../outputs/acceptance/refactor_reading_20261002/execution_manifest.json)。

## 从哪里开始读

[README](../../README.md) → [一个实际样本](../SAMPLE_WALKTHROUGH.md) → [代码阅读指南](../READING_GUIDE.md)。
执行时再看 [命令与计算边界](../COMMANDS.md)，配置选择看 [配置索引](../../configs/README.md)。
这条路线不要求先读重构历史。当前状态以 [任务清单](IMPLEMENTATION_TASKS_20261001.md) 为准；
各阶段报告的“未提交/待修复/待评审”保留其当时语境，不覆盖最新状态。

| 归属 | 当前职责 | 阅读入口 |
| --- | --- | --- |
| `densekp/core/datasets` | 路径解析、读图、增强、裁剪、归一化、标签与监督 | Dataset `__getitem__` → crop_pipeline → keypoint_supervision |
| `densekp/core` | 固定 138 点网络、训练目标、推理、训练审计与基础指标 | DenseKP、keypoint_objective、inference、evaluation |
| `src/densekp_retrain/body, clothing, camera` | 人体/衣物公共计算、世界和相机坐标、投影和有效域 | body/geometry、transforms、camera/rig；各历史算法明确保留 |
| `src/densekp_retrain/imaging, rendering` | DS/Omni 成像、标注输出、Blender 共用装配与任务命令 | process_scene / warp_case、rendering/blender、jobs |
| `src/densekp_retrain/data, evaluation, training, visualization` | 来源/打包/框、候选流程与报告、有界训练回调、展示 | source_split、packing、boxes、candidate_runner、trial_callbacks |
| `tools` / `configs` | 具体入口、场景与实验配方、配置组合 | 命令表及场景配方索引，不反向作为公共模块依赖 |
| `densekp/legacy` 与旧导出 | 旧裁剪/训练辅助；历史 loss 类路径另保留原位 | [保留依据](LEGACY_RETENTION_20261002.md)，不是日常主路径 |

## 实际样本的走读结论

主例固定为 heavy / cam2 / person 3。新版 amodal 标签沿用旧图和保存预测；
其 `imgname/center/scale/GT xy`、保存预测/affine/误差与旧版相同，监督策略明确区分。
P5 回读选定 NPZ 和 manifest、mmap 数组头与图片尺寸，没有创建 Dataset、加载模型或解码 EXR。
完整字段记录见 [sample_trace.json](../../outputs/acceptance/refactor_reading_20261002/sample_trace.json)，
当前符号及所在行见 [symbols.json](../../outputs/acceptance/refactor_reading_20261002/symbols.json)。

- 人物/姿态追到 BEDLAM 的 `rp_aaron_posed_009/1095`，男性，几何帧 16 对应动作切片内帧 80；衣物来自同人物同动作的既有模拟，其原时间轴帧为 181（含起始偏移 101）。
- 场景是 urban_street_assets，源 ERP 1536×768；person location/yaw、相机矩阵、源 EXR、normal ERP、标签和框均有现存产物。原始 EXR Depth 是源 ERP 射线距离，不替换成后处理深度。
- 固定框取 clear 条件，center=[1132,414]，scale≈1.068；heavy 图像与标签共用 affine。新版 123 点受监督，其中 101 个遮挡点；旧标签只监督 22 个可见点。
- 模型只读裁剪图像，不读人物 ID；看中间 192 列，但输出属于完整 256-square。主例不含训练 loss 制品；loss 链路通过当前函数和 P1 已评审证据解释，不伪造本例训练记录。
- 保存预测可追到原 checkpoint 的路径和 model_config；原始第三值不参与坐标还原或误差加权。报告明确是复用预测重新评分。

两个理解问题已有可直接核对的答案：同边长、无旋转时中心右移 Δx 使同一点归一化 x 减少 Δx/s，
图像也经相同 affine 改变，邻人上下文随之变化。新策略用 `in_view & inside_full_crop`，可靠遮挡 GT 继续参与，
域外或实际完整裁剪外不参与；第 4.1 节用既有 challenge_001 增强记录补足裁剪截断情况。

没有追溯完成的部分明确保留：

1. geometry manifest 指向的旧 `outputs/temporal_case_aaron_1095/geometry/canonical_motion.npz` 已不存在。
   baseline 目录有 canonical 副本且原动作文件存在；没有旧副本，不能证明它们曾字节一致。没有修改历史 manifest。
2. 主例生成早于真实镜头遮罩加入；当前默认后处理不能直接重跑以证明同一历史图像。其保存标签表达的是当时有效域。
3. 源 manifest 未冻结全部 appearance/HDRI/贴图版本；checkpoint 只有路径/历史配置等记录，不能仅凭文件现存证明自推理日起未变。
4. 本例不是来源隔离骑行小试，不能将另一条角色/姿态路线拼成它的来源；也不证明检测召回或真实场景效果。

这些是历史复现边界，不阻断当前保存样本阅读；没有发现需要扩大代码重构才能解决的阅读阻断。

## 本轮改动和核查

- 将样本导读主路径明确为新版 amodal 标签，补齐可执行的数据读取 imports、产物/符号索引和两个理解检查；旧标签仍作为历史对照。
- 新增命令表，明确 help、配置打印、数据检查、保存预测回放、真实推理、训练与渲染的输入和输出。特别注明去掉 `--cfg job`、省略 reuse、`--skip-render` 的实际效果。
- 同步 README、阅读指南、文档导航、配置索引、任务清单及规划页的当前进度；历史实施报告保持原样。
- 核查当前导航 Markdown 链接、关键 AST 符号、源码/配置/构建/测试 fixture 的忽略状态及未提交清单。只扫描选定源码/配置/文档和测试目录，不递归遍历 outputs 或资产。
- 复用已批准的 P1–P4 数值、生命周期、Blender 和无资产交付证据；本轮不重复运行 CPU 全套测试，不把文档核查说成新模型/渲染验收。

最终只读检查通过：所选导航 280 个链接（含锚点）、42 个关键函数/类、485 个交付检查文件的非忽略状态。
源相机标注 → 配对标注 → loader GT/mask → 保存预测报告的点数一致；TODO 与 HEAD 未改变。
`git diff --check` 排除用户原有 TODO 后通过。文件列表覆盖源码/配置/文档与测试必要文本 fixture；
可选历史 NPZ 不因此获得源码交付承诺，P4 的无资产 checkout 证据用于验证替代 fixture 路径。

## 最终交付清单

| 项目 | 当前交付状态 |
| --- | --- |
| P1–P3 公共计算与调用方 | 已提交 `aaf08bd`；两包/测试的当时 Python 文件已纳入，公共 data 与 docs/data 忽略问题已修复 |
| P4 安装与入口 | 未提交：pyproject.toml/setup.py、六份 Hydra data YAML、安装验证入口、依赖说明及安装记录 |
| P4 历史清理 | 未提交：legacy、image_normalization、兼容导出、12 脚本 main、日常检查、fixture/tests、清理/保留报告和 README 改动 |
| P5 | 未提交：本报告、命令表、样本导读及当前导航/任务状态更新；详见 delivery.json 中本轮文件集合 |
| 用户修改 | docs/TODO.md 保持本轮开始时原样，不能作为本轮交付内容混淆归属 |
| 大验收记录与历史输出 | outputs 仍被忽略；不假定 git 能交付 before/after、NPZ、图片、失败版本。需要迁机时另行保存，不改变忽略规则 |
| 必要源码/配置/fixture | 当前文件均检查可被 Git 交付；“非忽略”和“已经提交”分开记录；新增文件仍需未来显式纳入提交 |
| 权重和资产 | 不打包进源码：DenseKP checkpoint、ViTPose、SMPL-X、regressor、BEDLAM 动作/衣物/UV/贴图、场景/HDRI、真实标定与有效 mask；明细和缺省缺失见安装说明 |

文件级状态见 [delivery.json](../../outputs/acceptance/refactor_reading_20261002/delivery.json)；
P5 开始状态见 [starting_state.json](../../outputs/acceptance/refactor_reading_20261002/starting_state.json)。
关键结论保存在非忽略的本报告，outputs 大清单并非唯一交付说明。

## 已验证范围与仍在的限制

普通 Python：实际 Python 3.10.20 和既有依赖下的独立离线 editable、双包导入、两 cwd 配置/CLI、
小型 CPU Dataset 与无资产交付 checkout 已由 P4 验证。完整版本见
[依赖清单](../environment/DEPENDENCIES_20261002.json)。没有验证空环境安装全套依赖，也不承诺自带资产 wheel。
共享环境 OpenCV/NumPy 元数据冲突、rtmlib 缺 onnxruntime 保留记录，未升级或修复。

P1 真实固定预测、GPU 小网络回调与 CPU 生命周期证据继续适用各自路径；P3 Blender 装配/有限渲染
结论只覆盖 Blender 5.1.1、CPU 和已选代表场景，不扩展到 GPU 0、所有设备、版本或场景。
新机器须另备资产；默认 outdoor_courtyard.blend 缺失的历史路径不能作为现成可运行场景。

纯重构保持 138 点/regressor、人体/衣物/图像对应、DS/Omni 历史差异、ERP 周期、显式有效 mask、
完整 256-square、旧标签权重/新遮挡监督、第三输出语义、loss/优化器/RNG/模式切换、checkpoint 严格加载、
来源衍生关系、指标分母/空集/非有限策略。证据按模块分批保存；不作“全仓库每一分支均已实测”的承诺。

以下是独立挂起的行为事项，不能作为已修复交付：

| 事项 | 现状及处理边界 |
| --- | --- |
| 历史 evaluate_densekp_npz 的 visible 字段 | weights>0 不等于 amodal 可见点；需单独设计报告 schema 兼容修复 |
| portrait/closeup 图像旋转 | 保留名称分支；需核对其历史标签坐标后才能决定修正 |
| mask floor/rint | GT 有效域与预测诊断保留不同历史取整，未统一规则 |
| NaN 框拒绝记录写严格 JSON 失败 | P2 已证实并保留异常/部分输出；需独立修复 |
| LOSS_TYPE/head 等旧配置未生效 | 保留既有代码行为，未将配置声明变成新目标或模型结构 |
| 历史邻人元数据版本差异 | 活动修订有记录；保留旧字段/失败版本，不重写历史输出 |

P4 已完成评审，P5 本轮核查具备提交评审条件。若接受上述范围和历史复现边界，可由规划侧确认重构收尾；
效果优化、行为修复、资产迁机和代码提交均需作为后续明确任务处理。
