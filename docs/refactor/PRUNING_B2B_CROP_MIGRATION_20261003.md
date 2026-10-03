# B2b 首批：裁剪调用迁移与 Dataset capture 退役

日期：2026-10-03。**本批实现和有界 CPU 验收完成，未提交；停在验收点。旧模块与评分入口未删除。**

[执行清单及完整测试 ID](../../outputs/acceptance/pruning_b2b_crop_migration_20261003/execution_manifest.json) · [制品对照](../../outputs/acceptance/pruning_b2b_crop_migration_20261003/comparison.json) · [剩余依赖](../../outputs/acceptance/pruning_b2b_crop_migration_20261003/remaining_dependencies.json)

## 实际基线与修改范围

起始分支 master，HEAD `8db7ca5ec7583d1749db2e5f33e96abf3eff5a1e`。工作区已有 B2a 两批未提交成果（loop 改名、四继承者、八份预设/示例删除、相关说明/报告），另有用户 docs/TODO.md 修改；完整起始 status 保存在 starting_state.json。未重置、覆盖或重新实施 B2a。

实际修改六个源码/测试文件：

| 文件 | 改动 |
| --- | --- |
| tools/annotation/visualize_annotation.py | crop_example + CropBox 直调，读取 CropResult 具名字段；辅助函数直引定义模块 |
| tools/validate/validate_bedlam_densekp_tiny.py | 仅将 trans_points2d_parallel 导入改为 crop_geometry，其它计算不变 |
| tests/dataset_crop_fixture.py | 具名结果在测试内编码为原 result_N；original=True 先明确拒绝 |
| tests/test_dataset_crop_contract.py | 低层方法改名 test_named_outputs_mutations_failures_and_rng；全部原断言不变 |
| tests/test_densekp_boundaries.py | 仅几何测试改用 crop_example/sample_augmentation 及具名字段；不改 Objective/Lifecycle |
| tools/validate/verify_dataset_crop.py | capture 在读取基线、导入数据模块、mkdir/write 前报错；verify 原比较循环和报告构造保留 |

新增 `tests/test_dataset_capture_retirement.py` 两项有界拒绝检查。说明仅更新 docs/SAMPLE_WALKTHROUGH.md、docs/READING_GUIDE.md、tests/fixtures/README.md，另新增本报告。本轮没有新增删除项，没有修改公共裁剪算法、Hydra、注册表、模型或其它清理目标。

## 参数、返回值与顺序

旧 get_example_projverts 的 center_x/center_y/width/height 对应源图像素 `CropBox`；patch_width/patch_height 对应 `patch_size=(width,height)`。keypoints/proj_verts 的对象与调用方 copy 位置不变，其余 mean/std、增强、BGR、抗混叠、border_mode、dataset 参数保持。

| 原 tuple 下标 | 当前具名字段 |
| --- | --- |
| 0 / 1 / 2 | image / display / image_size |
| 3 / 4 | plan.box.center_x / center_y |
| 5 / 6 | plan.box.width / height |
| 7 / 8 / 9 | keypoints / dense_points / affine |

display 仍是未归一化 HWC BGR 图；image 为归一化 CHW float32。plan.box 的宽高尚未乘增强 scale，可视化报告继续使用这两个值。辅助 44 点与 DenseKP 138 点不混用，标签仍按完整裁剪坐标变换，不增加内部 192 列输入相关的 32 px 偏移。

visualize 仍先读图、按 dataset_name 旋转、绘制并保存原图，再传已读 ndarray 和 proj_verts.copy() 裁剪，dataset=None 防止重复旋转。之后画裁剪图并保存报告，保持调用/写出顺序。无效 flip_kp_permutation/return_trans 参数只从迁移后的调用处去除；公共旧包装没有改变。

run_lowlevel 的原 11 个用例全部保留。只在测试辅助中将具名字段排成原 result_N 编码，类型、顺序及输入变异也按原 fixture 检查；return_trans_flag 原用例仍在，新公共接口本来始终含 affine，不需要该标志。sample_augmentation 的调用次数/顺序不变，只改为按名读取 translation_x/y。

## 独立 before 与执行边界

新目录为 `outputs/acceptance/pruning_b2b_crop_migration_20261003/`，没有覆盖历史证据。编辑前完成：

- 保存 502 个相关源码、配置、文档、fixture 与前批具名证据的原字节到 protected/。
- 核实 P2 的 45 个规格、七固定样本及独立 before 文件存在；针对实际引用的 loader、图片和原 before 文件另保存 101 份字节快照。没有扫描全部 outputs 或权重。
- 复用原规格，使用有效的 verify 路线读取 P2 独立 before；另采集当前旧调用方的低层、可视化和受控 BEDLAM before，全部完成后写 before_gate.json，再开始编辑。
- 在新目录构造小型测试输入，另保存 input_snapshot/。未调用已经失效的历史 capture，没有重录或改动原 fixture。

一次性 acceptance.py 和 compare.py 位于新验收目录，前者只承担本批输入、采集和显式测试执行。解释器是已有 `/home/yy/conda_envs/camerahmr/bin/python`，禁用 bytecode，CUDA_VISIBLE_DEVICES 为空。阻断 torch.nn.Module 构造、Trainer 构造/fit、Optimizer 构造/step、torch.load、CUDA 初始化和 DenseKP 模块导入；写出仅限新验收目录，临时测试数据也置于其中。

执行顺序为 `acceptance.py prepare` → `capture --phase before` 与 `tests --phase before` → before_gate → 修改 → `capture --phase after` 与 `tests --phase after` → compare.py。未运行训练/推理/渲染、模型或 Dataset 之外的数据生成，也未下载、安装或升级依赖。未计算 SHA256。

## 分项验收

| 路径 | 实际结果 |
| --- | --- |
| Dataset | 原31个合成规格 + 七个真实样本各关闭/启用增强，共45规格；旧ID/显式路径90条路线，两次均通过原 P2 before verify；本批 before/after 2814数组精确相等 |
| 低层裁剪 | 11例、156数组精确一致；含RGB/RGBA、浮点/抗混叠、无归一化、无dense标签、旧无效flag、增强、无效宽度/类型/缺图/flip拒绝；输入原地变异、异常与Python/NumPy/Torch CPU RNG均保持 |
| 可视化 | 七真实样本 + 相对/绝对路径、portrait、closeup、bbox比例、越界索引、缺字段、缺图、零scale，共16例；12例成功、4例预期失败；含零scale失败前已写原图在内的25张PNG字节一致，报告及IO顺序轨迹一致 |
| BEDLAM validator | 两个文档候选真实tiny目录均不存在；新目录内2行受控图像/NPZ/debug输入，4张PNG及报告精确一致。只验调用迁移，不声称真实BEDLAM回读通过 |
| capture退役 | CLI退出码2，错误包含 capture retired in B2b；审计钩子阻断数据模块导入、数据读/写和mkdir，输出目录不创建。original=True 在第一分支 RuntimeError，旧导入、配置/采样和IO均未触及 |
| B1 metrics capture | 原有拒绝测试前后均通过，工具未修改 |

制品合计比较 101 个 NPZ 中的 **2970 个数组、115 份制品 JSON、29 张图片**，另比较可视化/validator 结果轨迹。数组除数值、dtype、shape、字段顺序外还比较原始数组字节；PNG 和 JSON 按文件字节比较，未放宽容差或过滤报告字段。

唯一随新输出目录变化的是各成功可视化/validator 的 stdout 最后一行 `audit report:` 或 `verification report:`（共13行）。comparison.json 逐项列出文件、行号、旧/新值；只校验该行输出路径替换，其余 stdout 内容一致。图像与报告没有任何忽略字段。测试日志耗时和方法名变化不作为业务输出比较。

前测13项、后测15项全部通过，无跳过。明确白名单为 DatasetCropContractTests 的6个方法、GeometryContractTests 的6个方法、ImportBoundaryTests.test_retired_metric_capture_rejects_before_data_imports_or_writes，后测另加：

- DatasetCaptureRetirementTests.test_cli_capture_rejects_before_data_imports_or_writes。
- DatasetCaptureRetirementTests.test_original_helper_rejects_before_imports_sampling_or_io。

完整逐方法 ID 位于执行清单和 before/after/tests.json。没有运行 unittest discover、整个 boundaries 模块、ObjectiveTests 或 LifecycleTests。Objective/Lifecycle/TinyHead 类 AST 原样，Dataset 测试除方法名外 AST 完全一致；verify 的原比较循环和报告 AST 完全一致。Geometry 原有 atol/rtol 断言保持，不借对照调整算法。

## 保护与仍待处理的旧消费者

493 个范围外保护文件逐字节一致；原 fixtures（不含获准更新的 README）、TODO、B2a 配置/成果以及具名历史输入/基线均保持。新长期测试未被忽略。日常现有检查列表未改，现行裁剪契约仍只依赖仓内原 JSON 与临时输入；新增拒绝测试也不依赖 outputs 或本机验收目录。

下列旧依赖仍在，不能宣称旧接口全部脱钩：

| 消费者/关系 | 下一批删除前置条件 |
| --- | --- |
| evaluate_omni_densekp.py:25,106 仍导入/调用 get_example_projverts | 旧评分能力退役需单独评审；先处理其动态消费者 |
| test_legacy_cleanup.py:23 导入 utils as crop，传入 legacy_cleanup_fixture.capture_crop | 旧裁剪能力和专用测试尚未退役；本批全部保留 |
| utils.py:101 仍调用 CropResult.legacy_tuple，并导出 legacy.crop | 等剩余实际消费者迁走或明确退役后才可删包装/方法/legacy包 |
| geometry_camera_fixture.TARGETS 的 erp_points 仍指向旧 Omni evaluator，geometry测试使用 api(original=True) | 未来删 evaluator 前需单独迁动态路由；本批未改 |
| verify_geometry_camera.run 仍走 api(original) | geometry capture 留后续处理，本批未触及 |

utils.py、legacy/crop.py、legacy/__init__.py、CropResult.legacy_tuple、两个旧评分入口、旧裁剪专用测试和所有原 fixture 均仍在且未改。constants、core/configs、注册表、misc、人物制作链、模型和 Hydra 未动。

没有超出预期的数值或行为差异。真实 BEDLAM 数据缺失是已明确的验收限制；本批不证明真实模型或 GPU 行为。HEAD 保持起始值，未暂存或提交；完成后停在本批验收点，不继续删除或其它清理。
