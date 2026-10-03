# B4-P 长期运行检查入口与有限真实验收

2026-10-03。两个替代入口均通过本批验收，两个旧驱动已退役，停在 B4 整体验收点；未提交，不进入 B5。

## 基线、范围与独立证据

实际起始 HEAD：`57d19dfdb76e20eac6c9d9c057e4f534c92ad694`，Complete B4-D historical tool retirement。
起始工作区只有用户 `docs/TODO.md` 修改，保留原字节。

新建证据目录 `outputs/acceptance/pruning_b4_runtime_20261003/`，总入口为
[execution_manifest.json](../../outputs/acceptance/pruning_b4_runtime_20261003/execution_manifest.json)。
`starting_state.json` 和 snapshot 保存两旧驱动、公共实现、fixture、TODO、选定小型输入/期望及恢复 checkpoint；大权重仅保存路径、大小、mtime，不复制、不计算 SHA256。
所有本次预测、tiny checkpoint、日志、源码副本和报告均在新目录内；历史输出/资产未移动或覆盖。

| 文件 | 变化 |
| --- | --- |
| tools/validate/check_inference_runtime.py | 新增长期手工/Lightning strict 加载与固定输入检查；直接调用公共推理函数 |
| tools/validate/check_fixed_validation_runtime.py | 新增长期回调检查；直接使用公共 FixedValidation 和原 tests.fixed_validation_fixture.run_case |
| tests/test_runtime_check_entries.py | 新增集中标准库安全、参数、设备桩、派发与 cwd 测试 |
| tools/validate/verify_refactor_inference.py | 对应验收通过后删除；旧 capture、三权重自动循环、四入口 AST 提取不迁入新入口 |
| tools/validate/verify_fixed_validation.py | 对应验收通过后删除；旧 capture、固定 GPU1 与历史目录推断退役 |
| docs/COMMANDS.md、docs/READING_GUIDE.md | 最小更新当前调用、输入、计算边界和退役说明 |
| tests/test_trial_callbacks.py | 只改首行说明指向新入口，全部测试实现字节不变 |

公共模型、推理、FixedValidation、训练算法和原 fixture 均未修改。未新增公共执行框架。
两个运行工具新增407行、删除旧工具351行（净+56）；安全测试166行，Python合计净+222行，包含明确输入检查、设备预检和更新预算保护。
此为长期能力替换，不将旧工具整份改名或隐去新增规模。

## 默认展示和执行边界

两个入口默认仅标准库检查显式路径并打印 JSON 计划，不读取权重/NPZ、不导入 Torch/NumPy/项目模型、不查询 GPU、不 mkdir、不启动进程。
必须显式 `--execute`；全部用户路径按调用 cwd 解析为绝对路径，output 必须不存在。
默认展示成功不代表数组、权重或设备可用。

推理必填 checkpoint、model-config、backbone、inputs、expected、route、profile、device、output。
route 为 explicit/lightning；每条路线独立指定原 JSON 及其同名 NPZ。profile 为 batch/item，分别对应独立期望中的 candidate/smoke；不再自动执行其它 evaluator。
输入为完整归一化 `img: float32[3,256,256]`，不是192列中间输入。
明确 backbone 路径设置 `DENSEKP_VITPOSE_BACKBONE` 后才导入模型；两个公共加载器保留先读 checkpoint、构造模型时读初始化资产、strict 加载的原顺序。
原始构造模式为 training=True，调用方执行 to(device).eval()；batch 保持模式，item 仍由公共实现调用 eval。
未增加裁剪偏移、置信度解释或评价公式。

回调必填 source、before、output、scenario、device、max-updates；resumed 另需独立 step2 resume-checkpoint。
复用原 TinyModel、Trainer、采样和报告编码，不复制 fixture 实现。源码 checkout 必须包含 tests，未承诺只带包的 wheel 可用。
十个既有场景的 failure/resume 参数映射保留；success需要4次、resumed2次、其它各2次替代小网络更新。
AdamW 更新前检查原小网络参数形状/顺序并递增尝试计数；预算到达即拒绝，不重试。
比较仍是原 metadata JSON 序列化一致及数组 dtype/shape/值精确一致，仅原 fixture 已有的 seconds/输出根规范化。

执行前对明确物理编号或完整 UUID 查询设备和进程；设备查询失败、不匹配、显存不足即停止，不自动换卡、不重置、不杀进程。
返回 UUID 经 CUDA_VISIBLE_DEVICES 映射为逻辑 cuda:0。记录进程不等于保证GPU无其它用户。

## 本次准确输入和计算

所有以下相对路径均相对仓库；实际绝对 argv、cwd、必要环境差量在 `commands.json` 和三个 `*_execution.json`，没有保存父进程完整环境。

- checkpoint：`assets_external/densekp.ckpt`（8049335298 bytes）；初始化：`assets_external/vitpose_backbone.pth`。
- model-config：`outputs/acceptance/refactor_inference_audit_20261001/before/checkpoint_loads/original_config.yaml`。
- 输入根：`outputs/acceptance/refactor_inference_audit_20261001/before/inference_gpu1/inputs/`，只读 `val_baseline_002.npz`、`val_baseline_001.npz` 的完整 img。
- 归一化沿原 RGB mean=(.485,.456,.406)、std=(.229,.224,.225)；本次检查完整已保存输入与独立期望精确一致，不重新生成图像或期望。
- 两路线 expected：同一 inference_gpu1 目录的 `original_explicit.json/.npz`、`original_lightning.json/.npz`。
- callback source：`outputs/acceptance/refactor_metrics_config_20261001/before`，cases.json 和 002/001/004 三个 NPZ，顺序由原 fixture 保持。
- callback before：`outputs/acceptance/refactor_reporting_callbacks_20261001/callback_before`。
- resume：该 before 下 `success/step_0002/checkpoint.ckpt`。只读元数据确认 global_step=2、一个optimizer及四个既有小网络参数，不补训练制作基线。
- 三个场景：success(4)、resumed(2)、save_failure(2)；只运行新入口一次，以原独立期望对照，未双跑回调 before/after。

设备为物理1、RTX3090、`GPU-6f046d56-acbe-d49b-774a-9d463e768cac` → cuda:0。
全卡/UUID查询受异常 GPU0 阻断；沙箱内亦无法访问驱动，失败均发生于计算前（0前向/0更新）。
改为明确物理编号1查询，核对仍是原定同一 UUID，然后在获准的沙箱外执行；不是自动换卡。
各次执行前记录有约23621MiB空闲和既有 Blender PID2824895 占82MiB，该进程未干预。
`device_initial.json` 保存初次失败记录，三份 `*_preflight.json` 和新入口 device.json 保存成功即时核查。
NVML告警及 checkpoint 的 Lightning2.0.7/本机2.0.2版本告警保留，未因此改比较规则。
环境：Python3.10.20、Torch2.0.0+cu118、Lightning2.0.2、NumPy1.26.4；未安装或升级。

## 真实验收结果

| 检查 | 实际结果 | 边界 |
| --- | --- | --- |
| 独立旧推理限定分支 | 原驱动源码快照提取，只保留原始权重、两输入、candidate与原predict_item、两加载路线；8前向通过 | 未直接运行旧硬编码main/三权重循环，不调用capture |
| 新推理入口 | 两路线各496 key，strict无缺失/多余，schema/输入/模式/RNG通过；8前向 | 共4次DenseKP构造（前后各两路线），没有参数更新 |
| 前后数组 | 每路线12数组，24数组 dtype/shape/顺序/值精确相同，预测最大差0 | 独立原期望 atol=1e-5、rtol=0不变；未统一不同预测函数 |
| CUDA success | 4更新、190数组及元数据精确一致 | 含step2/4保存、参数/优化器/模式、CPU/CUDA RNG |
| CUDA resumed | 从既有step2恢复，2更新、95数组及元数据精确一致 | 未重新训练恢复基线 |
| CUDA save_failure | 2更新、73数组及元数据精确一致 | 保留预期OSError、停止模式、只写validation/prediction而无checkpoint的部分制品 |
| 更新预算 | attempted=8、completed=8，DenseKP更新=0 | 无失败重试；不执行其它七场景真实更新 |

`numerical_comparison.json`、各 after/result.json 记录加载顺序、比较、state_key=FixedValidation、异常和计数。
回调358数组覆盖参数、优化器、RNG、保存状态；所有字段使用原比较规则，未增加忽略项或重录期望。
旧限定推理侧与新工具没有共同调用新工具的比较或构造逻辑；公共实现原字节保持，旧期待来自原历史独立采集。

## 安全测试、交付与保护

具体方法ID及调用闭包在 `test_ids.json`：

1. RuntimeEntryTests.test_default_cold_plan_has_no_writes_processes_or_heavy_imports
2. RuntimeEntryTests.test_explicit_execute_dispatch_keeps_plan
3. RuntimeEntryTests.test_missing_input_and_existing_output_rejected
4. RuntimeEntryTests.test_required_route_profile_device_and_pairs
5. RuntimeEntryTests.test_all_callback_scenarios_map_without_computation
6. RuntimeEntryTests.test_resume_and_update_budget_are_explicit
7. RuntimeEntryTests.test_device_query_failures_and_mapping
8. RuntimeEntryTests.test_relative_paths_follow_invocation_cwd

上述方法均在 tests.test_runtime_check_entries 下。未discover、未运行DenseKP lifecycle或其它训练入口。
8方法在原仓库和无资产源码副本、各根cwd/临时cwd共4轮全部通过，无skip；前期同8方法也通过。
冷进程审计阻断重型导入、权重/NPZ读取、写入、mkdir与进程；执行分支用桩，七个未实测回调场景不记为真实验收。
副本只含两个入口、安全测试与原fixture源码/包标记，以标准库Python隔离模式运行，没有原仓editable补模块；不是空环境安装或editable重装。

保护文件逐字节核验，包含TODO、全部原数值fixture、选定独立期望、原step2、公共模型/推理/回调和活动status。
两个大权重前后size/mtime一致，未称为全文件哈希验证；实现只读它们。
保留源码/测试/配置/命令无删除模块依赖；文档旧名只作明确退役和Git历史说明。新源码/测试可纳入Git，未被忽略。
日常检查无需本机acceptance目录；没有修改check_workspace范围。
一次性 final_checks 脚本初版误将 weight_stats 列表当字典，静态保护检查报错后按实际结构修正并继续尾部核查；未发生模型/参数更新重试。

## 停止点和限制

本批两驱动均可退役，无未解决删除依赖。B4-D与本批汇总见 [B4完成表](PRUNING_B4_COMPLETION_20261003.md)。
只验证原始一份权重、两个输入及所列profile/场景；其它权重、全数据评价、七个CUDA失败场景未实测。
未进行真实DenseKP训练、制作、SMPL-X、Blender或渲染；已有Blender进程非本次启动。
本次小网络更新是工具验收，不产生新的DenseKP效果结论，框增强权重仍未获部署替换认可。
未提交、未开始B5，等待B4整体验收。
