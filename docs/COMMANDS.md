# 推荐命令与计算边界

以下以仓库根为 cwd，`python` 指已准备依赖的独立开发环境。其他 cwd 下，tools 使用脚本绝对路径；
安装后的双包可直接 import。安装、版本和逐项路径解析见 [环境说明](environment/INSTALLATION.md)，
配置角色见 [配置索引](../configs/README.md)。本页代码块只提供检查/打印命令；计算入口在表中列出所需参数，
真正执行前应先准备输入、选择新输出目录和设备。`--help` 成功不代表资产已齐备。

## 先做检查

```bash
bash scripts/audit_workspace.sh
bash scripts/audit_workspace.sh --model-contracts
python -c 'import densekp, densekp_retrain; print(densekp.__file__); print(densekp_retrain.__file__)'
python -m densekp.train_keypoints --cfg job trainer=cpu
python tools/annotation/inspect_annotation.py --help
python tools/validate/evaluate_candidate_baseline.py --help
python tools/experiments/train_source_isolated_trial.py --help
python tools/render/run_temporal_case.py --help
python tools/fisheye/run_fisheye_camera_p0.py --help
```

前两条只检查源码、配置与无资产 CPU 契约，不遍历 outputs、权重或纹理。
`--model-contracts` 不加载真实模型；其中几何数组契约需要安装表中的 geometry 依赖（含 smplx 的纯公式），不读取人体资产。第三条验证双包导入，需要先完成 editable 安装或相应开发路径设置。
第四条只打印 Hydra 配置；`trainer=cpu` 本身不关闭训练，**不能删掉 `--cfg job` 当作数据检查命令**。
默认 `data=unselected` 不变。后五条仅打印帮助，需各入口的 Python 依赖。
这些入口的根目录/临时 cwd 行为已在 P4 验证；B5仅静态复核参数及执行分支，没有重新运行命令。

## 按用途选择入口

回归调度使用 `tools/validate/run_regression_suite.py`，默认只打印 JSON 计划（仅标准库，
不启动子进程、不查询 GPU、不写文件、不读取权重）。例如：

```bash
python tools/validate/run_regression_suite.py --suite source_trial --trial /path/to/trial --data-root /path/to/data
python tools/validate/run_regression_suite.py --suite box_robustness --experiment /path/to/experiment
```

计划列出完整 argv、输入/输出、reference、默认与显式 model_config/device、业务 gate 和汇总顺序。
缺训练报告或 gate 不通过会显示原因；展示成功不表示可推理。显式增加 `--execute` 才启动实际评价，
且必须给 `--gpu-uuid GPU-…` 或明确 `--physical-gpu N`；同时给出时必须指向同一设备。
执行先查询设备和进程，再执行原业务 gate，每个推理任务前复查，子进程经完整 UUID 映射到 cuda:0。
不自动换卡、重置或终止进程；记录进程不等于保证空闲。

source 按 main → paired → regression 执行，随后调用原配对汇总及 compare_source_trial。
box 按 challenge_A/B/C → val_C → regression_C → history_B/C 执行；compare_box_robustness
仅列为独立后续命令，不自动运行，且它固定读取 evaluation_v2，不跟随调度器的 `--output-name`。
box 保留 `--output-name`（默认 evaluation_v2）和 `--only`：筛选按原任务顺序，必须已有 execution.json；
未知名称可得到零任务，不能据此认定评价已完成。完整运行拒绝已有输出目录，部分重跑也不替 evaluator 覆盖旧结果。
source 先 gate 再 mkdir，box 先 mkdir 再 gate；新增设备前置检查先于两者，失败不产生业务输出。
`--reference-report` 是输入一致性约束，不复用预测。两套 evaluator 的公式、分母与单位保持各自规则。

旧 `tools/experiments/evaluate_source_trial.py`、`evaluate_box_robustness.py` 已退役，无兼容包装。
suite 路径配方见 [配置](../configs/regression/suites.json)，不自动选择最新权重或修改活动 status。
B3调度迁移只用桩验收调度，没有实际 GPU、推理、汇总或比较；框增强权重仍未获部署替换认可。

| 用途与入口 | 必要输入/选择 | 输出与执行边界 | 环境与外部资产 |
| --- | --- | --- | --- |
| 标签检查：`tools/annotation/inspect_annotation.py` | 位置参数为 NPZ 的绝对路径 | 读取并打印契约；不生成预测。它不代替图片/Dataset 回读 | data/model 依赖按安装表；显式 NPZ |
| 小型 CPU 安装验收：`tools/validate/verify_installation_runtime.py` | `--mode data --output` 指向不存在的新目录 | 创建小 PNG/NPZ 并走 Dataset；会写验收文件，不读取真实模型 | model 依赖；不需要项目资产。stdlib/light 模式有更小依赖面 |
| 保存预测重新评分：`tools/validate/evaluate_ds_occlusion_control.py` | `--root` 原场景组、`--reuse-predictions` 对应旧预测目录、`--output` 新目录 | 校验同源、框、GT、affine；写 NPZ、JSON 和图片。省略 reuse 参数会实际加载模型推理 | model 依赖和完整保存产物；重放不能证明图片内容从保存日起未变 |
| 候选推理：`tools/validate/evaluate_candidate_baseline.py` | `--indices` 与等长 `--labels`、新 `--output`；来源隔离输入另需 `--split-manifest`；明确 `--checkpoint`、`--device`，可选 `--model-config` | 实际推理，写冻结输入、逐例预测/图片、汇总报告；默认 device 为 cuda:0。`--reference-report` 只做一致性检查，不关闭推理 | model 依赖、完整 DenseKP checkpoint、ViTPose 初始化、NPZ/图像、来源记录 |
| 通用训练：`densekp.train_keypoints` | 显式 Hydra data 组、监督语义、trainer、权重/恢复路径；先用 `--cfg job` 查看 | 去掉配置打印参数后进入真实 train/test 生命周期，写 Hydra 运行目录、日志和 checkpoint；`init_checkpoint` 初始化权重，`ckpt_path` 恢复状态 | model 依赖、数据注册路径或适配入口、ViTPose 和初始模型权重 |
| 已冻结的 400 步配方：`tools/experiments/train_source_isolated_trial.py` | `--data-root` 含 split、indices、loader/acceptance、original_baseline、training_recommendation；可选 `--box-data`、`--output`、`--gpu` | 默认执行数据/基线审计，不能当作纯配置打印；`--execute-training` 才执行训练。步数等配方保持原值，不作为日常检查 | model 依赖、来源隔离全套资产和原 checkpoint；不自动修补缺失基线 |
| 透视时间序列渲染：`tools/render/run_temporal_case.py` | `--geometry --output-dir --frames --hdri`；可选 clothing、appearance、Blender 路径 | 直接启动 Blender，输出渲染制品与报告，没有通用 dry-run 承诺 | Blender、人体/衣物、UV/贴图/HDRI；详见配方索引 |
| Omni 生成/后处理：`tools/fisheye/run_fisheye_camera_p0.py` | 显式 `--geometry --clothing-geometry --output-dir --appearance-manifest`；核对 frame/分辨率 | 默认含渲染；`--skip-render` 仍写相机/几何副本并执行后处理，且依赖输出目录已有源 EXR。默认输出可复用旧目录，不建议裸执行 | Blender（渲染时）、普通 CPU 图像/EXR 依赖、已有几何与外观资产 |
| DS 多人及场景制作 | 从 [场景配方索引](rendering/SCENE_RECIPE_INDEX.md) 选择具体入口 | 原生多人、链接单人、专用场景制作保留独立默认值和旧 Blender `--` 参数，不互换命令 | 对应 `.blend`、贴图、人体/衣物；本轮不制作任何新资产 |

后续 DS 生成入口 `render_cycling_ds.py`、`render_cycling_characters_ds.py` 和
`render_adult_candidates_ds.py` 默认采用 compact 与 C 组尺寸（源/最终 ERP 均
5760×2880，DS 2880×2880）；需要调试证据时显式给 `--output-mode full`。
compact 临时目录可用 `--temporary-root` 指向尚不存在的任务独占目录；失败会保留，
成功回读后才清理。最小形式如下（其他人物、资产和真实鱼眼 mask 参数仍按各入口 `--help`）：

```bash
python tools/experiments/render_adult_candidates_ds.py \
  --assets /path/to/assets --output /new/output \
  --characters ID1 ID2 --output-mode compact
```

历史复核命令应显式 `--output-mode full` 并保留原 `--width`，不得用新默认改写旧实验。
有界验收入口为 `tools/validate/validate_ds_output_modes.py --output /new/acceptance`；它复用
已有 C 源缓冲，不启动 Blender、模型或推理，但会复制约 0.9 GiB 源文件并运行两次 CPU 后处理。
已完成结果和失败记录见 [验收报告](rendering/DS_OUTPUT_MODES_20261008.md)。

GPU 计算前先检查设备和进程，不假定 GPU 0 正常；选择的设备应显式传入相应入口。
文档收尾没有执行设备查询、推理、训练或渲染。

## 路径与复现

- 推荐对用户输入用绝对路径。Dataset 的 annotation_path/image_root 相对 cwd，imgname 相对 image_root；绝对 imgname 保持绝对路径。
- 仓库内配置查找以源码位置定位根目录；这不等于所有配置内相对资产路径都相对配置文件。scene blend、appearance 与历史 job 的各自规则见环境说明。
- ViTPose 默认初始化路径仍依赖 cwd 或 `DENSEKP_VITPOSE_BACKBONE`；从其他目录执行真实模型路径需显式准备它。P4 的跨目录 help/config 验收不证明真实权重路径可省略。
- 六份 Hydra data 配置已补齐交付，未启用原来无效的 loss/head 配置，也未改变默认数据选择。
- 历史命令/报告保留当时的设备、目录与状态。当前活动版本从 `outputs/experiments/box_robustness_400_20261001/status.json` 的 `active` 字段进入；其部署替换仍未获认可。

## 已退役的裁剪与评分入口

B2b 停止旧 get_example/get_example_projverts 导入与 tuple 返回，以及
`tools/validate/evaluate_omni_densekp.py`、`tools/validate/evaluate_densekp_npz.py` 两个 CLI。
前者的自动造框、ERP 居中与三相机评分、后者的注册 ID 评分不再支持；当前候选评价
不是其过滤规则、分母或报告的等价替代。Omni 生成、历史 NPZ 读取和监督权重保持。
当前裁剪使用 crop_pipeline.crop_example / CropBox / CropResult。
B4-D 已退役历史 Dataset/geometry/metrics 等迁移 capture/verify CLI；当前数值契约继续使用原 fixture，历史回放制品保留。
参见 [退役能力和保留边界](KNOWN_BEHAVIOR.md)。


## 按需长期检查与 B4-D 退役范围

B4-D 停止 BVH/BABEL 骑行检索和拟合输入、失败衣物修复/union/settling/polo、
独立 riverfront 制作、两套旧 Omni sweep、P0 pyrender 检查，以及列明的一次性迁移、smoke 和画廊命令。
它们不是当前入口的等价别名；原数据、权重、保存预测、失败结果和许可记录继续保留。
逐文件清单见 [最终交付及历史清单定位](refactor/PRUNING_FINAL_20261003.md)。

以下检查不属于默认日常命令，不应为了清理代码就启动：

| 检查 | 必要输入/输出 | 计算边界 |
| --- | --- | --- |
| `check_inference_runtime.py` | 必填 `--checkpoint --model-config --backbone --inputs --expected --route --profile --device --output`；expected 按 route 顺序提供独立 JSON 及同名 NPZ，inputs 为完整归一化 `img: float32[3,256,256]` | 默认仅标准库打印；`--execute` 才两类 strict 加载及所选前向。route 为 explicit/lightning；profile 为 batch/item；atol=1e-5、rtol=0，不执行参数更新 |
| `check_fixed_validation_runtime.py` | 必填 `--source --before --output --scenario --device --max-updates`；resumed 必须另给 `--resume-checkpoint`（独立 step2）；before 含所选场景 JSON/NPZ | 默认仅打印；`--execute` 使用公共 FixedValidation 和源码 tests.fixed_validation_fixture.run_case。success 4次、resumed 2次、其余每场景2次小网络更新，受累计尝试预算限制；不是无需 tests 的 wheel 入口 |
| `verify_densekp_model_lifecycle.py` | 显式 phase/checkpoint/output-dir；resume 另需 reference-dir；内部固定 p0_dressed_tiny、trainer=gpu，无 --config 参数 | 真实 DenseKP 更新/保存恢复；需要注册数据与初始化资产，不是无资产检查 |
| `verify_blender_assembly.py` | 显式 case、mode assembly/render、output | 真实 Blender 装配；只有触及渲染行为才安排有限渲染，不默认重做场景 |
| `compare_blender_acceptance.py` | 位置参数 `before after`，另给 `--report` | CPU 读取选定制品，含 EXR 通道；不启动渲染 |
| `verify_blender_postprocess.py` | evidence/run/output，另依赖原 val_000 lens、real masks 与 Aaron 几何 | 真实源 EXR 的 DS/Omni 后处理检查，尚非通用无资产入口；保持两种距离来源差异 |

任何后续 GPU 执行前即时检查设备与进程，不能沿用旧状态。上表不是自动执行清单。
普通 Python 导入安全只检查 audit_skin_texture、prepare_procedural_eyes、blender_headless_smoke_test；
三个场景脚本的导入检查在真实 Blender 内单独运行，允许 bpy/mathutils，但禁止 main、资产 IO、场景操作、写出和进程。
历史完整工具可只读查看，例如 `git show ada2295:tools/experiments/repair_commuter_top.py`；
Git 不备份 ignored 的 outputs/资产，也不保证历史代码在当前环境无需准备即可执行。


以上两个 runtime 入口的所有用户路径按调用 cwd 解析，建议给绝对路径；output 必须不存在。
`--device` 为明确物理编号或完整 GPU UUID，执行前查询该设备及进程，再以返回 UUID 映射到逻辑 cuda:0。
查询失败、UUID 不匹配或可用显存不足会拒绝，不换卡、不重置、不终止进程。
默认展示只检查文件存在性，不读取权重/数组、不查询设备、不创建文件；它不保证数值输入有效。
推理入口显式用 `--backbone` 设置初始化资产路径，不从旧历史目录或“最新权重”推断。
回调 source 需 cases.json 及 val_baseline_002/001/004.npz；保留既有 fixture 的采样、比较及异常语义。

旧 verify_refactor_inference / verify_fixed_validation CLI 和其历史 capture、AST 提取已退役。
历史代码可从 `git show 57d19df:tools/validate/verify_fixed_validation.py` 查阅；独立 before 与原结果保留在本地 outputs，Git 不为它们提供备份。
B4-P仅实测原始权重、两个固定输入、两路线×batch/item，以及 success/resumed/save_failure；其它回调场景只验参数派发。
具体输入与计算计数见 [真实验收范围](VALIDATION.md)。不代表新效果评价或部署认可。
