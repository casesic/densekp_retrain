# P2.3–P2.5：来源、打包、框与监督适配

完成日期：2026-10-02。P2.1/P2.2 已通过评审；本批完成 P2.3–P2.5 的实现与本地验收，
**P2 整体待评审**。没有开始 P3。

## 1. 独立基线与操作边界

分支 `master`，HEAD `a3a5e7bb641bf1dee774d015127fee6b39c28bda` 未变。
保留前批全部未提交实现及历史产物；本批数据仅写到
[refactor_sources_boxes_20261002](../../outputs/acceptance/refactor_sources_boxes_20261002/)。
`starting_state.json` 保存开始时状态/diff；`before/source_snapshot/` 保存迁移前源码。

在旧实现仍为活动代码时先执行完整打包和框派生，得到 `before/data/`：
198/79 行原始 loader、400/143 行衍生 loader/index、逐行裁剪和几何、419 次 Generator 调用及状态、
Python/NumPy/Torch RNG、37 项来源边界和契约/纯框结果。
不覆盖原 NPZ、索引、split、已有预测或任何 scene 文件。

独立拒绝基线在 `before/edges_v3/`，由冻结的旧源码执行；
`before/edges_v2/` 另保留非有限框导致严格 JSON 失败的原行为。
`before/edges/` 是首次注入过早、直接使参考回退失败的试验记录，保留但不作为成功回退基线。
标注适配的 `before/adapters/` 用冻结的旧 adapter 和旧监督函数执行；这组在主代码迁移后才执行，
但其实现来自迁移前快照，未调用新的被迁移函数或共用新包装来生成期望值。
before 从未覆盖或重新生成以匹配 after。

未运行 GPU、真实模型推理、DenseKP 效果训练、渲染或资产下载；没有 commit/reset/clean。
CPU 回归含原有小网络更新与保存恢复，未更新真实 DenseKP 参数。
`.gitignore` 未改；`docs/TODO.md` 在本批期间出现用户新增框增强备注，原内容和新增内容均保留，
详见 `concurrent_user_edit.json`。没有把这项并行编辑当作本批重构修改。

## 2. 实际代码与阅读顺序

| 文件 | 职责及兼容边界 |
| --- | --- |
| [data/source_split.py](../../src/densekp_retrain/data/source_split.py) | `validate_split` 分为声明来源检查、索引/场景人物检查、集合汇总；保留字段含义、检查顺序和错误文本 |
| [data/packing.py](../../src/densekp_retrain/data/packing.py) | `pack_index_rows` 按索引行序读取源 NPZ；`save_index_loader` 写原字段；`pack_derivative_rows` 保留衍生行全部字段及顺序 |
| [data/boxes.py](../../src/densekp_retrain/data/boxes.py) | `change_box`、`trial_crop_from_box`、框/支持量拒绝纯函数，以及保留独立策略的 BEDLAM `official_bbox` |
| [data/loading.py](../../src/densekp_retrain/data/loading.py) | `load_box_dataset` 保留 box-trial 名称、根目录 `/`、显式路径、关闭增强和训练/验证分支 |
| [keypoint_supervision.py](../../densekp/core/keypoint_supervision.py) | `supervision_for_crop` 共用 `in_view & inside_full_crop`；`legacy_weight_targets` 与 `in_view_targets` 明确区分旧权重和待实际裁剪重算的标签 |
| [build_source_isolated_trial.py](../../tools/annotation/build_source_isolated_trial.py) | 入口依次核查来源 → 公共打包 → 实际 Dataset 回读 → 输出检查记录；展开原密集单行 |
| [prepare_box_robustness.py](../../tools/experiments/prepare_box_robustness.py) | 保留 seed、配额、父样本顺序、随机参数配方和回退组合；纯计算直接调用公共模块，局部构建函数改为 `append_derivative` |

所有活动来源核查入口直接导入 `data.source_split`，不再依赖历史 validate 脚本。
source trial 的 box Dataset 工厂直接来自公共模块；audit 的 JSON writer 直接使用已存在的公共 writer。
旧 validate 脚本导出 `validate_split`，旧 box 脚本导出 `change_box`、`load_dataset`、`write`，
BEDLAM 入口导出 `official_bbox`，用于旧调用兼容。公共数据模块没有反向导入 tools。

`run_source_isolated_batch.py` 补上仓库 `src` 路径，使独立命令可定位新公共模块；
未改其场景生成计划或启动渲染。两包安装整理仍留 P4。

## 3. 保留的来源与监督策略

来源核查不仅遍历目标索引行，还读取每个 scene 的 `people.json`，检查所有已渲染人物，
包括未产生目标框的人及遮挡者。identity family、shape/face group、纹理、ancestry、动作来源、
图像/双视图 layout 和 paired-box 签名继续分别检查。声明字段不是未知真实身份无泄漏的证明。

打包保留行序和 source 行号，不重排、重采样或过滤。不同历史策略保持：

- 新 DS/omni adapter 继续使用 `make_targets`、显式有效 mask；Dataset 在实际 affine 后重算完整裁剪监督。
- paired source 初始标签仍只写原图 XY 与 in_view；不提前改成 make_targets 的固定框计算。
- 旧 rendered/omni 权重常数和 box-control 可见标签经 `legacy_weight_targets` 原样装配，
  不升级为新监督策略，不给旧 NPZ 增加 policy 或 visible 字段。
- BEDLAM 保留原齐次第三值及原序列化；不把它自动解释为新置信度/可见度。
  这些入口在 Dataset 消费时仍要求已有 `supervision_weight` 明确声明。
- commuter-studio 仍无 visibility/policy 元数据，仅作关闭增强的 preprocessing 检查。
- `validate_annotations` 与 objective 文件字节未改；零权重/非有限 XY、空监督、范围检查及混合 batch 规则不变。

不在 producer 写入前新增验证，以免改变错误发生时机或已有失败产物。
原 packing 在 loader 回读时执行契约；来源字段缺失的 KeyError、未知 policy、空行等原行为均保留。
普通打包允许索引计数匹配的空监督；框增强仍拒绝监督少于 10 或可见目标像素少于 16 的候选。
遮挡不会取消可靠 GT，目标像素支持阈值与逐点 GT 监督是不同检查。

`change_box` 保留 **1536×768 合成 ERP** 截断，移动中心以原框最大边 ×1.2 为基准，单侧缩短按原轴长。
`trial_crop_from_box` 保留 float32 转换与乘除顺序；BEDLAM 前 22 关节、至少 8 个有效关节及包含图像边界的规则独立保留。
这些不是 2880 鱼眼/5760×2880 真实 ERP 的通用默认值。框配额和随机采样没有迁入公共计算模块。

## 4. 已执行验收

所有下表结果是实际执行，不是拟验证。数组要求值、dtype、shape 和字段顺序精确一致；无新容差。
原有 roundtrip 门槛仍为 2e-4 px，原 loader 对索引 affine 的门槛仍为 1e-7；不改已有报告字段。

| 验收 | 结果与证据 |
| --- | --- |
| 原 198/79 打包及 400/143 衍生完整回放 | `after_final_comparison.json`：19 JSON、4445 数组、1088 图片一致；包括 37 来源边界、纯框/契约检查和 RNG |
| 父样本顺序与随机 | 400 个 parent_row 逐项等于原 completed_updates.jsonl；419 次 default_rng 方法、参数、结果、bit-generator before/after 状态及全局 Python/NumPy/Torch RNG 一致 |
| 拒绝与成功回退 | `edges_comparison.json`：25 JSON、4460 数组、1088 图片一致；倒置/过小框、支持量不足触发原框回退；原框也失败时停止；缺失 center/visible、错误 policy、空行、空监督比较旧结果 |
| 非有限框的旧失败 | `nan_edges_comparison.json`：错误文本和已有部分输出一致；NaN 拒绝记录最终被严格 JSON 拒绝，未修复 |
| 标注适配 | `adapters_comparison.json`：4 JSON、31 数组、19 图片、1 manifest 文本一致；旧常数权重、新 DS 显式 mask、黑 RGB、遮挡 GT、缺失 mask、空监督拒绝 |
| 543 行最终 loader | `loader_readback.json`：400 train/143 val 通过真实 Dataset 分支；输入、affine、裁剪 GT、监督与生成时保存值精确；训练监督 42878/其中遮挡 20722，验证监督 15798/遮挡 8065 |
| P2.1/P2.2 回归 | `dataset_regression/result.json`：90 次、2814 数组精确，含七个真实样本增强开/关及完整输入和 RNG |
| 保存预测回放 | `candidate_replay/result.json`：P1 独立 baseline 的 37 JSON、438 数组、78 图片一致，未运行模型 |
| CPU 测试 | `cpu_tests.json`：87 项通过、0 失败、0 跳过；含新 7 项便携测试及原混合 batch、遮挡梯度、完整裁剪边界测试；不含 Blender 集成 |
| CLI | `cli_checks.json`：8 个入口从 `/tmp`、无 PYTHONPATH 运行 --help 成功；训练/生成命令只测导入与参数解析 |
| 结构证据 | `structure.json`：来源校验拆阶段并重命名后计算 AST 一致；change_box/official_bbox/loader factory AST 一致；contract/objective 原样 |

正常及拒绝回放仅允许本次输出路径前缀、config/index 的 `frozen_utc`、acceptance 的 `seconds` 变化。
JSON key 和列表顺序均比较；图片比较解码像素，不依赖 NPZ 压缩容器时间戳。
候选回放沿用 P1 的时间/输出路径白名单，无指标、分母或来源字段豁免。

来源边界覆盖共享 face/shape/family/纹理/ancestry、动作组未隔离、同图/同场景、无目标框遮挡者的身份/姿态泄漏、
paired box/affine/几何签名、缺字段及未冻结索引。保持原 signature 的 camera 作为 key 的设计，不偷偷增加新 gate。

便携 [source_packing_v1.json](../../tests/fixtures/source_packing_v1.json) 保存独立旧结果；
[test_source_packing_contract.py](../../tests/test_source_packing_contract.py) 不依赖真实资产或 outputs，
兼容导出测试与独立 before 对照分开。

### 与历史 data_v2 的区别

`historical_comparison.json` 确认四个 198/79、400/143 loader 数组保持，400 次曝光顺序保持。
历史 `data_v2/*_index.json` 没有 `parent_neighbor_crop_pixels`，其 `neighbor_crop_pixels` 仍是父框值；
**本批开始前的代码**已经保存父值并重算实际衍生裁剪值。训练 99 行、验证 74 行的数值不同，
均精确匹配 status.json 指向的 `actual_crop_context_v2.json` 修订记录。
除这一既存元数据差异、时间和输出路径外，其余历史索引内容一致。
本批没有修改原始或修订索引；本批独立 before→after 的这些字段完全相等。

## 5. 复核与后续

从仓库根目录运行；所有输出目录必须是新目录，禁止覆盖 before：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/densekp-refactor-mpl PYTHONPATH=src:. /home/yy/conda_envs/camerahmr/bin/python -B tools/validate/verify_source_packing.py --before outputs/acceptance/refactor_sources_boxes_20261002/before/data --output /tmp/densekp-p2-source-review-new
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/densekp-refactor-mpl PYTHONPATH=src:. /home/yy/conda_envs/camerahmr/bin/python -B -m unittest tests.test_source_packing_contract tests.test_occluded_supervision -v
```

同环境：Python 3.10，NumPy 1.26.4、OpenCV 4.13、Torch 2.0.0+cu118、Pillow，沿用前批 CPU 环境。
拒绝/adapter 的完整运行脚本在证据目录 `capture_edges_final.py`、`capture_nan_edges.py`、`adapter_acceptance.py`；
对应 before 使用保存的旧源码，after 使用活动模块。主核查工具拒绝从已迁移函数捕获新 before。

行为修复仍单列：portrait/closeup 旋转、mask floor/rint 差别、历史报告字段，
以及本次证实的 NaN 框拒绝记录无法写入严格 JSON。没有统一这些规则。
未删除历史入口或资产，旧输出和失败版本均保留。

无阻碍本批验收的真实问题。P2 整体待评审；通过后进入 P3 几何/相机/生成链，
须在相关改动前单独冻结几何与渲染基线，不能用本批 CPU 结果替代 Blender 或真实 GPU 验收。
回退以本批 source_snapshot 的文件为边界，保留 P1/P2.1/P2.2、用户并行编辑和全部证据，不 reset/clean。
