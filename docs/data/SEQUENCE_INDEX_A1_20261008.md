# A1：最小序列索引与回读校验

日期：2026-10-08。状态：接口实现及有界 CPU 验收通过，停止于评审点。

**本轮通过的是序列数据接口，不是连续动作质量、曝光同步、动态渲染或训练效果验收。** A2 的 APD 核查独立进行，本轮未等待、选择、下载或制作 motion，未把其候选升级为已验收资产。

## 1. 范围与已有结构

已核查 A1 高分辨率框报告、总计划、双视角审计、A2 连续骑行方案及接口草案。接口草案未整体实施。相关路径未发现适用 AGENTS.md。保留用户 TODO、共享文档和上一轮 A1/A2 文件；本轮只新增下表三份代码/测试及本报告、独立 outputs。

| 新文件 | 职责 |
| --- | --- |
| `src/densekp_retrain/data/sequence_index.py` | JSON 旁表契约、校验及 packed row 上下文查询 |
| `tools/validate/validate_sequence_index.py` | 独立校验 CLI、6 样本 A1 静态复用验收 |
| `tests/test_sequence_index.py` | 人工表格 fixture、文件/尺寸/NPZ 校验及历史 Dataset 回读测试 |

复用已有 `people.json`、`rig.json`、`report.json`、`source_recipe.json`、逐相机 `boxes.json/annotations.npz` 及 A1 的 `index.json/packed.npz/geometry_*.npz`。原 `pack_index_rows` 仅固定保存 imgname、center、scale、proj_verts、in_view、visible、person_id、camera_id；本轮保持其字段与行序不变。未修改 Dataset 返回值、NPZ、打包器、监督规则、框算法或 A1 验收目录。

旁表不重复保存 RGB、mask、138 点、世界几何、pose 或变换矩阵。当前 v1 的边界是**一个 episode 中一个逻辑 sequence，对应一个完整 packed archive**，可含多个相机、人物、时刻和衍生框。混合多个 sequence 的训练大包、跨包合并/重排行映射不在本批实现；不能直接把某个 sequence 的部分行旁表冒充整个混合 NPZ 的完整映射。

## 2. 两层字段定义

两个 JSON 使用 `schema="densekp_sequence_index_v1"`，必须具有相同的 episode_id、sequence_id。相对路径统一相对于 **sequence_index.json 所在目录**，不按 NPZ 目录或文件名猜测。实际验收产物使用绝对的旧输入路径，可明确看到长期依赖。

### episode.json：序列级元数据

| 字段 | 含义/规则 |
| --- | --- |
| episode_id / sequence_id | 非空标识；每条框通过所属旁表继承，不逐框重复 |
| data_kind | synthetic / real / synthetic_test_only |
| temporal_kind | static / sequence；static 必须只有一条帧记录 |
| cameras | camera_id、image_size_wh、rig_ref、calibration_ref；标定未知允许 null，但须 calibration_unknown_reason |
| source_refs | 非空来源文件列表；路径属于长期依赖 |
| people | person_id、source_ref、character_family_id、asset_id、identity_status；家族未知为 null＋family_unknown_reason |
| assets | asset_id、asset_ref、motion_source_ref；动作片段来源未知为 null＋motion_source_unknown_reason |
| time_base | 约分后的正整数 `[numerator, denominator]`，单位为秒/tick |
| time_semantics / time_origin | static_anchor、synthetic_clock、nominal_index、measured_pts；明确零点，不从文件名推断时间 |
| timeline | first_frame_index、frame_step、frame_count、expected_tick_step；定义期望采样日程，最后一项可为 null 表示非均匀采样 |
| synchronization | status、basis，以及 evidence_ref 或 reason，见下文 |
| seed / recipe_ref / generator_version | 可选来源记录；本批不赋予 split 独立性或生成可复现性的认证含义 |
| ephemeral_roots | 可选临时目录黑名单，补充默认 compact 临时路径规则 |

`person_id` 是序列内的目标记录键。合成数据使用 `identity_status="synthetic_instance"`；真实数据使用 verified / track_only / unknown。verified 必须给 identity_evidence_ref，后两种须 identity_reason。可另记录观测的 track_id，**不能据 track_id 自动认定真实同人**。校验器只核对声明及证据文件存在，不进行身份审核。

来源一致性由 people/assets 注册表及观测引用检查。没有家族或 motion 来源证据时，明确记录 unknown；接口通过不等于来源隔离通过。seed 仅为来源信息，换 seed 不构成训练/验证独立性，v1 不替代现有 source_split 或未来按视频/动作家族的划分审计。

### sequence_index.json：帧、观测、框三个小表

| 表 | 必需字段/关系 |
| --- | --- |
| frames | frame_index、整数 time_tick、status=available/missing；missing 必须给 reason |
| observations | observation_id、frame_index、time_tick、person_id、camera_id、asset_id、asset_frame_index/status、source_ref、character_family_id、status |
| 观测文件引用 | image、annotations、annotation_person_index、valid_mask、person_mask、raster_manifest、image_size_wh；缺失状态的 null 规则见下文 |
| boxes | box_id、observation_id、packed_row_id、bbox_xyxy、bbox_kind、bbox_source、crop_policy、parent_box_id、origin_ref、origin_row |
| packed | 既有 NPZ 的 path、row_count；零行表可令 path=null，不能以它构造空 Dataset |

观测 annotations 指向保留的原始逐相机标注，annotation_person_index 对应其中 person_ids 的行。当前有界文件校验适配项目已有 DS `person_ids/xy` 标注，不是通用外部标签格式适配器。raster_manifest 采用已有 `final_erp_resolution` 字段。

有动作资产时，asset_frame_index 是非负整数且 asset_frame_status=known。真实观测没有动作帧时可为 null；必须同时使用 unknown 或 not_applicable，并给 asset_frame_reason。unknown 表示应该有映射但没有证据，not_applicable 表示该真实观测没有合成动作帧；两者均不能替换成 0。

box 的 origin_ref/origin_row 指向已有 `boxes.json` 列表或包含 rows 的旧 index。文件检查核对该行的框、person_id，以及存在时的 camera、bbox_kind。父框可以只作来源引用，packed_row_id=null；每个实际训练/验收框必须有唯一非负 packed_row_id。

衍生框使用 `bbox_kind="synthetic_perturbation"`，必须引用本表中的父框。父子必须属于同一 observation，因此保持同一 sequence、时刻、人物、相机和资产帧；检查缺父、自引用和环。多个衍生框可以共享父框，不能把它们误认为多个时刻。

## 3. 校验规则及缺失语义

### 时间与完整性

- `exact_time(time_tick, time_base)` 使用 Fraction 精确计算有理秒；拒绝 float tick、bool tick、非法或未约分的 time_base。不做浮点近似配对。
- frames 按期望的 first_frame_index＋i×frame_step 顺序保存，数量与 frame_count 一致；重复帧、顺序错误、未经声明的缺帧均拒绝。
- time_tick 严格递增；若配置 expected_tick_step，则相邻 tick 差也须一致。采样 step 描述索引的期望采样日程，不等同于原始视频没有丢帧的证据。
- 已知缺帧应保留 missing 帧记录、整数 tick 和原因，该帧全部相机/人物观测标为 camera_missing。不得删除该条后压缩时间轴，也不复制前帧补齐。
- 每个期望 frame×camera×person 槽位恰好一条观测。不能通过少写相机或人物记录掩盖缺失；同一相机/帧的 image、annotations、valid_mask 和缺相机状态须一致。
- 每条观测的 tick 必须等于其所属帧；同一人物的来源/家族/asset 引用必须等于序列注册表。合成双相机同帧必须引用同一 asset_frame_index，允许不同人物使用不同动作帧。

### 观测状态

| status | 文件与框规则 |
| --- | --- |
| observed | 有有效观测及至少一个框；图像、原标注、valid/person mask、尺寸 manifest 和标注人物行必须给出 |
| camera_missing | 该相机帧不可用；全部观测文件、尺寸和原标注行置 null，reason 必填，不能有框 |
| person_absent | 相机图像存在，该人物未出现；保留图像/相机标注/valid mask/尺寸，person_mask 和 annotation_person_index 为 null，不能有框 |
| empty | 相机和人物原标注/实例 mask 可追溯，但没有可用框；保留文件和标注行，reason 必填，不能有框 |

状态用于记录已知输入事实，本校验器不通过 RGB 黑色或模型预测推断状态。空观测不代表 138 点全部为零，也不改写可靠遮挡 GT 的监督资格。

### packed 与长期依赖

每个 packed row 唯一关联 box→observation→frame→episode。packed_row_id 必须恰好覆盖 `0..row_count-1`，不能重复或跳行。支持父框不打包，但 observed 观测必须有对应框。

校验分级，不批量解码高分辨率 RGB：

1. 表格与文件检查（默认）：结构、精确时间、引用、缺失状态、路径存在、manifest 声明尺寸、原始框来源行。
2. `check_image_headers=True`：只读每个不同 image/valid/person mask 的文件头，核对尺寸，不读全部像素。
3. `check_packed=True`：只读 NPZ 的 imgname/person_id/camera_id/center/scale 和原 annotations 的 person_ids 小列；核对原标注人物行、packed 图像/身份/裁剪策略。这里不载入 proj_verts 或 Dataset。
4. 实际 Dataset 回读是单独的有界验收，本轮严格 6 个 item。

当前裁剪元数据检查复用 A1 的 ERP 非周期策略、2 源像素最小边门槛和 deployment=1.0 / historical_trial=1.2；不改变原实验或新增框增强。

长期路径必须存在。默认拒绝 `source`、`_temporary_ds_source`、`_temporary_*`、`<ephemeral-source-dir>` 路径片段及 EXR；自定义临时根可经 ephemeral_roots 声明，且检查解析后的 symlink/相对路径。该规则针对本项目 compact 长期消费者，具有保守限制，不用于认证任意外部资产目录布局。历史 source_recipe 内记录的重算路径和 ephemeral_filename 是来源文本，不被递归当作活跃依赖。没有因此访问或要求恢复已清理 EXR。

### 同步与身份边界

`synchronization.status` 区分：

- synthetic_same_state：合成共用场景状态，必须有来源引用，并逐人物核对双相机动作帧。
- unverified：时间/帧索引可对齐，但曝光未确认，必须说明依据和缺口。
- verified_exposure：仅允许 real＋measured_pts＋basis=exposure_measurement＋证据文件。校验器仍只核对声明完整性，不阅读物理测量并作同步认证。

返回值使用 `declared_synchronization`，**始终保留 `exposure_sync_validated=false`、`identity_confirmed_by_validator=false`**。因此真实 dataset3 的名义帧号对齐和 tracker 记录只能保留 unverified/track_only；不会因为接口验证 status=passed 而自动成为“曝光同步通过”或“真实身份通过”。

## 4. 消费者用法

旁表与原 Dataset 分开读取，不给 Dataset 增加参数或返回字段：

```python
import json
from pathlib import Path
from densekp_retrain.data.sequence_index import validate_sequence_index, row_context
from densekp_retrain.data.loading import load_box_dataset

root = Path("outputs/acceptance/sequence_index_a1_20261008/actual_run_1")
episode = json.loads((root / "episode.json").read_text())
index = json.loads((root / "sequence_index.json").read_text())
validate_sequence_index(episode, index, base_dir=root, check_packed=True)

# dataset[i] 仍是原接口；时间、身份可信度和来源在旁表上下文中。
i = 0
context = row_context(episode, index, i)
dataset = load_box_dataset(index["packed"]["path"])
# 本批已完成 6 个 item 的预算；这里只展示后续使用形式，不再执行。
# sample = dataset[i]
```

context 包含 episode/sequence ID、time_base/time_semantics、synchronization、person 注册记录、observation 和 box。返回身份/同步状态用于防止调用者只拿 person_id 或 frame_index 就误当成已确认的配对。

独立 CLI（只查表格、文件和小型 packed 元数据，不解码图像）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
python tools/validate/validate_sequence_index.py \
  --episode outputs/acceptance/sequence_index_a1_20261008/actual_run_1/episode.json \
  --index outputs/acceptance/sequence_index_a1_20261008/actual_run_1/sequence_index.json \
  --check-packed --output outputs/acceptance/sequence_index_a1_20261008/<new_check>
```

输出目录必须不存在。成功写 validation.json；拒绝写 status=rejected、稳定 code、detail，退出码 1。函数采用 fail-fast，不声称一次列出所有问题。`--a1-acceptance` 会执行 6 样本实际回读，本轮已执行一次，后续不得无预算重复运行。

## 5. 定向测试与人工 fixture

命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/densekp-sequence-mpl \
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_sequence_index
```

首轮及最终均为 **10/10 通过**；最终增加上下文中身份/同步状态和非法 row 查询检查。只运行此模块，两次合计 20 次 test case 执行，没有全库 discover、模型构造或小网络参数更新。

测试覆盖：正常双视角三时刻、整数有理时间（含超过 float64 精确整数范围的 tick）、显式缺帧、未声明缺帧、相机缺失、人物未出现、空观测、时间冲突、重复观测/packed row、来源/家族/asset 冲突、双相机动作帧冲突、多个父框衍生版本、父子跨观测/成环、临时源、缺长期文件、mask 文件头尺寸错误、原标注行身份错误、原框来源错误、packed crop_policy 不匹配，以及无序列旁表的历史 Dataset 正常读取。

历史兼容测试用小型 fixture 构造旧 NPZ，直接调用原 Dataset；旁表验证前后的每个返回字段逐值相同，未新增 sequence_id，138 个有可靠 GT 的遮挡点继续监督。这个人工回读不属于本轮实际高分辨率 6 样本预算。

另保存 [synthetic_test_only/results.json](../../outputs/acceptance/sequence_index_a1_20261008/synthetic_test_only/results.json) 及 13 组可审阅表格：6 组合法状态、7 组预期拒绝。所有 fixture 都显式带 `synthetic_test_only=true`。其中测试真实观测语义的 fixture 使用 data_kind=real 来触发规则，但仍为人工数据，绝非 dataset3 实际观测或 motion 产物。保存的表格 fixture 使用占位文件引用，只用于 `check_files=False` 的表格测试；实际文件/NPZ测试在独立临时目录创建 80×60 小图，不修改原 fixture。

CLI 另实际检查一次现有静态旁表元数据并通过；对人工破坏 tick 的旁表返回 `observation_time_conflict`、退出码 1，保存了 [拒绝结果](../../outputs/acceptance/sequence_index_a1_20261008/cli_rejection_check/validation.json)。没有由此增加 Dataset item 读取。最终还核对 24 行上下文映射、可选 recipe_ref 的临时/缺文件拒绝、脚本语法与报告本地链接；见 `final_metadata_checks.json`，额外 Dataset item 数为 0。

## 6. 6 个实际样本回读

主产物：[actual_run_1/acceptance.json](../../outputs/acceptance/sequence_index_a1_20261008/actual_run_1/acceptance.json)、[episode.json](../../outputs/acceptance/sequence_index_a1_20261008/actual_run_1/episode.json)、[sequence_index.json](../../outputs/acceptance/sequence_index_a1_20261008/actual_run_1/sequence_index.json)。

只读来源：

- `outputs/acceptance/ds_output_modes_20261008/attempt_4/compact/`
- `outputs/acceptance/high_res_boxes_a1_20261008/compact_run_1/consumer/`

元数据完整映射原 A1 的 **24 行 packed NPZ＋6 个未打包父框引用＝30 条框记录**。本轮没有重新生成这 24 个框，也没有新写 NPZ。帧表只有 **frame_index=0、time_tick=0、time_base=[1,1]** 一个静态快照，零点只是本地静态锚点，不代表拍摄时间或 1 fps 连续数据。

原 people.json 的 frame 0/8/16/24 是不同人物的历史几何帧选择，分别保留为 asset_frame_index。它们不是四个场景时刻，不能把这份产物当作连续骑行。asset_id 使用本次本地的 legacy_compact_geometry_reference 来源句柄，引用保留的 people.json；原 motion 片段和角色家族证据没有在 compact 中认证，均明确 unknown，不将其冒充 A2 资产。

8 条人物/相机槽位：observed=6、empty=1、person_absent=1。cam4/person3 保留有域内 GT 但无实例紧框的空观测；cam4/person4 按原 report 的零 in_view 和空框记为未出现。没有静默丢掉这两条记录。

| 实际 packed row | 人物 / 相机 | asset_frame_index | crop_policy |
| --- | --- | ---: | --- |
| 0 | 1 / cam2 | 0 | deployment |
| 1 | 1 / cam2 | 0 | historical_trial |
| 4 | 2 / cam2 | 8 | deployment |
| 8 | 3 / cam2 | 16 | deployment |
| 16 | 1 / cam4 | 0 | deployment |
| 20 | 2 / cam4 | 8 | deployment |

实际读取严格为上述 6 个 Dataset item；row 0/1 验证同父框多版本，cam2/cam4 验证相同时刻同人同动作帧关联。主要结果：

- 所有 24 packed 行的小型身份/图像/center/scale 元数据通过映射检查；路径依赖共 21 个。
- 只读 11 个不同 PNG 的文件头核对 5760×2880；表格检查阶段 image_headers_read=0。未批量解码所有高分辨率图像。
- 6 个样本的 138 点原图坐标与对应逐相机 annotations 的人物行逐值一致，in_view/visible 逐值一致。
- 每个返回 affine 与裁剪标签均与 A1 geometry 逐值一致。共 574 个监督点，其中 361 个遮挡点继续监督，仍为 `in_view & inside_full_crop`。
- 原 diagnostic_loader 可在没有任何序列旁表参数时构造 Dataset（len=6），没有额外读取其 item；历史无旁表的实际数值回读由前述小型 fixture 覆盖。
- 旧输入树大小/mtime_ns 清单前后完全相同，无 SHA256；这是文件清单一致性证据，不声称内容哈希验证。
- 本轮没有写入 datasets、旧输出或 A2 文件，没有重复框增强、裁剪 PNG 导出、视频解码、渲染或动作接入。

## 7. 留给 A2/后续接入侧的字段

A2 后续提供已验收资产时，在本旁表中填入/引用：

1. 权威 asset_id、asset_ref、motion_source_ref，动作片段边界及来源/许可证据所在 manifest；本层引用小型清单，不复制 pose/衣物/groom 大数组。
2. 目标 person_id 的角色家族、source_ref、固定身份/shape 来源；源演员与目标角色不能混为同一身份。
3. 每个场景整数 time_tick 到 asset_frame_index 的明确映射；source frame/PTS、重采样、变速、循环等详细时间映射保留在资产 manifest。本层不推断 fps，不检查动作质量或执行重采样。
4. episode/sequence ID、相机列表、rig/calibration 引用、时间单位和零点；合成双相机同一人同 tick 使用同一资产帧。
5. 逐时刻全部人物/相机槽位与成功/缺失/空观测状态；长期图像、标注、mask、尺寸 manifest 及框来源行。
6. 打包结束后冻结 packed_row_id 与原 NPZ 行顺序，保存父框链和 crop_policy。若重排/合包，需要重新形成对应映射，旧旁表不能继续贴在新 NPZ 上。

本轮不要求或生成检查视频。未来视频入口需要：sequence_index 引用、输出 camera 顺序/布局、每个视频帧对应的 frame_index/time_tick/time_base、显式输出帧率及非均匀时间重采样映射、缺帧处理和错误状态。约定缺帧时拒绝导出或可见地标记缺口，不能静默复制上一帧、跳帧压缩时间或把检查视频替代训练 PNG。待真实连续资产具备后，再实施视频与逐帧渲染联合验收。

## 8. 计算与资源记录

计算前检查 GPU 和进程，日志位于验收根目录：两张 RTX 3090 当时 GPU 利用率均为 0%；GPU1 上已有 Blender 等进程，本轮未启动、停止或干预它们。执行使用 CUDA_VISIBLE_DEVICES=''，OpenBLAS/OMP=1；实际回读 OpenCV=1，仅 CPU。

| 范围 | 实测 |
| --- | --- |
| 首轮 10 项测试 | unittest 0.081 s；进程总墙钟 3.78 s |
| 最终 10 项测试 | unittest 0.095 s；进程总墙钟 2.97 s；峰值 RSS 564332 KiB |
| 6 样本验收主体（导入后） | 墙钟 1.78975 s；CPU 1.77088 s |
| 验收整个进程 | 墙钟 4.62 s；user/system 4.26/0.36 s |
| 验收进程峰值 RSS | 627824 KiB，约 613.11 MiB |
| actual_run_1 最终文件大小 | 48,543 B，无新 NPZ/图像/几何数组 |
| 人工表格 fixture | 13 组，仅元数据；真实观测语义也显式 synthetic_test_only |
| 实际数据范围 | 1 个静态时刻，24 行元数据映射，6 个 Dataset item |

资源来自 `/usr/bin/time -v` 和 `getrusage`，见 `consumer_resources.txt`、`tests_*_resources.txt`。最终文件大小不是临时磁盘峰值；未连续采集磁盘峰值。另有轻量 Git/文件查询、13 组表格检查和两次通用 CLI 检查，不把它们冒充额外真实回读。无全库测试、网络下载、安装、Blender 调用、模型/权重、推理、训练、参数更新、提交或 SHA256。

## 9. 评审边界

通过的是时间/身份声明/路径/行映射的接口一致性，以及 6 个已有静态样本的真实读取。未认证连续动作质量、衣物/groom 动态、曝光同步、真实身份关联、源视频许可、来源独立性、动态渲染或训练效果。

v1 不验证资产内部帧数/动作连续性、root 分解、循环接缝、视频 PTS 实测，也不改写 A2 接入契约。真实数据时间/身份可信度必须继续单独审核。完成后停在本批评审点，不自动进入下一阶段。
