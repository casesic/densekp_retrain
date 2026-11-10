# 混合行走／骑行：保留完全遮挡并续接恢复

执行日期：2026-10-09；报告收尾：2026-10-10。全部 `integration_only/shared_source_diagnostic`。

同一冻结场景已完成31时刻、10Hz、首末3秒。复用frame0–17，重新消费frame18，新增frame19–30；没有重摆人物或重渲染旧帧。数据流程、整段物理双边缘覆盖、原15/15遮挡恢复诊断分别通过；没有模型推理或训练。

- [完整双路视频](../../outputs/experiments/mixed_walk_cycling_edge_continuation_a1_20261009/run/pack/dual_roi.mp4)
- [最终验收](../../outputs/experiments/mixed_walk_cycling_edge_continuation_a1_20261009/acceptance.json)、[成本](../../outputs/experiments/mixed_walk_cycling_edge_continuation_a1_20261009/costs.json)
- [逐时刻曲线](../../outputs/experiments/mixed_walk_cycling_edge_continuation_a1_20261009/run/pack/target_curves.svg)、[CSV](../../outputs/experiments/mixed_walk_cycling_edge_continuation_a1_20261009/run/pack/target_curves.csv)

## 复用、时间与所有权

父运行：`outputs/experiments/mixed_walk_cycling_edge_a1_20261009/run`。父批次的 `partial_stopped`、frame18拒绝、冻结输入、1.9秒前缀视频和失败缓冲均保留；本批结果不改变当时结论。

| 范围 | 本轮处理 | 新相机调用 |
|---|---|---:|
| frame0–17 | 只读复用36张最终ROI及标签、mask、原生框 | 0 |
| frame18 | 复制必要原始缓冲到本任务拥有的临时目录，重新消费 | 0 |
| frame19–30 | 一个新Blender进程、同一场景装配和逐帧更新 | 24 |

一个新常驻同步CPU消费者处理frame18–30，成功正常关闭。新Blender进程一次加载，首个新渲染状态直接为frame19；没有从source0重走。跨父批次与本批次不能称作总共一次启动。

原31条episode/person时间映射、seed、轨迹、rig、world R138及scene矩阵保持一致并核对。person1是M01/APD骑行，person2是M04/KIT425行走；两种root所有权不变。APD整数原时间和KIT发布30Hz时间、各人的源帧及量化误差都保留，10Hz只是合成episode时钟。没有复制帧、插值、循环、重新绑定或改变M04原有外观验收限制。

`parent_recipe.json`、`parent_frozen_execution.json`、`time_matrix_audit.json`和`run/resume_manifest.json`记录输入及续接清单。`run/frame_sources.json`显式区分 `readonly_parent` 与 `this_run`；统一pack只读引用旧成功文件，不复制资产或依赖临时EXR。最终使用 `run/pack/packed.npz` 与同目录episode/sequence_index旁表，旧前缀目录仍是长期依赖，不可清理。

## 新策略与接口

具名策略 `record_unavailable_v1` 仅由显式新配置启用；缺省 `strict_dual_crop_v1` 仍保留旧共同crop门槛。新策略当前支持ROI-only。

- `data/observation_policy.py`：严格验证身份、mask、紧框与ROI支持；仅接受明确列举的无框、像素尺寸过小、crop跨接缝不支持、无crop内GT等观测不可用原因。未知异常、非有限GT、损坏文件、非法元数据、身份/时间/矩阵错误及ROI支持不足仍中止。
- `data/sequence_resume.py`、`frame_sources.py`：检查完整成功前缀、待再消费帧与未渲染后缀，拒绝未分类残留；复制缓冲而非链接父目录，清理只使用本任务所有权目录。
- 公共 `run_continuous_roi.py` / `render_continuous_roi.py` 增加显式续接，复用已有装配、逐帧更新和同步消费者，没有第二套渲染器。
- `roi_consumer.py`、`sequence_packing.py` 分离物理覆盖和输入可用性；`multiperson_acceptance.py`、`validate_persistent_roi.py`支持只读帧引用、完整统计及状态视频。没有修改Dataset、loss、R138或监督算法。

状态沿用现有schema：`observed`为有实际打包行；`empty`为已渲染但没有合格训练行；`person_absent`在本策略指本相机物理域外且无实例框，不代表世界人物被删除；`camera_missing`指未生成相机观测，不用于完全遮挡。域外槽位另保留几何标注索引与mask引用。

共同集合的定义：任一路没有合法crop时 `common_points=null`，同时记录具体缺失相机和 `evaluable_point_count=0`；两路有合法crop但交集为空时 `common_points=0`。`has_dual_input`表示两个合法图像crop，不保证两条训练行；零监督crop仍不进入NPZ。共同点≥8只作整段可用性统计，不再逐帧终止生成。

frame22行人cam2恰好出现“有合法crop、无crop内GT”。首版将其共同集合写成null，已做一次针对性CPU打包统计修正：最终为已定义空集0。首版保留在 `run/pack_attempt_1`；最终全部NPZ数组和序列索引与首版精确一致。额外渲染0、额外Dataset回读0；修正记录见 `support_count_correction.json`。本次没有再修改监督规则或伪造该槽位训练行。

运行命令（已执行；重用时必须提供不存在的新输出目录，不要覆盖本次目录）：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/densekp-roi-mpl PYTHONPATH=src:. \
/home/yuyue/anaconda3/envs/camerahmr/bin/python tools/experiments/run_continuous_roi.py \
  --output outputs/experiments/mixed_walk_cycling_edge_continuation_a1_20261009/run \
  --resume-parent outputs/experiments/mixed_walk_cycling_edge_a1_20261009/run \
  --observation-policy record_unavailable_v1 --storage-mode roi_only \
  --consumer-mode persistent --gpu-uuid GPU-d481acc7-106f-29e5-1298-8c0d398261b2 \
  --budget-seconds 1200
```

## 边缘、完全遮挡与恢复

物理门槛不变：同一目标同一时刻，两路各≥8域内且≥5域外可靠R138，共同物理域内≥8；整段≥16/31且连续≥0.5秒。实际frame0–19连续20时刻满足，20/31=64.52%，首末跨度1.9秒。其余11时刻没有从分母删除。

| 目标时刻 | cam2 | cam4 | 共同physical GT | 共同crop GT |
|---|---|---|---:|---:|
| frame18 / 1.8s | 130域内点全有邻人遮挡证据；框消失 | 有框 | 114 | null |
| frame19 / 1.9s | 框重现 | 124域内点全有邻人遮挡证据；框消失 | 118 | null |
| frame20 / 2.0s | 有框 | 框重现 | 121 | 120 |
| frame30 / 3.0s | 有框 | 有框 | 138 | 138 |

frame18与frame19都仍满足双路物理边缘条件，不能因无双路crop称作目标不在场。目标有双路合法输入且共同监督≥8的时刻共29/31=93.55%，窗口为0–17（跨度1.7秒）与20–30（跨度1.0秒）；定义了共同集合的目标时刻为92–138点。双边缘窗口内有18/20时刻具备≥8共同crop点，整体统计仍以31为分母。

原遮挡指标继续以每时刻共同physical GT为分母，依据前景邻人实例ID与range的保守证据，不靠总可见率下降替代互遮挡。每阶段3个相邻10Hz输出时刻；高窗口最小值分别减去两侧低窗口最大值，门槛均15个百分点。

| 相机 | 轻遮挡窗口 | 加重窗口 | 恢复窗口 | 加重 / 恢复 | 结论 |
|---|---|---|---|---:|---|
| cam2 | 0–2 | 17–19 | 20–22 | 53.04 / 45.61个百分点 | 通过 |
| cam4 | 0–2 | 18–20 | 21–23 | 53.51 / 18.21个百分点 | 通过 |

这里通过的是几何遮挡恢复与原生框重现。**恢复窗口已超出0–19的同时双边缘窗口，不能声称整个恢复阶段也保持双路部分域外。** 本轮没有DenseKP预测，不能声称预测恢复、定位误差改善或跟踪恢复。早晚窗口是同源时序对照，不是固定姿态/位置只改变遮挡的因果对照。

目标框宽度随遮挡和运动变化：cam2为120–339完整ERP像素，cam4为82–309；无框时刻保留null。完整中心、宽高、scale、共同点、邻人比例和边缘/输入布尔曲线见CSV/SVG，不把变化解释为检测误差。

## 数据和验收

最终62张逻辑相机ROI、124个槽位：101 `observed`、3 `empty`、20 `person_absent`，相机missing为0。3个empty分别是两次目标完全遮挡和frame22行人cam2无crop内GT。没有补框、全零训练图或沿用上一帧框。

| 人物/相机 | 训练行 | 监督点累计 | 其中遮挡监督点 |
|---|---:|---:|---:|
| M01 / cam2 | 30 | 3784 | 1767 |
| M01 / cam4 | 30 | 3768 | 1803 |
| M04 / cam2 | 25 | 2731 | 1354 |
| M04 / cam4 | 16 | 1827 | 870 |

所有合法行保持可靠GT的 `in_view & inside_full_crop` 监督，包含遮挡点。logical ERP仍5760×2880，图像只存实际ROI，physical mask与render coverage分开；全景背景和全图RGB生产读写均为0。相机、64 samples CUDA、色彩/灯光和ROI规划未变。

全31时刻核对身份、各人原时间、世界矩阵、有限标注、实例mask互斥、尺寸、框和ROI支持。旧18时刻是只读回查，不能计为新生成。frame18重新消费后的全部标注数组、编码RGB、mask和框与旧失败现场精确一致；父输出文件大小/mtime清单未改变。

实际Dataset共9行：frame0、18、19所有有效槽位；无框/域外槽位明确没有行。图像局部仿射、R138完整ERP仿射、监督集合、offset、`_trans`和`img_size=[2880,5760]`检查通过。frame19是cam2实际重现时刻；cam4在frame20重现由全帧元数据/标签检查确认，没有为此追加Dataset读取。修正打包后以NPZ全部数组和索引精确一致复用这9行验收，不重复解码成新Dataset次数。

首次7项定向CPU检查通过，涵盖具名旧/新策略、完全遮挡/域外/相机missing、null与0、硬错误、共享RGB多框、只读引用，以及既有混合时间/root和打包兼容；针对空监督修正后的5项策略测试再次通过。真实18→19映射、矩阵在 `time_matrix_audit.json` 及实际渲染/旁表中核对。无全库discover、模型构造或参数更新。

视频31帧，PTS为0.0…3.0秒，最后显示时长另记0.1秒，总播放3.1秒；没有复制观察帧。黄色线为physical domain边界，显示人物ID、episode时间和无框状态，灰色排版区域明确未渲染。

## 实际成本

本轮新GPU为RTX3090，UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`；即时资源检查未允许其他compute任务共卡，没有终止他人进程。

| 项目 | 实测 |
|---|---:|
| 一次场景载入/装配 | 32.76s |
| 常驻消费者初始化 / frame18再消费 | 2.20s / 2.40s |
| 首个新时刻frame19 | 34.82s |
| 后续11时刻均值 / 中位数 | 30.68s / 30.34s |
| 后续范围 | 28.17–33.11s |
| 两路ROI每时刻调用均值 | cam2 20.02s；cam4 7.43s |
| ROI平均像素量 | cam2 3.969M；cam4 3.956M |
| 几何更新均值 | 0.138s |
| CPU消费13帧均值 | 2.225s |
| Blender整段含同步消费 | 412.86s |
| 初次收尾进程 | 28.02s |
| 其中打包 / 回读阶段 / 视频 | 8.28s / 5.53s / 11.10s |
| 初次完整生成交付wall | 441.01s |
| 一次CPU统计修正 | 函数12.43s；含启动文件时标口径14.73s |
| 新增计算累计含修正及最终证据核查 | 455.82s，约7.60分钟，低于20分钟 |

回读阶段包含统计/作图等工作，不是纯9次Dataset耗时；相机调用包含场景同步、Cycles、compositor与写出，未声称纯GPU kernel时间。没有新性能优化，也不与旧前缀或单人作严格等负载加速比较。

GPU采样峰值4071MiB；Blender峰值RSS约11.28GiB，消费者约881.38MiB。临时磁盘采样峰值215.27MiB，成功清理本任务约2.528GiB临时缓冲。新run长期文件约114.43MiB，其中26份ROI RGB约106.29MiB，保留首版pack约2.86MiB；只读旧前缀约170.70MiB另列。父frame18约212.90MiB失败缓冲保持原样。

父批次721.032s（约12.02分钟）单列，未重复计入本轮20分钟预算；两批已记录计算合计1176.85s（约19.61分钟）。开发/阅读不计入此预算：从before快照到执行前资源记录约593.95s，未涵盖快照前阅读；中断至次日文档收尾的间隔不算计算。

## 限制与结束点

这一个完整事件证明同配方可保留完全遮挡并继续到重现。仍未验收来源独立泛化、曝光同步、预测恢复、跟踪框、正式遮挡比例或训练效果。整个episode按APD＋KIT425及两个角色的来源关系分组；KIT425历史验证来源限制不变，不能把邻帧/不同人物拆成独立train/val。

下一批仍需单独审定来源划分、采样配额和小试训练输入；没有自动扩episode、接新资产、推理、训练或提交commit。保留TODO、A2文件和全部旧输出，停在本次交付点。
