# A1：常驻同步 CPU 消费者

日期：2026-10-09。`integration_only`。本批只消除逐帧 CPU 进程启动，未改 ROI、采样、场景、标签、监督或 Dataset。完成同缓冲新旧对照及同配方 6 时刻实测：**一个 Blender 进程、一次场景加载、一个常驻帧消费者；12 次 ROI 渲染，全部成功，无渲染重试。**

产物：[验收 JSON](../../outputs/experiments/persistent_roi_consumer_a1_20261009/acceptance.json)、[逐帧时间](../../outputs/experiments/persistent_roi_consumer_a1_20261009/frame_timings.csv)、[同缓冲对照](../../outputs/experiments/persistent_roi_consumer_a1_20261009/same_buffer_attempt_2/comparison.json)、[双路 ROI 视频](../../outputs/experiments/persistent_roi_consumer_a1_20261009/short_run/pack/dual_roi.mp4)。旧参考：[ROI-only 验收](ROI_ONLY_RENDER_A1_20261009.md)。

## 实现和命令

新增 `src/densekp_retrain/runtime/frame_consumer.py`：标准输入/输出的逐行 JSON 请求与应答，无线程、队列、异步预取或并行消费。CPU 子进程初始化及导入一次，发 ready 后等待任务；Blender 完成一帧，发送 frame，阻塞至成功应答才进入下一帧。消费仍直接调用原 `consume_roi_frame`，没有第二套算法或缓存策略。每帧日志独立，协议 stdout 与诊断输出分开。

失败返回 frame、错误及 traceback，写 `consumer_failure.json` 后退出；父进程停止帧循环。超时、意外退出和协议错误也中止，不自动重试。失败现场不清理。正常结束发送 stop，核对已完成帧列表、等待 exit=0、关闭管道。退出异常时只回收自己启动的消费者，不操作其他进程。

现有入口增加显式选项，默认仍是旧 `per_frame`、11时刻、`full_image`，不改变历史默认模式：

```bash
/home/yuyue/anaconda3/envs/camerahmr/bin/python tools/experiments/run_continuous_roi.py \
  --storage-mode roi_only --consumer-mode persistent --frame-count 6 \
  --gpu-uuid GPU-6f046d56-acbe-d49b-774a-9d463e768cac \
  --output outputs/experiments/<new_directory>
```

本批 persistent 入口限定 ROI-only；可选已有配方的2–11时刻前缀。打包和有界抽查在帧消费者正常退出后，由原有收尾阶段的独立 CPU 进程完成；“一个消费者”指整段逐帧消费，不把收尾辅助进程隐去。新增 `validate_persistent_roi.py` 提供静态缓冲对照及短段收尾，后者针对当前冻结的单人双机配方，不代表多人接口已验收。

修改的既有文件仅 `tools/render/render_continuous_roi.py`、`tools/experiments/run_continuous_roi.py`、`tools/experiments/consume_continuous_roi.py`。原消费算法、ROI规划、渲染装配、Dataset、打包/旁表算法及全部资产未改；新增协议模块、定向测试和独立验收入口。

## 同缓冲及故障检查

只读使用上一轮保留的 `failure_attempt_1/retained_source/source` 首帧双路真实缓冲，复制到本任务各自拥有的临时目录，分别运行旧逐帧进程及新常驻进程。没有重渲染旧路线，没有从最终图反造缓冲。

两相机共14个成品文件检查通过：RGB/valid/person PNG字节一致，annotations NPZ所有数组一致，boxes、raster_storage和render_coverage仅规范化新旧输出根路径后完全一致，世界138点一致。经真实逐行打包器及 Dataset，各读双路2行，共4次实际回读：img_disp、归一化img、138点、监督/可见性、_trans、img_size、center/scale和其余返回字段逐值一致，差异为0。时间统计、所有权token和输出根路径不要求相等。

成功对照中，旧单帧进程 wall 3.779秒，新常驻进程完成这**一个**请求并关闭的总wall 3.729秒（启动2.182秒、请求0.831秒、关闭0.717秒）。单请求仍支付启动/退出成本；收益来自后续请求复用进程，不能宣称单帧启动也省掉这些时间。

第一次对照的消费者和成品比较已成功，但验收脚本把单静态帧送入“至少两帧”的序列校验器，报 `invalid_sequence_timeline`。已改为调用现有 `pack_derivative_rows`，不伪造连续帧；一次修正后完整对照成功。初次失败目录和日志保留，成功对照的自有临时缓冲在检查后清理。初次失败对照进程总wall未单独计时，不能从成功对照的7.831秒函数时间推算它；它属于 CPU 验收开销，不混入以下渲染吞吐。

`tests/test_frame_consumer.py` 的5项 synthetic_test_only CPU测试通过（0.307秒）：同一真实子进程连续应答及正常关闭、重复任务拒绝、消费异常后保留部分文件并停止后续帧、未正常关闭的EOF、超时/提前退出回收。没有全库discover、模型构造、推理、权重加载或参数更新。

## 短段实测

同一冻结 road_pass_00，frame_index 0–5，10Hz，首末跨度0.5秒；源帧12/15/18/21/24/27，名义30Hz、stride3，KIT/425共享动作来源，不是独立验证。身体/衣物/groom和世界变换沿用原实现。完整ERP采样栅格5760×2880、ROI64samples、CUDA和原色彩/光照设置保持；无背景渲染、无全ERP RGB生产读写。全部6时刻的 ROI 规划及相机矩阵与旧记录逐值相同。

Blender PID **1174824**；CPU消费者 PID **1174858**，6次应答均为此PID，正常退出。对象身份/persistent data检查通过。GPU为RTX3090/GPU1，UUID **GPU-6f046d56-acbe-d49b-774a-9d463e768cac**，与上一批ROI-only为同一卡。运行前后均记录GPU/进程/内存/磁盘；只有原有空闲桌面Blender上下文，没有其他活跃同卡计算任务，未终止或争用其他人的任务。

| 计时项 | 实测 |
| --- | ---: |
| 一次性场景/资产装配 | 17.271 s |
| 一次性消费者启动 / 关闭 | 2.352 / 0.717 s |
| 首帧循环，含消费、不含装配 | 20.623 s |
| 后续5帧均值 / 中位数 | **14.714 / 14.725 s** |
| 后续5帧最小 / 最大 | 13.870 / 15.245 s |
| 后续5帧几何更新均值 | 0.048 s |
| 后续5帧两路ROI调用均值 | 13.181 s |
| 后续5帧ROI规划等额外render回调开销 | 约0.760 s |
| 后续5帧同步消费回调均值 | **0.725 s** |
| 全6帧消费者请求往返均值 / 函数体均值 | 0.745 / 0.741 s |
| Blender进程wall，含同步消费者启动/工作/关闭 | **116.432 s** |
| 打包 / 6次Dataset抽查 / 视频 | 0.182 / 0.174 / 0.569 s |
| 独立收尾进程wall，含导入 | 3.978 s |
| **短段完整生成交付wall** | **120.415 s** |

逐帧循环：20.623、15.215、15.245、14.725、14.516、13.870秒。相机调用仍包含渲染、同步、compositor及写出，未伪造GPU kernel时间，也未继续细查第一路ROI成本。采样GPU信息的1秒监控精度包含在总wall中。

上一批后10帧消费进程约3.987秒，当前后5帧消费回调0.725秒，重复启动开销已转为段级启动/关闭。整帧历史参考18.391→14.714秒，约**1.25×吞吐、时间少20.0%**。两批长度/主机负载不同，且上一批frame1是故障恢复后的冷帧；这个比值不是控制所有因素的配对实验，也不能把全部差值归于本次改动。只测了单人6时刻，不能承诺多人/骑行也保持14.71秒。没有为严格配对再渲染旧逐帧路线。

## 产物、资源及结束点

`short_run/`交付12张ROI、局部valid/person masks、138点/框/coverage、配方、时间/来源旁表、12行NPZ、视频及6张代表crop。全部12行接受，无缺失或拒绝。监督点1466个，其中653个遮挡点仍参与；逐时刻双路共同GT为98、100、106、111、111、112。短段首中末（0/3/5）双路共6次Dataset抽查通过；加静态新旧对照4次，**本批实际Dataset回读共10次**。

视频6帧PTS0..0.5秒，末帧正常保持0.1秒；灰底仅为ROI展示占位，无复制帧或拉长。成功渲染的自有EXR及其它临时缓冲已按所有权清理，最终旁表不依赖它们；初次CPU对照失败现场和上一轮失败存档均保留。

ROI PNG总计17,726,039 B；短段目录测量约19,352,974 B（含日志/索引/视频）。临时源1秒采样峰值143,183,278 B；Blender RSS峰值9,434,532 KiB，消费者峰值805,696 KiB；GPU整卡显存采样峰值4068 MiB（含桌面占用）。CPU内存未随着6帧持续增长，不能据此认证长序列无泄漏。短段计算120.415秒，未触及30分钟上限；CPU开发/同缓冲验收另列，没有借预算扩大生成。

`before_code/`保留修改前三个入口；`preservation.json`确认源资产及所有非本批既有文件的size/mtime未变化。本批不改TODO、A2资产/实现、旧报告/输出或外部项目，没有SHA256、commit、下载或安装。

本项完成后暂停性能优化。骑行资产接入和多人场景应另批沿用此同步消费者；本批不自动接入资产、不扩场景、不训练，也不把单人吞吐和接口一致性当作动作质量或模型效果验收。
