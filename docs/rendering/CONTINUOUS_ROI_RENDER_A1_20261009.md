# A1：C 路线单进程连续渲染

日期：2026-10-09。`integration_only`。**一次 Blender 启动、一次场景/人物加载完成 11 个连续时刻、44 次相机渲染，产出 22 张最终 ERP 和 22 行可读取数据。完整生成与交付耗时 299.98 秒；后续 10 帧平均 23.19 秒/场景时刻。** 没有重做 A/B/C 对照，没有模型、训练或失败重试。

产物根目录：`outputs/experiments/continuous_roi_render_a1_20261009/`，下称 `OUT`。入口：[双路缩略视频](../../outputs/experiments/continuous_roi_render_a1_20261009/pack/dual_view.mp4)、[逐帧计时 CSV](../../outputs/experiments/continuous_roi_render_a1_20261009/frame_timings.csv)、[完整计时 JSON](../../outputs/experiments/continuous_roi_render_a1_20261009/timings.json)、[6 个实际回读记录](../../outputs/experiments/continuous_roi_render_a1_20261009/pack/readback.json)。这些路径从仓库根使用 `outputs/...`。

## 实际输入与流程

只读复用已冻结的 `pedestrian_seeded_pilot_a1_20261009/layout_attempt_2/road_pass_00`：urban_street、person1、M04 行走角色的原身体/固定绑定衣物/groom、rig、世界矩阵和源 root。未使用 A2 新 BEDLAM/APD 资产，未改动作、shape、衣物或 groom 算法。

输出 frame_index/time_tick=0..10，time_base=[1,10]，首末跨度 1 秒。asset_frame=3,6,…,33；本地源帧=12,15,…,42，源 time_base=[1,30]。这是名义 30 Hz 动作的 stride=3 抽样，不是原生逐帧；原始 100 Hz 映射继续标为 uncertain。没有循环、复制静态姿势或补帧。新 recipe 引用旧 recipe，仅声明本次 11 时刻前缀及 C 成像方式；源帧选择数组仍保留原编号。

仍是同一旧验证动作来源 KIT/425 walking_medium05，只有一个动作来源，不把 11 帧当作独立动作。全部 `shared_source_diagnostic`、`integration_only`，不是独立验证集。本轮没有权重或优化器更新。

现有 `render_road_pass_sources.py` 本来已有帧循环和 persistent data；本次只是把 C 接入该循环：

1. 场景、材质、身体、衣物、参考 groom 和 ERP 相机只创建一次。
2. 每帧复用 body/cloth mesh 更新顶点，groom 保持参考点与拓扑，仅更新 `M @ H[asset_frame]`。body/cloth 矩阵一致，body 经 R138 与 M 重算的世界点和缓存 GT 在既有 2e-6 m 容差内；身体、衣物顶点精确读回。
3. 从当前实际 Blender 对象的 body/cloth 顶点和 groom 参考点＋当前头矩阵计算 ROI。每帧两相机来自同一次几何更新；没有重复 root 或逐帧贴地。
4. 顺序渲染两路背景、两路 ROI，然后同步启动 CPU 消费子进程。消费检查通过后清理该帧自有临时目录，才进入下一帧。没有异步队列、多 GPU 或背景缓存。

Blender PID **876513**；`single_process_setup.json` 和 `continuous_render_result.json` 保存进程、场景加载计数及对象/mesh 指针身份检查。11 帧均复用同一对象集合。CPU 子进程不是新的 Blender 启动。

## 成像与 ROI 契约

高质量采样网格/最终 normal ERP 为 **5760×2880**，ROI **64 samples**；每时刻重新渲染 **1440×720、8 samples** 的全场景背景。CUDA、seed1009、animated_seed=False、persistent data、无降噪/自适应；原 AgX、曝光、灯光、材质、bounce、PNG 配置不变，未切换 OptiX 或渲染引擎。这里的 C 是**成像路线 C**，不经过 DS RGB 往返重采样；与旧“分辨率 C 档”含义不同。

按段保存配置、标定 mask 和每相机配置，不每帧复制冻结文件或建立对照报告。沿用已验收的 full-grid border/crop_to_border、整数 top-left/exclusive ROI 与 bottom-left border 换算（+0.01px 数值余量），不改变 FOV。没有调用 full/tiny 渲染，也不要求 B_geometry_gate、tiny_gate 或任何 A/B 源缓冲。

原生可见实例框随有效域截断可能偏移，因此本次用保守规划：把 body/cloth/全部 active groom 的投影包围框向外扩 2 个栅格像素，设宽高 W/H；任何包含于它的子框，经 deployment1.0 变成 square 后都位于 `[xmin-H/2, ymin-W/2, xmax+H/2, ymax+W/2]`。再留 **32 个最终 ERP 像素**。几何非有限、跨接缝或区域越过完整图像直接拒绝，不周期拼接或静默截边。实际框生成后再核查 crop＋2px 插值支持，不能只依赖预估。

例如 cam4 ROI 从首帧 `[3682,970,4859,2147)` 变化到末帧 `[3871,1100,4722,1951)`；cam2 末帧为 `[3929,1102,4775,1947)`。不是盲用原单帧矩形。首帧 ROI 比旧试验的936/940边长更大，后续人物投影变小，因此旧单时刻成本只能作参考。

标签完全来自本帧实际高分辨率 ROI range/IndexOB；每个 138 点的 5×5 可见性邻域检查通过。外围 range/ID 为 unknown，不用低清深度/ID假充GT，也不借历史完整缓冲。physical valid mask 使用同一 rig 下既定标定交集，与 render coverage 分开保存；不根据黑RGB推断有效域。外围 RGB 是逐时刻低清全场景，保留完整场景光照/遮挡交互。

只打包本路线实际 person mask 的原生紧框，`bbox_kind=visible_instance_tight`、`crop_policy=deployment`、扩张1.0。**新增/修改框必须再次检查高质量区域支持；本次不认证任意框增强。** `sequence_packing` 对 C 数据自动要求 coverage 文件并拒绝超界 crop；历史无 coverage 的路径保持原行为。Dataset 本身不增加返回字段；绕过生产者手改 NPZ 不在支持契约内。

## 吞吐和瓶颈

相机调用时间均为 **render/compositor/EXR/PNG 写出与同步 combined**，不能拆成纯 GPU kernel 时间。

| 分项 | 首帧 | 后续10帧均值 |
| --- | ---: | ---: |
| 几何更新及一致性检查 | 0.979 s | 0.053 s |
| 当前几何 ROI 规划 | 0.805 s | 0.760 s |
| 两路低清背景 | 13.383 s | 9.513 s |
| 两路高质量 ROI | 5.496 s | 4.044 s |
| 同步 CPU 消费子进程 | 8.792 s | 8.814 s |
| 每帧循环总计，含上述及少量元数据 | **29.459 s** | **23.188 s** |

后续10帧：中位数 **22.992 s**，最小 **22.262 s**，最大 **24.622 s**。每帧四次调用及两相机分项见 `frame_timings.csv`，未用静态第二相机时间代替实际连续吞吐。

一次性场景/资产加载及装配 combined **17.335 s**；加首帧循环约 **46.794 s**，不含进程启动/退出少量外围开销。装配分项不能进一步可靠拆开，摊到本次11时刻约1.58 s/时刻，不把旧18.7 s全部宣称为每帧节省。

| 整段计时 | 实测 |
| --- | ---: |
| Blender 进程 wall，**已包含同步 CPU 消费** | **280.743 s** |
| 其中11次 CPU 消费子进程 wall合计 | 96.933 s（不可再次加到上一行） |
| CPU 消费函数体合计 | 94.674 s |
| 最终打包 | 3.502 s |
| 6次 Dataset 回读、affine核查和crop保存 | 3.553 s |
| 视频编码/11帧PTS核对及最终元数据检查 | 9.120 s |
| 独立收尾进程 wall，含导入等 | 19.231 s |
| **完整生成到交付 wall** | **299.980 s，约5分钟** |

Blender wall包括约1秒监测轮询量化以及进程启动/退出；总生成wall不含开发、读文档及开发期间的轻量CPU测试。主计算预算1800 s，实际剩余约1500 s；无失败重试。后续统计/报告只读小型元数据，未增加渲染或 Dataset item。

后续 CPU 消费均值分项：读缓冲0.162 s、标签/全图缓冲0.158 s、合成0.343 s、编码/保存6.276 s、最终文件解码及框检查1.641 s，另有子进程导入与清理等开销。当前瓶颈是动态更新后背景调用的同步/场景更新/渲染，以及最终高分辨率 PNG/mask 保存；没有自动开展其它优化。

按**生产**口径，整段 Blender进程（包括消费）摊到每时刻 **25.522 s**，相对旧 C 单时刻生产参考46.206 s为 **约1.81×参考比值**（约44.8%时间减少）。这不是11次等负载冷启动对照：ROI更保守且随运动缩小，本次还增加逐帧消费检查。首帧同一状态的装配＋循环46.794 s与旧值接近；主要收益出现在实际后续时刻。未把减少诊断回读的节省归因于帧循环，**未用旧53.021 s（含十个诊断crop）计算加速比**。

包括本批打包、6个回读和视频，实际每时刻摊销 **27.271 s**。不外推多人同速；未另渲染11个冷启动版本，也不以ROI面积比宣称加速。

## 数据与精简验收

- **11/11时刻、22/22图像、22原生框、22 packed rows**；缺帧、无框/空观测、仅单侧有效、无共同GT均为0。`pack/rejections.json`为空，所有槽位仍通过序列旁表明确列出。
- 全部标签浮点有限；最终图、valid/person mask尺寸为5760×2880；框来自本帧实际实例支持，22个打包crop均通过coverage检查。逐帧最终PNG实际解码成功后才清理源。
- 22行合计2760个监督点，**1217个遮挡点继续参与监督**，仍为可靠GT `in_view & inside_full_crop`。没有使用visible排除遮挡，没有32px偏移，也没改138点或loss语义。
- 每时刻共同GT为98、100、106、111、111、112、111、113、122、128、130；两侧均非空。22个单视角观测均有有效域外语义点。单人自遮挡不标成全部可见。
- Dataset只读取frame0/5/10双路，对应rows0/1/10/11/20/21，**恰好6次**。图像独立warpAffine与loader显示输入精确一致；同affine变换138点精确一致，监督mask相等。只保存这6张原生crop。
- `pack/episode.json`、`sequence_index.json`、`packed.npz` 一一对应；11个整数时刻、动作帧、人物和相机一致。只声明 `synthetic_same_state`，`exposure_sync_validated=false`，不认证真实dataset3曝光同步。
- 视频是双路1440×360缩略图，实际11帧，PTS0、0.1、…、1.0；首末跨度1秒，末帧正常保持0.1秒、容器约1.1秒。没有拉伸、复制或替代训练PNG。视频供直接查看，不设置新的视觉批准前置条件。

定向 CPU 命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' \
python -m unittest tests.test_continuous_roi tests.test_render_regions
```

**12项通过**，unittest主体0.030 s。覆盖动态ROI移动、不同可见子框square覆盖、跨缝/非有限/越界拒绝、整数border、区域偏移/unknown外围、crop/可见性邻域边界、invalid physical mask和遮挡监督。新增fixture均为小型 `synthetic_test_only`，未改原fixture，未运行全库或参数更新测试。

## 资源、文件与调用

计算前重新检查GPU/进程/内存/磁盘。使用空闲 **RTX3090 GPU0，UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`**。GPU1的Blender PID43491保留且未争用。结束时GPU0为18MiB/0%，主机可用RAM约141GiB，仓库盘可用12TiB。

1秒采样显存峰值 **3679MiB**，不是不可漏峰的硬件峰值。Blender `ru_maxrss`峰值 **9331200KiB（约8.90GiB）**；独立打包/回读/视频进程峰值722364KiB（约705MiB）。逐帧CPU子进程日志的`ru_maxrss`与其父进程高水位重合，本批没有独立连续采样它的RSS，不把该值当成额外CPU内存峰值。

最终22张ERP共 **260917615 B**；compact目录在统计时约 **252.04MiB**（后来新增少量JSON/CSV/日志）。单帧清理前源缓冲最大 **122.49MiB**；11帧累计清理约1.202GB，不是同时占用量。没有连续采样磁盘峰值；当前目录无EXR或临时源残留。仅调用既有所有权marker清理本任务成功帧，历史目录和失败现场未动。

代码：

- `tools/render/render_road_pass_sources.py`：可选frame完成回调及分项计时；无回调时保留旧DS渲染默认行为。
- `tools/render/render_continuous_roi.py`：C回调，单进程/对象身份检查、动态ROI、44调用/预算计数、同步CPU消费。
- `src/densekp_retrain/imaging/roi_planning.py`：纯CPU保守ROI和已验收border坐标；`roi_consumer.py`：独立C缓冲合成、标签、支持检查和compact清理。
- `src/densekp_retrain/data/sequence_packing.py`：C数据打包前检查高质量支持，其余返回与监督保持。
- `tools/experiments/run_continuous_roi.py`、`consume_continuous_roi.py`：本次有界驱动、消费、打包和视频；`tests/test_continuous_roi.py`为4项新增测试。

简明命令（必须新输出目录；**本批已完成，不再执行**）：

```bash
python tools/experiments/run_continuous_roi.py \
  --output outputs/experiments/<new_continuous_roi_run>
```

驱动固定选择本批已冻结单人配方的0..10时刻，自动检查GPU，保存按段配置并拒绝覆盖已有run。通用Blender消费入口为 `render_continuous_roi.py --root <run_root>`，复用公共装配和资产manifest；没有第二套人体生成器。`run_config.json`只引用旧比较中的固定成像设置证据，不依赖A/B图像、range或gate文件。

执行清单在 `render_calls.jsonl`、`job_result.json`、`frame_*/render_frame.json`、`frame_*/consume_timing.json`、`task_wall.json`。`preservation.json`核对旧输入size/mtime、原有源码变更范围及TODO/README未变；没有计算SHA256。没有修改A2成果、外部pano-stitch、旧输出、历史数值fixture或执行prompt，没有提交commit、写入datasets或下载/安装。

## 交付边界

**具备继续同配置单人C路线有界批量生成的工程条件**：单进程连续更新、动态ROI、独立标签、逐帧compact清理、打包和6样本实际回读已连通。不是连续动作质量、无闪烁、真实曝光同步、模型效果或部署验收。

尚未覆盖：本片段以外轨迹/全31时刻的ROI覆盖、多人遮挡和成本、骑行/BEDLAM资产、跨缝多块ROI、任意增强框、其它seed/场景模式、故障后的自动续跑。新范围仍须保持现有明确拒绝和预算约束；本轮不扩展这些实现，也不以看视频作为本次交付的再次批准门槛。

已停在交付点，不自动生成62时刻、不做50步训练、不新增性能参数搜索。
