# A1：ROI-only 连续生成与 Dataset 接入

日期：2026-10-09。`integration_only`。**11时刻、22份ROI和22行打包数据已交付，生产背景渲染/全图RGB读写均为0；6组同缓冲Dataset对照全部精确一致。** 后10帧实测均值18.39秒，较旧C的23.19秒历史参考约快1.26×；ROI PNG体积减少89.15%。发生一次消费JSON类型故障并按授权修复：没有重渲染首帧，但实际共启动2次Blender，不能宣称本批11帧全程仅一次启动。

产物根目录：`outputs/experiments/roi_only_render_a1_20261009/`。入口：[双路ROI视频](../../outputs/experiments/roi_only_render_a1_20261009/pack/dual_roi.mp4)、[同缓冲对照](../../outputs/experiments/roi_only_render_a1_20261009/pack/same_buffer_comparison.json)、[逐帧计时](../../outputs/experiments/roi_only_render_a1_20261009/frame_timings.csv)、[机器可读验收](../../outputs/experiments/roi_only_render_a1_20261009/acceptance.json)。

## 坐标和存储接口

核实了 `make_targets`、`pack_derivative_rows`、Dataset和预测还原链：本流程NPZ的`proj_verts[..., :2]`是原始**完整ERP像素xy**；第三列为声明的监督权重，未预先归一化。Dataset使用实际crop仿射变换坐标，再按256-square归一化。`restore_crop_xy`读取返回的`_trans`，本扩展无需额外加offset。

旧NPZ完全不含新字段时走原full-image路径。新NPZ每行使用以下六个字段，必须全部存在且行数一致；任一字段缺失、版本不支持、尺寸/offset矛盾、矩形越界或实际解码尺寸不匹配均拒绝：

| 字段 | 本轮定义 |
| --- | --- |
| image_layout | `roi_image`；亦支持显式`full_image` |
| image_layout_version | 整数1 |
| full_image_size_wh | 逻辑完整图尺寸，本轮`[5760,2880]` |
| roi_xyxy | 整数top-left/exclusive，完整ERP坐标 |
| offset_xy | 必须等于roi_xyxy前两值 |
| stored_size_wh | 必须等于ROI右下减左上，且等于解码的局部RGB尺寸 |

生产者的`raster_storage.json`另保存physical mask引用、render coverage、局部文件名和外围unknown语义。框从局部person mask紧框加一次offset得到完整ERP框；标注138点、center/scale、返回`img_size=[2880,5760]`和`_trans`均保留完整坐标语义。物理有效域按段保存两张共享全图mask；每帧仅保存局部valid/person mask，不保存重复的全图mask。

核心实现位于`densekp/core/datasets/image_layout.py`，没有反向依赖`densekp_retrain`。Dataset返回字段不变，直接解码局部RGB，不构造完整空画布。先按旧算法计算`T_full`，图像采样的逆矩阵为`inverse(T_full)`减整数offset；标签仍用`T_full`。没有在生成仿射前减crop中心，没有双重offset、半像素或32px补偿。

## 支持与拒绝

实际增强plan确定后，用完整crop四角检查旋转/缩放/平移后的取样支持；四角包围范围还需在ROI内保留一个源像素的保守双线性余量。边界比较仅使用1e-9像素处理double逆矩阵末位误差，未放松成像对齐阈值。超界立即抛错，不补黑/replicate冒充观察、不钳框或重抽随机计划。

支持：原生deployment1.0框；有完整支持的平移、缩放、旋转、已有极端裁剪plan和逐通道color_scale。实际生成不增加随机增强。当前ROI路径明确拒绝skimage antialias预滤波、启用的Albumentations和portrait/closeup自动旋转组合；尚未验证的138点flip继续拒绝。旧全图路径的预处理行为不变。

`pack_derivative_rows`保留布局字段并拒绝不一致行字段，避免第一行缺字段导致后续ROI元数据被丢弃；`pack_index_rows`亦保留显式布局，混合旧隐式与新显式布局须先明确full-image元数据。序列校验分别检查逻辑全图、局部文件头、共享physical mask、coverage及packed行布局。没有把局部ROI定义为另一台相机。

如果人为删除所有新字段，裸NPZ无法与旧历史格式区分；本项目不根据文件名猜测。新ROI生产/旁表要求明确布局，正常接口中不会允许缺失；绕过生产校验并把文件伪装成旧格式不在支持契约内。

## 生成入口

```bash
python tools/experiments/run_continuous_roi.py \
  --storage-mode roi_only \
  --output outputs/experiments/<new_roi_only_run>
```

可显式传`--gpu-uuid`选择经检查空闲的设备。默认`--storage-mode full_image`保持上一轮C全画布行为，旧DS入口默认不变。复用原单进程帧循环、对象和persistent data；同一frame只渲染两相机ROI，既有规划算法/64 samples/完整采样网格/色彩/照明不变，不渲染低清背景。没有删除画外场景、缩小FOV或另造人体生成器。

标签仍用本帧实际ROI range/IndexOB。为避免改写投影/遮挡算法，内部仍临时回填全尺寸**数值**depth/ID缓冲和读取共享physical mask；生产不构造全尺寸RGB。这部分CPU/内存成本会如实记录。可靠遮挡点继续按`in_view & inside_full_crop`监督。

实际资产仍为同一冻结单人road_pass_00，KIT/425 walking_medium05，输出0..10、time_base=[1,10]，源帧12..42、stride3、名义源time_base=[1,30]。原始100Hz映射保持uncertain，源root/世界矩阵/衣物绑定/groom不改。单一共享动作来源不是独立验证；不接入A2的新BEDLAM/APD资产。

## 验收设计与当前结果

最终定向覆盖20项：10项ROI/打包/旁表/序列化用例，加直接相关的10项既有序列用例。修复后ROI模块10项全部通过（0.302秒）；已有序列10项在接口修改后通过。无全库discover、网络或参数更新。开发迭代及一次新增打包缓存检查的日志全部保留。初次边界用例暴露double逆矩阵在整数边界的末位误差，已用上述1e-9px数值容差修正；RGB等值断言未放宽。日志保留在独立输出目录。

首中末双路6个新缓冲已与旧全图路径逐值比较`img_disp/img/138点/监督/_trans/img_size/box_center/box_size/_scale`。完整RGB画布仅在这6个验收参考中生成，由自有临时目录管理；对照完成前保留对应源缓冲，失败保留现场。生产Dataset不会走该参考分支。

旧全图一个实际样本在修改前读回并保存独立before，修改后对同一行再次读回，所有返回字段精确一致；未重打包旧数据。视频仅展示ROI，附camera/tick/source时间，空白底色为展示占位，不是场景背景。

## 实测结果

本次完全移除低清背景调用，累计22/22次相机render均为64samples ROI；11个时刻的22个ROI矩形与上一轮C记录**逐值相同**，没有为提速缩小区域。没有生产完整ERP RGB；只有6张被明确标记的临时全图验收参考，均已按所有权清理。新的172个旁表依赖引用已验证，无临时EXR或旧C图像依赖。

每帧源时间、人物、双相机动作帧和世界变换检查通过，所有标注浮点有限，局部图像/mask尺寸一致，原生框来自实际实例支持并正确加offset。22个框全部具有高质量crop支持，无缺帧/空观测/单侧有效/拒绝行。共2760个监督点，其中**1217个遮挡点仍监督**；11时刻共同GT为98、100、106、111、111、112、111、113、122、128、130。22个单视角均有有效域外语义点。

6组对照的`img_disp`、归一化`img`、原始/裁剪138点、visibility/监督集合、`_trans`、`img_size`、`_scale`、`box_center/box_size`差异数量和最大幅度全部为**0**，未用容差掩盖RGB差异。实际Dataset读取为：新ROI6次＋同缓冲全图参考6次＋同一个旧全图样本before/after共2次＝**14次**，另有明确人工小型fixture。没有额外真实回放或模型前向。

| 单次实际Dataset读取，6样本均值 | 新ROI | 同缓冲全图参考 |
| --- | ---: | ---: |
| 解码 | 0.01491 s | 0.12882 s |
| 裁剪及返回元数据 | 0.00235 s | 0.00287 s |
| 合计 | 0.01726 s | 0.13169 s |

这是一次有界读取记录，参考全图的ROI外为占位零值；其PNG熵与真实全景不同，且存在文件缓存，不能当长期训练I/O benchmark。旧C实际全图单样本修改前读取0.34074秒，修改后数据逐值一致，时间记录在comparison JSON中。生产ROI读取没有重建全图空画布。

视频实际11帧，PTS0..1.0秒、步长0.1秒，末帧正常保持0.1秒（容器约1.1秒）；没有拉长或补帧。只展示局部ROI、相机和源时间，灰底是展示占位，不是渲染背景。没有新增人工画质批准前置条件。

## 成本与故障账

初次进程PID1109751完成frame0两次渲染后，在写coverage JSON时遇到`numpy.int64 is not JSON serializable`。对应缓冲、部分成品、设置、日志保留在`failure_attempt_1/`。修复仅将布局结构中的整数规范为Python int，增加实际writer定向测试。使用同一首帧缓冲重新消费；恢复进程PID1118286从frame1开始，一次加载完成余下10个时刻，没有重复任何成功相机渲染。

因此**实际启动/场景加载各2次**；严格“一次启动完成11时刻”的实测项没有通过。代码仍使用原单进程帧循环和persistent data，第二个进程的10帧对象身份、body/cloth顶点及groom拓扑/头矩阵复用全部通过。本轮不为消除这个故障记录而再跑一遍11帧。

| 计时 | 实测/口径 |
| --- | ---: |
| 初次进程至首个render回调（含装配及首帧几何，combined） | 19.138 s |
| 恢复进程一次性场景/资产/装配 | 17.633 s |
| frame0几何＋ROI渲染；修复后单独成功消费 | 19.869 s；4.076 s，非无中断首帧wall |
| 恢复进程首帧frame1循环，含消费 | 23.548 s |
| **frame1..10实际每时刻均值/中位数** | **18.391 / 18.024 s** |
| frame1..10最小/最大 | 16.131 / 23.548 s |
| 重启后稳定frame2..10（9帧）均值/中位数 | 17.817 / 17.693 s |
| 初次失败进程wall，含已成功的首帧渲染 | 43.487 s |
| 恢复Blender进程wall，含同步消费 | 203.310 s |
| 生产阶段计算wall合计，含失败及首帧重消费 | **250.873 s** |
| 额外6组同缓冲验收＋旧样本after读取 | 4.381 s |
| 打包 / ROI视频 | 0.299 / 0.876 s |
| 完整收尾进程，含导入/验收/清理/视频 | 8.692 s |
| **计算阶段累计，含失败和全部交付计算** | **259.569 s** |
| **首次启动至交付实际经过wall，含修复/开发暂停** | **501.456 s** |

259.569秒是执行阶段实测wall归属合计，不能冒充无中断端到端wall；中间约241.887秒用于修复、备份、检查和非计算间隔。30分钟预算以计算为主，本次连更保守的经过时间501.456秒也未超限，保守deadline余1298.544秒。没有第二次故障重试；未测无故障“一次启动11帧”的总wall。

后10帧均值分项：几何更新/检查0.152 s（包括恢复首帧）、ROI规划0.849 s、双路ROI调用13.400 s、CPU消费子进程3.987 s。第一路ROI平均**11.369 s**、第二路2.031 s，说明过去第一路背景调用承担的昂贵首次状态处理没有凭空消失。相机调用仍是render/compositor/写出/同步combined，没有GPU纯kernel计时。

CPU消费函数体均值0.713 s：读缓冲0.104、标签/数值缓冲0.167、局部物理mask合成0.020、局部PNG/mask保存**0.308**、消费检查0.105秒。子进程总wall与函数体约差3.27秒，含解释器导入等；当前布局模块位于`densekp.core.datasets`，该包初始化会导入torch/Lightning（没有构造网络/加载权重），是后续可研究的启动成本，本轮不追加优化。

与旧C的历史参考比较：后10时刻23.188→18.391秒，约**1.26×、时间少20.69%**；计算阶段含交付299.980→259.569秒，约1.16×。**实际首次启动至交付501.456秒因故障修复暂停长于旧299.980秒**，没有隐去退化。两批设备UUID及主机并行负载不同，以上不是同设备配对benchmark；没有用`23.19−9.51−6.28`代替测量，也没把少做诊断检查都归功于渲染。

## 存储、资源及保护

22张ROI PNG共**28,313,807 B（27.00MiB）**；旧22张完整ERP PNG为260,917,615 B，新值为其10.85%，减少**89.15%**。生产帧、共享mask、NPZ/旁表、视频和6张代表crop合计约**29,852,617 B（28.47MiB）**。失败证据另保留77,066,629 B，不能把它隐藏在“无EXR”的说法里；整个交付根目录统计时约108.85MB，含before、失败、日志和代码快照。

按逐帧实测文件大小及清理生命周期，活跃自有源＋验收参考的临时有效文件峰值约**172,692,817 B（164.69MiB）**；1秒采样同类峰值172,695,743 B（另含少量marker）。全根目录渲染阶段采样峰值272,030,336 B。六张临时全图参考自身峰值8,338,508 B。这里统计文件字节而非文件系统块，采样不是不可漏峰的硬件指标。成功清理原始源有效文件596,740,011 B及全部临时参考；失败存档的两张首帧EXR继续保留，但不属于训练依赖。

GPU0当时有其他人的活跃Python任务，GPU1也曾有A2预览；等待A2批处理退出后使用 **RTX3090 GPU1 / UUID `GPU-6f046d56-acbe-d49b-774a-9d463e768cac`**。仅共享已有118MiB、无脚本/无渲染任务的空闲Blender桌面上下文；没有终止他人进程。恢复前再次检查设备。上轮为另一张RTX3090/GPU0，UUID不同。

本次1秒采样GPU整卡显存峰值**4068MiB**（包含桌面基础占用）；Blender RSS峰值9,295,720KiB（约8.87GiB）；独立交付CPU进程峰值689,180KiB（约673MiB）。逐帧consumer的ru_maxrss可能与高RSS父进程高水位重合，不作为独立CPU峰值。仍有临时全尺寸float32 depth/ID各约66.36MB及共享bool物理mask，这是明确保留的标签内部成本，不是全图RGB生产。

## 文件与结束点

新增核心布局校验/局部warp：`densekp/core/datasets/image_layout.py`；修改现有Dataset、loading、crop_pipeline以读取元数据，未改image_processing旧full-image算法。应用层新增`data/raster_storage.py`，扩展既有roi_consumer、packing、sequence_packing、sequence_index。连续入口增加显式`--storage-mode roi_only`，默认仍为full_image；ROI规划、公共装配和旧DS入口保持。`validate_roi_only.py`仅实现有界实际验收；`resume_roi_only.py`只记录本次一次恢复，不是另一套渲染器或Dataset。

全部实际成品、标签、旁表和抽查通过；单进程连续10帧已实测，完整11帧则跨两次启动。尚未覆盖任意框增强、skimage/Albumentations局部预处理、多人/骑行/更多seed、长序列及模型效果。区域之外为unknown，训练方不能绕过coverage检查或把占位背景当真实观测。

保留旧输出、用户TODO、原fixture、A2并行修改与外部项目；没有下载、安装、SHA256、commit、数据集发布或网络/训练。完成后停在本批交付点，不追加62时刻、50步训练或其它性能搜索。
