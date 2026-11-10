# A1：直接 normal ERP 与人体区域高质量渲染对照

日期：2026-10-09。`integration_only`。单一已验收探针时刻、同一人物、同一双相机；没有训练、模型推理、检测器或动作/衣物/groom 修改。

**结论：B 直接 ERP 的几何正确；C 能独立用 ROI 的高分辨率 range/IndexOB 产出本次受限 crop 的可靠标签和训练输入。计入相同的打包及 10 个诊断 crop 回读，本例 A/B/C 约为 108.35 / 99.49 / 53.02 秒，B 相对 A 为 1.09×，C 为 2.04×。** 推荐 C 作为下一批受约束行人生成的候选路线，同时保留 A 作 DS 成像对照；没有把 B/C 设为正式默认，也不据此启动 62 时刻或 50 步训练。

产物：[direct_erp_roi_trial_20261009](../../outputs/experiments/direct_erp_roi_trial_20261009/)。以下简称 `OUT`。

## 1. 固定对象和执行边界

选择 `pedestrian_seeded_pilot_a1_20261009/layout_attempt_2/road_pass_00/frame_000`，person1、`cycling_M04_v1_local_walk`、asset_frame=3、source_frame=12、time_tick=0。同一 urban_street、灯光、身体、固定绑定衣物、groom、head/world 矩阵和两相机。未按预测挑样本，未混入 A2 新 APD 资产。

旧 compact 的两路图像、mask、标注和框齐全，5760×2880；旧源 EXR 已正常清理，不是损坏。本轮重新渲染且仅渲染一次完整双路源，绝未从旧 normal ERP 反推原始 range/ID。

计算前已保存 `frozen_comparison.json`：旧 recipe/rig、源帧及全部矩阵、资产 size/mtime 清单、Git 版本与 dirty=true、mask/DS 标定引用、框/ROI、容差、计算次数和预算。冻结后资产清单未变。没有 SHA256。

固定成像设置：Blender 5.1.1、CUDA GPU0 UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`；高质量源/最终画布 5760×2880，64 samples，seed=1009、animated_seed=False、adaptive_sampling=False、denoising=False。AgX、look=None、曝光 −0.649999976、gamma=1、sRGB，RGBA/8bit PNG；最终 Dataset PNG 由同一显示 RGB 转成 RGB PNG，无线性 EXR 直接量化。原生 bounce：max6、diffuse4、glossy4、transmission12、volume0、transparent8；Blackman-Harris filter 1.5。未切换 OptiX/引擎/材质/头发或搜索采样设置。

相机标定仍为 `20260320-riding-dataset3`，0.18 m 仍是合成基线，非实测。GPU1 当时有 Blender 与另一个 Python 任务，未使用或终止它们。GPU0 运行前/结束后约 18 MiB、0% utilization。

## 2. 实现与三条实际链

| 路线 | 实际 RGB | 标签/实例来源 |
| --- | --- | --- |
| A 原 DS | 新 5760×2880 AgX 显示 ERP → DS2880 → normal ERP5760，使用原公共 DS 函数 | visibility 来自原始 ERP range/ID；最终实例来自两次 nearest remap |
| B 直接完整 ERP | 共享 A 新渲染的 AgX 显示 ERP，直接应用既定最终物理 valid mask | 原始完整 ERP range/IndexOB，实例不经过 A 的 nearest 往返 |
| C 直接 ERP + ROI | 独立 1440×720/8samples 全场景背景双线性放大；独立 64samples 全像素密度 ROI 按整数偏移嵌入 | 仅使用 C 实际 ROI 高分辨率 range/IndexOB；外部深度/ID unknown，背景低分辨率深度/ID不用作 GT |

A/B 共享的完整渲染成本分别全额计入各自生产成本。B 并非“无需渲染”。C 的两个 ROI 都是新 Cycles 渲染，未从 A/B 图裁块冒充加速；没有单独渲染人体贴背景，所有射线始终与完整街道、人物、灯光、遮挡和阴影交互。

仅对原 `tools/render/render_road_pass_sources.py` 增加一个可选 `main(render_cameras_fn=None)` 回调；原命令调用默认仍为原 `render_cameras`，装配和逐帧变换不变。编辑前检查其 size/mtime 未被并行改动，没有复制第二套人体生成器。

新增最小公共函数：

- `src/densekp_retrain/imaging/direct_erp.py`：组合既有 `point_in_view`、`erp_visibility`，基于实际直接 ERP 缓冲生成标注。
- `src/densekp_retrain/imaging/regions.py`：全图坐标矩形检查、局部缓冲偏移嵌入、unknown 外围、crop/5×5 邻域覆盖拒绝、物理 mask 检查。

独立入口：`freeze_direct_erp_roi_trial.py`、`run_direct_erp_roi_stage.py`、`consume_direct_erp_roi_trial.py`、`validate_direct_erp_roi_trial.py`（`tools/experiments/`）；Blender 回调在 `tools/render/render_direct_erp_roi_trial.py`。现有目录不应直接重跑覆盖。独立阶段调用与参数保存在各 `*_process.json`，过程日志完整保留。

## 3. ERP 几何与 border 证据

`project_erp` 继续使用原连续栅格公式：`x=W*(atan2(qx,qz)/(2π)+0.5)`，`y=H*(0.5-asin(-qy/range)/π)`；没有新增半像素偏移或 32px 偏移。两路 normal ERP 的基底对齐，Blender 相机由 CV right/down/forward 转成 right/up/back。cam4 的 `diag(-1,1,-1)` 只属于 DS 物理镜头映射，B/C 不再次旋转图片或相机。

冻结几何容差为 0.0003 ERP px，world 容差 2e-6 m。两相机已保存的实际 Blender 相机矩阵重投影与冻结 double 矩阵的最大差分别为 **0.00009163 / 0.00007951 px**；float32 存储舍入最大约 0.0002423 px。A/B/C 保存的 world、xy 与 in_view 一致，未以放宽容差来通过。

ROI 前先做 cam2 `[4331,1347,4363,1379)` 的一次 32×32、64samples 小预检。保持完整全景 FOV、5760×2880 网格，`use_border=True`、`crop_to_border=True`；Blender bottom-left border 与文件 top-left xyxy 显式转换。归一化边界使用 +0.01 像素的内部数值余量，避免 float32 边界落到前一个整数格；目标仍为原整数区间，不改相机 FOV。实际输出 shape 为32×32，对应完整渲染的同一块 **range/IndexOB 全相等**；预检 RGB MAE0.397、max2/255，不要求显示 RGB 位级相等。

正式 C 两个完整 ROI 再次核查：其**每个像素**的原始 range 和 IndexOB 与 B 对应区域逐值相等，最大 range 差0、ID mismatch0。故不是把不同投影/分辨率的相机图硬贴回全图。详见 `tiny_gate.json`、`ROI_support_validation.json`。

## 4. 冻结框与高质量区域

每相机 4 个统一视觉比较框：旧 A 原生紧框参考、中心 x±5% 原最大边、尺度1.1；调用 A1 显式尺寸接口，`shift_reference_expansion=1.0`、`crop_policy=deployment`、扩张1.0。没有追加 y 扰动或随机增强。每条路线另计算一个自己的原生实例紧框，合计 `3路线×2相机×(4比较框+1生产框)=30` 次真实 Dataset 回读。

这些框只是区域覆盖诊断，不是训练增强范围、YOLO 误差或历史 trial1.2 的替代。统一比较框类型为 `fixed_reference`，声明旧框和变换来源；仍检查身份、尺寸、有限性、目标支持。各路线生产框为自己的 `visible_instance_tight`，需等于该路线自己的 mask 紧框。

| 相机 | 高质量 ROI，top-left/exclusive xyxy | 实际局部输出 | 面积占完整 ERP |
| --- | --- | --- | --- |
| cam2 | `[3855,1091,4795,2031)` | 940×940 | 约5.33% |
| cam4 | `[3838,1093,4774,2029)` | 936×936 | 约5.28% |

ROI 是所有测试的**完整 square crop**与实际 body/clothing/全部 active groom 点几何投影包围范围的并集，外加至少32个最终ERP像素，非仅138点最小框。radius2 的原始可见性邻域、2px crop 双线性采样保护都在区域内。本例没有邻人；没有跨缝情况，跨缝/非法矩形仍显式拒绝，不开发多块周期拼接。

C 使用整数硬嵌入，没有在允许 crop 内做羽化。高低质量边界位于全部允许 crop 之外至少32px。外围是独立低成本全场景 RGB，非任意黑填充；物理 valid mask 外的黑色与 A/B 保持同规则。

`C/cam*/high_quality_region.png` 与 `valid_mask.png` 是两个概念。`render_coverage.json` 明确 label coverage、offset、shape 和 unknown 外部状态。ROI 外 range/ID 在内存为 NaN，不用低分辨率深度假充高分辨率 GT；person mask 在 coverage 外的0不认证“无人”。本例全部实际人体像素均位于冻结 ROI 内（完整 B 缓冲仅作验收对照，区域外目标像素计数0），两路全部138点的5×5邻域亦在覆盖内。

新 crop 的生产入口必须调用 `require_crop_in_region(center, scale, roi, (5760,2880))`；超界测试明确返回 `crop_outside_high_quality_support`。原始 Dataset 保持兼容、不识别新的覆盖语义；**绕过生产入口自行改 NPZ center/scale，不在 C 的支持契约内**。不能把 C 成品宣传成任意全图框增强均可用的数据。

## 5. 实际图像、mask 与监督比较

以下 RGB 为0–255显示值；保护区统计排除物理无效域。其定义在看结果前冻结，没有要求RGB位级一致。

| 比较/相机 | 保护区 RGB MAE / P95 / max | 统一参考实际256crop MAE / max | person mask XOR / IoU |
| --- | --- | --- | --- |
| A→B cam2 | 3.622 / 12 / 79 | 1.565 / 44 | 143 / 0.998639 |
| A→B cam4 | 4.225 / 13 / 89 | 1.694 / 57 | 136 / 0.998651 |
| B→C cam2 | 0.393 / 1 / 2 | 0.211 / 2 | 0 / 1.0 |
| B→C cam4 | 0.392 / 1 / 2 | 0.207 / 2 | 0 / 1.0 |

其余诊断框、全物理有效图、人体并集的 MAE/RMSE/P95/max/变化比例都在 `comparisons.json`。A→B 清晰度/颗粒变化是真实链差异：B 省掉鱼眼2880采样和两次图像重采样，头发细节和渲染噪声均更明显；不能只描述为“无代价更清晰”。B/C 保留全景ERP拉伸、合成相机基线和标定有效域截断，**不是完整真实传感器成像仿真，也不与 DS 成品等价**。

补充只读比较原始 linear RGB：B/C 两路 ROI 的平均绝对差约5.32e-8/5.33e-8、最大2.575e-5/2.563e-5，见 `linear_RGB_comparison.json`。这比显示PNG差异小得多。另一次无渲染场景设置读取确认原生 `dither_intensity=1.0`，三路线未修改该值；局部输出的量化/dither坐标及浮点末位差异可能解释显示差异，但没有进行禁用dither或改采样对照，不能唯一归因于某一种机制，也不把RGB max2称为几何或明显蒙特卡洛质量退化。该补充没有增加Cycles调用或Dataset回读。

本例三路线的紧框数值一致（两端delta均0），但 A/B 实例轮廓确有143/136像素差异。分别从各自实例重算，没有为了结果好看复制 A 的mask或框。B/C整块原始ID相等解释了其轮廓一致，RGB积分的小差异不等于几何错位。

两相机 in_view 分别123/113，visible分别68/62；可见性或遮挡状态变化0个语义点，`visibility_difference_details.json` 显式为空集合，并保留全部276个点的两路线ID支持像素数与最小range差。相同的 visibility 结果来自各路线自己的实际 range/ID、原 periodic5×5、严格4cm规则，不是复制 A 标签。

每条路线、每个对应比较框的共同监督集合均为98个语义点，ID清单保留在 `loader_readback.json`。单视角每组123/113个点受监督，其中55/51个遮挡点仍受监督。每路有效域外分别15/25点，存在实际镜头截断且共同 GT 非空。监督保持 `in_view & inside_full_crop`，visible 不作为排除条件。

30条实际 Dataset 图像分别用自己的 `_trans` 独立 `cv2.warpAffine` 重算，与 `img_disp` 精确相等；同一个 affine 变换138点，与 `proj_verts_cropped` 精确相等，完整256-square，无32px补偿。`A/B/C/packed.npz`、`index.json`、`actual_inputs/geometry_*.npz` 保存原标签引用、row、框类型、frame/asset/source帧和 affine，不改原返回接口。没有任何模型前向。

新 A 对旧探针复现：xy、in_view、visible、person mask逐值一致；RGB只有极少量1灰度级差异，cam2约0.002224%、cam4约0.001965%像素变化，MAE约7.41e-6/6.55e-6。设置/角色/矩阵/设备未发现变化，**具体末位差异根因没有定位**；GPU渲染/量化非位级复现是可能解释而非已证实根因。旧原始源已经正常清理，无法逐缓冲追溯。本轮保留新基线，不声称位级复现，不增加第二次完整渲染来追差。

## 6. 可视化入口

下列面板均按 **A / B / C 从左到右**，统一框，不各裁各的框来比较。

![cam2实际输入](../../outputs/experiments/direct_erp_roi_trial_20261009/review/cam2_reference_actual_inputs.png)

![cam4实际输入](../../outputs/experiments/direct_erp_roi_trial_20261009/review/cam4_reference_actual_inputs.png)

![cam2头部同像素区域](../../outputs/experiments/direct_erp_roi_trial_20261009/review/cam2_head_same_pixels.png)

更多：[cam2完整ERP缩略对照](../../outputs/experiments/direct_erp_roi_trial_20261009/review/cam2_full_ERP_overview.jpg)、[cam4完整ERP缩略对照](../../outputs/experiments/direct_erp_roi_trial_20261009/review/cam4_full_ERP_overview.jpg)。完整缩略图从上到下为 A/B/C，C黄色框标出实际ROI。`review/` 另含另一侧头部、身体同像素图和其余固定crop。已查看实际crop和头部图，未发现人体相对像素错位；更锐利/更多噪声是否适合训练仍需评审，单时刻不能认证无闪烁或模型收益。

## 7. 成本、内存、磁盘与预算

所有实际 Cycles 相机调用 **7/10**：完整双路2、tiny1、低成本背景2、独立ROI2；没有失败渲染，没有ROI修正，没有其他采样试验。包括CPU场景设置检查、Blender冷启动和tiny在内，渲染进程累计墙钟 **163.14秒**，低于1200秒，还余1036.86秒；未使用余量扩算。

完成后补充的无渲染编码设置读取未独立采集精确wall，保守再占用10秒预算；本批预算账因此按 **173.14秒** 计、余1026.86秒。这10秒是预留而非伪造实测，不进入三路线生产成本。

| 每次调用 | combined 渲染/compositor/EXR/PNG 秒 |
| --- | ---: |
| 完整64samples cam2 / cam4 | 37.614 / 27.130 |
| tiny cam2 | 12.786 |
| 背景8samples cam2 / cam4 | 13.533 / 0.489 |
| 独立ROI64samples cam2 / cam4 | 1.900 / 1.884 |

第一次tiny/背景调用仍有场景GPU同步/加速结构成本，面积小不等于没有固定成本。相机调用内无法分离GPU纯kernel、compositor和写出，统一记combined，不制造精确kernel时间。

| 双路单时刻成本 | A 原 DS | B 直接ERP | C 直接ERP+ROI |
| --- | ---: | ---: | ---: |
| Blender进程墙钟，完整计入 | 86.172s | 86.172s（共享但全额计入） | 39.062s |
| 其中场景/资产/几何准备，combined | 18.324s | 同A | 18.703s |
| 生产后处理分项归属合计 | 15.084s | 7.289s | 7.144s |
| 冷启动生产成本合计 | **101.256s** | **93.461s** | **46.206s** |
| 相对A生产加速 | 1 | **1.083×** | **2.191×** |
| 再加相同10诊断crop打包/回读/导出 | **108.353s** | **99.495s** | **53.021s** |
| 相对A含回读加速 | 1 | **1.089×** | **2.044×** |

这是同一环境下**实测步骤归属合计**，不是三次独立完整流水线进程wall。A/B共用的EXR/显示RGB读取也各自计入；后处理分项不包括比较用图像统计和Python解释器导入。Blender外层监测存在最多约2秒轮询量化误差。C已计入每次背景和两次ROI、辅助输出/合成/编码，没有减去假设的背景缓存，也没有借B完整缓冲生产标签。tiny34.93s属于本次方法验收开销，列入本批预算，不放进每个生产时刻；若将tiny也摊入C首次验收，首次收益会相应降低。

B省去本例约4.003s的DS域LUT准备与3.617s RGB/ID往返remap；仍需要完整高质量渲染、原始range、mask、标签和PNG写出，故总体提升有限。A若提前缓存LUT，其相对差距会缩小；不能把本次冷启动差值当每帧固定节省。

没有实测完整的“下一动作时刻”。同进程第二相机/后续ROI仅说明当前相同场景状态的warm调用，不能拿1.9秒直接外推运动更新、多人物和31时刻吞吐。缓存背景需要相机/灯光/场景不变，还需处理运动人物对外围阴影/反射的贡献，本次没有省略这些成本。

CPU消费者函数体wall：A/B及几何/复现检查23.803s，C7.145s，打包/30回读/比较图29.086s，原始ROI支持/逐点核查1.505s，合计约61.54s；不包含解释器导入、冻结清单、轻量文件核查与测试。峰值RSS：A/B1618208KiB、C946336KiB、比较/回读1355224KiB。Blender full/region峰值约10249932/8833000KiB；GPU2秒采样观测峰值 full3757MiB、tiny2337MiB、region3669MiB，并非不可漏峰的硬件峰值。资源记录在 `*_resources_before.json`、`*_process.json`。

| 最终两张normal ERP PNG | 大小 | 相对A |
| --- | ---: | ---: |
| A | 24,966,209 B（23.81MiB） | 基准 |
| B | 34,717,808 B（33.11MiB） | +39.06% |
| C | 23,623,179 B（22.53MiB） | −5.38% |

直接全图保留更多细节/颗粒，PNG本例更大，不能承诺“直接ERP必然省存储”。连同标签、mask、10crop及行映射，A/B/C目录约26.03/35.84/24.78MB。原始临时缓冲约964.44MB，暂留供这次对照评审；所有目录带本任务所有权marker。最终NPZ/PNG/labels不依赖它们，未来仅使用原有所有权清理函数清理本任务成功缓冲；不动历史源或失败现场。

## 8. 测试、完整性和限制

`tests.test_render_regions` + `tests.test_high_res_boxes` 共 **17项定向CPU测试通过**。覆盖局部offset/shape、unknown外围、空/反向/越界矩形、crop保护边界与超界拒绝、5×5邻域、无效mask、物理域与render coverage分离、遮挡点仍监督、局部range/ID与完整可见性一致。没有全库discover、小网络更新或模型构造。

一次CPU汇总脚本因多余右括号在解析阶段失败，已修正，初始 `validation.log` 和 `validation_attempt_2.log` 均保留；没有产生额外渲染或Dataset回读。实际回读严格30次，其后只读NPZ/文件清单与原始缓冲统计，没有再解码Dataset item。

输入资产及旧compact文件size/mtime全部未变；TODO/README未变。执行期间发现 A2自己的 `CONTINUOUS_CYCLING_ASSET_INTEGRATION_A2_20261009.md` 和 `docs/history/prompt.md` 被并行更新，未改动、覆盖或回退它们；没有发现其他公共Python文件被并行修改。未改A2专用脚本、动作/衣物/groom、外部pano-stitch、历史fixture、旧DS默认或采样默认；未安装/下载、SHA256、commit或移入datasets。完整性记录见 `final_integrity.json`。

本轮通过的是**单时刻成像路线、局部缓冲支持、标签和实际crop消费接口**。不是连续动态无闪烁、多人泛化、真实曝光同步、真实传感器等价、模型精度或训练收益验收。

## 9. 路线建议与后续评审点

1. **直接ERP几何成立。** B适合需要完整高质量画布的候选场景；省掉DS采样/往返，但本例总收益只有约8–9%，且PNG更大、成像域改变。
2. **优先评审C用于受限crop的下一批小试。** 本例独立可靠标签/模型输入成立，实际端到端约2.04×；ROI必须逐时刻根据所有body/cloth/groom、邻人、允许crop和可见性邻域规划，超界必须拒绝，不能沿用本单帧ROI盲跑整段。
3. **保留A为成像对照，不自动替换正式默认。** B/C省略真实处理链中的鱼眼采样特性，需用户接受这类数据域差异；当前静态面板尚不能证明DenseKP或真实数据效果不退化。
4. **尚不直接恢复62时刻/50步训练。** 已具备选择候选成像路线的单帧技术证据；下一批仍需冻结选择，检查连续代表时刻和多人情况下的动态ROI覆盖/成本、边界闪烁、背景更新，并重新验收实际遮挡覆盖及原预算。A2 native timeline/root/support-policy接入缺口仍独立存在，本轮没有借机实施或阻塞A2。

停在本次对照评审点，剩余渲染预算不自动用于新增场景、后续帧或训练。
