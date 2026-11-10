# 本地公开骑行动作盘点（A2）

日期：2026-10-09。**已接受2段motion；本批推荐3段原生BVH候选，尚未把它们适配成SMPL-X或穿衣资产。** 其中第三段与第二段来自同一参与者的不同区间，可以增加姿态覆盖，不构成独立演员来源。全部 `integration_only`。

[统一观看页](../../../outputs/audits/cycling_motion_library_a2_20261009/index.html) · [机器库存及精确路径](../../../outputs/audits/cycling_motion_library_a2_20261009/inventory.json) · [原速解码核对](../../../outputs/audits/cycling_motion_library_a2_20261009/preview_time_validation.json) · [CPU执行记录](../../../outputs/audits/cycling_motion_library_a2_20261009/cpu_execution.json)。

## 已接受动作和阶段状态

| Motion来源 | 原动作可看 | 标准SMPL-X／R138 | 穿衣观感 | 公共场景消费 | 许可及来源 |
| --- | --- | --- | --- | --- | --- |
| APD6526，源帧30…105 | 原14点估计序列；M01/F02都继承同一段motion | 已通过目标体型求值和R回读 | M01已接受；F02-APD单独状态仍按其旧报告 | M01已用26原始时刻进入A1 ROI道路链；不代表全76帧正式数据 | [APD Zenodo 5724486](https://zenodo.org/records/5724486)，CC BY 4.0；原演员/session未知 |
| bikefit_XBLMidr4dps，源展示帧8662…8781 | 原视频120连续帧，整数PTS | 源身体已适配；F02目标身体/R回读通过 | **F02自身简单衣物和短波波头完整动态现已由用户接受**；局部穿插保留 | 当前仅资产适配器读取，未作道路ROI | [Bike Fit Adviser 原视频](https://www.youtube.com/watch?v=XBLMidr4dps)，来源许可unknown，仅内部可行性 |

F02新组合的用户后续决定已追加到 [原交付报告](MTB_F02_DRESSED_ASSET_A2_20261009.md)；其 [manifest](../../../outputs/experiments/mtb_f02_dressed_asset_a2_20261009/delivery_manifest.json) 和A2组合目录仅更新人工观感字段，修改前副本在本次审计 `acceptance_originals/`。身体参数、衣物、groom、120帧时间和视频字节未动；`public_consumer_tested=false`、许可unknown不变。M01与F02共用APD时只计一段motion。两段不同素材来源也未证明和既有预训练资料无重叠。

## 本地库存与筛选边界

先读历史索引、资源manifest和已保存官方Zenodo元数据，再只查它们指向的APD三条提取JSON、Cyclist Motion Capture两位参与者的BVH/小NPZ、已知BEDLAM/BABEL记录。没有扫整盘或解包服装大归档。深入核查5段未接受数据：participant_1两段、participant_2一段、APD6900/7500；只有前三段新制原生骨架视频，APD复用已有预览。BEDLAM motorcycle仅作索引线索，不把五个衣物配对或四个文件名当新motion。

公开动捕来源为 [Kyriakou、Chrysanthou、Kyriakou 的 Dataset of Motion Capture of Cyclists，Zenodo 10668611](https://zenodo.org/records/10668611)，本地官方记录 `outputs/motion_inputs/cycling_fit_20260929/zenodo_record.json` 中 `license.id=cc-by-4.0`。记录描述主动LED/24相机系统、涡轮骑行台以及公路车/TT车；本地两位参与者的ID分别带 `RB`，按发布方命名规则指公路车，并非山地车。数据集记录说明BVH是映射到骨架的发布物，**可能仍有误差**；C3D原始测量和BVH/SMPL-X人体表面不是同一种真值。本批没有使用自行车、C3D或原始视频。

本机的 `clip_native.npz` 实际为participant_1的240帧和participant_2的300帧，均75关节、372通道；两份原BVH文件也存在。源帧分别 `[7560,7800)` 与 `[5700,6000)`。前者BVH `Frame Time=0.0416667s`，后者为`0.0333333s`，这是发布BVH的**标称采样间隔**；没有逐帧原采集PTS，不能声称真实相机曝光或未重采样。原通道有位置和按 `Zrotation→Xrotation→Yrotation` 排列的旋转；position通道已经给出局部位移，不能再加OFFSET。依层级重新FK，在前三段中的既有两份小缓存回读最大误差1.28e-13原生单位。另从participant_1完整原BVH定向读取 `[9600,9792)`，并核对其层级头与原提取片段一致。该段不是把已有片段变速、复制或循环。

原生单位官方未明确；既有“0.01米/单位”来自人体比例推断，暂不当真实量纲。+Y为上轴推断。预览保留原坐标，X-Y和Z-Y固定视角，整段同一固定中心/尺度；没有逐帧root归零、逐帧缩放或旋转对齐。图中手、脚只有BVH关节，不表示确切把手/踏板接触。预览视频按BVH `frame_time` 编码，逐帧解码数准确，最长标称时刻漂移1.59微秒；没有时间插值或加速。助理查看了每段5时刻的两视角联系表、时间/连续数值，没有在播放器中实际连续观看，完整动态人眼判断留给用户。

## 三段推荐候选

| 顺序 | 原帧、时长与发布ID | 实际可见差异 | 连续性和不足 | 后续适配 |
| --- | --- | --- | --- | --- |
| **1 participant_2直臂** | `2_RB_M27_20230809_PA` `[5700,5940)`，240帧约8s；[视频](../../../outputs/audits/cycling_motion_library_a2_20261009/previews/p2_straight/native_skeleton.mp4) | 躯干较立、肘部近伸直，双腿交替屈伸；比已接受Bike Fit Adviser前倾视频更能补姿态覆盖 | 原帧连续、有限、无完全相同相邻通道行；左右相对足高相关约−0.994；远端关节仍需人工连续观看，没有真实车把/脚踏 | **优先一段试验**；须新建有界BVH→SMPL-X约束适配、核实统一尺度/前向与目标shape |
| **2 participant_1中度弯肘** | `1_RB_M42_20230719_PK` `[7560,7752)`，192帧约8s；[视频](../../../outputs/audits/cycling_motion_library_a2_20261009/previews/p1_comfortable/native_skeleton.mp4) | 相比participant_2肘部更弯，身体中度前倾；左右腿交替 | 原帧连续、无完全重复；足高相关约−0.998；另一段同一参与者/录制关联，不能彼此独立划分 | 可复用同一关节映射技术，但须独立检查最终动作和衣物观感 |
| **3 participant_1较前倾** | 同一原文件 `[9600,9792)`，192帧约8s；[视频](../../../outputs/audits/cycling_motion_library_a2_20261009/previews/p1_alternate/native_skeleton.mp4) | 与该参与者上一片段相比，联系表中躯干更向前、手臂相对位置有变化；原历史窗口测得倾角约53° vs前段约40°，均是本地骨架估计而非官方姿态标签 | 从完整BVH逐行取得，连续且无完全重复；足高相关约−0.998；与候选2同演员/session，姿态差异可计动作覆盖，不能计独立演员 | **可选第五段**；先由用户看连续播放是否值得保留；适配成本与候选2关联，不承诺自动无缝复用 |

三段的逐帧关节位置有限，最大相邻主要关节位移约6.18、6.42、6.79原生单位；绝对通道角可在±180°跨界造成近360°数值跳变，不能直接解释为肢体翻转。没有完全相同的相邻372通道行。数值与5时刻图支持“值得看、值得有界适配试验”，不构成动态人眼接受、更不等于SMPL-X/衣物通过。三者都在骑行台上，不提供道路root；将来的道路行进仍由A1场景轨迹拥有，源root需按明确约定处理。

候选1/2的发布参与者代码不同，代表不同命名参与者；真实姓名、与APD/video演员关系及预训练重叠均未知。候选2/3为**同一人、同一BVH源录制文件**，原时间区间不相交但必须关联分组；换目标角色、衣物、相机、速度或循环都不创建新motion来源或独立验证来源。

## 核查后未推荐的本地线索

- [APD6900原预览](../../../outputs/audits/apd_feasibility_20261008/samples/6900/preview.html)：官方split为validation、实际554帧；本地审计前251帧10秒中1.3–5.2秒有下肢反复屈伸，但右足/膝在65–67附近跳变、样本1缺3D骨架，5.5秒后近保持姿态。它是**估计14点**，不是SMPL-X或独立测量GT；与6526是否同演员未知，不能用官方文件或split名认定独立人。可作备用约束来源，目前不作为新增动作推荐。
- [APD7500原预览](../../../outputs/audits/apd_feasibility_20261008/samples/7500/preview.html)：官方test、原884帧；前251帧有下肢缺失段与后段足端跳动，起停标签也不能当准确静止约束。保留轨迹研究价值，不计本轮骑行动作。
- BEDLAM2训练映射的motorcycle有5个身体配对、4个不同BMLrub原文件；目前仅索引/预览URL线索，本机未取得本次对应动作NPZ，名称也不能证明自行车踩踏。BABEL公开类别曾有骑行/摩托混合标签，未建立本地完整动作关系。已有KIT/883 `wipe_arm` 与KIT/314 `bend_left` 虽有参数文件，动作内容并非骑行；两个COCO单照片姿态也不计连续motion。

APD原项目记录为[Zenodo 5724486](https://zenodo.org/records/5724486)，元数据明确CC BY4.0；本机保存三条提取JSON、旧预览和原下载证据。在本次限定的审计来源目录没有定位到完整`dataset.zip`，不能推断整机不存在。APD三条是轨迹文件而非验证过的不同演员；不改原项目历史train/validation/test划分，也不把它当本项目的独立验证证明。

## 推荐组合与下一批最小任务

最稳妥的覆盖目标先是**4段motion**：已接受APD6526＋bikefit视频，新增participant_2直臂＋participant_1中度弯肘。若用户连续看过第三段、确认差异足够，扩成**5段motion**：再加participant_1较前倾。这时4段出自公开数据集、1段出自视频重建；数据来源组少于5个，不能把数量写成5个独立人/录制。若第三段观感重复或不稳，就保留4段，不硬凑第五段。

下一批只做 `participant_2 [5700,5940)` 的一个短窗口（建议先前3–5秒，源frame_time/源帧号保持原样），目标现有 `cycling_M01_v2` 固定shape。M01模型、R138、simple衣物与groom以及公共骑行场景消费均有独立既有证据，能先聚焦新BVH动作适配；此选择会使M01跨APD和新动捕共享角色家族，后续来源划分必须保守处理。下一批需：核实BVH单位/前向或记录统一估计、具名22关节约束与手脚弱约束、整段固定shape/尺度和root所有权、连续SO(3)姿态拟合、R@标准身体回读、动态身体/穿衣人眼评审。原BVH→SMPL-X生产CLI已退役，**需要重新实现/恢复一条小型受控适配桥接**；本批没有做，也没有验证运行时长。两个BVH参与者可复用同类工具，但不能直接复制其中一人的shape或局部旋转给另一人。

正式数据集来源许可与角色关系仍需共同审查。官方元数据给出CC BY 4.0；后续派生、训练和发布仍需按该许可及SMPL-X模型、人物皮肤、groom各自条款逐项核对。bikefit视频许可unknown，现阶段只有内部可行性状态。所有新候选在进入SMPL-X之前均不可宣称R138或穿衣标签已就绪。

## 资源与保护

计算前只读检查：GPU0约4053MiB/100%，A1 Blender正在运行；GPU1约407MiB/0%；RAM可用约128GiB、磁盘可用约12TiB。本轮完全CPU，不启GPU、Blender、人体模型求值或重建网络。三段BVH解析/FK/骨架画图/编码实测合计见执行JSON，远低于15分钟；数据读取、文档与等待时间单列，不把静态文件扫描说成媒体计算。新预览与审计目录约24MiB，含必要图帧/联系表/小视频，没有复制BVH原文件或模型。

只新增本报告、审计JSON/HTML/预览和独立CPU脚本；F02旧报告及A2机器清单仅最小更新人工接受状态。原身体、衣物、groom、时间和视频字节、用户TODO、A1源码、外部pano-stitch及其它历史资产保留。没有下载、安装、SHA256、commit、拟合、重定向、DS/ERP场景、DenseKP推理或训练。本轮停在候选观看点。

## 2026-10-09 用户后续决定

用户看过候选后表示“可以这三个都接受”。本报告推荐的participant_2直臂、participant_1中度弯肘、participant_1较前倾三段，状态均更新为 **原生BVH连续动作候选已获用户接受**。这使计划中的姿态覆盖可按2段已交付＋3段已选候选安排下一批；当前仍只有2段SMPL-X／穿衣动作已交付，不能写成5段成品。三段BVH均未拟合、未绑定衣物/groom、未生成R138或公共场景。participant_1两段同演员同录制的来源关系不变，许可与时间、单位限制不变。下一步仍按上文顺序先试participant_2，待单段适配另批执行。
