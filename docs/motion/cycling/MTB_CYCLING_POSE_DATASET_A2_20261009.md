# Cycling Pose Estimation Dataset：山地车来源核查

日期：2026-10-09。范围：素材筛选，全部 `integration_only`；不生成新动作资产。

**找到两个画面上可确认的山地车候选，首选序列04，备选12。目前只能交付稀疏抽帧参考，不能直接开始有真实时间约束的连续重建。** 缺少与这两组一一对应的原视频、原时间/帧映射和原视频使用依据；不以两圈/24张推算fps。

打开 [素材评审页](../../../outputs/audits/mtb_cycling_pose_dataset_a2_20261009/index.html)，可查看20组首图筛选、两组各24张顺序浏览与标注联系表。页面的自动换图是人为浏览间隔，**不是原速视频**。机器入口：[candidates.json](../../../outputs/audits/mtb_cycling_pose_dataset_a2_20261009/candidates.json)。

## 用户评审与04原视频追加追溯（2026-10-09）

用户已表示“我觉得04效果可以”，确认其抽帧展示的姿态观感可接受；此前“待用户选择”的状态由本记录补充，原审计证据不改。该决定不代表原视频、真实时间或重建已验收。

本次进一步查找r/bikefit的山地车/骑行台/全避震相关帖子、发布者论文/源码索引、内部文件名及ORCID，仍未建立04到原帖或视频的可证对应。若干网页直接读取失败，没有绕过限制；也未把相似标题或假设车型当匹配。没有下载新媒体、联系作者或进行媒体计算。见 [追加记录](../../../outputs/audits/mtb_cycling_pose_dataset_a2_20261009/source04_followup_20261009.json)。

最直接的缺口是发布者提供 `04_unblurred.mp4` / CVAT task1798175 / 发布图片72…95的原帖或视频映射，以及原始时间/抽帧索引和视频复用依据。目前不能承诺原视频一定仍在线。

## A. 官方发布资料与实际文件

官方入口为 [Zenodo 20800829](https://zenodo.org/records/20800829)，DOI `10.5281/zenodo.20800829`，网页版本v1，2026-06-22发布。作者Matthew Vella、Kristian Guillaumier，机构University of Malta。API另有record revision=4和version index=0/latest=true；revision不冒充数据版本号。原始API、README及XML均保存在本轮输出目录。

| 官方实际文件 | API声明与已取小文件大小 |
| --- | ---: |
| images.zip | 665,114,542 B |
| annotations.xml | 886,882 B |
| README.md | 2,164 B |

ZIP中央目录实际为480张PNG及一个目录，图像路径是 `default/frame_000000.PNG`…`default/frame_000479.PNG`；不是README图示的 `images/default/`。没有视频、附加CSV、fps或帖子映射文件。只按需取图，没有全量解包。

发布者称来自Reddit r/bikefit的20段视频，每段抽24张覆盖两圈，脸部匿名化；涵盖多种车型。**这些是发布说明，不等于本轮已认证两圈原速视频。** README没有给出抽帧代码、原始开始/结束帧、时间戳、帧率、两圈边界或每段Reddit帖子URL。XML只有内部 `01_unblurred.mp4`…`20_unblurred.mp4` 文件名和CVAT任务链接；文件名中的“unblurred”不能推翻实际已匿名化图像，也不能据此声称原视频可下载。

API和README均标为CC BY4.0，是获取、核查与带署名展示本发布素材的依据。许可允许符合条款的复用与改编，但不自动证明未发布的Reddit原视频也采用同一许可。原上传者授权链、原视频下载/重建/训练依据仍unknown；发布页提及伦理审批，不等于提供了逐视频授权文件。参见 [许可证据](../../../outputs/audits/mtb_cycling_pose_dataset_a2_20261009/license_evidence.json) 和 [CC BY4.0条款摘要](https://creativecommons.org/licenses/by/4.0/)。本轮未联系作者、未登录CVAT或Reddit、未尝试恢复身份。

## B. 本轮实际核查

### 标注和图像映射

实际解析20个task、40个track、480个skeleton、480个head box、6,720个点。每个task一个person track和一个head track，不代表20个独立人物。

14个具名点是双侧shoulder、elbow、wrist、hip、knee、ankle、foot_index。没有APD的head/thorax三维点，也没有DenseKP138定义。`foot_index`精确落在脚趾还是鞋面何处，缺少更细协议。x/y是图像像素位置；无相机标定、3D、root、真实尺度或置信度。

`occluded=1`有2,597点次，`keyframe=0`有2,133点次，`outside=0`为全部6,720点次。README说明keyframe=0来自线性插值；不能把所有点当独立人工观测或把插值标记当检测置信度。遮挡点不因存在坐标就变成可见GT。字段解释参考 [CVAT XML说明](https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/)。逐关节计数见 [annotation_statistics.json](../../../outputs/audits/mtb_cycling_pose_dataset_a2_20261009/annotation_statistics.json)。

**发现README与XML的帧号描述冲突：** task元数据的start/stop为0/23，但实际track下frame在整个XML里是全局0…479。序列04实际为72…95，序列12为264…287，分别减72/264才得到任务内0…23。不能再次加task偏移。已用实际PNG尺寸、head/关节点位置检查对应；保存所有task映射与候选逐图记录。全局图像编号仍不是原视频帧号。

### 车型筛选与两个候选

先检查全部20组的首张图，不依赖不存在的车型标签；仅对04和12展开全部24张。其余组的首图及未选原因保留在 [screening_findings.json](../../../outputs/audits/mtb_cycling_pose_dataset_a2_20261009/screening_findings.json)。序列10虽是平把，但硬叉、窄胎更像通勤/fitness组合，山地车型证据不足；没有为凑第三名把它算成确认山地车。序列11原图可见弯把，也未误选。

| 项目 | 首选04 | 备选12 |
| --- | --- | --- |
| CVAT task / 内部源名 | 1798175 / 04_unblurred.mp4 | 1798186 / 12_unblurred.mp4 |
| 抽帧全局范围 | 72…95，共24张 | 264…287，共24张 |
| 发布图像尺寸 | 1920×1080 | 1080×1920 |
| 车型可见证据 | 平把、宽齿前胎、避震前叉、后避震 | 平把、避震前叉、宽胎、硬尾结构 |
| 环境与视角 | 室内骑行台，侧面略斜，画面向右 | 室内骑行台，侧前方斜视，画面向左 |
| 姿态特点 | 坐姿、躯干前倾、双手向前下方支撑；轮廓和近侧下肢较清楚 | 坐姿、躯干较直、双手向前伸；双臂可分辨，腿部运动模糊较重 |
| 主要遮挡 | 远侧左肩/肘/腕/髋全24张标遮挡，左踝22/24 | 远侧右肩23/24、右髋24/24、右踝22/24；宽松T恤遮体形 |
| 人体画面占比参考 | 标注+head包围高度70.3%–81.6% | 标注+head包围高度50.6%–58.2% |
| 时间和原视频 | 未发布fps/PTS、原帧号、帖子映射；未找到可靠原视频 | 同左 |

占比是包含遮挡/插值点的标注包围范围，不是分割测得的真实人体面积。外观成人样貌仅作筛选，年龄/真实身份未验证。两组都不是普通马路实拍；若后续动作成立，道路行进可由现有场景生成器负责，不能从骑行台虚构真实速度或道路root。

原始匿名化保持。PNG header没有可恢复时间的metadata；图像与task的original_size一致只说明发布内部一致，不能证明原拍摄视频未经裁剪、缩放或转码。相机方向/稳定性、踩踏变化与画质判断仅限实际已取抽帧，不能推断中间未取帧没有剪辑或异常。具体逐组观测见candidates及review_findings。

### 原视频追溯结果

已检查官方文件、内部文件名、数据集标题/作者相关公开索引，并有限检索r/bikefit山地车/骑行台帖子。没有建立任何“数据集序列→原帖子/视频”的可靠对应。搜索到的相似标题不作为对应证据；一个2026-07帖子晚于本数据集发布，明确不可能是其原始来源；另一帖子网页读取失败，未绕过访问限制。见 [source_search_evidence.json](../../../outputs/audits/mtb_cycling_pose_dataset_a2_20261009/source_search_evidence.json)。

因此两候选的原帖子URL、源作者账号、视频URL、可用原视频秒区间、fps、time_base和源帧号均为null/unknown，原视频下载量为0。不是证明原片已经消失，而是这次发布缺少可追溯映射、本轮未能找到。

## C. 推断、推荐与下一阶段条件

**推荐04作为优先补证/用户姿态参考；不标记continuous-motion-ready。** 它更清楚地体现平把坐姿、相对稳定的前伸双臂和前倾躯干，适合普通道路山地车的动作目标。对照既有APD6526已接受的较直躯干、手位/左臂快摆，这有可观察的姿态差异；没有计算跨素材真实3D角度，也不承诺重建后差异幅度。12保留为姿态更直的备选，腿部模糊和宽松衣物增加困难。

下一阶段最小前置证据是04内部文件名到原视频的可信映射、合法可取的原连续片段、原整数PTS/帧号及抽帧映射。目标约3–5秒或可用两圈，**本轮不编造起止秒数**。必须先核实视频是真实相邻帧且没有剪辑；若只能得到采样时间，则仍是稀疏序列，需另立稀疏适配试验，不能伪装为原视频连续拟合。

条件满足后，仅选一个片段、一个固定shape，使用时序联合拟合与有效/遮挡掩码。侧面图对深度、远侧肢体、关节twist欠约束，手指/面部可固定；脸部模糊不恢复。数据集2D点只作约束或检查，插值点/遮挡点须单独处理。身体求值仍用已验证SMPL-X模板，138点只由现有R@body生成，不把本数据集14点插值成138GT。

动作局部姿态与道路行进分开；骑行台不提供道路root。复用A1现已通过的原生时间、scene矩阵、固定骑行高度与ROI-only链，不返工M01、不另造场景生成器。本轮只引用 [M01公共场景接入](../../rendering/CYCLING_ROI_INTEGRATION_A1_20261009.md)；旧A2交接里的未接通状态是历史快照。

所有候选先保守并入 `CyclingPose_Zenodo20800829_reddit_identity_unknown`，task/内部视频名另存。即使任务号不同，也不能假设独立演员；同源邻段、裁剪、发布副本、重采样、换角色/衣物不能制造独立train/val。M01/F02仍共享APD6526，不把本次素材核查算成新已验收动作。

## 执行、成本与保护

开始前读取Git状态及指定报告，没有发现适用AGENTS.md。本地限定docs、outputs/audits/plans及资产文件名/下载记录和已知datasets顶层，没有目标数据既存记录；未扫描整盘。GPU0有其它任务，GPU1有桌面占用；记录两张卡UUID但**GPU计算为0**，仅CPU解析/PNG解码/联系表，无模型加载、拟合、推理、Blender或训练。

整包首次连接失败（0字节）；沙箱外整包实测速率约0.1MB/s，预计约两小时，因此停止本轮自己的curl并保留10,850,304B前缀。服务器明确支持Range，实际验证206和精确Content-Range后，读取64KiB目录及所需PNG成员。已有前缀可用部分复用；每个成员做路径/链接/大小约束和ZIP CRC32校验，不计算SHA256、不下载第二份整包。传输中断与失败记录不清理。

最终下载、解压、CPU与墙钟实测见 [execution_record.json](../../../outputs/audits/mtb_cycling_pose_dataset_a2_20261009/execution_record.json)。实际取得载荷 **128,644,892 B（128.6 MB）**，含中断前缀、小文件、ZIP尾部和所需成员；不含HTTP/TLS开销及网页检索服务流量。解出 **66张PNG、130,462,240 B**：20组首图，加04/12各23张新图，首图不重复下载。目录收尾前约147.9 MB，含保留的失败前缀和联系表，没有第二份全量包。

三阶段含网络墙钟分别为筛选128.84秒、04展开175.73秒、12展开222.70秒；后两阶段部分重叠，不能相加当总墙钟。审计Python内主要CPU时间合计 **8.395秒**，网页打包0.043秒；不含curl子进程CPU、解释器启动、小型临时查询、开发阅读和等待，未虚构完整端到端CPU成本。成员获取/CRC/PNG检查全部成功；初始整包网络失败、中断及一次SVG片段解析错误均记录，原数据未修改。

网络等待与CPU处理分别计数；联系表不作为原速视频或动态人工验收。全部原图保持发布字节，只有联系表/网页显示做缩放和标签；署名见 [ATTRIBUTION.md](../../../outputs/audits/mtb_cycling_pose_dataset_a2_20261009/ATTRIBUTION.md)。

新增独立只读源文件审计脚本 [audit_mtb_cycling_dataset_a2.py](../../../tools/validate/audit_mtb_cycling_dataset_a2.py) 和 [网页打包脚本](../../../tools/visualize/package_mtb_source_review_a2.py)，不改生产模块或任何公共契约。Python语法、网页JS语法、所有本地链接、两组24张映射和空时间字段检查通过；未做实际浏览器动态人工验收。见 [交付检查](../../../outputs/audits/mtb_cycling_pose_dataset_a2_20261009/delivery_validation.json)。

不修改TODO、A1文件、冻结资产、外部pano-stitch或用户prompt。size/mtime保护检查只发现 `docs/history/prompt.md` 在并行期间外部变化，A2没有写入或回退它；TODO及其它既有代码/文档无变化。无commit、无全库测试。

停在素材评审点：用户可先看04是否符合想要的姿态；即使观感接受，原视频映射与时间/许可缺口仍需补齐，不能直接宣布新动作已就绪。
