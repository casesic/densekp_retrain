# 双视角连续数据：首批核查与接口草案

日期：2026-10-08。状态：**T0 有界核查完成（缺口未消除）；T1 交付契约、T2 改动方案待评审，均未实施。**

总计划：[DUAL_VIEW_TEMPORAL_PLAN_20261008.md](DUAL_VIEW_TEMPORAL_PLAN_20261008.md)。本报告不授权后续下载、重建、生成或训练。

## 1. 结论与本轮边界

- 找到可复用的 dataset3 **4325–4575，共 251 帧双路 normal ERP**，不止历史02214单帧证据；已有记录混合全检测104帧、跟踪113帧、恢复检测34帧。它们不是251帧独立YOLO框，更不是GT。
- 显式有效域与检测ROI已复核。1550条保存框记录中373条跨有效域、0条跨ERP接缝；不能把“边缘”定义为左右接缝。
- 两路视频头均为2880×2880、30000/1001 fps、8359帧；同索引映射成立于现有准备代码，但**曝光同步、偏移、漂移未证实**。时间清单标注 nominal，不冒充逐帧PTS。
- 有限距离共同覆盖必须由同一组三维语义点分别投影。既有C组合成person3两侧域内点123/42，共同仅27且全部双侧遮挡；person4共同0点。真实片段缺可靠138点GT，不能给真实精度或共同可用点数。
- 连续骑行交付仍缺少同一人物的连续motion及动态衣物/头发。两个现有canonical_pose均为1帧，不是连续运动；10成人库仅12个静态组合获候选使用认可。
- T2最小改动不只是帧号循环：尺寸感知框、世界变换的时间语义、逐帧groom、序列索引/打包旁表、compact生命周期和检查视频均需明确接口。
- 另发现旧来源脚本只传源宽1536、未传最终宽高；子入口现默认最终5760×2880。**调用行为已确认，历史复跑兼容性存在风险**；本轮不修复、不复跑。

初始HEAD `e53dc60`；外部仓库HEAD `830dc51`。初始已有 `docs/README.md`、`docs/TODO.md` 修改及未跟踪总计划，全部保留。本轮没有修改这些文件或任何生产代码、原配置、旧结果、外部文件。查阅README、用户指定七份文档及相关代码；未找到适用的AGENTS.md。

本轮仅新增本报告、[有界审计脚本](../../tools/validate/audit_dual_view_temporal.py)及 `outputs/audits/dual_view_temporal_20261008/`。未加载模型、推理、训练、渲染、下载、安装、视频帧解码、删除、提交或计算SHA256。外部纯CPU ROI函数被用于重算ROI，关闭了Python字节码写入。

## 2. 证据清单与核查层级

以下 `X` = `/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch`，`P` = `X/data_process/tmp/production_regression_20260320_dataset3_multiperson`，`O` = 本轮audit输出目录。所有尺寸均宽×高。

| 证据 | 本轮实际读取及限制 |
| --- | --- |
| `X/Circular_Video/01_send/20260320_riding/dataset3/VID_20260317_112543_00_013.{0,1}.mp4` | 两文件ffprobe流/容器头；没有show_frames、count_frames或解码 |
| `P/prepare_selected.py`、`prepared/preparation_summary.json` | 核查0→cam2、1→cam4，20260320，闭区间4325–4575 |
| `P/prepared/erp/{cam2,cam4}/04325.png…04575.png` | 检查502个明确文件名存在；只解码首、中、末各一对，共6张既有PNG |
| `P/prepared/valid_masks_20260320.npz` | 两个bool[2880,5760]显式mask；参考镜头mask由4325帧生成 |
| `P/standalone_sequence/sequence_result.json` | 251帧记录、已配对检测框、跟踪框、保存后端与导出空间；不重新执行其计算 |
| `P/temporal_sync_audit.json` | 复用既有10次抽查结果；不是本轮解码，也不是曝光同步证明 |
| `P/raw_all_fixed_cases/sequence_result.json` | 核实只有3帧固定案例，不当作全251帧原始检测日志 |
| `X/data_process/Input/20260320_riding/dataset3/{review,seg,candidates}.json` | 517条选帧、47段、709条候选记录；candidate含17点鱼眼候选和分数，不是138点人工GT |
| `outputs/experiments/ds_source_resolution_20261008/C/` | 两相机标签、mask、同一world_densekp[4,138,3]重新CPU投影，未读取EXR或运行后处理 |
| 两照片canonical_pose、10成人character_manifest/combination_status | 核查数组shape、候选数与接受范围；没有生成身体/衣物 |
| 400步普通comparison、框增强status；输出模式源码和定向测试源码 | 复用已有结果，未重跑训练/测试；普通effect_passed=false，框增强deployment_replacement_accepted=false |

`O/audit.json`记录完整路径、两repo版本、ffprobe命令/结果、统计及3D检查；`frames.json`包含251帧索引与名义时间，`boxes.json`含1550条逐框记录；`guided_changes_not_detector_error.json`是250条相邻跟踪框变化。`interface_draft.json`是机器可读**提案**，未获批准的阈值为null。

注意版本证据：外部准备脚本导入 `dist/human_fitting_delivery/vendor`。本轮逐字节比较了它与src的 `overlap_roi.py`、`camera/fisheye_equirect.py`、`camera/erp_rotate.py`，三者相同；这不证明所有dist文件相同，也不证明历史执行时整个工作区干净或资产版本完全冻结。

## 3. T0：真实时间、坐标与交界区域

### 3.1 时间链：索引对齐不等于曝光同步

两路媒体头完全一致：HEVC，2880×2880，`r_frame_rate=avg_frame_rate=30000/1001`，`time_base=1/30000`，start=0，流duration=278.911967 s，容器278.912 s，nb_frames=8359。不能把帧率近似成30后用于精确时间映射。

`P/prepare_selected.py:25`定义两个视频，`:29`固定区间，`:42`同时seek到START，之后两边每轮read一次，按同一个0基帧号写PNG（`:59`）。因此本轮清单按现有生成约定写：

`cam2 frame n ↔ video .0 frame n; cam4 frame n ↔ video .1 frame n; nominal t=n×1001/30000 s`。

首帧144.310833 s、末帧152.652500 s，首末时刻差8.341667 s；251帧的名义占用时长8.375033 s。显示用帧4325/4450/4575在查看预测之前按首/中/末固定，不按误差选样。

现有 `temporal_sync_audit.json`比较651、2131、4175、6146、8356各相机解码图与**本相机同索引旧JPEG**，reported_next均n+1，MAE约1.276–1.796。这支持旧JPEG索引解释；没有比较两路同一外部事件的曝光时间，不能估计相机间offset/drift。容器同时从0起算同样不足以证明同步。本轮没有逐帧PTS、丢帧/VFR或硬件同步证据，`measured_pts`、`sync_offset_s`均保留null。

路径陷阱已确认：旧review使用Windows分隔符及 `00_send`，现存视频在 `01_send`；`X/data_process/Output/config.json`的dataset4条目指向实际dataset3，而其dataset3条目指向dataset2。接入应冻结真实路径、basename、相机映射和标定组，不能按泛化的“dataset3”配置键猜测。

后续补齐方式（本轮未执行）：由拍摄方提供同步说明/硬件时间；获单独授权后抽查有限首中末事件及对应PTS，估计固定偏移与漂移、记录不确定度。未达到约定时间容差前，不宣称时序点对或速度指标可靠；不能简单把同索引当已同步真值。

### 3.2 坐标链及其消费者

| 阶段 | 已确认行为/单位 | 源码证据（X下路径为外部只读） |
| --- | --- | --- |
| 鱼眼→normal ERP | 2880-square→5760×2880；DS标定20260320；new是20260320别名，不是0311 | `X/src/densekp_delivery/camera/fisheye_equirect.py:14,18,53,59` |
| normal ERP射线 | lon=2πx/W−π，lat=πy/H−π/2；CV射线[cos(lat)sin(lon),sin(lat),cos(lat)cos(lon)]，y向下 | 同文件`:59`；主仓库 `camera/erp.py:13`世界→CV投影与之相容 |
| cam4轴向 | cam2矩阵I；cam4为diag(-1,1,-1)，作用在射线后查DS LUT；不是再翻最终ERP图片 | 同文件`:32,38,92` |
| ROI局部→normal ERP | x加ROI起点再mod W，y加起点；框保留unwrapped x及wrap标志 | `X/src/densekp_delivery/overlap_roi.py:183,243,252` |
| ROI内检测器输入 | BGR，按min(640/h,640/w)缩放，右/下补114，float32；框除缩放比还原 | `X/src/detectors/yolox_common.py:18,45` |
| DenseKP单人裁剪 | 外部框最大边square，扩张1.0；合成trial历史使用1.2，两者不可偷换 | `X/src/densekp_delivery/runners/ort_runner.py:229,243,451`；主仓库 `data/boxes.py:52` |
| 模型与还原 | 完整256-square，内部中间192列；inverse affine回局部再回normal ERP，不额外+32 | `X/src/densekp_delivery/runners/ort_runner.py:282,439`；主仓库Dataset`:84` |
| erp_rotate导出 | 对normal ERP射线乘R，R将(x,y,z)变为(x,-z,y)，再投影；不是二维平移 | `X/src/densekp_delivery/camera/erp_rotate.py:25,65` |
| 输出框与点 | schema即使点coord_space=erp_rotate，bbox_coord_space仍为erp；person_id为本帧matches枚举 | `X/src/densekp_delivery/schema.py:39,47,62` |

normal ERP中的像素不等于旋转ERP或鱼眼像素；不能拿旋转后的点与normal ERP框直接裁剪/评分。球面旋转非二维仿射，旋转框也不能只变换两角。saved sequence导出是erp_rotate且optimized结果不是GT。本轮框来自 `delivery.matched_bboxes` / `guided_crops.predicted_bbox_erp`，始终按normal ERP解释，没有用优化人体反推真值。

ROI方向对齐不提供真实相机平移/基线。主仓库 `camera/rig.py:12`及C/rig.json的0.18 m明确是合成参数，不能作为实测外参；本轮审计的实际重映射链未提供经验证的真实双机六自由度外参。

### 3.3 有效域与实际检测ROI

`X/src/densekp_delivery/pipeline.py:44`的活动策略为component_bbox；`overlap_roi.py:82`从两显式mask的同坐标交集提取周期连通域，极区行覆盖≥0.8W只在种子连通分析排除，扩张160 px，最终取矩形ROI。不是默认使用compact_strips，也不是直接对整张ERP跑一次检测。

本轮由保存mask重算的结果：

| 项目 | 实测 |
| --- | --- |
| cam2/cam4有效像素 | 10,662,811 / 10,612,995 |
| 同坐标strict交集 | 4,687,006 px |
| 扩张detection mask | 7,724,371 px |
| ROI 0 | x=[118,2742)，y=[0,2880)，2624×2880 |
| ROI 1 | x=[3019,5643)，y=[0,2880)，2624×2880 |
| 两ROI矩形总面积/全ERP | 91.1111% |

ROI矩形、扩张mask、各相机物理有效mask是三个概念。矩形包含无效区与上下文，其边界不等于镜头边界。160是5760×2880下固定像素上下文；不称为各纬度相同球面角度距离。

以下图使用已有normal ERP，缩小到原尺寸1/5，仅供定位说明；青色=显式mask轮廓，黄色=ROI矩形，红色=保存框。轮廓在最近邻缩小mask上绘制，数值统计仍用全分辨率mask，不从图中量精确边界。

![4450双视角有效域与框](../../outputs/audits/dual_view_temporal_20261008/real_roi_04450.png)

[首帧4325](../../outputs/audits/dual_view_temporal_20261008/real_roi_04325.png) · [末帧4575](../../outputs/audits/dual_view_temporal_20261008/real_roi_04575.png)

4450的保存track20例子：cam2框[1205,1503,1313,2391]，cam4框[1070,1566,1508,2441]。cam2在x≈1200处被镜头有效边缘截断，cam4在对应位置看见较多骑行者身体；都不靠ERP左右接缝。框内有效mask比例为55.154%/100%，**不是人体可见比例，也不是经验证的同人GT**。哪些138点缺失、身份是否正确、具体身体部位缺失数，在真实图没有可靠标注时不能从框推断。

上游限制：准备脚本首次从4325鱼眼生成lens mask，`fisheye_equirect.py:130`使用灰度>5、轮廓填充与形态学处理。这是既有mask来源的启发式，不等于本轮按黑RGB判断有效性。后续应保留其来源并审核镜头边缘误差；本轮始终读取保存的显式mask，不对各新帧按黑色重建。

点采样边界也不暗中统一：主仓库 `camera/domain.py:12`采用floor、水平mod；外部 `valid_filter.py:33`采用round加in-bounds。外部点对过滤`:71`在两边各自坐标采样各自mask，不能替换为同一像素的二维mask交集。

### 3.4 三维投影验证：共同覆盖不能由二维交集代替

本轮读取C组保存的**同一份**world_densekp[4,138,3]，分别用两camera_to_world与20260320 DS内参投影，逐点比较保存xy与in_view。两侧最大差0.000243618/0.000244044 ERP px，in_view不一致均0；差值符合保存float32舍入量级。未运行几何重建、渲染或可见性射线计算，visible沿用既有源距离/实例计算结果。

| 合成人物 | cam2域内 | cam4域内 | 共同域内 | 双侧可见 | 单侧可见 | 双侧遮挡 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| person1 | 91 | 87 | 40 | 5 | 4 | 31 |
| person2 | 90 | 93 | 45 | 24 | 2 | 19 |
| person3 | 123 | 42 | 27 | 0 | 0 | 27 |
| person4 | 138 | 0 | 0 | 0 | 0 | 0 |

`O/audit.json → synthetic_projected_support.persons`还保存每人共同semantic_id列表及两侧缺失semantic_id，均索引0…137。它是**有效域层**集合，尚未交参考框完整裁剪；不能把它当最终评分分母。也未映射成未经核验的人体部位名称。

例子说明：两侧都“有人”可能共同点很少，双侧可见甚至为0；可靠遮挡GT仍可监督/离线评分，但当前图像未必有足够线索准确恢复。person4作为单视角完整回归对照有用，不能硬凑成双视角共同点评价。这是合成证据，不是对真实4450人体的GT结论。

后续“边缘”定义提案：分别报告每视角GT点/身体表面到**本视角显式mask边界**的有符号距离，再报告检测ROI边缘和完整裁剪边缘距离；不要合成一个“边缘分数”。ERP像素距离需横向周期处理；跨纬度比较可另算单位射线角距（rad/deg），不能把ERP欧氏像素距离称为球面距离。真实有限距离共有区域需实测相对位姿和深度/可靠人体GT，目前缺失。

## 4. T0：已有框、变化统计与不能声称的指标

保存后端为rtmlib_yolox_trt，阈值0.25，TensorRT FP16 pre-NMS+ONNX tail；这不同于02214历史单帧rtmlib_yolox ORT记录。`X/src/lib_pano/human_fitting/sequence_orchestrator.py:173`记录全检测matches的框，`:261`单独保存guided预测框。按mode分离后共870个检测框（435个双视角配对）和680个guided框（340对），不能称为1550个人或全部检测输出。

| 来源/相机 | 框数 | 宽中位数px | 高中位数px | 跨有效域框数 | 最低框内有效域占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FULL_DETECTION/cam2 | 271 | 87.750 | 285.750 | 69 | 71.60% |
| FULL_DETECTION/cam4 | 271 | 101.250 | 321.750 | 31 | 81.62% |
| RECOVERY/cam2 | 164 | 88.875 | 279.000 | 31 | 76.48% |
| RECOVERY/cam4 | 164 | 99.000 | 280.125 | 21 | 82.23% |
| TRACK_GUIDED/cam2 | 340 | 146.000 | 468.000 | 139 | 55.15% |
| TRACK_GUIDED/cam4 | 340 | 156.000 | 500.500 | 82 | 62.68% |

统计以框floor/ceil包围的像素矩形采样mask，x周期、y限图内；域比例只描述矩形。全部1550框均未跨左右接缝。每侧相邻guided帧同track有125对；中心变化除以前框最大边，绝对dx/dy的P90分别cam2=0.0376/0.0712，cam4=0.0453/0.0474；最大边比例中位数1.0179/1.0096。**含真实运动、跟踪估计与框策略变化，不是YOLO误差或可直接采用的增强范围。**

sequence保存775个人时刻观测、54个算法track；不是54名确认独立人物。match_index是本帧配对序号，不擅自映射为跨帧身份，因此本轮没有按full检测框数组顺序计算“同人轨迹”。记录仅存matched_bboxes；未匹配检测只保留数量，存在匹配成功选择偏差。全部检测器框的真实分布仍缺，不能评估漏检率。

| 所需指标 | 当前能做/缺什么 | 补齐方式 |
| --- | --- | --- |
| 相对人体参考框dx/dy、尺度比 | **不能报告检测误差**：无可靠完整人体参考框 | 对冻结的有限真实帧人工/可靠多视角标定参考框，注明amodal/visible定义及不确定度；再按参考maxside归一化 |
| 单侧截断 | 能报告跨有效域、ROI/图边关系；不能确定缺失身体比例 | 标注人体GT/部位及每侧域内状态，区分镜头截断、框截断、遮挡 |
| 可见人体/语义比例 | 框mask比例不能代替 | 实例/遮挡标注及可靠138点；当前合成可算，真实缺失 |
| 邻人侵入 | 框重叠也不能等同像素侵入，本轮不伪造该数值 | 实例mask与冻结实际crop；统计邻人像素占裁剪面积及目标像素面积比 |
| 真实双侧精度与“可用点对” | 无可靠GT；优化点、预测valid、时间平滑都不是独立GT | 冻结真实标注预算/来源；未知点单列，严禁按预测误差筛点 |

参考框草案：真实人工完整人体框与可见框分别记录 `bbox_kind`，不混用；合成以预先冻结的几何amodal投影/clear定义得到每帧各视角reference_box，随后所有模型或扰动共用。允许人体真实运动引起参考框逐帧变化，不是整个序列固定一个像素框；同一时刻对照间固定。新增扰动范围需真实误差证据评审，现在不从上述变化P90反推增强上限。

## 5. T0：配对评价协议草案

机器字段见 `O/interface_draft.json → evaluation`。以下只冻结定义建议，不宣称已完成效果验收；具体门槛在训练/预测之前由下游批准。

1. 键为 `(sequence_id, sample_time_s, person_id, semantic_id)`，semantic_id固定138体系。实际时间可用整数PTS+time_base建立无浮点歧义键；跨视角必须先验证时刻对应，track_id需注明是否经过身份审核。
2. 单视角监督/评分支持 `S_c = reliable_GT ∩ in_view_c ∩ inside_full_crop_c`，完整crop=256-square；可靠遮挡点不被visible剔除。旧权重标签维持原语义，不自动转换。
3. 配对支持 `C = S_cam2 ∩ S_cam4`，在预测前冻结。按双侧可见、仅cam2可见、仅cam4可见、双侧遮挡分组；另外记录仅单侧有效、无共同点及完全遮挡情况。**共同集合只用于双视角评价，不替代单视角训练监督。**
4. 每端分别对自身GT计算crop欧氏误差（256像素单位），以及统一normal ERP的周期误差：`dx=((pred_x−gt_x+W/2)%W)−W/2`，`sqrt(dx²+dy²)`，W=5760。不能直接减两相机的点坐标作为对应误差。
5. 点对误差取较差一端 `max(e2,e4)`；两端同时≤阈值才算达标。报告达标数量、达标/C比例、人体部位空间分布；用同一138语义映射表，不能临时按索引段猜部位。预测第三值不参与权重/阈值/筛选。
6. `|C|=0`时误差/比例为null，另计no_common_support，不能置0误差；有GT支持但预测非有限时计失败，不缩小分母。有限项均值与失败率同时给出，明显身份错误另记。
7. 可预注册探索曲线crop阈值1/2/4/8 px（**候选网格，非通过门槛**）；ERP阈值、最少点对数、身体分布要求、完整/轻遮挡/旧骑行回归退化容忍度当前均待定。ERP↔crop阈值换算必须按实际affine，不对不同crop统一乘固定比例。
8. 等点、等人物时刻、等序列口径分列；按边缘距离、人体尺寸、截断、遮挡、检测/参考/扰动框类型分组；样本清单与框不按模型误差选。时序用 `(pred−GT)` 残差随真实Δt的变化/跳变和身份稳定性；不把真实运动计为抖动，不把平滑预测当GT。
9. 用GT得出的可用点对是离线评价，不是部署时可直接读取的可靠性输出。标注不可靠点单列；允许部分点不参与loss不等于模型会输出“未知”。最终合图仍另行验收。

## 6. T1准备：连续骑行资产交付与接入契约

### 6.1 已有资产能复用什么

`outputs/experiments/cycling_smplx_20260929/{coco_292226_person_1330527,coco_306800_person_225160}/canonical_pose.npz`：body_pose[1,63]、global_orient[1,3]、transl[1,3]、source_frame_ids[1]；各自betas[10]。两独立照片来源，不能拼成同shape连续骑行动作。

`cycling_ten_adults_candidates_20260930/character_manifest.json`确认5女5男；`combination_status.json`有12 accepted、8 not_attempted、3 excluded_ds_rows、3 failed_appearance_versions。全部integration_only；accepted仅为对应静态组合的有限视角候选认可，不保证任意动作衣物适配。F04家族仍不进入来源隔离划分，不因换v3而解除。原照片人物身份与目标角色shape身份也须分开。

本轮在已列资产和 `outputs/motion_inputs` 当前骑行入口中未找到已验收连续骑行交付；这不是对所有磁盘的全量搜索结论。外部拟合后的251帧人体不能直接当本项目已验收连续motion：来源时间、身份、root/shape、动态衣物/头发、许可和质量契约均未满足。

### 6.2 交付字段（建议v1）

| 层 | 必需字段、shape与单位 | 接入/拒收条件 |
| --- | --- | --- |
| 来源 | asset/source_video/source_person/clip ID、原文件或URI、许可证据/允许用途、处理版本；identity/appearance family、split_group | 不用“公开视频”等同许可充分；本轮未选新视频或联网获取 |
| 时间 | source_frame_index[T]、source_pts[T]+time_base、有序sample_time_s[T]、回映source_time_s[T]；采样/变速/循环/缺帧规则 | fps不能代替全部时间戳；重复帧、插值帧、拼接点显式标记 |
| 姿态 | body_pose[T,63]；左右手各[T,45]；jaw/eyes各[T,3]，轴角rad；root_transform[T,4,4]，平移m | 关节/手指fixed或推测必须有标志；不逐分量线性插值轴角掩盖跳变 |
| 身份与shape | target_character_id、固定betas[B]、B/model/gender版本；source actor与target shape分开 | 单序列不逐帧改变身份、betas或拓扑；衍生角色重新标记适配质量 |
| 身体/138点 | vertices[T,10475,3] m、faces[F,3]、densekp[T,138,3]；算子引用/版本 | 使用仓库R[138,10475]作用于身体；不以BEDLAM索引、衣物点或重归一化R替代 |
| 衣物 | 每件vertices[T,Vc,3]、faces[Fc,3]、UV/材质/绑定、碰撞与验收状态 | 时间轴和坐标与身体完全一致；允许轻微局部穿模，不接受大面积轮廓/身份错误 |
| 头发 | 每时刻points[P,3]或逐帧目录索引、curve_offsets/radius/active mask/绑定 | 当前静态groom不能通过复制路径假称动态；无拓扑变化或显式新版本 |
| 世界放置 | 右手Y-up模型坐标、原点、单位、root运动保留策略、世界Z-up变换、场景轨迹 | 身体、衣物、头发、标签共用同一时刻同一变换；禁止每对象独立贴地 |
| 验收 | 数值检查、质量flags、按实际时间轴播放的多视角穿衣视频、失败帧清单 | 检查蹬踏/躯干真实变化、闪烁、翻转、漂移及接缝；不以静姿沿路平移替代 |

建议root协议采用列向量 `p_world=T_scene(t)·C_yup_to_zup·T_root_residual(t)·p_local`，C为(x,-z,y)，p_local为pelvis中心、未重复烘焙global orient/transl的几何。必须二选一并记录分解：

- in_place_plus_scene：移除源行进位移、保留所需姿态/朝向残差，场景轨迹仅叠加一次；移除/保留哪些旋转需明确。
- source_root_plus_static_placement：保留源root行进，场景只给固定放置，不再加动态行进轨迹。

序列落地使用常量初始变换或显式受控轨迹，不允许当前逐帧min-Y自动贴地作为默认时间语义。身体root、全局朝向、场景yaw的归属必须写明，不能同时保留两份旋转/平移。本轮只给契约，不自动转换现有资产。

下一批资产验收应先无模型回读：时间严格递增/所有数组同T、有限值、拓扑固定、旋转RᵀR≈I/det≈1、已知点变换闭环、保存densekp与R@body误差；再报告关节角增量、root速度/加速度、脚部/衣物突变分布及最坏帧。数值质量阈值由首段真实时长/采样率和制作方共同冻结，不编造已通过的阈值；必须加连续播放人工检查。

来源隔离按人物家族、源视频+源人物+原始片段、同源外观、全部遮挡者成组；相邻段、重采样/变速/循环、双视角、换seed不创造独立来源。先分配split再生成场景，首段可作integration诊断，但独立验证仍需另一个来源。

## 7. T2规划：最小改动和有界验收（未实施）

### 7.1 高分辨率框

[change_box](/mnt/data1/yy/home/yy/Projects/densekp_retrain/src/densekp_retrain/data/boxes.py:11)在21–22行写死1536/768；dx/dy按原框maxside×1.2归一化。建议保留旧函数的缺省/数值运算顺序，在新尺寸感知入口强制 `image_size_wh=(W,H)`，调用方从真实PNG/mask取尺寸并验证与manifest一致。历史1536策略具名，不修改旧实验的框/单位。

新增规则需显式：finite且x1>x0/y1>y0，空框拒绝；历史2 px门槛是像素门槛，不称相同角度容差；clip到图像不等于clip到有效mask；不按黑色RGB裁边。部署1.0与trial1.2扩张作为独立box_policy记录，不顺带统一。

ERP接缝提案：框采用周期展开xyxy（x0∈[0,W)，0<宽≤W，x1可>W）。不能对两端分别mod再求min/max造成巨大框。**第一版若不实现周期crop，则明确拒绝并计数跨缝样本**，不得默默clip；本轮真实1550框没有跨缝不意味着未来可忽略。若后续实现，图像、GT、valid/person mask必须同一roll/unwrap+affine，还原x再mod。周期误差已支持不代表Dataset裁剪已支持。

`prepare_box_robustness.py:96`还把parent给定框强制等于逐人可见mask tight bbox（`:109–110`），且unit文本写1536×768（`:71`）。接新参考框时需增加bbox_kind：现有可见框保留该断言；clear/amodal/检测框不可被这个断言误拒绝。保留person_*_mask.png接口，用实际裁剪统计目标/邻人，不为节省小文件改接口。

### 7.2 序列输出与消费者

| 最小改动位置 | 方案及理由 |
| --- | --- |
| 新sequence driver（tools层） | 消费已冻结episode/时间清单，逐frame独立子目录；每人显式asset_frame、T_scene/T_root、camera/seed，禁止依赖列表顺序关联身份 |
| `body/transforms.py:13`与renderer`:235–244` | 增加具名序列常量/显式世界变换策略，两边一致；保留legacy min-Y策略；避免逐帧min-Y造成高度跳变/源root抵消 |
| `clothing/groom_transfer.py:18,35`、`rendering/blender/groom.py:9` | 现为vertices[0]和静态points；最小接入可由制作方交逐帧groom目录，driver按asset_frame选取；不在本阶段重写毛发系统 |
| `imaging/ds.py:76,287` | 现有单scene/双相机输出作为每帧单元复用，frame索引一致；保留源/最终尺寸解耦、源距离可见性、cam4与显式mask逻辑 |
| `data/packing.py:13` | 当前仅打包imgname/center/scale/proj_verts/in_view/visible/person_id/camera_id；新sequence字段会丢失。最小方案新增固定row_id旁表，保持顺序和字段契约；不改网络 |
| `DatasetKeypoints.__getitem__:84` | 保持完整256-square和实际affine监督；评估器用packed_row_id旁表定位episode/time/person，不能只依赖person_id跨场景关联 |
| `imaging/output_modes.py:155,205` | 每帧任务独占临时目录，验收后清理；sequence manifest只引用长期文件；逐帧failed/pending/completed保留，不能一帧失败把其余成功状态覆盖 |
| 序列检查视频 | 从成功帧normal ERP顺序流式编码双路/并排视频，标frame/time/状态；仅视频保留必要标注，不默认恢复所有逐帧overlay和拼图；播放timebase/变速有记录，PNG仍为训练输入 |

建议目录为 `episode_manifest.json + frames/<frame_id>/{cam2,cam4,...} + sequence_rows.jsonl + review/`。每帧保存的camera/person ID局部稳定，跨episode由复合键区分。序列索引字段、身体/衣物/点引用和seed实际抽样结果见 `O/interface_draft.json`；来源清单包含遮挡者。检查视频失败单独记状态，已验证训练图不必重渲染；断点只补显式缺失单元，遇目录碰撞拒绝覆盖，失败现场保留。

compact依赖边界继续保持：PNG、138标签、pair/world小数组、valid/person masks、boxes、配方/来源/rig/版本/日志长期留存；源EXR/RGB/鱼眼预览/默认调试图不长期保留。重算成像/遮挡需按配方重新渲染，不把已删除临时路径列成必需依赖。full用于指定诊断。

清理状态修复已确认：[complete_output](/mnt/data1/yy/home/yy/Projects/densekp_retrain/src/densekp_retrain/imaging/output_modes.py:155)保留data_validation=passed、cleanup_error、residual_path；[record_failure](/mnt/data1/yy/home/yy/Projects/densekp_retrain/src/densekp_retrain/imaging/output_modes.py:205)保护该状态；`tests/test_ds_scene_pair.py:94`包含外层覆盖回归用例。本轮只读，不重复修复或重跑。

### 7.3 活动入口与历史复核的显式参数

| 入口 | 当前默认/显式调用（源码已核查） | T2处理建议 |
| --- | --- | --- |
| `render_cycling_ds.py:75` | 源5760×2880，最终5760×2880，compact | sequence driver显式传全部尺寸与模式 |
| `render_cycling_characters_ds.py:35` | 同上 | 同上；不自动宣称动态资产可用 |
| `render_adult_candidates_ds.py:70` | 同上 | 同上；接受静态/序列策略具名 |
| `run_ds_source_resolution.py` | A/B/C源配置各自保留，最终5760×2880，full | 不改历史组定义，不改“B总体最好、C非全面改善”结论 |
| 公共 `process_scene` | 未指定mode保留full，output_size=None沿源尺寸 | 保留旧直接调用兼容 |
| `run_source_isolated_batch.py:68`、`run_source_occlusion_pairs.py:34`、`render_source_ground_repairs.py:32` | 显式源宽1536/full，**未传最终尺寸** | 当前实际继承adult的5760×2880；旧1536复核入口需另批显式补1536×768并做回归；新生产则显式C，不悄悄更名为历史复现 |

最后一项是本轮发现的兼容风险，不是说旧输出已被修改。`run_source_occlusion_pairs.py:52`还读取source/person_*_world_{vertices,clothing,groom}.npy作逐值对照，故历史full仍有真实消费者；不能把这些旧实验入口强改compact。新序列若需同类对照应先冻结必要几何证据，再决定留哪些数组，不依据文件名清理。

### 7.4 后续定向验收建议

本轮没有执行以下生产实现或验收；下一任务需另外冻结数量与预算。

- 纯CPU框：旧1536完全回归、新5760不误裁、非有限/空/2px策略、图像/mask尺寸不一致、左/右/上下边缘、跨缝明确拒绝或同变换、部署1.0/trial1.2各自验证。
- 时间/几何：至少连续3帧且姿态确有变化，同time身体/衣物/groom/138点；同一世界矩阵，root无重复，T与索引错位须失败；固定变换不引入min-Y抖动；cam4不多翻一次。
- 打包：原字段逐值相同；旁表row_id一一对应；双视角/邻帧/同源人物与遮挡者分组无泄漏；无共同点仍保留记录。
- compact：先读回完整产物、标验证结果、清理任务自有临时文件，再从最终目录实际打包、Dataset回读和**生成至少一个新扰动框**，确认不是仅原框回读。模拟清理失败保护原成功验证状态；full接口不减少。
- 视频：帧数/时刻/缺帧策略可追溯；播放变化来自真实姿态，不把图像平移伪装motion；无默认逐帧大调试图。时间不均匀时显式timebase或显示重采样映射。
- 存储：记录每帧双路最终大小、视频大小、临时峰值和失败保留量。历史25.80 MiB/帧对只作估算，不乘帧数后称实测峰值；不强求本轮规划即实现逐相机流水线。

## 8. 新验证结果、未完成项与评审决策

有界审计主脚本实际执行1次成功，耗时5.24 s（不含人工阅读、报告编写及其他轻量查询）。断言通过：251帧索引连续、502个图像路径存在、6张图尺寸正确、1550框finite正尺寸、所选src/dist几何代码逐字节一致、两侧合成投影误差<0.001 px且in_view完全一致。未运行全库discover，也未调用Dataset或模型构造。辅助只读探查曾因预想字段大小写/路径不符报错，调整查询后完成，没有生成失败渲染/训练目录。

可复用命令（输出必须使用新的不存在目录；不覆盖本轮产物）：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python tools/validate/audit_dual_view_temporal.py \
  --external /mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch \
  --output outputs/audits/dual_view_temporal_<new_run_id>
```

脚本不扫描/解码视频；ffprobe只查两份头。GPU未使用，无需申请设备。主审计产物写出时4,783,141 B（约4.56 MiB，含3张说明PNG/框/帧/审计JSON；不含随后接口草案及本报告）。未采集RAM/临时磁盘峰值，不将最终体积冒充峰值。outputs被git忽略不是备份，交接时应一并保留。

| 具体缺口/待决定 | 对结论或下一批的影响 | 最小补齐方/方法 |
| --- | --- | --- |
| 曝光同步、PTS、偏移漂移 | 不能正式评价真实时序/点对 | 拍摄方同步说明；另授权有限事件/PTS核查 |
| 实测双机相对位姿 | 不能把0.18m合成覆盖当真实覆盖 | 用户/部署方提供标定、坐标方向与单位 |
| 可靠真实138GT、身份、参考框/实例 | 无法报告真实检测误差、可见比例、邻人侵入、定位提升 | 冻结有限标注清单、协议与预算；已有预测不能替代 |
| 全部原始YOLO框 | matched-only统计有选择偏差 | 优先寻找该已保存运行的完整日志；若不存在，未来另授权计算，不在本轮运行 |
| 连续骑行视频人物motion与动态资产 | T1尚不能验收；T2不能靠循环静帧完成 | 制作方按第6节交一段最小资产及许可/时间表 |
| root/轨迹所有权、动态groom格式 | 可能双重位移/头发冻结/逐帧跳动 | 评审interface_draft选项、明确制作侧与接入侧责任 |
| 跨缝crop第一版策略 | 影响新边缘样本覆盖与复杂度 | 建议先明确拒绝计数，需覆盖时另实现同变换周期crop |
| 点对误差/数量/部位门槛、正常回归门槛 | 不能宣布部署可用 | 下游确定单位/容忍度，预测前冻结 |
| 独立motion来源与划分 | 同片段邻帧不是独立泛化 | 独立验证来源另行准备，F04及同源外观/遮挡者限制继续保持 |

本轮停止于评审。未自动继续T1制作、T2生产实现、更多帧导出、检测/推理、场景扩展或训练。
