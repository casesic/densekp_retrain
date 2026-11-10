# BEDLAM2 原配行走：眼睛与头发补充

日期：2026-10-09。用户已接受上一版原身体＋原动作＋原配衣物，随后授权补上眼睛和头发。本轮沿用 `us_2495_L/2200 + nt_belle_007` 的91个原时刻，原30 Hz、帧0–90，首末跨度3秒；身体、shape、动作、root、衣物、UV和时间表不变。

交付入口：[连续播放页](../../../outputs/experiments/bedlam_native_walk_eyes_hair_a2_20261009/preview.html)、[MP4](../../../outputs/experiments/bedlam_native_walk_eyes_hair_a2_20261009/native_walk_two_views.mp4)、[第45帧头部细节](../../../outputs/experiments/bedlam_native_walk_eyes_hair_a2_20261009/head_detail_frame045.png)、[新版manifest](../../../outputs/experiments/bedlam_native_walk_eyes_hair_a2_20261009/manifest.json)。**新增眼睛/头发的动态观感待用户观看决定**，不把先前的身体/衣物接受扩大为新版已接受。

## 新增内容和来源

原locked-head身体已有两套独立眼球，各546个顶点，合计2176个面。使用同一原模型reference的 `left_eye_smplhf` / `right_eye_smplhf` 权重确认组件和面，没有外加眼球、改变视线pose或改变网格。程序化材质提供眼白、棕色虹膜、深色瞳孔；以原眼球rest坐标作为固定属性，随原眼球几何自然运动。材质方案沿用本项目已有原生眼球着色思路，只在独立预览适配代码中实现，保持原身体哑光材质和衣物材质。

头发选择本地已有的 BEDLAM2 `vinefx_groom_18-1_short-bob-F.abc`（短波波头），使用已有导出审计，不下载、不解整包。保留84,069条曲线、1,890,570个点和原点序、radius；原导出厘米转换到米。来源是官方BEDLAM2发型资源，**不是已核实的us_2495_L原演员发型配对**。新增棕色眼睛与短波波头属于本次外观组合，不增加独立身份。

许可沿用 `assets_external/bedlam2_pilot/LICENSE.txt`、`b2_groom_abc.tar.gz.source.json` 和原BEDLAM2/SMPL-X来源；不把代码许可推广为资产许可，不作公开再分发授权声明。

## 固定绑定与变换

groom的原作者reference与本人物都是本地locked-head neutral模板；先核对reference faces与已接受身体faces相同。使用原头皮顶点对应，在第0帧建立**一次**reference到目标头部的仿射适配，三个奇异值约0.9881、0.9841、0.9763。这个小幅缩放只影响派生头发，不改变身体，也不是逐帧缩放。

以23个原模型纯head权重顶点为锚，从已接受frame0头部到每个asset帧求刚性变换；求解使用float64，逐帧检查正交性。最大锚点残差0.601mm，头皮核对区全段最大残差1.474mm。这是头发附着近似误差，不是动作重拟合，也不声称有物理摆发。没有按帧重筛有效毛发、碰撞迭代、删毛或修改头型。

`groom/points.npy` 位于已接受身体第0帧的模型坐标；`head_transforms.npy[n]` 将其移到第n帧。展示或场景使用 `scene @ head[n]`；身体与衣物使用同一个 `scene`。源行进和头部旋转已经包含在这些头部矩阵中，不再叠source root。展示沿用原固定轴变换 `C(x,y,z)=(x,-z,y)`，无逐帧贴地、跟随相机或动态缩放。

完整groom只保存一份约30MB参考点/radius/offsets，91份小型4×4矩阵逐帧消费，不保存91套毛发缓存。`groom_manifest.json`、`appearance.json` 和 `binding_validation.json` 记录所有参数及来源。

## 检查与预览

复用原两视角相机、固定全段画面范围、灯光、分辨率512×512/视角、AgX、Cycles64 samples、不降噪、seed29。5个探针帧为0、22、45、68、90，额外一张第45帧头部近景用于看眼睛和发根；这张近景是静态检查，不替代动态验收。探针检查可见双眼材质、头发轮廓和原身体/衣物配对，没有发现明确脱体；64 samples仍有可见毛发采样颗粒，未展开提采样/降噪实验。

探针阶段实测30.282秒，完整阶段据此估计501.0秒（约8.4分钟，含余量），在30分钟总渲染预算内。复用10张原探针图，补齐172张；完整实际成本以 `execution_record.json` 为准。预览视频保留91帧和30 Hz，显示asset/body/cache帧、原仿真帧以及原时刻，无补帧、循环、变速。

全91帧全部groom点经过有限值和共同变换检查，曲线/点数固定；身体和衣物直接消费上一版不可变缓存，并在渲染器更新时回读核对。R138仍从原身体回读，数值不因新增材质/头发而变化；**locked-head模板上的R语义兼容状态仍pending**，本轮不将它提升为监督就绪。

技术检查不能代替动态人眼判断。请观看新版中眼睛是否自然、头发是否正确跟随且无明显脱体或几何闪烁。轻微局部穿插不自动否决，原动作与衣物的已接受状态保持；本轮仅新增外观待选择。

## 资源、失败与交接

即时GPU检查显示GPU0有三个活动任务，因此选择GPU1：UUID `GPU-6f046d56-acbe-d49b-774a-9d463e768cac`，计算前407MiB/24576MiB、0%利用率，已有低占用idle进程43491；没有停止或修改其它任务。CPU前可用内存约135GiB、磁盘约12TiB。沙箱内驱动访问失败后使用授权的只读设备检查，不将其视为GPU损坏。

一次CPU准备在保存任何groom前因float32 SVD正交误差1.19e-7超过1e-10门槛而失败；改为float64小矩阵求解后通过，保留 `preparation_attempt_1_failure.json`。这不是修改身体或选段。具体计算、存储、全段检查和保护快照见 `execution_record.json`、`full_geometry_validation.json`、`delivery_validation.json`。

实现只新增独立外观准备/眼睛材质代码，并为A2自己的studio渲染和打包脚本增加可选 `--appearance-overlay`，历史不传该参数的默认路径不变；未修改公共groom/material模块、A1接口、TODO、原fixture、APD或上一版接受缓存/视频。未下载、安装、训练、推理、计算SHA256或提交commit。

新版manifest引用原身体/衣物，不复制模型。A1以后需读取显式groom_reference与head_transforms路径、眼部材质映射，同时保留原garment face-corner UV/贴图及真实时间契约；不能因这里新增头发就宣称当前公共消费者已支持完整组合。仍标记 `integration_only`、`A1_schema_effective=false`、`public_consumer_tested=false`；继承KIT425/walking_medium05来源组和BEDLAM2 body family，不改变数据划分。停在新版外观评审点，不运行道路、DS/ERP或正式数据生成。

## 完成时实测

探针阶段30.282秒，完整补齐阶段461.701秒，渲染阶段累计491.982秒；编码1.642秒。CPU绑定准备0.448秒、全帧全部毛发点验证21.166秒。保存183张诊断图（双视角182张及头部近景1张）、91帧MP4；输出大小快照60,149,180 B（57.36 MiB）。旧身体/衣物/时间表、旧视频和TODO保护检查通过。新增眼睛与头发仍待用户动态观看确认，未运行公共场景。
