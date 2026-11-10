# APD 6526 单角色 SMPL-X 身体适配试验

日期：2026-10-08。状态：**身体参数与数值回读完成；动态观感待用户评审；本轮停止。** 全部产物为 `integration_only`。

本轮使用用户已观看并认为值得适配的 APD 6526，冻结一个3秒片段和 `cycling_M01_v2`。用户新的评审决定优先于上一轮报告：不再以两个完整周期、精确踏板圆周、握把或足端解剖定义全部确认作为拟合阻断。未知项通过显式适配假设处理，没有升级为官方事实。

三个交付结论：

1. **拟合与数值链通过。** 完成5帧探针、一次76帧完整拟合、一次针对左臂跳变的修正。固定shape、拓扑、有限值、参数→身体回读、现有 R→138点均通过；这不是对真实人物的重建精度认证。
2. **动作观感仍待用户观看。** 下肢屈伸保留，逐帧双方向图未见明显反膝、翻腿或严重穿体；左臂仍有较快抬落，不能把数值通过宣布为“无明显突跳”已人工验收。
3. **有条件建议下一批衣物/groom接入。** 先评审同一身体片段，重点源35–48帧左臂；若接受该适配观感，可只推进这个角色/片段的既有绑定。若不接受，先另批处理身体动作，不能用衣物遮住问题。本轮没有继续穿衣或双路渲染。

直接打开：[评审页](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/index.html)、[逐帧浏览器对照](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/fit_attempt_2/preview/index.html)、[轻量MP4](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/fit_attempt_2/preview/comparison.mp4)。权威参数是 [fit_attempt_2/parameters.npz](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/fit_attempt_2/parameters.npz)，机器交接见 [delivery_manifest.json](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/delivery_manifest.json)。

## 1. 冻结范围、资源与保护

已读取 APD 审计及字段/判断清单、6526原数据与统计/时间表、A2方案、A1高分辨率框和新增的 [序列索引报告](../../data/SEQUENCE_INDEX_A1_20261008.md)及接口。没有发现适用 AGENTS.md。没有重新下载APD或重跑上一轮审计，没有接管A1实现。

| 冻结项 | 实际内容 |
| --- | --- |
| 来源 | APD v1.0，`dataset/6526.json`，原官方split=train；源人物真实身份unknown |
| 原帧 | **`[30,106)`**，即30–105，共76个连续原始观察时刻 |
| 源相对时间 | **1.200018–4.200018秒**，按最接近用户建议1.2/4.2秒的原观察时刻选取；首末跨度3.000000秒 |
| 时间处理 | 原速、原索引、无重采样、无循环；保留整数时间戳。微秒单位仍为上一轮基于25 Hz的推断 |
| 目标角色 | `cycling_M01_v2`，身份家族`cycling_M01`，male，固定原manifest的30维betas，不估计源人物shape |
| 模型 | 既有 `cycling_commuter_pilot_20260930/models/SMPLX_MALE.npz`；10475顶点、20908三角面，顺序不改。精确语义发行号unknown，文件引用、大小/mtime及载入buffer规格保存 |
| 探针 | 原帧30、49、68、86、105，5帧；Adam180步 |
| 完整预算 | 76帧500步一次；只允许针对明确问题再500步一次；单次墙钟上限1200秒，探针预测若超过1800秒则停止 |
| 执行设备 | CPU，2线程；无GPU计算，`gpu_uuid=null` |

上述范围在任何优化前写入 [frozen_manifest.json](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/frozen_manifest.json) 和 [time_mapping.csv](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/time_mapping.csv)。同一片段没有被替换，也没有参数网格搜索。所有1064个选中3D关节点观察有限；本段没有需要补齐的缺失点。

计算前检查两张RTX3090：GPU0约18 MiB、GPU1约402 MiB，两者利用率当时0%；已有Blender PID43491占约122 MiB显存，未接触该进程。本轮固定CPU进行姿态优化、SMPL-X求值、绘图/编码。见 [GPU记录](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/gpu_before.txt)、[进程记录](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/processes_before.txt)。

TODO、README、A1框/序列文件与测试、既有APD/A2报告、角色manifest、原数据、模型和R的大小/mtime前后相同，见 [保护记录](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/protected_after.json)。没有计算SHA256；没有修改公共生产代码、Dataset、监督规则、fixture、配置、外部pano-stitch或已有产物，没有commit或移入datasets。

## 2. 坐标、尺度与root契约

APD的完整坐标/单位/足端定义仍未获官方确认。本轮只建立**合成适配用**的全段统一变换，不声称恢复真实相机、世界坐标或root轨迹。

对于原点列向量，统一轴矩阵为：

```text
A = [ 0.7079027233, -0.7063099421, 0 ]
    [ 0,             0,            1 ]
    [-0.7063099421, -0.7079027233, 0 ]
det(A) = +1
```

假设原`+z`为向上，映射到模型`+Y`；全段左右髋/肩差的中位方向定义模型`+X`，与向上叉乘定义模型`+Z`。本段膝/手的前向关系用于探针合理性检查，没有每帧重新定向。全段尺度 **s=1.2016299166模型米/原单位**，取双侧上臂、大腿、肩→髋六段“目标rest长度/源中位长度”比的中位数；足端段不参与尺度估计。六个比值约1.070–1.309，显示源估计比例与固定目标不同，不能靠逐帧缩放消除。

行向量写法：

```text
H_t = (APD_lhip_t + APD_rhip_t) / 2
target_t = (APD_t - H_t) @ A.T * s + c
c = [-0.0008379724, -0.1016405225, -0.0232656486] 模型米
```

`c`是目标rest姿态中相对pelvis的髋中点偏移，全段固定。原 `H_t` 和原APD点保存于 `observations.npz`，被移除的平移可追溯；没有叠加 `head_smoothed`。**输出pelvis固定在模型原点**，`transl`恒为负的shaped-rest pelvis：`[-0.0009769164, 0.3770247698, -0.0096359504]` 米，位移残差为零。`global_orient`是拟合的身体朝向残差，和局部姿态一起连续约束；它不是每帧独立的观测对齐，也没有场景yaw。

身体仍保留由关节旋转引起的躯干倾斜与骨盆姿态。未来场景平移/朝向必须由接入侧只应用一次，本轮不交付场景轨迹。没有任何逐帧脚底或min-Y贴地；身体最低点随姿态变化，不能据此反算真实地面。完整数值见 [alignment.json](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/alignment.json)。

## 3. 具名约束与优化

关节编号来自本地SMPL-X标准关节表，映射保存于 [joint_mapping.json](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/joint_mapping.json)。左右不交换。

| APD | SMPL-X对应 | 权重 / 状态 |
| --- | --- | --- |
| head | joint15加该关节局部Y方向0.08米虚拟偏移 | 0.3，头部landmark近似；不是精确头顶GT |
| thorax | neck，12 | 0.5，胸颈位置近似 |
| l/r shoulder | 16 / 17 | 各1.0 |
| l/r elbow | 18 / 19 | 各0.8 |
| l/r wrist | 20 / 21 | 各0.45 |
| l/r hip | 1 / 2 | 各1.0；允许固定目标比例与源不同 |
| l/r knee | 4 / 5 | 各1.0 |
| l/r foot | ankle，7 / 8 | 各0.3，弱足端近似；不当精确踝、趾或踏板真值 |

只使用3D有限值掩码，没有把 `pose2d` 第三分量用作置信度。选段原数据无缺失，既未零填观测也未插值补帧。损失为向量残差的pseudo-Huber（δ=0.05模型米）、四肢/躯干方向约束、固定初始化的弱姿态先验，以及按实际时间差归一化的时间正则；没有下载学习型姿态先验。

多轴关节用6D连续表示，经Gram–Schmidt构造旋转矩阵；时间约束作用在矩阵和位置上，不对跨界轴角直接作平滑。膝关节为正向局部X铰链，范围0.02–2.5 rad，防止以反向屈膝吸收噪声。导出标准轴角，同时保留矩阵作回读检查。只有12个多轴关节和两个膝铰链参与估计；骨骼twist仍是欠约束、受先验影响的量。

手指45维/侧固定为flat-neutral；jaw、eyes、expression均零；neck/head、ankle/foot、wrist的局部旋转固定identity，但会随父骨骼运动。没有精确握把、踏板或自行车接触约束。字段的fixed/estimated/missing含义见 [parameter_semantics.json](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/parameter_semantics.json)。

拟合使用目标固定shape的原生关节运动学，优化阶段不必每步计算所有表面顶点；之后通过原生SMPL-X LBS生成完整身体，逐帧验证两者关节一致。因此这是SMPL-X姿态参数优化，**不是DenseKP训练、学习新模型或用14点插值伪造人体/138点**。

## 4. 一次首版与一次修正

5帧探针的参数/映射/尺度和双方向身体图通过基本检查，平均关节残差约37.55 mm；稀疏探针不能证明相邻帧连续性。以探针实测成本保守估算完整拟合约157.07秒，低于30分钟门槛，遂执行冻结76帧。探针决定在完整优化前保存。

首版存在明确失败：源35→36帧左肘局部旋转约29.64°，手腕位移约0.288米；45→46帧手部表面最大位移约0.460米。查看连续帧图可见左臂跟随原观测突然抬起/落下。首版完整参数、网格、数值日志与预览保留在 `fit_attempt_1`，没有换段或覆盖。

唯一修正的决定先保存为 [correction_config.json](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/correction_config.json)。保持shape、尺度、轴矩阵、映射、root和76帧不变，从首版参数初始化：旋转速度权重0.001→0.01，加速度0.008→0.08，另加映射关节点位置速度0.05、加速度0.5。目标是抑制孤立跳变，同时保留下肢数据与方向约束。又运行500步后停止，**没有第三次拟合或额外调参搜索**。

| 指标 | 首版 | 修正版（交付评审） |
| --- | --- | --- |
| 14点残差均值 | 36.17 mm | **39.07 mm** |
| 残差P95 / 最大 | 102.29 / 124.72 mm | **104.01 / 141.13 mm** |
| 相邻局部旋转最大 | 29.64° | **13.42°** |
| 相邻表面顶点位移均值 / P95 | 25.20 / 63.01 mm | **14.89 / 34.52 mm** |
| 相邻表面顶点位移最大 | 460.44 mm | **231.55 mm**，源45→46帧，vertex5060 |
| 优化步数 | 500 | 500 |
| 优化耗时 / 含加载与导出进程内部耗时 | 3.218 / 4.323 s | 3.299 / 4.398 s |

残差使用本轮估计变换后的模型米制坐标；它不是APD真实3D误差。髋部因源比例/landmark差异保持约9–11厘米量级偏差，没有为压低残差逐帧改shape。修正增加了拟合残差，换取更平缓的运动，权衡已公开。

**剩余问题没有被隐藏：** 左手腕最大单步位移仍0.1431米，右手腕0.0753米，45→46附近左臂仍快摆。13.42°小于脚本25°提示线不等于主观“无突跳”通过。衣物接入前仍需用户按真实时间观看35–48帧及整段。

## 5. 全段回读与变化保留

独立校验器从权威NPZ重新读取参数，对76帧再次原生SMPL-X求值，与缓存及R结果比较；没有新优化。结果见 [readback/validation.json](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/readback/validation.json)、[逐帧CSV](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/readback/per_frame.csv) 和 [连续诊断图](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/readback/temporal_diagnostics.png)。

- `betas[30]`逐值等于角色manifest；`global_orient[76,3]`、`body_pose[76,63]`、`transl[76,3]`及固定手脸参数全部有限。模型faces与角色参考faces逐值一致，10475顶点顺序保持。
- 参数重新求值与 `vertices.npy[76,10475,3]` 最大差 **0**；重新求值后用既有 **`R[138,10475] @ body`** 与 `densekp_3d.npy[76,138,3]` 最大差 **0**（本环境dtype/实现下），容差分别2e-5和2e-6米。没有对R重新归一化或改权重。
- 轴角还原矩阵最大差约1.55e-7，正交性最大误差约3.58e-7；原生关节与快速运动学在浮点容差内一致。
- 四条大腿/小腿长度的最大CV约5.64e-7，为固定骨架浮点误差量级。pelvis原点误差0，平移全段恒定。身体min-Y约 **−1.0545至−0.9494米**，作为诊断保留，没有逐帧拉回地面。

| 变化保留诊断 | 左 | 右 |
| --- | --- | --- |
| APD膝角范围 | 43.65° | 32.22° |
| 拟合膝角范围 | **50.28°** | **35.16°** |
| 膝角与原序列相关系数 | **0.9830** | **0.9389** |
| 原弱足端垂直范围（统一尺度后） | 0.3044 m | 0.3048 m |
| 目标踝垂直范围 | **0.2365 m** | **0.1376 m** |

下肢没有被时间正则压成静态；膝角交替关系和曲线变化保留。但足端幅度被约束和平滑改变，**右侧缩小更明显**，不宣称精确保留脚踏运动。该指标只诊断适配结果，不再作为“两周期认证”门槛。

本轮查看修正版全部76帧的前/侧方向逐帧对照图，未观察到明显反膝、翻腿或严重人体穿插；手靠近大腿、开放手指和中性脚部方向仍可见。没有做全三角形自交穷尽检测，也没有实际在浏览器/播放器连续观看成片。**动态人工验收明确为“待用户观看”**，不能拿数值检查或逐帧图替代。

## 6. 预览、来源与交付契约

HTML及MP4同时显示四格：APD前视/目标身体前视、APD侧视/目标身体侧视。前后使用同一时间表；每一版内原始/目标、所有帧、两个方向共用固定画面尺度和垂直中心，没有各自自动缩放。原APD只是应用已声明的统一变换并移除行进平移，未平滑原观测。目标面采用CPU平面着色三角形绘图，关节点在表面上方绘出供诊断；无材质纹理、布料、毛发、Blender或场景相机。

两版MP4都是1280×760、76帧；time_base=`1/1000000`，逐帧PTS与实际源时间差**全部一致，最大误差0**。PTS首末0–3秒，容器长度3.04秒包含末帧40 ms展示时长，不代表新增观察时刻。HTML按原时间差播放，终点停止，支持逐帧定位，不循环。网页脚本全部时刻映射经过执行校验；不是实际浏览器人工观感验收。

默认编码器缺少`fps_mode`，系统编码器的旧concat接口又不支持逐文件`option`；两次失败日志保留。最终使用现有系统ffmpeg的`settb/setpts`赋精确微秒PTS编码成功，没有安装/升级依赖，也没有重新绘图覆盖失败现场。

来源仍为 [APD Zenodo记录](https://doi.org/10.5281/zenodo.5724486)，CC BY4.0；署名、裁段、坐标估计、平移移除、姿态拟合与平滑修改见 [ATTRIBUTION.md](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/ATTRIBUTION.md)。本地SMPL-X沿用独立的 [非商业研究模型许可](https://smpl-x.is.tue.mpg.de/modellicense.html)；目标shape沿原M01父manifest的BEDLAM来源及 [BEDLAM研究许可](https://bedlam.is.tuebingen.mpg.de/license.html)。本轮不重发模型/纹理，不把APD许可泛化到其它资产，不宣称目标就是APD源人物。

| 交付 | 用途 |
| --- | --- |
| `frozen_manifest.json`、`time_mapping.csv` | 选段/角色/预算在先；资产帧0–75↔源帧30–105↔整数时间 |
| `model_reference.json`、`alignment.json`、`joint_mapping.json` | 本地模型、固定shape、坐标/尺度/近似语义 |
| `parameter_semantics.json`、`fit_attempt_2/parameters.npz` | **参数是权威产物**，明确fixed/estimated/unknown及root所有权 |
| `fit_attempt_2/{vertices,densekp_3d,joints,mapped_joints}.npy` | 回读与预览所需派生缓存；faces仅根目录保存一份 |
| `probe/`、`fit_attempt_1/`、`correction_config.json` | 有界探针、失败首版和唯一修正记录，不能伪装为多个独立来源 |
| `readback/`、`preview_validation.json`、`execution_record.json` | 数值、全段变化、时间一致性、资源与限制 |
| `fit_attempt_2/preview/`、根目录`index.html` | 原始/拟合动态对照与静态逐帧图 |

所有上述文件继承根交付清单的`integration_only`范围。APD来源组仍为 `APD_5724486_v1_identity_session_unknown`，官方train字段原样保留；同片段邻段、循环、变速、换角色衍生、两路相机将来均不能成为独立验证来源。角色家族`cycling_M01`也单独保留。

当前只有身体资产，没有DS观测、mask或packed rows，**不伪造A1的episode/sequence_index成品**。未来接入直接复用A1接口，逐帧绑定同一asset_frame，保留原始源tick及单位推断声明；本段时间间隔略不均匀，不能强制假定固定`expected_tick_step`。本轮未修改或执行A1公共接口。

## 7. 实际资源与停止点

探针180步优化1.150秒，进程内部总计2.787秒，峰值RSS684360 KiB；完整与修正各500步见上表，峰值RSS约684600 KiB（约669 MiB）。总计**1180次姿态参数更新、优化约7.666秒**，无GPU计算；逐帧网格原生回读校验约1.544秒。这些数字不含人工阅读、脚本编写及全部预览编码墙钟，也不能推作衣物/渲染成本。完整输出约45.5 MB，包含两版与探针，低于冻结的80 MB缓存/交付预算；没有复制整套角色资产。

新增脚本只有：

- [fit_apd_6526_pilot.py](../../../tools/experiments/fit_apd_6526_pilot.py)：冻结输入的探针/首版/唯一修正，输出目录必须不存在。
- [preview_apd_6526_pilot.py](../../../tools/visualize/preview_apd_6526_pilot.py)：已有身体的CPU双方向预览与精确PTS编码。
- [validate_apd_6526_pilot.py](../../../tools/validate/validate_apd_6526_pilot.py)：权威参数全76帧独立回读及时间变化诊断，无优化。

没有运行不相关生产测试。独立回读、JSON/CSV/HTML/MP4映射、脚本语法、文件引用和原有文件保护检查通过，详见 [执行记录](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/execution_record.json) 与 [交付检查](../../../outputs/experiments/apd_6526_smplx_pilot_20261008/delivery_validation.json)。

**评审点：** 请以修正版整段观感及左臂35–48帧为主。若接受，则下一批仅需在同一76帧身体上评审既有简化衣物与随头groom的绑定、时间一致性和穿插；轻微局部穿模可接受，明显翻折/脱体/闪烁仍拒收。若左臂快摆或足端幅度变化不可接受，应另立身体修正批次；本轮优化预算已用完，不再运行。当前不接衣物/groom、不叠道路轨迹、不启动双路/DS渲染或训练。
