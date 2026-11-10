# APD 6526：同一角色、76 帧衣物与 groom 动态接入

批次日期：2026-10-08；交付收尾：2026-10-09。全部为 **integration_only**。状态：**数值与绑定流程通过，穿衣动态观感待用户连续观看；停止于本阶段评审点。**

直接查看：[HTML 双方向播放与逐帧页](../../../outputs/experiments/apd_6526_dressed_pilot_20261008/preview/index.html)、[轻量 MP4](../../../outputs/experiments/apd_6526_dressed_pilot_20261008/preview/comparison.mp4)、[机器交付清单](../../../outputs/experiments/apd_6526_dressed_pilot_20261008/delivery_manifest.json)。

三个交付结论：

1. **同一身体上的数值与绑定流程通过。** 76 帧原生参数回读、身体→R138、固定衣物面集、固定 groom 曲线集合及头部变换检查完成。没有改身体、删衣物面或逐帧重筛头发。局部穿插和小面积面法向翻转保留并记录，不宣称零相交。
2. **穿衣动态观感待用户确认。** 查看了覆盖全部 76 帧的正/侧联系图，未见明显大面积翻折、拉飞、脱体或头发滞留；没有实际在浏览器/播放器按真实时间连续观看。脚本播放检查和静态逐帧检查均不冒充动态人工验收。8 samples 预览有可见采样颗粒。
3. **具备下一批有界双路接入试验的资产基础，尚未完成双路链验收。** 先确认本段穿衣播放观感；后续仍需显式场景变换归属、逐帧渲染消费者、同步两相机和真实图像/标签/A1 旁表回读。本轮没有自动进入该阶段。

## 1. 冻结输入与保护

用户已接受身体观感，包括左臂快摆；这更新了旧身体报告中“待评审”的状态，但没有改写历史报告。身体权威仍为：

`outputs/experiments/apd_6526_smplx_pilot_20261008/fit_attempt_2/parameters.npz`

| 项目 | 本轮固定内容 |
| --- | --- |
| 角色 | cycling_M01_v2，成人 male；角色家族 cycling_M01；原有 30 维 betas |
| 来源 | APD6526 原帧 **[30,106)**，asset_frame **[0,76)** |
| 时间 | 源相对时间 1.200018–4.200018 s；首末跨度 3 s；原速度、原索引、不循环、不重采样 |
| 身体 | 10475 顶点、20908 面，原模型和顶点顺序；shape、pose、root、手脸参数完全不调整 |
| 衣物 | 原 `reference/simple_clothing_v1`，2612 顶点、5018 面，sport 贴身衣裤 |
| 头发 | 原 `reference/groom_v1`，short-casual；原固定有效集合 |
| 探针 | asset 0、6、16、38、75，即源 30、36、46、68、105；涵盖首末及源 35–48 快摆区域 |
| 范围 | 衣物/随头 groom、数值检查、低成本角色预览；没有自行车、道路轨迹或 DS/ERP |

先读取身体报告及 delivery/semantics/alignment/time/readback、A2 方案、A1 序列索引报告，核对角色和衣物/groom manifest、实际绑定数组与消费者。未发现适用 AGENTS.md。检查 Git 状态，保留已有 A1/A2、TODO、README、fixture 和历史资产。保护记录采用文件大小与 mtime，不计算 SHA256。最终检查发现 `docs/TODO.md` 相对批次开始的大小从 1116 B 变为 1122 B、mtime 改变；本轮没有写入它，变化来源未核实，当前内容原样保留，不回退。其余已记录文件前后不变，见 [protected_after.json](../../../outputs/experiments/apd_6526_dressed_pilot_20261008/protected_after.json)。没有修改外部 pano-stitch、公共生产模块、A1 接口、Dataset、NPZ 契约、框计算或监督规则；没有 commit。

计算前保存 [frozen_recipe.json](../../../outputs/experiments/apd_6526_dressed_pilot_20261008/frozen_recipe.json)，探针后、完整计算前保存 [probe_decision.json](../../../outputs/experiments/apd_6526_dressed_pilot_20261008/probe_decision.json)。没有按结果换段或挑选其它角色。

## 2. 衣物：复用固定绑定，不重新裁面

实际复用 `clothing.shells.vertex_normals`、原 `body_barycentric_indices/weights`、`clothing_faces`，以及 `rendering.blender.materials.add_clothing_materials`。没有逐帧调用可能按姿态过滤退化面的 `clipped_shell` 或重新运行制衣流程。

对固定顶点绑定 `(ids_i,w_i)`：

```text
surface_i(t) = sum_k w_ik * body[ids_ik](t)
normal_i(t)  = normalize(sum_k w_ik * body_vertex_normal[ids_ik](t))
cloth_i(t)   = surface_i(t) + 0.004 * normal_i(t)    # 模型米
```

参考身体与当前本地模型按固定目标 shape 求出的零姿态 **逐值一致**；参考绑定的重心点一致；上述配方在参考姿态重新生成衣物，与原衣物顶点最大误差 **0**。因此沿用的是既有参考的 4 mm 壳配方，没有新增衣型或用逐帧 shape 变化吸收问题。

衣物始终保留 2612 顶点和 5018 面，顶点顺序、索引、面材质分配及连续材料坐标不变。该资产没有独立衣物 UV 文件；已有 shader 使用固定 `clothing_rest_vertices` 的 `GarmentRest` 属性划分上衣、短裤和条带，不能虚称验证了不存在的衣物 UV。皮肤仍通过原 UV 和 material_v6，眼睛/头皮局部材质沿用角色 manifest。

**本轮绑定/局部偏移修正次数为 0。** 原资产不覆盖；没有删面遮住穿插。允许的唯一局部修正额度没有使用，也没有身体更新。

## 3. Groom：固定整束随头刚性运动

现有 groom 已包含头皮重心绑定、局部 frame、静态 collision_correction 和 active mask。公开的 `groom_transfer.transfer_groom` 当前按第 0 帧文件夹做头皮 frame 转移；本轮不重写该静态入口。对这段手脸/头部局部姿态固定、仅随躯干运动的短发，采用用户允许的更小方案：**原已修正 reference points 整束随 SMPL-X head joint15 运动**。

设 `Q_head(t)` 为从冻结轴角沿模型父链累积的全局头旋转，`J_head(t)` 为原生模型当前 head 位置，`J_head(rest)` 为同 shape、pelvis 居中的参考位置：

```text
H(t) = [ Q_head(t), J_head(t) - Q_head(t) @ J_head(rest) ]
groom_model(t) = H(t) @ groom_reference
```

头部链已经包含身体 global_orient；**不能再叠一次身体 global/root 变换**。原固定 pelvis 平移已反映在当前/参考关节点中，不叠 APD 头部路径。展示时身体和衣物使用 `C(x,y,z)=(x,-z,y)`；头发对象使用 `C @ H(t)`，各只应用一次。展示相机全段固定，目标 `[0,0,-0.1]`、正/侧方向、正交尺度 2.3 m；没有逐帧 min-Y 贴地或自动缩放。

复用 `rendering.blender.groom.add_groom` 导入原 ABC 一次、应用原静态有效集合一次。原 63656 条曲线有 4 条已被历史 active mask 排除，本轮始终渲染 **63652 条曲线、1591300 个点**；原 points 文件仍含 1591400 点。曲线/点顺序、radius、各对象曲线数量和有效集合固定；没有按每帧碰撞结果重新裁剪。Blender 每帧只更新头发对象矩阵，检查点坐标本身没有被重复变形。

刚性跟随与当前身体头皮重心锚点的全段最大偏差 **4.284 mm**，说明存在混合蒙皮/姿态形变与整束刚性的局部差异，并非完全相同的头皮表面形变。这是明确的第一阶段近似，不是毛发惯性、摆发或物理仿真。没有保存 76 份巨大头发数组；只保存 76 个 4×4 矩阵，引用原 groom。

## 4. 全段验收、局部问题与预览

数值依据：[全段回读](../../../outputs/experiments/apd_6526_dressed_pilot_20261008/full/validation.json)、[逐帧绑定记录](../../../outputs/experiments/apd_6526_dressed_pilot_20261008/full/per_frame.json)、[补充诊断](../../../outputs/experiments/apd_6526_dressed_pilot_20261008/diagnostics/summary.json)。

| 检查 | 本轮实际结果 |
| --- | --- |
| 原生参数→身体缓存，76 帧 | 最大差 **2.3842e-7 m**，在 2e-5 m 容差内；逐帧求值与历史小 batch 的浮点差异，没有写回或替换身体 |
| 原生身体→既有 R[138,10475] | 与原 138 缓存最大差 **1.1921e-7 m**，在 2e-6 m 容差内；不从 APD14 插值 |
| Blender 实际身体顶点 | 152 次视图记录对应的身体缓存坐标回读逐值一致；无 body modifier |
| 参考 body / cloth 配方 | 最大误差均 0；shape 和拓扑一致 |
| 衣物/groom 有限值 | 76 帧通过；全部源 groom 点逐帧变换检查有限，不只验少数根部 |
| 探针/完整对应帧 | 衣物顶点和头部矩阵逐值相等，允许复用探针图片 |
| 衣物偏移 | 保持 4 mm；相对冻结缓存的最大浮点偏差约 1.09e-7 m |
| 衣物相邻位移 | 最大约 67.41 mm；相对重心面的局部 offset 相邻变化最大约 7.19 mm，无超过壳偏移量级的额外拉飞 |
| 拓扑/时间/root/head | 固定面集、固定曲线集合；同一 asset_frame、同一原时间表；无逐帧贴地或重复 head/root |

**保留的小范围问题：**

- 每帧最多 26 个衣物小面与其绑定基面法向负向对齐，最大总面积占比 **0.062576%**，最大单面约 **95.43 mm²**。没有退化为零面积的衣物面，也没有删掉这些面。该检查诊断局部折转，不等价于屏幕上可见的大面积翻折。
- 衣物顶点近表面的最小有符号诊断值 **−6.475 mm**，出现在源帧 51；每帧低于 −1 mm 的顶点最多 23/2612。探针最深约 4 mm，完整序列出现更深局部值，未隐去。
- 固定选择 995 条有效曲线，每条根/中/末 3 点，共 **2985 点/帧** 做毛发近表面诊断；最小值 **−3.272 mm**，源帧 70；低于 −1 mm 最多 43 点，没有低于 −5 mm 的样本。抽样不能保证其它点不穿插。
- 以上使用“24 个近邻面心候选中的最近面及法向”近似，不是完整内外判定或全三角形碰撞证明。身体本身已接受的动作限制保持原样。

问题定位图保存在 [source46](../../../outputs/experiments/apd_6526_dressed_pilot_20261008/diagnostics/issue_source_46.png)、[source51](../../../outputs/experiments/apd_6526_dressed_pilot_20261008/diagnostics/issue_source_51.png)、[source70](../../../outputs/experiments/apd_6526_dressed_pilot_20261008/diagnostics/issue_source_70.png)。红圈标记衣物负向法线小面的投影，可能位于背面，不应误读为可见破洞；具体面 ID 保存在逐帧诊断。

查看全 76 帧正/侧联系图，尤其源 35–48 快摆区域，未见明显大面积翻折、拉飞或脱体，短发轮廓随头变化。**仍须用户按真实时间观看**袖口/腋下、躯干条带和头发：固定拓扑不自动证明没有视觉闪烁；静图也不能判断瞬态观感。低 samples 的皮肤/毛发采样颗粒是预览质量限制，不能据此声称几何闪烁已经人工排除。

MP4 为 **1024×640、76 帧、time_base=1/1000000**。原图读取虽以 25 Hz 初始化，最终逐帧 PTS 使用原始微秒差明确设置，误差 **0**；不是静默改成均匀时间。首末 PTS 0–3 s，容器长 3.04 s，末尾 40 ms 是末帧展示，不是新增观察。原 CSV 原样复制，包含 body/cloth/groom 共用的 asset/source/time 映射。

HTML 提供 MP4 与按源时间的图片播放/逐帧定位，不循环。脚本边界检查发现过浮点秒换算导致精确时刻落到前帧的问题，已仅在本轮网页改为整数微秒比较；失败记录保留。最终 76 个精确时刻和 75 个前一微秒位置检查通过。这是模拟 DOM 的程序检查，**不是实际浏览器人工播放验收**。

## 5. 资源、失败保留和交付范围

计算前 GPU0 约 18 MiB、利用率 0%；选择 RTX3090，UUID **GPU-d481acc7-106f-29e5-1298-8c0d398261b2**。GPU1 上已有 Blender PID43491，未停止或占用其设备。实际本轮 Blender PID120092 出现在 GPU0，抽查使用约 1716 MiB；这不是连续采集的显存峰值。CPU 几何检查 2 线程，Blender CPU 配置 4 线程。

初次 Blender 在沙箱内未枚举到 CUDA，未产生图像；失败目录和日志保留。取得沙箱外执行许可后，在指定单一 UUID 上成功运行既有 Blender5.1.1，没有安装依赖或下载模型。探针测得双方向每时刻约 3.9–5.3 s；按最慢帧×76×1.5、加初始化及 30 s 余量，完整预算预测 **642.58 s**，低于 1800 s 门槛。

| 计算 | 实测 |
| --- | --- |
| 5 帧 CPU 参数/绑定探针 | 1.53 s |
| 全 76 帧 CPU 参数/绑定 | 9.17 s，峰值 RSS 715080 KiB |
| 全段独立补充诊断 | 3.57 s |
| 探针：5 帧×2 方向 | 36.16 s，含约 15.20 s 初始化 |
| 完整阶段：补 71 帧×2 方向 | **302.06 s**，含约 14.29 s 初始化；复用同配方探针 5 帧 |
| 两次成功 Blender 进程内部时间合计 | **338.22 s**；峰值 RSS 2355532 KiB，约 2.25 GiB |
| 图片合成/编码/打包 | 8.48 s |

全程 512×512/方向、Cycles CUDA、8 samples、固定 seed29、无降噪、简单三盏区域灯。没有场景环境、DS/ERP 或高质量渲染。资源数字不包含全部阅读、脚本编写、工具等待和评审墙钟，不能当正式场景吞吐量。

成功验证后，仅清理本次独占、可重建的 142 张完整阶段分视图临时 PNG，释放 **27,675,724 B**；保留 76 张双方向合成帧、10 张联系图、MP4、5 帧探针视图、问题图、配方/日志及失败证据。清理前验证合成面板除明确标注区外与原像素一致。见 [cleanup_record.json](../../../outputs/experiments/apd_6526_dressed_pilot_20261008/cleanup_record.json)。交付约 **28.2 MB**，没有复制模型/整套角色或 76 份头发；所有历史输出、源资源和失败证据均保留。

| 新增文件 | 职责 |
| --- | --- |
| `tools/experiments/prepare_apd_6526_dressed.py` | 冻结配方；原生回读；固定衣物绑定；小型头部矩阵；有界数值诊断 |
| `tools/render/preview_apd_6526_dressed.py` | 复用已有材质/groom 消费者；固定两方向；逐帧更新；设备和相机记录 |
| `tools/validate/validate_apd_6526_dressed.py` | 衣物局部面与固定毛发样本诊断；探针/完整回读一致性 |
| `tools/visualize/package_apd_6526_dressed.py` | 原时间图片、MP4、HTML、逐帧与联系图交付 |

没有公共模块修改，历史静态调用默认行为完全未动；未跑不相关全库测试。定向检查覆盖本轮原生参数、绑定、R138、Blender 回读、相机/时间、编码、网页脚本和脚本语法。详见 [execution_record.json](../../../outputs/experiments/apd_6526_dressed_pilot_20261008/execution_record.json)、[delivery_validation.json](../../../outputs/experiments/apd_6526_dressed_pilot_20261008/delivery_validation.json)。

## 6. 来源与下一阶段前置条件

APD 署名和修改说明见 [ATTRIBUTION.md](../../../outputs/experiments/apd_6526_dressed_pilot_20261008/ATTRIBUTION.md)。APD 数据 CC BY4.0 与 SMPL-X、BEDLAM shape、Meshcapade 皮肤、BEDLAM2/VINEFX groom 的许可各自保留；本轮使用已有本地资产的来源记录，没有重新下载、重新授予许可或外部发布。源坐标/尺度/foot 未确认项继承身体适配报告，没有伪称官方 GT。

继续继承 `APD_5724486_v1_identity_session_unknown` 与 `cycling_M01` 家族；同段不同视图、同视频邻段、变速、循环和换角色派生都不是独立验证来源。只交衣物缓存和 groom 矩阵配方；原身体参数仍是权威，138 标签仍由身体生成。本轮没有 DS 观测，**没有伪造 A1 的 sequence_index/打包成品**。

下一批最小任务应在用户接受本段穿衣播放后另行启动，仍限定同一角色和这 76 帧：

1. 接入逐帧资产消费者，选择同一 asset_frame 的身体、衣物、头部矩阵和 R138；验证 groom 只做 `scene @ H(t)`，全身只做一次共同 scene 变换。静态入口继续兼容。
2. 明确选择“原地动作＋场景轨迹”，保留当前固定 pelvis 约定；单独声明整段放置/地面路径、yaw 与坐标转换，不恢复或叠加 APD 头部路径，不按每帧 min-Y 落地。
3. 配置真实同步双相机与小规模观测输出，核对图像/可见性/标签同刻一致；再使用既有 A1 序列索引和回读接口。当前 studio 相机不能冒充 DS 观测。
4. 复查投影视角下这些小面积衣物折转和毫米级头皮穿插是否仍可接受；动态观感、真实场景资源成本、遮挡/截断及 compact 来源隔离尚待该批验证。

**本轮停止。** 不扩角色、不叠道路轨迹、不开始双路/DS 渲染、模型推理或训练。当前需用户决定的唯一事项是本段穿衣与头发连续播放观感是否可接受。
