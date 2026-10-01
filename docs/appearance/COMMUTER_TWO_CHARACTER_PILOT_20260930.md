# F02 / M02 通勤角色试制记录

日期：2026-09-30。结论：**需修正，不能扩十人**。

后续衣物修复记录见 [COMMUTER_CLOTHING_REPAIR_20260930.md](COMMUTER_CLOTHING_REPAIR_20260930.md)。
该轮拆分了静态衣型、绑定、人体重叠和组件接触，原始 P2 几何也发现腋下相交；
仍未通过跨姿态衣物验收，DS 保持未运行。下文保留首轮历史记录，不用后续结果覆盖旧失败。

两名角色的完整模型参数、四份骑行姿态几何及 138 点链路通过独立回读；
两份真实三维 groom 已取得、导入、绑定至两姿态并完成修碰/有限发丝清理。
用户提供的 add-on 已核验，参考模型访问障碍已解除。通勤衣物四版候选及两次静态布料求解仍未通过。
遵守“先棚拍、后 DS”，本轮 DS 样例为 **0**；
不把旧 DS 验收结果转记到这些新资产，也不宣称完整角色试制通过。

全部新资产与诊断均为 `integration_only`，两个身份、两个独立姿态、四个组合；没有训练、推理或重跑照片重建。
原角色、原始照片/重建、旧输出和既有未提交修改均保留，未修改外部项目、计算 SHA256、提交 commit 或移入 datasets。

## 产物与当前状态

实验根目录：[cycling_commuter_pilot_20260930](../../outputs/experiments/cycling_commuter_pilot_20260930/)。

| 组件 | 产物 | 验收状态 |
|---|---|---|
| 完整形状空间 | `models/SMPLX_{FEMALE,MALE}.npz`、`model_regression.json` | 从本地现有归档解出；旧 10 维回归通过 |
| F02 / M02 新版本 | `characters/cycling_{F02,M02}_v2/manifest.json` | shape 候选；未冻结，父身份仍是各自 v1 |
| 四组合人体 | `derivatives/<asset_id>/<pose_source_id>/geometry/` | 参数、pelvis、R @ V 回读通过 |
| 四版衣物＋两次求解 | `commuter_clothing_v1` 至 `v4`；F02/P2 的 `cloth_simulation_v5`、`v6` | 全部保留；衣物仍未整体通过 |
| 发型 | `groom_reference/`、`groom_binding_v1/`、每份姿态的 `groom_v1` 至 `v5` | v5 绑定/身体采样检查通过；完整服装接触、DS 实例尚未验收 |
| 统一条件对照 | `studio_uniform_comparison/` | F02/F01、M02/M01，24 张同光照同服装预览 |
| 通勤诊断 | `studio_clothing_v2_diagnostic_gpu/`、`studio_clothing_v4_diagnostic/` | 各 24 张；后者为最新失败候选审查图 |
| 带三维发型的诊断 | `studio_groom_v5_clothing_v4/`，F02 材质最新局部复查 `studio_F02_scalp_v5/` | 36＋8 张；另有首版 groom 4 张，所有渲染共 120 张 |
| 实际预处理 | `studio_groom_v5_clothing_v4/loader_diagnostic/` | 最新 18 行 actual loader，bbox 包含保留发丝；256×256 与 256×192 输入；不运行网络 |
| 独立核验 | `independent_readback.json`、`acceptance.json` | 身体/绑定通过与衣物/发型未通过分别记录 |

建议先看：

- [统一脸部对照](../../outputs/experiments/cycling_commuter_pilot_20260930/studio_uniform_comparison/identity_heads.jpg)。
- [统一衣着体型及灰模头部](../../outputs/experiments/cycling_commuter_pilot_20260930/studio_uniform_comparison/shape_comparison.jpg)。
- [P1 正侧背面](../../outputs/experiments/cycling_commuter_pilot_20260930/studio_clothing_v4_diagnostic/pose_P1_multiview.jpg)。
- [P2 正侧背面](../../outputs/experiments/cycling_commuter_pilot_20260930/studio_clothing_v4_diagnostic/pose_P2_multiview.jpg)。
- [实际方形裁剪与网络中心输入](../../outputs/experiments/cycling_commuter_pilot_20260930/studio_clothing_v4_diagnostic/loader_diagnostic/square_and_network_center.jpg)。
- [最新两角色三维发型与脸部](../../outputs/experiments/cycling_commuter_pilot_20260930/groom_and_face_review.jpg)。
- [两姿态颈肩近景](../../outputs/experiments/cycling_commuter_pilot_20260930/groom_neck_review.jpg)。
- [带发型的实际网络输入检查](../../outputs/experiments/cycling_commuter_pilot_20260930/studio_groom_v5_clothing_v4/loader_diagnostic/square_and_network_center.jpg)。

## 来源、身份和完整模型

复核了规划报告、候选 JSON、四角色 manifest/最终预览以及现有生成、材质、渲染和监督入口。
新版本没有增加身份数量：

| 新资产 / 稳定身份 | 父资产 | 皮肤沿用 | 衣物候选 | 发型候选 |
|---|---|---|---|---|
| cycling_F02_v2 / cycling_F02 | cycling_F02_v1 | skin_f_african_01_ALB | 长袖上衣＋直筒裤 | vinefx_groom_18-1_short-bob-F |
| cycling_M02_v2 / cycling_M02 | cycling_M02_v1 | skin_m_african_01_ALB | 短袖 polo＋直筒裤 | vinefx_groom_01-1_curly-M |

皮肤、UV 和材质沿用原有出处；新 shape 重新生成原生眼球局部坐标。
统一运动衣着对照仍保留旧头皮贴图，仅用于身份比较。
新发型预览使用真实 NURBS groom；F02 新材质 v5 以头部专用顶点 mask 混合原贴图自身的额头肤色，
保留原 PNG，未修改标签人体或更换脸部来源。v4 的亮度门控留下明显侧额边缘，已留存失败图；
v5 改为局部连续混合，在新近景中旧短发贴图边缘已不明显，但填充区域缺乏微纹理，仍是试制方案。
同条件对照可见 F02/F01 的唇部纹理、下颌和身高差别，以及 M02/M01 的身高、肩躯干和下颌差别；
两组灰模仍有相近基础脸型。当前仅支持保留这两个候选继续试制，不外推到十人多样性通过。

完整模型来自只读外部归档：
`/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools/PEAR/assets/models_smplx_v1_1.zip`
（870,108,517 bytes）。未重新下载模型。

直接比较了模板、面序、pose directions、LBS weights、joint regressor、parents、前 10 个 shape directions 和 expression directions。
完整文件的 **0:300 是 shape，300:310 是本次使用的 expression**；当前精简文件的 10:20 对应 expression，未当作身份形状。
旧 10 维只在已验证相同 gender、相同前 10 维基底内补零；没有跨模型版本复制 betas。
旧两身份在参考姿态、P1、P2 共 6 个回归实例：网格最大误差 0，138 点最大误差 5.9604645e-8 m。

新 shape 使用前 30 个真实 shape 方向，在全身模型内做有正则项的温和测量目标拟合，expression 保持模型零默认值并独立检查。
30 不是验收指标；没有局部改顶点、拼头或使用表情冒充身份。
参数变化 ±1 是工程保护界限，**不是统计标准差或已验证人群先验**。
测量使用固定解剖区域顶点带宽，肩宽项并非临床外肩宽；不能把这些代理量单独当作合理性验收。

| 测量代理（m） | F02 v1 → v2 | M02 v1 → v2 |
|---|---:|---:|
| 身高 | 1.71724 → 1.71731 | 1.79378 → 1.79389 |
| 肩部带宽 | 0.46090 → 0.47628 | 0.46020 → 0.48673 |
| 腰部带宽 | 0.27772 → 0.28559 | 0.32206 → 0.34028 |
| 头部横向带宽 | 0.16433 → 0.17072 | 0.17669 → 0.18354 |
| 下颌带宽 | 0.11467 → 0.12075 | 0.14329 → 0.15117 |
| 全身最大静态顶点位移 | 0.00983 | 0.01679 |

P1=`coco_292226_person_1330527`，P2=`coco_306800_person_225160`。
沿用 `outputs/experiments/cycling_smplx_20260929/<pose>/canonical_pose.npz`，每个目标 shape 重新计算 pelvis/transl、身体、附件绑定和 R @ V。
四个组合互相独立，不组成 motion；同一角色跨两姿态 betas 和外观来源一致。

## Groom 获取与实际导入

[BEDLAM2 官方下载页](https://bedlam2.is.tuebingen.mpg.de/download.php)的 Hair/Alembic 包是本次使用的最小已列出下载粒度；
没有个别 groom 的官方直链可用证据，因此获取一个包，仅解出两份候选及 LICENSE。
未下载 Maya groom、motion、Starter Pack、旧动作服装缓存或 bald 纹理大包。

资源目录：`assets_external/bedlam2_pilot/`；实际目录清单见 `inventory.json`，来源/大小见 `.source.json`。

| 文件 | 实际 bytes | 状态 |
|---|---:|---|
| b2_groom_abc.tar.gz | 1,335,726,114 | 已下载；40 份 groom＋许可；全包解压大小总计 4,365,348,841 bytes |
| vinefx_groom_01-1_curly-M.abc | 104,340,124 | 已解出；Blender 5.1.1 导入为 4 个 CURVES 对象 |
| vinefx_groom_18-1_short-bob-F.abc | 77,647,013 | 已解出；Blender 5.1.1 导入为 4 个 CURVES 对象 |
| smplx_model_lh_20230302.blend | 344,983,880 | 用户放入本地 add-on，已读取 neutral 参考；不替换标签人体 |

包内许可指向 [BEDLAM2 license](https://bedlam2.is.tuebingen.mpg.de/license.html)；保留该限定用途许可记录。
模型沿用 [SMPL-X model license](https://smpl-x.is.tue.mpg.de/modellicense.html)，旧 Meshcapade 皮肤为 CC BY-NC 4.0。
登录凭据没有写入仓库、脚本、报告或 cookie 文件；下载会话已关闭。

M02 导出 80,411 条曲线 / 2,560,079 个控制点；F02 导出 84,069 条 / 1,890,570 点。
保存了原始控制点、曲线 offsets、导入半径、对象矩阵、分组和属性信息。
首次检查脚本只识别旧 CURVE 类型，已保留该诊断并修正为支持 Blender CURVES；没有误判为格式不可用。

实际核验 add-on `bl_info.version=(2024,11,29)`；neutral mesh 标记 `locked_head`、无非零 shape key。
与本地标准 neutral 比较：完整面序一致，非头部模板在 SMPL-X→Blender 轴转换后最大误差约 1.2e-7 m，
差异集中在头部，最大约 12.4 mm。结合该证据使用相同三角形语义迁移，未因顶点数相同就直接认定兼容。

源 groom 坐标乘 0.01 后直接匹配参考模型；所有发根到限定 scalp 的距离不超过约 1.27 mm，
首端比末端更贴近头皮，180° 反向比较明显更差。保留 NURBS order/knots 和导入 radius，半径随厘米→米同步转换。
这不是仅靠包围盒试摆。

v1 用发根三角形/重心及局部头皮框架；v2 的整体 scalp affine 试验反而增加局部穿插，保留但不用。
v4 回到 v1 绑定做有界身体修碰；v5 **实际移除**剩余异常发丝，渲染 helper 调用 `remove_curves`，
不是只从检查中排除。F02 移除 7/84,069 条（0.0083%），M02 移除 14/80,411 条（0.0174%）；
每人参考/P1/P2 使用三者失败发丝的并集，因此保留的发丝 ID 和拓扑跨姿态一致。
原 ABC 和所有失败版本不变。若清理超过 1% 则脚本拒绝继续，该上限没有触发。

发型新门槛：修碰小于 30 mm、采样 signed-normal 不低于 -1 mm，发根漂移小于 1e-6 m。
v5 六例保留点最大修碰 29.79 mm，最差采样 -0.903 mm；回读绑定与发根最大误差均 5.9604645e-8 m。
独立数据见 `groom_independent_readback.json`。NURBS 控制点采样不是完整曲线碰撞证明；
当前身体接触检查通过，**与最终衣物的接触及 DS 深度/实例仍未验收**。
头部与两姿态颈肩图未见原生头部明显穿出或大块发根漂浮；旧模板/回归器未因 head-bun 被删点或雕改。
官方绑定入口以 locked-head neutral authoring reference 建立 groom binding：
[create_groom_bindings.py](https://github.com/PerceivingSystems/bedlam2_render/blob/main/unreal/import/groom/create_groom_bindings.py)。

参考资源已由用户放在 `assets_external/smplx_blender_addon/`；当前无需用户再次下载或提供账号信息。

## 衣物制作、失败与下一步修正

本轮衣物从各自 shaped rest body 的表面对应出发，形成独立上衣/裤装，加入下摆松量、
近似恒截面的下裤腿、袖口、折领和门襟；与旧运动 4 mm 壳层使用不同的几何和绑定记录。
所有服装均在本仓库生成，无付费软件或旧 motion 缓存复用。
但它仍是**表面派生的程序化衣型候选，不是已经合格的通勤成衣**；当前胸腹有不自然的波纹/壳状轮廓。
统一身份对照中另用旧运动衣型仅为了同条件比较，不算通勤衣物成果。

绑定保存 `barycentric posed-body reference + blended joint rotation @ rest_offset + collision_correction`；
目标身体保持不变，皮肤权重和旋转均在独立回读中重新计算。

- v1：裤腿松量侵入裆部，固定尺寸领圈穿入肩颈，失败。
- v2：裤腿松量在大腿中段渐变，领圈改用实际截面；仍有肩部开口及姿态碰撞，失败。
- v3：加入姿态表面法线修正和胸部支撑；局部修碰位移过大，失败。
- v4：提高裁切领口以保留肩峰，加入修碰预算终止；保留更合理轮廓的失败候选，不再让修碰任意拉长衣物。
- 静态求解 v5：F02/P2，70 帧仅作求解初始化，边界 pin＋身体/自碰；最大偏移 266.12 mm，失败。
- 静态求解 v6：130 帧、增加软 pin 并减慢姿态过渡；最大偏移 311.41 mm、顶点/面中心各 59/139 个样本低于 -1 mm，失败。
  两次都只跑最困难实例，没有把失败方法扩大到四组合。v6 独立确认碰撞体启用，末帧与目标人体误差 5.9604645e-8 m；
  因而不能将结果归因于碰撞体未开启或坐标接错。插值不是照片之间的 motion，未修改原姿态。
  v5 未另存实际末帧碰撞体，该项不追认为独立检查通过。

新服装事先单独设置的诊断门槛：顶点/面中心 signed-normal 样本不低于 -1 mm，修碰最大位移严格小于 60 mm；
60 mm 是阻止异常修碰的失败上限，不是合身松量标准。未放宽旧 0.4 mm clearance / 10 mm shell gate。
最近表面只在 24 个最近三角形质心候选中求解，属于局部采样诊断，**不是穷举穿插/自碰证明**。

| v4 情况 | 最小顶点 / 面中心（mm） | 最大修碰（mm） | 数值结果 |
|---|---:|---:|---|
| F02 参考 | -0.700 / -0.034 | 16.91 | 采样通过，外观待修正 |
| F02 P1 | -1.414 / 0.039 | 48.15 | 失败 |
| F02 P2 | -1.577 / -19.692 | 60.00 | 失败，修碰预算耗尽 |
| M02 参考 | -3.344 / -0.698 | 30.90 | 失败 |
| M02 P1 | -0.873 / -0.541 | 51.22 | 采样通过，不能替代外观通过 |
| M02 P2 | 1.821 / 1.015 | 43.07 | 采样通过，不能替代外观通过 |

拓扑回读：上衣各 4 个边界连通分量、裤装各 3 个，polo 折领/门襟各 1 个，无非流形边；
这排除了部分意外开边，却不证明没有面片互穿。
F02 P2 的一处 v3 初始腋下采样 winding number 接近 2，提示原始 posed body 局部自重叠；
记录在 `body_overlap_probe.json`，不是全网格自交结论，不能为此静默改动原姿态或人体。

当前瓶颈已具体定位到衣型/绑定及姿态折叠处的求解稳定性，继续增加推开次数或 pin 权重没有验收依据。
下一轮衣物工作必须转向连续衣片/重拓扑和明确的衣物层间约束：
保留裤腿截面与绑定成果，重做胸腹过渡、裆部桥接和 polo 领片，不继续用增加推开迭代数凑通过。
先在 rest/P1 验证新衣片，再处理 P2 腋下；Blender Cloth 的免费路径已实际试过，但当前表面派生网格不合格，不能直接批量沿用。
若人体自重叠使该 pose 无法可靠着装，应保留该组合失败，提出有来源记录的派生姿态修正方案，不修改原重建。

## 几何、裁剪与 DS 门槛

参考＋四组合共 6 份身体独立回读：

- 参数网格误差 0（沿用 2e-5 m 门槛）；pelvis 误差 0（2e-6 m）。
- R @ V 最大误差 5.9604645e-8 m（1e-6 m）；138 点固定，未使用 BEDLAM 点索引。
- 新服装绑定回读最大误差 1.4901161e-8 m（新绑定序列化检查 2e-6 m）。
- 带发型实际 DatasetKeypoints 的 18 行预处理：逆裁剪误差最大 1.8808769e-5 px，沿用 2e-4 px 门槛。
- 已查看 256×256 与 trainer 中 `x[:,:,:,32:-32]` 对应的 256×192 中心输入，未执行网络。

棚拍记录真实相机矩阵与投影，使用显式全矩形有效 mask，不凭黑色像素判无效。
这些裁剪只验证预处理与外观；未计算棚拍可见性，不作为训练样本，不替代 DS 有效 mask/遮挡验收。

DS 入口已支持新的绑定回读，但要求 manifest 为 `studio_accepted` 且碰撞门槛通过。
本轮没有任何组件被写成该状态。旧 shell 的距离检查保持原样。
新增测试覆盖旋转后的松量还原、错误身体来源、篡改网格和未经棚拍审查时拒绝进入 DS；相关 17 个单元测试通过。

后续 DS 仍须重新验证：20260320 独立标定及显式有效域、同高/相反光轴、实际鱼眼/ERP 尺寸与合成外参、
同一世界人体的双相机观测、头发/衣物的光照深度实例归属、逐人物材质与框、重新计算可见性。
沿用实际渲染世界人体/衣物 5e-6 m、投影/裁剪 2e-4 px 等既有门槛；目前这些项目均为未执行。
监督仍为 `in_view & inside_full_crop`，遮挡不自动取消监督；预测 keypoint_value 不用作置信度或权重。

## 耗时、继续条件与工具入口

本轮从 09:18 左右开始，10:10 收到参考资产继续，收尾约 10:50，约 92 分钟墙钟，含中断及并行等待；
精确结束时间和 subprocess 计时见 `timing.json`，各任务不能简单相加作为墙钟。
模型准备 3.98 s；四轮衣物共 145.18 s；两次布料求解共约 78.08 s；
下载耗时 368.45 s，资源等待与衣物/审查并行。完整渲染、groom 适配计算分别在机器记录中逐次列出。
编码/适配约 55–65 分钟、目视审查约 10–15 分钟属于回顾估计，未事先逐段打表，不能伪称精确人工工时；
它们也包含上述计算进行时的工作。原有约 19 分钟样例时间没有被用作制作成本。
完整收尾时间及计时范围以机器记录为准。

继续条件明确：**修正衣片/绑定并通过两姿态棚拍；复核发型与新衣物接触；再执行最少 DS 双相机样例**。
不需要再次讨论十人资源规划、扩大 shape 维数或采购软件。两人通过后，仍须对十人逐人及同性别配对审查。

可复现入口：

- `tools/experiments/prepare_commuter_pilot.py`：新目录模型回归、shape 候选和四组合。
- `tools/experiments/build_commuter_clothing.py --version <新版本>`：衣物候选，禁止覆盖同版本。
- `tools/experiments/prepare_commuter_comparison.py`：独立统一衣着对照。
- `tools/experiments/inspect_bedlam2_groom.py`：Blender 原始 groom 导入审计。
- `export_groom_reference.py`、`bind_commuter_groom.py`、`relax_commuter_groom.py`、`prune_commuter_groom.py`：参考、绑定、修碰和跨姿态一致清理。
- `prepare_commuter_cloth_path.py`、`settle_commuter_cloth.py`：已保留的静态布料失败试验。
- `tools/render/review_cycling_characters.py --record-projection`：棚拍和真实相机记录。
- `tools/validate/validate_commuter_pilot.py`、`validate_commuter_studio.py`：参数/绑定与实际预处理。
- `tools/validate/validate_commuter_groom.py`：发型独立回读。

后续计算前仍须重新检查 GPU/主机进程。本轮使用空闲 GPU0，保留原有 Blender GUI 与编辑器服务。
