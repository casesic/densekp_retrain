# A2：BEDLAM 原配行走人物核查与连续预览交付

**当前状态（2026-10-09，用户授权下载后续执行）：已补齐第三个候选 BEDLAM2 `us_2495_L/2200 + nt_belle_007`，完成原身体＋原动作＋原配衣物的91帧、30 Hz、双固定视角连续预览。技术回读通过，动态观感待用户评审；R138 locked-head 模板语义仍 pending，未接入公共场景。**

当前入口：[连续播放页](../../../outputs/experiments/bedlam_native_walk_a2_20261009/native_attempt_2/preview.html)、[MP4](../../../outputs/experiments/bedlam_native_walk_a2_20261009/native_attempt_2/native_walk_two_views.mp4)、[实际资产 manifest](../../../outputs/experiments/bedlam_native_walk_a2_20261009/native_attempt_2/manifest.json)、[技术验收](../../../outputs/experiments/bedlam_native_walk_a2_20261009/native_attempt_2/validation.json)、[当前交付指针](../../../outputs/experiments/bedlam_native_walk_a2_20261009/current_delivery.json)。下载、冻结和实施证据见本文第7节。下方第1–6节是最初“只用本地资源”阶段的原始记录，保留当时缺口与停止结论，不代表授权补齐后的当前状态；原blocked JSON不覆盖。

## 初次本地核查记录（历史状态）

日期：2026-10-09。状态：**在最多三个具体组合内，未找到资源完整、可确认普通行走的原配组合；停在资源评审点。** 未运行身体求值或 Blender，因此没有本轮连续 HTML/MP4，也没有可直接交给 A1 的已验收人物资产。下面交付的是实际资源检查结果与候选交接，不把 manifest 当作消费成功。

入口：[候选清单](../../../outputs/experiments/bedlam_native_walk_a2_20261009/candidates.json)、[选择结论](../../../outputs/experiments/bedlam_native_walk_a2_20261009/selection.json)、[精确缺口/获取入口](../../../outputs/experiments/bedlam_native_walk_a2_20261009/missing_resources.json)、[候选交接 manifest](../../../outputs/experiments/bedlam_native_walk_a2_20261009/delivery_manifest.json)。所有候选仍为 `integration_only`。

## 1. 本轮边界与选择方法

遵循用户最新决定：优先本地原身体、原动作与原衣物配套，不换 cycling_M01、不补齐或截断源 betas、不重新绑定衣服。APD6526、已接受简单衣物/groom及旧 BEDLAM 骑行试制不动；motorcycle 未下载、未适配。没有等待或修改 A1 的直接 ERP 对照。

已读取资源参考及库存、source_evidence、既有衣物/贴图成员索引、官方本地配对表、ASSETS/CONTRACTS/DATA_AND_MODELS、A1 序列索引和骑行交接，以及 A1 动态资产 manifest/消费者。检查祖先和相关目录，未发现适用 AGENTS.md。Git 原有修改保留，保护记录见本轮 `protected_before.json` 与最终 `protected_after.json`；未改 TODO、README、历史实验或外部 pano-stitch。

先按已知本地配套关系收敛到三个组合，提取前写入 [candidate_freeze.json](../../../outputs/experiments/bedlam_native_walk_a2_20261009/candidate_freeze.json) 和 [budget.json](../../../outputs/experiments/bedlam_native_walk_a2_20261009/budget.json)：

1. `BEDLAM1/rp_aaron_posed_009/1095`：唯一已解出的完整 motion/cloth 关系，先确认是否适合本轮动作。
2. `BEDLAM1/rp_aaron_posed_009/1033`：同一已有身体/贴图角色下，衣物分片索引的首个仿真条目；检查是否还带有所需身体/动作信息。**它是资源候选，事先不标为行走。**
3. `BEDLAM2/us_2495_L/2200`：本地已有的行走来源参数，检查其原配 `nt_belle_007` 是否可直接使用。

没有因为前三项不足再扩人、扩动作。已知示例 `KIT/314/bend_left01` 和历史 P0 geometry 仅作为来源排查：该例不是普通行走，不新增为第四个深入适配候选。没有重新做上轮全库研究。

## 2. 三个组合的实际结果

| 组合 | 身体/动作 | 衣物、UV、纹理 | 动作与选择结论 |
| --- | --- | --- | --- |
| B1 `rp_aaron_posed_009/1095` | 原 motion 本地存在；male，10 betas，192帧/30Hz；`motion_info` 指向 KIT/883/wipe_arm_dabbing01 | 39帧/6Hz，原 NPZ、face-corner UV、diffuse/normal均存在；历史配对映射存在 | metadata 类别 stretch，源为 wipe_arm；**不选，不能冒充普通走路** |
| B1 `rp_aaron_posed_009/1033` | 对应 motion 未在限定路径找到；原动作文件、实际shape、身体帧偏移未知 | 从本地归档实际提取 NPZ，24帧/6Hz；固定 faces/UV 基础检查通过；同角色原 texture_01 已存在 | **未确认动作类别；缺原身体动作，因此不选** |
| B2 `us_2495_L/2200` | 118,606 B原 NPZ，neutral，16 betas，127帧/30Hz；对应 KIT/425/walking_medium05 | 官方配置配 `nt_belle_007`；本地已知衣物分片没有这个body，未找到相应缓存/UV/贴图 | **最接近行走需求的未齐候选**；没有完成原配动态观看，不作为可用资产发布 |

候选 JSON 保存路径、数组维度、原形状系数、官方映射、缓存元数据、贴图选项及每项缺口。这里的“文件有效”与“原配关系正确”“普通行走观感通过”是三个不同状态。

### 原配 1095：为什么不使用已有预览凑交付

原 motion：

`/mnt/data1/yy/home/yy/datasets/BEDLAM/motion_data/tiny/rp_aaron_posed_009/1095/motion_seq.npz`

原衣物：

`/mnt/data1/yy/home/yy/datasets/BEDLAM/clothing_data/tiny/rp_aaron_posed_009/clothing_simulations/1095/1095.npz`

本轮读到衣物 metadata：`fps=6, source_fps=30, start_frame=101, end_frame=292`。旧 temporal_case 记录 body local `[0,5,…,190]` 与 cloth `[0,…,38]`、source30 `[101,106,…,291]` 对应。本轮没有重算该身体/衣物配对或重播旧动画；旧39帧产物不能成为本轮“新行走人物通过”的证据。

### 定向提取 1033：实际做了什么

归档：`/mnt/data1/yy/home/yy/datasets/BEDLAM/clothing_data/clothing_npz_6fps/clothing_npz_6fps.0.tar`。

唯一提取成员：`rp_aaron_posed_009/clothing_simulations/1033/1033.npz`，索引和实物均 **3,805,798 B**。另有同目录 `1033_0101.obj`（索引925,915 B），本轮没有提取，因为 NPZ 已包含网格和 UV；没有全量解包或写共享 datasets。

结果：`vertices_seq[24,13027,3]`、`faces[26328,3]`、`vertices_uv[14776,2]`、`faces_uv[26328,3]`。全部顶点/UV有限，两个面索引范围正确，面数对应，拓扑为全段共享数组。metadata 为 `fps=6,source_fps=30,start_frame=101,end_frame=220`。这只给出缓存的时间说明，**不能据此确定未知 body NPZ 的 local0 对应什么**，也没有原动作名称。

安全提取逐项检查白名单成员名、绝对/上级路径、链接类型、声明大小、目标存在与目标父目录链接；使用普通文件流复制，不调用 extractall。五个定向边界检查拒绝了越界名称与选中成员的符号链接，见 [提取记录](../../../outputs/experiments/bedlam_native_walk_a2_20261009/extraction.json) 和 [边界检查](../../../outputs/experiments/bedlam_native_walk_a2_20261009/extraction_boundary_checks.json)。

同角色已解出的 `texture_01_diffuse_1001.png` 和 `texture_01_normal_1001.png` 已检查文件头，精确路径/尺寸在 `texture_headers.json`；不新复制或更改纹理。这份衣物不是新造 shell，但缺 motion 时仍不能运行原配预览。

### B2 行走：模型资源的新发现与尚未确认项

源参数没有变化，官方 CSV 是训练 motion 映射；项目历史仍将 KIT425作为验证来源，不改既有划分。它的官方 body 性别元数据是 f，NPZ求值模型是 neutral，不能从目标角色或尺码推断源模型性别。

上轮“support_data约定位置没有 locked-head NPZ”仍成立，但**不能扩大成整机没有相应模型**：本轮找到插件已有

`assets_external/smplx_blender_addon/data/smplx_model_lh_20230302.blend`（344,983,880 B）。

插件代码的 `smplx_version=locked_head` 分支明确选择这个文件；也有相应 betas-to-joints 数据。尚未加载验证源16维形状、pose correctives、模板及衣物参考的一致性；旧 `rig_compatibility.json` 只说明骨骼旋转约定一致，明确未证明 shape space 相同。因此候选记录为“已有可核查 Blender 模型，兼容性待验证”，**不是要求先下载一个新模型**。本轮没有调用会把源beta补到30维的旧人物适配器。

## 3. 有界搜索与真正缺失的资源

检查仅覆盖已知 BEDLAM motion_data、该角色tiny路径、B1/B2示例目录、已有 source_isolated sources、bedlam2_pilot、声明的模型目录与已有基线 provenance。`/home/yy/datasets/BEDLAM` 与 `/mnt/data1/yy/home/yy/datasets/BEDLAM` 解析到同处，不是两套独立库存。已知 motion_data 内本次仍只找到1095。详见 [bounded_search.json](../../../outputs/experiments/bedlam_native_walk_a2_20261009/bounded_search.json)；**不声称整盘不存在其它资源**。

最接近候选保留为 `BEDLAM2/us_2495_L/2200 + nt_belle_007`，但没有将其冻结为预览资产。

| 缺口 | 准确定位/现有证据 | 获取或补证方式 |
| --- | --- | --- |
| 原配仿真衣物 | body=`us_2495_L`、motion=`2200`、outfit=`nt_belle_007`；不是任意 nt_belle_007 的其它motion缓存 | 官方 B2 下载区 clothing NPZ 的 archive_map 按组合查具体分片和成员 |
| 精确 NPZ 成员/分片/大小 | **unknown**，未取得对应 archive_map 行；不能编造tar编号或假定下载大小 | 先拿小型目录条目，再决定取得一个必要成员；不下全部分片/Starter Pack |
| 服装材质 | 需该服装的一套纹理、独立UV及法线/材质关系；配置包含 texture_01 | 随服装资源定位，未确认本地路径与字节数 |
| 原模型一致性 | 现有 locked-head Blender 文件可核查；普通NPZ不能直接替换 | 模型/模板/原16beta/pose correctives与官方服装reference对齐后再求值 |
| B1备选1033的原motion | 逻辑成员 `rp_aaron_posed_009/1033/motion_seq.npz`；归档前缀和成员大小unknown | 找原 `gendered_ground_truth` 仿真动作，不拿 neutral发布GT替代；到位后仍先看 motion_info/动作 |

官方 ABC 导入代码示例支持 `body_id/motion_id/motion_id.abc` 的组织约定，所以 `us_2495_L/2200/2200.abc` 只能记为**按代码推得的候选路径**，不是已核实归档成员。需要补的精确身份已给出；无法确定的物理成员名/大小明确保留unknown。

本轮网页只读核查：[B1下载](https://bedlam.is.tuebingen.mpg.de/download.php)重定向登录；[B2衣物分片索引](https://download.is.tue.mpg.de/download.php?domain=bedlam2&sfile=assets/clothing/npz/archive_map.json)网页工具未能取得内容。没有绕过访问限制、读取凭据、注册或下载新包。本轮权限不包含新motion/服装大包，故在这里停止。

本地资源的研究使用继续保留[BEDLAM许可](https://bedlam.is.tuebingen.mpg.de/license.html)、[BEDLAM2许可](https://bedlam2.is.tuebingen.mpg.de/license.html)与独立SMPL-X许可；不是代码MIT授权。B1引用 Black等，CVPR2023；B2引用 Tesch等，NeurIPS2025。内部核查产物不作为许可已批准公开再分发的声明。

## 4. 时间、几何与138点：当前通过和未通过

| 检查 | 本轮状态 |
| --- | --- |
| 两份已有motion的参数维度、数值有限 | 通过文件检查；没有修改、求值或平滑 |
| 1095/1033衣物所有缓存帧的有限值、faces/UV合法性 | 通过基础读取；没有改拓扑或删面 |
| 选定行走身体与衣物时间/几何一致 | **未执行：没有完整选定组合** |
| 模型与138点语义兼容 | **pending**；没有因10475顶点或R可乘就宣布通过 |
| 预览探针/连续动态/用户观感 | 未生成，因资源门槛未满足；不是穿衣质量失败 |
| 公共场景消费/DS/ERP/打包/Dataset | 未执行，且不在本轮范围 |

本轮 `selected_interval=null`、`timeline_ref=null`，没有人为造19帧或91帧的“成品时间表”。[timeline_contract.json](../../../outputs/experiments/bedlam_native_walk_a2_20261009/timeline_contract.json)只是资源到位后的规则：若cloth6/body30，精确取19个配对时刻覆盖首末3秒，time_base=1/30、tick步长5；若两者30Hz，最多91个时刻。保留实际源offset，不插值、不复制帧、不变速。19帧以6Hz编码的播放占时19/6秒与首末跨度3秒要分开记录。

root必须保留原动作，并核实何时已进入衣物缓存；统一一次固定展示/场景变换，不逐帧贴地。无原配头发可先省略，不新做绑定；不引入鞋/去趾模板。未来标签依然使用本项目R作用于经语义核查的身体，不能替换成官方点索引或衣物表面。

## 5. 给 A1 的准确交接

本轮 `delivery_manifest.json` 的 schema 是 **候选交接提案**，`asset_ready=false`、`A1_schema_effective=false`、`geometry=null`。没有替换 A1 的行人，也没有伪造 sequence_index 的 observed/packed 行。

| 只读核查的调用点 | 后续原配资源到位后需要的能力 |
| --- | --- |
| `prepare_pedestrian_seeded_assets.py` | 当前用目标30beta/simple shell的生产方式不能拿来生成“原配”资产；原配应另行读取原body与cloth cache |
| `data/road_pass.py` 与 `plan_road_pass.py` | 当前30Hz/stride3/31帧假设不能套6Hz衣物；要从统一native timeline显式选帧 |
| `render_road_pass_sources.py` 的 clothing_vertices 消费 | 逐帧缓存入口已有基础；原配衣物应直接读缓存，不运行bound_garment重新生成 |
| 同渲染器 `add_clothing_materials` | 增加原衣物faces、face-corner UV、diffuse/normal等表示的显式消费；不能把简单shell材质当成原材质 |
| 同渲染器强制 groom/head 分支 | 允许明确 `groom=none` 的身体＋衣物资产；不能为凑接口绑定新发型 |
| `sequence_packing.py` | 目前source_time_base固定1/30并读取original_100fps_mapping；后续使用资产真实时间，不替其它来源伪造100Hz字段 |
| `sequence_index.py` | 现有整数有理时间机制可沿用；本轮没有生成图像，故不创建成品观测记录 |

原配root在body/cloth中只保留一次，scene施加共同放置；是否另加轨迹须显式声明。R138、模型模板、头发/鞋情况按manifest真实状态传递。以上仅交接建议，没有修改这些公共文件或测试，也没有阻塞等待A1改接口。

## 6. 实测资源与停止点

计算前检查主机GPU、进程、内存和磁盘。沙箱内nvidia-smi无法访问驱动，随后沙箱外只读检查成功；这是权限可见性问题，不据此判GPU损坏。记录见 `resources_before_cpu.json`：

- GPU0：`GPU-d481acc7-106f-29e5-1298-8c0d398261b2`，18MiB/24576MiB，0%；GPU1：`GPU-6f046d56-acbe-d49b-774a-9d463e768cac`，407MiB/24576MiB，0%。实际计算只用CPU，未选择或占用GPU。
- 主机已有 Blender PID43491，不干预；可用内存约141GiB、项目盘空闲约12TiB。状态仅对应该次检查。
- 独立审计脚本实测墙钟 **0.876s**，CPU **0.190s**，峰值RSS **45,664KiB**；其中定向提取 **0.749s**。另有少量索引/网页/文件头检查，不把该数字当整轮工作耗时。
- 新提取 **3,805,798 B**；没有新身体/衣物派生缓存、模型或纹理副本，远低于1GiB上限。最终产物大小和保护差异见 `final_validation.json`。
- 身体求值、探针渲染、完整渲染及视频编码均 **0秒**；没有失败渲染版本。未满足资源门槛，所以没有以未实测单帧时间估算或占用30分钟渲染预算。
- 开发/阅读/等待和计算分开记录；没有可靠全程CPU/等待拆分的部分明确为未单独计量，不编造成已验证成本。

可复查入口：[audit_bedlam_native_walk.py](../../../tools/validate/audit_bedlam_native_walk.py)。使用新的输出目录，它会按冻结三个候选重新做小型CPU检查并提取一个3.8MB成员；不要为了复查而反复生成副本。没有全库测试、依赖安装、SHA256、commit、训练、推理或公共场景渲染。

**本轮结论：没有找到完整原配行走组合；没有新的连续预览待用户选外观；身体/衣物原配时间与几何、R138和公共消费均未验收。** 最小下一步是补齐一个明确组合的原配资源，不是重做衣物适配。允许范围已到停止条件，保留现场供评审，不自动下载或扩候选。

## 7. 用户授权补齐后：真实原配连续预览

### 7.1 下载与原配关系

用户随后明确授权代为登录下载，取代先前“不下载新衣物”的限制。仅补齐已选的第三个候选，没有扩候选、下载新模型/motion、Starter Pack或整包。官方认证成功后按官方 archive_map 定位，使用HTTP Range定向取得普通tar成员，验证状态码、Content-Range、成员类型/路径和声明大小。登录会话已关闭；邮箱、密码、cookie及带令牌URL不写入项目产物。

| 实际资源 | 官方归档与成员 | 保存字节数 |
| --- | --- | ---: |
| 原配衣物NPZ | `assets/clothing/npz/b2_clothing_npz_256.tar` → `us_2495_L/2200/2200.npz` | 31,617,436 |
| 原diffuse | `assets/clothing/textures/b2_clothing_textures_11.tar` → `nt_belle_007/texture_01/texture_01_diffuse.png` | 4,748,207 |
| 原normal | 同纹理归档 → `nt_belle_007/texture_01/texture_01_normal.png` | 4,893,858 |

两个完整tar分别为5,048,780,800和5,375,447,040 B，**均未整包下载**。三个必要成员合计41,259,501 B（约39.35 MiB），成员下载含索引头探测的网络读取41,847,937 B；另有两个小型archive_map共1,101,193 B和512 B头检查。详细成员、偏移、字节和耗时位于 `acquisition_attempt_3/4/5`，原始获取记录不因后续通过回读而改写。

这是官方受认证资源，不以代码MIT替代数据许可。继续引用[BEDLAM2研究许可](https://bedlam2.is.tuebingen.mpg.de/license.html)、[SMPL-X模型许可](https://smpl-x.is.tue.mpg.de/modellicense.html)、已存 `source_evidence.json` 和原KIT/AMASS来源；本轮不作公开再分发许可声明。原官方motion/body/outfit配对见已有候选记录与官方配置，不由衣物外观猜配对。

### 7.2 冻结与原模型求值

在求值前保存 `native_attempt_1/frozen_recipe.json` 与时间表，读取器修正后 `native_attempt_2` 沿用相同冻结条件，**没有换片段或选最好结果**：

- 身体参数保持本地 `source_isolated_pilot_20260930/sources/us_2495_L_2200.npz` 原127帧中的 `[0,91)`；源16维betas保持不变，原pose/global_orient/trans保持不变。不是项目30维角色，不重定向、不拟合、不改变shape。
- 使用现有 `smplx_model_lh_20230302.blend` 中 `SMPLX-mesh-neutral`，其属性为 `smplx_version=locked_head`、`smplx_gender=neutral`，及配套locked-head关节回归器。按本地官方插件公式驱动原16个Shape键、55个骨骼旋转和486个pose corrective键；未观测shape/expression键保持模型默认零值，没有构造一个补零的项目beta向量。
- 官方身体元数据f与求值模型neutral分别记录。没有下载/替换普通SMPL-X模型、修改模板、加去趾或鞋位移。
- 衣物NPZ为127帧、30 Hz，`vertices_seq[127,19639,3]`，`faces[38861,3]`，`vertices_uv[20327,2]`，`faces_uv[38861,3]`；metadata为start101、end227。取前91帧，仅float64→float32派生，不重绑定、不重新仿真、无逐帧删面。
- 原衣物使用其独立face-corner UV、texture_01 diffuse/normal。粗糙度0.58、normal strength1是明确的预览配方，不宣称这些标量来自官方材质。身体采用中性哑光预览材质，不宣称还原原皮肤身份。无原配groom和鞋，本轮不新增。

原身体求值代码：[prepare_bedlam_native_walk_a2.py](../../../tools/experiments/prepare_bedlam_native_walk_a2.py)。它只读取现有模型，不安装或全局注册插件。

### 7.3 时间、坐标、root与138点

`asset n = body NPZ n = clothing cache n`，衣物仿真文件帧为 `n+101`，n=0…90。官方 `third_party/bedlam2_render/unreal/import/import_abc_clothing.py` 丢弃100个warmup帧、从101开始；身体ABC导入从1开始。两者127帧与原NPZ对应，且5个代表时刻实际身体/衣物空间对齐支持该映射。此处的不同编号不是再叠101帧位移。

原发布处理后时间表为整数tick0…90、time_base=1/30，首末跨度3秒。原AMASS/KIT动作有100 Hz来源说明，但这里只保留发布后的30 Hz时钟，不宣称精确恢复原100 Hz物理时刻。MP4保留全部91帧，播放时长91/30秒，无复制、丢帧、插值、变速或循环。播放器不自动循环。对应[时间表](../../../outputs/experiments/bedlam_native_walk_a2_20261009/native_attempt_2/time_mapping.csv)与[编码检查](../../../outputs/experiments/bedlam_native_walk_a2_20261009/native_attempt_2/video_validation.json)。

模型缓存采用SMPL-X Y-up米制；原姿态和trans已进入身体网格，同一原root已进入衣物仿真缓存。展示共同施加一次 `C(x,y,z)=(x,-z,y)` 转为Blender Z-up，无额外源root、逐帧贴地、跟随相机或动态缩放。相机与画面范围根据全段共同边界一次冻结，保留真实行进。以后scene也应共同左乘一次，不能再把source trans加到衣物或身体。

全91帧身体、衣物与点数组有限，衣物float32数组与原NPZ选段严格相同，faces/face-corner UV与原数据严格相同，模型faces全段固定且与现有R参考faces一致。5帧的衣物点到身体最近顶点中位数约10.46–10.61mm，仅是空间配对诊断，**不是贴身shell门槛或无穿模证明**。全段相邻顶点位移统计见validation，仅定位，不作为观感自动判决。

本项目 `R[138,10475] @ body` 已计算并回读，最大绝对误差 **2.384185791015625e-7 m**，低于2e-6原容差。但locked-head/no-head-bun模板头部表面与项目R定义的语义适用范围尚未独立认证。故文件命名为 `densekp_3d_pending.npy`，`densekp_label_ready=pending_locked_head_template_semantics`，**不能用于已验收监督标签**；顶点数、faces相同和R乘法成功均不替代语义证据。

### 7.4 实际预览和用户选择

5个探针时刻固定为asset 0、22、45、68、90；两视角共10图。静态查看全部探针联系图，没有发现明确错坐标、错帧或大范围脱体；这不等于连续观感通过。依据探针实测冻结完整阶段预计66.2秒，小于30分钟，复用10图再渲染172图。

实际Blender 5.1.1、Cycles CUDA、每视角512×512、64 samples、不降噪、seed29、animated seed关、adaptive关、AgX；2个固定正交视角、简单3盏area灯及world，无场景。完整设置在 `render_settings.json`。并排MP4为1024×568（含帧号/时间栏），H.264 CRF18、30 Hz。182个PNG尺寸/解码校验、91个视频帧和帧率核查通过。

**未实际连续观看，动态人眼验收待用户进行。** 用户只需判断这段原配行走能否接受，尤其衣物是否明显翻折、断裂、持续闪烁；轻微局部穿模不自动否决。不能仅凭静帧或程序验证宣称原配比已接受APD更好。现有APD、其简单衣物及groom完全不动，本资产也未替换A1诊断行人。

渲染与打包脚本：[preview_bedlam_native_walk_a2.py](../../../tools/render/preview_bedlam_native_walk_a2.py)、[package_bedlam_native_walk_a2.py](../../../tools/validate/package_bedlam_native_walk_a2.py)。

### 7.5 成本、失败及保护

三个成员实际网络取得159.33秒，两个小型索引另约4.63秒；不把开发/阅读/登录等待算成下载或渲染成本。已有本地原模型求值8.196秒；探针阶段2.107秒、完整补齐阶段17.518秒，渲染阶段累计19.625秒，其中实际render调用17.964秒；编码1.493秒。原帧小人物studio非常轻量，**不能外推为道路ERP或正式数据成本**。

即时检查GPU0为18MiB/24576MiB、0%，GPU1为411MiB/24576MiB、0%；选GPU0 UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`。渲染期间自身PID900945约1782MiB；其他Blender PID43491在GPU1，不干预。CPU前检查约141GiB内存可用、12TiB磁盘可用。一次沙箱内GPU记录子进程exit9，直接只读工具检查正常，未视为设备故障。详见 `execution_record.json`。

保留失败：首次网络受沙箱限制、首次认证302未跟随、错误纹理归档名探测被拒，以及 `native_attempt_1` 因Blender append未设置active object而在求值循环前失败；后者仅修正读取器选择原对象，再用原冻结配方成功。没有失败渲染、衣物重绑定、模型调参或换片段。失败求值的单独秒数未计量，不编造为零。

当前全部新增/提取/派生产物约104MiB，低于1GiB；精确结束快照在 `delivery_validation.json`。不复制344.98MB模型，不存整包，不删原图/旧产物；保留182张小型诊断PNG方便定位。未计算SHA256、提交commit、安装依赖或训练/推理。

保护比较显示TODO、README及既有APD accepted manifest等保持原size/mtime；执行期间 `docs/history/prompt.md`、A1 `sequence_packing.py` 与 `render_road_pass_sources.py` 的快照发生变化，本任务未写这些文件，也未回退它们。此处不冒认并行变化归属，记录见 `protected_comparison.json`。

### 7.6 A1交接与停止点

实际manifest仍是显式版本的资产侧提案，`A1_schema_effective=false`、`public_consumer_tested=false`。当前只读消费者已可读取逐帧 `vertices.npy` / `clothing_vertices.npy`，但仍要求character appearance、simple-shell材质、groom/head矩阵，以及31场景帧配方；本任务不伪造这些字段。

需要A1接入时明确：直接消费原配衣物缓存及独立UV/材质，允许无groom和明确预览body材质，使用真实30 Hz/91帧时间表。这一组合若采10 Hz可以显式选0,3,…,90得到31帧，但**本轮未做这种采样或公共消费**，也不伪造原100 Hz映射。原root已烘焙，scene统一放置一次；R模板语义确认后才标监督就绪。需要语义证据而不是再下载模型来凑顶点数。

当前已找到完整的**身体＋动作＋原配衣物＋UV/纹理**组合并完成真实连续预览，未提供原配头发/原皮肤身份材质，不能称整套官方人物外观全复现。所有产物integration_only，继承BEDLAM2 body family、KIT425/walking_medium05来源及既有项目划分；邻段、换衣/seed不生成独立验证来源。下一步是用户观看选择，再评审R语义和公共消费者能力；停在这里，不启动motorcycle、DS/ERP或正式数据生成。
