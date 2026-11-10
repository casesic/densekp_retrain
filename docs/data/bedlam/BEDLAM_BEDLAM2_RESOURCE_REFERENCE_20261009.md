# BEDLAM / BEDLAM2 资源参考与 motorcycle 专项核查

核查日期：2026-10-09。用途：DenseKP 后续人物、动作、穿衣资产选型与来源追溯。**这是资源研究，不是新动作或穿衣质量验收。** 本轮不改变已接受的 APD6526 身体、简单衣物和 groom，也不恢复已暂停的 BEDLAM 换衣试制。

证据标记：**官方**＝公开项目页、论文或官方代码说明；**本地实查**＝本轮读取文件、索引、数组头；**历史记录**＝既有审计/下载说明，本轮未重做其计算；**待核查**＝没有相应实际文件、映射或连续观看证据。资料中的数量与大小是对应发布版本的统计，不是本机下载量。

机器清单：[本地库存](../../../outputs/audits/bedlam_resources_20261009/local_inventory.json)、[motorcycle 五条完整记录](../../../outputs/audits/bedlam_resources_20261009/motorcycle_candidates.json)、[候选 CSV](../../../outputs/audits/bedlam_resources_20261009/motorcycle_candidates.csv)、[网页与访问状态](../../../outputs/audits/bedlam_resources_20261009/source_evidence.json)、[执行记录](../../../outputs/audits/bedlam_resources_20261009/execution_record.json)。复查脚本：[audit_bedlam_resources.py](../../../tools/validate/audit_bedlam_resources.py)。

## 1. 对当前项目最有用的结论

1. **走路人物可以优先整套复用 BEDLAM 的原身体、原动作和配套衣物缓存。** 这样能保留已经做过的衣物仿真，不必把别人的衣服重新绑到 cycling_M01。仍需对齐帧、坐标、模型和材质，并由用户观看短段；“数据集有这个组合”不能替代观感判断。
2. **motorcycle 在 BEDLAM2 训练映射中有 5 个身体—动作组合，只有 4 个不同原动作文件。** 全来自 BMLrub，索引时长约 4 秒。已定位配套服装，其中一个在 Starter Pack 白名单内；本轮没有读到这五条的实际 motion NPZ 或配套衣物，也没有观看其连续动作，因此尚不能认定它们适合自行车蹬踏。
3. **BEDLAM 不是只有图像，也不是一套可直接换任意动作的通用蒙皮服装库。** 它同时提供身体参数、指定身体/动作的衣物仿真缓存、贴图和渲染工具；BEDLAM2 还提供曲线头发和鞋的位移贴图。缓存复用与重新绑定是两种不同工作量。
4. **已有本地行走 NPZ 不等于已有原版穿衣行人全套。** `us_2495_L/2200` 的原配服装是 `nt_belle_007`，其完整衣物缓存尚未在本轮限定路径中确认；现有项目行人使用源动作驱动本项目角色，与原版整套复用不同。

## 2. 两代资源的区别

| 项目 | BEDLAM，CVPR 2023 | BEDLAM2.0，NeurIPS 2025 | 对我们的影响 |
| --- | --- | --- | --- |
| 图像序列 | 10,450，30 Hz，1280×720 | 27,480，30 Hz，1280×720 | 是官方透视渲染，不是本项目 DS/ERP 成品 |
| 身体模型 | 仿真/渲染：SMPL-X v1.1 male/female，10 betas；发布 GT：neutral，11 betas | neutral、locked head、no head bun，16 betas | 不可混用模型后仍声称与源衣物精确配对 |
| 深度 | EXR 32-bit float | EXR 16-bit float；官网列出覆盖约 44% 图像 | 不能假定每张 RGB 有深度 |
| 相机 GT | CSV | CSV + JSON | 还要读取坐标转换与中心时刻定义 |
| 发型/鞋 | 图像含头发；具体发型资产是否可再分发需逐项核实；无 B2 的鞋增强 | strand groom；鞋颜色/法线/位移贴图 | 头发需头部绑定；鞋不是随手附加一个独立鞋网格 |
| 便捷资产包 | 按原项目下载目录组织 | Starter Pack：51 个 body shape、150 个动作组合，预制 Unreal uasset | 方便官方 UE 流程；不能直接把 uasset 当 Blender NPZ |

模型差异来自[官方 FAQ](https://bedlam2.is.tuebingen.mpg.de/faq.html)；图像与资产发布范围见[BEDLAM](https://bedlam.is.tuebingen.mpg.de/)和[BEDLAM2](https://bedlam2.is.tuebingen.mpg.de/)。不要由 body ID 末尾 `M` 推断男性，它是尺码；实际性别元数据另有表，B2 求值模型仍为 neutral。

论文附录统计 B2 的动作池 4,643、身体 1,615、服装 187、鞋 182、发型 40，身体/动作/衣物组合 10,592；动作片段通常 4–16 秒。这里“动作池”“重定向组合”“渲染序列”是不同计数。论文前文还出现“超过 4K body shapes”的表述，与附录的 1,615 不一致；本参考采用附录口径，保留这一差异，不自行补成精确统一值。[论文 §7.2、§7.9](https://arxiv.org/html/2511.14394v1#A7.SS2)

## 3. 到底有哪些 motion

### 3.1 原动作来源与处理

B2 包含 AMASS 的若干子库，并加入 MOYO、BEAT2。来源涉及普通移动、跑跳、运动/舞蹈、过渡动作、姿态变化、瑜伽及讲话手势；它不是以“走路、骑车”分类并逐类保证质量的成品动作包。官方重定向会适配目标身体；手部也可能用额外来源增强，不能把所有自由度都视为原始动捕。[官方重定向代码](https://github.com/PerceivingSystems/bedlam2_retargeting)、[资产处理代码](https://github.com/PerceivingSystems/bedlam_clothing)

本轮实查训练映射 `b2_bedlam2amass_motion_training.csv` 共 **9,951 行**，按 `(ds_name, motion_rel_fname)` 去重为 **4,046 个源文件**。这是本地这一版训练映射，不代表全数据集仅有 4,046 个动作。每行包含 body_id、motion_id、来源子库、源文件、转换后区间以及原始帧率/区间。

| 来源 | 训练映射行数 | 来源 | 训练映射行数 |
| --- | ---: | --- | ---: |
| CMU | 3137 | KIT | 2513 |
| BMLmovi | 1247 | HDM05 | 631 |
| BMLrub | 509 | Transitions | 378 |
| BEAT2 | 332 | ACCAD | 314 |
| MoSh | 286 | SOMA | 257 |
| MOYO | 154 | PosePrior | 126 |
| DFaust | 67 | 合计 | 9951 |

### 3.2 可用于选候选的名称检索，不能当正式动作统计

| 文件名子串 | 匹配的身体—动作行数 | 例子/解释 |
| --- | ---: | --- |
| walk | 911 | `walking_medium05`、`walk_001`；也会命中过渡动作 |
| run | 237 | `run_001`、`jumpingtwist_running` |
| jog | 1 | `treadmill_jog` |
| jump | 260 | 跳跃及其过渡 |
| dance | 128 | 舞蹈及其过渡 |
| sit | 39 | 只表示名字匹配，不保证截取段含可直接使用的坐姿 |
| stair | 1 | `KIT/359/downstairs07_stageii.npz`；仍需看片段 |
| motorcycle | 5 | 详见下一节 |
| bicycle / cycling / bike | 各 0 | 仅这一索引文件名未命中，不证明全库无骑行动作 |

这些规则可重跑，但类别会重叠，数字不应相加；CMU 等数字文件名还会漏掉很多语义。论文说明曾过滤需要外部物体支撑或不平地面的动作，而本地表仍有 sit/downstairs 命名。可能涉及截取或筛选阶段，**本轮未核实具体保留内容**。因此修正此前较宽泛的口头分类：名称只能帮助寻找候选，不能宣称已验证“坐下/上下楼”成品动作。

MOYO 是重要例外：论文说明其复杂姿态使用贴身纹理表示衣服，未逐条做服装仿真。不能因为条目在 motion 表里，就假定必有松衣缓存。[论文 §7.5、§7.9](https://arxiv.org/html/2511.14394v1#A7.SS9)

## 4. motorcycle：已经查到哪一步

### 4.1 精确条目、原动作和服装

下表由本地训练映射、`motion2outfit_training.json`、`gender_training.csv` 和两个动画白名单交叉取得。五条都在训练动画白名单中；这不等于已取得服装人工评分。

| B2 body_id / motion_id | BMLrub 原文件 | 身体性别元数据 | 原配 outfit | Starter Pack |
| --- | --- | --- | --- | --- |
| `us_2444_4XL / 2000` | `rub060/0004_motorcycle_stageii.npz` | m | `gr_aaron_009_4XL` | 否 |
| `us_1282_4XL / 2103` | `rub011/0004_motorcycle_stageii.npz` | m | `gr_corey_002_4XL` | 否 |
| `us_2173_2XL / 2314` | `rub057/0004_motorcycle_stageii.npz` | f | `gr_alexandra_006_2XL` | 否 |
| `us_2467_4XL / 2305` | `rub011/0004_motorcycle_stageii.npz` | m | `gr_ben_008_4XL` | 否 |
| `it_4191_M / 2400` | `rub094/0004_motorcycle_stageii.npz` | m | `nt_corey_006` | **是** |

每条 `start=0,end=120,orig_fps=120,orig_start=0,orig_end=480`，索引含义支持约 4 秒、目标约 30 Hz 的片段。未读取实际参数，**不把区间推算冒充已验证的数组帧数/时长**；若确为 120 个均匀采样，首末跨度是 119/30 秒，和 120/30 秒播放占时不同。

两条 `rub011` 是同一个原动作换了身体，后续必须同源分组；其余不同 rub 编号也不自动构成独立验证集。四个 `gr_*` 服装各有 10 个 texture ID 选择，`nt_corey_006` 有 15 个；这是配置选项数，不是本机已具备这些贴图。

依据：[官方配对表](https://github.com/PerceivingSystems/bedlam2_render/blob/main/config/motion2outfit_training.json)、[Starter Pack 白名单](https://github.com/PerceivingSystems/bedlam2_render/blob/main/config/whitelist_animations_starterpack.json)，本地完整行见候选 JSON。

### 4.2 统计透露了什么、不能透露什么

官方仓库的 `motion_stats_training.csv` 有 9,797 行，和 9,951 行映射数量不同；不默认两表一一覆盖。本次五条均能关联。

| body / motion | ground_coverage_x | ground_coverage_z | pelvis_height_min–max |
| --- | ---: | ---: | ---: |
| us_2444_4XL / 2000 | 0.008905 | 0.022180 | 0.871700–0.878646 |
| us_1282_4XL / 2103 | 0.006361 | 0.011353 | 0.865494–0.867618 |
| us_2173_2XL / 2314 | 0.005409 | 0.010566 | 0.797888–0.800828 |
| us_2467_4XL / 2305 | 0.006718 | 0.011992 | 0.914419–0.916662 |
| it_4191_M / 2400 | 0.003366 | 0.004845 | 0.765227–0.767192 |

`filter_animations.py` 把厘米阈值除以 100 后与这些字段比较，说明消费端按米制使用。**本轮没有找到统计生成公式**，因此保留原字段名，不能擅自解释成全身包围盒、脚的活动范围或精确 pelvis 水平极差。它们提示空间覆盖较小，不能证明静态，也不能证明没有蹬踏。[统计表](https://github.com/PerceivingSystems/bedlam2_render/blob/main/stats/motion_stats_training.csv)、[筛选代码](https://github.com/PerceivingSystems/bedlam2_render/blob/main/tools/animations/filter_animations/filter_animations.py)

### 4.3 找到的公开人眼预览入口：尚未与上述条目配对

本轮读取 BABEL 官方 Explore 页面实际使用的公开 `ride_babel.json`，共 5 条。下列是其提供的远程预览链接；**没有下载或播放视频**，标签仅转述元数据。这些通常是动作的身体可视化，不应称为原始实拍录像。

| BABEL 公开预览 | 标注 | 标注时间区间（秒） |
| --- | --- | --- |
| [008265.mp4](https://babel-renders.s3.eu-central-1.amazonaws.com/008265.mp4) | riding a bike | 0–3.97 |
| [006007.mp4](https://babel-renders.s3.eu-central-1.amazonaws.com/006007.mp4) | riding motorcycle | 0–2.43 |
| [007288.mp4](https://babel-renders.s3.eu-central-1.amazonaws.com/007288.mp4) | riding a motorcycle | 0–4.00 |
| [010670.mp4](https://babel-renders.s3.eu-central-1.amazonaws.com/010670.mp4) | ride motorcycle | 0–3.97 |
| [007924.mp4](https://babel-renders.s3.eu-central-1.amazonaws.com/007924.mp4) | ride motorcycle | 0–2.27 |

公开类别 JSON 只有标签、时间和 video_url，没有 AMASS `feat_p` 源路径。**不能按时长或“四条摩托车”把它们硬配到四个 BMLrub 文件。** 完整 BABEL 注释的源路径才是配对依据；本轮数据下载入口要求登录，未获取受限注释、未绕过权限。

入口：[BABEL Explore](https://human-movement.is.tue.mpg.de/explore/)、[页面所用公开类别 JSON](https://human-movement.is.tue.mpg.de/data/cat_dsets/ride_babel.json)。本轮保存了 7,517 字节 HTML 和 1,571 字节 JSON，见审计目录；公开能播放不代表可自由再分发。

### 4.4 对 motorcycle 的判断与最小下一步

**值得查看，但尚不具备替换 APD 的证据。** 摩托车姿态与自行车交替屈伸不能凭名称画等号；目前没有脚运动、膝角连续性、坐姿变化或穿衣质量的实际观察。用户已放宽精确踏板约束，因此后续以是否看起来合理为主要选择依据，不预设必须满足两个可认证周期。

若后续选择验证一个 B2 原配组合，优先 `it_4191_M/2400 + nt_corey_006`：理由是组合明确且列入 Starter Pack，**不是已知它的动作最好**。先确认对应 motion、模型、服装 NPZ/ABC、UV/贴图及许可齐全，再做原配身体连续预览供用户观看；若只有 uasset，要先查同组合的独立衣物分片，不要求为此先安装 Unreal。本轮停在资料核查，不发起下载或适配。

## 5. 资源目录：分别拿到的是什么

### 5.1 身体、动作、表面与形状

| 资源 | 表示/关系 | 使用注意 |
| --- | --- | --- |
| SMPL-X motion | NPZ：poses、trans、betas、gender、帧率等；原动作与 B2 body/motion 通过 CSV 对应 | 参数文件不包含 SMPL-X 模型许可和权重；先核对模型变体 |
| 渲染身体 | 从参数求值，再导出 Alembic GeometryCache；pose correctives 可烘焙 | 缓存已含的 pose/root 不可再施加一次 |
| 身体纹理 | 皮肤颜色贴图及服装 overlay；B2 渲染代码采用 UV map 2023 | UV、faces_uv、肤色纹理版本须匹配；仅有同样顶点数不够 |
| 发布 GT | 供学习/评价的参数、投影和场景元数据 | B1 neutral GT 与仿真 male/female 原身体要区分；不能随意互换来驱动原衣物 |

本地 B2 行走 NPZ 实查：`poses[127,165]`、`trans[127,3]`、`betas[16]`、`gender=neutral`、`mocap_frame_rate=30`。B1 衣物示例对应 motion：`poses[192,165]`、`global_ori[192,3]`、`trans[192,3]`、`betas[10]`、male、30 Hz。**这是两个样本的实际结构，不是保证全库每个 NPZ 完全同构。**

B2 模型说明与导出流程见[官方 render README](https://github.com/PerceivingSystems/bedlam2_render#render-pipeline)。B1 流程见[bedlam_render](https://github.com/PerceivingSystems/bedlam_render)。

### 5.2 衣物：缓存、贴图和制作资产要分开

| 层次 | 已确认的资源/形式 | 能直接做什么 | 不能默认做什么 |
| --- | --- | --- | --- |
| 仿真缓存 | B1 本地有 NPZ、OBJ；官方渲染消费 ABC；B2 下载历史说明有分片 NPZ，渲染器消费 ABC | 在原身体、原时间上播放已经仿真的衣服 | 更换 shape/pose 后原样贴上仍无穿模 |
| 网格与 UV | B1 实例有 vertices_seq、faces、vertices_uv、faces_uv | 固定拓扑和 face-corner UV 的逐帧消费 | 用 body UV 替代 garment UV，或逐帧删面隐藏问题 |
| 材质/纹理 | 本地原 diffuse/normal，B2 outfit→多个 texture ID 配置 | 在正确 UV 上换色/花纹、保持原材质关系 | 仅改简单 shell 颜色就称为复用原服装 |
| 可编辑制作源 | 官方有衣物制作/处理工作流 | 为后续制作提供参考 | 本轮未核实每件服装均提供 ZPRJ、官方 rest mesh、蒙皮权重；不能保证有 |
| 配对/质量表 | motion2outfit；另有服装仿真人工评分 whitelist | 选指定组合和更可靠候选 | 动画白名单不等于服装评分表；未取得分数不能编造 |

B1 本地衣物实例实际为 39 帧、13,027 顶点、26,328 面、14,776 UV 点，UV 面与几何面分开。历史审计记录其 6 Hz 缓存对应 30 Hz 身体每 5 帧采样；本轮读取数组头，未重新解包 object metadata 或重算几何。沿用该对齐时须引用历史映射，不能把 39 帧改名为 30 Hz 动画。

B2 历史下载说明列 `b2_clothing_npz_[000-275].tar` 与 `assets/clothing/npz/archive_map.json`。这意味着可以先定位成员属于哪一分片；本轮未获得完整 archive_map、各分片大小或 motorcycle 对应成员，不建议盲下 276 个包。官方入口：[B1 衣物](https://bedlam.is.tuebingen.mpg.de/clothingsim.php)、[B2 下载](https://bedlam2.is.tuebingen.mpg.de/download.php)。入口可能要求登录，本文未验证未登录即可直接取包。

### 5.3 头发、鞋、环境与灯光

| 资源 | 已确认范围 | DenseKP 对接重点 |
| --- | --- | --- |
| B2 groom | 本地归档索引 40 个 Alembic；已解出的 short-casual、curly、short-bob 可定位 | 曲线/点固定集合、头部 reference 与 head transform；groom 文件本身不等于任意动作的动态仿真 |
| 鞋 | B2 以颜色/法线/位移贴图施加到去趾模板，有每鞋高度 offset 表 | 不与足部原模板混用；高度偏移要记录所有权；不把修改后的足表面标签自动当原模板定义 |
| 环境 | 官方列 3D 场景与第三方来源；HDRI 与完整场景不同 | 不保证第三方场景全打包或同许可，具体模型、材质、HDRI 分别引用 |
| 相机动作 | be_camera_animations.json、逐帧相机 GT；B2 有多种相机运动 | 可参考，但不替代我们自己的双相机 rig 和 20260320 标定 |

鞋在本地 `bedlam_clothing/shoes/README.md` 和代码中是特殊模板＋材质流程，而非现成独立鞋动画。`support_data` 下当前只有下载脚本，未找到其约定位置的 locked-head 模型、toeless OBJ 和鞋贴图；不能把仓库存在说成资源已齐。官方实现入口：[资产处理仓库](https://github.com/PerceivingSystems/bedlam_clothing)、[渲染导入流程](https://github.com/PerceivingSystems/bedlam2_render/tree/main/unreal/import)。本地 README/实现可能有项目内整理，未保证逐字等同当前上游。

### 5.4 已渲染图像、视频、深度、mask、GT

| 项目 | 已确认的发布/代码能力 | 适合用途与缺口 |
| --- | --- | --- |
| RGB PNG / MP4 | 两代均有，B1 还发布论文用 6 fps 图像子集 | 浏览动作或用官方数据；不能直接变成我们的 ERP 监督 |
| 深度 | B1 EXR32；B2 EXR16、部分覆盖 | 缺失深度不能以黑图冒充；数值语义还需读取消费者 |
| Mask | B1 官网页列身体/衣物/头发/环境二值 PNG；B2 渲染/后处理支持 cryptomatte→mask | B2 代码能生成不等于下载包对每帧都提供独立 mask，需查具体包 |
| 相机/人体场景 GT | be_seq.csv、camera CSV/JSON、motion NPZ、相机动画设置 | 必须关联人物实例、帧偏移、场景变换和相机时刻 |
| Normal | B2 后处理支持在所需 EXR 信息存在时生成 | 不能据代码功能宣称全库已有 normal PNG |
| 预处理训练标签 | 历史下载说明有 6 fps CameraHMR NPZ；B1 有转换 SMPL GT | 不等同本项目 DenseKP138 或 A1 序列契约 |

官方 B2 网页列 RGB 约 11 TB、视频约 160 GB、深度约 15 TB。它们是全量资源规模参考，不是当前动作试验的必要下载量。2026-02-26 两代官网均宣布图像/深度 Hugging Face 镜像，仍从注册下载说明进入；不是无条件公开再分发许可。[B2 资源入口](https://bedlam2.is.tuebingen.mpg.de/#dataset)

### 5.5 代码、Starter Pack 与模型权重

| 入口 | 作用 | 本项目是否必须 |
| --- | --- | --- |
| [bedlam_render](https://github.com/PerceivingSystems/bedlam_render) | B1 Blender 预处理＋Unreal 5.0 渲染 | 读取 NPZ 不必运行完整 UE 流程 |
| [bedlam2_render](https://github.com/PerceivingSystems/bedlam2_render) | B2 预处理、序列生成、UE5.3.2 渲染、后处理；官方要求 Windows 工作流 | 参考坐标、配对表、缓存消费即可；不替换 A1 road_pass |
| [bedlam2_retargeting](https://github.com/PerceivingSystems/bedlam2_retargeting) | 动作到不同身体的重定向 | 复用原组合无需重新重定向 |
| [bedlam_clothing](https://github.com/PerceivingSystems/bedlam_clothing) | 动作、鞋和衣物相关资产处理 | 工具不包含所有模型和资产 |
| [ML 数据处理](https://github.com/pixelite1201/BEDLAM/tree/master/data_processing) | 图像、参数、投影与标签整理 | 可核对 GT，不照搬其点定义为138 |
| [CameraHMR](https://github.com/pixelite1201/CameraHMR/blob/master/docs/bedlam2.md)、[GVHMR_BEDLAM2](https://github.com/mkocabas/GVHMR_BEDLAM2)、[PromptHMR](https://github.com/yufu-wang/PromptHMR) | 论文模型、训练/评价代码或 checkpoint | 已有参数和衣物缓存的资产接入不需要这些推理模型 |

Starter Pack 提供 body/clothing、纹理、hair、shoes、HDRI 的 uasset 子集，能跳过官方 ABC 预处理与导入；仍需其 Unreal Core Assets、插件、样例项目以及相应序列输入。**51/150 是子集，并不是 150 类动作或 150 个独立演员。** 本地只验证其配置索引，未验证整包已下载。[Quickstart](https://github.com/PerceivingSystems/bedlam2_render/blob/main/unreal/render/unreal_quickstart.md)

## 6. 本地实际持有：可复用与缺口

只检查既有 manifest 指向的目录、指定文件和已有成员清单；没有扫描整盘，也没有重新展开大 TAR。精确字节数和数组头见 `local_inventory.json`。

| 本地资源 | 本轮状态 | 限制 |
| --- | --- | --- |
| `outputs/experiments/source_isolated_pilot_20260930/sources/us_2495_L_2200.npz` | 118,606 B，127 帧 B2 行走参数 | 本次未求值；原配衣物未确认本地拥有 |
| 同目录映射 CSV | 9,951 行；motorcycle 五条可追溯 | 索引不是 motion 本体 |
| `/mnt/data1/yy/home/yy/datasets/BEDLAM/motion_data/tiny/rp_aaron_posed_009/1095/motion_seq.npz` | 267,098 B，192 帧 | 源为 KIT/883/wipe_arm_dabbing01，metadata 类别 stretch；不是走路 |
| 同一 B1 角色 `clothing_simulations/1095/1095.npz` | 5,967,101 B，39 帧几何＋UV | 原动作缓存；不等于通用 rest 服装 |
| B1 `clothing_npz_6fps.0.tar` | 28,151,562,240 B，存在 | 旧成员索引列 1,792 NPZ + 1,792 OBJ；未逐件读取/验收 |
| B1 `clothing_textures.0.tar` | 4,870,778,880 B，存在 | 复用已有成员索引，没有全解压 |
| B1 body texture ZIP | 1,906,155,828 B，存在 | 不复制、不把贴图许可泛化到人体/衣物 |
| `assets_external/bedlam2_pilot/b2_groom_abc.tar.gz` | 1,335,726,114 B，已有 40 ABC 成员清单 | 本轮未重新读取归档曲线 |
| B2 已解出的 ABC | curly 104,340,124 B；short-bob 77,647,013 B；short-casual 65,187,310 B | short-casual 位于已有 cycling 候选目录，避免重复解压 |
| B2 locked-head 模型、toeless 模板、鞋贴图约定位置 | 不存在，只有 support_data 下载脚本 | 不等于全机器不存在；也不说明本项目原 SMPL-X 模型不存在 |
| `/mnt/data1/yy/home/yy/datasets/BEDLAM2` | 此路径不存在 | 不据此否认 repo 内已保存的 B2 motion/groom |

本地 B1 tiny 图像与 camera/be_seq 曾审计 24 图、186 人物行、20 个序列，需要 157 个 motion 路径；这些是[历史帧映射报告](BEDLAM_TINY_FRAME_MAPPING.md)的结果，**本轮没有重跑这 24 图的全链路，也没有把旧“缺 motion”状态当今天的全盘结论**。

`us_2495_L/2200` 的官方 body 性别表为 f、求值 NPZ 为 neutral，原配衣物 `nt_belle_007`；二者不冲突。其源是 `KIT/425/walking_medium05_stageii.npz`。当前用它驱动 cycling_M01 等目标角色，是项目自己的重定向/消费选择，不能标成 BEDLAM 原版人物。

## 7. 时间、坐标、标签：复用时最容易错的地方

### 7.1 图片帧、动作帧和缓存帧不总相同

按 B2 FAQ，先在 be_seq.csv 找 sequence，再找 Body 与 `start_frame`；30 Hz 动作索引为 `start_frame + image_index`。单人常从 motion 第 1 帧开始对应图像 0000，多人偏移可能不同。运动模糊需中心时刻前后样本，所以渲染长度会短于源动作。不能把 image 0000 一律对到 motion 0。[官方帧映射说明](https://bedlam2.is.tuebingen.mpg.de/faq.html)

本地 `bedlam_clothing/motion/framerate.py` 对可整除帧率按步长抽取；100→30 使用每 3 帧、59→30 每 2 帧、200→30 每 7 帧、250→30 每 8 帧。这些非整除分支不能保证严格原时间不变。因此旧行走文件的“源100 Hz、目标30 Hz”不能直接承诺精准原速。motorcycle 的120→30可整除，但实际参数仍未读取，映射不能仅凭规则盖章。

缓存 fps、源整数帧、截取区间、采样策略、图像中心时刻分别保留。B1 6 fps 图片的文件编号也不能不看规则就除以5；以实际 filename 与 GT 行为准。后续若重采样，身体、衣物、头发必须共用一张时间表。

### 7.2 坐标与 root

NPZ 的 SMPL-X 取向、AMASS 原数据世界方向、Blender 与 Unreal 轴向不是同一个概念。现有 B2 下载说明明确提到 SMPL-X orientation；不能因为来源叫 AMASS 就假定每个转存 NPZ 均为同一 Z-up 世界。消费代码、导出变换与场景米/厘米必须一起检查。

原配身体与衣物缓存应采用同一 `scene @ model`。已烘焙到缓存的 source translation/global orient 不再重复施加；groom 的 `scene @ head` 也只应用一次。若保留原行走 root，就让场景只做固定放置/明确剩余轨迹；若改原地动作，要同时处理衣物与头发，不能只对身体减 root。

APD 当前 pelvis 固定、时间略不均匀，和上述 B2 行走不同；本次目录研究不改变既有[骑行接口交接](../CYCLING_SCENE_INTERFACE_A2_20261009.md)。

### 7.3 DenseKP138 仍从正确身体顶点生成

本项目既有 R 是 `[138,10475]`。BEDLAM 处理链中出现的 437 点算子不是同一标签，不能把437任意选取/插值冒充138；`proj_verts` 的第三分量也不能未经核实当置信度。已有[算子审计](BEDLAM_OPERATOR_AUDIT.md)可复用。

后续取得 B2 locked-head/no-bun/toeless 变体后，需要检查顶点顺序、拓扑和关键点表面语义。顶点数量相同不等于头部/足部位置定义未变；衣物、头发或鞋的外表面不能替换 R 所依赖的身体。本文没有执行该模型兼容性验证。

## 8. 许可与来源隔离

BEDLAM/BEDLAM2 数据许可为免费、非商业科研/教育/艺术用途的受限许可，要求注册与引用，并限制转让和再分发；不是 CC0，也不因某个工具采用开放代码许可就允许自由发布全部资产。B2 官方 render 仓库本身也使用其科研许可，不应一概叫 MIT。许可还对修改及向第三方提供作出限制：**本资料核查不替代派生资产发布、商业使用或训练用途的许可确认**。[B1 许可](https://bedlam.is.tuebingen.mpg.de/license.html)、[B2 许可](https://bedlam2.is.tue.mpg.de/license.html)

| 对象 | 需保留/核对的依据 |
| --- | --- |
| 图像、身体参数、衣物、groom | 对应 BEDLAM 版本许可、取得时间、文件 ID、论文署名 |
| SMPL-X 模型文件 | SMPL-X 自身许可与模型来源；参数文件不替代模型授权 |
| AMASS / BABEL | 原子库/原文件、各自使用条款；公开预览不等于任意再利用许可 |
| 身体纹理、鞋、环境、HDRI | 各资源原许可和派生包装限制；不能只引用总项目许可或一张 CC0 HDRI |
| 后续训练/发布 | 明确用途及输出范围；本报告不声称所有下游训练、公开派生数据或商用都已获准 |

在本项目中，换 body shape、衣服、纹理、seed、邻段、重采样或循环都不自动形成独立动作来源。B2 的同源 motion 及其衍生版本应同组；还需保留 target body family、outfit 来源。官方保留测试参数不发布，不能从原 AMASS 还原测试源来伪造合法独立训练材料。[官方 split 说明](https://arxiv.org/html/2511.14394v1#A7.SS9)

署名至少包括 Black, Patel, Tesch, Yang 的 BEDLAM（CVPR2023）或 Tesch 等的 BEDLAM2.0（NeurIPS2025），并继承实际使用的 SMPL-X、动作子库、皮肤、鞋和环境来源；BABEL 注释另引用其项目。仅列论文名不能代替接受数据许可。

## 9. 质量限制与后续选型建议

官方承认部分地面不平导致脚穿地/悬空；服装仿真人工评分 0–6，低于3的组合已丢弃，但余下仍可能穿模或抖动。本轮未取得各候选评分，不能把训练动画 whitelist 当该评分。[Known issues](https://bedlam2.is.tuebingen.mpg.de/knownissues.html)

官方[具体渲染问题表](https://bedlam2.is.tuebingen.mpg.de/media/upload/b2_render_issues.txt)还记录贴图、曝光/噪声、出画/遮挡、人物相交及个别动作问题。选择数据时结合 renderjob Notes；原配缓存也需要用户看连续动作，轻微穿模可按我们的目标接受，不要求零相交。

| 需求 | 推荐最小路线 | 尚需实际取得/验证 |
| --- | --- | --- |
| 快速加入自然穿衣行人 | 一套原 B2 body + motion + 对应 outfit/cache，先保持原 shape/root/速度 | motion、匹配模型、衣物成员/UV/贴图、时间与坐标、短段人眼预览 |
| 查 motorcycle 是否可用 | 先看公开候选；取得精确原文件关系后，优先一个 Starter Pack 配对组合 | 公开预览与 BMLrub 路径对应尚缺；实际参数及连续质量待查 |
| 继续当前骑行 | 保留已接受 APD6526 + 简单衣物 + 现有 groom | 不因资源研究重开换衣试制；A1 公共消费能力另按交接推进 |
| 仅增加发型 | 从已持有 groom 库选一个，再做固定头皮/头部绑定 | reference、单位、有效集合、动态观感；不必先做物理毛发 |
| 增加鞋 | 先独立评审特殊模板与鞋材质对身体/R138 的影响 | 模型/template/maps 当前约定位置缺失，不作为走路接入前置 |

本轮没有提出全量下载、完整 Unreal 管线迁移或新重建模型。对于走路，**先找齐一套原配组合比扩大衣服库更有价值**；对于 motorcycle，**先确认具体动作看起来是否合用比按名称替换 APD 更有价值**。人眼选择是后续动作/穿衣观感的决定依据，数值检查负责发现坏文件、错帧、错变换等明确错误。

## 10. 本轮执行与可复查入口

已读 Git 状态和适用指令路径；本轮只新增本参考文档、独立核查脚本及审计目录。不写用户 TODO、README、历史报告、A1 模块/测试、角色源资产或外部 pano-stitch。并行 A1 的未提交修改保留，不回退。

执行内容为官方网页/论文/代码检索、本地 CSV/JSON 关联、NPZ 数组头和小型标量读取、既有归档成员索引复用。没有解压大包、下载视频/模型/衣物、求值人体、渲染、拟合、训练、安装依赖、SHA256 或 commit；没有占用 GPU。网络新增仅 BABEL 页面及类别 JSON 两个小文件。

复查时输出新目录，避免覆盖本次证据：

```bash
python tools/validate/audit_bedlam_resources.py --output /tmp/bedlam_resources_recheck_20261009
```

脚本只检查指定路径，不保证盘上没有其他副本；文件存在/数组维度/配对成功都不是动态质量通过。motorcycle 仍为 **candidate_unviewed**，本报告不构成新的 train/val 划分或骑行动作替换决定。
