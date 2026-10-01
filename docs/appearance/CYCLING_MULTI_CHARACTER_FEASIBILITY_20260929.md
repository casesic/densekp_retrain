# 公开多角色资产可行性核查

后续状态：四角色已完成接入，见 [四角色 × 两姿态验收](../rendering/CYCLING_FOUR_CHARACTERS_20260929.md)。下文保留可行性核查当时的候选状态和资源边界。

日期：2026-09-29。范围：本地资源及压缩包目录、已有预览、BEDLAM / BEDLAM2 官方资料。
本轮没有下载新资产、重建、生成角色网格、渲染或训练；没有计算 SHA256。
仅新增本文、只读审计脚本、审计结果和包内原有预览的原样副本；外部项目及旧输出未修改。

## 1. 结论

**现有资源支持设计 2 女 + 2 男的合成角色接入样例，但尚没有四个已组合、已验收的角色。**
推荐最小路线：本地标准 SMPL-X female/male 模型 + 固定的角色 shape + BEDLAM/Meshcapade
脸部和皮肤纹理 + 按新人体重新生成的贴身上衣/短裤。两个骑行姿态均可作为迁移来源。
这里的角色是有稳定配置的合成人物，不宣称复现四位 BEDLAM 扫描对象或原照片人物。

本地已有 100 套皮肤纹理及官方预览；缺口主要是四角色 shape 的实际实例化和验收、
逐角色材质接口、眼部适配，以及独立三维头发。纹理中的头发颜色/发际线可以记录为初版
外观，但**不能把预览里的发型当作已获得的三维发型资产**。如接入样例要求三维发型，
完整接入目前不具备条件，不能用光头/头皮贴图替代后宣布达标。

若必须使用 BEDLAM 发布的四个具体 subject shape，本地已明确定位的具名 shape 只有
`rp_aaron_posed_009`；两位女性及第二位男性的对应 shape 尚缺。服装缓存不补足这个缺口。

## 2. 状态定义和核查范围

四种状态表示组件对当前骑行管线的适用程度，不能用“文件存在”代替“可直接使用”。

| 状态 | 本文含义 |
|---|---|
| 已存在 | 已定位文件/压缩包成员，但目标组合尚未验证 |
| 可直接使用 | 已有目标用途的接口或数值证据；只对指定组件成立 |
| 需要适配 | 文件或实现已存在，但换角色、换姿态、UV/材质/绑定需要处理 |
| 尚未获取 | 在已检查范围内未定位到必要原始资产；代码/名称/预览不算资产 |

搜索范围：`/home/yy/datasets`、本仓库 `assets`、`assets_external`、`third_party`、
`outputs`，以及外部 `pano-stitch/tools` 的模型和资产路径。未声称扫描整台机器所有挂载盘。
本地旧审计中的“没有皮肤/服装”已被随后取得的压缩包和本次清单核查更新；保留历史文件。

本地审计证据位于：

- [归档清单与四张原有预览](../../outputs/audits/cycling_character_feasibility_20260929/)
- [机器可读归档结果](../../outputs/audits/cycling_character_feasibility_20260929/archive_inventory.json)
- [模型、UV 和回归器检查记录](../../outputs/audits/cycling_character_feasibility_20260929/model_compatibility.json)
- [只读归档审计脚本](../../tools/experiments/audit_cycling_character_assets.py)

## 3. 本地资产盘点

路径缩写：`B=/home/yy/datasets/BEDLAM`；
`P=/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools`。

| 组件 | 实际路径/内容 | 状态与使用边界 |
|---|---|---|
| 标准身体模型 | `P/smpl_models/smplx/SMPLX_{FEMALE,MALE,NEUTRAL}.npz` | 可直接使用模型加载接口；新角色仍需生成和几何验收 |
| 原始模型包 | `P/PEAR/assets/models_smplx_v1_1.zip` | 已存在；不能只凭文件名判版本，见下节 |
| 具名男性 shape | `B/motion_data/tiny/rp_aaron_posed_009/1095/motion_seq.npz`，male、10 betas | 已存在；迁移到骑行姿态需要适配，不使用原 motion 衣物 |
| 未具名示例 shape | `third_party/bedlam_render/blender/smplx_anim_to_alembic/smplx_animation.npz` | 已存在；male、10 betas，缺少独立人物身份依据，不计第二个男性 |
| 原照片姿态 | `outputs/experiments/cycling_smplx_20260929/<sample>/canonical_pose.npz` | 可直接用作两份独立姿态来源；原 neutral shape 保留，不复制给 gendered 模型 |
| 皮肤/脸部 | `B/body_textures/bedlam_body_textures_meshcapade.zip`，1,906,155,828 bytes | 已存在；214 个目录/文件成员、50 女 + 50 男 albedo，版本说明 20220207 |
| 已接入皮肤 | `B/body_textures/tiny/rp_aaron_posed_009/skin_m_white_01_ALB.png` | 可直接复用既有皮肤材质入口；不能代表所有新角色眼部已正确 |
| 身体 UV | `P/PEAR/assets/SMPLX/smplx_uv.obj` | 可直接使用其标准拓扑的 face-corner UV；具体贴图接缝/眼睛需要检查 |
| 原始眼贴图 | 上述 ZIP 的 `eye/SMPLX_eye.png`，已有 tiny 副本 | 需要适配；此前官方眼贴图与当前眼部 UV 未通过直接兼容检查 |
| 程序化眼部 | `tools/experiments/prepare_procedural_eyes.py`、`tools/render/eye_background_render.py`、`outputs/eye_fix_v2/` | 需要适配；旧单人试验有通过记录，但当前 DS renderer 没有接入该眼部材质分支 |
| 服装几何缓存 | `B/clothing_data/clothing_npz_6fps/clothing_npz_6fps.0.tar`，28,151,562,240 bytes | 需要适配且禁止直接套用；1,792 个 NPZ + 1,792 个 OBJ，23 subject 条目 |
| 服装纹理 | `B/clothing_data/clothing_textures/clothing_textures.0.tar`，4,870,778,880 bytes | 需要适配；262 张 PNG、13 subject 条目，属于原衣物 UV，不能套在当前壳层上 |
| 新姿态贴身衣物实现 | `tools/experiments/prepare_cycling_clothing.py`；已验收示例 `outputs/experiments/cycling_clothing_20260929_v2/` | 需要适配；能重新构造上衣/短裤，但模型写死 neutral/10 betas，需要按角色读取 |
| BEDLAM v1 三维发型 | 官方配置里有名称；本地未找到可复用原始 hair mesh/groom | 尚未获取；官方论文说明 v1 发型受不可再分发许可限制，不能从 RGB/配置恢复资产 |
| BEDLAM2 三维发型 | 官网 `b2_groom_abc.tar.gz`；本地只有导入/绑定代码和白名单 | 尚未获取；不可将脚本存在记成 groom 已存在 |
| BEDLAM2 模型/角色 | locked-head/no-head-bun 模型、`b2_motions_npz_training.tar` | 尚未获取相应原始模型/参数；本地旧模型不能冒充 |
| BEDLAM2 Starter Pack | 官方为 UE 5.3.2 `.uasset`，含 51 shapes / 150 motions | 尚未获取；不是可直接驱动当前 Blender 骑行姿态的通用角色包 |

服装几何分片里的 23 subject 条目，按 `_posed_` 前缀保守归组仅有
`aaron/alexandra/alison/aneko/anna/antonia/ashley/beatrice` 八组；其中男性组只有 Aaron。
纹理分片则是 Aaron、Alexandra、Alison、Aneko 四组。
这是保守防泄漏分组依据，不等同于已经核验的真实人物身份证明。
抽查 Aaron、Alexandra、Alison 的 NPZ 均只有
`vertices_seq/vertices_uv/faces/faces_uv/metadata`，没有 betas；归档文件清单也没有单独身体参数文件。

## 4. 身体版本、UV 和 138 点兼容性

本轮直接比较，三个本地模型均为 10,475 顶点、20,908 三角面；`f` 与当前骑行几何
**逐项一致**。本地 `shapedirs` 为 `[10475,3,20]`；原始 ZIP 对应模型为 `[10475,3,400]`。
每个 gender 的 `v_template`、`f`、前 10 个 shape directions、`posedirs` 均与该 ZIP 内
同 gender 模型逐项一致。它们之间不能跨 gender 复制 betas。

模型包文件名写 `v1_1`，内部 `version.txt` 却写 `Version 1.0`。因此身份记录必须保留
具体模型路径、数组规格及上述直接比较依据，不写成“仅凭名字已确认最新版本”。
官方说明 BEDLAM v1 渲染使用 v1.1 female/male 10 维形状，neutral GT 为 11 维；
BEDLAM2 使用 neutral locked-head/no-head-bun 的 16 维形状。不能截断、补零或跨版本
复制这些 betas。[官方版本说明](https://bedlam2.is.tuebingen.mpg.de/faq.html)

UV OBJ 的几何面与当前几何逐项一致，UV 为 `[11313,2]`，UV 面 `[20908,3]`，
索引合法且数值有限。当前 `R=assets/densekp_regressor_R.npy` 为 `[138,10475]`，
数值有限、行和最大偏差 `5.96e-8`。因此可以沿用当前标准人体的顶点序与语义回归路径，
不改拓扑、不重排点，也不用 BEDLAM 的其他点集代替 138 点。
这些证据不替代新姿态上的语义位置、网格/标签同变换及实际投影验收。

皮肤包 readme 明确：纹理原本为 SMPL UV；用于 SMPL-X 时眼周会有误差，需要相应眼层。
当前 DS renderer 直接连接皮肤 Base Color，不能假定已处理眼部。
旧 `outputs/eye_fix_v2/acceptance.json` 记录程序化眼部方案通过，但四张新皮肤仍需各自审查，
不能把旧白色男性纹理的局部修正无条件套给所有纹理。

BEDLAM2 则使用更新的 UV 2023 和 bald-head 纹理。其头/脚模板改变对 138 点语义的影响
尚未在本地验证；仅有同样顶点数不能证明兼容。若采用 B2，须另作模板/顶点序/回归器支持
区域和 UV 检查。[官方渲染输入说明](https://github.com/PerceivingSystems/bedlam2_render)

## 5. 推荐的四个合成角色候选

以下 ID 从此保留；都标记 **candidate / 需要适配**。女性/男性表示资产配置类别，依据
对应模型和官方纹理目录；不是对照片中真实人物的性别推断。
下面的 shape 是明确可复现的**建议配置**，本轮没有实例化，更没有把 beta 数值解释成
身高、BMI 或具体脸型。实例化后若外观差异不足或几何不合理，应修订角色版本再冻结。

共用路径：`T=bedlam_body_textures_meshcapade/smpl/MC_texture_skintones`，
为上述本地皮肤 ZIP 内部路径；四张选定 albedo 均已读取头信息，4096×4096 RGBA。
`U` 为上节已核查的 `smplx_uv.obj`；`C` 为当前贴身衣物脚本的按新人体重新生成方案。
共用许可记录为下文 L1 + L2；M01 的具名 shape 另外关联 L3。

| 稳定角色 ID | shape 来源 / 身体模型 | 脸部与肤色来源、差异证据 | 发型来源和边界 | 服装方式 / 缺失项 |
|---|---|---|---|---|
| `cycling_F01_v1` | 本地 FEMALE；建议 `betas=[0,0,0,0,0,0,0,0,0,0]`，合成 shape preset F01 | `T/female/skin/skin_f_asian_01_ALB.png`；已有预览脸部、眉眼、较浅肤色纹理，与 F02 不同 | 同贴图预览有棕红头发区域；暂记 albedo-only，无独立发丝/发髻资产 | U + C；拟用红上衣/深色短裤。缺新 shape/脸部/眼周及骑行衣物验收 |
| `cycling_F02_v1` | 本地 FEMALE；建议 `betas=[0.75,-0.5,0.25,0,0,0,0,0,0,0]`，合成 shape preset F02 | `T/female/skin/skin_f_african_01_ALB.png`；已有预览面部色彩/眉唇及较深肤色，与 F01 不同 | 同贴图预览有深色头发区域；albedo-only，三维发型未获得 | U + C；拟用蓝上衣/深色短裤。缺新 shape/脸部/眼周及骑行衣物验收 |
| `cycling_M01_v1` | 本地 MALE；shape 取 Aaron/1095 NPZ 的 10 betas，完整值见下 | `T/male/skin/skin_m_white_01_ALB.png`；已有预览眉眼、胡须/体毛区域及较浅肤色，与 M02 不同；此纹理已有本地材质使用记录 | 同贴图预览棕色短发区域；albedo-only，无短发网格 | U + C；拟用灰白上衣/深色短裤。缺迁移后 body/clothing/眼部验收；不声称此脸是 Aaron 原脸 |
| `cycling_M02_v1` | 本地 MALE；建议 `betas=[0,0,0,0,0,0,0,0,0,0]`，合成 shape preset M02；与 Aaron shape 不同 | `T/male/skin/skin_m_african_01_ALB.png`；已有预览不同的面部色彩、较深肤色和头皮外观 | 同贴图预览近似剃短/光头外观，显式记录无三维头发 | U + C；拟用绿上衣/深色短裤。缺新 shape/脸部/眼周及骑行衣物验收 |

M01 采用已存在的 shape 值，迁移时保持该角色 shape 固定：

```text
[-1.02578818798, -0.81927132607, -0.43943831325, 0.20268304646,
 -0.02746478096, 0.10601449758, -0.02193670720, 0.10608315468,
 -0.30950465798, -0.08776225150]
```

四个候选的区别来自**各自固定模型/shape 与不同脸部/皮肤 albedo 的组合**，不是换衣换色。
预览只能证明已有外观资源不同，不能证明新组合已实现四个明显不同的三维脸型；包内也
没有与每张预览一一对应的 shape 参数。因此当前不把四张预览当作已完成角色照片。
文件名中的族群词只保留作官方资产定位符，不据此给合成人物添加真实族群标签。

已审查的原有预览（原样复制，无新渲染）：

- [F01 纹理预览](../../outputs/audits/cycling_character_feasibility_20260929/skin_f_asian_01_ALB_preview.1.jpg)
- [F02 纹理预览](../../outputs/audits/cycling_character_feasibility_20260929/skin_f_african_01_ALB_preview.1.jpg)
- [M01 纹理预览](../../outputs/audits/cycling_character_feasibility_20260929/skin_m_white_01_ALB_preview.1.jpg)
- [M02 纹理预览](../../outputs/audits/cycling_character_feasibility_20260929/skin_m_african_01_ALB_preview.1.jpg)

## 6. 若要求官方具名 shape 或独立三维发型

官方具名 shape 的备选表如下，不能与上面的合成 ID 混为一谈。gender 来源为
[官方 gender.csv](https://github.com/PerceivingSystems/bedlam_render/blob/main/config/gender.csv)，
分片定位依据本地 `third_party/bedlam_clothing_download/subject2archive.json`。

| subject 候选 | 当前可定位证据 | 缺失及下一步入口 |
|---|---|---|
| `rp_alexandra_posed_002` / female | 服装 NPZ 分片 0、纹理分片 0 有成员，官方 gender=f | 身体 shape/motion 参数缺；先查 BEDLAM gendered ground-truth 对应 subject；原服装仍不能直接转骑行 |
| `rp_alison_posed_001` / female | 同上，有不同 subject 基础名，官方 gender=f | 同上；不同名字还需独立 shape/头部审查 |
| `rp_aaron_posed_009` / male | 本地 shape、服装、纹理已有 | 骑行姿态下新服装/头发/眼部仍未完成 |
| `rp_ben_posed_004` / male | 官方 gender=m；与 Aaron 为不同基础名 | shape 未取得；若以后检查衣物资源，映射为 NPZ6fps 分片 1、纹理分片 2、CLO 工程分片 13，均非当前分片 |

新下载不在本次授权范围。本地目录不足以复现这四个官方具名人物；不建议为骑行任务
先下载大量原动作服装缓存，因为取得后依然不匹配新姿态。

BEDLAM2 备选发型可从本地 `third_party/bedlam2_render/config/whitelist_hair.json` 定位：
F01 `vinefx_groom_04-1_pixie-MF`、F02 `vinefx_groom_18-1_short-bob-F`、
M01 `vinefx_groom_03-1_short-casual-MF`、M02 `vinefx_groom_01-1_curly-M`。
这些只是候选名称，未看本地 groom 几何，不能保证适合骑行或选定头型。

官方入口为 BEDLAM2 Downloads → Hair → `b2_groom_abc.tar.gz`；
另需 authoring reference、每个目标 shape 的 T-pose 和头部绑定。
官方 `create_groom_bindings.py` 绑定的是 locked-head neutral reference 到目标 GeometryCache，
不能直接拿来绑定本项目标准 female/male 模型。在 Blender 中需检查 Alembic 曲线导入、
作者坐标系、头皮形变/根部绑定，再随头部姿态变换；只移动到头顶不够。
若保留现有人体拓扑，就适配头发到该人体；若迁移 B2 模型，则先验证 138 点兼容性。
[官方导入与绑定说明](https://github.com/PerceivingSystems/bedlam2_render/blob/main/unreal/import/README.md)

B2 原始 bald textures `bedlam2_body_textures_meshcapade.zip` 也未取得。
不能在带头发图案的 v1 皮肤上叠加新发型后，忽略外露的旧发际线。
Starter Pack 可用于了解官方组合，不能免去新骑行姿态的衣物重建和当前相机链适配。

## 7. 两个骑行姿态的最小接入方案

保留以下原始资产及其照片来源，全部新结果另存候选实验目录：

- P1：`coco_292226_person_1330527`。
- P2：`coco_306800_person_225160`。
- 根目录：`outputs/experiments/cycling_smplx_20260929/`；上游 HMR2/原图目录也不改。

建议分阶段实施，以下均是后续工作，**本轮未执行**：

1. 冻结角色 manifest：`character_id`、版本、模型文件/配置、shape 值及来源、纹理和 UV
   的路径/包内成员、脸部外观来源、hair mode/资产/绑定、衣物生成方式及许可记录。
   建立 `character_id + pose_source_id + derivation_version` 衍生键，角色身份跨姿态保持稳定。
2. 每角色分别接 P1、P2：迁移相同 SMPL-X 关节约定下的局部 body rotations 与已规范化
   global orientation，保留 flat-hand/jaw/eyes/expression 的明确设置。角色使用自己的
   betas，不复制原 neutral betas；不把八个组合包装成共享 shape 的连续 motion。
   新骨长会改变手足位置，需检查腿/躯干穿插、手臂姿态；不宣称保持自行车接触点。
3. 按新 shape 重新计算 pelvis/root 基准和平移。原 fitted transl 和原身高抬升不可直接复用。
   全部沿用当前 y-up→场景轴变换；衣物、头发与标签共用人体的抬升/旋转/平移。
4. 重生成 `V[1,10475,3]` 和 `R @ V`，新 shape、新姿态重新生成 C 衣物；不复用旧衣物
   `reference_body_vertices`。维持标准人体为标签网格，衣物和头发只作为渲染几何/遮挡物。
5. 小范围接口适配：`prepare_cycling_clothing.py` 根据角色读取 gender/model/num_betas；
   `render_equirectangular_source.py` 从每个 person 读取 texture/UV/眼部及可选 hair 配置。
   当前全局 `--body-texture` 会让所有人共用一张皮肤，需要更改后才能同图多角色。
6. 先检查八组合的正面/侧面/背面和头部外观，之后才接既有 DS 场景。保持明确 mask、
   138 点语义和遮挡监督策略；新增头发/衣物须进入深度及目标人物实例 ID。
   实际渲染世界网格、标签与裁剪验收通过后才能称为接入完成。

身份记录还要区分 `source_person_id`、`shape_source_id`、`appearance_source_id`、
`pose_source_id`；不能给新合成外观沿用 COCO 人物 ID 来冒充同一真实人物。
对 shape 的后续修订需增加版本并记录衍生关系，而不是悄悄修改已有角色。

## 8. 身份和姿态分组：避免八组合全部连通

四角色 × 两姿态 = 八个**组合**，仍只有四个候选合成身份、两个照片姿态来源。
若八组合全都产生，并要求角色与姿态均不跨集合，则它们构成一个连通分组，
**不能再拆成独立 train/val/test**。适合作接入审查，不能当独立泛化评测。

两种明确用途：

| 用途 | 分组办法 | 限制 |
|---|---|---|
| 推荐：八组合接入样例 | 八组合及所有相机/衣服/场景版本统一 `integration_only` | 不声称独立验证集 |
| 最小双因素留出示意 | train：F01/M01 × P1；val：F02/M02 × P2；禁止其他交叉组合进入这两个集合 | 每边仍只有一个姿态，无独立 test，统计远不足正式验收 |

正式扩量前先划分角色组和原照片/原人物/姿态来源组，再在各自集合内组合；增加新来源
才能建立独立 test。跨角色复用的脸部纹理也应记录并对“外观独立”评价施加分组约束。
同一个 BEDLAM 基础人物的 `_posed_*` 版本、角色换装/换色、相机、场景衍生均归原组。
若多人共图，把共现角色/姿态和图像联系一起纳入分组；不能从同图裁出两个人后跨集合。

## 9. 许可记录

本节记录所查文件和官方条款，不据“公开可查”推断可商用/可再分发。

| 编号 | 组件 | 本次记录和边界 |
|---|---|---|
| L1 | 本地 SMPL-X 模型及衍生形状 | [SMPL-X Model License](https://smpl-x.is.tue.mpg.de/modellicense.html)：非商业科研等限定用途；商业许可另行取得。本地文件存在不证明项目已取得商业许可 |
| L2 | Meshcapade 100 套皮肤 | 包内 `Meshcapade_textures_CC-by-nc.md` 和 readme 声明 CC BY-NC 4.0、来源 Meshcapade、版本 20220207；保留署名及修改记录。脸部/皮肤不是独立于整个贴图的无许可素材 |
| L3 | BEDLAM shape/motion/clothing 数据 | [BEDLAM License](https://bedlam.is.tuebingen.mpg.de/license.html)：非商业科研等限定用途及分发限制；使用具名 shape 或衣物时关联此许可，不能因贴图有 CC 许可就放宽所有资源许可 |
| L4 | BEDLAM2 groom / 模型参数 / 服装 | [BEDLAM2 License](https://bedlam2.is.tuebingen.mpg.de/license.html)：非商业科研等限定用途及分发限制；源资产尚未取得，获取/授权记录为空，不能标“许可已齐备” |
| L5 | 当前程序化衣物和眼部代码 | 来源为本仓库实现，保留实现版本；其人体衍生几何仍关联 SMPL-X，不把自写代码等同于解除上游约束 |

官方原始 BEDLAM 发型的不可再分发限制见
[BEDLAM2 论文 §3.4](https://bedlam2.is.tuebingen.mpg.de/media/upload/BEDLAM2_NeurIPS2025.pdf)。
原照片两姿态仍携带原输入 `manifest.json` 的逐图许可和来源；此次角色组合不覆盖这些记录。
本轮没有验证原资产的账号授权主体或项目用途，不给出“全部商业可用”的结论。

## 10. 工作量和下一步检查入口

以下是单人工程量估计，非测得耗时；不含等待资产权限、人工审美迭代、正式数据扩量或训练。

| 后续阶段 | 估计 | 交付/条件 |
|---|---|---|
| 四角色 manifest、shape 实例化、两姿态迁移及几何检查 | 0.5–1 人日 | 八份独立衍生几何；不更改原重建 |
| 按角色材质/眼部、衣物脚本适配 | 1–2 人日 | 四套不同皮肤/脸部与匹配衣物；确认贴图眼周和 UV，无假发资产声明 |
| 外观审查、DS 小样及标签/loader 验收 | 0.5–1 人日 | 几何与语义一致、衣物/眼睛合格；后续计算前先查 GPU/进程 |
| 独立 BEDLAM2 三维头发 | 资产到位后另计约 3–7 人日探索 | Blender 曲线、参考头型、绑定和四角色两姿态检查；失败可能需要其他资产路线 |

因此最小的**贴图头部外观 + 贴身衣物**四角色原型预计 2–4 人日；包含独立三维发型时
不能承诺同样工期。若要求 BEDLAM 原设计的宽松衣物，需额外取得可改造衣物或源工程并
重绑定/重新模拟，当前证据不足以可靠估算，不包含在上表。

下一步具体入口：

- 按第 5 节候选表决定是否先接受 albedo-only 头部外观作为接入阶段；它不满足三维发型多样性。
- 若要求官方具名 shape，先检查 BEDLAM 下载区 gendered ground-truth 的 subject 参数和
  模型版本；不要误取 neutral 11-beta GT 到 male/female 10-beta 模型。
- 若要求三维发型，检查 B2 groom 包、locked-head authoring reference、bald textures
  和目标绑定所需资源；包大小、可单独获取粒度在本次公开页面中未确认，不编造数值。
- 若要求真实服装款式，检查对应 `clothing_zprj` 或静态可绑定源网格；既有逐动作缓存不能
  直接充当新骑行姿态衣物。当前原型衣物可先承担几何接入验证。

归档审计复现命令（必须换新的输出目录，不覆盖本次结果）：

```bash
/home/yy/conda_envs/camerahmr/bin/python -B tools/experiments/audit_cycling_character_assets.py \
  --bedlam-root /home/yy/datasets/BEDLAM \
  --output outputs/audits/cycling_character_feasibility_review_v2
```

本轮验收结论：资源核查和候选方案完成；四个完整角色接入及八个骑行组合的实际质量仍待后续验证。
