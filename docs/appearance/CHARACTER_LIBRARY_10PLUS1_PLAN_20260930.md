# 十名成人＋可选一名儿童：角色设计与公开资源接入方案

日期：2026-09-30。状态：**规划完成；十人组合未制作、未验收**。
F02/M02 后续执行见 [两角色通勤试制记录](COMMUTER_TWO_CHARACTER_PILOT_20260930.md)：完整模型回归、同源参考核验和三维 groom 绑定已完成，衣物及共同接触验收仍需修正，尚不可扩十人。
本轮仅只读核查文件、数组、已有图片和官方资料，并写入本报告及候选 JSON。
未下载资产，未运行重建、角色生成、渲染、推理或训练；未修改外部项目、计算 SHA256 或提交 commit。

## 1. 推荐路线和关键缺口

保留四个旧身份，升级为带三维发型的版本；新增三男三女，形成五男五女。
建议男女各两人运动、三人通勤，共 **4 运动＋6 通勤**。先以 F02/M02 验证发型绑定和合身通勤衣物。
儿童走独立验证分支，不进入成人方案的完成前提。

推荐保持标准 SMPL-X 标签人体，优先复用本地完整形状空间与 Meshcapade 皮肤；
三维发型优先 BEDLAM2 Alembic groom，衣物采用免费 Blender 中制作的合身衣型及当前重心映射思路。
这是一条**可执行但尚有两项技术关卡**的路线：当前形状空间能否满足十人的合理面部差异；
BEDLAM2 发型能否在保留标准头部的前提下通过绑定及轮廓审查。
若关卡失败，明确停在候选状态，不用换色、光头贴图或夸大 beta 来补人数。

机器可读产物：

- [candidates.json](../../outputs/plans/characters_10plus1_20260930/candidates.json)：10 名成人、1 名可选儿童、资源状态、来源、父子关系、缺项。
- [local_evidence.json](../../outputs/plans/characters_10plus1_20260930/local_evidence.json)：本轮文件/数组复核与已查看图片清单。

资源许可仅作为来源记录，不作为本轮规划交付的问答前置条件。“无需付费”不等于不限用途。
SMPL-X、BEDLAM2 和 AGORA 的公开许可
限定非商业科研等用途，现有 Meshcapade 纹理为 CC BY-NC 4.0；本报告不据免费访问承诺商业使用权。
[SMPL-X 许可](https://smpl-x.is.tue.mpg.de/modellicense.html)、
[BEDLAM2 许可](https://bedlam2.is.tuebingen.mpg.de/license.html)、
[AGORA 许可](https://agora.is.tue.mpg.de/license.html)。

## 2. 当前状态复核

读取了用户指定三份文档，以及当前四份 manifest、总报告、48-view manifest、八份保存的
body/point/parameter 数组、形状对照及头部和 P1 多方向图片。没有重跑旧验收或旧渲染。

| 项目 | 本轮直接证据 | 能说明什么 / 不能说明什么 |
|---|---|---|
| 四角色资产 | 四份 manifest 均 frozen_after_review，材质 v3、衣物 v2；实际贴图/UV/眼映射存在 | 旧集成已完成，不能代表本轮三维发型完成 |
| 八组合 | 每份 V 为 [1,10475,3]，138 点为 [1,138,3]；保存 betas 与各自角色相等 | 重新只读计算 R @ V 的最大误差 5.96046e-8 m；未重新生成身体 |
| 旧验收 | 总报告记录 4 角色、2 姿态、48 预览、4 DS 视图、13 行 loader | 作为既有 integration_only 证据，不扩称独立训练集 |
| 头部与体型 | 实际头部、灰模及统一服装图已查看 | 同性别头部轮廓很接近；眉唇纹理有差异，不能声称已覆盖充分脸型变化 |
| 肩宽 | F01/F02 joint width 为 0.29102/0.29169 m；M01/M02 为 0.359809/0.359801 m | 当前同性别肩宽覆盖很窄；joint width 也不是人体测量中的外肩宽 |
| 发型 | manifest 明写 albedo only；原生身体有头皮/后脑轮廓 | 无独立三维发型；标准模型的 head-bun 也不能算发型资产 |
| 本地皮肤包 | ZIP 目录直接读取：214 项、100 张皮肤；另查看六张候选的原有官方 JPEG | 六张预览只在 /tmp 原样读取，无新拼图/新预览；相似基础脸型仍须重审 |

本地完整模型 ZIP 为 870,108,517 bytes；female/male 的 shapedirs 为 [10475,3,400]。
本轮直接数组比较确认：其模板、面序、前 10 个 shape directions 与当前同 gender 模型完全一致；
归档 300:310 的方向对应当前 10:20 的 expression directions。结合加载代码，完整文件是
300 shape＋100 expression，**不能把当前精简文件的第 11–20 维当作新增身份形状**。
文件名 v1_1 与内部版本历史差异仍沿用先前审计记录，不以名称单独认定版本。
[官方完整形状空间说明](https://github.com/vchoutas/smplx)、
[SMPL 模型规格汇总](https://smpl-made-simple.is.tue.mpg.de/smplcentral.html)。

在 /home/yy/datasets、本仓库 assets_external/third_party 和外部 pano-stitch/tools 指定范围按路径/文件名核查，
未定位到 BEDLAM2 groom/bald 包、已安装 locked-head 模型和 SMPL-X kid template；只有相关代码入口。
这是所查范围结论，不宣称扫描了所有磁盘。

## 3. 十名成人设计表

以下都是设计目标，身高范围不是已生成数值，脸型描述不是对纹理文件中人物的身份推断。
F/M 为模型配置类别；文件名中的族群词仅为原资产定位符，不额外给角色赋真实族群身份。
每人 shape 最终值暂为 null，避免把概念参数伪装成已验收模型。

character_id=cycling_F01 等为新增的稳定逻辑键，asset_id=cycling_F01_v2 为候选版本。
旧 manifest 的 character_id=cycling_F01_v1 **原样保留**，映射为同一 identity_family 下的父资产；
不是删除旧 ID 或把 v1/v2 计成两个人。F01/F02/M01/M02 都执行这一规则。

| 稳定 ID / 版本 | 身体与目标相貌（尚未制作） | 皮肤定位 | 三维发型候选 | 初始穿搭建议 |
|---|---|---|---|---|
| cycling_F01 / 旧 v1 → v2 | 保留约1.66m、较窄肩的旧体型；面部延续较窄唇部，目标较窄椭圆轮廓；目标身高 1.63–1.69 m | skin_f_asian_01_ALB | vinefx_groom_04-1_pixie-MF | 运动：短袖立领骑行上衣＋中长骑行短裤；蓝/白拼片，细网眼 |
| cycling_F02 / 旧 v1 → v2 | 保留约1.72m；脸部保留浓眉/唇部细节，目标更宽颧部和较圆下颌；目标身高 1.69–1.75 m | skin_f_african_01_ALB | vinefx_groom_18-1_short-bob-F | 通勤：合身长袖圆领针织上衣＋直筒长裤；米色细罗纹/深蓝斜纹 |
| cycling_F03 / 新增 v1 | 较矮、肩较宽、腰腹较饱满；目标短圆脸、较宽下颌；目标身高 1.55–1.62 m | skin_f_white_01_ALB | vinefx_groom_07-1_medium-layered-bob-MF | 通勤：合身短袖圆领T恤＋膝上通勤短裤；砖红细横纹/卡其织物 |
| cycling_F04 / 新增 v1 | 较高、肩背较宽、较精瘦；目标偏长脸和清楚下颌轮廓；目标身高 1.74–1.8 m | skin_f_indian_03_ALB | vinefx_groom_10-1_bun-F | 运动：长袖圆领运动上衣＋脚踝长骑行裤；青绿斜向拼片/黑色 |
| cycling_F05 / 新增 v1 | 中等身高、腰髋及大腿较饱满；目标偏宽椭圆脸与较短下巴；目标身高 1.63–1.69 m | skin_f_asian_03_ALB | vinefx_groom_20-1_low-afro-F | 通勤：七分袖小立领合身上衣＋踝上直筒裤；橄榄色细格/炭灰 |
| cycling_M01 / 旧 v1 → v2 | 保留Aaron约1.88m高个体型；面部延续旧眉口特征，目标偏长椭圆轮廓；目标身高 1.85–1.91 m | skin_m_white_01_ALB | vinefx_groom_03-1_short-casual-MF | 运动：短袖立领骑行上衣＋膝上骑行短裤；橙/深蓝纵向拼片、网眼 |
| cycling_M02 / 旧 v1 → v2 | 保留约1.79m；目标更宽肩、较壮实，面部较宽下颌；目标身高 1.76–1.82 m | skin_m_african_01_ALB | vinefx_groom_01-1_curly-M | 通勤：合身短袖polo＋直筒长裤；灰蓝珠地/浅灰斜纹 |
| cycling_M03 / 新增 v1 | 较矮、肩较窄、精瘦；目标较窄脸与较短下脸；目标身高 1.64–1.71 m | skin_m_asian_01_ALB | vinefx_groom_05-1_straight-hair-M | 通勤：合身长袖小翻领衬衫＋膝上短裤；浅蓝细条/沙色 |
| cycling_M04 / 新增 v1 | 中高个、宽肩、腿部较结实；目标短宽脸与较宽颧部；目标身高 1.77–1.84 m | skin_m_indian_03_ALB | vinefx_groom_06-1_slicked-back-M | 运动：长袖半拉链运动上衣＋七分骑行裤；酒红几何纹/深灰 |
| cycling_M05 / 新增 v1 | 中等身高、较宽腰腹、较厚躯干；目标圆脸和饱满面颊；目标身高 1.7–1.77 m | skin_m_white_03_ALB | vinefx_groom_13-1_curtained-M | 通勤：合身长袖亨利领上衣＋直筒长裤；森林绿罗纹/棕色织物 |

十张皮肤 PNG 均已定位于现有 ZIP，对应的六张新增候选原有预览已查看。
例如 M04 预览含胡须纹理，M05 有不同眉毛/皮肤细节；F03/F05 的基础脸部轮廓仍与部分女性预览接近。
这些只能提供纹理选择线索，不能代替目标 shape 下的同条件脸部验收。
表中的十种 groom 名称来自本地官方仓库的白名单，均是**未取得、未导入、未适配**的资源候选；
不要求最后保留十种不同发型，适配失败可以换候选，但不能以发色代替三维轮廓差异。

逐行共用资源定位及状态（完整展开在 JSON）：

| 组件 | 路径 / 来源 | 当前状态与许可 |
|---|---|---|
| 标准模型 | /mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools/smpl_models/smplx/SMPLX_{FEMALE,MALE}.npz | 已有且旧角色已接入；SMPL-X 限定用途许可 |
| 完整 shape | 同项目 tools/PEAR/assets/models_smplx_v1_1.zip 内 models/smplx/SMPLX_{FEMALE,MALE}.npz | 已有；高维候选尚未接入或验收，不需再下载 |
| 六名新增 shape | 在同 gender 官方完整形状空间中按测量目标和先验选择/优化 | 没有现成六个 betas；是拟制作合成人物，不是六个已获得扫描身份 |
| 四名升级 shape | 旧 manifest 的 betas；M01 保留 Aaron/1095 来源 | 优先沿用，再按脸型关卡决定是否修订；派生版本持续保留旧 shape 来源 |
| 十张皮肤 | /home/yy/datasets/BEDLAM/body_textures/bedlam_body_textures_meshcapade.zip 内 bedlam_body_textures_meshcapade/smpl/MC_texture_skintones/{female,male}/skin/<表中ID>.png | 已有；CC BY-NC 4.0，版本 20220207；新六张仅预览已读 |
| UV / 眼 | outputs/body_texture_audit/body_uv.npz；tools/render/character_materials.py | 旧四张 material v3 已接入，新 shape/皮肤必须逐人复查 |
| 三维发型 | BEDLAM2 Hair / b2_groom_abc.tar.gz；third_party/bedlam2_render/config/whitelist_hair.json | 官方列出，本地无文件，免费注册及接受限定用途条款 |
| 初始衣型 | tools/experiments/refine_cycling_character_clothing.py 作为技术起点，在 Blender 新建计划衣型 | 当前只存在短袖短裤壳层；表中通勤版尚未制作，没有宣称已下载成衣 |

四个旧身份保留全部原图、姿态、纹理和 shape 指针。头皮修整/替换 bald 纹理、换 groom、
换衣服均写新 appearance_version，仍属于旧身份。若高维 shape 改动较大，保留 shape_version 和
parent_asset_id；不得为了“独立人物数”切断同源关系。

## 4. Shape 与相貌扩展路径

优先保持同 gender、同模板、同顶点语义的标准 SMPL-X，用完整文件探索前 30 个 shape 方向；
30 是初始工程上限建议，不是生物学结论。仅在不足时评审扩大维度，最多使用真实 300 维 shape 空间。
不直接放大某个 beta 来指定“胖”“宽脸”，不局部改点、拼头或用 expression 冒充身份。

下一阶段先确认参数尺度/先验及正则化依据，再根据身高、外肩宽、腰/髋/大腿围和头面比例设约束。
候选采用受约束全身 shape 优化或筛选，保存配置、种子、目标测量及最终实测值；
系数差异、顶点 RMS、默认随机正态采样都不能单独作为体型或相貌验收。
无可核实先验时，不把任意“±3 beta”称为合理统计范围。

升级完整模型后须检查 f、v_template、全部实际使用 directions、posedirs、LBS weights、joint regressor、
关节/手部约定、expression 索引、UV、138 行回归器支持区域；后几项尚未在本轮全量重新比对。
先用旧 10 维 shape 的同空间嵌入做数值回归，确认高维参数为零时旧 mesh/points 可恢复，再生成新 shape。
这项嵌入仅因本轮已验证同 gender、相同前 10 个方向才可提议，不能推广到 BEDLAM2 16 维或其他模型。

当前 refine_cycling_character_clothing.py 仍显式 num_betas=10；共享 load_smplx_model
虽接收 num_betas，但儿童 age/kid_template 尚未暴露。因此高维/儿童方案都需要明确接口适配，
不能仅更换模型路径。若合理全身空间无法产生可辨认的五男五女，停止人数验收并记录难区分的配对；
下一入口是核查同版本公开 subject shape/更完整官方模型，不默认采购扫描，也不转为局部拼头。

## 5. BEDLAM2 发型、头皮与衣物适配

### 发型参考与免费路径

官方使用 locked-head/no-head-bun neutral 模型、16 维 shape，与当前标准男女模型不同。
绑定代码以 /Engine/PS/Bedlam/Core/SMPLX/smplx_lh_n_abc 为 authoring reference，
向每个目标的 _0000 T-pose GeometryCache 建 groom binding；这不是可直接复用的 Blender 绑定。
[官方模型差异](https://bedlam2.is.tuebingen.mpg.de/faq.html)、
[官方绑定代码](https://github.com/PerceivingSystems/bedlam2_render/blob/main/unreal/import/groom/create_groom_bindings.py)。

建议在 Blender 中实现以下路线，均为待验证工程方案：

1. 先取得 groom 和同源参考头部。核实曲线数、根端方向、宽度、单位、坐标及组属性；
   官方 UE 导入脚本设置旋转 [-90,0,180]，不能直接当作 Blender 或本仓库世界变换。
2. 对参考静态头皮建立语义对应，核实后将发根记录为参考表面三角形/重心坐标；
   映射至目标 rest head，传递局部切向框架与偏移，变形的是头发，不是标签人体。
   不对整个身体用无约束最近点匹配，避免根部吸到脸、耳或后颈。
3. 随目标 posed scalp/LBS 变形，之后统一走人体 world transform；短发可先验证近似随头绑定，
   bob/长发在颈肩附近必须查碰撞，需要时增加限位/局部变形或短程静态松弛。
4. 原生 head-bun 若穿出短发，先改用可覆盖且有轮廓差异的 groom。若为遮盖而严重拉坏发型，
   判为失败，不删除或雕改人体头部顶点。迁移 B2 标签模型只能另开兼容性验证，不能静默替换。
5. 发丝进入光照、阴影、深度及目标人物实例 ID，核对曲线/转换网格的 pass index 和深度表现。
   头发虽遮住语义点，标签仍在人体，重新计算可见性后沿用遮挡监督。

Blender 官方列出 Alembic Curve 导入能力，但**通用格式支持不代表本项目已成功导入这包 groom**；
宽度、分组属性、渲染半径、内存和实例通道均须做样例。免费路径为本地 Blender 原生工具加必要转换代码，
不依赖 Maya、Houdini 商业版或付费 groom 插件；失败时先诊断曲线属性/转换，不用付费软件绕过。
[Blender 管线格式说明](https://www.blender.org/features/pipeline/)。

### 头皮纹理冲突

现有 albedo 自带发际线、短发图案。加真实 groom 后，耳边、后颈、额头可能同时露出两套发际线。
保留原 PNG，优先在派生纹理中只处理头皮区并留编辑遮罩，脸部细节保持来源；不得直接把旧发色涂黑。
若不能自然覆盖，才获取 B2 bald 包。其 UV 2023 与当前 UV 不默认相同，同名贴图也不证明像素对应：
需比对 face-corner UV、脸部/耳部和接缝，必要时显式重烘焙并保留来源及变换记录。
旧四人的脸部来源继续关联，不把新 bald 贴图或新发型算成第五个旧身份。
[官方 bald 纹理入口及材质流程](https://github.com/PerceivingSystems/bedlam2_render/blob/main/unreal/import/README.md)。

### 衣物与付费依赖

本地 BEDLAM 服装 NPZ/OBJ 是旧 motion 的形变缓存。衣物贴图也绑定原衣物 UV，不能直接套新人体或新壳层。
CLO 工程能否重拟合是另一条路线，不能把“有公开工程”写成“可免费编辑”；CLO 官方是付费订阅，
本方案不使用其试用期作为长期免费依赖。[CLO 官方费用说明](https://support.clo3d.com/hc/en-us/articles/115012815487-How-much-does-CLO-cost)。

免费主线：在 Blender 中制作少量可参数化、独立上/下装的合身衣型，先包含短袖 T 恤、长袖圆领、
polo/简单衬衫、短裤和直筒长裤；通过领口、袖长、裤长、裤腿截面、下摆及缝线形成款式差异。
程序纹理生成网眼、罗纹、斜纹、条纹和格纹，不冒用旧衣物 UV，不把色块变化当作全部多样性。
这需要真正的新衣物制作，**不是已有六套免费通勤成衣可直接使用的承诺**。

每个 shape 重新适体；紧贴部件可使用表面重心映射，具有领口/袖口/裤腿松量的部件需要权重或表面绑定，
再按目标姿态重变形，检查腋下、肘窝、裆部、髋腹、膝窝、后腰及颈肩头发交界。
表面映射失败或衣型需要重力褶皱时，用 Blender Cloth 的碰撞、pin 和短程静态 settle，逐姿态保存缓存。
起始中性姿态到目标的求解过渡仅为数值初始化，不能包装成照片之间的真实 motion。
不默认宽松衣物动态模拟；若合身衬衫仍不稳定，先调整衣型/绑定，再降级款式，并保留多样性验收。
[Blender 官方 Cloth 示例](https://docs.blender.org/manual/en/latest/physics/cloth/examples.html)。

鞋袜单独可选：初版可以保留赤足并如实记录。袜子壳层预计额外 2–4 适配人时、0.2–0.5 小时计算、1–2 审查人时；
一两款独立简鞋预计 8–16 人时适配、0.5–2 小时计算、2–4 人时审查，非本轮成人必备。
B2 鞋采用 toeless 模板加 displacement；不能直接改标签脚部来套鞋，应保留标准脚部作为标签，
鞋作为独立遮挡几何。若采用其映射须再验对应和脚部语义，资源大小未知。
[官方鞋处理说明](https://github.com/PerceivingSystems/bedlam_clothing)。

## 6. 资源状态与最小获取清单

“官方列出”“本地存在”“Blender 导入通过”“本项目适配通过”是四个独立证据层级。
候选 JSON 对缺失发型三个后续状态均为 false；旧皮肤仅四张在本项目通过，不扩大到全部 100 张。

| 资源 | 官方 / 本地 / Blender / 项目状态 | 最小获取或使用粒度 | 大小、条件 |
|---|---|---|---|
| 标准 SMPL-X | 已列出 / 有 / 旧路径通过 / 旧四人通过 | 直接复用既有男女模型 | 本地已有，不再下载 |
| 完整 shape 模型 | 已列出 / 有 / 新高维未验 / 未验 | 本地 ZIP 内 female/male NPZ | 整 ZIP 870,108,517 bytes；无需联网 |
| v1 皮肤包 | 已列出 / 有 / 四张通过 / 四张通过 | 只读选取十张和相应预览 | 整 ZIP 1,906,155,828 bytes；无需重复获取 |
| v1 旧衣物缓存/纹理 | 已列出 / 有 / 旧用途证据 / 新姿态不适配 | 不作为本轮最小下载项 | 28,151,562,240 / 4,870,778,880 bytes，本地已有 |
| B2 groom | 已列出 / 无 / 未验 / 未验 | b2_groom_abc.tar.gz 整包入口，取得后先选 F02/M02 两个，再选其他；单 groom 下载能力未知 | **大小未知**；BEDLAM2 注册、邮箱验证、接受许可 |
| Groom authoring reference | 官方代码依赖 / 无 / 未验 / 未验 | 先核实 SMPL-X Blender addon ≥20241129 或 locked-head 模型包哪个最小包提供同源 no-head-bun reference | **大小未知**；smplx_lockedhead_20230207.zip 有官方脚本入口，但包内容与 grooming reference 的完全一致性待验，不先同时下载两包 |
| B2 bald 纹理 | 已列出 / 无 / 未验 / 未验 | bedlam2_body_textures_meshcapade.zip；仅头皮修整失败时补；单图获取未知 | **大小未知**；BEDLAM2 条款，包内单项许可届时核对 |
| B2 motion/Starter Pack | 已列出 / 无 / 未验 / 未验 | **不列最小获取项**；已有骑行姿态，.uasset 包不免除新衣物适配 | 不为 shape/头发直接下载大包，大小未知 |
| 新通勤衣型 | 设计项 / 无 / 未验 / 未验 | 本地制作独立衣型，保存 blend/UV/binding/各姿态几何 | 无下载成本，有制作成本；继承身体衍生许可边界 |
| 儿童 template | AGORA 官方列出 / 无 / 未验 / 未验 | smplx_kid_template.npy 所在 kid-template 条目，具体包粒度待登录核实 | **大小未知**，AGORA 注册及许可 |
| 儿童皮肤/独立姿态 | 未定位合格候选 / 无 / 未验 / 未验 | 逐身份/逐图筛选并记许可，不能以成人包替代 | **来源与大小未知**，只影响儿童 |

BEDLAM2 下载页当前匿名访问跳转登录。本轮未登录、不读取凭据、不下载；未能核实的字节数、
单文件下载能力和参考包内容均记为未知，没有用推荐磁盘容量代替下载大小。
官方入口：[BEDLAM2 下载](https://bedlam2.is.tuebingen.mpg.de/download.php)、
[资源注册说明](https://bedlam2.is.tue.mpg.de/index.html)、
[Blender reference 准备说明](https://github.com/PerceivingSystems/bedlam2_render/blob/main/blender/smplx_anim_to_alembic/README.md)。

## 7. 可选学龄儿童

结论：**有公开身体接入方向，但当前不足以批准一个完整儿童角色进入制作验收**。
ID 预留 cycling_C01；具体年龄外观、性别待定。AGORA 提供 kid template 与 SMPL-X API 用法：
age='kid'，kid_template_path，额外一个形状分量表示成人到儿童模板差值插值，不是年龄单位。
官方建议男童用 male，女童用 neutral；不能直接沿用成人 female 配置。
[AGORA kid-model 文档](https://github.com/pixelite1201/agora_evaluation/blob/master/docs/kid_model.md)。

本地 smplx/body_models.py 有该实现，但本仓库加载接口没有 age/kid_template 参数，目标文件也未定位。
SMIL 本身是婴儿模型，不能把它的模型定义或成人等比缩放称作已经获得学龄儿童。
AGORA 页面包含儿童扫描研究，但这些扫描来自商业资源；公开图像/拟合参数不等于赠送原始皮肤、发型、衣物资产。
[SMIL 官方说明](https://is.mpg.de/ps/code/skinned-multi-infant-linear-model-smil)、
[AGORA 项目说明](https://agora.is.tue.mpg.de/)。

儿童独立检查清单：

- 模板：验证具体 kid template 的顶点对应、f、单位、body proportions 与关节约定；n＋1 参数维度显式保存。
- 138 点：检查回归器每行支持区域在儿童身体的解剖语义、左右一致性和面/手/脚位置；不能只看 10475 顶点。
- 外观：缺少已核实适龄皮肤/面部；不能把成人胡须、体毛和成年五官贴到儿童上。必要的原创派生外观也需单独审查。
- 头发：成人公开短发可作为资源起点，但按儿童头皮重新绑定，不整体缩小发型和身体。
- 衣物：按儿童比例重新构造上衣/裤子及绑定；保留头身比例、躯干长度和肢体粗细的合理性。
- 姿态：现有成人骑行姿态可做技术压力测试，不能称为独立儿童骑行姿态；尚缺可追踪许可的学龄骑行照片/姿态来源。
- 划分：若先完成一个儿童，只标有限 integration_only 覆盖，不设计独立儿童训练与泛化测试。

## 8. 分阶段任务、产物、失败处理与工作量

以下估计针对一位熟悉本仓库的工程人员、现有 Blender/Python 和单张 RTX3090，非承诺工期。
人时与计算时分开；计算可和审查部分重叠。资源等待不确定，不计入有效工作人时。

| 阶段 | 输入 → 产物与检查 | 失败后处理 | 编码/适配人时 | 计算墙钟估计 | 人工审查人时 | 资源等待 |
|---|---|---|---:|---:|---:|---|
| S0 资源就绪 | 本计划/资源条件 → 最小包及 authoring reference 清单、逐项大小/许可记录 | 无权限则标记缺失，不采购替代 | 1–2 | ≤0.5h，仅解包检查 | 1–2 | 注册/访问/网络未知 |
| S1 形状空间与身份关卡 | 本地 full-space＋旧4人 → 先制作 F02/M02 shape 候选与测量表，同条件分别对照旧 F01/M01；其他六人的 shape 制作留至 S3 | 比例不合理/同脸退回参数选择；无法区分则报告缺口，不靠 beta 极值 | 8–16 | 0.5–2h | 4–8 | 本地资源，无等待 |
| S2 一女一男完整试点 | F02/M02＋两 groom＋长袖长裤/短袖polo长裤 → 发根绑定、头皮派生纹理、独立衣型、2人×2姿态样例 | 曲线失败修属性/转换；穿插修绑定/衣型；必要时改候选并保留失败记录 | 20–36 | 2–8h | 4–8 | groom/reference 未取得，未知 |
| S3 成人扩展 | S1/S2通过 → 其余8人，10套初始搭配；两现有姿态可形成20个集成组合 | 不合格人物留candidate，逐人补修，不先批量批准 | 16–28 | 3–10h | 6–12 | bald包若需补充，未知 |
| S4 有限DS集成与划分决策 | 10人通过棚拍 → 森林/街景有限覆盖、实际loader、世界网格/深度/实例检查、来源分组清单 | 任一几何门槛失败退回对应阶段；不放宽旧门槛 | 6–12 | 1–4h | 3–6 | 独立姿态扩充另立任务 |
| C 可选儿童分支 | template＋适龄外观＋合格来源 → 一人独立验证包 | 缺模板/语义不兼容/外观不合理则保留未完成，不延期十成人 | 12–24 | 1–4h | 4–8 | 模板/外观/姿态未知 |

成人合计约 **51–94 编码/适配人时＋18–36 审查人时，计算约 6.5–24.5 小时**，另加资源等待。
这覆盖一个主要实现迭代，复杂发丝转换或衣物多次返工可能超出；不把旧样例约19分钟计算当制作总成本。
鞋袜为上节独立增量，儿童成本不计入成人合计。
后续任何生成/渲染开始前检查 GPU/主机进程；本轮没有启动这些计算。

外观验收拟采用以下固定审查协议，正式批量前把选择结果冻结：

1. 身份对照：统一灯光、姿态、服装、视角、相机尺度；无发型遮挡的脸部正/侧/斜面，
   带纹理与灰模两套；记录同性别 20 个配对，凡“只能靠肤色/衣服/发色区分”的配对退回。
   至少说明轮廓/比例与眉眼鼻唇细节的差异，不以仅一个 RMS 数值批准。
2. 体型对照：相同单位尺度量身高、外肩宽、腰髋/大腿围及比例，统一穿衣轮廓检查；
   不把默认相机缩放后的高度差当身高，不以 betas 不同替代实测，也不从 mesh 自称 BMI。
3. 完整角色：各自发型、服装前侧后对照，检查额头/耳/后颈双发际线、发根漂浮、head-bun 穿出、
   领口/袖口/裤脚、衣物透体，确认三维剪影和款式差异，而非只看颜色。
4. 两独立骑行姿态：正面、侧面、背面及腋、肘、裆、髋腹、膝、后腰、颈肩近景；
   原身体姿态不能为了穿衣而偷偷修改，必要修订需新姿态版本及原来源。
5. 实际输入尺度：通过真实 loader 看完整 256×256 crop 和网络实际 256高×192宽中心输入；
   保留标签完整256-square坐标，不另加32像素偏移。以边缘/畸变/小人体场景检查发丝闪烁、衣纹混叠、
   眼部错贴、遮罩及裁剪效果；不要求小裁剪能识别每张脸，但不能只展示高分辨率棚拍。

数值验收沿用旧参数还原2e-5 m、R@V 1e-6 m、pelvis/目标world点2e-6 m、实际渲染身体/附件
world变换5e-6 m、投影/crop2e-4 px、DS光线往返1e-8。旧紧身壳仍保留最大偏移1 cm、
局部间距0.4 mm及既有求解/float32读回容差。新衣型若需要不同松量，必须单独解释建模距离与碰撞准则，
不能暗中放宽旧壳门槛；头发不套用紧身壳1 cm距离上限。
最终人体和附件共用人体抬升、轴转换、yaw和位置；显式有效mask不从RGB黑色推断。
标签固定为最终SMPL-X人体的138点，头发/衣物进入深度与实例，监督继续 in_view & inside_full_crop，
不把 DenseKP keypoint_value 当置信度或权重。

## 9. 来源、划分和待决定事项

保存链路：原照片/人物 → 原重建 → 独立 pose_source → 目标 shape_source＋appearance_source →
稳定 identity_family → asset_version → outfit/hair variant → camera/scene/crop。
每次 shape 改动都重新生成身体、pelvis/root、衣物/发型绑定及 R@V；保留所有父版本和失败输出。
同一人物换装/换色/换发型/换视角不增加独立身份；十个角色也不增加现有两个独立姿态来源。

S1–S4 的试验仍 integration_only；现有八组合保持原标记。正式批量前先选角色家族、
照片原人物/近邻帧/姿态来源、复用脸部纹理和多人共图关联的分组，再只在集合内部组合。
官方共享模型本身作为模型来源记录；相同实际 shape、皮肤、扫描人物或其派生关系另外做防泄漏组，
不能靠新角色 ID 隐藏克隆。若全部十角色都跨用 P1/P2，这些组合仍构成连通图，不能随机拆分。
独立姿态来源数量和正式训练规模另定，本轮不生成训练/验证清单。

后续只需评审两项规格，不妨碍本轮交付：

- 是否采用推荐的 4运动＋6通勤，以及免费 Blender 自建合身衣型；鞋袜暂为可选。
- 儿童何时启动、具体学龄外观与性别；推荐十成人试点稳定后再定，不让它阻塞成人。

本轮没有需要宣称“已完成”的新角色、发型、通勤衣物或儿童资产。
下一步可直接下发 S0–S2 的限定制作任务：先核实最小资源及形状空间兼容，
再用 F02/M02 验证发型和通勤衣物；未通过不批量扩十人。
