# BEDLAM2 原配衣物适配到 M03 行走（A2，2026-10-10）

本批只替换已获用户接受的 M03＋KIT/11 120 帧行走组合中的衣物。M03 标准 SMPL-X 身体、原参数/时间、头发和 R138 保持原样；现有贴身简单衣物版本保留供同帧比较。新增衣物是 BEDLAM2 原发布组合 `nl_2066_3XL/2205` 的 `gr_eric_033_3XL` 仿真缓存，经明确近似转移到 M03；**不是 BEDLAM2 原身体的整套原配呈现，也没有在 M03 上重新做布料物理仿真**。全部仅供 `integration_only` 人眼评审。

## 来源、获取和配对

[冻结范围](../../outputs/experiments/bedlam_m03_native_cloth_transfer_a2_20261010/frozen_scope.json)记录了源/目标、所选发布帧 `[0,120)`、归档成员和保护对象。官方 [`motion2outfit_training.json`](https://github.com/PerceivingSystems/bedlam2_render/blob/main/config/motion2outfit_training.json)把该身体/动作指向 `gr_eric_033_3XL`；本地既有官方 `archive_map.json` 将衣物成员 `nl_2066_3XL/2205/2205.npz` 定位到 `b2_clothing_npz_046.tar`。本批通过已授权的 BEDLAM2 账号定向获取了该 **132,276,684 B** 成员，没有下载整个约 5 GB 衣物分片。单次大范围请求卡住后，改用 4 MiB 范围请求并保留失败记录；登录凭据和带签名链接未写入产物。数据仍受 [BEDLAM2 非商业科研许可](https://bedlam2.is.tue.mpg.de/license.html)约束，不对外分发原衣物或纹理。

实际衣物[成员获取记录](../../outputs/experiments/bedlam_m03_native_cloth_transfer_a2_20261010/acquired_attempt_3/nl_2066_3XL_2205_clothing.npz.acquisition.json)与两份定位/读取[失败说明](../../outputs/experiments/bedlam_m03_native_cloth_transfer_a2_20261010/retrieval_issue_attempt_1.json)、[续查说明](../../outputs/experiments/bedlam_m03_native_cloth_transfer_a2_20261010/retrieval_issue_attempt_2.json)单独保留。原 NPZ 只做本地依赖，不复制到交付缓存里。

衣物 NPZ 实际含 163 帧、每帧 71,794 个顶点，固定 133,979 个三角面，另有 51,346 个 UV 顶点和同面数的 face-corner UV 索引。选取的前 120 帧与原动作发布帧逐一对应，未插值、循环或调整播放速度。原缓存首帧至 BEDLAM2 原身体的最近顶点距离中位约 16 mm、95 分位约 46 mm，支持身体/衣物坐标配套，但不是零穿模或真实物理正确性的证明。

同一官方纹理索引把 `gr_eric_033_3XL/texture_01` 指向 `b2_clothing_textures_04.tar`；已定向取得原 diffuse 和 normal 两张 4096×4096 图，并通过原 face-corner UV 消费。未把 M03 贴身衣物的 UV 或颜色冒充原服装材质。下载的动作、衣物和贴图均只存于本批本地受限产物目录，不公开发布。

## 转移方法和证据

使用本地 BEDLAM2 neutral locked-head 模型、原 16 维 shape 及源姿态求值出 120 帧原身体。[源身体缓存](../../outputs/experiments/bedlam_m03_native_cloth_transfer_a2_20261010/source_body/manifest.json)只用于衣物对应，不替换 M03。原身体与目标标准 male 身体均为有序 10,475 顶点、20,908 面；本次逐元素核实有序面集一致。源帧 0 对原衣物每个顶点建立一次固定身体三角形/重心绑定，之后逐帧把该三角形上“目标身体−源身体”的加权位移加到**原仿真衣物逐帧缓存**，保持原衣物面、UV、顶点顺序及布料褶皱时间变化。没有逐帧换最近面、删面、修改身体参数或另加源 root。

[转移检查](../../outputs/experiments/bedlam_m03_native_cloth_transfer_a2_20261010/transfer_attempt_1/validation.json)确认输出 120 帧有限且时间/拓扑固定，5 个代表帧里衣物到目标身体的距离分布与原衣物到原身体相近。代表帧衣物边长比第 1–99 百分位约 0.78–1.92；这是需要人眼注意的局部形变诊断，不能当作全片段质量通过。最大相邻衣物顶点位移约 0.125 m，也可能包含原动作/松衣运动，须在连续播放中判断是否闪烁或局部翻折。小面积穿模可接受；明显脱体、持续翻折或闪烁不能仅凭数值检查放行。

同一 120 帧里，原衣物最大相邻顶点步长约 0.130 m、转移版约 0.125 m，二者最大值都在 71→72 帧；全体顶点步长的 99 分位分别约 0.0728 m 与 0.0711 m。转移没有在此指标上放大整体抖动，但它仍无法证明局部褶皱的动态观感。

定向回算选定 5 帧，源衣物加固定绑定位移与输出逐值一致，绑定重心权重和误差不超过 1.2×10⁻⁷。与原衣物同帧三角面法向相比，探针约 0.4%–1.1% 面发生反向；这不是动态质量通过，应在视频中注意是否集中成可见折转。没有通过删面或改源姿态消除这些局部变化。

[权威资产保全回核](../../outputs/experiments/bedlam_m03_native_cloth_transfer_a2_20261010/authority_preservation.json)确认新旧两版引用同一 M03 body、faces、R138、head、groom、来源组和 120 帧时间基；仅新衣物几何/UV/材质发生变化。原贴身版的缓存与预览没有覆盖。

固定视角的 5 帧[灰色材质探针联系图](../../outputs/experiments/bedlam_m03_native_cloth_transfer_a2_20261010/probe_contact_gray.png)与已接受简单衣物同姿态、同镜头对照：转移版呈现有褶皱的长袖上衣和长裤，探针未见大面积拉飞。这还不是动态人工验收，也不是最终贴图效果。

带原 `texture_01` diffuse/normal 的 5 帧探针同样保留 M03 身体、头发、光照和镜头，只更换衣物几何/材质；[同镜头贴图对照](../../outputs/experiments/bedlam_m03_native_cloth_transfer_a2_20261010/probe_contact_textured.png)里正侧视图的 UV 映射正常，未见明显贴图错位。采样 64、每向 512×512、无降噪，约 45.17 秒。探针和完整预览均是人物 studio，不是 A1 道路或 DS 输出。

## 后续评审状态

[完整同时间轴对照页](../../outputs/experiments/bedlam_m03_native_cloth_transfer_a2_20261010/index.html)将已接受的 M03 简单衣物放左侧，BEDLAM2 转移衣物放右侧，各有前/侧固定视角。[120 帧逐帧页](../../outputs/experiments/bedlam_m03_native_cloth_transfer_a2_20261010/builds/M03_KIT11_BEDLAM_native/preview_full/index.html)及[机器交付](../../outputs/experiments/bedlam_m03_native_cloth_transfer_a2_20261010/delivery.json)核实两版均为发布 30 Hz、120 帧，合成 MP4 显示 PTS 与原时间表逐帧一致、误差 0 tick。全片段新渲染 230 张视角图，加上复用的 10 张探针图，未删帧或用静帧冒充连续预览。助理只核查过探针与少量相邻画面，**未完成整段动态人工验收**。

[执行记录](../../outputs/experiments/bedlam_m03_native_cloth_transfer_a2_20261010/execution_record.json)给出：衣物成员 132.28 MB、两张原贴图合计约 9.64 MB；成功范围传输累计约 145.85 MB（含归档头核查），没有整包下载。数值转移约 5.65 秒；灰材质探针 44.15 秒、原贴图探针 45.17 秒、完整原贴图渲染 667.55 秒，GPU0 UUID 已记录；打包约 9.47 秒。任务目录约 314 MiB，旧资产、失败现场与源文件保留。两张 GPU 同时被占用时等待至 GPU0 空闲，没有终止其它进程。本批没有下载模型、安装依赖、运行公共道路或训练。

**用户后续决定（2026-10-10）：**已观看完整对照，认为 M03 转移衣物观感可以接受。独立[接受记录](../../outputs/experiments/bedlam_m03_native_cloth_transfer_a2_20261010/user_acceptance_20261010.json)和交付 manifest 已同步；修改前的 manifest 单独保留。接受范围仅为 M03＋KIT11_walk 原 120 帧＋此版 `gr_eric_033_3XL` 转移衣物及 M03 groom，不等于原 BEDLAM2 身体重建、其它动作的穿衣质量或公共场景验收。原 M03 简单衣物版本依然保留。

统一组合清单现将此衣物登记为 `M03_KIT11_BEDLAM_transfer`，其类型是**已生成缓存的限定组合**，不是可任意换人体或动作的衣型。`compose_asset_a2.py --name M03_KIT11_BEDLAM_transfer --plan` 核对 M03 身体、KIT11_walk 动作、M03 groom、完整 `[0,120)` 原时间窗及权威缓存；`--build` 只回用这个已验收成品，不重新变形或仿真。任何换人体、换动作、换 groom 或截段均拒绝，若未来要适配须另建派生版本并重新验收。A1 原消费者目前偏向 simple 衣物材质；转移版还需要固定 faces、face-corner UV 和贴图读取能力，公共场景消费仍未验证，不能自动加入 A1 活动道路/训练池。来源身份仍沿用 KIT/11 与 M03 家族，换衣服不产生新人物或独立验证来源。

工具参数查询（本批输出按 append-only 规则保留；以下命令只显示参数，不重复生成）：

```bash
/mnt/data1/khb/app/blender-5.1.1-linux-x64/blender -b -t 4 --python tools/experiments/evaluate_bedlam2_locked_body_a2.py -- --help
PYTHONPATH=src python tools/experiments/transfer_bedlam2_cloth_to_m03_a2.py --help
python tools/experiments/prepare_bedlam_m03_transfer_preview_a2.py --help
python tools/visualize/package_bedlam_m03_transfer_a2.py --help
PYTHONPATH=src python -m unittest tests.test_asset_composition_a2
```
