# participant_1 两段 M02 骑行动作：衣物与头发动态接入

**当前状态（2026-10-10）：两段 M02 身体动态及 simple 衣物、groom 的连续预览均获用户人眼接受。** 本报告仅涉及 `integration_only` 资产预览，不代表道路场景消费或独立验证来源。下文的“待评审”保留为本轮最初交付时的历史状态。

统一观看入口：[两段穿衣连续预览](../../../outputs/experiments/bvh_p1_dressed_assets_a2_20261010/index.html)。每段各有原速 MP4 和逐帧页；原[身体双视角对照](../../../outputs/experiments/bvh_p1_smplx_pilot_a2_20261010/index.html)供比较。两段均已完成 96 帧穿衣渲染与编码，用户现已接受其连续观感。

## 接受决定与权威身体

用户已观看并接受 `cycling_M02_v3` 的 participant_1 两段身体动态。只把总/逐段身体交付清单中的 `body_human_status` 更新为 `accepted_by_user`，修改前分别保存 `delivery_manifest_before_body_acceptance_20261010.json`，另存[接受记录](../../../outputs/experiments/bvh_p1_smplx_pilot_a2_20261010/body_acceptance_20261010.json)。原身体参数、网格、138点、预览与技术验证未改；穿衣观感待新一轮观看，公共场景仍未消费。

后续用户回复“可以接受”，确认两段 **M02 穿衣与头发的连续预览** 观感均可接受；见[穿衣接受记录](../../../outputs/experiments/bvh_p1_dressed_assets_a2_20261010/dressed_acceptance_20261010.json)。这一决定只更新人工观感状态，不改变下述原始数值证据、已知轻微局部穿插、来源分组、许可或 `public_scene_status=not_consumed`。

两段权威身体分别是 `p1_comfortable/fit_attempt_1/parameters.npz`（原帧 `[7560,7656)`）与 `p1_alternate/fit_attempt_1/parameters.npz`（`[9600,9696)`）。各 96 帧，asset `0…95`，BVH 原文 `Frame Time=0.0416667 s` → `416667/10000000 s`。原始帧、源标称时间和片段相对时间各自保留；这不是实际相机曝光 PTS。两段未拼接、不循环、不变速、不插值。身体已由前批完成一次共享坐标/尺度标定并在模型 Y-up、pelvis-centered 空间，源 root 不进入道路位移，本轮不再应用标定或逐帧脚底贴地。

## M02 绑定与全段数值检查

[角色 manifest](../../../outputs/experiments/cycling_ten_adults_candidates_20260930/characters/cycling_M02_v3/manifest.json)指向自身 v3 的 `simple_clothing_v1` 和父 M02_v2 的标准身体 reference、`groom_v5`，其 body geometry 路径与角色 reference 对应。衣物使用已有 body barycentric 权重、固定顶点/面集和 manifest 记录的 4 mm 法线偏移；不是 M01 逐帧衣物。groom 使用同一固定 active 曲线集合、一份原参考及本动作各 96 个 head 矩阵；不是旧动作的 head 矩阵，也没有物理毛发。头发 reference 在 M02 shaped-rest pelvis-centered 模型空间。以后 body/cloth/R138 应用同一个 `scene_from_model[t]`，groom 应用 `scene_from_model[t] @ head[asset_frame]`，全局朝向只应用一次。

CPU 全段准备结果：中度弯肘/较前倾的重新求值身体对冻结缓存最大差均约 `2.38e−7 m`，R138 回读最大差均约 `2.38e−7 m`；各段衣物 3116 顶点、6032 面，groom 固定 active 80397 曲线、2559631 个点，数组有限且无逐帧删面/删毛。头皮锚点近似最大差约 `8.47/9.90 mm`。局部衣面相对绑定基面的反向面最多 `23/30`，仅作定位，不是全身穿插或质量分数；轻微局部穿模与小面积折转允许。头发始终由各段身体的 head15 刚性变换驱动，未再次应用 root。源身体全段固定shape、参数和时间回读未改。数值不代替连续人眼检查明显脱体、闪烁或拉飞。

## 有界预览与接口

[冻结配方](../../../outputs/experiments/bvh_p1_dressed_assets_a2_20261010/frozen_manifest.json)分别选了首末、中段和真实膝屈极值各 5 帧。探针沿用先前已接受的 studio 双固定方向、每向 `512×512`、Cycles `64 samples`、无降噪、固定灯光/相机/色彩。两段静帧检查中未见明显衣物脱体或头发冻结；**连续观感仍由用户看原速视频决定**。探针实测每帧约 7.3 s，保守估计两段完整预览加封装约 1677 s，低于 1800 s 总计算预算；实际完整计算更快，见下文。

每段交付独立 manifest，引用权威身体、标准 male 30 维 shape、原时间表、衣物缓存/面集/绑定/材质、groom reference/active 集合和该段 head 矩阵。最低身体点与 body/cloth 全段范围仅供 A1 选择一次固定骑行高度；不继承 M01/F02 的道路高度。A1 的 `dynamic_scene.adapt_proposal`、`geometry_paths`、`select_timeline` 和 `person_time.map_timeline` 已分别只读消费两段全部 96 条原时间，精确映射误差为 0；见[中度弯肘适配检查](../../../outputs/experiments/bvh_p1_dressed_assets_a2_20261010/p1_comfortable/a1_adapter_check.json)与[较前倾适配检查](../../../outputs/experiments/bvh_p1_dressed_assets_a2_20261010/p1_alternate/a1_adapter_check.json)。这仅表示资产与整数时间可读，**不是**道路轨迹、相机、ROI、观测或 packed rows 已生成。

中度弯肘段的模型空间身体最低 Y 在 `−0.958…−0.829 m`，较前倾段为 `−0.961…−0.846 m`；它们是全段几何范围，不是自行车轮胎或路面接触高度。独立的 A2 资产侧检查器也对两段实际 proposal 执行身体/R138 及任意刚性场景矩阵交换检验、衣物绑定与固定 groom 集合回读，均通过：[中度弯肘](../../../outputs/experiments/bvh_p1_dressed_assets_a2_20261010/p1_comfortable/asset_side_validation.json)、[较前倾](../../../outputs/experiments/bvh_p1_dressed_assets_a2_20261010/p1_alternate/asset_side_validation.json)。A1 仍需选择固定放置高度及场景矩阵，并实际生成/检查双路观测，才能把 `public_scene_status` 升级。

各段最终 MP4 经解码均为 96 帧，标称 `10000000/416667` 的视频帧率；容器 PTS 与原 BVH 有理数时刻最大差约 `1.2 µs`，只属于封装舍入。探针成功的 5 帧直接复用，各补渲 91 帧，画面标注 target M02、asset 帧、源帧与相对标称时刻。整段固定相机视野，无逐帧跟踪缩放。轻微局部穿插、小范围衣面折转以及既有头发亮度特征按用户标准允许；本报告只确认代表静帧无明显脱体，不能替代完整动态人眼验收。

## 来源、预算与保护

两段都是 Zenodo 10668611 中 `1_RB_M42_20230719_PK` 的同一参与者和录制，目标也同属 `cycling_M02` 家族。换片段、外观或 scene seed 不能产生独立 train/val。源[官方记录](https://zenodo.org/records/10668611)及本地 `zenodo_record.json` 标 CC BY 4.0，应署名 Kyriakou、Chrysanthou、Kyriakou；SMPL-X、M02 材质/纹理和 groom 保留各自许可。全部仍为 `integration_only`。

计算前检查两张 RTX 3090、显存/进程、内存和磁盘；CPU 准备未用 GPU。Blender 在沙箱中首次无法枚举 CUDA，失败目录保留；随后使用获准的 GPU0 UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2` 完成各 5 帧探针。A1 的 `render_continuous_roi.py` 曾占用 GPU0，因此等待其 31 帧任务完成、显存回到空闲状态后才顺序渲染两段，不停止或争用。最终[执行记录](../../../outputs/experiments/bvh_p1_dressed_assets_a2_20261010/execution_record.json)：中度弯肘 CPU 准备约 `5.5 s`、探针 `52.0 s`、补渲 `652.9 s`、封装 `2.1 s`；较前倾为 `5.2/51.5/653.4/2.2 s`。成功计算合计约 `1424.8 s`，短暂失败约 `3 s`，仍低于 1800 s。新增目录约 `91.5 MB`（含保留的失败现场）。一次封装器局部变量覆盖导致首版已编码视频未生成 HTML，该版本保存在 `p1_comfortable/preview_package_failure_attempt1`；修正后两段最终 HTML/MP4 均回读通过，身体和渲染图像未改。未下载、安装、重拟合、跑公共道路或训练。

[两个组合的总 manifest](../../../outputs/experiments/bvh_p1_dressed_assets_a2_20261010/delivery_manifest.json)和 A2 组合索引分别追加 `p1_comfortable`、`p1_alternate`，已保留追加前索引副本。当前状态分开记录：身体 `accepted_by_user`、衣物/groom 数值 `passed`、穿衣观感 `accepted_by_user`、A1 适配器 `passed_read_only`、公共场景 `not_consumed`。两段仍属同一原人物/录制和 M02 家族，不能据此作两个独立验证来源。

复用方法：先按[冻结入口](../../../tools/experiments/freeze_bvh_p1_dressed_a2.py)生成新的独立配方，再以 `--asset-root` 将[共用准备器](../../../tools/experiments/prepare_bvh_p2_dressed_a2.py)、[studio 渲染器](../../../tools/render/preview_apd_6526_dressed.py)和[预览封装器](../../../tools/visualize/package_bvh_p2_dressed_a2.py)指向各段新目录。现有目录禁止覆盖；p2 历史入口和产物保持不变。
