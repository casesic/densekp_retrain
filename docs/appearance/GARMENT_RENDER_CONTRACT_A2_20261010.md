# 衣物描述、Blender 装配和固定缓存窗口（A2，2026-10-10）

本批将 M03＋KIT11 的简单衣物和用户已接受的 BEDLAM2 转移衣物写成同一版本化资产描述，并把旧 studio 的材质分支接到一个可供后续道路渲染调用的 Blender 函数。**只改资产侧与 studio；A1 道路、ROI、打包和训练代码均未修改。** 旧 120 帧身体、衣物、UV、贴图、R138、groom、时间与接受记录保持权威状态。本批 `integration_only`，不产生新道路图像或训练样本。

## 交付接口

CPU 可导入的 [`garment_contract.py`](../../src/densekp_retrain/assets/garment_contract.py) 定义 `densekp_garment_asset_v1` 和 `densekp_garment_window_view_v1`。这两个实例可直接供程序读取：[M03 简单衣物](../../outputs/experiments/garment_render_contract_a2_20261010/M03_simple.garment.json)、[M03 BEDLAM2 转移衣物](../../outputs/experiments/garment_render_contract_a2_20261010/M03_BEDLAM.garment.json)。描述分别声明：

| 字段 | 简单衣物 | 固定转移缓存 |
|---|---|---|
| `geometry_kind` | `simple_barycentric_shell` | `fixed_vertex_cache_face_corner_uv` |
| `material_kind` | `GarmentRest_continuous_shell` | `native_face_corner_uv_diffuse_normal_v1` |
| 几何/材质 | 目标 rest 绑定和程序化材质引用 | 已生成逐帧顶点、固定面与 face-corner UV、原 diffuse/normal |
| 组合能力 | 原统一组合入口的 simple 兼容检查 | 仅 M03＋KIT11_walk＋M03_groom 的原完整 `[0,120)` 权威成品；消费时可选连续子窗口 |

描述还引用原模型空间 Y-up、米制、source-root 所有权、角色/动作来源组、许可和独立的技术/人工/场景状态。缺几何、UV、贴图、拓扑或时间依赖时 `validate_description` 报错；未知描述版本/材质类型拒绝。完整固定资产仍由原 manifest 和 [用户接受记录](../../outputs/experiments/bedlam_m03_native_cloth_transfer_a2_20261010/user_acceptance_20261010.json)授权；子窗口并不解锁换人体、换 motion、换 groom 或重做仿真。

[`garment_materials.py`](../../src/densekp_retrain/rendering/blender/garment_materials.py) 提供 `add_garment_materials(mesh, description)`。simple 分支委托原 `materials.add_clothing_materials`；固定衣物按 `uv_vertices[faces_uv]` 写入实际 mesh loops，使用独立 sRGB diffuse 与 Non-Color normal 图像、NormalMap 节点、原粗糙度 0.58 和原默认底色。每次装配先清除已有 UV 层与材质槽，重复调用不会叠加；不改衣物面、顶点、物体矩阵或实例 ID。旧 studio recipe 的 `native_clothing_material` 经薄适配仍可读，也可直接传 `garment_description`；它们进入同一函数。普通资产描述模块不会导入 bpy、构造模型或触发渲染。

## 原缓存时间视图

两个固定转移衣物视图是[完整 `[0,120)`](../../outputs/experiments/garment_render_contract_a2_20261010/views/M03_BEDLAM_full.json)及[原缓存 `[15,106)`](../../outputs/experiments/garment_render_contract_a2_20261010/views/M03_BEDLAM_15_106.json)。后者是 91 条原发布 30 Hz 记录，首末跨度 3 秒。另有一份[简单衣物同窗口对照](../../outputs/experiments/garment_render_contract_a2_20261010/views/M03_simple_simple_15_106.json)。视图只保存权威引用、原 `asset_frame=15…105` 与局部 `view_frame=0…90` 的对应；无人体/衣物数组副本、插值、重采样、循环或新的制作 window。原 `source_root_relative` 的参考帧仍是完整资产的 frame 0；视图开始不重置 root。

[CPU/A1 只读检查](../../outputs/experiments/garment_render_contract_a2_20261010/view_validation.json)实际回读 120＋91＋91 帧的 body、cloth、head 和 R138，全部与原帧数组一致，`R @ body` 最大回读差 0。A1 `geometry_paths`、`select_timeline` 可读取对应路径/时间；`person_time.map_timeline` 把 91 条发布帧按 10 Hz 映射成原帧 `15,18,…,105` 的 [31 条记录](../../outputs/experiments/garment_render_contract_a2_20261010/a1_person_time_31.json)。原 KIT 100 Hz 采集帧到 BEDLAM2 发布帧及真实曝光时间仍未知，映射只使用发布 30 Hz 时钟。对视图首/中/末，在固定场景矩阵下用 A1 `person_matrices` 验证 episode 起点源 root 仅扣一次；body、cloth、R138 同一模型坐标，groom 仍按 scene×head 合成。这个代数回读不等于道路消费者已经接通。

## 独立装配和静帧对照

修改 studio 前，原 `native_clothing_material` 实现单独冻结在[旧分支文本](../../outputs/experiments/garment_render_contract_a2_20261010/legacy_native_material_before.py.txt)，验证时从独立的 [`garment_material_before_a2.py`](../../tools/validate/garment_material_before_a2.py)装配，未让新旧都调用新函数。[Blender 无渲染装配检查](../../outputs/experiments/garment_render_contract_a2_20261010/material_assembly_audit_complete.json)对 simple 和固定衣物分别比较旧/新：顶点、面、UV loop、面材质分配、节点参数与连接、图片颜色空间、物体矩阵及实例 ID 一致；重复装配的槽和 UV 层仍固定。Blender 同场景创建两个材质时会给第二个 datablock 名称自动加 `.001`，比较时仅规范化该自动后缀，保留节点与材质设置逐项比较。

固定衣物在已接受 studio 配方、GPU1、64 samples、512×512、固定 front 镜头下，旧/新各渲染 frame 0 和 60，最终对照为 **4 张静帧**。首次新装配得到第 0 帧 2 个、第 60 帧 1 个颜色通道相差 1/255；随后有界修正了面序显式核查及同图同颜色空间的 image datablock 复用，保留初版静帧并只重算新装配 2 张。[最终像素与几何比较](../../outputs/experiments/garment_render_contract_a2_20261010/render_comparison/validation_final.json)显示最终旧/新两帧 **像素逐值一致**，衣物顶点/面/UV loop/材质槽、对象矩阵和实例 ID 也逐值一致，材质节点记录一致。这里共有 6 次实际成功静帧渲染，其中 2 张为保留的修正前记录；未重做 120 帧视频。旧、新图及审计记录均在[对照目录](../../outputs/experiments/garment_render_contract_a2_20261010/render_comparison/)中。首次沙箱内 GPU 不可见，失败空目录保留；经既有授权在空闲 GPU1 重新执行。旧、首次新、最终新渲染分别 28.73/29.17/27.41 秒。

13 项定向 Python 测试通过，覆盖描述类型/缺失项/错误 UV、锁定组合、非法窗口、源索引、旧统一入口等。CPU 全视图回读约 0.67 秒；3 次成功静帧批次累计渲染约 85.31 秒，低于 20 分钟预算。[执行记录](../../outputs/experiments/garment_render_contract_a2_20261010/execution_record.json)记载本批新目录约 20.4 MiB，以 6 张 PNG 和少量元数据为主；不含完整新网格缓存或纹理副本。[冻结保护清单](../../outputs/experiments/garment_render_contract_a2_20261010/frozen_scope.json)和 [能力/来源清单](../../outputs/experiments/garment_render_contract_a2_20261010/capability_and_sources.json)可供交接。

## 来源与 A1 交接

机器清单分列五段已接受骑行动作、M04/KIT425、M03/KIT11 两种衣物组合，并区分组件可组合性、固定缓存限制、材质、root、人工与历史场景证据。p1 两段同参与者/录制；M03 换衣不产生新人物或新 motion；KIT11 与 KIT425 发布路径不同，演员是否独立仍 `unknown`。bikefit 视频许可仍未知，BEDLAM2 服装仍受非商业科研许可约束。所有条目都只供 `integration_only`，不据此划分 train/val；当前 A1 场景状态须以 A1 独立证据为准，不能从旧 A2 manifest 的 pending 字段推断最新进度。

[A1 机器交接](../../outputs/experiments/garment_render_contract_a2_20261010/a1_handoff.json)列明三个接入点：`dynamic_scene.adapt_proposal` 当前只接受 `GarmentRest_continuous_shell`，需识别新材质描述和视图的原资产帧；`render_road_pass_sources.py` 创建衣物处当前无条件调用原 simple 材质，需加载真实衣物 faces 后调用 `add_garment_materials`，逐帧按原资产帧更新顶点；`mixed_road_pass` 时间规划应消费所选视图，保持完整 source-root 参考与一次固定场景变换。本批未修改这些 A1 文件。材质模块准备好不等于双相机、ROI、Dataset 或训练已验收。

复用命令（输出路径已存在时按 append-only 规则拒绝覆盖；另选新目录运行）：

```bash
python tools/experiments/select_garment_window_a2.py --describe-simple outputs/experiments/walking_composition_a2_20261010/builds/M03_KIT11_walk/delivery_manifest.json --output /tmp/M03_simple.garment.json
python tools/experiments/select_garment_window_a2.py --describe-fixed outputs/experiments/bedlam_m03_native_cloth_transfer_a2_20261010/builds/M03_KIT11_BEDLAM_native/delivery_manifest.json --base-authority outputs/experiments/walking_composition_a2_20261010/builds/M03_KIT11_walk/delivery_manifest.json --output /tmp/M03_BEDLAM.garment.json
python tools/experiments/select_garment_window_a2.py --select-window /tmp/M03_BEDLAM.garment.json --window 15 106 --output /tmp/M03_BEDLAM_15_106.json
PYTHONPATH=src python -m unittest tests.test_garment_contract_a2 tests.test_asset_composition_a2
```

原完整组合的 `compose_asset_a2.py --name M03_KIT11_BEDLAM_transfer --plan` 仍检查强绑定，`--build` 只引用既有已接受成品。子窗口须走新选择视图入口，不能修改该制作 recipe 的 `[0,120)` 锁定范围。
