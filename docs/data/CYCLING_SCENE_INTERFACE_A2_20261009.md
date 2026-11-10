# A2 → A1：APD6526 连续骑行资产与 road_pass 接口交接

日期：2026-10-09。全部 `integration_only`。资产侧 76 帧 CPU 回读通过；**骑行 road_pass 实际消费、投影、双路成像、compact/旁表/Dataset 回读尚未执行**。不能将资产 manifest 或公共绑定辅助函数通过写成场景链通过。

## 已交付资产

入口：`outputs/experiments/continuous_cycling_asset_a2_20261009/accepted/manifest.json`。

权威仍为 `apd_6526_smplx_pilot_20261008/fit_attempt_2/parameters.npz`：cycling_M01_v2，固定 30 betas、male、10475 顶点/20908 面；源帧 `[30,106)`，asset 帧 `[0,76)`。精确模型发行语义版本 unknown，`model_reference.json` 给出实际本地文件和模型 buffer；不另命名官方版本。用户本轮接受身体＋简单衣物＋原 groom 的 64 samples studio 观感，包括左臂快摆、轻微穿模和顶部亮度。旧报告的待评审记录保留，新决定在本轮冻结配方中记录。

`geometry/` 使用指向稳定旧产物的符号链接：vertices、faces、joints、densekp_3d、clothing_vertices、head_transforms。新 source_frames/asset_frames/timestamps 是小型索引；没有复制模型、纹理或逐帧 groom。parameters 是权威，缓存只供消费。clothing_reference/groom_reference/character_manifest 等沿用 A1 `densekp_dynamic_asset_v1` 的字段名。

manifest 使用显式 `densekp_dynamic_asset_a2_extension_proposal_v1`，**不是已被 A1 支持的新版本声明**。`extensions_effective_in_A1=false`；不要直接塞进旧采样器并期待能运行。

## 必须保持的时间与变换

原始整数时间戳首末：`1476871677560014`–`1476871680560014`；首末跨度 3 s，间隔 39949–40010 ticks。`time_base=[1,1000000]` 为根据 APD 25 Hz 推断的微秒，未升级为官方单位确认。保留 `time_mapping.csv` 和 `accepted/timeline.json`；名义 25 Hz 不等于精确均匀时钟。

每条 timeline 含 `asset_frame, source_frame, source_time_tick, episode_time_tick, time_base`，episode tick 为原 tick 减固定首 tick。一个 episode 的两相机必须消费同一条记录；禁止 `start+3*k`、改文件名伪装 30 Hz、各组件独立 nearest 或重复帧。

body/cloth/138 已在模型米中，pelvis 固定，transl 固定为负 shaped-rest pelvis。global_orient 已进入 body 和 H；不再额外叠它。APD 原髋路径在旧 `observations.npz` 中保留但已从动作移除，不能再叠 head_smoothed。

```text
body_world[t]  = scene_from_model[t] @ body_model[asset_frame[t]]
cloth_world[t] = scene_from_model[t] @ cloth_model[asset_frame[t]]
kp138_world[t] = scene_from_model[t] @ (R @ body_model[asset_frame[t]])
groom_world[t] = scene_from_model[t] @ head[asset_frame[t]] @ groom_reference
```

`scene_from_model[t]` 包含一次轴转换 C(x,y,z)=(x,-z,y)、场景 yaw/placement/path；H 已有头部链与身体整体运动。行进平移归场景，资产 root 残差为零。本轮 CPU 的固定任意矩阵只检查上述代数关系，**不是道路轨迹、道路高度或公共投影结果**。

## 当前 A1 实现的具体缺口

本轮只读检查 A1 源码和新报告 `docs/training/PEDESTRIAN_SEEDED_PILOT_A1_20261009.md`。A1 已有行人 3 时刻/6 ERP 的实际 compact 回读；完整两段、50步训练未完成。它的成功不等于 APD 接口已具备。

| A1 调用点 | 当前行为 | 需要的最小能力／建议字段（尚未实现） |
| --- | --- | --- |
| `data/road_pass.py::plan` | 固定31帧、30Hz、stride3、time_base1/10；source-root 模式 | 根据资产 native timeline 选帧；`timeline_ref`、`sampling_policy=native_observations`、实际 frame_count/time_base；按能力选择 root_mode |
| `tools/experiments/plan_road_pass.py::preflight` | `start+3*arange(31)`；读取 `source_root_relative`；只有一个固定 M；source_nominal_seconds=frames/30 | 支持 APD `root_mode=fixed_pelvis_model_origin_scene_trajectory_owned_by_scene`；逐条 asset/source/tick 映射；由公共道路逻辑生成 `scene_from_model[t]`，不能伪造行走参数 |
| 同上，放置高度 | 片段统一 min-Y lift（并非逐帧，但仍依据最低身体点） | `support_policy=fixed_scene_placement_height_no_pedestrian_foot_contact`、显式固定 `placement_height_m` 与依据；结合道路几何检查身体/道路，不能要求骑行脚底贴路 |
| `tools/render/render_road_pass_sources.py` | 索引范围<31；ground probe 固定0/15/30；脚底射线间隙−2cm..12cm；body/cloth固定M，hair=M@H | 动态帧数；依据 support_policy 执行合适检查，不粗暴关闭所有地面检查；每帧统一更新 body、cloth、hair 和 R138 检查中的同一 scene matrix |
| 同上衣物材质 | 直接 `add_clothing_materials`，偏向 GarmentRest shell | 已接受 simple 版可沿用；若未来采用 BEDLAM，需要 `material_kind=uv_textured_principled`，独立 face-corner UV、贴图色彩/法线/材质参数入口，不能把换颜色当 BEDLAM |
| `tools/experiments/consume_road_pass_probe.py` | report time_tick=k/time_base1/10；视频10Hz、PTS=k | 传递真实 episode/source ticks；按显式时刻编码；其它公共 DS/compact/框/Dataset 消费保持 |
| `data/sequence_packing.py::pack_rendered_sequence` | frame/obs tick=k；source tick=source frame，base1/30；必读 original_100fps_mapping | 旁表直接取 timeline；源 timestamp/base 独立；移除对特定行走来源映射的强制要求，unknown 不伪造 |
| `data/sequence_index.py` | 已支持 `expected_tick_step=null` | 当前 schema 无须为 APD另造一套；让上述生产者真实填写非均匀表，实际图像才创建 observed/packed rows |

A2 **未修改**这些 A1 文件及其测试。扩展应保留旧行人默认行为，由 A1 完成定向回归。不要新增 walking=true、伪造 `source_root_relative`/100fps 证明，或复制生成器绕过检查。

## 建议的下一个有界公共消费任务

先合入并验证 native timeline、scene-owned motion、support policy 以及生产者准确传时三项；简单衣物版不等待 BEDLAM 材质入口。

1. 同一 road_pass 入口，一角色、一短道路场景、一个固定 seed（提案 `20261009`）。A1 选择实际道路 placement/path，沿道路穿过有效域交界，rig 内部关系固定；冻结实际矩阵、固定高度和渲染参数后计算。
2. CPU 消费全部76条原时间表，身体/衣物/头发/138变换与双相机投影一致；不新增 motion、不循环、不重采样。
3. 建议探针 asset `[5,6,7]`／source `[35,36,37]`，原 ticks `1476871677760032,1476871677800042,1476871677840021`，episode ticks `200018,240028,280007`。这是未来任务建议，**本轮没有建立 job 或产生图像**；不能把三个时刻填成0/0.1/0.2s。
4. 同一公共 renderer→DS后处理→compact→框→sequence packing→Dataset。最多3时刻/6最终ERP；源和最终5760×2880、DS2880²、标定20260320。渲染设置由公共配方冻结，studio64不强制推广。实测预算含DS转换/打包，累计≤900s；不擅自降C分辨率。
5. 检查显式valid/person masks、框坐标与原图、图像/138 affine和真实索引；监督保持 `in_view & inside_full_crop`，双视共同集仅评价。没有成像的时间保持明确未生成状态，不伪造observed/packed行。

若必须重采样，另立目标时间表、左右源帧和权重；旋转采用旋转插值，先参数统一求值再生成所有组件，单独回归验证。本轮推荐保持原时刻，尚未实现重采样。

## 可运行的资产侧检查

```bash
python tools/experiments/prepare_cycling_asset_a2.py accepted --check-only \
  --validation-output /tmp/cycling_a2_readback_new.json
```

输出路径必须不存在。CPU 2线程、隐藏CUDA；实测每次约7秒（仅本机本资产），读取冻结模型，不拟合。检查76帧参数、时间、固定shape、身体/R138、公共衣物与head函数复算、groom有限值、scene/head复合代数；不会写源文件、运行公共场景或生成A1旁表。

来源组继续 `APD_5724486_v1_identity_session_unknown`，角色家族 `cycling_M01`。BEDLAM试制只是额外衣物来源；换衣、seed、邻段、循环或重采样不新增独立验证来源。仍停在资产与接口评审点。
