# BEDLAM2 KIT/11 行走动作接入 M03（A2，2026-10-10）

> 用户后续观看确认：本组合的连续动态效果可以接受，同时指出衣物看起来贴身。实际衣物是 M03 自身 `simple_clothing_v1` 的贴身表面衣型（最大参考偏移约 4 mm），不是 BEDLAM2 原配宽松服装或布料仿真。该观感接受仅适用于本组合，不提升来源独立性、许可或公共场景状态；原始数值与预览证据保持不变。见[独立人工评审记录](../../outputs/experiments/walking_composition_a2_20261010/M03_KIT11_walk_human_review_20261010.json)。

在已有统一 `compose_asset_a2.py` 行走入口和 M04/KIT425 回放基础上，本批从 BEDLAM2 官方资源**定向取得一条 158,659 B 的动作参数**，适配到现有 `cycling_M03_v1`，生成 120 个连续发布帧的身体、简单衣物、头发和项目 R138。数值流程通过，双视角 studio 动态观感待用户观看；尚未进入 A1 公共道路场景。全部为 `integration_only`。

## 获取与来源

- 官方下载目录的 `ground_truth/motions/b2_motions_npz_training.tar` 实际大小为 2,863,616,000 B。服务端支持字节范围，本批仅读取 TAR 头并取出 `motions_npz_training/nl_2066_3XL_2205.npz`，没有下载整包。[提取记录](../../outputs/experiments/walking_composition_a2_20261010/official_motion_access_escalated_v2_20261010/nl_2066_3XL_2205.npz.acquisition.json)记录了成员位置、传输字节和耗时。[本地动作文件](../../outputs/experiments/walking_composition_a2_20261010/official_motion_access_escalated_v2_20261010/nl_2066_3XL_2205.npz)不得对外分发。
- 本地官方 BEDLAM2→AMASS 映射 CSV 把发布 `body_id=nl_2066_3XL, motion_id=2205` 指向 `KIT/11/WalkingStraightForwards05_stageii.npz`；旧 M04 指向 `KIT/425/walking_medium05_stageii.npz`。两者不是同一个文件或录制目录，原演员是否不同尚无充分证据。即使最终确认不同，也仍须由 A1 连同目标、遮挡者和历史训练来源冻结划分，不能由本批标记为独立验证集。
- [候选机器记录](../../outputs/experiments/walking_composition_a2_20261010/downloaded_candidate_20261010.json)与[增量来源关系](../../outputs/experiments/walking_composition_a2_20261010/source_graph_updated_20261010.json)保留这一区分；原先的无参数候选清单仍作为当时的审计快照保存。
- 后续核查官方 [`motion2outfit_training.json`](https://github.com/PerceivingSystems/bedlam2_render/blob/main/config/motion2outfit_training.json)确认，这一发布组合的原配 outfit 是 `gr_eric_033_3XL`；本地已有官方 `archive_map.json` 将 `nl_2066_3XL/2205/2205.npz` 指向衣物分片 `b2_clothing_npz_046`，纹理索引将该 outfit 的 `texture_01` 指向 `clothing_textures_04`。这只是定位证据；本批没有下载/验证该衣物缓存和贴图，也没有声称它们与 M03 标准 male 身体兼容。本批要保留目标角色 M03，因此采用它自身已绑定的贴身简单衣物。若改用 BEDLAM2 原配衣物，需选择原配 BEDLAM2 身体整套消费，或另做显式转移与动态观感验收。
- 数据使用受 [BEDLAM2 官方非商业科研许可](https://bedlam2.is.tue.mpg.de/license.html)约束，未取得公开再分发许可。只使用既有账号访问，未记录凭据、签名下载链接或接受新的条款。
- 发布参数 `poses` 为 163×165、`trans` 为 163×3，记录标称 30 Hz；映射 CSV 则写原 KIT 100 Hz、原帧 `[0,489)`。二者时长不相等，准确的原采集帧到发布帧对应及真实曝光时刻**未确认**。预览和资产时间仅采用 BEDLAM2 **发布 30 Hz 时钟**，不宣称还原原 KIT 速度。

## 冻结组合与计算

[冻结输入](../../outputs/experiments/walking_composition_a2_20261010/frozen_new_motion.json)指定发布源帧 `[0,120)`、目标 M03 标准 male SMPL-X 的固定 30 维 shape、已有 M03 simple 衣物和 groom。120 帧的展示时长为 4 秒，首末时刻跨度 119/30 秒；无重采样、补帧或变速。源为 BEDLAM2 neutral locked-head、16 维 beta，目标形状采用 M03 自身 beta，**没有把源 shape 截断或补零**。源 165 维姿态块按已核查的标准 SMPL-X 关节次序复制到目标模型；未进行新拟合。表情使用中性默认。`target transl = source trans[t] − source trans[0] − target shaped-rest pelvis`，原 root 位移保留一次，后续 A1 只需给固定场景放置，不叠加道路恒速。

[规范化记录](../../outputs/experiments/walking_composition_a2_20261010/normalized_KIT11_M03/normalization_manifest.json)、[统一组合交付 manifest](../../outputs/experiments/walking_composition_a2_20261010/builds/M03_KIT11_walk/delivery_manifest.json)、[时间表](../../outputs/experiments/walking_composition_a2_20261010/builds/M03_KIT11_walk/timeline.json)以及 [A1 行走读取视图](../../outputs/experiments/walking_composition_a2_20261010/builds/M03_KIT11_walk/a1_walking_view.json)相互引用；不复制模型或 M03 原衣物/发型源资源。

[全段数值检查](../../outputs/experiments/walking_composition_a2_20261010/builds/M03_KIT11_walk/validation.json)通过：120 帧全部有限，目标 shape 与 body/cloth 拓扑固定，R138 从标准 10475 顶点身体回读差 0；衣物和 head 与源时间同步，未逐帧贴地。[补充 CPU 回核](../../outputs/experiments/walking_composition_a2_20261010/m03_kit11_readback.json)显示发布窗口 root 行进约 2.44 m，源姿态拷贝最大数值差约 2.4×10⁻⁷ rad，相邻关节最大 SO(3) 变化 13.76°（约在第 54→55 帧），非首帧子窗口只扣一次 root 起点。速度数值基于**发布时钟**，不是实测真实步速。A1 `geometry_paths`、`select_timeline`、`person_matrices` 的只读回读通过；**公共场景实际消费未执行**。

## 预览与评审边界

使用 M03 自有衣物、groom，在固定视野下以 512×512/视角、Cycles 64 samples、无降噪渲染前/侧两个方向。先检查源帧 0、60、119 的探针，再生成全部 120 帧；编码按发布 30 Hz 原时间表，页面显示 asset/source 帧。静帧探针无明显脱体或错位；这不等于连续动作已获人工接受。

观看入口：[连续双视角评审页](../../outputs/experiments/walking_composition_a2_20261010/builds/M03_KIT11_walk/preview_full/index.html)及 [统一入口](../../outputs/experiments/walking_composition_a2_20261010/index.html)。[编码检查](../../outputs/experiments/walking_composition_a2_20261010/builds/M03_KIT11_walk/preview_full/delivery.json)确认 MP4 解码显示帧 120 个，PTS 与发布 30 Hz 时间表误差 0 tick。新的 M03 身体和穿衣观感均 `pending_user`；已有 M04/KIT425 回放的数值一致性与先前人工/道路接受范围不因本组合改变。

[实际执行记录](../../outputs/experiments/walking_composition_a2_20261010/execution_followup_20261010.json)：探针 33.66 秒、完整 Blender 668.11 秒、CPU 制作 3.85 秒、打包 10.09 秒，已量测核心计算约 11.93 分钟，低于 30 分钟上限。探针在空闲 GPU1、完整段在后来空闲的 GPU0，记录 UUID；初次受沙箱限制的无 CUDA 渲染失败保留，没有终止其他进程。产物目录现约 136 MB（含本批之前的 M04 回放），定向动作文件只有 158,659 B。7 项定向测试通过。账号凭据及带签名下载链接未写入产物，未修改 A1、TODO、外部项目或既有权威资产。

复用命令（产物存在时按 append-only 规则不能覆盖）：

```bash
python tools/experiments/prepare_bedlam2_walk_motion_a2.py --help
PYTHONPATH=src python tools/experiments/compose_asset_a2.py --name M03_KIT11_walk --plan
PYTHONPATH=src python tools/experiments/compose_asset_a2.py --name M03_KIT11_walk --build
python tools/visualize/package_composed_asset_a2.py --output-root outputs/experiments/walking_composition_a2_20261010 --name M03_KIT11_walk --folder preview_full
PYTHONPATH=src python -m unittest tests.test_asset_composition_a2
```

新增的是一个可看、可由统一入口制作的**候选**。A1 下一步仍须确定目标与遮挡者的来源划分、真实场景固定放置高度、行走支持策略及实际公共场景消费；不要将旧 KIT425 人物或这个新角色直接写为正式 train/val。
