# participant_1 两段 BVH → cycling_M02_v3 身体适配

**状态：两段各 96 个原始相邻帧的标准 SMPL-X 身体、网格及项目 R138 已完成数值回读；用户随后观看并接受两段 M02 身体动态。** [统一评审页](../../../outputs/experiments/bvh_p1_smplx_pilot_a2_20261010/index.html)依次播放中度弯肘和较前倾两段，各有[逐帧页：中度弯肘](../../../outputs/experiments/bvh_p1_smplx_pilot_a2_20261010/p1_comfortable/preview/index.html)与[逐帧页：较前倾](../../../outputs/experiments/bvh_p1_smplx_pilot_a2_20261010/p1_alternate/preview/index.html)。本报告记录身体阶段当时的待评审状态；新的[人工接受记录](../../../outputs/experiments/bvh_p1_smplx_pilot_a2_20261010/body_acceptance_20261010.json)仅更新观感决定，不改参数、网格、时间或预览。身体接受不代表衣物、头发或道路场景通过。

另已按用户新决定，将 [participant_2＋M01 穿衣报告](BVH_P2_DRESSED_ASSET_A2_20261010.md)的观感状态更新为 accepted，并保存[独立接受记录](../../../outputs/experiments/bvh_p2_dressed_asset_a2_20261010/dressed_acceptance_20261010.json)及旧 manifest 副本。该决定没有改动 participant_2 的身体、衣物、groom、视频或数值历史；其公共场景状态仍为 `not_consumed`。

## 冻结输入和角色

[总冻结清单](../../../outputs/experiments/bvh_p1_smplx_pilot_a2_20261010/frozen_manifest.json)在拟合前锁定同一个原 BVH `1_RB_M42_20230719_PK.bvh` 的两个**不拼接**窗口：`p1_comfortable=[7560,7656)`、`p1_alternate=[9600,9696)`。每段 96 帧，asset 帧 `0…95`；原帧和逐帧标称时间见各段 `time_mapping.csv`。原 BVH 头的 `Frame Time` 原文为 `0.0416667 s`，保存为 `time_base=1/10000000`、每帧 `416667 ticks`；首末间隔 `3.9583365 s`，含末帧展示约 `4.0000032 s`。这是发布 BVH 的标称采样，**真实曝光 PTS 未知**。没有改成精确 24/30 Hz、插值、循环、复制帧或把缓存索引双加到原帧。第一段回查本地 `source_clip.bvh`、`clip_native.npz` 与原文件，FK 最大差 `1.14e−13` 原单位；第二段从原 20060 帧 BVH 直接读取，既有局部缓存不覆盖该窗口，故没有虚构缓存回查结果。

选择 [cycling_M02_v3 manifest](../../../outputs/experiments/cycling_ten_adults_candidates_20260930/characters/cycling_M02_v3/manifest.json) 的真实标准 male SMPL-X、原 30 维固定 betas 和 reference faces。其身份家族是 `cycling_M02`，父资产 M02_v2；模型、父 manifest、geometry、simple clothing reference 和 groom reference 路径均存在，本轮只使用模型/shape/reference faces，**未**计算服装与头发。不同角色名本身不能证明身份独立；M02 家族与 M01/F02 家族的现有清单关系不同，但正式来源划分还要把场景中所有遮挡者和衍生关系纳入。

## 同一桥接方法与坐标权属

复用并最小参数化 [participant_2 拟合器](../../../tools/experiments/fit_bvh_p2_smplx_a2.py)、[回读器](../../../tools/validate/check_bvh_p2_smplx_a2.py)和[CPU 预览器](../../../tools/visualize/preview_bvh_p2_smplx_a2.py)；p2 的默认路径与旧产物不动。原 BVH 按已核查的层级及每关节 channel 顺序做 FK：位置通道含本地位移、取代 OFFSET，旋转按列出的 `Zrotation→Xrotation→Yrotation` 组合。具名[关节映射](../../../outputs/experiments/bvh_p1_smplx_pilot_a2_20261010/p1_comfortable/joint_mapping.json)覆盖骨盆、脊柱/头、肩肘腕和髋膝踝/弱趾点。原 Euler 没有直接填给 SMPL-X；优化固定 shape 上的 SO(3) 局部旋转，膝关节采用正向弯曲约束，手指/面部/眼/表情为中性默认。位置、骨段方向与适度时间正则共用 p2 方案，头颈只弱约束。

[共享对齐](../../../outputs/experiments/bvh_p1_smplx_pilot_a2_20261010/shared_alignment.json)先利用**两段冻结窗口**估计一次 proper yaw `−0.123609 rad` 和统一尺度 `0.0104936` 模型米/原单位；八组主要骨段比例取中位数。两段均使用这一相同矩阵/尺度，没有逐帧重定向、每帧缩放或按足底贴地。源 `+Y` 上轴与原单位尚属推断；此对齐用于目标角色动作适配，不能当作真实米制或摄影机/世界标定。源 root 保存于各段 `observations.npz`，目标导出 `transl=−M02 shaped-rest pelvis` 的全段常量，pelvis 居中；未来道路位移、yaw 和固定放置高度只归场景所有。

## 数值结果与观感边界

先用第一段 asset `0/24/48/72/95` 做 5 帧探针，核对左右髋膝、躯干方向、坐标和回读；随后各只做一次 96 帧完整拟合。第二段沿用同一冻结标定，也做了同样 5 帧低成本探针。没有修正版、换窗或参数搜索。各段的 [拟合验证：中度弯肘](../../../outputs/experiments/bvh_p1_smplx_pilot_a2_20261010/p1_comfortable/fit_attempt_1/validation.json)、[拟合验证：较前倾](../../../outputs/experiments/bvh_p1_smplx_pilot_a2_20261010/p1_alternate/fit_attempt_1/validation.json)和独立全段回读均有限、固定 shape、10475 顶点标准 faces、源帧及整数时间连续。保存参数重新求值身体、前 22 关节和 `R[138,10475] @ body` 的全段缓存差均为回读精度内 `0`；138 点来自标准身体表面，不是 BVH 关节插值。两段 pelvis 中心最大误差为 `0`，不要求零自交或精准车把/脚踏接触。

| 指标 | 中度弯肘 | 较前倾 |
|---|---:|---:|
| 具名关节位置残差均值 / P95（模型米） | 0.0411 / 0.1170 | 0.0420 / 0.1209 |
| 最大相邻局部旋转 | 13.40° | 13.42° |
| 目标躯干向上轴偏角中位数（pelvis→spine3，本地几何近似） | 29.49° | 40.86° |
| 目标左右踝高度相关 | −0.998 | −0.996 |
| 目标肘内角中位数，左/右 | 139.0° / 123.2° | 126.8° / 117.1° |

同一共享坐标下，第二段较前倾且肘更弯，左右足高交替未被时序项抹平。原骨架相应躯干角约 `32.16°/45.08°`，不同体型与关节定义使数值无需完全相等。关节残差是**源骨架与目标固定体型适配约束**的误差，不是现实三维重建精度；脚高相关也不单独认证精确踩踏。两段视频以同一段内固定视角/尺度显示源 FK 和目标身体的 X–Y、Z–Y 对照，画面保留 source/asset 帧与标称时间，原源 root 被显式移除供原地资产评审。MP4 各解码 96 帧，理想有理数时间与容器 PTS 最大差约 `1.2 µs`。助理只检查了静帧与程序回读，没有将其冒充连续人工验收；用户请观看统一评审页判断有无反膝、突跳或姿态不自然。

## 交接、来源与资源

[两个身体候选的机器清单](../../../outputs/experiments/bvh_p1_smplx_pilot_a2_20261010/delivery_manifest.json)分别链接权威参数、身体/关节/R138 缓存、原时间、映射、源 root 观察与预览。A1 当前只能将其视为 `body_only` 候选：衣物/groom 动态未生成、固定骑行场景高度未选、公共道路生成器未消费、无相机/ROI/观测/packed rows。其时间接口需保留 `416667/10000000 s` 原表，不能套 p2 或场景默认的 30 Hz。两段均为同一 `zenodo_10668611_participant_1` 参与者和录制，目标 M02 家族也同一；不能相互划成独立 train/val。A1 原方案里关于 p2 未交付的文字是历史快照，不在本轮改写；独立行人遮挡者及完整来源关系仍待 A1 后续解决。

源由 [Kyriakou、Chrysanthou 和 Kyriakou 的 *Dataset of Motion Capture of Cyclists*](https://zenodo.org/records/10668611)发布，本地[官方元数据](../../../outputs/motion_inputs/cycling_fit_20260929/zenodo_record.json)记录 CC BY 4.0；SMPL-X 和 M02 人物/纹理各有自己的许可。所有派生内容仅 `integration_only`，不移入 datasets。

[执行记录](../../../outputs/experiments/bvh_p1_smplx_pilot_a2_20261010/execution_record.json)保留计算前 GPU UUID/显存、内存/磁盘与 CPU-only 约定。优化、回读和预览含失败重制累计约 `110 s`，低于 1800 s 限额；新目录约 `60.0 MB`（含保留的失败预览）。两次预览实施错误已显式留档：首次画面标签用了被关节循环覆盖的角色名；中度弯肘首次编码还把最后一帧 tick 当作帧间隔。原失败预览保存在各段 `preview_label_error_attempt1`，随后仅修正预览器并重新生成，**身体、参数和时间表未改**。最终视频帧率、解码数和显示文字均重新核查。全程不占用 GPU，没有下载、安装、训练或公共场景计算。

复用入口（需指定**新的空输出目录**，冻结脚本拒绝覆盖本次结果）：

```bash
/home/yuyue/anaconda3/envs/camerahmr/bin/python tools/experiments/freeze_bvh_p1_smplx_a2.py
/home/yuyue/anaconda3/envs/camerahmr/bin/python tools/experiments/fit_bvh_p2_smplx_a2.py --stage calibrate-p1 --root outputs/experiments/bvh_p1_smplx_pilot_a2_20261010
/home/yuyue/anaconda3/envs/camerahmr/bin/python tools/experiments/fit_bvh_p2_smplx_a2.py --stage probe --root outputs/experiments/bvh_p1_smplx_pilot_a2_20261010/p1_comfortable
/home/yuyue/anaconda3/envs/camerahmr/bin/python tools/experiments/fit_bvh_p2_smplx_a2.py --stage fit --root outputs/experiments/bvh_p1_smplx_pilot_a2_20261010/p1_comfortable
# 对 p1_alternate 使用相同 probe/fit、独立回读与预览入口；本次冻结目录已存在，勿直接覆盖。
```

完成于两段**身体**评审点；用户确认后才另批讨论 M02 的衣物/groom 和道路场景。
