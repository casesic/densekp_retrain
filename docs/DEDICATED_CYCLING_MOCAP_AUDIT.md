# Phase 13 — Dedicated Cycling Mocap Audit

2026-09-06。主 motion source 转为专门 cyclist mocap。本轮没有 retarget、
SMPL-X fitting、自行车下载、渲染或训练，也没有修改现有 renderer/DenseKP。
运动学图是原始 BVH 关节轨迹分析，不是将 BVH mesh 接入 renderer。

## 下载、完整性与许可

来源：[Zenodo record 10668611](https://zenodo.org/records/10668611)，
DOI `10.5281/zenodo.10668611`，v1，作者 Kyriakou / Chrysanthou / Kyriakou。
实际以 `wget --continue` 下载文件至
`/home/yy/datasets/cycling_mocap/Dataset_cycing_mocap.zip`。

- 字节数 497387739，即 497.387739 MB（十进制）。
- 完整文件 MD5 `f5188f3088a928502142ed00f06637cc`，与官方一致，PASS。
- 只读取 ZIP central-directory 完成全 archive 清单；仅提取两个 BVH，
  未批量解压或处理 C3D/GPX。
- 官方 HTML 的 License 区域未显示具体名称，但
  [官方 API](https://zenodo.org/api/records/10668611) 明确返回
  `metadata.license.id = cc-by-4.0`、`access_right = open`。
  LICENSE_VERIFIED=YES，CC BY 4.0；不能仅凭 Open 作许可判断。
  元数据快照保存为 `outputs/dedicated_cycling_mocap_audit/zenodo_record.json`。
  后续使用应保留作者、DOI、许可及修改说明。

## Archive inventory

11 个 participant 目录，Road Bike 7、TT Bike 4；BVH/C3D/GPX 各 11。
文件命名中的 RB/TT 含义由官方记录说明确认。目录没有逐姿态/逐 RPM 的 BVH
子片段，每个参与者只有一段长录制。完整 members、压缩前后大小及参与者列表
在 `outputs/dedicated_cycling_mocap_audit/archive_audit.json`。

提取至 `/home/yy/datasets/cycling_mocap/candidates/`：

1. `1_RB_M42_20230719_PK/1_RB_M42_20230719_PK.bvh`
2. `2_RB_M27_20230809_PA/2_RB_M27_20230809_PA.bvh`

两者均是 Road Bike，避免先处理 TT/Standing。文件选择不预设姿态；片段选择
依赖后续数值检查。archive 没有 README 或额外 posture 时间表。

## BVH 结构与解释

| 项目 | participant 1 | participant 2 |
|---|---|---|
| ROOT | Hips | Hips |
| channel joints / End Sites | 62 / 13 | 62 / 13 |
| 每帧通道数 | 372 | 372 |
| 全文件帧数 | 20060 | 23385 |
| Frame Time | 0.0416667 s | 0.0333333 s |
| fps | 23.9999808000，标称 24 | 30.0000300000，标称 30 |
| 录制时长 | 835.834002 s | 779.4992205 s |
| root X min/max | -93.4666 / 1.88201 | -78.6188 / 5.49938 |
| root Y min/max | 96.3212 / 121.146 | 95.062 / 119.288 |
| root Z min/max | 0.361244 / 52.5347 | -0.416299 / 84.209 |
| root XYZ range | 95.34861, 24.8248, 52.173456 | 84.11818, 24.226, 84.625299 |

全文件逐行校验：motion 行数符合 Frames；每行恰为 372 个值；全部有限；
层级完整闭合，OFFSET 存在；62 个可动关节名称唯一。两者名称/父子结构一致。
全部 channel joints 都是
`Xposition Yposition Zposition Zrotation Xrotation Yrotation`。
旋转单位 degrees，局部旋转矩阵按 `Rz @ Rx @ Ry`，即 intrinsic ZXY；
不得直接当 XYZ 欧拉角，也不能在 retarget 时复制三个数字到 SMPL-X。

**位置通道语义关键点**：非根位置通道基本重复静态 OFFSET。审计按
[Blender 官方 BVH importer](https://raw.githubusercontent.com/blender/blender-addons/main/io_anim_bvh/import_bvh.py)
的通道位置语义，存在 position channel 时使用其值替代对应 OFFSET，缺失轴/
End Site 使用 OFFSET。导入器通过减 rest_head_local 构造 pose translation。
不能再叠加静态 OFFSET，否则重复计算骨长/root 偏移。没有修改源 BVH。

BVH 本身没有单位/坐标元数据。大腿/小腿约 42–45 单位、直立身高约 180 单位，
故 **厘米为强运动学推断**，未来初始单位换算候选为 0.01 m/unit，但须校准。
头相对脚主要沿 +Y，推断 Y-up；左右髋给出 anatomical-left，
`left × up` 给出 forward。按右手旋转 convention 做 FK，未声称 exporter
明确声明 handedness。当前片段朝前方向接近 +Z，非任意将相机方向当身体前方。
完整 joint names、父结构、offset、channel order 见每个 participant 的
`hierarchy.csv` / `bvh_audit.json`，均保存所有 75 个节点。

主要父结构：Hips → Spine → Spine1 → Spine2 → Neck → Head；
Spine2 → Left/RightShoulder → Arm → ForeArm → Hand；
Hips → Left/RightUpLeg → Leg → Foot → ToeBase。
Hand 各有 Thumb/Index/Middle/Ring/Pinky 1–4 链和 End Sites。

## 片段选择及证据

每个文件用 10 秒窗口、5 秒步长检查交替性、周期能量、双手与骨盆稳定性及
脚步连续性；没有使用不可靠的 workout 开始时间推断 posture/RPM。
阈值以 native units 定义：两足高度相关系数 < -0.65，周期谱能量集中度 > .65，
双足高度幅度 > 8，双手相对骨盆波动范数 < 5，骨盆波动 < 3，单帧脚位移 < 15。
这是 P0 筛选启发式，不是完整临床 mocap 质量认证。完整排序在 clip_windows.csv。

| 证据 | participant 1（主选） | participant 2（备选） |
|---|---:|---:|
| 0-based 半开 frame range | [7560,7800) | [5700,6000) |
| segment frames | 240 | 300 |
| 时间 | 315.000252–325.000260 s | 189.999810–199.999800 s |
| 实测 cadence | 约 72 RPM | 约 78.26 RPM |
| 左右踝高度相关系数 | -0.99830 | -0.99387 |
| 周期谱能量集中度 | 0.99493 | 0.99544 |
| 双足高度振幅至少 | 35.386 | 34.339 |
| 双手相对骨盆 std 范数最大 | 0.6573 | 0.9884 |
| 骨盆 std 范数 | 0.5148 | 0.8318 |
| 脚单帧位移最大 | 6.0825 | 5.4386 |
| 左/右肘角 | 140.98° / 125.51° | 171.70° / 172.90° |
| 躯干相对竖直倾角 | 40.16° | 31.20° |
| 左/右手前方距离均值 | 63.02 / 59.37 | 53.25 / 50.07 |

两段双手全帧都在人体前方；踝轨迹闭合近椭圆，10 秒约 12 周期，足迹持续
反相；主要骨段长度 CV < 1e-14，没有选中窗口内的可见跳变/丢帧/数值损坏。
已人工查看两张 `kinematic_diagnostic.png`：周期、相位、双手稳定性符合骑行。
此为原生 BVH 的检查结果，不是 SMPL-X motion 已验证。

SELECTED_PARTICIPANT=1_RB_M42_20230719_PK。
SELECTED_BIKE_TYPE=Road Bike。
SELECTED_POSTURE=COMFORTABLE_LIKE（肘角、躯干倾角的推断，无官方片段标签）。
优先主选是因为手/骨盆更稳定，双踝轨迹更一致，具备多周期数据。participant 2
为 STRAIGHT_ARMS_LIKE 备选，双肘接近伸直。

pelvis 的坐姿稳定性有证据，但数据无鞍座锚点或自行车模型，因此**不能直接
验证 pelvis-saddle 距离或接触**。保持 seated-compatible 推断，不伪造鞍座位置。

## SMPL-X mapping 设计与下一阶段

`outputs/dedicated_cycling_mocap_audit/BVH_TO_SMPLX_JOINT_MAPPING.csv`
包含 22 个主要关节的对应，全部在两个 BVH 中存在；映射可行，但非直接角度复制。
注意 BVH LeftShoulder 对应 SMPL-X left_collar，LeftArm 才对应 left_shoulder；
Foot 是 ankle 位置，ToeBase 对应 SMPL-X foot。右侧同理。
手指额外 4 节不直接一一复制给 SMPL-X 的 3 节；jaw/eyes 无观测，后续需中性
值/先验。这些都属于下一阶段，不在本轮拟合。

下一阶段仅建议：

1. BVH → 原始局部 joint rotations + FK joint positions，保留原 fps 与帧索引。
2. 以初始站姿/骨长检查厘/米尺度、rest frame 和轴 convention；局部动画位置
   不可复制成 SMPL-X 的可变骨长。
3. SMPL-X pose fitting/retarget：保持 SMPL-X 模型拓扑，拟合 pelvis/root、身体
   主要关节及时间连续性，评估手脚轨迹、相位和接触残差。必要时明确形状拟合范围。
4. 生成 SMPL-X vertices，再计算固定 `R @ vertices` 的 DenseKP 138，通过既有
   renderer/projection/visibility 流程；绝不以 BVH mesh 替代 SMPL-X。

本轮实际 stopping point：BVH audit + mapping design，未执行第 2–4 步。

## 可复现工具

`tools/audit_cyclist_bvh.py`：完整层级/通道校验、只读 FK；
`tools/find_cyclist_clip.py`：窗口排序；
`tools/review_cyclist_audit.py`：mapping CSV、层级表及运动学图。
审计产物全部在 `outputs/dedicated_cycling_mocap_audit/`。
