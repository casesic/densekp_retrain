# Phase 12.1 — BABEL Cycling Candidate Search

2026-09-06。入口：`tools/search_babel_cycling.py`。只解析 annotation、检查已有
motion metadata，不下载 motion、自行车，不渲染，不修改 renderer 或 DenseKP。
本报告替代 Phase 12 中“完整 BABEL metadata 不可用”的旧状态。

## 实际 schema 与完整性

输入 `/home/yy/datasets/BABEL/babel_v1.0_release/` 的五个 JSON 均已解析。
每份文件顶层为以 SID 字符串为 key 的字典；条目 `babel_sid` 为整数，与 key
逐项核对相同。`feat_p` 是字符串，`dur` 是秒。五份文件 SHA256 和逐字段类型、
标签数量、null 计数见 `outputs/cycling_motion_audit/babel_cycling_summary.json`。

| split | sequence 记录 | 标签行 | sequence 字段 | frame 字段 |
|---|---:|---:|---|---|
| train | 6615 | 43137 | seq_ann: dict | frame_ann: dict/null |
| val | 2193 | 15744 | seq_ann: dict | frame_ann: dict/null |
| test | 2084 | 14431 | seq_ann: dict | frame_ann: dict/null |
| extra_train | 7921 | 13152 | seq_anns: list | frame_anns: list/null；47 条非空 |
| extra_val | 2636 | 4944 | seq_anns: list | frame_anns: list/null；22 条非空 |

annotation group 有 `labels`；标签含 `raw_label`、`proc_label`、`act_cat`、
`seg_id`。act_cat 可为 list 或 null。frame 标签带 start_t/end_t 秒数；
sequence 标签无明确片段边界时，用 [0,dur] 表示**整段标注范围**，不是持续
骑行的几何证据。CSV 保留 timing_basis、group index、babel_lid、seg_id。

test 的 14431 条标签 raw/proc/category 全部为 null；它已被遍历，但不能判定
有无 cycling。因此“0 个已确认 cycling”不代表 test 中不存在这种动作。
合计 split 记录 21449，按 SID 去重 12660，独立 feat_p 也是 12660。
extra 是不同 annotator 的补充信息，不能把跨 split 重复 SID 当新 motion。

## 搜索与计数口径

sequence 和 frame（含 extra 多组）均搜索 raw/proc/category。宽检索覆盖
bike/bicycle/cycling、pedal/peddle、ride/riding 与 motorcycle/horse/scooter，
保留否定证据。匹配标签行数 78，不是独立 motion 数。

排除 backwards/backpedal、bicycle arms、侧向 step ride、明确摩托车/骑马。
来源路径冲突不等于凭文件名证明动作，但必须隔离复核。明确 bicycle 标签
仍保留 HIGH 的**标签置信等级**，另写 exclusion_reason；不能直接放行下载。

- 明确骑车措辞 HIGH：2 条，2 个 SID，均为 BMLrub；可直接采用 HIGH：0。
- 未排除 MEDIUM：1 条，KIT SID 1960，extra_val，sitting on bike，9.43 秒。
  其他 annotator 仅写 squat / Sitting，没有 pedaling 证据；不升级 HIGH。
- 原始宽匹配词法等级为 HIGH 3 / MEDIUM 61 / LOW 14；其中 HIGH 的第三条
  是 bicycle arms（SID 1349），已剔除骑行 HIGH 统计。
- 总排除/隔离 77 行，1 行可保留为 MEDIUM 待查，不等于它已通过姿态检查。

| feat_p 首段 source dataset | 宽匹配标签行数 |
|---|---:|
| KIT | 36 |
| BMLrub | 20 |
| CMU | 7 |
| EyesJapanDataset | 5 |
| Transitionsmocap | 3 |
| ACCAD | 3 |
| TotalCapture | 2 |
| SFU | 1 |
| MPIHDM05 | 1 |

source 从真实 feat_p 的第一段解析，第二段 collection 单独保存；不靠标签猜测。

## Top：仅标注复核顺序，不是 motion 下载推荐

| rank | SID | raw label | feat_p | 标注范围 |
|---|---|---|---|---|
| 1 | 8265 | riding a bike | BMLrub/BioMotionLab_NTroje/rub076/0004_motorcycle_poses.npz | 0–3.97 s |
| 2 | 10670 | Riding a bike | BMLrub/BioMotionLab_NTroje/rub079/0004_motorcycle_poses.npz | 0–3.97 s |

两者标签明确程度、范围和总长相同。8265 排前是因为 10670 的 extra_train
明确写 ride motorcycle，存在更强的跨 annotator 冲突；8265 的 extra_train
写 standing holding arms outstretched。两者路径均命名 motorcycle。
**Rank 1 也是隔离候选，未推荐为真正 cycling motion。**没有为了凑 Top 10
而混入后退、手臂绕圈或摩托车。Top 按 feat_p 去重。

第 4–6 项（交替踏腿、前方稳定双手、坐姿骨盆）全部
NEED_MOTION_INSPECTION；没有 motion 几何时不能验证。获取成本 UNKNOWN，
不编造大小作为排序依据。未使用 fps 推算帧数；annotation 本身不含可靠 fps。

## 最小获取范围

Rank 1 的 AMASS_SOURCE_DATASET=BMLrub，collection=BioMotionLab_NTroje。
TARGET_RELATIVE_PATH（按 BABEL 原样）=
`BMLrub/BioMotionLab_NTroje/rub076/0004_motorcycle_poses.npz`；
相对 collection 内路径为 `rub076/0004_motorcycle_poses.npz`。
SMPL-X 发布版是否改名为 stageii 及目录结构，须核对所获 release 的索引；
不能把 `_poses.npz` 自动宣称为已验证的 SMPL-X 文件。

[AMASS 官方入口](https://amass.is.tue.mpg.de/download.php)目前未登录时重定向
登录页。单 NPZ 官方直取能力未确认，不能断言必须或可以单文件下载；若按
collection 打包获取，预计单位为 **一个 BioMotionLab_NTroje/BMLrub 子数据集
的 SMPL-X 包**，不是整个 AMASS。准确包名、字节数、内部路径尚未验证。
[官方工具仓库](https://github.com/nghorbani/amass)说明其模型兼容路径，但
这不能代替目标 release 格式检查。

本轮 REQUIRED_DOWNLOAD=NONE。应先复核 annotation 自带的官方可视化链接
或官方 source 动作说明以消歧：SID 8265 的 URL 是
`https://babel-renders.s3.eu-central-1.amazonaws.com/008265.mp4`，10670 对应
`010670.mp4`；本轮未请求/下载视频，也未把视频当已检查证据。若证实为
motorcycle，放弃这两段并转向其他官方 cycling 来源，而非下载整个包碰运气。

## BEDLAM overlap 与本地性

与现有 local_motion_audit.json 中的两段 BEDLAM motion_info 比较 collection/
subject/source basename，并容忍 poses/stageii 后缀差异，无匹配。另对本地
bedlam_render、bedlam2_render 和 BEDLAM 文本元数据搜索 rub076、rub079、
0004_motorcycle、wipe_leg_smallcircle03，无命中。BEDLAM2 stats/outfit 映射
未提供完整 AMASS 来源索引。因此 BEDLAM_OVERLAP_FOUND=UNKNOWN（已有两段
不重叠；全库未知），不虚构 subject/animation ID。

在上轮两根目录的可访问 motion 资产清单中没有对应 NPZ；
AMASS_MOTION_LOCAL=NO（受已记录的扫描范围和访问权限限制）。

## 产物

- `outputs/cycling_motion_audit/babel_cycling_candidates.csv`：78 行完整命中及排除理由。
- `outputs/cycling_motion_audit/top_cycling_candidates.csv`：2 行隔离 HIGH 的复核优先级。
- `outputs/cycling_motion_audit/babel_cycling_summary.json`：计数、schema、哈希及 context。

NEXT_ACTION：先消歧 SID 8265/10670；当前没有已确认可用于自行车骑行生成的
HIGH 候选，不进入 geometry、IK、renderer 或 training。
