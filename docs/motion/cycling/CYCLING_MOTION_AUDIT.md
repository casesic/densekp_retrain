# Phase 12 — Cycling Motion Asset Audit

审计日期：2026-09-06。只读检查 motion/metadata；未修改 renderer、appearance、
DenseKP、相机、几何或训练代码，未渲染，未下载 motion archives 或自行车模型。

## Baseline

初始提交 `1ad425978882ac96fbdaacec6fa15d1e21ec2a1b`，语义
`synthetic-render-baseline-before-cycling`，作者使用用户提供的 Git 身份。
原目录没有有效 Git 历史，本轮初始化仓库。提交仅含源代码、配置与小型文档；
`.gitignore` 排除 outputs、third_party、模型数组、图片、HDRI、archives、凭据文件。
旧代码的空白格式未做无关清理。冻结依据见 `SYNTHETIC_RENDER_BASELINE.md`。

## 本地范围及结果

扫描 `/home/yy` 与 `/mnt/hdd/yuyue/home/yuyue` 的可访问文件，包括忽略目录中的
资产；排除环境包、缓存、Git、agent 状态和已生成 outputs。搜索错误单独保存，
因此“未找到”不是对不可访问目录或任意压缩包内容的绝对不存在证明。
机器可读结果：`outputs/audits/cycling_motion_audit/local_motion_audit.json`；
文件清单和错误分别为 `local_inventory.txt`、`search_errors.txt`。
NPZ 通过实际 key/metadata 检查，不以文件名给动作分类，不启用 pickle。

确认的本地原始 motion：

| 文件 | motion_info 标签 | 来源路径末尾 | 帧数 | 输出 fps | 兼容性 |
|---|---|---|---:|---:|---|
| BEDLAM tiny/rp_aaron_posed_009/1095/motion_seq.npz | stretch | KIT/883/wipe_arm_dabbing01_stageii.npz | 192 | 30 | poses 为 165 维 SMPL-X |
| third_party/bedlam_render/blender/smplx_anim_to_alembic/smplx_animation.npz | turn | KIT/314/bend_left01_stageii.npz | 126 | 30 | poses 为 165 维 SMPL-X |

两者均非明确 bicycle motion，不作为 cycling 候选。没有选择动作，也没有仅凭
坐姿推断 pedaling。已检查的本地资产中没有确认可用的自行车模型。

## motion_info：能证明什么，不能证明什么

[官方 BEDLAM preprocessing](https://github.com/pixelite1201/BEDLAM/blob/master/data_processing/df_full_body.py)
读取整个 `smplx_param_orig['motion_info']`，直接 append 到输出数组，未解析六个
元素。[官方说明](https://github.com/pixelite1201/BEDLAM/blob/master/data_processing/ReadMe.md)
只称其包含 AMASS 信息和 BABEL 动作标签，没有定义全部槽位。审阅本地
bedlam_render/bedlam2_render 也未找到生成这个六元素数组的写入代码。

以下同时对照两个不同 motion；不能把推断当官方 schema：

| 槽位 | tiny / 示例值 | 审计结论 |
|---|---|---|
| 0 | stretch / turn | 动作标签文本；官方说明支持其与 BABEL 有关，但未证明是 raw_label、proc_label 还是 act_cat |
| 1 | 55 / 0 | 未命名数值字段，语义 UNKNOWN；不能称为 BABEL SID、类别 ID 或 animation ID |
| 2 | /ps/project/datasets/AMASS/.../smplx_neutral/KIT/...stageii.npz | 明确源文件路径；包含 AMASS collection KIT；是发布者原机器路径，不表示本地存在 |
| 3 | 100.0 / 100.0 | 疑似原始采样率，缺少生成代码证实；不能覆盖实际 mocap_frame_rate=30 |
| 4 | 0 / 0 | 疑似源区间起点，单位及端点约定未确认 |
| 5 | 575 / 377（带换行） | 疑似源区间终点，单位及闭开区间未确认；不是输出帧数 |

结论：可提取标签和来源路径；**六槽位完整语义仍未恢复**。后续 join 仅使用
已验证路径和正式 BABEL `feat_p`；不要根据槽位 1 或臆测的起止时间裁剪。
BABEL 官方 schema 本身有 `babel_sid`、`feat_p`、`raw_label`、`proc_label`、
`act_cat` 和以秒为单位的 `start_t/end_t`，这不等于 BEDLAM 六槽位 schema。
参见 [BABEL 数据格式](https://babel.is.tue.mpg.de/data.html)。

## BABEL 标签和 BEDLAM 关联

[官方公开 raw-label 表](https://babel.is.tue.mpg.de/actionsrawlabels.html)
的 `ride` 类包含明确自行车标签，以及 motorcycle 和泛化标签。
只接纳 `ride a bike` 作为明确 bicycle 标签；`ride motorcycle` 排除，
`ride`、`step ride` 留待人工消歧。`bicycle arms` 是手臂动作，
`back pedal/backpedal` 属于后退相关动作，不能按子串误选为骑行。
未在这张表中确认独立的 cycling/biking 标签。

公开表不提供 sequence ID、AMASS path、gender、frames、fps。官方 GitHub
提供 top-120 action-label 分类表，但它不是全部类别，也不是 sequence annotation。
本地未找到可建立路径关联的 BABEL annotation。
[BABEL annotation 下载入口](https://babel.is.tue.mpg.de/download.php)
当前重定向登录；没有尝试绕过登录或读取用户凭据。

BEDLAM2 的 `stats/motion_stats_training.csv` 提供 body/motion ID 和空间统计；
`config/motion2outfit_training.json` 提供服装/动作关联，均不能证明 bicycle 语义。
本地两段已读 motion 没有 cycling 标签。**尚未建立
BABEL → AMASS → BEDLAM subject/animation 的对应关系**，不能宣称整个 BEDLAM
没有骑行动作，也不能虚构 subject/animation ID。

`outputs/audits/cycling_motion_audit/cycling_candidates.csv` 保留要求的全部表头，当前
0 个已定位 source sequence。标签存在数量不等于 motion candidate 数量。

## 最小获取方案

立即所需是用户在 BABEL 官方站登录后获取轻量 annotation release，优先
train/val；检索不足再检查 extra annotations，保留原 split。无需提供密码。
该步骤只取 JSON，不取 AMASS motion。记录 bicycle 的 SID、feat_p、时间区间后：

1. **A / BEDLAM 优先**：先用已有或官方可提供的 motion_info 索引核对路径。
   若匹配，申请或获取对应单个 subject/animation 的 motion_seq.npz；目前未确认
   官网支持单文件下载，因此不能承诺具体下载量或对应动作已存在。
2. **B / AMASS 次选**：若 BEDLAM 不匹配，只获取候选所属的单个 source collection
   的 SMPL-X 版本。当前尚无 feat_p，不能负责任地指定 KIT、CMU 或包大小。
   先确认文件格式/大小和 SMPL-X 版本，不能将 SMPL-H 直接视为同拓扑 SMPL-X。
3. **C / 其他官方 mocap**：作为备选，仅接受明确 cycling 标签、周期、fps、
   许可和转换依据；当前没有证据充分且已定位的更小候选，不推荐盲目下载。

[BABEL 许可](https://babel.is.tue.mpg.de/license.html)、
[AMASS 许可](https://amass.is.tue.mpg.de/license.html)、
[BEDLAM 许可](https://bedlam.is.tue.mpg.de/license.html)均有非商业研究用途限制，
不是可任意再分发的公共领域资产。下载量最小的当前动作是获取 annotation，
而非完整 AMASS/BEDLAM 包。下一步需要用户登录获取元数据。

## Bicycle asset 设计（未实现、未下载）

V1 分层 rigid objects 足够，不需要蒙皮 rig。单位米，明确车前方向、轮轴和
曲柄轴；每个旋转件原点必须在真实轴心。至少 frame、fork、handlebar、
front/rear wheel、saddle、crank、左右 pedal，以及握把/鞍座/脚踏接触锚点。

```text
bike_root
├── frame
│   └── saddle
├── steering_root
│   ├── fork
│   ├── handlebar
│   └── wheel_front
├── wheel_rear
└── crank
    ├── pedal_left
    └── pedal_right
```

steering_root 绕头管轴，前轮绕自己的轮轴，后轮绕后轮轴；曲柄绕中轴，左右
脚踏相位相差 π。脚踏在曲柄下的位置随圆周运动，其自身姿态需独立控制，避免
脚踏板机械地随曲柄翻转。轮速由车速/轮半径决定；曲柄频率来自 motion，
通过齿比/滑行状态关联，不能假定轮和曲柄转速相同。

## Human ↔ bicycle 约束及 DenseKP policy

A. global alignment：统一坐标/单位，求 body-bike 刚性相对姿态；记录变换，
不能把未知尺度误差藏进 SMPL-X 形状或 DenseKP 定义。
B. hands：左右掌握持锚点分别约束到左右握把，检查位置及握持朝向。
C. pelvis：骨盆参考点经固定人体接触偏移对应 saddle；避免将 pelvis joint
直接等同于皮肤坐垫接触点。
D. feet：鞋底/足底接触锚点对应 pedal，保持左右归属、脚踏半径及侧向间距。
E. phase：用双足相对骨盆轨迹估计踏频和相位，检查交替周期、手部稳定性及
坐姿，再拟合曲柄 phase；记录接触残差。后续必要时才讨论 IK，本轮没有实现。

DenseKP 始终为 `densekp_3d = R @ smplx_body_vertices`，这里 R 是固定 138 点
回归算子，不是新增自行车姿态旋转。相机/世界变换按既有约定处理。
自行车仅是 occluder/scene geometry；点被车遮挡时 `valid=1, visible=0`。
保留两个字段，不改定义、loss 或点数。新增动作的 clothing 也需要对应动作
的兼容模拟资产，不能把当前 1095 clothing 动画直接套到 cycling。

NEXT_ACTION：获取最小 BABEL cycling sequence metadata，确认 source，再请求
最小对应 SMPL-X motion；当前不进入 Cycling P0 渲染。
