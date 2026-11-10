# 城市端部 T 形路口：三人转弯与三组双机的有界预检

日期：2026-10-11。结果：**人物道路轨迹通过；相机候选未通过冻结门槛，因此未启动正式渲染**。本批没有新的路口视频、ROI、NPZ 或 Dataset 回读。此结论只针对本批48个冻结机位，不能推断该路口不存在任何可行机位。[几何查看页](../../outputs/experiments/t_junction_multirig_a1_20261011/index.html)和[执行决定](../../outputs/experiments/t_junction_multirig_a1_20261011/execution_decision.json)保留原始证据。

用户已接受[五人修正版正面视频](../../outputs/experiments/front_rig_size_repair_a1_20261011/production/rigs/rig_front_revised/pack/dual_roi.mp4)的**已展示观感**；其中 cam4 主目标原生框高度中位数260→438个完整 ERP 像素。旧正面过小、修正版部分人物域外及双路边缘0/31不改变。这个接受不转移到尚未渲染的路口机位。

## 场景和轨迹

只读检查实际 `urban_street_assets.blend`。主路 `Road_surface_7m` 宽7米，至 Y=80；正端 `Cross_street.001` 在 Y=80..88、X=−80..80，二者路面同为 Z≈0。负端也有横路；选择正端，因为现有三套骑行动作沿道路正时间前进，转向此处无需倒放动作或突然逆转人物前向。未修改原 `.blend`，不需要路面patch。[162处端部射线与对象记录](../../outputs/experiments/t_junction_multirig_a1_20261011/junction_inspection.json)可查实际路面、人行道与建筑，**不是**全连续障碍碰撞证明。

按[冻结的12套种子轨迹候选](../../outputs/experiments/t_junction_multirig_a1_20261011/frozen_trajectory_candidates.json)顺序，首套通过。三人只读复用已接受城市三人缓存：[输入谱系及文件引用](../../outputs/experiments/t_junction_multirig_a1_20261011/source_references.json)列出p1=M01／APD6526／M02_simple，p2=M02／participant_1 comfortable／M01_simple，p3=F02／participant_2 straight／F02_simple；头发分别为M01／F02／F02 groom。三人各保留原 source frame、整数时间、shape、身体／衣物／头矩阵和31个10Hz合成时刻，未重新组合或拟合。全部rig同一`scene_episode_id=city_t_junction_3p_s20261040`，同一共享来源组。

p1 从主路连续圆弧转入 +X 横路；p2 沿横路 +X，p3 从另一侧沿 −X。p1从约(0,77.96)到(3.86,81.93)m，实际转向接近但不到完整90°，故称**自然转弯窗口**，不称完成90°转弯或真实骑车压弯。三人的3秒根位移分别5.53／4.70／4.26米。圆弧与直线位置和切线连续；31时刻采样的最大相邻方向变化2.92°，p1原动作身体前向与当帧道路切线最大夹角24.94°。三人保留各自固定支撑高度，场景矩阵作用于身体、衣物、groom、R138，原动作不变速或循环。[轨迹检查](../../outputs/experiments/t_junction_multirig_a1_20261011/trajectory_checks.json)和[俯视图](../../outputs/experiments/t_junction_multirig_a1_20261011/junction_topdown.svg)记录实际路径。

道路支持不是包住路口的大矩形：每帧所有body／cloth顶点按主路与横路的**联合区域**检查，路肩留4厘米平面余量；三人所有人物对的离散XY包络最小分离3.06米。独立Blender回读在31×3个时刻、1,047个近地身体射线上命中原生`Road_surface_7m`或`Cross_street.001`，实测身体离路面间隙0.203–0.347米，未命中人行道或建筑。[原路面复核](../../outputs/experiments/t_junction_multirig_a1_20261011/native_road_check.json)是离散近地采样；它不是三角形连续碰撞证书。为使公共renderer的原生路面检查识别横路，新配方只给这三人的`support_checks.road_object_names`加入`Cross_street.001`，不改旧行人或骑行默认。

[首／中／末×三人独立R138与原时间回核](../../outputs/experiments/t_junction_multirig_a1_20261011/independent_R138_time_check.json)以项目原回归器对缓存body重算，世界点最大绝对差约1.5e−14米；对应head矩阵有限、每人的asset/source帧映射未混用。这是CPU数值检查，不代替真实渲染标签验收。

## 冻结机位与拒绝

[配置](../../configs/t_junction_multirig_a1_20261011.json)预先冻结相机master seed20261041、front/back/side各16份，总48份，不追加候选。每组仍为0.18米**合成**基线的原绑定双机，物理mask、ERP和cam4变换未动。目标固定p1；三组各需双机同类严格视角、共同physical R138≥8连续6帧，且检查较小一路的投影身高；至少一组还需连续6帧双路物理边缘。正面投影身高筛选线280像素，仅为前次已接受cam4原生框中位438像素约64%的本批工程代理，不是所有场景通用验收线，更不是真实可见框。

| 组别 | 冻结候选 | 最长双机同类＋共同窗口 | 最长双路边缘 | 最大小一路R138投影身高中位 | 本批尺寸线 |
| --- | ---: | ---: | ---: | ---: | ---: |
| back | 16 | 24帧 | 0帧 | 231px | 180px |
| front | 16 | 29帧 | 0帧 | 187px | 280px |
| side | 16 | 4帧 | 0帧 | 352px | 180px |

完整[候选摘要](../../outputs/experiments/t_junction_multirig_a1_20261011/candidate_summary.json)与逐帧[back](../../outputs/experiments/t_junction_multirig_a1_20261011/light_back_checks.json)、[front](../../outputs/experiments/t_junction_multirig_a1_20261011/light_front_checks.json)、[side](../../outputs/experiments/t_junction_multirig_a1_20261011/light_side_checks.json)日志保留48个原attempt编号及两机独立角度、域内外点、共同点和R138代理尺度。各组轻量筛查通过数均为0，因此未消耗每组最多3份的精确body／cloth／active-groom ROI检查配额，也**没有**冻结可生产的rig列表。不能把背面的24帧共同观测称作双边缘，也不能把侧面的4帧称作本批6帧窗口。实际原生框、ROI和人眼视角均未产生。

主要规划缺口可以定位：这批静态候选把整体yaw设为朝目标中段的方位±4°，未求解原物理mask的双侧边缘朝向；背面虽然共同点多，却始终没有双侧同步截断。正面候选距转弯者偏远，较小一路的尺度只有本批尺寸线以下；侧面人物转向时角度变化快，限定的东南侧位置只给出最多4帧窗口。这说明**本批候选构造不够好**，不能声称路口硬几何不可能。后续需要在新授权和新候选预算下，把physical边界求解纳入yaw构造、在相机与三人全段净空约束下布置更近的正面rig，并检查另一侧可放置的侧向机位；不能回头改写本批48份候选或放宽本批门槛。

## 实施、成本和剩余范围

新增公共[圆弧转弯轨迹](/mnt/data1/yy/home/yy/Projects/densekp_retrain/src/densekp_retrain/data/t_junction.py)、[三人准备入口](/mnt/data1/yy/home/yy/Projects/densekp_retrain/tools/experiments/prepare_t_junction_a1.py)、[有界相机规划入口](/mnt/data1/yy/home/yy/Projects/densekp_retrain/tools/experiments/plan_t_junction_rigs_a1.py)和只读Blender地面检查；已有多rig逐帧renderer无需复制或改人数轴。旧相机预检器只加入可选的路口目标值，旧back边缘及side横截面默认门槛维持。16项定向CPU测试（含旧城市多rig契约）通过。可重放步骤：`prepare_t_junction_a1.py freeze` → `prepare`；`plan_t_junction_rigs_a1.py freeze` → `light`；只有三组通过后才能做`exact`、原生相机净空检查、冻结rig列表，再交给已有`run_multirig_roi_a1.py --rig-list-manifest`。**本批按拒绝结果停在light阶段**，尚未检查任何最终相机中心的原生障碍净空。

[独立目录的同seed重放](../../outputs/experiments/t_junction_multirig_a1_20261011/seed_replay_check.json)确认12套人物轨迹候选、三组各16个相机参数逐值一致，所选三人scene矩阵与世界R138数组逐值一致；此次回放未再次投影或渲染。

从仓库根目录在独立目录重放CPU部分的命令如下；入口采用只创建新文件的输出策略，不能覆盖本批证据：

```bash
PYTHONPATH=src python tools/experiments/prepare_t_junction_a1.py freeze --output-dir /tmp/t_junction_replay_a1
PYTHONPATH=src python tools/experiments/prepare_t_junction_a1.py prepare --output-dir /tmp/t_junction_replay_a1
PYTHONPATH=src python tools/experiments/plan_t_junction_rigs_a1.py freeze --output-dir /tmp/t_junction_replay_a1 --source-prepared /tmp/t_junction_replay_a1/prepared_scene
PYTHONPATH=src python tools/experiments/plan_t_junction_rigs_a1.py light --output-dir /tmp/t_junction_replay_a1 --source-prepared /tmp/t_junction_replay_a1/prepared_scene
```

计算前GPU0 `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`空闲约24,144MiB；系统可用内存约139GiB，工作盘余量约12TB。实际轨迹求值1.32秒、48机位轻量投影约0.97秒、Blender原场景端部打开/射线约1.94秒、原路面支持回读约2.72秒，加上定向测试约0.42秒，均远低于20分钟场景与CPU限额。未做Cycles相机渲染，60分钟生产额度未消耗；Blender静态检查打开场景两次，**不是**一次正式场景装配。计划186次、558槽；实际相机调用0、合法行0、Dataset回读0、视频0。任务目录目前约1.9MiB；没有本任务EXR，也未清理旧产物。不存在可报告的实际原生框身高、训练输入或渲染吞吐。

旧正面修正版观看接受、历史三rig／4–5人和旧失败证据保持。本任务只证明路口原生道路与三人连续转向轨迹可用；**路口三rig生产尚未通过**。A2资产、外部pano-stitch、TODO及正式来源划分没有修改。下一步相机几何需单独冻结候选和预算后再执行，不能把这份仅CPU通过的准备数据加入训练或独立验证。
