# A1：种子视角与整体双机 rig 扩展（CPU 预检交付）

日期：2026-10-10。**本批没有通过可渲染候选，因此没有新增正面／斜前视频、ROI、NPZ 或 Dataset 行。**既有[背面单人和混合视频](../../outputs/experiments/seeded_scene_modes_a1_20261010/index.html)只读保留。当前[查看页](../../outputs/experiments/seeded_viewpoints_a1_20261010/index.html)列出四份配方与拒绝结果，不将旧视频冒充新视角成品。机器可读结果见[acceptance.json](../../outputs/experiments/seeded_viewpoints_a1_20261010/acceptance.json)。

新增的可复用接口是 `seed_scene_modes.py --view-classes front,...` 与[版本化覆盖配置](../../configs/seeded_viewpoints_a1_20261010.json)。资产组合仍从原已接受池选择，不重新绑定衣物。资产、布局、相机各用独立子种子；相机范围改变不会改已选角色、原动作起点或布局随机流。配方记录请求视角、相机 x/y/yaw、道路正反方向、过界策略和每候选子种子。同 seed、配置与版本重现同一配方；旧配置缺字段时仍使用固定 rig、`+Y` 路径和原 `enter_valid` 检查。

视角是世界水平面内**身体前向与骨盆根点指向相机光心**的夹角，0°朝向相机，180°背对相机。前向来自已验收的 `fixed_scene_rotation @ global_orient @ 模型+Z`，没有再加一次资产旋转；根点是当前 pelvis-centered 骑行资产的场景原点。分档为 `[0,45)` 正面、`[45,80)` 斜前、`[80,100)` 侧面、`[100,135)` 斜后、`[135,180]` 背面。零长度水平向量拒绝。CPU 预检逐帧记录 rig 中点及 cam2、cam4 各自的角度；类别名本身不构成通过证据。正面／斜前目标要求至少16/31帧**两台同时**处于目标角度，且与物理双边缘窗口连续重合至少0.5秒。

相机通过原 `rig_poses` 整体改变道路平面 x/y 与 yaw，原函数只加一次 1.15m 高度；相机相对 ERP 矩阵与相对物理 DS 镜头矩阵保持不变，0.18m 仍是合成基线。两张 normal ERP 继续共用同向坐标基，cam4 的物理镜头反向只在 DS 域判断中应用一次。标定有效 mask 在相机局部 ERP 坐标下不变，世界点经**新** camera-to-world 逆变换后投影；没有从 RGB 黑色推断有效域。道路方向 `+Y/-Y` 以正速度和时间正向行进；反向时转动骑行者的固定场景朝向，不倒播动作。行人继续由本人的 source-root 拥有位移，其 yaw 与骑行路线解耦，避免双重平移。相机一段内固定，不追踪人物。

冻结 `20261012` 与 `20261013` 两个 master seed，每 seed 各有单人 `front`、混合 `front_oblique` 一份，共4份配方，每份12个候选、48个**唯一**候选。相机工程范围与道路范围见配置；这些范围不是实测相机运动分布。CPU 预检全部31时刻，并继续要求双机各≥8域内、≥5域外 R138、共同域内≥8，至少16/31且连续≥0.5秒；单人还要求首末两机同步完成 `enter_valid` 或 `exit_valid` 的20→5／5→20域外点与有符号边界距离翻转。保守按人 square-crop 并集不是实际 visible-instance 框；其全图边界与7M像素预算仍是生产前硬约束。

| 模式 / seed / 资产 | 48候选中的本份12个结果 | 已能证明 / 尚不能证明 |
|---|---|---|
| 单人正面 / 20261012 / M01 | 6个双边缘不足；6个首末双路过界不足 | 最佳候选两台同时正面31/31、双边缘22/31；端点条件未通过，不能称穿越。 |
| 单人正面 / 20261013 / F02 | 2个双边缘不足；10个首末双路过界不足 | 最佳候选两台同时正面31/31、双边缘31/31；端点仍不符合两机同步20→5或5→20。 |
| 混合斜前 / 20261012 / M01＋M04 | 12个完整保守 square-crop 支持包络越出 ERP | 不能把保守包络失败说成真实原生框已越界；因无栅格输出，也不能确认实际斜前角度和遮挡。 |
| 混合斜前 / 20261013 / F02＋M04 | 12个人物根点间距低于既有0.9m门槛 | 未进入视角／遮挡判断；不能以人穿人获得交会。 |

每份最终预检耗时分别6.33、7.01、6.43、1.77秒；这是对应有效代码版本的 CPU 几何阶段，不是总开发墙钟。中途发现并修正三处接入错误：首次正面预检缺少前向缓存、首次混合误让行人随骑行者反向旋转、随后把“目标完全在物理域外”误作硬错误。对应失败目录保留；针对性重算复用原12个参数，没有抽新候选。最终代码将域外视为合法空观测，仍对非有限标签、ROI 不足、矩阵／时间／身份错配保留硬拒绝；没有借修错放松视角、过界、碰撞或ROI门槛。

[正面候选7的逐时刻CPU曲线](../../outputs/experiments/seeded_viewpoints_a1_20261010/cpu_front_candidate7.svg)展示两台的实际身体观察角与域外R138数；这是被拒绝的投影诊断，没有 RGB／视频，不能当作新场景样本。

定向 CPU unittest **25项通过**，涵盖视角分档／退化、种子重现与相机随机流隔离、旧配方、正反向时间、过界策略、两机相对矩阵与局部投影。`py_compile`通过。新收尾入口会在首次 Dataset item 读取前写 `readback_plan.json` 并将两模式最多8行作为硬抽查上限；视频顶部可标实际两台角度，但本批没有运行该生产／回读路径，不能据此声称实际 Dataset 验收。历史混合段曾读10行的事实不改写。

资源快照：GPU0 `GPU-d481acc7-106f-29e5-1298-8c0d398261b2` 18MiB使用，GPU1 `GPU-6f046d56-acbe-d49b-774a-9d463e768cac` 407MiB使用，均0%利用率；另有他人 Blender 进程，未终止或使用。可用内存约140GiB、磁盘约12TiB；本批输出约12MiB。Blender 启动0次、相机渲染0次、Dataset item回读0次，45分钟渲染预算消耗0。没有推理、训练或权重修改。

复用命令（新目录；`--view-classes` 一项对应一个 episode，多个类别按 seed 确定性打乱）：

```bash
PYTHONPATH=src:. python tools/experiments/seed_scene_modes.py \
  --config configs/seeded_viewpoints_a1_20261010.json \
  --mode single_cyclist_boundary_pass --master-seed 20261012 \
  --episode-count 1 --view-classes front --output /tmp/a1_view_recipe_new
PYTHONPATH=src:. python tools/experiments/plan_road_pass.py \
  --config /tmp/a1_view_recipe_new/single_cyclist_boundary_pass_front_s20261012_e00/config.json \
  --output /tmp/a1_view_preflight_new/plan.json --geometry-preflight
```

第二条命令前需建立其输出父目录。`mixed_walk_cycle_crossing` 与 `front_oblique` 走同一 planner；只有 `geometry_preflight.json` 为 `passed` 才能把该目录显式交给 `run_continuous_roi.py`。本批四份均为 `rejected`。下一轮若要得到正面视频，需**先作为新设计批次**重新确定道路端点／速度与相机／行人范围，再另冻 seed 和候选预算；不能把本批视角覆盖高的沿边候选伪称过界，也不能在本批继续加候选。来源划分、随机框增强、队伍／路口与8段／200步方案仍待评审。
