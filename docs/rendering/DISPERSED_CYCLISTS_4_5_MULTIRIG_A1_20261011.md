# 城市分散骑行人群：4／5人双 rig ROI 数据

日期：2026-10-11。本批为 `integration_only/shared_source_diagnostic`。观看入口：[并列视频页](../../outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011/index.html)。灰色区域未渲染，不代表真实背景；黄色线是物理有效域边界。新 M03／M04 组合的人工观感仍待观看。旧三人双 rig 视频已获用户观感接受，但其 23/31 双边缘、8帧斜前窗口与邻人遮挡恢复未达标的技术记录没有改变。

## 冻结输入、组件与来源

前3人精确只读复用城市三人双 rig 的组合缓存、原帧起点和道路世界轨迹。新制两套标准 SMPL-X／simple 衣物／登记 groom 组合：p4 是 M03＋participant_1 alternate＋M03_simple＋M03_groom，p5 是 M04＋participant_2 straight＋M04_simple＋M04_groom。A2 `composition.plan/build` 原接口数值通过；未修改其 registry、参数或绑定算法，未使用锁定 KIT11 的 BEDLAM 衣物。两套成品不因数值通过而继承人工外观接受。p4／p5 组合缓存跨本批两个 episode 复用，第4人没有在5人段重新 build。

每段31个合成10 Hz时刻，首末3秒；每人从各自原整数 timeline 选择严格递增的asset/source帧。p1/p2/p3起点0/4/19；新增p4/p5起点分别18/7。没有循环、倒播、插值、静态姿势复制或统一伪造30 Hz。实际是5个不同登记身体家族、4个 motion ID、3个原动作来源组：APD、participant_1、participant_2。p1两段关联、p2两段关联；M01/M04与M02/M03的保守外观父关系仍保留。两个 episode 的所有人物、rig及相邻帧必须一起作来源分组，不是正式train/val。见[来源图](../../outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011/source_provenance.json)。

[配置](../../configs/dispersed_cyclists_4_5_multirig_a1_20261011.json)固定4人 layout seed20261030、5人 seed20261031，斜前rig候选seed20261032/33；人数由任务显式给定。旧三人坐标和`rig_edge`原位姿是锚点，第4／5人从结构化道路纵向／横向单元经稳定子seed洗牌与小扰动。12个布局候选先全部冻结，再按序投影；每段第0个通过。每段12个斜前候选同样先冻结，第0个沿用旧合格机位即通过，备用候选未投影。改变camera seed会改变备用相机计划，**本批实际选中位姿仍是继承机位**，不能称实际相机由新seed改了位置。前3人新旧31帧世界R138和场景矩阵逐值相同。见[预检摘要](../../outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011/preflight_summary.json)及各段冻结候选。

两个布局沿道路Y正方向分散：4人起点Y约0.73、2.07、3.54、4.91米；5人另有Y约5.49、7.05米，速度约0.44–0.53米/秒。各人场景拥有一次固定朝向、支撑高度与恒速道路位移；身体、衣物、R138同矩阵，头发复合本人的head矩阵。预检检查全部31时刻、4人6对／5人10对间距、道路范围、两组相机人体／衣物／精确活动groom、逐人完整square crop和联合ROI≤700万像素。该包络与顶点距离是工程近似，不是连续三角形无穿透证明。遮挡15/15事件与严格端点过界只作诊断；自然无框不阻止后续帧。两组rig内相对外参与0.18m**合成**基线固定。

## 4人段已交付结果

31时刻在一个Blender进程中加载城市与4人一次，正常路径31次人物几何更新，每帧按`rig_edge(cam2,cam4)→rig_front_oblique(cam2,cam4)`完成，共124次ROI相机调用、一个常驻同步CPU消费者。没有低清背景、全ERP RGB或DS往返；逻辑ERP5760×2880、64 samples CUDA和原城市场景灯光／材质保持。两rig共享同一时刻R138世界状态，独立投影和局部ROI；全帧相机与预检ROI边界一致。

4人共496个人物／相机／时刻槽位，415条合法原生框行；edge 242行／6不可用，斜前173行／75不可用。`record_unavailable_v1`保留无框与域外，不造黑色输入或上一帧框。冻结24个Dataset槽位实际读回20行，其余不替换；局部RGB仿射、138点完整ERP→256-square、可靠`in_view & inside_full_crop`监督、`_trans`和`img_size`核对通过。4人各在至少一台相机连续31时刻具有合法框和≥8个可见R138点。主目标edge双路物理边缘23/31、共同物理点≥8为31/31；斜前双边缘0/31、共同物理点≥8为24/31，目标双机斜前最长8帧。斜前rig不计入双边缘配额。实际视频见[4人edge](../../outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011/city_4p/production/rigs/rig_edge/pack/dual_roi.mp4)与[4人斜前](../../outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011/city_4p/production/rigs/rig_front_oblique/pack/dual_roi.mp4)。

4人渲染与逐帧消费完成后，第一次CPU打包因新配方组件键为`components`而旧打包器只读`selection`失败；失败时创建的空`pack_failed_attempt1`目录保留。打包器现对两个明确键做一致性校验，**只重做CPU收尾，未重渲染帧**。该段最初还遗漏了持久化邻人IndexOB＋range证据开关，且成功缓冲已按compact生命周期清理，所以真实15/15邻人恢复在4人段为**不可评价**；不以总遮挡或CPU投影代替。5人段启动前已单列元数据修正并启用这条原有证据链，几何、ROI与采样未改。

4人Blender wall 1367.25秒；装配62.75秒，首时刻54.30秒，后30时刻均值41.50秒。edge 62次ROI相机调用累计866.98秒，斜前62次229.93秒；逐帧同步CPU消费累计72.40秒。首次失败收尾2.93秒，修正后两个rig的pack／读回／视频分别16.48／15.58秒，计算口径合计1402.24秒；含定位代码错误与验证的交付墙钟另计，不把它说成纯渲染。GPU0 UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`，采样显存峰4691MiB，Blender峰RSS约15.13GiB，任务盘峰约421MB、临时峰约299MB，compact后生产目录约285MiB。详见[成本与故障](../../outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011/city_4p/production/task_wall_corrected.json)和[全帧验收](../../outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011/city_4p/production/acceptance.json)。

## 5人段已交付结果

5人沿用前4人的组合与原时间映射，加入M04／participant_2 straight／M04_simple／M04_groom；五种身体家族、四个motion ID和三个原录制来源组都由实际缓存与首／中／末顶点变化核对。31时刻在一个Blender进程加载城市与5人一次，正常路径31次五人几何更新，依序对两rig各两机共124次ROI调用，一个常驻同步CPU消费者。两rig合计**620**个观测槽位（31×2rig×2相机×5人）；edge 310槽位中230条合法行／80不可用，斜前310槽位中204条合法行／106不可用，共434条合法行。冻结24个槽位实际Dataset读回17条，其余无框不替换；全帧ROI／mask／实例ID／时间／来源、138点仿射、可靠监督、`_trans`、全ERP `img_size`及长期路径通过。[5人全帧验收](../../outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011/city_5p/production/acceptance.json)。

五人均在至少一组rig／一台相机连续31时刻具有合法框和≥8个实际visible R138。第4、5人在edge组的最长连续合格段仅1／7时刻，在斜前组则各为31；这符合分散人群与自然入／出有效域的设计，**不能把edge组说成五人全程可见**。主目标edge双路物理边缘23/31、共同物理点≥8为31/31；斜前双边缘0/31、共同物理点达标24/31，两机斜前共同窗口最长8帧。原严格端点过界、各人的无框、共同crop和实际实例遮挡均与物理边缘分开存档。5人段启用了逐点邻人IndexOB＋range证据；两rig的cam2／cam4均未满足原15／15个百分点、每阶段连续3时刻的完整加重—恢复事件，不能以某单帧较高遮挡比例宣称恢复。[5人edge视频](../../outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011/city_5p/production/rigs/rig_edge/pack/dual_roi.mp4)、[5人斜前视频](../../outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011/city_5p/production/rigs/rig_front_oblique/pack/dual_roi.mp4)。抽看视频中段，两套新增组合确实以不同衣型和持续姿态进入道路画面；人工外观接受仍待用户观看。

按p1…p5次序，edge组合法行分别为62／62／57／5／44，斜前组为56／55／31／31／31；分母每人每rig为62槽位。主目标edge首末域外R138点为cam2的22→3、cam4的18→0，旧严格端点过界所需两机起点各≥20未同时成立，因此**边缘覆盖通过、严格端点过界未通过**。四份视频均由31帧、10 Hz编码，展示时长3.1秒；最后0.1秒仅是末帧展示，场景真实首末跨度仍为3秒。

5人Blender wall 1664.83秒；装配78.81秒，首时刻63.22秒，后30时刻均值50.56秒（47.06–57.55秒）。edge 62次ROI调用累计1075.41秒，斜前62次250.97秒；逐帧同步CPU消费累计80.95秒。两rig打包／冻结Dataset回读／视频收尾分别22.74／25.09秒，整段生产交付wall 1712.69秒；没有失败重试。GPU0同一UUID，采样显存峰5153MiB，Blender峰RSS约18.06GiB，任务盘峰约448MB、临时峰约303MB，compact后该段目录约303MiB。两段总渲染＋收尾计算约3114.92秒，CPU预检392.28秒、新组合制作8.13秒，约3515.33秒、低于本批90分钟总限额；两段渲染＋收尾约51.9分钟、低于70分钟分项限额。新输出整体约629MiB，其中组合缓存约36MiB。不同人数与联合ROI面积使耗时不可作严格等负载速度比；边缘rig仍占相机调用主要成本。[5人成本](../../outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011/city_5p/production/task_wall.json)。

## 结束范围

[机器覆盖清单](../../outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011/coverage_inventory.json)分开列技术通过、用户观感与未通过的事件诊断。此次完成的是**城市直道的分散4／5人、两组静态双机、ROI-only数据链**。新增M03／M04组合及两段新画面的人工观感待用户看视频；未证明4／5人任意布局、三岔路口转弯、严格端点过界、15／15遮挡恢复、来源独立划分或训练效果。下一批才考虑路口道路核查、框增强ROI预留、正式来源分组与训练设计评审；本批没有运行DenseKP／YOLO或optimizer。

复用命令：`python tools/experiments/prepare_dispersed_cyclists_a1.py freeze`，按人数依序运行`preflight4/5`、`freeze_camera4/5`、`preflight_camera4/5`、`rig_manifest4/5`；使用版本化rig清单执行`python tools/experiments/run_multirig_roi_a1.py --rig-list-manifest <清单> --output <新目录> --gpu-uuid <空闲GPU UUID> --budget-seconds <有界值>`。本批因先做静态核查而使用同入口的`--run-prepared`，不能拿旧专用frame0-side续接选项冒充通用恢复。定向CPU unittest共17项通过，未跑全库测试、DenseKP/YOLO或训练。
