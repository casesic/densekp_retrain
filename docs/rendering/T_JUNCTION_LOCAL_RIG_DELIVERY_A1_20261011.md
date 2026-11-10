# T 形路口：局部机位修正与普通三视角配置

日期：2026-10-11。当前状态：**同一三人转弯场景的正／背／侧三组双机已完成31时刻真实生产，独立数据验收通过。** 本批先按新授权尝试双边缘，未通过精确ROI；随后使用事前确定的普通路口路径，选定正／背／侧各一组机位。前三次Blender生产启动均在首个相机调用前遇到数值检查实现错误，保留三个失败目录；用户另外批准第四次启动后，冻结配方成功完成。[执行状态](../../outputs/experiments/t_junction_local_rig_delivery_a1_20261011/execution_state.json)与全部日志可复核。**本段不计入双边缘配额，新的路口视频观感尚待用户观看。**没有改写[上一批48机位](T_JUNCTION_MULTIRIG_A1_20261011.md)或[1376机位](T_JUNCTION_RIG_REPAIR_A1_20261011.md)的拒绝结果。

## 固定场景与机位结果

人物场景只读引用原[prepared_scene](../../outputs/experiments/t_junction_multirig_a1_20261011/prepared_scene/)，scene_episode_id仍为city_t_junction_3p_s20261040。p1=M01/APD6526转入横路，p2=M02/participant_1 comfortable、p3=F02/participant_2 straight按各自原时间向前运动；31个场景时刻、三套身体和世界R138数组与原文件逐值一致。沿用原生Road_surface_7m与Cross_street.001的路面支持，不修改.blend、人物、轨迹或physical mask。

[本批配置](../../configs/t_junction_local_rig_delivery_a1_20261011.json)冻结camera seed20261043、两个**不同位置**的近临界锚点（旧back attempt152、front attempt68）、最大0.6m平移与锚点±12°yaw。按全段最危险人物根点到真实cam2/cam4光心的方向构造位移。512个唯一提案事前写入[frozen_local_proposals.json](../../outputs/experiments/t_junction_local_rig_delivery_a1_20261011/frozen_local_proposals.json)；先做双光心对所有三人31帧的0.5m净空，再做完整physical投影。本批512份都过了根点初筛，224份在轻量阶段具备视角、尺度和连续双边缘机会。逐项记录在[local_checks.json](../../outputs/experiments/t_junction_local_rig_delivery_a1_20261011/local_checks.json)。

精确检查预算冻结为最多3个优先边缘候选、每视角最多3个普通候选；实际检查8个。三个边缘候选分别因完整ERP裁剪越界或front在第16帧的联合ROI超过700万像素被拒绝，**不能生产**。普通方案选中：

| rig | 原候选 | 整体XY米／yaw | p1同类＋共同窗口 | 双边缘最长窗口 | 较小一路R138高度中位 | 最大单路ROI |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| back | prior_back_88 | (−1.937,75.953)／52° | 26帧 | 1帧 | 269px | 166万像素 |
| front | prior_front_5 | (1.969,83.920)／75° | 20帧 | 2帧 | 383px | 542万像素 |
| side | prior_side_27 | (6.035,75.978)／298° | 10帧 | 0帧 | 248px | 75万像素 |

上述数值是CPU投影代理，非真实原生框或实际可见输入。三组各自通过31帧三人的body、cloth、active groom完整square crop、ERP接缝及单路ROI≤700万像素；[原生Blender静态短射线](../../outputs/experiments/t_junction_local_rig_delivery_a1_20261011/static_clearance.json)对所有12个冻结备选光心无障碍命中。全段人物根点净空也通过。根点距离和离散短射线均不等于连续人体／场景无碰撞证明。选择和全部拒绝见[rig_selection.json](../../outputs/experiments/t_junction_local_rig_delivery_a1_20261011/rig_selection.json)。最终[rig_list_manifest.json](../../outputs/experiments/t_junction_local_rig_delivery_a1_20261011/rig_list_manifest.json)显式标记ordinary_no_edge_quota，三组共享同一个scene episode与来源组。冻结了每rig6槽、合计18个[Dataset回读槽位](../../outputs/experiments/t_junction_local_rig_delivery_a1_20261011/readback_plan.json)，无框不替换。

三组计划ROI总面积约3.062亿像素，与旧五人三rig的约2.991亿像素同量级；这只是启动前负载参考，人物数量、近场ROI和同步成本不同。按旧完成任务32.70分钟估计本批约35–50分钟并设置60分钟总生产上限，没有承诺实际耗时或降低64 samples/CUDA画质。

## 生产故障与成本

先行沙箱试启动在Python子进程读取nvidia-smi时被NVML拒绝，没有启动Blender；随后在批准的运行环境中使用空闲GPU0，UUID GPU-d481acc7-106f-29e5-1298-8c0d398261b2。GPU1当时另有任务，未使用或终止。系统约126GiB可用内存，工作盘约12TB空闲。

第一次Blender装配耗时52.73秒，在首个相机前的R138核对中被固定2微米绝对容差拦下。Blender对象矩阵在约80米的世界坐标以float32表示，实际差约5微米。针对性修正确认对象矩阵**精确等于计划矩阵的float32表示**，世界R138再按各坐标一个float32 ULP与原2微米下限核对；全部93个人物时刻CPU回核最大差约7.39微米且无超限，故意错1毫米root会被拒绝。第二次Blender装配又耗时52.62秒，但当前Blender内NumPy的assert_allclose报错格式不支持数组容差，抛TypeError，同样未进入相机循环。获用户批准的第三次启动耗时52.51秒，又在同一逐人物循环发现groom世界点核对残留固定2微米断言：在世界Y约82.6米处出现6.05微米float32舍入。现在groom也先核对矩阵的精确float32表示，再按不超过一个float32 ULP（原2微米下限）检查世界点；全文件同类断言已搜索，10项相关unittest通过。用户批准的第四次启动完成生产。

前三次Blender启动合计约157.87秒，成功相机调用0；不存在可复用的成功帧。用户另准第四次启动，在独立目录`production_retry3`成功产出。局部候选投影约数秒，精确支持8次合计321.64秒，加静态Blender净空与定向测试，低于20分钟CPU检查上限。三个独立失败目录production、production_retry1、production_retry2保持原状。未做背景、完整ERP RGB、DenseKP/YOLO或训练。

第四次的单次Blender装配52.05秒，首时刻六路渲染合计38.49秒，后续30个时刻均值36.37秒、中位35.38秒、范围31.99–42.08秒；此数包括相机调用与每帧ROI规划，不把第一路的动态同步误称纯像素成本。各rig两路调用、常驻消费和pack/回读/视频细项见[成本表](../../outputs/experiments/t_junction_local_rig_delivery_a1_20261011/delivery_cost.json)。成功生产总wall为1355.56秒（22.59分钟），前三次失败157.87秒另计；四次启动合计1513.42秒（25.22分钟），低于60分钟总生产预算。成功Blender进程峰值RSS约13.05GiB；中途采样GPU0显存约3.2GiB，未建立严格显存峰值。最终生产目录约312.1MB；各同步消费单元最大原始临时缓冲约276.6MB，累计约7.95GB的临时数据按成功消费清理，仅余约0.32MB任务临时元数据。全部四次启动均使用同一冻结rig、64 samples和原城市渲染设置。

## 实际覆盖、数据与观看

第四次正常路径由**一个**Blender进程加载城市及三人、一个常驻同步CPU消费者，在31个场景时刻各更新一次三人的几何，然后按back→front→side每组cam2/cam4共186次ROI相机调用。`scene_episode_id=city_t_junction_3p_s20261040`、世界R138和scene矩阵与原prepared逐值一致；两机在组内保持相对外参。三组共享来源，绝非三个独立人物/动作样本。全[558个观察槽](../../outputs/experiments/t_junction_local_rig_delivery_a1_20261011/production_retry3/junction_acceptance.json)核对时间、身份、GT、mask、ROI、框及索引，361条有合法框，197槽不可用并保留原因。所有三人在至少一组的一台相机都取得连续31时刻合法框且≥8个实际可见R138。成功帧使用ROI-only，不产生低清背景或完整ERP RGB。

| rig | p1双机同类视角＋共同physical点 | 最长连续窗口 | 双路边缘时刻／最长窗口 | 原生p1框高度中位数cam2／cam4 | 合法框行／计划槽 |
| --- | ---: | ---: | ---: | ---: | ---: |
| back | 30/31 | 26时刻 | 1／1 | 278／277px | 145／186 |
| front | 20/31 | 20时刻 | 2／2 | 393／510px | 113／186 |
| side | 10/31 | 10时刻 | 0／0 | 251／248px | 103／186 |

表中的角度均按p1身体水平前向到各相机光心计算；同类窗口还要求共同physical R138≥8。原生框高度的两机分母不同，详见机器表。双路边缘各组最长仅2时刻，**没有连续6时刻的边缘配额**。严格首末端点过界亦未通过；轻量IndexOB/range邻人证据已保存，但三组的15/15个百分点三阶段加重—恢复均未达到。二者是独立诊断，不影响本批事前选择的普通路口目标。

渲染前固定每rig6个Dataset观察槽，共18槽；其中12个有框槽各真实读取一次，图像crop、138点affine、可靠GT监督、`_trans`及完整ERP `img_size`通过；其余6槽无框，不替换抽样或伪造输入。可靠GT仍按`in_view & inside_full_crop`监督，包括遮挡点；physical valid与render coverage分开。[全槽验收](../../outputs/experiments/t_junction_local_rig_delivery_a1_20261011/production_retry3/acceptance.json)和各rig `pack/readback.json`保留明细。三组视频分别31帧、10Hz，PTS 0–3.0秒，末帧展示0.1秒另列；灰底是未渲染区域，不是环境背景。直接观看[三组并列页面](../../outputs/experiments/t_junction_local_rig_delivery_a1_20261011/index.html)，俯视[轨迹和机位](../../outputs/experiments/t_junction_local_rig_delivery_a1_20261011/junction_topdown.svg)。

重放命令（保持冻结rig和来源引用）：`PYTHONPATH=src python tools/experiments/run_multirig_roi_a1.py --output <new-output> --rig-list-manifest outputs/experiments/t_junction_local_rig_delivery_a1_20261011/rig_list_manifest.json --gpu-uuid <idle-UUID> --budget-seconds <remaining-budget>`。完成后用`PYTHONPATH=src python tools/experiments/validate_t_junction_local_delivery_a1.py --root <new-output>`复核，再用`python tools/experiments/finalize_t_junction_local_delivery_a1.py --root <new-output>`生成成本与观看页。全部结果仍是integration_only/shared_source_diagnostic，不进入正式train/val。
