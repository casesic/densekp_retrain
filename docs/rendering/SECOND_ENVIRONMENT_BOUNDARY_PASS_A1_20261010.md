# 第二环境双机交界骑行探针

日期：2026-10-10。第一环境 `urban_street_assets` 保持原样。第二环境选择 **riverfront**：已完成同一 M01/APD 单骑行者 31 个合成 10 Hz 时刻的双路 ROI-only episode；[直接观看双路视频](../../outputs/experiments/second_environment_boundary_pass_a1_20261010/riverfront_complete/pack_corrected/dual_roi.mp4)，[统一页面](../../outputs/experiments/second_environment_boundary_pass_a1_20261010/index.html)。森林已接入并真实生成一个双路时刻，但三时刻探针在8分钟硬限前未完成，因此没有把森林列为本批可生产的第二环境。所有新数据仅限 `integration_only/shared_source_diagnostic`；环境、seed和邻帧变化不产生独立动作来源。本批未做框增强、推理或训练。

## 接入与冻结依据

沿用已通过端点过界的 `single_cyclist_boundary_pass_s20261010_e00` prepared recipe，核对它的 `selected_projection.json`、实际31帧产物和 M01/APD 原 timeline。人体/衣物/groom、原动作帧、合成10 Hz 时钟、0.496 m/s 场景速度、64 samples、5760×2880逻辑 ERP、双机相对外参/物理mask和ROI-only消费者均未重做。源外观为原已绑定的 M01 body、M01_simple 衣物、M01_groom、APD6526 动作；这是**冻结接入 fixture**，不声称本批重新随机抽了组件，也未调用A2重制几何。三秒道路位移约1.49米，没有动作循环、变速或补帧。

[版本化环境配置](../../configs/second_environment_boundary_pass_a1_20261010.json)用 master seed `20261021` 分别派生森林/河岸布局子seed，候选锚点从各自已冻结路段取值，每环境上限12。首次候选均通过，实际各检查1/12；[候选及拒绝记录](../../outputs/experiments/second_environment_boundary_pass_a1_20261010/prepared_riverfront/candidate_checks.json)和[森林记录](../../outputs/experiments/second_environment_boundary_pass_a1_20261010/prepared_forest/candidate_checks.json)留存。新公共适配器只对**整个**人物世界轨迹和两台 `camera_to_world`/physical lens 做同一个刚性变换，旧城市配方不调用它；相对外参和全ERP投影因此保持，实际两环境全段138点重投影最大差异约 `1.8e-12 px`。旧有效域mask沿相机本地 normal ERP 坐标使用，没有改FOV、像素画布或二次翻转cam4。身体、衣物和R138共享每帧新矩阵，groom继续使用该矩阵与原head变换；root仍由场景施加一次。预检31帧人体/衣物均在5米原生道路内，最小路缘余量森林1.87米、河岸1.90米；Blender真射线再核实地面对象/高度。

森林锚点 `y=-4.1288m`，按 `x=15+10sin(y/32)` 和切线放置；真正的射线都命中 `Asphalt_5m_continuous`，足底/道路间隙0.216–0.247m。森林节点确有13处引用旧 `Camera1` 的对象输入；公共装配仅在本次运行场景将它固定到双机光心中点，整段、两路不切换，不修改源 `.blend`。129个文件贴图路径核查无缺失。河岸锚点 `y=-0.0677m`、局部中心线 `x=0`，真射线命中 `Promenade_ground`/`Promenade_limestone_*`，间隙0.216–0.249m；护栏和草坡未被当道路。河岸场景无外部图像依赖，源1842对象。两者均用公共 rig，不使用场景自带预览相机。运行期环境设置分别记录原森林 Filmic/曝光1.0/7 bounces、河岸 AgX/曝光0/5 bounces；其它ROI生产策略固定，**两环境不是相同材质与ROI负载的严格计时基准**。河岸实图未见明显缺材质或路面错位，本批未制作 overlay/派生 `.blend`，原地面、护栏、植被保持只读。

## 探针、选择和成本

计算前 GPU0 UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2` 空闲，GPU1只有已有空闲桌面 Blender；系统约139 GiB可用内存，仓库卷约12 TiB可用。两环境均在 GPU0 串行运行，未终止他人任务。旧森林环境预览时长和 `.blend` 大小没有用于预测。

| 环境 | 探针实际结果 | 相机调用 | GPU采样峰值 | 选择判断 |
|---|---|---:|---:|---|
| cycling_forest | 483.04秒触及8分钟硬限；仅frame0双路成功并消费，frame1未完成 | 成功2，第三次启动未完成；首两路各219.97/161.04秒 | 14975 MiB；Blender RSS峰值约44 GiB | 原生路面与人物画面可用，但无法完成三时刻探针，也不能支持本批31时刻≤25分钟执行门槛；失败缓冲保留 |
| riverfront | 3时刻双路成功，含冷装配和同步消费54.11秒；后两时刻两路相机合计各约8.85/8.65秒 | 成功6 | 2187 MiB | 成本、地面、ROI、框/标签均可用，按冻结规则选中 |

森林首时刻的[cam2](../../outputs/experiments/second_environment_boundary_pass_a1_20261010/probe_forest/frame_000/cam2/roi.png)／[cam4](../../outputs/experiments/second_environment_boundary_pass_a1_20261010/probe_forest/frame_000/cam4/roi.png) ROI 留存；它证明该时刻能消费，不是三时刻稳态吞吐测量。河岸未选中前的三时刻共6次 ROI 调用和最终段是同一冻结版本。选中后新启动一次 Blender＋常驻同步CPU消费者，只渲染frame3..30（56次调用），只读复用探针frame0..2；跨探针/续跑共**两次**Blender装配，不宣称单进程完成全部31帧。新后缀 Blender wall 258.01秒，含首时刻后缀冷启动；其28时刻两路相机调用均值约7.27秒，ROI面积全段0.518–1.600百万像素/路。后缀 GPU峰值2183 MiB、Blender RSS峰值约3.23 GiB。探针＋后缀墙钟317.22秒，首次收尾在263.11秒的后缀任务内失败；针对性CPU收尾约10.24秒，未重渲染。连森林失败探针计入的本批已计时主要计算约810.50秒（13.51分钟），另有短暂Blender只读设置检查/CPU测试；低于45分钟总限。仓库本任务输出约148 MiB，含保留的森林失败现场；河岸长期图像及pack引用本任务探针/后缀目录，**不依赖已清理EXR**。相机调用时间含Blender同步与I/O，不是纯GPU kernel时间。

## 实际过界、数据与验收

选中环境的全31时刻生产ROI坐标与冻结规划逐一相等，注释 `in_view` 与预检期望逐点一致，世界R138和每人物原时钟记录均一致；图像/局部mask尺寸和全部GT有限。物理交界不是ERP接缝、ROI边界或实体双机0.18米间距。[逐时刻曲线](../../outputs/experiments/second_environment_boundary_pass_a1_20261010/physical_boundary_curve.svg)及[道路俯视图](../../outputs/experiments/second_environment_boundary_pass_a1_20261010/trajectory_plan.svg)分别展示变化与空间放置：cam2首/中/末域外R138数 **25→11→1**，cam4 **20→16→2**；共同物理域内点 **93→111→135**。两机边界距离第10百分位均由负转正，原 `enter_valid` 首末20→≤5端点条件通过；同时双边缘门槛满足 **21/31** 时刻，主连续窗口frame0..19、跨度1.9秒。两路共用同一个动作帧和世界状态，身体沿河岸道路正时间推进；这是一段实际进入有效域的过界过程，而非始终沿边停留。

最终31时刻/62逻辑相机图、62观测槽位全部有合法原生可见框，序列旁表和 `packed.npz` 校验通过。可靠GT仍按 `in_view & inside_full_crop` 监督，遮挡点继续参加；预先列出的frame0/15/30双路共**6条真实Dataset回读**逐项检查了局部ROI取样、138点仿射、监督、完整ERP `img_size` 和 `_trans`。实际[双路视频](../../outputs/experiments/second_environment_boundary_pass_a1_20261010/riverfront_complete/pack_corrected/dual_roi.mp4)解码31帧，PTS为0..3.0秒；画面灰色排版处未渲染，黑色镜头区域由physical mask定义，均不冒充真实背景。完整数值与资源证据见[acceptance.json](../../outputs/experiments/second_environment_boundary_pass_a1_20261010/acceptance.json)。

收尾发生一次明确接入错误：旧续接选择器假设必有被拒绝帧重消费，探针成功复用时 `reprocessed=[]`，首次已打包但在Dataset/视频前抛出索引错误。[故障与修正记录](../../outputs/experiments/second_environment_boundary_pass_a1_20261010/finish_correction.json)保留原 `pack`，新 `pack_corrected` 对空列表选择冻结首/中/末；旧有重消费选择行为不变。未重渲染成功帧，也未把失败归入自然无框。成功探针frame0保留的本任务自有EXR直到6条回读结束，随后按所有权标记清理；森林未完成frame1失败现场未清理。

定向CPU unittest **24项通过**，覆盖环境rig刚性变换、seed/原时间重放、缺失依赖及非法道路拒绝、探针只读续接，以及相关旧种子/视角/多人入口兼容；旧城市入口不带新字段时仍走原逻辑。未运行全库discover或任何模型更新。尚缺：森林当前原密度下可承受的连续生产方案、河岸人眼环境观感确认、其它场景模式、来源独立划分、框扰动ROI预留与正式训练。本批不自动开展这些工作。

复用入口（输出目录必须不存在；渲染前即时检查资源并明确设备）：

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python tools/experiments/prepare_second_environment_a1.py --environment riverfront --output outputs/experiments/<new>/prepared
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python tools/experiments/run_continuous_roi.py --prepared-episode outputs/experiments/<new>/prepared --output outputs/experiments/<new>/probe --storage-mode roi_only --consumer-mode persistent --observation-policy record_unavailable_v1 --probe-first 3 --frame-count 31 --gpu-uuid <idle_UUID> --budget-seconds 480
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python tools/experiments/run_continuous_roi.py --resume-parent outputs/experiments/<new>/probe --output outputs/experiments/<new>/full --storage-mode roi_only --consumer-mode persistent --observation-policy record_unavailable_v1 --frame-count 31 --gpu-uuid <idle_UUID> --budget-seconds <remaining_budget>
```

新计划路径会核查 `.blend` 大小和mtime契约；这项依赖核查是在本次预检/渲染后补入，**本次冻结产物仅保存场景路径、环境ID、rig及时间，未保存源 `.blend` 的运行前mtime**，不能反称本次已经验证文件版本未变，更不能宣称跨版本RGB逐位复现。所有源 `.blend`、旧城市数据、A2组合资产和用户TODO均未修改，也没有下载、安装、SHA256、commit或移入datasets。
