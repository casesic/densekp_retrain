# 城市五人同一场景：正面／背面／侧面三组双机

日期：2026-10-11。本批 `integration_only/shared_source_diagnostic`。用户已接受上一批城市4／5人道路视频中所展示的M03／M04观感；接受范围不扩大到完整资产窗口或任意重组。原4人邻人恢复不可评价、5人15／15恢复未通过的历史结果不改写。本批新三机位视频的人眼状态待观看。查看页：[正／背／侧三组视频](../../outputs/experiments/front_back_side_multirig_a1_20261011/index.html)。灰色排版区域未渲染，不是真实背景。

## 冻结人物与机位

共享`scene_episode_id`为`city_dispersed_5p_s20261031`，三组rig只是该场景的不同相机观察。

只读复用[城市5人权威prepared](../../outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011/city_5p/prepared_edge/recipe.json)与原组合缓存；五个角色家族、四个motion ID、APD／participant_1／participant_2三个原动作来源组、逐人源整数时刻和31个10Hz合成时刻均不变。新旧世界R138、`scene_matrices`逐值相同。未build新资产、没有对人物平移／旋转或改动作相位。所有rig和邻帧属于同一个`scene_episode_id`与来源组，不能按机位拆作独立训练／验证。

[版本化配置](../../configs/front_back_side_multirig_a1_20261011.json)固定camera master seed 20261034，front／back／side使用独立child stream。20261035仅生成另份计划，未投影或渲染。道路侧方候选从固定目标中段位置、道路横截面和到目标的方位构造；每种最多16份，seed决定候选顺序和小扰动。先保存[冻结候选和人物证据](../../outputs/experiments/front_back_side_multirig_a1_20261011/frozen_scene.json)，再做投影。正面与侧面的第0份各自通过，精确支持检查均只核查1份，远低于每类4份上限；背面直接核对继承旧`rig_edge`的准确中心与yaw，不把斜前方改名。所选rig内相对外参、安装高度、pitch／roll和0.18m**合成**基线保持，physical mask与cam4映射不改。

事前门槛与[预检证据](../../outputs/experiments/front_back_side_multirig_a1_20261011/frozen_execution.json)：统一目标p1。`rig_back` 两机同属严格back `[135°,180°]` 且共同物理R138≥8至少16时刻、连续≥6，并保留双路边缘≥16／31与连续≥0.5秒；实际31／31背面共同窗口、23／31双边缘。`rig_front` 严格front `[0°,45°)` 共同达标29／31、最长20帧。`rig_side` 严格side `[80°,100°)` 共同达标最长6帧、全段共同物理点达标23／31，目标首末跨过该rig道路横截面；其双边缘0／31，仅作为侧面观察对照。正面／侧面不要求双路同时截断；三组严格端点过界与邻人15／15恢复都另列诊断。全部31帧、全部五人的body／cloth／active groom、完整square crop和插值支持、ERP接缝拒绝、相机与人物距离、道路侧净空及每相机联合ROI≤700万像素已由相同生产规划算法核查。六个相机中心另作城市静态场景12向×0.25m水平短射线净空，均无命中；不是全三角形扫掠碰撞保证。[人物角度曲线](../../outputs/experiments/front_back_side_multirig_a1_20261011/target_view_angles.svg)、[逐人两机角度](../../outputs/experiments/front_back_side_multirig_a1_20261011/view_angles_by_person.json)、[俯视轨迹和rig](../../outputs/experiments/front_back_side_multirig_a1_20261011/topdown_rigs.svg)提供可重读定义。

[相机seed谱系](../../outputs/experiments/front_back_side_multirig_a1_20261011/camera_provenance.json)区分每rig child seed与其第0候选seed。生产snapshot在公共runner增补可选`camera_seed`字段前已准备，故背面snapshot记null、正面／侧面记候选seed；冻结rig manifest保留真正的rig child seed，三份prepared与生产的两机矩阵均逐值相同。这是元数据层级差异，未改变已渲染机位或人物状态。后续新任务的公共入口已传播两级seed。

## 公共链与本批执行

现有`scene_episode_rig_list_v1`直接列`rig_back→rig_front→rig_side`，每rig仍有cam2／cam4，单个场景时刻只更新五人各一次再顺序渲染六路、由一个常驻CPU消费者同步消费。旧单rig、两rig入口保持。新批只把明确含三rig的manifest预算上限扩到3600秒，旧默认2700秒不变，不能靠重试重置累计预算。ROI-only、逻辑ERP5760×2880、64samples CUDA、原城市光照／材质／色彩、无降噪与无低清背景保持。原生框与可靠`in_view & inside_full_crop`监督不改，无遮挡外区域继续标unknown。三rig计划31×3×2＝186次相机调用、930个人物观察槽；训练行取决于真实原生框。

本批CPU正面与侧面完整预检分别112.08／112.89秒；背面继承核对和静态场景净空另计，合计低于15分钟CPU预检上限。GPU0 UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2` 开算前空闲，GPU1另有他人Blender进程，未终止或争用。16项本批／相关定向CPU测试通过，未跑全库、模型或训练。

## 实际数据、视角与观测

[全槽位验收](../../outputs/experiments/front_back_side_multirig_a1_20261011/production/front_back_side_acceptance.json)通过：同一Blender进程装配一次城市及五人、31次共享人物几何更新、一个常驻同步CPU消费者，事件顺序为每帧`rig_back→rig_front→rig_side`各`cam2→cam4`，实际完成186次ROI相机调用。五人与旧场景的people、原整数时间表、世界R138逐值相同；三rig组内相对矩阵差≤1e-9。背面机位继承旧`rig_edge`的两机矩阵，31时刻两机的`xy`及physical `in_view`均与旧权威标注逐值一致。本批没有重新制作资产，也没有生产低清背景或完整ERP RGB。

| rig与用途 | p1两机同类且共同physical点≥8 | 最长连续窗口 | 同窗口两机实际原生框 | p1双路物理边缘 | 合法框行／310槽 | 冻结Dataset实读／12槽 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rig_back` 背面兼边缘困难 | 31／31 | 31帧 | 31帧 | 23／31 | 230 | 9 |
| `rig_front` 正面对照 | 29／31 | 20帧 | 29帧 | 0／31 | 233 | 10 |
| `rig_side` 侧面对照 | 12／31 | 6帧（2.0–2.5秒） | 12帧 | 0／31 | 186 | 7 |

侧面目标只在短窗口满足严格`[80°,100°)`；例如1.5秒仍是斜前方，不能称全程侧面。其共同physical点≥8为23／31，人物首末确实跨过侧方rig所在道路横截面。背面／正面共同physical点≥8均为31／31。三组实际视角窗口都有两路合法原生框；这只验证有模型输入，不是预测效果。五人在至少一组rig／相机中各有≥6个相邻时刻的合法框及≥8个实际可见R138点。所有930槽保留：649条合法行，281条不可用，其中背面80、正面77、侧面124。背面缺失分为邻人完全挡住无框53、physical域外23、框边过小4；正面77及侧面124均为physical域外无实例框。这些槽不生成假训练行。

每组冻结的frame15五人双机与frame0主目标双机共12槽，[实际回读](../../outputs/experiments/front_back_side_multirig_a1_20261011/delivery_summary.json)合计26条：背面9、正面10、侧面7；缺框槽没有替换补抽。ROI尺寸、offset、valid/person mask、原生框完整square支持、138点全ERP仿射与可靠遮挡监督、`_trans`和逻辑`img_size`均通过实际Dataset及全帧序列索引检查。可靠GT仍按`in_view & inside_full_crop`监督，含遮挡点。三组均保留IndexOB＋range邻人逐点证据；真实15／15三阶段加重—恢复在两台相机的三组中均**未达到**。严格端点过界也未通过：主目标两机首→末域外R138数量背面为22→3／18→0，正面为0→0／0→81，侧面为138→0／0→138，没有一组满足两机同步的原20→5或5→20必要条件。这些指标不能因新视频观感而写成通过。

[统一查看页](../../outputs/experiments/front_back_side_multirig_a1_20261011/index.html)并列链接[背面](../../outputs/experiments/front_back_side_multirig_a1_20261011/production/rigs/rig_back/pack/dual_roi.mp4)、[正面](../../outputs/experiments/front_back_side_multirig_a1_20261011/production/rigs/rig_front/pack/dual_roi.mp4)、[侧面](../../outputs/experiments/front_back_side_multirig_a1_20261011/production/rigs/rig_side/pack/dual_roi.mp4)。每段实测31帧、10Hz、首末PTS 0–3.0秒，最后一帧展示另占0.1秒，容器时长3.1秒。灰区是未渲染区域。首／中／末可看人物与物理边界，但本批新机位观感仍待用户观看，不自动继承旧视频的人工接受状态。

## 计算、失败与限制

[成本清单](../../outputs/experiments/front_back_side_multirig_a1_20261011/delivery_summary.json)记录Blender装配79.36秒、首时刻69.40秒、后30时刻均值58.07秒／中位57.31秒／范围55.71–64.20秒。Blender渲染＋同步消费wall 1890.93秒；三rig收尾分别22.49／22.09／21.34秒；完整生产与收尾wall 1961.95秒（32.70分钟），低于60分钟生产上限。31帧几何更新计时合计3.53秒；render callback combined合计1703.40秒、consumer callback合计104.61秒。各相机调用combined另见成本JSON；首个背面cam2调用承接动态场景同步，其均值31.50秒，其余五路约2.96–3.17秒，不能把该差值解释为纯像素GPU kernel时间。背／正／侧各路ROI平均约174／157／152万像素、最大388／208／165万，均低于700万上限，负载并不相同。长期目录约412.5 MB（十进制）；生产目录406.4 MB，成功清理后任务临时文件余397 KB。CPU峰值RSS 19,048,516 KiB。GPU使用UUID已记录，运行期峰值显存没有可靠独立采样，记为未测，不编造数值。

沙箱中的首次资源查询因`nvidia-smi`子进程限制退出，发生在Blender启动和任何渲染之前；[启动审计](../../outputs/experiments/front_back_side_multirig_a1_20261011/sandbox_launch_attempt.json)保留。随后相同冻结任务在授权设备正常启动一次，生产无渲染／消费重试，3次finish均成功。未修改旧失败记录或旧成果。当前技术通过只覆盖这一个城市直道五人场景和三组静态rig；front/side各自0／31双边缘，不能算边缘配额。所有rig与邻帧仍属于APD＋participant_1＋participant_2混合的同一共享来源组，不拆正式train/val，不移入datasets。

下一项三岔路口需要先核查城市端部T形道路的实际可行区域与障碍净空，定义转弯路径、每人随路径变化的朝向和一次性root所有权，再预检双机physical域及多人ROI支持；直道三rig通过不证明路口或任意rig数已可用。框增强ROI预留、来源隔离、原权重基线与受控训练也未实施。本批在交付点停止。

复用命令：`python tools/experiments/prepare_front_back_side_a1.py freeze`，接着分别运行`preflight_front`、`preflight_side`、`manifest`；在空闲设备上以全新目录调用`python tools/experiments/run_multirig_roi_a1.py --rig-list-manifest <rig_list_manifest.json> --output <新目录> --gpu-uuid <UUID> --budget-seconds 3600`。本次先做CPU静态准备，随后用同入口`--run-prepared`启动。旧专用side-frame0修复选项不用于三rig。
