# 城市五人正面双机机位修正

日期：2026-10-11。范围仅为已交付城市五人同一`scene_episode_id=city_dispersed_5p_s20261031`的正面机位；原背面、侧面与旧正面结果保留。全部数据仍是`integration_only/shared_source_diagnostic`，未运行模型或训练。

## 问题与冻结选择

用户指出原正面视频的cam4人物过小。旧正面rig的目标person1原生实例框高度在31时刻的中位数为260像素，frame15为257像素；cam4在中段对person3只剩很小的实例框，person4／5域外。这是原ERP的真实成像尺度与physical视场关系，不通过预览缩放或改crop伪装修复。

只读复用[五人原prepared](../../outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011/city_5p/prepared_edge/recipe.json)、五套缓存及原source时间。新旧五人的世界矩阵、R138和31个10Hz时刻须逐值一致；只移动、旋转整套双机，组内相对标定、0.18m合成基线、安装高度、normal ERP和physical mask不变。新相机名`rig_front_revised`，保留旧`rig_front`供直接比较。

[配置](../../configs/front_rig_size_repair_a1_20261011.json)在计算前冻结camera master seed 20261036及32个道路侧方候选：道路两侧|x|=2.65／2.95／3.25／3.55m，沿路y=5.8／6.2／6.6／7.0m，小幅确定性位置／yaw扰动。筛查条件为两机同属严格front且共同physical点≥8至少16帧、连续≥6帧，cam4目标可见R138投影高度中位数≥旧机位1.35倍；全五人实际body／cloth／active groom完整crop、接缝拒绝与单相机ROI≤700万像素仍须通过。几何代理只用于候选选择，实际尺寸以下游原生框为准。

[冻结候选](../../outputs/experiments/front_rig_size_repair_a1_20261011/frozen_candidates.json)32份中10份通过轻量筛查，按cam4目标高度比和原attempt确定性排序；首位原attempt27（x≈−2.651m，y≈5.815m，yaw≈−49.23°）通过唯一一次精确全31帧支持检查，原候选ID不重新编号。正面共同physical窗口24／31、最长17帧，cam4目标R138代理高度中位数从255.11增至429.97像素（1.685倍）；frame15从约253增至422像素。相机在原城市场景中12向×0.25m静态短射线净空通过，但这不是完整障碍扫掠证明。新旧rig、相机seed、原source及完整支持记录见[冻结执行](../../outputs/experiments/front_rig_size_repair_a1_20261011/frozen_execution.json)。

公共CPU精确预检原先在复制source prepared时会把刚写的`candidate_checks.json`覆盖成旧布局候选；已让该文件免于source复制，并从本批`selected_projection.json`恢复任务自有的真实相机检查记录。旧source及历史产物未修改。该修复不改变投影或ROI算法。

## 实际结果

[新旧正面并列查看页](../../outputs/experiments/front_rig_size_repair_a1_20261011/index.html)含两段完整双路视频及同一第15帧静图；[新正面MP4](../../outputs/experiments/front_rig_size_repair_a1_20261011/production/rigs/rig_front_revised/pack/dual_roi.mp4)为31帧、10Hz、首末PTS 0–3秒，末帧展示0.1秒。灰区仍为未渲染，黄色线仍是physical有效域，不把黑区当背景。

[实际验收](../../outputs/experiments/front_rig_size_repair_a1_20261011/production/front_rig_size_acceptance.json)证明：一次Blender装配／一个常驻CPU消费者，31次共享五人几何更新、62次双路ROI相机调用，未重跑旧背面／侧面。五人people、原整数时间、世界R138和scene矩阵与旧城市五人场景逐值一致；两机相对矩阵最大差1.78e−15，仅整组位置／yaw改变。新机位cam4目标person1真实原生框高度中位数438像素（旧260，**1.685倍**），frame15为430像素（旧257），31帧实际范围378–498像素。cam2在frame15也从258增至430像素。这是完整ERP原生像素尺度变化，未放大预览或改Dataset采样。

新rig目标两机同属front `[0°,45°)` 且共同physical R138≥8达24／31时刻、最长17帧，这24帧都有双路合法原生框；旧正面同条件29／31，故尺寸改善伴随5个正面共同机会的减少。两机边缘截断及严格端点过界另列，不把此修复称边缘数据增益。新前视310人物／相机槽位打包221条合法行（旧233）：域外无实例框86、crop内无GT点2、完全邻人遮挡无原生框1，均保留原状态而非伪造输入。五人在新rig至少一台相机都各有≥6个相邻时刻合法框及≥8个实际可见R138。cam4第15帧只有p1、p2有效，p3–p5仍域外；这一机位改善主要是主目标尺寸，**没有让分散五人同时在cam4大幅入镜**。

预先固定的12个Dataset观察槽实际回读9条；缺少`f15_cam4_p3/p4/p5`三槽，未补抽。其余图像局部ROI、完整ERP坐标的138点仿射、可靠监督、`_trans`、`img_size`、mask、实例ID、框支持及序列索引均通过。遮挡可靠GT仍按`in_view & inside_full_crop`监督，模型与loss未改。旧正面结果保持可读，新视频的人眼观感待用户观看；用户之前对旧正面“太小”的反馈如实保留，不追认旧机位已获人工接受。

[成本清单](../../outputs/experiments/front_rig_size_repair_a1_20261011/delivery_summary.json)：装配78.43秒、首时刻45.42秒、后30时刻平均40.70秒；Blender渲染＋同步消费1344.83秒，打包／回读／视频收尾24.80秒，本批单rig完整生产wall 1374.70秒（22.91分钟），低于冻结1800秒预算。新任务目录约171 MB；生产目录清理后168.24 MB，运行时任务临时磁盘峰值约117.82 MB。CPU峰值RSS 18,983,208 KiB；GPU0 UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`，进程每秒采样显存峰值5117 MiB（不是GPU kernel独立计时）。CPU完整支持预检112.53秒、静态Blender净空约2.45秒另计；32份轻量投影耗时约1秒，不把它们并入渲染提速口径。

沙箱内第一次运行仅在资源查询的`nvidia-smi`子进程处失败，尚未启动Blender或渲染；后以相同冻结配方在可访问GPU的环境启动。实际生产一次Blender进程，未发生帧渲染、消费或finish重试。必要定向16项CPU测试与本批全槽位验证通过。新成果仍与旧三rig、邻帧和全部五位人物同一个混合APD／participant_1／participant_2来源组，不移入datasets，不作为正式独立验证或训练效果证据。

覆盖诊断单列：新rig共同physical点达标31／31，双路同时物理边缘0／31；目标首→末两机域外点数分别60→0、0→115，未满足同步严格端点过界。邻人IndexOB＋range证据已保留，但真实15／15三阶段恢复两路均未达标。这些结论不由正面尺寸改进替代。
