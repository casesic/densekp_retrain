# A1：M01 原时序骑行接入公共 ROI-only 场景

日期：2026-10-09。全部 `integration_only / shared_source_diagnostic`。本批仅接入已接受的 `cycling_M01_v2_APD6526_30_106_simple_v1`，不接F02/BEDLAM、不扩多人或训练。性能优化暂停，保持64samples CUDA、完整ERP逻辑栅格5760×2880、原色彩/光照/无降噪设置。

产物：[验收记录](../../outputs/experiments/cycling_roi_integration_a1_20261009/acceptance.json)、[冻结执行范围](../../outputs/experiments/cycling_roi_integration_a1_20261009/frozen_execution.json)、[实际配方](../../outputs/experiments/cycling_roi_integration_a1_20261009/run/recipe.json)、[原时序双路ROI视频](../../outputs/experiments/cycling_roi_integration_a1_20261009/run/pack/dual_roi.mp4)。

**本轮场景实际验收通过：26原始时刻、52张ROI、52行NPZ/旁表、6次真实Dataset回读及原整数PTS视频；一次Blender加载、一个常驻消费者，零重试。完整生成交付280.52秒。** 这是新增场景数据链的验收，不提升为真实曝光同步、独立验证或训练效果通过。

## 资产已有证据与本轮适配

A2的76帧资产检查、身体/简单衣物/groom接受状态只作为既有证据引用。本轮读取原parameters、缓存、绑定、head和角色材质，不改shape、姿态、拓扑、服装或头发，也不重复studio/SMPL-X重建。原A2 proposal中的 `A1_schema_effective=false` 和 `public_consumer_tested=false` 保持历史原值；本轮在自己的 `plan/cycling_m01_road_pass/asset_adapter.json` 使用显式 `densekp_scene_asset_v2`，实际消费证据由本报告和run产物给出。

薄适配层 `data/dynamic_scene.py` 映射明确的body/cloth/faces/head/groom引用。仅支持既有character/simple材质和固定参考groom；显式groom文件必须与已审计参考目录一致，不能悄悄替换。权威参数、原timeline、source_group、角色家族和人工状态均保留引用。APD与F02同属 `APD_5724486_v1_identity_session_unknown`，换角色/seed/邻段不构成独立train/val。

公共 `road_pass.plan` 通过显式schema分支进入 `dynamic_preflight.py`，原行人默认分支不变。`render_road_pass_sources.py` 复用原场景、body/cloth对象、groom和persistent data，按每时刻矩阵更新；原ROI渲染回调和常驻同步消费者直接复用。没有另一套骑行渲染、Dataset、投影或监督算法。

## 原时间、变换与放置

原timeline有76条记录，本批选择asset0..25/source30..55；原source tick `1476871677560014` 至 `1476871678560014`，首末跨度1秒。selected timeline为26个相邻原始观察，没有10Hz采样、插值、循环或复制；未选asset26..75明确列为unselected，不计缺帧。原tick和每条time_base保留，episode tick统一减本次首source tick，另保留资产原episode tick。

`time_base=[1,1000000]` 的微秒单位是根据APD名义25Hz推断，仍非官方时间单位/曝光标定。名义rate仅供视频编码描述；实际PTS来自整数tick。渲染manifest、annotations NPZ、消费报告和序列旁表共用同一selected record，两相机不分别nearest取帧；新资产不强制或伪造original_100fps_mapping。

身体已经包含global_orient，transl固定为负目标rest pelvis。场景唯一拥有行进：

```text
M[t] = fixed_yaw @ Y_up_to_Z_up + translation(start + speed * exact_tick_seconds)
body / cloth / R138 : M[t]
groom                : M[t] @ head[asset_frame]
```

这里“+translation”表示齐次矩阵平移列，不是再次叠加资产root。固定yaw依据选段已有global_orient的平均前向选择，仅用于一次道路朝向匹配，不再次施加逐帧global_orient。场景yaw约185.0418°，行进为世界+Y；各帧身体前向与道路方向最大差约25.45°（冻结门槛35°），不是侧身沿道路漂移。相邻身体缓存最大逐坐标变化0.01585–0.21433m，确有原动作变化，不是静态人偶平移；保留已接受的快摆限制。

固定seed20261009；第一候选即通过，未尝试另外三个候选。起点 `[-0.02,2.0,1.2]m`，终点 `[-0.02,4.0,1.2]m`，合成场景速度2m/s，不是APD真实速度估计。保持原urban_street及rig整体位置/朝向，双机0.18m仍为合成基线。相对镜头/normal ERP域规则不变，所以按段复制共享physical masks；没有额外翻转cam4或沿用错误世界投影。

固定骨盆高度1.2m：原选段身体最低model-Y约−1.011m，给出约0.189m最小净空，符合允许“虚空骑车”的范围，不要求车架、脚踏或把手接触。独立support policy检查body/cloth最低净空0.05..0.70m、骨盆距路面1.0..1.6m、路面Z误差≤0.03m、向上法线及道路对象；CPU检查身体在道路走廊内。实际Blender在导入人物前对26时刻射线核查，均命中 `Road_surface_7m`，低点间隙约0.18944..0.27029m。没有逐帧min-Y贴地；旧行人−2cm..12cm门槛仅保留于旧support分支。

## 冻结预检与兼容

所有26帧先执行CPU时间/身体R138/矩阵/投影/ROI预检。ROI覆盖本帧body、cloth、所有有效groom点及保守完整square裁剪支持，沿用32个最终ERP像素余量。区域外是unknown，physical valid mask与render coverage分开；没有通过收缩ROI提速。

冻结门槛：每时刻两侧各≥8个域内点、预计完整crop共同点≥8；每视角至少4个截断时刻；实际共同监督最少1。预检cam2域内117..138、cam4域内122..138；两侧分别22/18时刻含域外点，共同投影crop点101..138。这些是几何机会，实际栅格框和深度可见性统计另列，未用几何机会代替实际验收。

定向CPU覆盖22个唯一用例：7个新时间/root/support/旁表用例、5个旧road_pass用例、10个旧sequence用例。首次21项0.093秒通过，新增旁表冲突用例后新模块7项2.380秒通过；无全库discover。拒绝缺失或非整数时间、重复时间/asset帧、越界asset帧、未知root/support、非法非刚体矩阵、双重root所有权及源tick冲突。

同seed/配置的配方相等、scene矩阵重放逐值一致。修改前独立保存的旧planner与新legacy分支输出逐值一致；旧行人固定输入frame0/5/10的世界138点亦逐值一致。没有重渲染旧行人或重打包旧数据。32×32人工 `synthetic_test_only_native_pts.mp4` 验证不均匀PTS精确编码；不是动作产物。CPU几何预检未单独记录进程wall，不虚构其耗时。

开发时修正了顶层plan说明误沿用行人coverage字段，以及补入名义source rate说明，修改前元数据副本保留；没有改变已通过的几何矩阵。执行用配方和代码清单在渲染前冻结。

## 运行与后续复用

```bash
# 模型/渲染无关的计划与CPU预检；使用新的episode目录
mkdir -p outputs/experiments/cycling_m01_new_plan
python tools/experiments/plan_road_pass.py \
  --config outputs/experiments/cycling_roi_integration_a1_20261009/config.json \
  --output outputs/experiments/cycling_m01_new_plan/plan.json --geometry-preflight

# 同一公共连续入口消费已预检episode；按原selected timeline运行
python tools/experiments/run_continuous_roi.py \
  --storage-mode roi_only --consumer-mode persistent \
  --prepared-episode outputs/experiments/cycling_m01_new_plan/cycling_m01_road_pass \
  --gpu-uuid GPU-6f046d56-acbe-d49b-774a-9d463e768cac \
  --output outputs/experiments/cycling_m01_new_run
```

既有runner默认仍走旧行人/旧存储选择。显式prepared episode不允许在renderer内另行截短或改时间；先在planner选择timeline。常驻CPU消费者等待当前帧成功才处理下一帧。收尾不再强制11帧/22行/10Hz，按实际槽位、原生框和整数时间打包/编码；无框、empty、missing、reject保留原因，不为凑行数删除时刻。

F02以后可复用显式缓存、原timeline、场景拥有轨迹、固定骑行高度、head复合、同相机成像/标签及序列旁表能力，但必须使用F02自身shape/绑定/head并遵循其人工状态，本批不提升接受状态。BEDLAM原配行人仍缺其baked source-root＋固定场景放置分支、独立服装face-corner UV/贴图/normal材质，以及无groom/不同参考空间的显式支持；不能走本批simple衣物分支冒充接入。本轮未实施这些能力。

## 本轮实际场景验收

全部26时刻的source/asset/time、两机同状态和scene矩阵引用一致；annotations所有浮点数组有限，world138与冻结矩阵结果一致。body/cloth逐帧顶点回读、同世界矩阵、groom拓扑/头矩阵复合检查通过。没有缺帧/empty/无框/拒绝：52个真实observed槽位对应52行NPZ，不是预先凑行数。原76条中的未选择50条另列，未冒充缺帧。

| 实际ROI标签与原生框 | cam2 | cam4 |
| --- | ---: | ---: |
| 有效观察时刻 | 26 | 26 |
| 每时刻域内语义点范围 | 117–138 | 122–138 |
| 每时刻可见点范围 | 64–83 | 66–85 |
| 存在域外语义点的时刻 | 22 | 18 |
| 域外点累计（点×时刻） | 206 | 121 |
| 完整crop监督点累计 | 3378 | 3463 |
| 其中遮挡点仍监督 | 1452 | 1494 |
| 域内但完整crop外的点累计 | 4 | 4 |
| 不同原生框数 | 26 | 26 |
| 框宽/高范围（完整ERP px） | 182–330 / 397–749 | 184–291 / 397–746 |

双路共同监督每时刻100–138点，26个时刻均非空；具体逐时刻数量见acceptance JSON。总单视角监督6841点次，其中2946个遮挡点次继续监督。单人只有自遮挡及场景遮挡，未宣称多人互遮挡。40/52个单视角观察含有效域截断，达到事前冻结覆盖要求；实际紧框/裁剪与预投影不完全相同，例如首时刻共同点从预估101变为实际100，未复制预估统计。

首、中、末frame0/13/25双路共6次真实Dataset回读，独立重算局部图采样及完整坐标的标签affine，逐值一致；监督集合也一致。保持full ERP坐标、offset、_trans、img_size和256-square语义，无32px补偿，无随机框增强。原生框及全部打包crop通过既有高质量支持区检查。抽查所需缓冲保留至收尾检查完成，再按所有权清理；最终数据不依赖EXR或旧行人GT。

实际双路视频26帧，解码PTS按有理数与26个原episode tick逐值一致，首末1秒；末帧展示40010 ticks（0.04001秒，沿用最后一个已观察间隔），不是新增观察或循环。ROI灰底仅为排版占位，不是真实外围背景。视频供观看本轮道路接入，不设置重新接受旧资产的前置门槛。

## 实测成本、资源与保护

| 阶段 | 本轮实测 |
| --- | ---: |
| 一次性场景/资产装配 | 17.281 s |
| 一次性CPU消费者启动 / 关闭 | 2.292 / 0.716 s |
| 首帧循环，含消费 | 16.321 s |
| 后25帧均值 / 中位数 | **9.468 / 9.215 s** |
| 后25帧最小 / 最大 | 8.401 / 11.395 s |
| 后25帧几何更新均值 | 0.050 s |
| 后25帧ROI规划均值 | 0.449 s |
| 后25帧双路ROI调用均值（render/compositor/I/O/sync combined） | 8.445 s |
| 后25帧同步CPU消费回调 / 函数体均值 | 0.522 / 0.520 s |
| 函数体内局部保存均值（已包含于消费，不能重复相加） | 0.235 s |
| Blender进程wall，含常驻同步消费 | 274.930 s |
| 打包 / 六样本回读 / 视频 | 0.655 / 0.258 / 1.615 s |
| 独立收尾进程wall，含导入 | 5.583 s |
| **完整生成交付wall** | **280.521 s** |

没有故障或渲染重试，30分钟预算余1519.479秒。一次Blender PID1279164、一次消费者PID1279196。GPU1 RTX3090 UUID `GPU-6f046d56-acbe-d49b-774a-9d463e768cac`；计算前后均记录GPU、进程、内存与磁盘。只共享原空闲桌面上下文，未终止/争用其他活跃任务。GPU整卡显存1秒采样峰值3920MiB，包含桌面基础占用；Blender RSS峰值8,064,272KiB，消费者815,756KiB，收尾CPU678,512KiB。RSS不是显存，也未把相机combined耗时称为GPU kernel时间。

52张ROI PNG共49,375,465 B；run目录统计约52,593,590 B，含compact标签/共享mask/NPZ/旁表/视频/日志。临时源1秒采样峰值143,560,080 B，采样可能漏峰；长期成品无本任务EXR。成功清理的源有效文件共1,036,506,518 B，旧目录和失败现场一律未清理。

9.47秒只代表这一M01单人短段；人物沿路远离相机、ROI宽从约1188降到654像素，资产和轨迹均不同于旧行人14.71秒，不能当成新的性能优化收益，也不能承诺骑行/多人同速。本批没有OptiX、采样、异步、ROI范围缩减等探索。

保护核查采用size/mtime及已保存入口副本，不计算SHA256。用户TODO、A2文件和引用资产、原fixture、旧输出及外部项目保持；没有下载/安装、commit、Dataset发布、模型构造、DenseKP/YOLO推理或参数更新。

## 未覆盖与结束点

已有证据是A2接受的M01动作/衣物/groom；本轮新增通过的是原时序、道路行进、双路边缘ROI及标签/旁表/Dataset/视频消费链。未认证真实速度、真实曝光同步、自行车接触、多人交会、其它seed/长段、F02/BEDLAM接入或任何模型效果；138点语义和可靠遮挡监督没有改变。

公共能力已经接通，下一批可按各资产接受状态复用。当前停在交付点，不自动扩76帧、多人物、其它模式或训练。
