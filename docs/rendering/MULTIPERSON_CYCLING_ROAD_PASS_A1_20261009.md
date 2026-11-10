# A1：可复现的多人骑行 road_pass

日期：2026-10-09。`integration_only / shared_source_diagnostic`。本轮只有一个正式episode、两个M01实例；不增加人物外观或独立动作来源，不训练、不推理。当前执行证据位于 [输出目录](../../outputs/experiments/multiperson_cycling_road_pass_a1_20261009/)。

**交付结论：seed→双人配方→76时刻双路ROI→304行打包/旁表→12次Dataset回读已通过；互遮挡恢复覆盖未通过。** 整段实际17.06分钟、零渲染重试。未追加episode、降低门槛或启动训练。完整状态见 [delivery.json](../../outputs/experiments/multiperson_cycling_road_pass_a1_20261009/delivery.json)。

## 冻结范围与来源

唯一组合为 `cycling_M01_v2_APD6526_30_106_simple_v1`，复用已接受的身体、simple衣物、groom、head缓存、shape及材质。两个人分别为person1（稳定目标）和person2（遮挡者），同一个资产、同一个动作来源、同一相位。F02、BEDLAM及A2新山地车素材均未使用或修改。

完整76条原记录：asset0..75/source30..105；源tick1476871677560014..1476871680560014，`time_base=[1,1000000]`，首末3秒。保留原不均匀整数间隔，未插值、变速、循环、复制或改成10Hz。微秒单位仍是根据名义25Hz推断，不能作为官方曝光标定。两机消费同一selected record；所有人物的asset/source帧必须与该记录相同，原`frame_record`检查未放宽。

本次只随机道路起点、横向/纵向间距、恒定行进速度。角色、衣着、动作相位、相机、固定高度及朝向不随机。局部`random.Random`生成63位attempt子seed，再用独立RNG抽样，不依赖全局随机状态。固定算法版本为`road_pass_multiperson_scene_owned_v1`；承诺配方和矩阵重现，不承诺跨Blender版本RGB逐位一致。

| 参数 | 事前抽样范围 | person1实际 | person2实际 |
| --- | --- | ---: | ---: |
| target起点X | −0.5..0.5 m | 0.2363713 | 0.4758816 |
| target起点Y | 3.5..5.0 m | 3.9524716 | 2.8492180 |
| 遮挡者相对X/Y | −0.6..0.6 / −2.2..−1.1 m | — | +0.2395103 / −1.1032536 |
| 恒速 | 1.0..2.8 m/s | 1.6857243 | 1.6162772 |
| 骨盆高度 | 固定1.2 m | 1.2 | 1.2 |
| 道路方向 | 固定世界+Y | +Y | +Y |

速度是合成场景参数，不是APD真实速度估计，也没有改变源动作播放速度。本轮不要求自行车及脚踏/把手接触。

## 有界CPU计划与选择

固定seed为20261009、20261010、20261011，每个最多12个候选，总共36个独特候选。参数范围、抽样顺序和实际值见[配置](../../outputs/experiments/multiperson_cycling_road_pass_a1_20261009/config.json)及[candidate记录](../../outputs/experiments/multiperson_cycling_road_pass_a1_20261009/plan/multiperson_m01_road_pass/candidate_checks.json)。选中第一个满足本轮几何规则的候选：seed20261009、attempt5；未增加到3人。

最初CPU草案错误沿用了旧单人条件“两路各至少4个截断时刻”，导致36个候选均被拒绝。原配置、检查和失败日志保留在`before/config_inherited_single_gate.json`、`plan_inherited_single_gate/`。修正为本轮要求的双路持续观察、各路至少3个64px边缘机会、至少一路4个截断时刻后，只重新核对同一批样本，新增候选为0。没有修改下述实际互遮挡15个百分点门槛，也没有看渲染结果后重选配方。此实现修正单列于`preflight_correction.json`，不冒充最初一次通过。

CPU检查全76时刻的时间、R138、一次root、道路包络、固定高度、投影支持、重叠变化和联合ROI。目标每时刻两侧共同physical-domain点至少8；选中候选cam2域内95..138点、28个截断时刻、76个64px边缘时刻；cam4域内138点、无截断、3个64px边缘时刻。这是事前几何机会，后续深度/实例验收单列。

最小人物根间距1.12895m。每个原时刻的body/cloth世界XY包围盒须至少一个轴分离≥0.1m，根间距≥0.9m；实际最小分离轴间隙0.47145m。身体/衣物最低路面净空0.14546..0.25061m，骨盆固定1.2m，使用原骑行support policy。Blender仍须对实际道路进行射线核查。该包络检查保守、廉价，不是全网格相交、所有连续中间瞬间或真实自行车碰撞认证。

继续使用已验收的固定185.041806°场景yaw和一次Y-up→Z-up；身体已有global_orient，不能再次叠加。每个人用原整数时间计算唯一`scene_from_model[t,person]`：body/cloth/R138同矩阵，groom为scene×原head。未逐帧贴地。rig与urban_street不变，0.18m仍为合成基线。

联合ROI沿用所有live body/cloth/groom几何的保守完整deployment-square支持包络＋32个最终ERP像素余量。选中候选还对全部有效groom点逐帧预检。每机ROI上限7,000,000像素，实际预估cam2为150,544..872,356像素、cam4为148,225..859,329像素。这个上限只用于拒绝过大配方，不截短ROI或降低采样。没有跨缝分块或按人单独合成。

## 事前冻结的互遮挡口径

[执行清单](../../outputs/experiments/multiperson_cycling_road_pass_a1_20261009/frozen_execution.json)在渲染前写入。分母为同一时刻目标在两相机physical domain内的可靠138点交集，不取决于可见紧框或crop，不删除困难时刻。

邻人遮挡证据要求同时满足：原规则判为不可见；中心射线IndexOB是另一已知person_id；range比该GT距离近超过0.04m；3×3邻域至少5/9像素有同一个前景ID且同样近超过0.04m。这个3×3证据位于原可见性5×5支持内，既有visible计算不改。保存前景ID、实际range、GT-range差和一致像素数，临时EXR清理后仍保留每点证据。

证据码区分域外、可见、邻人、本人实例前景（含body/cloth/groom）、场景前景及未确定。未确定不自动归为自遮挡；本人实例证据也不声称区分身体与本人的衣物遮挡。物理mask与render coverage分开，ROI外unknown，不按黑RGB判定。

按时间寻找三个不重叠、各含3个相邻原记录的窗口。中段**最低**邻人比例须高于前段**最高**比例至少0.15，并高于恢复段最高比例至少0.15。空分母为null，不能拼接跳帧窗口。两个相机独立计算，任一路满足即可；返回首个符合规则的时间三元组。CPU凸包投影重叠仅作预筛，不是实际遮挡证据。

这只是同源时序的相对轻/重/恢复对照。动作、位置、域内点及投影尺度也随时间变化，不是“同姿态同位置、只改遮挡”的因果对照。

## 公共接口与运行

新增薄的`data/multiperson_road_pass.py`处理有界RNG和多人CPU预检，复用`dynamic_scene`时间/变换、现有相机投影及ROI规划。原行人和单M01分支不改默认行为。

`scene_matrices.npy`显式支持`[T,P,4,4]`，配套`scene_matrix_person_ids`；每条旁表记录`scene_matrix_row`和`scene_matrix_person_index`。校验器检查人物顺序、矩阵合法性、recipe逐值对应及源时间。旧`[T,4,4]`仍可读。NPZ、Dataset返回、标签和监督语义不变。

既有renderer和consumer的多人循环直接复用；同相机/时刻只保存一张ROI，多个person框引用同一路径。body/cloth/groom各自使用所属person实例ID。无可见框仍保留observation及render coverage引用，实际有效行数由观测决定，不强制304行。新证据只是旁路统计，不替代`in_view & inside_full_crop`监督；遮挡GT继续参与。

```bash
# 新输出位置；相同配置/seed恢复相同轨迹。
mkdir -p outputs/experiments/multiperson_new_plan
python tools/experiments/plan_road_pass.py \
  --config outputs/experiments/multiperson_cycling_road_pass_a1_20261009/config.json \
  --output outputs/experiments/multiperson_new_plan/plan.json --geometry-preflight

python tools/experiments/run_continuous_roi.py \
  --storage-mode roi_only --consumer-mode persistent --budget-seconds 2700 \
  --prepared-episode outputs/experiments/multiperson_new_plan/multiperson_m01_road_pass \
  --gpu-uuid GPU-6f046d56-acbe-d49b-774a-9d463e768cac \
  --output outputs/experiments/multiperson_new_run
```

runner默认预算仍1800秒，显式最大2700秒；native prepared episode最多76条且禁止在renderer内重新采样。当前命令是复用说明，不表示授权自动再跑一段。

## 定向检查

7个新增synthetic_test_only CPU用例覆盖局部seed重现、多人物共享时间/相位拒绝、近邻包络、联合ROI/越界、保守深度ID证据、三连续窗口、共享图像多框/空槽位及人物矩阵映射。另跑7个既有动态时间/root/support用例，共14项最终通过，0.237秒；首次发现的空槽位coverage引用缺失已修复，首次失败记录保留。

旧M01单人planner使用修改前独立副本对照，结果完全一致；frame0/13/25世界点与原输出逐值一致；旧NPZ实际回读1次，局部图像采样与完整ERP affine一致。新选中矩阵重新计算逐值一致。没有重渲染、重打包或覆盖旧M01输出，没有全库discover、模型构造或参数更新。

## 实际生成与成本

**数据流程通过；冻结的互遮挡加重→恢复门槛未达到。** 一次Blender PID1374402、一个同步消费者PID1374434，完成76时刻、152次成功ROI相机调用、152张ROI、304个真实有效原生框/NPZ行。无缺帧、empty、reject、渲染失败或重试；没有按预期行数伪造样本。物理域外语义点仍逐点记录，不因该观察有框就声称全部可见。

[机器验收及成本](../../outputs/experiments/multiperson_cycling_road_pass_a1_20261009/costs_and_acceptance.json)、[旁表验收](../../outputs/experiments/multiperson_cycling_road_pass_a1_20261009/run/pack/validation.json)、[原时序双路视频](../../outputs/experiments/multiperson_cycling_road_pass_a1_20261009/run/pack/dual_roi.mp4)、[目标时间曲线](../../outputs/experiments/multiperson_cycling_road_pass_a1_20261009/run/pack/target_curves.svg)、[逐人逐时刻统计](../../outputs/experiments/multiperson_cycling_road_pass_a1_20261009/run/pack/per_person_time_statistics.json)。视频灰底是未渲染排版区，黑色镜头无效区来自physical mask；没有补渲染全景背景。

全部152个人物时刻通过实际道路射线检查，命中Road_surface_7m及已允许Center_dash.013/.014；抽样低点间隙0.14892..0.27029m。CPU全顶点最小净空与此抽样射线范围不同，分别保留，不把抽样值称为全网格最小值。body/cloth逐帧顶点、138点世界坐标、head复合矩阵和groom拓扑检查通过；八个人物对象（每人body、cloth、两组CURVES）按person_id=1/2正确归属。对象/mesh identity在全段不变。

全部annotation浮点值有限，source/asset/tick/person/camera/matrix一致；局部mask尺寸正确、人物mask互斥、physical支持和原生框紧框校验通过。旁表记录304个observed槽位与304行一一对应，仅152个不同RGB路径，未按人物复制图像。临时EXR不是长期依赖。

| 人物 / 相机 | 域内GT范围 | 可见GT范围 | 64px边缘时刻 / 截断时刻 | 完整crop监督点次 | 遮挡仍监督点次 |
| --- | ---: | ---: | ---: | ---: | ---: |
| person1 / cam2 | 95–138 | 44–67 | 76 / 28 | 9907 | 5503 |
| person1 / cam4 | 138 | 46–64 | 3 / 0 | 10457 | 6151 |
| person2 / cam2 | 32–135 | 19–82 | 76 / 76 | 6241 | 2352 |
| person2 / cam4 | 138 | 69–89 | 0 / 0 | 10476 | 4275 |

person1双路共同physical GT为95–138点，实际共同crop监督为94–138；person2为32–135点。全部人物时刻共同crop非空。每人每机76个原生框均不同。person1框宽：cam2 73–152px、cam4 73–179px；框高174–402/175–403px，对应deployment scale约0.870–2.015。中心、宽高、scale、共同点及域外点的完整曲线/表保留。框是当前可见实例mask的oracle框，不是YOLO检测框，变化不构成“框变化导致模型误差”的证明。

### 实际互遮挡与失败口径

person1确实受到person2遮挡：域内邻人证据累计cam2 2821点次、cam4 3715点次。本人实例前景分别2687/2422点次，未确定23/33点次，场景前景0。person2没有邻人遮挡证据，只有本人实例/未确定遮挡。这里的累计域内点次与下面共同physical GT比例的分母不同，不能相除混用。

共同physical GT上的邻人遮挡比例范围为cam2 **14.74%–36.96%**、cam4 **22.11%–40.58%**；单纯最大减最小不是本轮三阶段验收。按冻结的三相邻时刻窗口，两个相机均无合格事件。

[缺口量化](../../outputs/experiments/multiperson_cycling_road_pass_a1_20261009/coverage_gap.json)在事后只用于解释失败，没有更改门槛：

- cam2的最佳对称起伏窗口为frame0–2 → 34–36 → 67–69，即约0–0.080020s → 1.360027–1.439986s → 2.679983–2.760003s。对应比例14.74–16.33% → 35.51–36.96% → 26.81–27.54%。可保证加重**19.18个百分点**，但恢复只有**7.97个百分点**，未达到15。
- cam4的最佳对称起伏窗口为frame0–2 → 30–32 → 69–71，可保证加重14.64个百分点、恢复10.14个百分点，同样未达到15。

这说明本次凸包投影预筛不足以保证真实可见栅格上的恢复幅度，不能把预测的前/中/后窗口提升为已通过事件。已有深度/实例证据确认变化涉及邻人，不拿总不可见点增加替代互遮挡。没有删帧、更换分母、放宽阈值、换seed或追加第二个episode。

### Dataset与视频

因实际门槛未通过，按事前冻结的fallback，使用CPU预筛窗口中心frame1/12/19（source31/42/49，tick40010/479965/760003）作前/中/后接口抽查；这不是声称三个时刻属于已验收遮挡事件。双路×2人共**12次真实Dataset回读**，另有旧M01兼容回读1次；未重复遍历全部304行Dataset。

12项图像采样与138点完整ERP affine逐值一致，监督集合一致，full-image img_size、ROI offset、完整256-square及无32px补偿保持。遮挡仍监督。实际输入crop及[三张深度/ID证据叠加图](../../outputs/experiments/multiperson_cycling_road_pass_a1_20261009/run/pack/event_evidence_frame_012.png)仅用于解释，未生成模型预测或逐帧全套调试图。

视频76帧按原整数tick编码，解码PTS的有理数值逐条相等；首末3秒，最后展示40010ticks=0.04001秒单列，文件约3.04001秒不代表增加观察。无复制帧或拉长。同步声明仅synthetic_same_state，真实曝光同步未验证。

### 实测成本与资源

| 项目 | 实测 |
| --- | ---: |
| 一次场景/资产装配（combined） | 33.986 s |
| 常驻CPU消费者启动 | 2.316 s |
| 首时刻完整循环，含消费 | 19.112 s |
| 后75时刻均值 / 中位数 | **12.721 / 12.596 s** |
| 后75时刻最小 / 最大 | 11.875 / 14.610 s |
| 几何更新均值（全76） | 0.061 s |
| 联合ROI规划均值（全76） | 0.893 s |
| cam2首调用 / 后75次均值 | 15.271 / 10.705 s |
| cam4首调用 / 后75次均值 | 1.520 / 0.676 s |
| CPU同步回调 / 函数体均值（全76） | 0.394 / 0.390 s |
| Blender进程wall，含消费 | **1012.541 s** |
| 打包 / 全帧统计及12次回读 / 视频 | 2.126 / 2.103 / 3.541 s |
| 收尾进程wall，含导入 | 11.196 s |
| **完整生成交付wall** | **1023.755 s，17.06分钟** |
| 追加统计 / 三张证据叠加图函数体 | 0.730 / 0.391 s |

CPU计划开发另列：首次错误gate检查19.355秒；同批候选重核及选中候选全groom预检90.098秒；14项定向测试最终0.237秒。它们不计入45分钟渲染＋必要消费预算。少量只读进度检查、文档和代码开发也不冒充GPU时间。相机调用包括render、compositor、写出和同步，不能拆成未经测量的GPU kernel时间；第一路仍承担动态场景同步成本。

实际ROI像素量：cam2每时刻150,544–872,356、均值359,252；cam4 148,225–859,329、均值353,978。两路总共54,205,488个目标分辨率渲染像素；场景始终完整参与光照/阴影/遮挡。无低清背景调用、无完整ERP RGB生产读写、无DS RGB往返。

GPU1 RTX3090，UUID `GPU-6f046d56-acbe-d49b-774a-9d463e768cac`；GPU0同时有他人任务，未使用或终止。GPU整卡1秒采样峰值4152MiB，包含已有桌面上下文；Blender RSS峰值9,829,128KiB，消费者790,192KiB，收尾753,096KiB。计算前后资源记录保留。临时磁盘采样峰值89,284,993B（可能漏掉瞬间峰值），成功清理原始缓冲1,435,328,759B；本run最终EXR为0。失败CPU计划和历史目录均保留。

152份ROI RGB共76,250,476B；含标签、共享mask、NPZ/旁表、视频、曲线、三张证据叠加图及日志的run目录92,873,152B（约88.57MiB，统计时刻口径）。全程未将数据移入datasets。

旧M01单人后续均值9.468秒，本批双人12.721秒，表面约增加34.4%；两段时长、轨迹、人物距离和ROI像素量不同，不能视为严格等负载性能对照，也不按人数线性推算。当前成本满足本轮45分钟预算，未展开其它性能优化。

## 后续边界

本轮M01双实例、三个seed、双相机和全部邻帧仍只有一个人物外观家族、一个原动作来源，不构成独立train/val；不能与共源F02拆成独立组。进入来源划分与小试训练前，仍需正式来源关系与整组隔离、更多已接受外观/独立动作、数据规模及事件配额、训练/评价预算和原权重基线方案。多人数/其它seed/模式覆盖及真实数据GT也未由这一episode认证。

本轮不接新资产、不运行检测/推理/训练、不扩大模式、不做性能参数搜索。无下载、安装、SHA256或commit，不移入datasets。TODO、A2、历史输出、原fixture及外部项目保留；运行期间生产代码与冻结副本一致，所有输入依赖的size/mtime未变。全局diff检查发现非本轮编辑的TODO尾随空白，未借机修改；本轮计划文件及新增代码定向检查通过。当前停在交付点。
