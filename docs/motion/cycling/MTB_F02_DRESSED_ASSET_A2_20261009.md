# F02 山地车连续穿衣资产（A2）

日期：2026-10-09。源身体120帧已获用户接受；本轮使用 **cycling_F02_v3** 自身女性体型、简单衣物和既有短波波头。全部 `integration_only`。目标穿衣动态观感待用户观看，不能继承为已接受。

[观看入口](../../../outputs/experiments/mtb_f02_dressed_asset_a2_20261009/index.html) · [MP4](../../../outputs/experiments/mtb_f02_dressed_asset_a2_20261009/preview/comparison.mp4) · [资产描述](../../../outputs/experiments/mtb_f02_dressed_asset_a2_20261009/delivery_manifest.json) · [A1适配器结果](../../../outputs/experiments/mtb_f02_dressed_asset_a2_20261009/a1_asset_adapter.json) · [已接受源身体](../../../outputs/experiments/mtb_video_smplx_pilot_a2_20261009/index.html)。

## 动作与时间冻结

唯一父动作是 `mtb_video_smplx_pilot_a2_20261009/fit_attempt_1/parameters_asset.npz`。源身体报告末尾追加了用户接受决定，交付状态更新为 `human_body_motion_accepted`；原检查和限制保持。没有重新运行SAM/4D Humans、拟合、平滑或修正头颈。

| 字段 | 实际交付 |
| --- | --- |
| source family | `bikefit_XBLMidr4dps`；Bike Fit Adviser，原演员身份unknown |
| source展示帧 / asset帧 | `[8662,8782)` / `[0,120)` |
| 原整数PTS | 8670662…8789781，步长1001，time_base=1/30000 |
| 速率与时长 | 30000/1001 fps；首末3.9706333秒；120帧展示4.004秒 |
| 旋转/表情 | global、body（含neck/head）、双手45维、expression10维、jaw/双眼均逐元素继承 |
| 采样策略 | 不插值、不复制、不循环、不变速、不预降至10Hz |

`timeline.json` 为A1已有字段：asset_frame、source_frame、source_time_tick、episode_time_tick、time_base；`time_mapping.csv`逐字节复制原文件。原相机到资产的 `diag(1,-1,-1)` 仅作为父动作已应用的来源记录保留，**本轮没有再次应用**；也不使用pred_cam_t或原视频路径作为道路轨迹。

## 女性模型与参数语义

使用F02原 `SMPLX_FEMALE.npz`、原固定30beta；不使用父male的10beta、不补零、不截断、不复制男性网格。标准female模板/R语义复用 [已有证据](../../../outputs/experiments/cycling_female_asset_a2_20261009/model_semantics.json)，没有改顶点或faces。两条求值接口均为 `use_pca=False, flat_hand_mean=True`，完整双手轴角45维、零pose mean；无需PCA转换。expression继承源10维，目标完整模型使用300:310表达方向。F02角色参考manifest的“zero expression”描述适用于原参考；本组合明确记录继承源expression的覆盖策略，未改角色原manifest。源jaw/eye默认零继续保留，隐藏头部仍为模型估计。

女性shaped-rest pelvis为 `[0.0005660951,-0.3464756012,0.0128384270] m`，新参数全段固定transl为其负值。source与目标有意变化仅为female模型、原30beta、常量transl及gender/root描述；其余源参数/时序数组逐元素一致。目标网格与源male不同是预期，并不以两者点位相等作为验收。

身体/衣物/R未来统一使用 `scene @ model`，groom使用 `scene @ head`。global orient已在几何与head中，未来scene只拥有道路行进、一次朝向/轴适配和固定放置高度。无逐帧min-Y贴地。

## 衣物与头发

F02_v3自身 `reference/simple_clothing_v1`：3616顶点、7058面，固定barycentric索引/权重、4mm法线offset、原材质及rest坐标。衣物为原程序化simple材质，不伪称独立服装UV或BEDLAM缓存；身体纹理/眼部材质继续由原appearance消费。全部帧固定拓扑，无裁面、重绑定或衣物求解。

头发真实依赖F02_v2的 `reference/groom_v5`，由F02_v3原manifest继承。原84,069曲线，固定active84,062曲线、1,890,420点；本轮不重筛，不保存120份完整头发。参考点空间为女性shaped-rest、pelvis-centered；每帧女性自身关节求得 `H=[Q, J_head(t)-Q J_head(rest)]`，保存120个小矩阵，root/global只进入一次。无男性head矩阵或毛发动力学。

## 技术检查与观感范围

[全120帧准备记录](../../../outputs/experiments/mtb_f02_dressed_asset_a2_20261009/validation.json) 和 [独立参数回读](../../../outputs/experiments/mtb_f02_dressed_asset_a2_20261009/readback.json)：源旋转/时间完整；目标固定shape、标准10475顶点/20908面、有限值通过；参数重新求值body最大误差0，R@body回读误差0。R来自项目固定算子，未从源观测点插值。

身体/衣物同帧同坐标；head刚性正交与scene复合检查通过，诊断矩阵下R变换交换误差3.58e-7m。全部毛发参考点逐帧变换有限，复合代数另用stride1024抽样。最大头皮锚点近似误差8.38mm，不宣称零穿插/物理毛发。最多27个局部衣面相对绑定基面法向反向，保留逐帧记录；这不是整网格相交认证，也不是自动否决条件。

探针asset `[0,47,59,78,119]` 于目标求值前按源数据选取：首、中、末及最大右膝屈曲/躯干前倾。实际看过10张图，未见衣物明显拉飞、头发脱头或坐标错位；高屈膝时远侧手臂与腿部有投影重叠，连续观感留给用户。补查asset16–19的两视角后，远侧手部在后方视角靠近髋/上腿，存在局部穿体可能，明确保留供用户判断（`knee_jump_context.jpg`），没有用衣物或姿态修正隐藏。原右膝约19.49°最大相邻变化、头部出画和远侧估计继承，不通过改姿态“修好”。未使用绑定修正次数。

沿用studio两固定方向、每方向512×512、Cycles64 samples、无降噪、seed29、非animated seed、AgX和原灯光，整段2.3m固定视野/中心。front/side是旧相机方向名称，本动作朝向不同，实际可见斜侧和后方，不冒称解剖正侧面。没有调亮度、材质、灯光或做DS/ROI渲染。

视频/成本最终结果见本报告末尾完成记录。静帧、解码与数值检查不是动态人工验收；本轮目标组合仍为 `pending_user`。

## A1交接与来源限制

只读实际调用 `dynamic_scene.adapt_proposal`、`select_timeline` 与 `geometry_paths`，已成功读取新manifest并核对120条原时间。新 `a1_asset_adapter.json` 使用已实现的 `densekp_scene_asset_v2`，不是另造场景系统；原A2 proposal格式用于来源描述，`public_consumer_tested=false`仍表示未做场景消费。[接口检查](../../../outputs/experiments/mtb_f02_dressed_asset_a2_20261009/a1_interface_check.json)明确区分适配器读取与场景验收。

当前已有可变原时序、scene-owned行进、固定骑行高度、simple材质、head复合及每人物时间映射能力；不继续把旧30Hz/stride3限制当现状。A1的 `person_time.py` 已提供显式nearest原观察映射及有理数误差记录，本轮没有为它采样或伪造episode时钟。之后多人物时间策略仍由A1明确选择。

身体全段model-Y最低点范围 `[-0.951465,-0.767322] m`。完整body/cloth范围见readback；固定放置高度留空，**不把M01的1.2m沿用为本角色已验收高度**。A1需要选择道路yaw/路径/支撑高度、适用时间策略，并在后续授权批次执行实际ROI、旁表和Dataset回读。当前没有相机图像、scene matrix、observations或packed rows。

F02与M01具有不同原有体型/外观，本次又使用与APD不同的视频动作来源，提供新的躯干前倾/手臂/踩踏组合；不声称准确恢复山地车接触或真实演员。F02-APD与F02-bikefit共享角色家族，所有同视频邻段/角色衍生继续归bikefit来源族。换动作不消除同角色泄漏；源license仍unknown，不因用户接受观感提升，全部仅内部integration_only。

## 可复现入口与保护

```bash
/home/yuyue/anaconda3/envs/camerahmr/bin/python tools/experiments/prepare_cycling_female_asset_a2.py --asset-root outputs/experiments/mtb_f02_dressed_asset_a2_20261009
/home/yuyue/anaconda3/envs/camerahmr/bin/python tools/validate/validate_mtb_f02_asset_a2.py
# Blender命令、GPU UUID、探针预算与成功帧复用见 frozen_recipe / probe_decision / render manifests。
/home/yuyue/anaconda3/envs/camerahmr/bin/python tools/visualize/package_mtb_f02_asset_a2.py
```

入口拒绝覆盖已存在成功输出，重放需新输出目录并更新引用，不能直接重跑覆盖本次证据。只参数化A2女性准备器、studio源帧标注和A2资产检查器；公共求值、binding、R与A1源码只读。未下载、安装、运行重建、训练、生成道路、多角色或计算SHA256/commit。源模型、原视频、已接受动作、历史预览、TODO和A1成果保留。

## 完成记录

完整120帧、240张视角图已完成（含复用10张探针）。MP4实际解码120帧，所有展示PTS相对原8670662的差逐项相等，精确时基1/30000，末帧展示结束4.004秒；没有只看容器fps。页面链接全部本地核对可用。源身体仍已接受；F02新比例、衣物和头发完整动态待用户观看，助理未实际连续播放验收。

CPU准备 13.229s，参数回读 1.912s；5帧探针 41.480s，余115帧 625.699s，渲染共 667.179s；打包/编码/逐帧解码 11.812s，五组合只读检查 0.457s。各阶段实测合计 **694.589s（11.58分钟）**；启动/导入及小型inline检查未逐项计量，预留30秒后仍低于30分钟。阅读、开发和等待时间未另行精确计时，不计作已验证计算成本。无失败渲染、无修正重跑。

实际设备 GPU0 / `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`，开始18MiB/0%，运行中显存快照1741MiB，结束18MiB；快照不是连续峰值。Blender完整阶段峰值RSS约2.61GiB（不是GPU显存）。另卡原进程保持，未干预。新增目录约 **85.57MiB**（最终小型JSON可能略增），包含原分视图/合成预览和必要缓存，未复制模型、原视频或120份groom，未删除旧产物。

[保护与执行记录](../../../outputs/experiments/mtb_f02_dressed_asset_a2_20261009/execution_record.json)确认源fit目录、TODO与快照内A1源码size/mtime未变。源报告追加接受记录，源manifest仅更新身体接受状态；修改前manifest保留。旧四组合条目逐值保留，只追加第五组合与相应计数；目录检查器在原四组合及新组合均通过。所有A1公共文件、原模型/绑定、已接受APD资产与外部pano-stitch未由本任务修改。

本轮停在穿衣动态评审。下一步待用户选择是否接受本F02组合，再由A1在单独授权任务中选时间/道路yaw/固定放置高度并实际消费公共ROI场景；本轮没有这些观测，也未启动其它motion盘点。

## 2026-10-09 用户后续接受记录

用户已观看本次 F02＋山地车源动作＋F02自身简单衣物和短波波头的完整120帧动态，并明确表示接受当前效果。目标穿衣观感状态更新为 `accepted_by_user`；已记录的局部穿插保留，不做修正。此决定仅覆盖当前组合的观感，不改变原视频许可unknown、真实人体精度、public_consumer_tested=false或正式数据集状态。
