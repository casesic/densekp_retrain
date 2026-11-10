# F02 连续骑行资产与预览

日期：2026-10-09。**实际采用 cycling_F02_v3，完整76帧身体、简单衣物及原有short-bob头发已完成求值、检查与双固定视角预览；动态观感待用户观看。** F02资源齐全，没有启用F01，没有扩展其它角色或动作。

入口：[连续播放及逐帧页](../../../outputs/experiments/cycling_female_asset_a2_20261009/preview/index.html)、[MP4](../../../outputs/experiments/cycling_female_asset_a2_20261009/preview/comparison.mp4)、[资产manifest](../../../outputs/experiments/cycling_female_asset_a2_20261009/delivery_manifest.json)、[76帧数值记录](../../../outputs/experiments/cycling_female_asset_a2_20261009/validation.json)、[原时间映射](../../../outputs/experiments/cycling_female_asset_a2_20261009/time_mapping.csv)。全部integration_only。

## 固定输入与派生关系

原动作权威仍为 `apd_6526_smplx_pilot_20261008/fit_attempt_2/parameters.npz`，APD6526源帧30…105。所有global/body/手/脸/眼部旋转、局部旋转矩阵、源帧和原时间数组逐元素保留；没有重拟合、平滑、变速、循环、插帧或换段。新增目标女性不代表原APD演员是女性，也不是一个新独立动作来源。

使用F02原有female模型与固定30beta，模型文件来自已有 `cycling_commuter_pilot_20260930/models/SMPLX_FEMALE.npz`。本轮重新对照标准female文件：v_template、faces、LBS weights、posedirs、J_regressor、kinematic table与前10个shape方向完全相同；30beta reference重新生成误差0。结合原完整模型回归记录，R仍适用于该标准female顶点体系，未换成男性网格/shape，未进行头部拼接或拓扑编辑。证据：[model_semantics.json](../../../outputs/experiments/cycling_female_asset_a2_20261009/model_semantics.json)。

女性自身未平移rest pelvis为 `[0.0005660951,-0.3464756012,0.0128384270] m`；全段使用一次常量平移 `[-0.0005660951,0.3464756012,-0.0128384270] m`。它不同于M01原平移 `[-0.0009769164,0.3770247698,-0.0096359504]`。76帧pelvis保持模型原点，没有每帧min-Y贴地。源global_orient只通过模型求值进入身体及head一次；未来行进和固定放置高度由场景负责。

新 `parameters.npz` 为目标参数权威，关联原动作权威，不覆盖男性参数。模型仍是30个真实shape方向、默认零expression（原动作相应数组不变），不是把16维BEDLAM形状搬到女性。字段说明见 `parameter_semantics.json`。

## 衣物与头发

直接复用F02_v3的 `reference/simple_clothing_v1`：3616顶点、7058面、固定body barycentric indices/weights、4mm法线offset和既有GarmentRest材质；reference回读误差0。没有挪用M01缓存、BEDLAM缓存、重新制作衣服或逐帧裁面。

F02_v3正式reference引用其父F02_v2的 `reference/groom_v5` 和 `groom_binding_v1`；这不是路径错误或换用别人的头发。女性同shape/reference身体重算完全一致。原84,069条曲线中固定active为84,062条、1,890,420点；继承原静态有效集合，不新删毛、每帧重筛或复制76份毛发缓存。

头发参考点已在女性shaped-rest、pelvis-centered模型坐标中。按女性关节求得 `H=[Q, J_head(t)-Q J_head(rest)]`，Q来自原动作旋转链；不复用M01头矩阵。全段固定参考点＋76个小矩阵，身体/衣物/R使用scene，groom使用 `scene @ H`。原root/global orient已在H中，不再次叠加。最大头皮锚点近似误差4.082mm，记录为刚性附着诊断，不宣称毛发物理或零穿插。

## 检查与实际预览

全76帧：固定shape/拓扑、有限值、原姿态/时间数组一致、参数回读、身体与衣物的共同时刻、head正交性/复合变换、R138回读均通过；R缓存回读最大差0。资产侧只读检查器另做共同固定诊断矩阵的R交换性和groom复合检查，没有伪造道路轨迹或场景观测。

最多32个局部衣面相对绑定基面的法向反向，最坏帧这些局部面的总面积约0.000923m²；无退化面。这不是完整人体相交证明，也不拿零交叉数或统一毫米阈值否决。已检查探针正/侧画面，未发现明显拉飞、脱体或错坐标；轻微穿插、小褶皱及原动作左臂快摆仍保留。静态检查不能替代动态人眼验收。

先渲染asset `[0,6,16,38,75]`（源 `[30,36,46,68,105]`），覆盖快摆及首末。探针没有明确接入错误，实测后估计完整补齐421.862秒，继续生成其余71个时刻，复用10张探针图。每方向512×512、Cycles64 samples、无降噪、seed29、AgX、既有studio灯光、两固定相机、整段共同2.3m视野，无跟踪缩放或地面lift。

APD整数时刻首末跨度3秒、间隔39949…40010 ticks。time_base为按APD25Hz推断的1/1,000,000秒，仍标记单位推断。复用既有 `settb/setpts` 编码方案：MP4包含76帧，PTS与源时刻差 **0秒**，stream time_base=1/1,000,000，时长3.040000秒（末帧展示40ms，不新增观察）。没有假装30Hz或重采样到10Hz。页面提供实际时间播放和逐帧定位。

## 成本、保护与停止点

CPU准备实测9.077秒；5帧探针42.469秒，完整补齐392.838秒，渲染累计435.306秒（约7.26分钟），低于30分钟；打包/编码8.867秒。没有失败求值、失败渲染、绑定修正版或新衣物迭代。图形准备前GPU0已恢复18MiB、0%空闲，实际UUID为 `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`；最初两卡忙时只做CPU，没有干预其它进程。

原模型、F02参考、M01参数/接受资产、BEDLAM原资产、TODO、A1文件均未由本任务修改。A1并行更新的ROI报告/源码只记录快照差异，不回退。新增存储、保护清单和本轮完整验收在 `delivery_validation.json`；本轮没有下载、安装、SHA256、commit、训练或DS/ERP场景渲染。

结束存储快照：女性产物约56.39MiB，BEDLAM语义审计约10.91MiB，小型资产目录约0.052MiB；不含符号链接重复计数，不复制整套模型或76份groom。模型属性CPU导出耗时未单独计量，其余实测计算分别记录，不把阅读和等待时间填成已验证成本。

用户待决定：这名女性的完整穿衣骑行动作是否看得过去。接受前保持候选状态；不因旧F02静态通过或男性动态通过而自动接受。M01/F02同属APD动作来源组，换目标女性只增加人物外观多样性，不能将男女版本拆成独立train/val。

生成器：[prepare_cycling_female_asset_a2.py](../../../tools/experiments/prepare_cycling_female_asset_a2.py)；预览/打包沿用A2 studio入口新增显式 `--asset-root`，不修改公共材质、Dataset或A1生成器。下一批公共消费条件见 [动态资产交接](../../data/DYNAMIC_ASSET_HANDOFF_A2_20261009.md)。
