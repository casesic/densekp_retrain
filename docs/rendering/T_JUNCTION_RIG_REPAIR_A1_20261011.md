# T 形路口三组机位修复：搜索结果与生产决定

日期：2026-10-11。**本批未生成新视频或训练行。** 三人路口场景继续使用[上一批已核验的轨迹与原生路面](../../outputs/experiments/t_junction_multirig_a1_20261011/)，人物、动作缓存、31条原时间映射、世界矩阵和 R138 均未改。旧48个机位的拒绝记录也未改。本批扩大了有界 CPU 搜索，并把组级“至少一组双边缘”与各组的正／背／侧视角、尺度分别筛查；实际搜索后仍没有同时满足相机与人物净空的双边缘机位。因此依硬门槛停止，未启动 Cycles 或多rig生产。机器决定见[execution_decision.json](../../outputs/experiments/t_junction_rig_repair_a1_20261011/execution_decision.json)。

## 本批修复及实测

[新配置](../../configs/t_junction_rig_repair_a1_20261011.json)冻结相机 seed 20261042、道路周围位置网格、每组最多16个位置、15°全周粗 yaw 和围绕粗最佳 yaw 的±2/4/6/8°细化。位置先依据 p1 每帧真实身体前向筛选观察类别；yaw 阶段对同一31帧使用实际绑定双机、physical mask 和 R138 投影。每一个全段投影候选均记账，未把内部 yaw 试算藏在限额外。目标人始终是 p1；front/back/side各自要求连续6帧双机同类视角及≥8共同 physical 点，较小一路的原 R138 投影高度中位数分别≥280/180/180 px。组级双边缘仍要求同一组连续6帧、两路各≥8域内和≥5域外、共同≥8点。角度窗口与边缘窗口没有强制重叠。完整人物 crop、接缝、≤700万像素、场景障碍和人体净空仍是后续精确条件。

| 组别 | 完整31帧投影候选 | 视角＋尺度＋人物净空合格 | 同时具备连续6帧双边缘 | 最长原始双边缘窗口 |
| --- | ---: | ---: | ---: | ---: |
| 背面 | 512 | 90 | 0 | 12帧，但该机位不满足人物净空 |
| 正面 | 512 | 23 | 0 | 7帧，但该机位不满足人物净空 |
| 侧面 | 352 | 23 | 0 | 0帧 |

原始[light_summary.json](../../outputs/experiments/t_junction_rig_repair_a1_20261011/light_summary.json)保留粗筛当时的结果；按**组内两台相机的真实旋转后中心**和全部三人31帧根点重新核算的[corrected_light_summary.json](../../outputs/experiments/t_junction_rig_repair_a1_20261011/corrected_light_summary.json)是最终资格依据。最接近的背面候选 attempt152 具有19帧视角／共同窗口、12帧双边缘和较小路 R138 高度中位602px，但两机到三人根点的最小距离只有0.473m，低于既有0.5m硬线；同位置的其余四个背面边缘候选为0.466–0.471m。正面四个边缘候选为0.477或0.490m，也均低于0.5m。侧面有合格视角和尺度，但本批候选没有连续边缘窗口。不能因为差几厘米便降低净空线，也不能将侧面候选改名为双边缘。

轻量筛选第一版漏掉了 yaw 旋转所改变的 cam2/cam4 中心与人物根点的最小距离，致使背面 attempt152/153/154 被送入下一阶段，均在第8帧的廉价根距离门槛拒绝；没有进入 body/cloth/active groom 精确 ROI 计算。这三次失败和原始搜索日志保留。[修正后的搜索入口](../../tools/experiments/repair_t_junction_rigs_a1.py)在 light 阶段计算两真实光心的全段净空；定向检查确认旧背面默认仍要求边缘，显式本批配置仅解除“背面单独承担边缘”的限制，视角和其它硬条件未被跳过。九个冻结备选光心的原生 Blender 水平短射线[静态净空检查](../../outputs/experiments/t_junction_rig_repair_a1_20261011/static_clearance.json)没有命中场景障碍；这不抵消人物动态净空失败，也不是连续碰撞证明。

最终背面与正面各已用满512个全段投影候选，且两组在净空纠正后均无合格双边缘项；侧面只用了352个，因为继续评估也无法让已用尽的正／背候选产生组级合格解。按冻结预算及最多3个精确候选／组的范围停止，不追加机位、改人物轨迹或渲染一组三机位不合格的替代视频。exact_summary.json保留上述三次早期拒绝；本批**没有**精确支持通过的 prepared rig，故没有 rig list、186次ROI、558个生产槽、NPZ、Dataset回读或视频。这里的0是生产数量，不代表无框或零定位误差。

## 成本、范围与重放

CPU完整投影搜索约20.19秒；三次早期精确入口拒绝合计0.74秒；原生 Blender 静态场景仅供相机短射线检查，未进行正式场景装配或相机渲染。三项新定向测试及原 T 路口5项回归通过。pytest在当前解释器未安装，因此定向测试直接调用、历史用例用unittest运行；未安装依赖。GPU开算前检查记录 GPU0 UUID GPU-d481acc7-106f-29e5-1298-8c0d398261b2，空闲24144MiB，系统可用内存约139GiB、磁盘余量约12TB；本批无GPU渲染。任务目录约30MiB，主要为逐候选31帧投影记录。20分钟CPU上限与60分钟生产额度均未耗尽；**候选和联合几何门槛**而非计算预算阻止了渲染。[资源记录](../../outputs/experiments/t_junction_rig_repair_a1_20261011/resources_record.json)保留；后续从 Python 子进程调用 NVML 在沙箱内返回错误，未据此宣称GPU占用变化。

从仓库根目录复现同一候选与筛选规则时须使用新的空输出目录，不覆盖本批或旧批证据：

    PYTHONPATH=src python tools/experiments/repair_t_junction_rigs_a1.py --output /tmp/t_junction_rig_repair_replay
    PYTHONPATH=src python -m unittest tests.test_t_junction_a1 -q

当前交付的是可重放搜索与明确不可生产结论，不是新的路口观感验收。要再试机位，需新批次明确新的候选范围和上限，并优先将**两光心到全段所有人物的净空**置于 yaw 全段投影之前，同时对已有近临界机位做有界局部位置／yaw联合构造；不能把本批已拒绝机位直接交给 renderer。全部数据仍为 integration_only/shared_source_diagnostic，未修改 A2、旧资产、源 .blend、TODO 或正式来源划分。
