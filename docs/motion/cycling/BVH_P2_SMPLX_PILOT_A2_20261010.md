# participant_2 直臂短段 → cycling_M01_v2：身体适配评审

**当前状态：120 帧身体参数、网格和 DenseKP 138 点数值通过；用户已观看目标身体预览并表示“效果可以”。** 这项接受仅针对 `fit_attempt_1` 的 M01 身体动态观感。没有制作衣物、groom、自行车或道路场景。

观看入口：[HTML 四栏对照](../../../outputs/experiments/bvh_p2_smplx_pilot_a2_20261010/preview/index.html)；[原速 MP4](../../../outputs/experiments/bvh_p2_smplx_pilot_a2_20261010/preview/comparison.mp4)。画面左右分别为原 BVH FK 骨架与拟合的 M01 身体，各有 X–Y 和 Z–Y 两个全段固定视角；画面显示 asset 帧、原始源帧和 BVH 标称时刻。页面支持逐帧查看。助理未连续人眼观看视频；用户在交付后完成了身体动态观感评审，接受记录单独保存，不改写原预览或数值证据。

## 冻结来源、模型与时间

- 来源为 [Kyriakou、Chrysanthou、Kyriakou 的 *Dataset of Motion Capture of Cyclists*，Zenodo 10668611](https://zenodo.org/records/10668611)，发布 session `2_RB_M27_20230809_PA`。本地官方元数据 [`zenodo_record.json`](../../../outputs/motion_inputs/cycling_fit_20260929/zenodo_record.json) 记录 `cc-by-4.0`；本试验及改动应保留上述署名。SMPL-X 模型及项目人物资产另有各自条款，不能由 BVH 许可代替。
- [冻结清单](../../../outputs/experiments/bvh_p2_smplx_pilot_a2_20261010/frozen_manifest.json) 在优化前确定原 BVH 帧 `[5700,5820)`，共 120 个相邻样本，asset 帧 `0…119`。原 BVH、本地截取 BVH、`clip_native.npz` 的前 120 行及 FK 均核对；FK 对既有缓存最大误差约 `9.95e-14` 原生单位。没有双加 `5700`。
- BVH 头的 `Frame Time` 原文为 `0.0333333` 秒，保存为整数 `333333/10000000`。首末标称跨度 `3.9666627` 秒；120 帧含末帧展示时长约 `3.999996` 秒。这只是发布 BVH 的标称采样，并非真实曝光 PTS。[逐帧时间表](../../../outputs/experiments/bvh_p2_smplx_pilot_a2_20261010/time_mapping.csv) 同时保存源帧、asset 帧和整数时刻；未插值、循环、变速或复制。
- 目标为 `cycling_M01_v2` 的标准 male SMPL-X 和原 30 维固定 betas。实际 faces 与 M01 reference 完全一致，10475 个顶点使用项目原始 `R[138,10475]`，并未从 BVH 关节推造 138 点。人物外观不是 BVH 演员的重建。

## 接入方法与权属

复用[原生 BVH 审计器](../../../tools/validate/audit_cycling_motion_library_a2.py)中的已验证解析/FK：每非末端关节的 3 个 position channel 已含局部位移，**替代** OFFSET；局部旋转按实际 `Zrotation → Xrotation → Yrotation` channel 顺序相乘。具名 [BVH→SMPL-X 映射](../../../outputs/experiments/bvh_p2_smplx_pilot_a2_20261010/joint_mapping.json) 覆盖 pelvis、脊柱、颈头、肩肘腕、髋膝踝和弱 toe 约束。原 Euler 角没有直接填入目标 body pose；由 FK 位置和骨段方向优化目标 SO(3) 旋转，膝弯曲限制在合理正范围，手指/脸/眼/表情为中性默认。头颈为弱位置约束，非精确头部恢复。

[对齐记录](../../../outputs/experiments/bvh_p2_smplx_pilot_a2_20261010/alignment.json) 给出一次固定的 proper yaw (`−0.04518 rad`) 和统一尺度 (`0.011036` 模型米/源单位)，尺度取八组主要骨段长度比的中位数。这是合成角色动作适配估计：源 `+Y` 上轴、厘米量级均未获官方单位/坐标确认，不宣称恢复真实米制、摄影机或绝对世界朝向。拟合目标是 `(BVH关节 − 同帧BVH Hips) × 一次固定轴变换 × 一次固定尺度`；原 root 数列保存在 `observations.npz`，不当道路速度。导出全段常量 `transl=−M01 shaped-rest pelvis`，骨盆原点固定，保留源的躯干/骨盆旋转变化；未来 scene 独占行进、道路 yaw 和一次固定高度。没有逐帧脚底/min-Y 贴地。

5 帧探针固定在 asset `0/30/60/90/119`，350 次迭代，实测拟合及检查 `2.37` 秒，保守全段估计 `85.2` 秒，小于 30 分钟上限。随后仅运行一次完整 120 帧、650 次迭代的拟合；没有换区间、调参网格或修正版。[适配脚本](../../../tools/experiments/fit_bvh_p2_smplx_a2.py) 和 [独立回读脚本](../../../tools/validate/check_bvh_p2_smplx_a2.py) 均只写本试验目录。

## 实际检查

全 120 帧固定 shape、拓扑、源/目标时间轴与有限值通过。[拟合统计](../../../outputs/experiments/bvh_p2_smplx_pilot_a2_20261010/fit_attempt_1/validation.json) 的具名关节位置残差平均 `0.0440 m`、P95 `0.1261 m`、最大 `0.1485 m`；这是 BVH 骨架到不同体型模型的映射残差，**不是现实 3D 精度**。所有相邻局部旋转最大 `11.75°`，没有超过脚本记录阈值 `25°` 的孤立步；顶点相邻位移 P95 `0.0410 m`、最大 `0.0626 m`。这些数值不能代替连续人体观感。

[独立 120 帧回读](../../../outputs/experiments/bvh_p2_smplx_pilot_a2_20261010/readback_validation.json) 对保存参数重新求值，身体/关节和 `R@body` 与缓存全段吻合（记录精度内最大差为 0）；骨盆中心最大误差为 0。原骨架左右肘内角中位数约 `171.7°/172.5°`，目标约 `169.9°/171.8°`，较直手臂基本保留。原对齐骨架胸段的中位相对位移约 `[0.006,0.342,0.150] m`，目标对应段约 `[0.004,0.289,0.123] m`，较立但略前倾的姿态保留，胸段长度差反映固定 M01 体型/关节定义。目标左右踝高度变化约 `0.345/0.362 m`，相关系数 `−0.993`，交替踩踏未被时间正则抹平。没有精准车把/踏板接触认证，也没有自相交零误差要求。

MP4 使用 `10000000/333333` 的标称帧率、容器 time base `1/10000000`；解码回读 120 帧，最大容器 PTS 与理想有理数时刻偏差 `0.5 µs`，这只是封装刻度误差。两个视角使用相同的一次对齐和固定画面尺度，不逐帧缩放。首轮图片生成成功，但编码因 `1600×535` 奇数高度失败；零字节失败文件与 [失败记录](../../../outputs/experiments/bvh_p2_smplx_pilot_a2_20261010/preview/failure_record.json) 留存。随后只对现有图片编码时补一行像素，**未重算身体或重渲图片**。

## 资源、交付与下一步

计算前记录两张 RTX 3090 的 UUID、显存/进程、内存与磁盘，实际全程仅用 CPU，GPU UUID 为 `null`。探针约 `2.37` 秒，完整拟合/导出约 `5.61` 秒，独立回读约 `1.86` 秒；120 帧预览图片生成按首末文件时间约 `25` 秒，二次编码/PTS 核查 `1.51` 秒，失败编码耗时未单独精确计时。合计远低于 30 分钟。试验目录约 `26 MiB`，没有复制模型、原 BVH 全文件或其它角色资产。

[机器交付清单](../../../outputs/experiments/bvh_p2_smplx_pilot_a2_20261010/delivery_manifest.json) 以 `fit_attempt_1/parameters.npz` 为权威，身体、关节、138 点缓存仅供回读和预览。所有结果为 `integration_only`：来源组 `zenodo_10668611_participant_2`，角色家族 `cycling_M01` 也已用于 APD，不能据此自动拆独立 train/val。用户已表示目标身体“效果可以”，见[接受记录](../../../outputs/experiments/bvh_p2_smplx_pilot_a2_20261010/body_acceptance_20261010.json)。后续可单独评估 M01 简单衣物/groom 绑定和公共场景时间消费；participant_1 两段仅可复用这一适配思路，仍需各自解析/模型/观感验证。此接受不改变拟合残差、真实精度、许可、穿衣或场景状态。

复现命令（现有环境，无下载/安装）：

```bash
/home/yuyue/anaconda3/envs/camerahmr/bin/python tools/experiments/fit_bvh_p2_smplx_a2.py --stage probe
/home/yuyue/anaconda3/envs/camerahmr/bin/python tools/experiments/fit_bvh_p2_smplx_a2.py --stage fit
/home/yuyue/anaconda3/envs/camerahmr/bin/python tools/validate/check_bvh_p2_smplx_a2.py
/home/yuyue/anaconda3/envs/camerahmr/bin/python tools/visualize/preview_bvh_p2_smplx_a2.py
```

脚本默认拒绝覆盖已有拟合阶段；复现须传新的独立 `--root`，并先复制/生成相同冻结清单。预览脚本针对本次目录，已存在的帧图会复用；不得把上述命令直接当成可覆盖历史产物的批处理步骤。
