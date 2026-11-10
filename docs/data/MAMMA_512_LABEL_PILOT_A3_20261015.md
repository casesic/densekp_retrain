# MAMMA 512 标签与矩形裁剪小试 A3

日期：2026-10-15。状态：小试完成；visibility 训练资格待补代表性负例。

## 恢复状态

- 任务 ID：`mamma512_a3_20261015`；输出根：`outputs/experiments/mamma_512_label_pilot_a3_20261015/`。
- 范围冻结为 first-four e03、`rig_back`、源帧 0/15/30、cam2/cam4、M02/p1_alternate 与 M03/KIT11；开始生产前仍需逐文件核对人物和时间映射。
- 原 R138 `annotations.npz` 保持原样；MAMMA 512 写独立 `mamma_512.npz`、证据和 observation manifest。
- GPU 0 在首次核查时空闲；GPU 1 有两条非本任务 Blender 进程。本批不停止或争用它们。
- 旧 A3/其它执行耗时不重置。当前新任务已确认 GPU 调用 0；历史相关计算成本不计为 0，无法归属部分记 unknown。

## 当前核查

- A1 只读 reader 已识别 `mamma_observation_v1`，模板为 512×10475 one-hot 行序；三份资格身体缓存已登记。
- 当前 reader 的 `coverage_crop` 来源仍是局部 valid mask，因此不能据此宣称 render coverage 已与 physical mask 分离。本批 manifest 将显式提供两者。
- 当前公共 R138 可见性和遮挡证据存在 138 固定形状，需要增加通用点数接口并保留旧包装行为。
- 当前服务与历史产物不作为本批修改目标；新 probe 使用独立目录。

## 计划与预算

1. 冻结并核对 e03 输入、模板和资产资格；实现通用投影/证据及七框规划。
2. 运行定向 CPU 测试和 R138 冻结对比。
3. 再次核查资源后，最多执行 3 时刻×1 rig×2 相机=6 次正常 RGB ROI 调用；仅一次定向修复重试，总上限 8。
4. 输出稀疏 observation manifest、点级检查页、A1 handoff、成本与清理记录。稀疏帧不称连续 episode。

## 已知问题

- 官方 `vertex_visibility` 的完整生成细节尚未核证，首版策略须标记 `compatible_definition_pending`。

## 实现里程碑：渲染前冻结

- 已核对 e03 两人：P1 为 M02/p1_alternate，P2 为 M03/KIT11；frame 0/15/30 的资产帧分别为 P1 8/44/80、P2 9/54/99，源帧为 P1 9608/9644/9680、P2 9/54/99。
- 新增任意点数与显式 point→person 映射的 ERP range/instance 可见性及遮挡证据接口；旧 R138 包装保持 138 分块、5×5、4 cm、水平周期和 floor 取整策略。
- 新增 MAMMA 512 模板校验、准确资产帧世界点、身体 BVH 等价的有界三角形射线核对、保守 known/unknown 策略，以及固定七框支持计算。contact 与 floor contact 固定 unknown/valid=false。
- 定向 CPU 测试通过：任意点映射与独立旧规则参考一致；旧遮挡字段形状一致；七框支持/拒绝；已知几何的前后表面射线区分。当前环境没有 pytest 模块，直接调用四个测试函数并另做语法检查。
- 旧 e03 实际 ROI 只读预检：12 个人物槽中 10 个有框，70/70 个七框变体有完整采样矩形支持；2 个无框槽保留 absent。新任务会按新实际 mask 再验。
- 已冻结独立 `run/`：三个时刻、一个 `rig_back`、两相机，计划 6 次调用，预算 1800 秒。512 世界点 shape 为 `[2,512,3]`；三个时刻抽查 P1 身体射线均 512/512 可解析。

## 实际执行范围

- 输入固定为 e03、`rig_back`、原 frame 0/15/30、cam2/cam4。P1 是 M02/p1_alternate 骑行者，P2 是 M03/KIT11 行人。人物、动作、世界矩阵、标定、灯光、64 samples 和逻辑 ERP 5760×2880 均沿用冻结源配方。
- 实际运行 6 次 RGB ROI 相机调用，重试 0 次；一个 Blender 进程、一次场景装配、三个完整人物状态更新和一个常驻消费者。没有生成连续 episode 或视频。
- GPU 为 `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`。ROI 纯渲染累计 79.070 秒，Blender 总墙钟 138.346 秒；消费 13.005 秒，其中 MAMMA 512 投影、证据和身体射线 7.384 秒，源缓冲读取 0.779 秒、合成 7.543 秒、保存 2.952 秒。消费子阶段有重叠，不能相加当作墙钟。
- 计算过程丢弃本任务临时缓冲 477,023,413 字节。512 文件和 manifest 回读后清理；现存 EXR 0、保留原始校准缓冲 0、失败渲染文件 0。

## 公共实现与兼容

- `camera/visibility.py` 提供显式 point→person 的任意点数接口；旧 `erp_visibility` 仍使用每人 138 点、5×5 邻域、严格 4 cm、floor 像素和 ERP 水平周期。
- `data/occlusion_events.py` 提供任意扁平点证据接口，旧 R138 包装的 shape 和默认值保持。
- `data/mamma_labels.py` 校验模板 512 行顺序，从准确资产帧只应用一次 `scene_from_model`，生成世界点、完整 ERP 坐标、physical、coverage、原始 range/instance 邻域证据和保守 visibility。
- `roi_consumer.py` 在深度/实例临时缓冲清理前写出 `mamma_512.npz`、`mamma_box_support.json` 和 `label_policy.json`。
- `run_multirig_roi_a1.py --label-probe` 只接受一个 rig 和最多三个显式帧，不走连续序列 pack。旧 preview 单帧和完整 31 帧检查保留。
- 正式序列打包对 MAMMA 侧文件是可选的。旧目录不新增字段；新目录必须同时存在标签、七框表和策略，并通过人物顺序、512 shape、帧及相机校验后，观测才获得 `mamma_512_ref` 等引用。

## 512 坐标、R138 与观察槽结果

- 12 个人物观察槽中 10 个有真实实例框，2 个为 P2 在 frame 0/15 的 cam4 域外/无框槽，原样保留 `person_absent`，没有补抽。
- 所有相机文件均为 `[2,512,3]` 世界点、`[2,512,2]` 完整 ERP 坐标及对应 `[2,512]` 证据；人物 ID、整数源帧、源时间和 scene matrix 引用通过自动回读，数值有限。
- 与旧 e03 的六个 `annotations.npz` 独立比较，每个文件原有 26 个 R138 字段逐数组完全相等，差异字段为 0。
- `physical_domain`、`render_coverage` 和 `visibility_known/value` 分开存储。cam4 的部分点物理域外，但 ROI 数值缓冲仍覆盖其位置；物理域外不被解释为遮挡。
- contact 和 floor contact 全部为 `value=-1, known=false, valid=false`，本批未计算人体或地面接触。

## Visibility 证据与资格

- 6144 个点观察中，known positive 1137、known negative 0、unknown 5007。固定场景没有出现满足保守判定的另一人物/场景前方遮挡负例，因此 visibility 训练资格为 `pending_representative_negative_evidence`。
- 正例要求冻结身体三角形首交点与目标点相差不超过 3 mm、实际渲染为同人物表面且 range 相差不超过 15 mm、3×3 证据完整。负例要求另一注册人物或场景在前方超过 4 cm，并有中心像素和至少 5/9 邻域一致。
- 本人物实例前景、衣物或头发厚度、轮廓 ID 混杂、自遮挡未解和邻域不完整均保留 unknown。同人物 ID 不直接当作身体点可见。
- 固定点级检查最多 24 例，覆盖正例、本人物歧义、轮廓、物理域外和本地模板身体部位。该配方缺少另一人物与场景高可信负例，未额外摆场景。
- 官方 `vertex_visibility` 生产细节仍未核证，策略名明确包含 `compatible_definition_pending`；15 mm 外观表面容差需要 A1 在扩大训练前复审。

## 七框矩形支持

- 每个真实框独立检查 native、中心 x±0.03w、中心 y±0.03h、中心固定同比 0.9/1.1，共七种。随后应用 1.2 扩张、384:512 宽高比、部署 Gaussian 和插值 halo。
- 10 个有框槽的 70/70 个变体均被实际渲染区域完整支持，拒绝 0；两个无框槽对应 14 个变体未测试。结果只证明这七个离散框，不代表连续随机框族。
- 原几何保守 ROI 已覆盖七框，本次计划区域和实际区域没有因为 MAMMA 再扩大，也没有两遍 RGB 试探。六次相机调用合计由 18,217,913 像素保持为 18,217,913 像素，新增 0；逐相机坐标见 `roi_region_comparison.json`。

## 回读与定向测试

- 五项定向 CPU 测试通过：任意点数与独立旧 R138 参考、旧遮挡包装、七框通过/拒绝、已知网格前后表面射线、正式 pack 的可选 MAMMA 引用及旧目录兼容。语法检查通过。环境没有 pytest，因此使用相同测试函数直接调用。
- A1 正常包导入会触发当前环境缺失的可选 `wandb`；未安装依赖。验证脚本只读加载 A1 的 `MammaManifestDataset` 模块并实际回读固定前 6 个有框观察，图像 `[3,512,384]`、坐标 `[512,2]`、有限性和源采样支持均通过。
- 输出共 141 个文件、44,249,893 字节（约 42.2 MiB）；该数值在最终状态记录写入后以文件系统复核。

## 产物入口

- 稀疏 manifest：`outputs/experiments/mamma_512_label_pilot_a3_20261015/probe_manifest.json`
- A1 契约：`outputs/experiments/mamma_512_label_pilot_a3_20261015/a1_handoff.json`
- A1 reader 配置与实读结果：`a1_reader_config.json`、`a1_reader_readback.json`
- 静帧检查页：`point_checks.html`；绿点是 known visible，黄点是 unknown，红点预留给 known occluded。页面仅含三个源时刻，不能按连续动作解释。
- 汇总：`acceptance.json`、`execution.json`、`visibility_summary.json`、`seven_box_support.json`、`roi_region_comparison.json`、`r138_regression.json`、`point_checks.json`。

## 复现命令

从冻结 first-four manifest 新建独立输出并执行相同六次调用：

```bash
python tools/experiments/run_multirig_roi_a1.py \
  --output <new-output>/run \
  --gpu-uuid GPU-d481acc7-106f-29e5-1298-8c0d398261b2 \
  --budget-seconds 1800 \
  --rig-list-manifest outputs/experiments/first_four_episodes_a1_20261011/e03/rig_list_manifest.json \
  --frame-index 0 --frame-index 15 --frame-index 30 \
  --rig-id rig_back --label-probe
```

当前交付的纯 CPU 收尾和 A1 只读回读：

```bash
python tools/experiments/finalize_mamma_512_label_probe_a3.py \
  --output outputs/experiments/mamma_512_label_pilot_a3_20261015
python tools/experiments/validate_mamma_512_handoff_a3.py \
  --config outputs/experiments/mamma_512_label_pilot_a3_20261015/a1_reader_config.json \
  --output outputs/experiments/mamma_512_label_pilot_a3_20261015/a1_reader_readback.json \
  --limit 6
```

本批停止于稀疏标签与文件契约交接，不启动整段生产、接触标签、模型前向或训练。
