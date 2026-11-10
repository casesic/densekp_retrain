# A1：短真实渲染验证——从当前准备目录到 MAMMA 读取

日期：2026-10-23。工作目录：`/mnt/data1/yy/home/yy/Projects`。

## 目标与执行边界

生成器结构整理已完成。本轮不继续做大重构，目标是用一个很小的任务验证真实 Blender 装配、GPU 渲染、CPU 消费、标签导出和 MAMMA Dataset 读取是否连通，并检查新组件组合的实际画面。

最新已记录提交：human_data_generator `b446040`。请先核对实际工作区与用户最新指令，保留其他人的修改。README 已按用户要求改为使用手册，不改回交接日志风格。

**用户已明确授权本次真实 GPU 渲染（原话：“GPU也许可啊”）。**请直接完成 CPU 准备并执行下述小任务，无需再次请求渲染许可或要求用户指定 GPU。自行选择通过现有空闲检查的 GPU UUID，采用下述累计 1800 秒执行预算。此授权覆盖本任务必要的 Blender、GPU 渲染及实际 CPU consumer 子进程；不覆盖训练、模型推理或扩大生产规模。

不要构造或运行 MAMMA/DenseKP/SAM 网络，不训练、不评价模型、不修改训练执行许可。允许 CPU SMPL-X 几何准备和 Dataset 读图。不要重启编辑器，不操作其他任务的进程，不下载/安装依赖，不改上游代码，不自动提交或推送 Git。

## 1. 只做一个小任务

建议使用现有城市双人规则作为起点，冻结成当前 human_scene_v2 场景：
- 两个人，独立 person_id，每个人都是 target。
- 一组已知可用机位，cam2/cam4 均执行。
- 仅选择两个帧，例如原场景的 frame 0 和 frame 5，不跑完整序列。
- 至少一人使用新组件系统支持的跨身体衣物或发型组合，优先复用前轮已经 CPU 验证的资源与缓存。另一人可用已知组合。
- 正常标签任务，不加 --preview。预期 2 帧 × 1 rig × 2 相机 = 4 次相机渲染、8 个 target 观察槽；observed/absent 按真实结果统计，不强行全可见。
- 延用正常分辨率、标定、样本数、ROI、mask 裁剪与标签算法，不为让验证通过私自降低配置或改标签。
- 不制作视频，不扩展机位或场景数量，不追求本轮覆盖所有组件组合。

路线和相机优先复用现有规则，CPU 检查不通过时先在准备阶段修正具体原因；不要进行大量随机搜索。正常输出放 `outputs/runs/<新的任务名>/`，场景放 `outputs/scenes/`。历史失败现场与原输出不覆盖。

## 2. 准备具体可执行结果

阅读 README、PREPARED_LAYOUT_SIMPLIFICATION_A1_20261023.md 和 COMPONENT_SELECTION_A1_20261023.md，核对当前 run CLI。

使用正常 plan/prepare 路径，确认：
- 新准备结构：根目录四文件、每 rig 两文件，无 job.json/multirig.json/camera_configs。
- run_config 中选定帧与 rig 顺序正确，组件选择确实进入 asset descriptor。
- mask/标签所需的 ROI 留边与相机物理域没有明显冲突。
- 外部场景、组件缓存、标定、模板及可执行文件可访问。

命令形式如下，执行时替换为本轮实际路径：

```bash
cd /mnt/data1/yy/home/yy/Projects/human_data_generator
export PYTHONPATH="$PWD/src"
GEN_PY=/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python

"$GEN_PY" -m human_data_generator.run plan --scene <本轮场景.json> --frame 0 --frame 5
"$GEN_PY" -m human_data_generator.run prepare --scene <本轮场景.json> --name <新任务名> --frame 0 --frame 5

# 用户已授权本轮执行，准备成功并检查设备后直接运行：
"$GEN_PY" -m human_data_generator.run run-prepared \
  --root outputs/runs/<新任务名> \
  --gpu-uuid <实际空闲GPU的UUID> --budget-seconds 1800
```

只读查看 GPU 状态并选择符合现有程序空闲检查的设备，执行前重新检查；没有空闲 GPU 就报告当前情况，不抢占别人的资源。本任务真实执行累计预算为 1800 秒；若用户后续指定其他预算或设备，以其最新要求为准。不要通过改任务名、重启进程或新增尝试重置累计预算。

## 3. 授权后的运行与故障处理

走正常 run-prepared 入口，让 Blender 和实际持久 CPU consumer 子进程执行；不要用 mock 或手工造成功标记代替真实链路。

记录实际相机调用次数、用时、输出路径和失败阶段。遇到代码错误，先保留日志及现场，定位具体原因，做最小修复和必要 CPU/mock 回归。失败后先定位修复；允许在同一累计 1800 秒预算内做一次修复后的重试，使用新任务目录且保留失败现场。不得扩大场景、帧数或预算，超出范围时交付诊断与修复结果。不要把架构重构作为默认故障处理方式。

结束时核对自己启动的 Blender/consumer 均已退出；只处理本任务子进程，不通过宽泛 pkill 或端口清理影响其他工作。

## 4. 必须查看真实产物

成功时检查：
- 实际 ROI RGB，人物外观、衣物材质、发型、位置和两帧运动是否明显异常；制作一张简单联系图或可查看的图片索引即可，不生成大批展示材料。
- 逐人 mask 与 RGB 的人物归属是否一致；不能只验证文件存在。
- 每个目标的 512 点顺序、二维坐标、身份/时间、坐标有效性和 visibility 的取值；叠加图要将完整 ERP 坐标正确转换为 ROI 坐标。
- observations.json 为 mamma_bundle_v1，8 槽对应实际帧/机位/相机/人物，缺 mask 按 absent 记录；不漏目标、不伪造负例。
- 记录 visibility 1/0/-1 数量。出现 0 也不自动等于可信遮挡标签合格；没有可信负例也不应隐藏或改数值。本轮目标是验证渲染与数据链路，不以“负例必须大于零”强行判定运行成败。
- 正常数据完成标记、消费日志、预算与临时缓冲清理情况；失败现场不清除。

CPU 使用 mamma_retrain 正常 check-data 入口及 --manifest 指向新 observations.json，读取最多 2 个 observed 样本，优先覆盖两个人；不修改默认训练配置或切换 execution_authorized。核对输入 RGB/mask、mask 派生框与坐标/可见性读取，不构造网络。

## 5. 交付并停止扩展

给用户一个简短结果：是否真实跑通、渲染了几帧/几次相机、数据在哪里、图片入口、MAMMA 回读是否通过，以及实际发现的问题。说明这次是小样本链路验证，未训练、不代表模型效果或全部组合质量。

报告写 `human_data_generator/docs/FIRST_CURRENT_RENDER_A1_20261023.md`，任务内可保留一份简短 checks.json；不要再建一套庞大验收框架。协调记录写根目录 COORDINATION_NOTES.md，不写 TODO.md，不覆盖本 prompt。

若因设备无空闲或实际故障无法执行，明确报告阻碍和已完成的准备，不将其描述为缺少用户许可，也不得写成功报告。执行成功后停止，不自动追加大批生产或训练任务。
