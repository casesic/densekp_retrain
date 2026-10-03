# B3 整体交付与验收清单

日期：2026-10-03。B3.1–B3.3 已经用户评审；B3.4 实现及有界验收完成，**B3 整体待评审**。
HEAD 仍为 `48b074901140a4ae9571ea3ff6828f2e851c5e83`，B3 全部成果未提交，用户 TODO 修改保留。
没有自动提交或开始 B4/B5。

## 现在从哪里读

- 参数组装：`src/densekp_retrain/body/parameters.py::parameters_from_pose`。
- 简单衣物保存：`src/densekp_retrain/clothing/preparation.py::prepare_simple_clothing`；
  十套 STYLES 属于 `tools/experiments/adult_candidate_recipe.py`，调用方显式传表。
- groom：`src/densekp_retrain/clothing/groom_transfer.py::transfer_groom`，消费既有绑定/几何。
- 相交：`src/densekp_retrain/rendering/blender/intersections.py::crossings`，仍在 Blender/mathutils 边界内。
- 回归任务：`tools/validate/run_regression_suite.py` 的两个具名 builder + 两个执行分支；
  固定历史输入角色在 `configs/regression/suites.json`。默认只展示，见 [当前命令](../COMMANDS.md)。
- 测试：`test_source_asset_preparation.py`、`test_commuter_intersections_blender.py`、
  `test_import_boundaries.py`、`test_regression_suite.py`；没有为了精简丢掉原数值或导入安全断言。

## 分批完成与证据边界

| 批次 | 已完成 | 实际验收 | 尚未验证/不承诺 |
| --- | --- | --- | --- |
| [B3.1](PRUNING_B3_1_20261003.md) | 参数函数、衣物保存和配方数据归位；直接消费者迁移 | 原函数独立 before；四个既有男女/照片/行人组合；参数切片、别名、写出/异常/来源及小型契约 | 保存 reference/rest 是共同输入，不是 rest_body 位级复现；未实例化模型或执行完整制作 main |
| [B3.2](PRUNING_B3_2_20261003.md) | groom 公共化；固定 mixed 制作 CLI 退役 | 四个真实冻结绑定，float32/64 frames；数组、manifest、失败部分写出；既有 mixed/adult 回归引用保全 | 未重新制作27衍生、拟合人物或验证外观/渲染；全 inactive 原失败保留 |
| [B3.3](PRUNING_B3_3_20261003.md) | crossings 公共化；九消费者迁移；最小DQS/scan测试辅助；严格有限AST映射 | Blender 5.1.1 CPU 合成三角形/立方体七项；独立完整结果；两cwd导入；原fixture不变 | 未读人物资产/源blend，未运行修复或审计main；不代表衣物质量或其它Blender版本通过 |
| [B3.4](PRUNING_B3_4_20261003.md) | 一个入口两套suite；默认展示；显式执行与设备前置检查；两旧调度CLI退役 | 38条独立桩轨迹；11项标准库测试；两cwd、无资产副本、8次冷CLI展示 | 没有真实GPU查询、评价/汇总/比较子进程、模型加载或推理；不证明设备可用或实际运行通过 |

逐批机器证据分别位于：

- `outputs/acceptance/pruning_b3_1_20261003/execution_manifest.json`
- `outputs/acceptance/pruning_b3_2_20261003/execution_manifest.json`
- `outputs/acceptance/pruning_b3_3_20261003/execution_manifest.json`
- `outputs/acceptance/pruning_b3_4_20261003/execution_manifest.json`

前三批复用已有验收记录，B3.4 未为了凑数量重跑模型/几何或图片检查。
这些 outputs 被忽略，不能当成已有 Git 备份；关键结果与限制同时保存在上述 docs 报告。

## 精确退役与保留能力

B3 实际删除四个整脚本：

1. `tools/experiments/prepare_mixed_pedestrians.py`：停止固定 JOBS 制作 CLI 和旧 helper 导入；不承诺在当前树重制旧配方。
2. `tools/validate/commuter_intersections.py`：旧 Python 导入退役，九个消费者使用公共 crossings。
3. `tools/experiments/evaluate_source_trial.py`：旧自动执行调度 CLI 退役，使用 source_trial suite。
4. `tools/experiments/evaluate_box_robustness.py`：旧自动执行调度 CLI 退役，使用 box_robustness suite。

还退役 audit_pedestrian_poses.parameters 与 prepare_adult_candidates.clothing 的旧 helper 导出，
原专用脚本继续存在。没有新增兼容包装；已跟踪历史可查 Git，不清除任何旧资产/输出/失败版本。

保留 source-isolated 的完成→复用→制作优先级、旧 mixed report/manifest/geometry 的读取、
mixed27行/成人24行回归索引与保存预测。render_adult_candidates_ds、freeze_mixed_candidates、
当前 candidate、DS遮挡/框控制、历史骑行/行人评价和独立汇总/比较能力保留。

人体、衣物、groom、138点共同几何变换、参数维度/切片及来源隔离不改；独立照片不拼共享shape motion。
同场景遮挡者继续纳入来源检查，可靠遮挡GT在有效域且完整裁剪内继续监督。
裁剪256-square/内部192列、keypoint_value语义、旧NPZ标签权重、DS/Omni差异和指标分母均未改变。
没有新的训练或效果结论。活动 status 仍拒绝框增强权重部署替换。

B3.4 的允许变化单列：默认只展示、显式 --execute + 明确设备、执行前及任务前查询设备/进程、
完整UUID匹配与逻辑设备映射，以及source源码快照改为新入口。
原业务gate、任务/日志/失败顺序保持；GPU新增前置条件失败会在业务建目录前停止。
box compare 始终独立执行，仍读 evaluation_v2，不偷偷接到 suite 尾部或改变默认目录。

## 仍保留的专用关系及后续待办

- SOURCES、mesh_panel 属于姿态审查/画廊配方；角色、照片、材质、场景布局与随机参数不塞入公共包。
- repair_commuter_top 的 edges_of/average/project、commuter_union_volume 的 capped 等失败支线关系仍存在，
  本轮仅已授权的 crossings import 迁移；等待 B4 逐项依赖和复现用途评审。
- 独立测试参考 DQS/scan 暂留；其期望仍来自解析矩阵与已知体内外关系，不用同一 helper 自证正确。
- evaluate_ds_box_control → evaluate_ds_occlusion_control 的历史导入耦合仍存在，未随调度整理拆分。
- B4 可评审失败制作工具、一次性验收工具等；不能根据目录/日期直接删除，先核清实际闭包。
- B5 再处理历史报告归并、长期规范与导航收尾；不让有效几何/监督、来源、许可与未通过效果门槛失去落点。
- GPU/真实制作/外观或推理路径如果后续确需运行，须另按对应范围授权并即时检查设备；本批桩验收不能替代。

无阻碍本次 B3 已授权范围验收的未解决依赖。上述保留关系是后续范围，不自动授权清理。
建议评审 B3 整体后再决定下一批；当前停止，所有新成果仍未提交。
