# 场景配方、资产与消费入口

更新：2026-10-03，B1–B4已交付，B5归并导航。这里描述现有路径，不是要求重新制作资产的运行清单。
配方决定布局、灯光、材质选择与随机参数；公共库负责几何、相机、装配、成像和任务执行。
本批不新增资产、不重新制作森林或街景。场景链的既有验收范围见 [验证范围](../VALIDATION.md)。

## 先找配方，再找消费者

以下路径均相对仓库根目录。制作脚本仍可能写其既有默认输出；新增 `main()` 只保证导入不会制作场景，不把直接执行变成只读操作。
带 `bpy` 的配方应在 Blender Python 中读取，不要求普通 Python 安装 Blender。

| 制作入口（`tools/scene/`） | 输入和资产依赖 | 默认制品与用途 |
| --- | --- | --- |
| `build_cycling_forest.py` | 已打开的 Pine Forest 原场景，`assets/scenes/photoreal/pine_forest/` 贴图及 cycling_forest 本地 asphalt_02 贴图 | `assets/scenes/photoreal/cycling_forest/cycling_forest.blend`；`outputs/scenes/cycling_forest/` 中心线 CSV、validation 和预览；骑行林道配方 |
| `prepare_hidden_alley.py` | 已打开的 `assets/scenes/photoreal/hidden_alley/ph_hidden_alley.blend` 和相邻 textures | 同目录 `hidden_alley_ready.blend`，`outputs/scenes/photoreal_hidden_alley/`；原场景贴图路径、相机适配，保留原源文件 |
| `build_urban_street.py` | 程序式街景配方，固定 seed 默认 73 | `assets/scenes/photoreal/urban_street/urban_street.blend`，`outputs/scenes/urban_street/` 报告/预览；资产升级的基底 |
| `download_urban_assets.py` | Poly Haven apartment kit 下载元数据和网络（本批不运行） | `assets/scenes/photoreal/urban_street/source/modular_urban_apartments_facade/`；下载及既有校验配方 |
| `upgrade_urban_street_assets.py` | urban_street 基底、apartment kit、Hidden Alley 道具及贴图 | 同目录 `urban_street_assets.blend`；`outputs/scenes/urban_street_assets/asset_manifest.json` 和预览；baseline 与输出必须不同 |
| `validate_cycling_forest.py` | 已打开的保存后森林、贴图与场景对象 | `outputs/scenes/cycling_forest/reopen_geometry_check.json`；射线/路面核查，不制作场景 |
| `validate_urban_street.py` | 既有街景资产及报告 | 由原参数指定的检查报告；Blender 几何与道路探针，保留专用约束 |
| `render_scene_fisheye.py` | `--scene`、`--camera`，冻结 Omni-Radtan 内参 | `--output-dir` 下 equirectangular、source.json、鱼眼、valid_mask、fisheye_report；纯环境预览，无人体标签 |
| `__init__.py` | 包标记 | 场景读取直接使用公共 `rendering/blender/scene_bridge.py`；旧 bridge 包装已在 B1 删除 |

森林/街景消费路线：

1. `configs/fisheye_scene_cycling_forest.json` 和 `configs/fisheye_scene_urban_street_assets.json` 引用相应 `.blend`。
2. `tools/experiments/render_cycling_ds.py`、`render_cycling_characters_ds.py`、`render_adult_candidates_ds.py` 组合人体/衣物、布局、场景配置和双相机；各自输出目录承载 scene JSON、源 ERP 与后处理制品。
3. `tools/fisheye/render_equirectangular_source.py` 打开原生多人场景；`render_equirectangular_scene_source.py` 是历史单人链接路径。它们直接使用 `densekp_retrain.rendering.blender`，保留不同灯光、实例 ID 和默认值。
4. `densekp_retrain.imaging.ds` / `omni` 读取源制品，生成有效 mask、人物框、138 点和可见性；标注入口消费 manifest，随后打包为 Dataset。

`render_source_ground_repairs.py` 是选定候选的修复重放配方，消费已有探针/布局与场景，输出修复目录及执行状态；其固定分辨率、采样和 GPU 配方保持，不是通用渲染 API，本批未运行。
成人候选复用 `render_cycling_ds.layout`，街景升级复用 `build_urban_street.blender_main` 作特定预览；这是配方之间明确的组合，公共库没有反向导入它们。

人物准备先读 `tools/experiments/prepare_source_isolated_assets.py`：来源/角色/帧及复用顺序仍在
该配方中；姿态块使用 `densekp_retrain.body.parameters.parameters_from_pose`，简单衣物写出使用
`densekp_retrain.clothing.preparation.prepare_simple_clothing`。adult 准备也直引这些定义，
显式传入 `tools/experiments/adult_candidate_recipe.py` 中的十套 `STYLES`；公共模块不选择角色或颜色。
旧脚本的 `parameters`、`clothing` helper 导入已退役，无兼容包装。
groom 转移直引 `densekp_retrain.clothing.groom_transfer.transfer_groom`，读取冻结绑定和现有身体数组；
固定 `prepare_mixed_pedestrians.py` 制作 CLI 已退役，不提供旧名包装或当前树重制承诺。
已有 mixed 资产/report/27行索引以及成人24行索引仍保留；`render_adult_candidates_ds --assets`
与 `freeze_mixed_candidates` 不依赖退役脚本，source-isolated 仍可复用其保存资产。
`SOURCES`、`mesh_panel` 的专用配方关系保留。历史实现查 Git，ignored 资产并不由 Git 备份。
衣物保存与 groom CPU 回放仅写新验收目录，没有运行人物制作 main 或场景链；
分别见 [验证索引](../VALIDATION.md) 中对应B3.1/B3.2证据。

## 历史透视与外观检查路径

相交诊断的公共计算位于 `densekp_retrain.rendering.blender.intersections.crossings`，
仍需 Blender/mathutils；轻量包入口不自动导入它。当前 `diagnose_adult_contacts` 直接使用公共实现。
旧相交包装在 B3.3 退役；B4-D 进一步停止失败修复、union、settling、polo 及专用审计制作链，
并移除只服务其历史 DQS/scan 的参考测试。公共 crossings 四项解析断言保持，不能据此宣称衣物质量通过。
独立 riverfront 再制作、两个旧 Omni sweep 和 P0 pyrender 专用检查也已退出；已有成功/失败制品不删除。
历史算法和七项原迁移证据见 [历史数值证据](../VALIDATION.md)，不作为当前CLI推荐。

以下入口均位于 `tools/render/`，保留旧命令、默认值及报告格式。它们不被强制改成 ERP 管线。

| 入口或支持模块 | 输入/依赖 → 输出 | 归属与边界 |
| --- | --- | --- |
| `run_eye_background_fix.py` | baseline manifest/geometry、衣物贴图、眼部 mapping → 透视输出目录 | 薄 CLI，直引公共 `rendering.jobs.run_eye_background_fix` |
| `run_temporal_case.py` | 时序 geometry、appearance manifest、HDRI、可选衣物 → 透视序列目录 | 薄 CLI，直引公共 `rendering.jobs.run_temporal_case` |
| `render_p0_blender.py` | geometry、UV/贴图、HDRI、camera placement → RGB、Depth、Mask、p0_render_report | Blender 透视实现；Mask 来自 Alpha，不能替代 ERP IndexOB；参数默认引擎/设备保持 |
| `eye_background_render.py` | 眼部映射、线性图像和 shadow 数据 → 透视眼部/阴影合成 | 仅透视渲染器消费的私有支持，不伪装成全局通用材质库 |
| `run_camera_stress.py` | `configs/perspective_camera_stress.json` 八种配置、baseline → 逐配置渲染目录 | 特定实验配额与跳过已有输出的规则保留；调用 eye CLI |
| `solve_camera_stress_placement.py` | 几何/目标约束 → placement 配置目录 | SciPy 最小二乘摆放配方，不改求解规则 |
| `finalize_camera_stress.py` | 八种渲染目录、EXR Depth/manifest → 汇总、复制制品与图表 | 历史压力实验的固定汇总 |
| `finalize_p0_outputs.py` | P0 渲染、投影、深度 → 可见性标签/报告 | 保留旧深度/可见性算法，不接入 DS 标签规则 |
| `finalize_temporal_case.py` | 序列渲染与标签 → CSV、时序统计与 ffmpeg 视频 | 专用归档/展示，不在导入时执行 |
| `review_cycling_characters.py` | 候选人体、衣物/groom、材质 → studio 预览与报告 | 角色外观配方；直接调用 P3.4 公共材质/groom |
| 公共 `rendering/blender/materials.py`、`groom.py` 与 `clothing/contracts.py` | 材质/发型/衣物契约 | 活动调用方直引公共实现；对应 tools 纯导出已在 B1 删除，制作与渲染配方不变 |

`src/densekp_retrain/rendering/jobs.py` 是两条冻结透视任务的公共组织层：
`build_eye_background_job` / `build_temporal_job` 读取 manifest、解析原路径规则并返回具名 `GeometryRenderJob`；
`execute_geometry_job` 依次创建输出、调用 `run_logged`、检查报告存在、链接 geometry。
两个原 `run_*` API 直接组合这些步骤。argv 参数和顺序仍分别可见，没有新的通用场景注册表。

## 应当保留的差异

- eye baseline 相对仓库根解析，但 clothing texture、camera placement 以调用者 cwd 解析；temporal geometry/output 的 resolve 时机保留。不要顺手统一。
- 两条任务缺报告的异常文字不同；report 仅检查文件存在，不新增 JSON 内容校验；已有断链仍会触发 FileExistsError。
- 环境 `render_scene_fisheye.warp` 使用 Lanczos4，DenseKP 后处理有自己的插值/周期和标签规则；不以函数外形相似为合并依据。
- 灯光、道路参数、森林清障、随机 seed 和材质选择属于具体制作配方。P3.4 公共装配已承接实际复用部分，本批不新增空泛配方基类。
- 旧 `--` Blender 入口继续保留。新增 main 的四个脚本不承诺原有制作后临时变量作为可导入 API；仓库调用核查没有此类消费者。


## 必要外观制品的生产者继续保留

- `tools/experiments/audit_skin_texture.py` 实际写出 `outputs/body_texture_audit/body_uv.npz`
  的 UV、faces、faces_uv、component_labels；角色准备和程序眼部仍消费它。
  原 `skin_texture_audit.json` 及资产来源/许可附件保留；通用 UV JSON 审计不等于该 NPZ 生产能力。
- `tools/experiments/inspect_bedlam2_groom.py` 在 Blender 导入 Alembic，导出 points/offsets/imported_radius
  和包含 cases/export 的 report，供 bind_commuter_groom/bind_adult_grooms 使用。
  既有 `outputs/experiments/cycling_commuter_pilot_20260930/groom_import_audit_v2/report.json`
  记录 Blender 5.1.1、对象变换和来源。导出单位仍须按原报告与绑定入口解释，不能直接假定为米；
  绑定入口原有缩放未变，原许可/来源限制未解除。
- `prepare_source_trial_config.py` 仍生产 training_recommendation/regression_reference_audit；
  它还含历史提案/status写出，不应在活动目录直接重跑。B4-D 只改说明中的 CheckUpdates 公共路径。

人物、简单衣物与 groom 的当前链及 mixed/adult 保存资产消费完整保留。
B4-D 仅做合成相交与真实 Blender 受控导入，没有制作、渲染或外观验收。
历史代码使用 `git show ada2295:<仓内路径>` 只读查看；ignored 资产/失败输出没有因此得到备份。
