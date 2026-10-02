# P3.5 场景配方与执行入口

执行日期：2026-10-02。P3.1–P3.4 已通过用户评审。本批实现及验收完成，**P3.5 与 P3 整体待评审**；P4 未开始。
入口与资产全表见 [场景配方索引](../rendering/SCENE_RECIPE_INDEX.md)，跨批次结论见 [P3 完成检查表](P3_COMPLETION_CHECKLIST_20261002.md)。

## 基线与独立证据

实际分支 `master`，HEAD `a3a5e7bb641bf1dee774d015127fee6b39c28bda`。
保留前批所有工作树修改、未跟踪代码、历史制品和失败版本；本批没有修改 `.gitignore`、`docs/TODO.md` 或提交 commit。
证据根目录为 [refactor_scene_recipes_20261002](../../outputs/acceptance/refactor_scene_recipes_20261002/execution_manifest.json)。

迁移前已保存 `starting_state.json`、`source_snapshot/`、32 文件 `inventory_before.json`，并在独立旧实现上采集 `jobs_before/records.json`。
命令 before 使用真实 manifest 读取、真实临时目录/符号链接，替换 Blender resolver 与进程启动为记录器；**没有渲染**。
这些记录复制到非忽略的 `tests/fixtures/scene_jobs_v1.json`，不会在测试时调用新实现生成期望值。
四个原始制作/校验脚本的有序 AST 独立冻结到 `scene_recipe_bodies_v1.json`（压缩文本，不是文件哈希）。

## 实际改动

1. `src/densekp_retrain/rendering/jobs.py`：两条命令构造保留独立的 `build_eye_background_job`、`build_temporal_job`，具名结果携带 argv 和 geometry 路径。共用的创建输出 → 执行 → 检查报告 → 链接 geometry 归入 `execute_geometry_job`。原 `run_*` 接口仍是活动 CLI 的公共入口，直接组合新步骤；没有经旧脚本跳转或公共包反向导入 tools。
2. `tools/scene/build_cycling_forest.py`、`build_reference_riverfront.py`、`prepare_hidden_alley.py`、`validate_cycling_forest.py`：保留导入定义，把原顺序执行体及仅供该配方使用的辅助函数放入 `main()`，增加标准入口守卫。直接执行仍采用原 `--` 参数、默认路径和语句顺序；导入不再制作场景、写报告或调用随机数。
3. 新增 `tests/scene_jobs_fixture.py`、`tests/test_scene_recipes.py`、两份小型独立 fixture，以及真实 Blender 导入/CLI 核查工具 `tools/validate/verify_scene_recipe_boundaries.py`。
4. 阅读路线增加配方、资产、消费者和历史透视用途索引；关键任务与验证要求保存在 docs 非忽略路径。

四个场景脚本原有辅助函数未发现外部调用者，也没有依赖 globals/locals/eval/exec 的动态查找；本地闭包继续引用同一顺序生成的场景变量。
未改场景对象创建、灯光、材质、随机参数或渲染表达式；四份原始有序执行体 AST 与 main 函数体精确相同。
因此本批不重新制作森林、河岸或街景，也不重复 P3.4 装配和像素对照；AST 与入口检查不是这些场景完整运行的验收证明。

## 验收结果与限制

| 证据 | 已核查结果 | 不能外推的范围 |
| --- | --- | --- |
| `jobs_comparison.json`、before/after records | 30 用例、20 条记录的启动 argv 精确一致；路径、manifest 行读取、异常类型/消息、调用顺序、输出文件和链接行为一致 | mocked launch 不证明 Blender 渲染正确 |
| `recipe_ast_comparison.json` | 四个原有执行体含私有 helper 的 AST 精确一致 | 没有运行完整场景制作；直接执行证明限于结构与 CLI |
| `blender_import_cli.json` | Blender 5.1.1：4 脚本导入，场景对象/数据块及 Python/NumPy RNG 不变，无写文件/进程启动；3 条 `--help` 文本和退出码精确一致 | 无 parser 的森林 validator 未执行 main；不支持据此声称所有 Blender 版本可用 |
| `cpu_tests.json` | 100 项 CPU 测试通过，无失败/跳过；含新 3 项，原有小网络生命周期测试仅为回归 | 未训练 DenseKP 效果模型、未推理三权重、未运行 GPU |
| `formatted_tests.log`、`blender_import_cli_formatted.log` | 新文件排版后复验相关 3 项及 Blender 入口通过 | 排版没有更改 fixture 期望值 |
| `final_audit.json`、`asset_references.json` | 本批变更范围、原装配实现不变、选定资产存在性及交付忽略项 | 资产 stat 不代表完整贴图或所有场景加载成功 |

30 用例覆盖两路线各 15 种输入：相对/绝对路径、camera、无衣物、已有目录/链接/断链、缺报告、非 JSON 报告、进程失败、resolver 失败、缺/空 manifest、缺字段及不存在的贴图。
fixture 的调用 cwd 与 root 不同，检查两条路径原有 resolve/expanduser 差异；不因迁移统一路径语义。
只有验收临时根目录与 HOME 被替换为占位符；manifest 内容、字段、命令参数、异常消息其余部分精确比较，无浮点容差。
生产代码没有新增 manifest，也不重写历史 manifest。

旧 CLI 对照读取的是**迁移前源码快照的 parser 前缀**，在迁移后独立执行到 `--help` 退出；没有执行可能写源资产的旧完整制作体。它不同于已在迁移前采集的 jobs before，应分别解读。
首轮 Blender 导入检查失败于 Python 字节码缓存写入，保留 `blender_import_cli.log`。Blender 不默认采用 PYTHONDONTWRITEBYTECODE，验收器随后设置 `sys.dont_write_bytecode=True`；无须修改配方或允许写资产，retry 和最终检查通过。

## 保留项与后续

保留独立森林、河岸、Hidden Alley、程序街景、街景升级、场景验证、环境 Omni 预览和历史透视/时序配方。
保留 P3.4 两条渲染入口、兼容材质/groom/scene bridge；未验证分支本批字节不变。
专用 eye/shadow helper 保留透视私有归属，配方间特定 layout/preview 调用不强行抽成公共通用接口。

已存在而未修复：report 只检查文件存在、断链时报错、环境 Omni Lanczos4 与 DenseKP 插值不同、历史 mask/旋转/报告字段语义差异。
本批没有阻碍实施的运行问题。P4 必须检查整个公共包的安装、入口、忽略和跟踪状态；当前 `data/` 规则仍忽略公共数据包，不能宣称普通提交会交付完整代码。

复验新 CPU 合约：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. /home/yy/conda_envs/camerahmr/bin/python -m unittest tests.test_scene_recipes -v
```

Blender 入口工具需要已有本批 evidence 内的源码快照，命令见 `execution_manifest.json`。不要为了复验入口执行完整制作脚本。
回退边界为本批五个生产文件；应从独立快照逐项恢复，不使用 git reset，以免丢失其他批次修改。
