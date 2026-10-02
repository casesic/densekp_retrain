# 推荐命令与计算边界

以下以仓库根为 cwd，`python` 指已准备依赖的独立开发环境。其他 cwd 下，tools 使用脚本绝对路径；
安装后的双包可直接 import。安装、版本和逐项路径解析见 [环境说明](environment/INSTALLATION.md)，
配置角色见 [配置索引](../configs/README.md)。本页代码块只提供检查/打印命令；计算入口在表中列出所需参数，
真正执行前应先准备输入、选择新输出目录和设备。`--help` 成功不代表资产已齐备。

## 先做检查

```bash
bash scripts/audit_workspace.sh
bash scripts/audit_workspace.sh --model-contracts
python -c 'import densekp, densekp_retrain; print(densekp.__file__); print(densekp_retrain.__file__)'
python -m densekp.train_keypoints --cfg job trainer=cpu
python tools/annotation/inspect_annotation.py --help
python tools/validate/evaluate_candidate_baseline.py --help
python tools/experiments/train_source_isolated_trial.py --help
python tools/render/run_temporal_case.py --help
python tools/fisheye/run_fisheye_camera_p0.py --help
```

前两条只检查源码、配置与无资产 CPU 契约，不遍历 outputs、权重或纹理。
`--model-contracts` 不加载真实模型。第三条验证双包导入，需要先完成 editable 安装或相应开发路径设置。
第四条只打印 Hydra 配置；`trainer=cpu` 本身不关闭训练，**不能删掉 `--cfg job` 当作数据检查命令**。
默认 `data=unselected` 不变。后五条仅打印帮助，需各入口的 Python 依赖。
这些入口的根目录/临时 cwd 行为已在 P4 验证；P5 静态复核参数及执行分支，没有重新运行计算入口。

## 按用途选择入口

| 用途与入口 | 必要输入/选择 | 输出与执行边界 | 环境与外部资产 |
| --- | --- | --- | --- |
| 标签检查：`tools/annotation/inspect_annotation.py` | 位置参数为 NPZ 的绝对路径 | 读取并打印契约；不生成预测。它不代替图片/Dataset 回读 | data/model 依赖按安装表；显式 NPZ |
| 小型 CPU 安装验收：`tools/validate/verify_installation_runtime.py` | `--mode data --output` 指向不存在的新目录 | 创建小 PNG/NPZ 并走 Dataset；会写验收文件，不读取真实模型 | model 依赖；不需要项目资产。stdlib/light 模式有更小依赖面 |
| 保存预测重新评分：`tools/validate/evaluate_ds_occlusion_control.py` | `--root` 原场景组、`--reuse-predictions` 对应旧预测目录、`--output` 新目录 | 校验同源、框、GT、affine；写 NPZ、JSON 和图片。省略 reuse 参数会实际加载模型推理 | model 依赖和完整保存产物；重放不能证明图片内容从保存日起未变 |
| 候选推理：`tools/validate/evaluate_candidate_baseline.py` | `--indices` 与等长 `--labels`、新 `--output`；来源隔离输入另需 `--split-manifest`；明确 `--checkpoint`、`--device`，可选 `--model-config` | 实际推理，写冻结输入、逐例预测/图片、汇总报告；默认 device 为 cuda:0。`--reference-report` 只做一致性检查，不关闭推理 | model 依赖、完整 DenseKP checkpoint、ViTPose 初始化、NPZ/图像、来源记录 |
| 通用训练：`densekp.train_keypoints` | 显式 Hydra data 组、监督语义、trainer、权重/恢复路径；先用 `--cfg job` 查看 | 去掉配置打印参数后进入真实 train/test 生命周期，写 Hydra 运行目录、日志和 checkpoint；`init_checkpoint` 初始化权重，`ckpt_path` 恢复状态 | model 依赖、数据注册路径或适配入口、ViTPose 和初始模型权重 |
| 已冻结的 400 步配方：`tools/experiments/train_source_isolated_trial.py` | `--data-root` 含 split、indices、loader/acceptance、original_baseline、training_recommendation；可选 `--box-data`、`--output`、`--gpu` | 默认执行数据/基线审计，不能当作纯配置打印；`--execute-training` 才执行训练。步数等配方保持原值，不作为日常检查 | model 依赖、来源隔离全套资产和原 checkpoint；不自动修补缺失基线 |
| 透视时间序列渲染：`tools/render/run_temporal_case.py` | `--geometry --output-dir --frames --hdri`；可选 clothing、appearance、Blender 路径 | 直接启动 Blender，输出渲染制品与报告，没有通用 dry-run 承诺 | Blender、人体/衣物、UV/贴图/HDRI；详见配方索引 |
| Omni 生成/后处理：`tools/fisheye/run_fisheye_camera_p0.py` | 显式 geometry、clothing、output-dir 和 appearance；核对 frame/分辨率 | 默认含渲染；`--skip-render` 仍写相机/几何副本并执行后处理，且依赖输出目录已有源 EXR。默认输出可复用旧目录，不建议裸执行 | Blender（渲染时）、普通 CPU 图像/EXR 依赖、已有几何与外观资产 |
| DS 多人及场景制作 | 从 [场景配方索引](rendering/SCENE_RECIPE_INDEX.md) 选择具体入口 | 原生多人、链接单人、专用场景制作保留独立默认值和旧 Blender `--` 参数，不互换命令 | 对应 `.blend`、贴图、人体/衣物；本轮不制作任何新资产 |

GPU 计算前先检查设备和进程，不假定 GPU 0 正常；选择的设备应显式传入相应入口。
这不是 P5 需要执行的步骤：本轮没有推理、训练或渲染。

## 路径与复现

- 推荐对用户输入用绝对路径。Dataset 的 annotation_path/image_root 相对 cwd，imgname 相对 image_root；绝对 imgname 保持绝对路径。
- 仓库内配置查找以源码位置定位根目录；这不等于所有配置内相对资产路径都相对配置文件。scene blend、appearance 与历史 job 的各自规则见环境说明。
- ViTPose 默认初始化路径仍依赖 cwd 或 `DENSEKP_VITPOSE_BACKBONE`；从其他目录执行真实模型路径需显式准备它。P4 的跨目录 help/config 验收不证明真实权重路径可省略。
- 六份 Hydra data 配置已补齐交付，未启用原来无效的 loss/head 配置，也未改变默认数据选择。
- 历史命令/报告保留当时的设备、目录与状态。当前活动版本从 `outputs/experiments/box_robustness_400_20261001/status.json` 的 `active` 字段进入；其部署替换仍未获认可。
