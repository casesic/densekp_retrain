# DenseKP 重训练暂存工作区

本仓库是从 CameraHMR 中单独整理出来的一份小型、隔离的 DenseKP **训练代码**副本，主要用于代码审计、数据准备和后续重训练迁移。

## 当前代码从哪里读

先看 [代码阅读指南](docs/READING_GUIDE.md)，再看 [一个实际样本的完整调用链](docs/SAMPLE_WALKTHROUGH.md)。
两篇依据 2026-09-28 工作树，讲解数据生成、裁剪、模型、训练与诊断之间的关系。
下方保留的 Phase 记录描述各阶段当时的资产和实现状态，不代表全部当前行为。

## 最小外观基线

真实动作 P0 已在不改变人体几何、相机、DenseKP 投影、深度与 mask 的前提下加入程序化肤色材质、地面、世界背景和三点光。结果与前后对照图位于 `outputs/baselines/p0_appearance_baseline/`；详见 [外观资产审计](docs/appearance/APPEARANCE_ASSET_AUDIT.md) 与 [最小外观基线](docs/appearance/APPEARANCE_BASELINE.md)。当前没有本地动作兼容服装缓存，因此人物仍是无服装 SMPL-X body。

Phase 6 穿衣数据集目前因缺少经过验证的 body × motion × clothing Alembic 配对而阻塞，没有生成伪造的 dressed dataset。官方资产要求、用户操作和后续接入门槛见 [服装接入约束与方案](docs/appearance/CLOTHING_INTEGRATION.md) 与 [P0 穿衣数据集状态](docs/rendering/P0_DRESSED_DATASET.md)。

## P0 穿衣渲染

Phase 7 已解除上述阻塞：使用官方匹配组合 `rp_aaron_posed_009 / 1095` 的 6 fps clothing NPZ 与 30 fps motion，按显式帧映射生成了 16 帧 dressed dataset。RGB、full-scene depth、body+clothing person mask、DenseKP valid/visible、overlay、manifest 和 contact sheet 均已通过；详见 [P0 穿衣渲染接入](docs/rendering/P0_DRESSED_RENDER_INTEGRATION.md)。

## Appearance V1-A

Appearance V1-A 已通过：按用户授权以 `texture_01` 作为确定性服装纹理基线，NPZ face-corner UV 索引与方向验证通过，无需 V flip。16 帧纹理输出位于 `outputs/baselines/p0_dressed_textured/`；body/clothing geometry、相机、投影、depth、mask 和 visibility 回归全部通过。详见 [服装纹理接入](docs/appearance/CLOTHING_TEXTURE_INTEGRATION.md) 与 [Appearance V1 计划](docs/appearance/APPEARANCE_V1_PLAN.md)。

Appearance V1-B 已完成 body texture 前置审计：当前 10475-vertex SMPL-X mesh 与本机 UV OBJ 的 20,908 个 faces 全量一致，UV 索引合法；但官方 `bedlam_body_textures_meshcapade.zip` 尚不在本机，需要从 BEDLAM 许可下载页面手动取得，因此 skin render 尚未执行。详见 [Body Texture 接入审计](docs/appearance/BODY_TEXTURE_INTEGRATION.md)。

为保持范围清晰，本仓库有意不包含以下内容：

- pano-stitch 的交付代码；
- ONNX / TensorRT 产物；
- 人体检测器；
- CameraHMR 网格回归、CamSMPLify 与拟合流程；
- 数据集和大型模型权重。

## 来源与版本

审计来源为 `pano-stitch/tools/CameraHMR`，对应上游提交：

```text
65d11a0640d23c60f55ec82ae54298e9e7e549e0
```

该版本还包含本地推理和部署相关改动，但这些改动没有复制到本仓库。DenseKP 训练实现最早出现在上游提交 `0ca750f` 中。

## 建议阅读顺序

从 [文档导航](docs/README.md) 开始；其中按架构、训练、数据、相机、渲染、外观、动作、场景和环境分类。

## 目录概览

```text
densekp/                 DenseKP 模型、数据模块和训练入口
src/densekp_retrain/     本项目 synthetic/retraining 可复用实现
tests/                   infrastructure 单元测试与只读数值回归
configs/hydra/           Hydra 训练配置
configs/datasets.example.json
                         数据集路径配置示例
tools/                   合成数据 pipeline（generate/render/fisheye/annotation/scene/
                         validate/visualize/experiments）
scripts/                 工作区审计脚本
assets/                  静态资源说明及小型资源文件
datasets/                数据集放置说明（不包含实际数据）
docs/                    审计、数据格式和迁移文档
```

## 快速检查

以下检查不需要模型权重或数据集：

```bash
bash scripts/audit_workspace.sh
python -m unittest discover -s tests
python tools/annotation/inspect_annotation.py /path/to/labels.npz
```

第一条命令用于检查表面算子、Python 语法以及工作区中是否存在超过 100 MB 的文件；第二条运行 camera 单测和落盘数值回归（本机没有 baseline 时相应测试会 skip）；第三条检查指定的 DenseKP 标注文件。请将 `/path/to/labels.npz` 替换为实际标注路径。

## 运行环境路径

从仓库根目录直接运行 `python tools/.../script.py` 仍是受支持的方式；这些入口会加载 `src/densekp_retrain/`。开发环境也可用 `python -m pip install -e .` 做最小 editable install；原 CameraHMR `densekp/` 不属于该 package。调用 Blender 的 wrapper 按以下优先级寻找可执行文件：`--blender-bin`、环境变量 `BLENDER_BIN`、`PATH` 中的 `blender`。例如：

```bash
BLENDER_BIN=/opt/blender/blender \
python tools/fisheye/run_fisheye_camera_p0.py --skip-render
```

使用 frozen appearance 的 wrapper 默认读取 `outputs/baselines/p0_dressed_eye_background_fixed/manifest.jsonl`，可用 `--appearance-manifest` 覆盖。`run_eye_background_fix.py` 的 BEDLAM 服装纹理目录通过 `--clothing-texture-dir` 或 `BEDLAM_CLOTHING_TEXTURE_DIR` 提供。

## 与上游代码的差异

暂存的 Python 文件保留了上游网络结构和数据变换流程，仅做了以下清理：

- 包导入路径统一指向 `densekp.core`；
- 恢复训练时使用配置中的 `ckpt_path`，不再强制使用 `last`；
- 从数据集类中移除了未使用的 `smplx`、`pickle` 和 `torchvision.Normalize` 导入。

## 当前状态

本仓库目前处于**训练代码审计暂存阶段**，尚不能视为已经验证可复现训练。数据集标注和 ViTPose-H 初始化权重均未包含在仓库中；开始实际训练前，请先按照上述文档完成数据转换、外部资源准备、环境配置和最小实验验证。
