# DenseKP 重训练暂存工作区

本仓库是从 CameraHMR 中单独整理出来的一份小型、隔离的 DenseKP **训练代码**副本，主要用于代码审计、数据准备和后续重训练迁移。

## 最小外观基线

真实动作 P0 已在不改变人体几何、相机、DenseKP 投影、深度与 mask 的前提下加入程序化肤色材质、地面、世界背景和三点光。结果与前后对照图位于 `outputs/p0_appearance_baseline/`；详见 [外观资产审计](docs/APPEARANCE_ASSET_AUDIT.md) 与 [最小外观基线](docs/APPEARANCE_BASELINE.md)。当前没有本地动作兼容服装缓存，因此人物仍是无服装 SMPL-X body。

Phase 6 穿衣数据集目前因缺少经过验证的 body × motion × clothing Alembic 配对而阻塞，没有生成伪造的 dressed dataset。官方资产要求、用户操作和后续接入门槛见 [服装接入约束与方案](docs/CLOTHING_INTEGRATION.md) 与 [P0 穿衣数据集状态](docs/P0_DRESSED_DATASET.md)。

## P0 穿衣渲染

Phase 7 已解除上述阻塞：使用官方匹配组合 `rp_aaron_posed_009 / 1095` 的 6 fps clothing NPZ 与 30 fps motion，按显式帧映射生成了 16 帧 dressed dataset。RGB、full-scene depth、body+clothing person mask、DenseKP valid/visible、overlay、manifest 和 contact sheet 均已通过；详见 [P0 穿衣渲染接入](docs/P0_DRESSED_RENDER_INTEGRATION.md)。

## Appearance V1-A

Appearance V1-A 已通过：按用户授权以 `texture_01` 作为确定性服装纹理基线，NPZ face-corner UV 索引与方向验证通过，无需 V flip。16 帧纹理输出位于 `outputs/p0_dressed_textured/`；body/clothing geometry、相机、投影、depth、mask 和 visibility 回归全部通过。详见 [服装纹理接入](docs/CLOTHING_TEXTURE_INTEGRATION.md) 与 [Appearance V1 计划](docs/APPEARANCE_V1_PLAN.md)。

Appearance V1-B 已完成 body texture 前置审计：当前 10475-vertex SMPL-X mesh 与本机 UV OBJ 的 20,908 个 faces 全量一致，UV 索引合法；但官方 `bedlam_body_textures_meshcapade.zip` 尚不在本机，需要从 BEDLAM 许可下载页面手动取得，因此 skin render 尚未执行。详见 [Body Texture 接入审计](docs/BODY_TEXTURE_INTEGRATION.md)。

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

1. [`docs/TRAINING_AUDIT.md`](docs/TRAINING_AUDIT.md)：训练代码审计结果与已知风险；
2. [`docs/ANNOTATION_AUDIT.md`](docs/ANNOTATION_AUDIT.md)：真实标注的搜索结果、已验证事实和当前阻塞项；
3. [`docs/BEDLAM_OPERATOR_AUDIT.md`](docs/BEDLAM_OPERATOR_AUDIT.md)：BEDLAM 与 DenseKP surface operator、bbox 和投影约定的实证对比；
4. [`docs/BEDLAM_TINY_GENERATOR.md`](docs/BEDLAM_TINY_GENERATOR.md)：tiny BEDLAM DenseKP 生成接口、验证办法与当前阻塞项；
5. [`docs/BEDLAM_TINY_DATA_AUDIT.md`](docs/BEDLAM_TINY_DATA_AUDIT.md)：选定 BEDLAM v1 scene 的 tiny RGB 下载、成员选择与 GT archive 审计记录；
6. [`docs/BEDLAM_TINY_FRAME_MAPPING.md`](docs/BEDLAM_TINY_FRAME_MAPPING.md)：24 张 tiny 图片到 camera row、person motion 和 SMPL-X 参数帧的实测映射；
7. [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md)：现有数据加载格式和标注约定；
8. [`docs/BEDLAM_PLAN.md`](docs/BEDLAM_PLAN.md)：BEDLAM / BEDLAM2 数据转换与可行性方案；
9. [`docs/BEDLAM_RERENDER_FEASIBILITY.md`](docs/BEDLAM_RERENDER_FEASIBILITY.md)：BEDLAM / BEDLAM2 定制重渲染、相机域、鱼眼与 ERP 的可行性审计；
10. [`docs/HEADLESS_ENVIRONMENT.md`](docs/HEADLESS_ENVIRONMENT.md)：当前 Linux/RTX 3090 的 Blender 与 EGL 无界面实测结果；
11. [`docs/HEADLESS_RENDERER_SELECTION.md`](docs/HEADLESS_RENDERER_SELECTION.md)：Unreal、Blender、pyrender 的角色划分和鱼眼/ERP 方案；
12. [`docs/SYNTHETIC_RENDER_P0_PLAN.md`](docs/SYNTHETIC_RENDER_P0_PLAN.md)：单一 SMPL-X 几何源的最小合成渲染实施计划；
13. [`docs/P0_PERSPECTIVE_CLOSED_LOOP.md`](docs/P0_PERSPECTIVE_CLOSED_LOOP.md)：16 帧 procedural SMPL-X 的 perspective RGB/depth/mask/DenseKP 闭环实测；
14. [`docs/P0_COORDINATE_CONVENTION.md`](docs/P0_COORDINATE_CONVENTION.md)：SMPL-X、Blender、camera 与像素投影的数值坐标约定；
15. [`docs/P0_DEPTH_VISIBILITY.md`](docs/P0_DEPTH_VISIBILITY.md)：Blender depth 语义和 DenseKP visibility 判定；
16. [`docs/P0_REAL_MOTION_REGRESSION.md`](docs/P0_REAL_MOTION_REGRESSION.md)：官方真实 SMPL-X 示例 motion 的 schema、帧映射与闭环回归；
17. [`docs/MIGRATION.md`](docs/MIGRATION.md)：环境、数据和权重的迁移说明。

## 目录概览

```text
densekp/                 DenseKP 模型、数据模块和训练入口
configs/hydra/           Hydra 训练配置
configs/datasets.example.json
                         数据集路径配置示例
tools/                   标注与表面算子检查工具
scripts/                 工作区审计脚本
assets/                  静态资源说明及小型资源文件
datasets/                数据集放置说明（不包含实际数据）
docs/                    审计、数据格式和迁移文档
```

## 快速检查

以下检查不需要模型权重或数据集：

```bash
bash scripts/audit_workspace.sh
python tools/inspect_annotation.py /path/to/labels.npz
```

第一条命令用于检查表面算子、Python 语法以及工作区中是否存在超过 100 MB 的文件；第二条命令用于检查指定的 DenseKP 标注文件。请将 `/path/to/labels.npz` 替换为实际标注路径。

## 与上游代码的差异

暂存的 Python 文件保留了上游网络结构和数据变换流程，仅做了以下清理：

- 包导入路径统一指向 `densekp.core`；
- 恢复训练时使用配置中的 `ckpt_path`，不再强制使用 `last`；
- 从数据集类中移除了未使用的 `smplx`、`pickle` 和 `torchvision.Normalize` 导入。

## 当前状态

本仓库目前处于**训练代码审计暂存阶段**，尚不能视为已经验证可复现训练。数据集标注和 ViTPose-H 初始化权重均未包含在仓库中；开始实际训练前，请先按照上述文档完成数据转换、外部资源准备、环境配置和最小实验验证。
