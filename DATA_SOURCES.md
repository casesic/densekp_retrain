# 数据来源与重建方式

本仓库只包含源码、配置、测试与文档。运行时依赖的数据、模型权重和渲染产物
全部被 `.gitignore` 排除，需要按下列来源自行取得。

许可细节、逐资产的来源 URL 与限制说明已随仓库提交，见：

- [`docs/ASSETS.md`](docs/ASSETS.md) — 资产、来源与复现边界（逐资产来源表）
- [`docs/DATA_AND_MODELS.md`](docs/DATA_AND_MODELS.md) — 数据根、索引与活动版本

## 不在仓库里的内容

| 路径 | 体积 | 内容 | 取得方式 |
| --- | --- | --- | --- |
| `assets_external/` | 12 GB | BEDLAM2 pilot 资产（groom、场景、许可附件） | `https://bedlam2.is.tuebingen.mpg.de/`，需注册并接受许可 |
| `assets/scenes/` | 12 GB | 场景几何、纹理、HDRI（含 Hidden Alley / Riverfront） | Poly Haven `https://polyhaven.com/`（CC0），另见 `assets/README.md` |
| `outputs/` | 163 GB | 实验产物：重建、训练、渲染、评估、对比报告 | 由本仓库脚本重新生成，无法直接下载 |
| `datasets/*_tiny/` | 139 MB | 合成小数据集（p0_dressed、omni/ds multiperson） | 生成脚本见 `tools/`，或从 `outputs/` 重建 |
| `third_party/` | 110 MB | 上游依赖源码（如 4D-Humans） | 上游仓库，版本见 `docs/legacy/requirements.upstream.txt`（历史快照） |
| `auth.json` | — | 本机 Codex CLI 登录态（含 API key），与本项目代码无关 | **不备份、不分发**，换机后重新登录即可 |

## 三类关键外部依赖

1. **SMPL-X 人体模型** — `https://smpl-x.is.tue.mpg.de/modellicense.html`，需单独申请授权。
   许可不允许把模型权重转发给第三方，因此仓库里只有加载代码。
2. **BEDLAM2 人物与 groom** — `https://bedlam2.is.tuebingen.mpg.de/license.html`，需注册。
3. **HMR2 重建资产** — `https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz`。

照片来源为 COCO train2014 / V-COCO（`https://cocodataset.org/`），
逐图许可记录在 `outputs/motion_inputs/cycling_images_20260929/manifest.json`，
该文件随 `outputs/` 一起不在仓库内。

## 重建顺序

```bash
pip install -e .        # 当前依赖：pyproject.toml 的 [project] dependencies
# 上游的一次性环境快照（不再维护，chumpy 在 Python 3.10+ 通常装不上）：
#   docs/legacy/requirements.upstream.txt
# 1. 取得 SMPL-X / BEDLAM2 资产，放到 assets_external/ 与 datasets/
# 2. 准备场景与人体几何
# 3. 按 docs/COMMANDS.md 依次运行重建 → 训练 → 渲染
```

`outputs/` 属于可重跑的中间产物，不做版本控制；需要长期保留的结果
请单独归档（本机的 `/mnt/1T`、`/mnt/hdd`、`/mnt/16T` 均为只读挂载，
归档前需先重新挂载为可写）。
