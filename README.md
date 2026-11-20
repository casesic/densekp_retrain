# DenseKP 重训练

DenseKP 给定一个人体框，回归固定的 **138 个身体语义点**，用于跨相机「同一人体」的点对应。

| | |
|---|---|
| **输入** | 图像 + 目标人物 mask + 138 点标签（合成数据来自 [human_data_generator](https://github.com/casesic/human_data_generator)） |
| **输出** | 138 点预测（完整 ERP 坐标）与评价指标 |
| **研究范围** | 多人遮挡、框中心/尺度变化、截断、鱼眼展开畸变、骑行姿态 |
| **不做** | 检测漏检、最终合图、部署流水线 —— 都在外部仓库 |

整条链：`人物/姿态来源 → 人体/衣物/groom → 场景/相机/成像 → 图像 + 138 点 + 框 → 读取 → 定裁剪 → 同变换的图与标签 → 监督 → DenseKP → 还原与评价`

## 结果与状态

- **当前暂停效果优化**。已完成公共职责拆分（B1–B4）与文档归并、只读交付核查（B5）。
- 普通 BCE 与框增强两次 **400 步**试验：框增强改善了合成框扰动下的表现，但完整对照和旧骑行数据仍有退化，**两次都不构成部署替换认可**。细节见 [400 步试验](docs/training/BOX_ROBUSTNESS_TRIAL_400_20261001.md) 与 [验证](docs/VALIDATION.md)。
- **重构保持行为 ≠ 效果提升**，别把「没坏」当成「变好了」。

## 怎么跑

先按 [安装说明](docs/environment/INSTALLATION.md) 准备独立环境（双包 editable 安装；权重、SMPL-X、场景、纹理不打包）。

```bash
bash scripts/audit_workspace.sh --no-tests      # 源码/配置语法
bash scripts/audit_workspace.sh                 # 基础 CPU 契约
bash scripts/audit_workspace.sh --model-contracts
python -m densekp.train_keypoints --cfg job trainer=cpu   # 只打印配置
```

> ⚠️ 训练默认 `data=unselected`；**去掉 `--cfg job` 就会真的开始训练**，`trainer=cpu` 不代表只检查。
> 回归 suite 与两个 runtime 检查默认只展示，要显式 `--execute` 才计算。

## 读代码

1. [一个实际样本的完整链路](docs/SAMPLE_WALKTHROUGH.md)：框怎样同时改变图像、标签和监督。
2. [技术契约](docs/CONTRACTS.md) → [代码阅读指南](docs/READING_GUIDE.md)：138 点算子、坐标、mask、loss、模块归属。
3. [命令与计算边界](docs/COMMANDS.md)：哪些是展示、哪些是检查、哪些真的在算。

按需查：[数据与权重](docs/DATA_AND_MODELS.md) · [资产与来源](docs/ASSETS.md) · [已知行为与退役能力](docs/KNOWN_BEHAVIOR.md) · [场景配方索引](docs/rendering/SCENE_RECIPE_INDEX.md) · [配置](configs/README.md) · [全部文档](docs/README.md)

## 交接与边界

- 四个仓库怎么交接（本仓的输入从哪来、138 点模型给谁用）：[PIPELINE_CONTRACT](https://github.com/casesic/mamma_retrain/blob/main/docs/PIPELINE_CONTRACT.md)
- 真实目标 `20260320-riding-dataset3`：鱼眼 2880²、ERP 5760×2880；合成 ERP 常为 1536×768，必须按自己的栅格单位解释，**不要混 031 / 0311 标定**。
- `densekp/` 是模型与直接数据处理；`src/densekp_retrain/` 是生成、来源组织、评价与实验回调；`tools/` 组合配方，公共实现不反向依赖实验脚本。完整裁剪 256-square，网络取中间 192 列，还原不加 32 像素偏移。
- Git 不备份本地被忽略的 `outputs/` 和资产；历史文档全文固定在 `d8bd387`。
