# DenseKP 重训练

DenseKP 为一个指定人体框预测固定 **138 个语义点**，服务跨相机同一人体的点对应。本仓库负责人物与合成数据准备、Dataset、训练、推理和评价。多人遮挡、框中心/尺度变化、截断、鱼眼展开畸变和骑行姿态属于研究范围；检测漏检、最终合图和部署流水线在外部项目。

当前暂停效果优化，已完成公共职责拆分和 B1–B4 代码精简；B5 已完成文档归并与只读交付核查。普通和框增强两次400步试验均不构成部署替换认可：框增强改善合成框挑战，但完整对照和旧骑行仍有退化。重构保持行为不等于效果提升。

## 最短阅读路线

1. [一个实际样本的完整链路](docs/SAMPLE_WALKTHROUGH.md)：框怎样同时改变图像、标签和监督。
2. [技术契约](docs/CONTRACTS.md) → [代码阅读指南](docs/READING_GUIDE.md)：138点、坐标、mask、loss及模块归属。
3. [命令与计算边界](docs/COMMANDS.md)：区分默认展示、数据检查和真正计算。

按需查 [数据与权重](docs/DATA_AND_MODELS.md)、[资产与来源](docs/ASSETS.md)、[验证与历史证据](docs/VALIDATION.md)、[已知行为和退役能力](docs/KNOWN_BEHAVIOR.md)。无需先读阶段报告。

四个仓库怎么交接（本仓的输入从哪来、138 点模型给谁用）：[PIPELINE_CONTRACT](https://github.com/casesic/mamma_retrain/blob/main/docs/PIPELINE_CONTRACT.md)。

```text
人物/姿态来源 → 人体、衣物、groom → 场景、相机、成像 → 图像、138点、框
 → 读取 → 确定裁剪 → 同变换图像与标签 → 计算监督 → DenseKP → 还原与评价
```

`densekp/` 负责模型及直接数据处理；`src/densekp_retrain/` 负责生成、来源组织、应用评价和实验回调。`tools/` 组合具体配方，公共实现不反向依赖实验脚本。完整裁剪是256-square，网络取中间192列，输出还原不增加32像素偏移。

## 开发与日常检查

先按 [安装说明](docs/environment/INSTALLATION.md) 准备独立开发环境。支持保留完整 checkout 的双包 editable 安装；权重、SMPL-X、场景和纹理不打包。已有依赖下的安装与源码交付已验证，未验证空环境安装全部依赖。

下面是供后续使用的命令，本轮文档核查没有执行：

```bash
bash scripts/audit_workspace.sh --no-tests
bash scripts/audit_workspace.sh
bash scripts/audit_workspace.sh --model-contracts
python -m densekp.train_keypoints --cfg job trainer=cpu
```

依次是源码/配置语法、基础CPU契约、更多CPU数据契约、配置打印。前三条不读取大权重或遍历outputs；不是全测试发现。训练默认 `data=unselected`，**去掉 `--cfg job` 会进入实际训练入口，trainer=cpu 不表示仅检查**。
回归 suite 和两个 runtime 检查默认只展示，显式 `--execute` 才计算；其它制作/评价脚本没有统一 dry-run 承诺。

真实目标为 `20260320-riding-dataset3`：鱼眼2880×2880、ERP5760×2880。合成1536×768 ERP必须按自身单位解释，不混031/0311标定。
场景与人物工具从 [配方索引](docs/rendering/SCENE_RECIPE_INDEX.md) 查找，配置从 [configs](configs/README.md) 查找。

[主题导航](docs/README.md) · [最终精简交付](docs/refactor/PRUNING_FINAL_20261003.md)。历史文档全文固定从 `d8bd387` 取回；Git不备份本地ignored输出和资产。
