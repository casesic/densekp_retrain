# P3 几何与生成链完成检查表

更新：2026-10-02。**P3.1–P3.4 已通过用户评审；P3.5 实现与验收完成，P3 整体提交评审。**
本表不是 P4/P5 完成声明，也不将代表路径实测扩大为全部资产、设备和 Blender 版本覆盖。

| 阶段 | 已迁移内容 | 已有独立验收 | 状态 |
| --- | --- | --- | --- |
| P3.1 人体、衣物 | body/parameters、measurements、transforms；clothing/shells、binding；io/geometry | 男/女×两骑行姿态：人体顶点、面、138 点、衣物对应及保存恢复；与 P3.2 合计 409 数组/12 JSON 精确 | 已评审 |
| P3.2 坐标与相机 | camera/erp、rig、domain、visibility | 两相机/周期/有效域，原始 EXR 另 20 数组和既有标签；区分 DS、Omni、历史可见性 | 已评审 |
| P3.3 成像与标注 | imaging/source、ds、omni、remap；visualization/imaging | 完整 189 数组/44 图片/29 JSON；边界另 55/35/7 精确；保留源 ERP 射线距离与重映射鱼眼距离差异 | 已评审 |
| P3.4 Blender | rendering/blender 的 assembly、materials、groom、scene_bridge；clothing/contracts | 原生多人/链接单人，世界空间装配 78 数组；9 EXR/3 预览；后处理 39 数组/23 图片/6 JSON 精确 | 已评审，限 Blender 5.1.1 CPU 代表输入 |
| P3.5 场景配方 | jobs 具名构造/共用执行；4 配方显式 main；32 文件归属索引 | 独立 30 命令/路径用例（20 argv），4 有序 AST、真实 Blender 4 导入/3 CLI，100 CPU 测试 | 待本次评审 |

## 交付检查

- [x] 公共几何/相机/成像/装配不反向依赖实验脚本；活动调用方直引公共模块。
- [x] 人体、衣物、138 点共用世界变换；保留单位、矩阵方向、投影与深度来源约定。
- [x] 保留 DS、Omni、ERP 周期、显式有效区、可靠遮挡 GT、完整裁剪监督及历史规则差异。
- [x] 制作配方保留布局、灯光、材质选择和随机参数；已发现的四条导入即执行路径有显式 main。
- [x] 旧 Blender `--` CLI、两个 jobs `run_*` API、旧材质/groom/contract/scene bridge 导入兼容保留。
- [x] 未改动路径复用原证据，本批没有为了覆盖更多分支重新渲染或制作资产。
- [x] 配方 → 资产 → 配置 → 渲染 → 后处理 → 标注路线已在非忽略 docs 中记录。
- [ ] P3 整体用户评审（本次停止点）。
- [ ] P4 全公共包忽略/跟踪修复及双包 editable、入口安装验收（不得跳过）。
- [ ] P4 历史清理、P5 完整阅读路线验收；整个重构仍未完成。

## 专用配方与兼容接口

[场景配方索引](../rendering/SCENE_RECIPE_INDEX.md) 列出核查的 32 文件、输入、输出和消费者。
保留森林、河岸、Hidden Alley、程序街景/资产升级、专用场景校验、环境 Omni、DS 候选布局、修复重放、历史透视/眼部/压力/时序和 studio 配方。
这些脚本不因“仍在 tools 下”成为遗漏：专用选择留在配方，已经复用的数值/装配与执行职责进入公共库。
P3.1/P3.3 旧几何/成像符号继续兼容导出；P3.4 三个 render 支持文件与 scene bridge 保留旧导入路径。具体导出与调用方见各批报告。

## 未实测与不应推断的范围

- P3.4 只验证 Blender 5.1.1、CPU、限定种子/分辨率/采样和代表原生街景/链接单人；不包含 CUDA、旧 Blender API、所有衣物/材质/groom/集合/地面分支。
- 本批未完整执行任何场景制作或专用验证配方；新 main 保持语句序列、导入及 CLI 已核查，不代表重新验证森林/街景资产质量。
- 历史透视 renderer、眼部/shadow、时序视频、环境 Omni Lanczos4、ground repairs 未重渲染；实现保留，不统一历史行为。
- P3.4 链接验收使用既有街景代表资产；缺失的原 courtyard 默认资产不因测试替代而恢复，详见该批报告。
- 源资产只做选定引用存在性核查；不递归加载贴图、压缩包或 outputs。P3.5 无 GPU 计算，不对 GPU 0 健康作判断。
- portrait/closeup 旋转、mask 取整、历史报告/来源字段与 NaN 拒绝写入等既有事项继续单列，不随重构修复。

P4 必须检查整个 `src/densekp_retrain/` 和 `densekp/` 的安装与交付状态，特别是被 `data/` 规则忽略的公共数据包；不能只 force-add 一个 source_split.py，也不能假定本地通过等于新 checkout 可用。
本批保留 `.gitignore` 既有改动，未提交 commit。证据大清单留在 outputs，小型任务、fixture 和验证门槛保存在非忽略路径。

详细证据：[几何/相机](P3_GEOMETRY_CAMERA_20261002.md)、[成像](P3_IMAGING_20261002.md)、[Blender](P3_BLENDER_20261002.md)、[场景配方](P3_SCENE_RECIPES_20261002.md)。
