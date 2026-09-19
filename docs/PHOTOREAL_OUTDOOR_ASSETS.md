# 写实户外场景选型

2026-09-19。目标：使用成熟的几何、UV/PBR 材质和场景布置，获得可信的户外环境；不要求复刻 `assets/example/00001.jpg`。原 `assets/scenes/riverfront/riverfront.blend` 保留。

## BEDLAM / BEDLAM2 调研结论

BEDLAM2 论文 Table 6 明确列出：多数三维场景另购于 Unreal Marketplace / Fab；不是随公开渲染代码一起分发的 Blender 场景。示例包括 busstation、citysample、rome、stadium、chemicalplant。HDRI 来自 Poly Haven。

- 论文和资产表：https://arxiv.org/html/2511.14394v1#S7.T6
- 官方渲染代码：https://github.com/PerceivingSystems/bedlam2_render
- Starter Pack 说明：https://github.com/PerceivingSystems/bedlam2_render/blob/main/unreal/render/unreal_quickstart.md
- BEDLAM v1：https://github.com/PerceivingSystems/bedlam_render

Starter Pack 为 Unreal 渲染准备了人物、衣物、头发、鞋与 HDR 图像；不能据此认定所有论文中的完整第三方环境都包含在下载包中。当前项目使用 Blender，优先选择原生 Blender 完整场景，以保留材质和布置。

## 已选来源与备选

| 资产 | 内容 | 使用方向 |
| --- | --- | --- |
| Poly Haven — Hidden Alley | 完整城市巷道、建筑、路面、城市杂物及材质 | 本轮优先落地，用实际渲染检验写实质量 |
| Poly Haven — Pine Forest | 完整松树林环境 | 自然植被环境备选，尚未下载验证 |
| Poly Haven — A Verdant Trail | 自然步道、草地与岩石环境 | 开阔步行环境备选，尚未下载验证 |
| BEDLAM2 所用 Fab 场景 | 城市、车站等专业场景 | 后续如果已有相应资产，可另做导入与许可核对 |

来源：

- 场景目录：https://polyhaven.com/collections
- Hidden Alley 项目及作者：https://blog.polyhaven.com/hidden-alley/
- Poly Haven 资产许可：https://polyhaven.com/license （CC0）
- 官方完整包：https://dl.polyhaven.org/file/ph-assets/Scenes/hidden_alley.zip

下载包公布大小：1,505,326,815 字节；服务端 SHA-1：`582ba7a02e188f1265fbe437b08b24518270b9fa`。ZIP 索引包含 406 个条目，解压总大小约 1.86 GB。

资产创作归 Poly Haven 及其社区作者。本轮工作是下载、导入、兼容性检查、相机配置与渲染验证，不将现成场景描述为自行建模。

## 本轮落地结果

- 完整 ZIP 下载成功，字节数和服务端 SHA-1 一致，解压完成。
- 原版：`assets/scenes/photoreal/hidden_alley/ph_hidden_alley.blend`。
- 适配版：`assets/scenes/photoreal/hidden_alley/hidden_alley_ready.blend`。
- 125 个材质、390 个文件纹理引用，缺失纹理数为 0；重新打开适配工程后仍为 0。
- CPU Cycles 32 samples：透视图 `outputs/photoreal_hidden_alley/perspective.png`（1600×900），经纬全景 `outputs/photoreal_hidden_alley/panorama.png`（1600×800）。两张图均已实际渲染并目视检查。
- 主视角有完整建筑、城市道具、树木与庭院；全景背后为有门洞的巷道结构。当前是较封闭的室外巷道，不是开阔公园。
- 暂未加入人物，也未接入训练数据流水线。全景相机可编辑，进一步移动相机时仍需检查几何覆盖与遮挡。
