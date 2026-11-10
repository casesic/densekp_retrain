# APD6526：头发颗粒闪烁诊断与预览修正

日期：2026-10-09。状态：**完成 8/64 samples 短段对照和 64 samples 完整候选预览；动态减闪效果待用户连续观看，停止于评审点。** 全部为 `integration_only`。

入口：[总评审页](../../outputs/experiments/apd_groom_flicker_20261009/index.html) · [头部原像素对照](../../outputs/experiments/apd_groom_flicker_20261009/comparison/heads/index.html) · [全身同步对照](../../outputs/experiments/apd_groom_flicker_20261009/comparison/frames/index.html) · [新 76 帧预览](../../outputs/experiments/apd_groom_flicker_20261009/full_preview/frames/index.html)。

**结论：采样噪声是顶部亮暗颗粒的主要成分，得到本轮单变量对照的强支持。** 同帧 64 samples 头部裁剪中的颗粒明显减轻，轮廓和头皮覆盖未见明显改变。没有改身体、衣物、groom 或材质，没有引入降噪、模糊或时间平滑。这个结论不等于证明全部动态闪烁已消失，也不把合理高光变化或细发丝像素覆盖变化归为几何错误。

## 1. 冻结范围与旧配方核实

已读取穿衣报告、frozen_recipe、探针及完整渲染 manifest、`full/validation.json`，以及实际渲染脚本和被调用的材质/groom 消费者。检查 Git 和适用指令，未发现 AGENTS.md。旧 A1/A2、TODO、旧预览、角色和公共代码均保留；本轮记录761个既有文件；最终发现 `docs/history/prompt.md` 在工作期间大小/mtime发生变化（3726 B→13869 B），本轮未写入、变化来源未核实，当前内容原样保留。**其余760个记录文件，包括TODO、角色资产和旧预览，大小/mtime前后一致**。未计算SHA256。

计算前保存 [frozen_comparison.json](../../outputs/experiments/apd_groom_flicker_20261009/frozen_comparison.json)：

| 项目 | 固定内容 |
| --- | --- |
| 角色和身体 | cycling_M01_v2；原 `fit_attempt_2/parameters.npz`，不改已接受的左臂快摆 |
| 短段 | **asset 9–18，源帧 39–48，连续 10 个原观察时刻** |
| 选择理由 | 在新渲染前按旧联系图选取，覆盖源 45→46 快速回落，头顶清楚可见；不根据新结果换段 |
| 源相对时间 | **1.560045–1.920011 s** |
| 原动作片段时间 | **0.360027–0.719993 s**；短视频减去固定起点，首末跨度 **0.359966 s** |
| 基线 | 原 8 samples 的无损 PNG；从既有合成图取出原生 512×512 面板，不解码旧有损 MP4 |
| 头部裁剪 | 每个方向固定 `[184,32,328,176)`，**144×144、1:1 像素**；不跟踪、不逐帧缩放 |
| 复现校准 | 短段内 asset16/源46，两方向各重复一次 8 samples，与仍保留的旧探针原 PNG 比较 |

实际图像生成设置：

| 设置 | 8 / 64 对照 |
| --- | --- |
| 引擎/设备 | 同一现有 Blender 5.1.1、Cycles CUDA、同一 RTX3090 GPU0 |
| samples | **唯一变化：8 → 64** |
| 降噪/自适应采样 | 两者均关闭 |
| seed | 29；`use_animated_seed=False`；未改变 scene 帧或随机策略 |
| 分辨率/相机 | 每方向 512×512、100%；原正/侧正交相机，尺度 2.3 m，固定目标/矩阵 |
| 色彩 | 同一 factory 默认状态及 AgX；image/display/view 设置快照一致 |
| 光线反弹 | max4、diffuse1、glossy2、transmission2，其它默认项一致 |
| 灯光 | 原三盏区域灯与 uniform world；未调整能量、位置或尺寸 |
| 材质/几何 | 原皮肤 material_v6、衣物材料和 groom shader；同一身体/衣物缓存、头部矩阵、ABC 与 active mask |
| 渲染执行 | persistent data 开启，CPU threads4；无身体重拟合、资产制作或物理仿真 |

新增独立驱动 [preview_apd_groom_flicker.py](../../tools/render/preview_apd_groom_flicker.py)，保留旧脚本不动。它继承旧图像生成语句，改变输出路径、帧选择、采样数和审计记录；初始/最终源码差异分别保存为 `renderer_changes.diff` / `renderer_final_changes.diff`。

**复现证据超过“设置看起来一样”：** 源46的正/侧两张 8 samples 新图与旧原图 **全部 RGB 像素逐值一致，最大差 0**。随后比较新执行的完整标量设置快照，唯一差异为 `.cycles.samples: 8 → 64`；短段20张图的相机矩阵与旧记录逐值一致。见 [calibration_readback.json](../../outputs/experiments/apd_groom_flicker_20261009/calibration_readback.json)、[controlled_comparison_validation.json](../../outputs/experiments/apd_groom_flicker_20261009/controlled_comparison_validation.json)。这验证了当前机器上的基线复现；不泛化为其它机器的位级可复现承诺。

## 2. 对照产物与编码控制

短段提供两个同步视频：

- [全身并排无损 MP4](../../outputs/experiments/apd_groom_flicker_20261009/comparison/frames/lossless.mp4)：1024×1120，上排正视、下排侧视，每排左8/右64；每个人物面板均为原生512×512。
- [头部裁剪无损 MP4](../../outputs/experiments/apd_groom_flicker_20261009/comparison/heads/lossless.mp4)：640×208，从左至右为正视8/64、侧视8/64；四个144×144裁剪原像素呈现，没有单独自动缩放。

全部 40 个短段头部面板与对应旧/新 PNG 指定区域逐值一致，见 [head_crop_validation.json](../../outputs/experiments/apd_groom_flicker_20261009/head_crop_validation.json)。面板左上角已有/新增的视角文字位于裁剪之外，不参与颗粒判断。

为避免压缩影响，本轮使用 **libx264rgb、CRF0、RGB4:4:4**，不做色度下采样。短段两份视频和完整视频均解码回 RGB，与全部输入 PNG 比较，**最大像素差 0**。没有用 JPEG 联系图或旧有损视频进行噪声归因。额外的完整片段缩略联系图仅用于粗略几何浏览。

视频 time_base 为 `1/1000000`，逐帧显式赋原始时间差；短段10帧和完整76帧的 **PTS 最大误差均为 0**。短段视频长0.399966 s，其中末帧按原展示策略保持40 ms；首末观察跨度仍为0.359966 s。完整视频仍为首末跨度3 s、容器长3.04 s。没有重采样、循环或变速。

无损 MP4 为 H.264 High4:4:4 Predictive，浏览器支持可能不同；每页同时提供直接读取 PNG 的原时间播放器和原像素显示按钮。PNG 播放不以另一个有损编码替代对照。页面时间边界检查通过，但没有把程序运行或解码成功称为实际浏览器人工动态验收。

## 3. 证据、判断与不确定性

本轮查看了**所有10帧、两个方向的同像素头部裁剪**。8 samples 顶部及侧面有较强、孤立的亮暗颗粒；64 samples 中这些颗粒明显减轻，发型整体轮廓、额头发际线和头皮覆盖未见明显变化。未在这些裁剪中观察到新的成束消失、头发留在原地、明显几何跳动或大范围穿插。

同时，两个设置使用完全相同的身体、衣物和头部矩阵；64 samples 全部152张视图的身体缓存回读、固定groom曲线数量、头部变换只施加一次等检查通过。改变采样数不会修正原绑定，本轮也没有声称修好了历史毫米级穿插。既有短发整束刚性跟随及局部穿插限制原样继承穿衣报告。

据此区分三类现象：

| 现象 | 本轮能支持的判断 |
| --- | --- |
| 顶部/皮肤的随机亮暗颗粒 | 同帧、同几何、同seed下仅增加samples便明显减轻；强支持欠采样噪声是主要成分 |
| 随头部朝向改变的亮度/高光 | 不自动判为闪烁错误；本轮没有把64图当无噪声GT，也没有消除正常高光 |
| 束状消失、几何错位/穿插 | 未改资产；数值/裁剪检查未见新增明显异常，但不能用固定拓扑证明全时刻无视觉异常 |

**未使用原始相邻帧像素差作为闪烁指标。** 头部本身在运动，这样的差分会混合轮廓位移、高光变化与噪声。本轮也没有通过画面模糊、时间平均或运动补偿后的视频替代原始输出。

**实际连续播放是否足够稳定仍待用户评审。** 只查看同帧裁剪和部分完整联系图，没有实际连续观看新视频；因此不写“动态人工验收通过”或“闪烁完全消失”。剩余细颗粒、细发丝亚像素覆盖/抗锯齿变化、合理高光以及已有轻微穿插仍可能影响观感。完整联系图另检查了 asset0–15、40–47、72–75，未见明显新增脱体或头发冻结；不把这项抽查冒充全段动态验收。

若用户观看64版后仍看到明显束状跳变，应在新一批中先定位具体源帧、方向和区域，再结合对应头部变换/曲线集合及考虑头部运动的比较方法区分噪声、高光和几何问题。本轮不自动提高到128/256、不引入降噪，也不先修改绑定。

## 4. 有条件更新完整预览

由于短段同帧颗粒明显改善、没有发现新增明显几何异常，且成本估算低于30分钟，按照本轮授权完成64版全段。决定在完整计算前写入 [short_decision.json](../../outputs/experiments/apd_groom_flicker_20261009/short_decision.json)，明确该决定依据静态单变量对照与数值检查，**不是动态人工验收**。

完整入口：[HTML](../../outputs/experiments/apd_groom_flicker_20261009/full_preview/frames/index.html) · [无损RGB MP4](../../outputs/experiments/apd_groom_flicker_20261009/full_preview/frames/lossless.mp4)。视频画布1024×608含标题/说明，两个实际渲染面板仍是原512×512、同一像素尺度；画布边栏高度不属于相机分辨率变化。

仍是 cycling_M01_v2、源 `[30,106)`、同一76帧、原时间关系、正/侧固定视角。复用已完成的10帧64版，只补渲染其余66帧。旧8版所有产物不覆盖。完整阶段设置快照与短段64版逐值一致，没有为完整预览调整材质、光照或额外提高质量参数。

## 5. 实测资源、失败记录与存储

计算前GPU0为18 MiB、利用率0%，选择健康空闲的 RTX3090，UUID **GPU-d481acc7-106f-29e5-1298-8c0d398261b2**。本轮完整进程 PID644237 在该设备上抽查约1716 MiB；这不是连续监测的显存峰值。原GPU1上的 Blender PID43491 保持不动。既有Blender在沙箱内不能枚举CUDA，沿用本轮获准的沙箱外单GPU执行；没有安装或下载。

| 阶段 | 实测/预算 |
| --- | --- |
| 8 samples校准 | 1时刻×2方向，实际渲染调用合计约5.01 s；日志最后图像时刻约19.76 s（含初始化） |
| 64 samples短段 | 10时刻×2方向，**55.98 s**，其中初始化14.67 s |
| 短段双方向每时刻 | 约3.98–5.18 s |
| 完整阶段事前估算 | 初始化＋66×最慢双方向时刻×1.5＋45 s打包余量＝**572.31 s**，低于1800 s |
| 完整阶段实际补渲染 | 66时刻×2方向，**288.14 s**，含初始化14.79 s |
| 全部64 samples渲染合计 | **344.12 s**；共152张视图，没有重复渲染短段 |
| 短段/完整打包与无损回读 | 3.30 / 10.89 s |
| Blender峰值RSS | 短段2334060 KiB；完整2335016 KiB，约2.23 GiB |

这些时间包含本资产的初始化、几何/BVH更新及相机操作，不含全部阅读、编写、工具审批等待和评审；不同初始化/片段耗时不能作为8与64的严格性能比例。尤其不能外推正式DS场景、其它发型/光照或采样策略。

**失败证据保留：** 校准两张8版图已成功渲染，但新驱动最终写JSON时，设备列表和帧集合重用了变量名，导致集合不能序列化。图像生成与渲染设置没有受此元数据错误影响；两图仍与旧版逐像素一致。随后只修正新驱动变量名，没有重渲染或覆盖校准目录；`calibration.log`、原图、settings/timings和初版源码diff均保留，`calibration_readback.json`如实记录失败。

成品验证后，仅删除**本轮成功的152张可重建临时分视图PNG**，释放 **27,571,404 B**。短段/完整无损合成PNG、三份视频、联系图、校准失败现场、配方和日志保留；旧目录和源资产完全不清理。新增交付约 **36.2 MB**；三个无损视频分别约2.64 MB（全身对照）、0.743 MB（头部）、7.178 MB（完整）。没有复制角色模型、纹理或巨大groom缓存。

资源与生命周期依据：[resource_record.json](../../outputs/experiments/apd_groom_flicker_20261009/resource_record.json)、[cleanup_record.json](../../outputs/experiments/apd_groom_flicker_20261009/cleanup_record.json)、[protected_after.json](../../outputs/experiments/apd_groom_flicker_20261009/protected_after.json)。

## 6. 交付边界与停止点

新增独立脚本只有渲染驱动和 [package_apd_groom_flicker.py](../../tools/visualize/package_apd_groom_flicker.py)。旧驱动、公共消费者、身体参数、衣物/groom绑定、A1接口、监督逻辑、TODO和旧预览均未修改。定向验证覆盖设置差异、校准像素、原生裁剪、152条相机/几何记录、三份视频RGB/PTS、网页时间边界、文件引用和输入保护；没有运行模型推理、训练或不相关全库测试。

交付清单：[delivery_manifest.json](../../outputs/experiments/apd_groom_flicker_20261009/delivery_manifest.json)；最终检查：[delivery_validation.json](../../outputs/experiments/apd_groom_flicker_20261009/delivery_validation.json)。原时间映射、APD CC BY署名及独立角色许可见同目录CSV和 [ATTRIBUTION.md](../../outputs/experiments/apd_groom_flicker_20261009/ATTRIBUTION.md)。来源组继续为 `APD_5724486_v1_identity_session_unknown`，角色家族 `cycling_M01`；不同samples不构成独立验证来源。

**本轮停在角色预览质量评审。** 请以同像素头部对照和新76帧播放判断顶部颗粒是否已可接受。没有宣布正式DS采样数，没有启动双路/DS/ERP场景渲染、资产适配或训练，也没有commit。
