# BEDLAM2 当前原配行人的 R138 模板语义核查

日期：2026-10-09。结论：**compatible，限本地 `smplx_model_lh_20230302.blend` 的 neutral locked-head/no-head-bun 模型，以及本轮核对的 `us_2495_L/2200` 未改鞋/脚趾身体缓存。** 这里的兼容指138个有编号的身体表面位置仍对应原算子定义，不表示头型坐标不变、真实解剖精度或整个BEDLAM系列通过。

入口：[机器结论](../../../outputs/audits/bedlam_r138_compatibility_a2_20261009/compatibility.json)、[138行完整对应](../../../outputs/audits/bedlam_r138_compatibility_a2_20261009/rows.json)、[同尺度头部表面图](../../../outputs/audits/bedlam_r138_compatibility_a2_20261009/head_surface_points.png)、[本地证据目录](../../../outputs/audits/bedlam_r138_compatibility_a2_20261009/local_evidence.json)。旧manifest、`densekp_3d_pending.npy` 文件名及历史pending报告全部保留；只有新交付manifest引用此结论后标记 `densekp_label_ready=true`。

## 算子来源与编号

只读外部代码 `tools/SingleFitting/single_fitting/fitting/constraints/densekp.py` 明确从CameraHMR的 `data/train-eval-utils/downsample_mat.pkl` 与 `smplx2smpl.pkl` 构造：

```text
R[138,10475] = D[138,6890] @ T[6890,10475]
```

本轮实际读取这两个既有文件：D每行是一个原SMPL顶点的one-hot选择；按这些原顶点逐行取T、转为float32，**与项目 `assets/densekp_regressor_R.npy` 逐元素完全相同，最大差0**。没有重新计算SHA256、下载或更换R。204个非零权重和其中3个约1e-15负值原样保留，不归一化/裁零。

`rows.json`逐行保存 point_id、原SMPL vertex、SMPL-X support vertex IDs、原权重、所处三角形、UV表面坐标及主要骨骼区域。所有138行的support都能在实际SMPL-X面集中找到共同三角形。没有完整官方人体点名称字典，因此不编造138个解剖名称；语义精确定义为**原编号所对应的原SMPL采样点及其SMPL-X表面坐标**，不是BEDLAM437点的插值。

## 模板来源、修改和对应证据

官方本地插件README的20230302变更说明引入“locked head, removed head bun”，保留v1.1可选；插件 `SMPLXAddGender.execute` 明确选择两个不同blend，并按 `smplx_version` 使用相应关节回归器。BEDLAM2官方本地生成器 `third_party/bedlam2_render/blender/smplx_anim_to_alembic/smplx_anim_to_alembic.py` 明确使用locked-head模型。内部雕刻制作脚本不在已提供文件中，本轮不声称能够复刻其制作过程。

通过CPU模式Blender分别读取同一发行目录里的 standard与locked-head neutral对象，记录实际对象属性和零shape键；未求值新动作、换身体或生成替代资产。取得的对应证据超过顶点数/faces一致：

| 实际检查 | 结果与意义 |
| --- | --- |
| standard模板与既有原版neutral NPZ | 顶点最大误差1.1921e-7m；确认比较的标准模型基准 |
| 完整有序faces | 逐元素一致；保留相同表面三角形身份 |
| 每个face-corner的UV | 逐元素一致；同一局部表面参数化，没有用同顶点数掩盖重新编号 |
| 全部vertex-group名称及每顶点权重 | 逐元素一致；头/颈/身体骨骼区域未交换 |
| 所有138行surface support/UV | 同一三角形上的同一权重，保留同一UV位置及骨骼区域 |
| 实际模板修改区 | 361顶点、778面（2µm数值容差）；最大位移12.417mm，集中在头/颈相关区域 |
| 修改面法向方向 | 最小cosine0.97648，无反向；未发现借变形把点翻到另一侧表面 |
| 当前91帧R回读 | 最大差2.3842e-7m；仅为实现回读补充，非独立语义证据 |

shape basis并不相同：前16个方向的最大差约6.745mm/系数，不能跨模型复制betas去宣称同一身体。这个差异与顶点语义编号是不同问题；当前人物继续使用其原locked-head模型及原16beta，不做标准模型替代。

## 受修改影响的5个点

133行不碰到模板修改顶点，另5行如下。区域描述来自同尺度表面图的定位，**不是官方点名**。

| point_id | support顶点 | 模板点位移 | 图示区域 |
| --- | --- | ---: | --- |
| 2 | 1298,1308,1309 | 7.397mm | 后脑 |
| 6 | 2014,8972,9009 | 2.172mm | 头顶前部 |
| 75 | 2593,2772,2961 | 5.924mm | 后上方头顶 |
| 77 | 112,2483,2484 | 0.309mm | 后颈 |
| 79 | 2224,2375,2376 | 3.625mm | 后脑下方 |

已实际检查头部正/侧两方向的同尺度CPU图示：这些点仍附着原三角形和对应头/颈位置，头顶凸起被收回造成坐标变化，没有跳到脸部、耳部或另一侧。图不独立缩放、不放大位移、不隐藏被挡点；这属于语义位置检查，不是动态外观验收。

据此，本地当前模型是**保持表面对应关系的头部形状修改**，而非R编号含义改变。不能仅因头型坐标不同判不兼容；同样，若没有UV、权重、原算子和局部位置这些证据，仅凭faces或R乘法也不够。

## 适用边界与复查

新交付的原配行人及其新增眼睛/头发组合共用同一身体，均可沿用此R语义结论。眼部着色/头发不改变R作用的身体。新增眼睛/头发的观感仍待用户，公共场景仍未消费。

不推广到鞋位移、toeless/sock模板、重拓扑/重编号缓存、其它性别/发行模型、其它BEDLAM版本或官方437点。后续这些改变需重新核查。没有删头部点、改可见性、替换R或更改监督规则。

只读核查脚本：[audit_bedlam_r138_compatibility.py](../../../tools/validate/audit_bedlam_r138_compatibility.py)；模型属性导出：[export_bedlam_r138_templates.py](../../../tools/validate/export_bedlam_r138_templates.py)。前者CPU审计/图示实测约2.40秒（不包含解释器启动、读取研发时间），数据来源为本地既有文件；模型属性导出另为CPU Blender读取，无GPU渲染、无模型下载。产物仅保存必要模型属性/逐行记录与一张图，不复制578MB转换矩阵或整个模型。所有源文件只读。
