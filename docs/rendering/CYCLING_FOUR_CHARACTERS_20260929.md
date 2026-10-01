# 四角色 × 两骑行姿态接入验收

2026-09-29 生成，2026-09-30 完成最终复查。结论：**通过本轮 integration_only 接入验收**。
四个合成身份（2 female + 2 male）、两个独立照片姿态、八份衍生几何均已接入。
未进行 DenseKP 推理或训练，未建立独立 train/val/test，不据此宣称模型效果提升。

## 产物与审查入口

实验根目录：`outputs/experiments/cycling_four_characters_20260929/`，下文路径相对此目录。
保留原照片重建、旧实验、失败版本和已有未提交修改；未移动至 datasets。

| 产物 | 路径 / 数量 |
|---|---|
| 总验收 | `acceptance_report.json` |
| 独立数值验收 | `final_numeric_acceptance.json` |
| 身份 manifest | `characters/cycling_{F01,F02,M01,M02}_v1/manifest.json`，4 份 |
| 原始姿态衍生 | `derivatives/<character>/<pose>/geometry/`，8 份；不是时间序列 |
| 当前衣物 | 对应 `clothing_v2/`，每个角色按自身模型和 shape 生成 |
| 最终棚拍 | `review_material_v3_clothing_v2/`，48 张实际渲染 |
| 棚拍决策 / 冻结记录 | `visual_review_decision.json`、`review_acceptance.json` |
| DS 场景 | `ds_v1/{cycling_forest,urban_street_assets}/`，4 张 ERP、13 行 loader 样本 |
| 场景视觉复查 | `ds_visual_review.json` |
| 测试 | `tests_final.log`，16 项通过 |

直接查看：[头部近景](../../outputs/experiments/cycling_four_characters_20260929/review_material_v3_clothing_v2/identity_heads.jpg)、
[体型对照](../../outputs/experiments/cycling_four_characters_20260929/review_material_v3_clothing_v2/shape_comparison.jpg)、
[姿态 P1 多方向](../../outputs/experiments/cycling_four_characters_20260929/review_material_v3_clothing_v2/pose_P1_multiview.jpg)、
[姿态 P2 多方向](../../outputs/experiments/cycling_four_characters_20260929/review_material_v3_clothing_v2/pose_P2_multiview.jpg)、
[森林裁剪](../../outputs/experiments/cycling_four_characters_20260929/ds_v1/cycling_forest/crop_review.jpg)、
[街景裁剪](../../outputs/experiments/cycling_four_characters_20260929/ds_v1/urban_street_assets/crop_review.jpg)。

## 身份、来源与冻结

所有角色使用本地标准 SMPL-X female/male、10 个 shape 参数，10475 顶点、20908 面。
模型只读来源为外部 `pano-stitch/tools/smpl_models/smplx/`；完整路径、许可入口、betas 和版本在 manifest。
标准模型的前 10 个 shape 方向已在前期核查中与归档模型比对；没有将 BEDLAM2 的 betas 直接复制进来。

| 身份 | Shape 来源 | 静态身高 m | Meshcapade 贴图 basename | 同性别内部身份差异依据 |
|---|---|---:|---|---|
| F01 | female，10 个零值 | 1.65881 | `skin_f_asian_01_ALB` | 较矮体型；细眉、较窄唇部纹理 |
| F02 | female，`[.75,-.5,.25,0,0,0,0,0,0,0]` | 1.71724 | `skin_f_african_01_ALB` | 较高体型；更浓眉、不同唇部及脸部细节 |
| M01 | male，既有 `rp_aaron_posed_009/1095/motion_seq.npz` 的 10 维 betas | 1.88294 | `skin_m_white_01_ALB` | 较高体型；不同眉毛、发际线与嘴部细节 |
| M02 | male，10 个零值 | 1.79378 | `skin_m_african_01_ALB` | 较矮体型；不同眉唇、剃发式头皮贴图 |

贴图从已有 `/home/yy/datasets/BEDLAM/body_textures/bedlam_body_textures_meshcapade.zip`
提取四个成员，许可记录为 Meshcapade CC BY-NC 4.0、version 20220207。
COCO 图片逐图来源/许可仍在 `outputs/motion_inputs/cycling_images_20260929/manifest.json`。
这些是稳定配置的合成人物，不宣称对应四位扫描对象或照片人物。

四角色统一灯光、观察方向、正交相机尺度和蓝上衣／白横带／深色短裤，防止颜色与姿态代替身份差异。
同性别体型顶点 RMS 差异约女性 10.84 mm、男性 16.28 mm；居中头部 RMS 仅约 0.41 / 0.66 mm。
因此脸部身份差异较依赖纹理细节，不宣称覆盖广泛三维脸型。原生眼球使用程序化虹膜/瞳孔，仍显简化。
头发只记录为 albedo 方案，没有独立三维发型。shape v1 经审查未改，冻结使用 material v3、clothing v2。

## 修正记录与几何一致性

material v1 的眼区移除导致眉眼间距不自然；v2 的局部椭圆 UV 位移产生眉毛折叠，均保留并拒收。
v3 使用宽区域单调 UV 采样变换对齐原生眼睑，四角色头部正面/斜面复查无明显双眼、重眉或错贴。
采样 UV Jacobian 最小值 0.49506 > 0；这一数值检查补充视觉审查，不等同于照片级真实度。
身体 UV 文件 `outputs/body_texture_audit/body_uv.npz` 的 faces 与目标模型严格一致，采样变换不改变网格拓扑。

clothing v1 按整三角面选择造成领口、袖口、裤脚和横带锯齿；v2 在静态人体三角形内裁切边界，
记录重心坐标映射至各姿态，使用连续静态坐标配色。所有八组合多方向复查无明显破洞、错位或穿插。
边界检查只允许领口、袖口和裤脚，未发现意外开边；局部外偏检查不构成穷尽的自相交证明。

姿态 P1=`coco_292226_person_1330527`，P2=`coco_306800_person_225160`，
来源 `outputs/experiments/cycling_smplx_20260929/<pose>/canonical_pose.npz`。
每个目标角色重新计算自身 shaped pelvis，设置 transl 为其负值，再按原姿态旋转生成 body 与 `R @ V`。
不复制照片人物 shape，不将两张独立图片拼成共享 shape 的连续 motion；不重新运行照片重建。
每份 `motion_params.npz` 独立加载模型还原后与保存网格最大差异为 0。

世界变换由既有 `world_points` 规则和 Blender 实际导出的 world mesh 交叉验证：
人体、衣物及 138 点共用人体决定的抬升、轴转换、yaw 与位置。衣物不能依据自身最低点另行抬升。
当前仓库 `assets/densekp_regressor_R.npy` 为唯一 138 行语义依据；不使用 BEDLAM 关节索引替代。

## 双相机与监督结果

| 场景 | 四个人物依次为 | Loader 行 | 监督点 | 其中遮挡点 |
|---|---|---:|---:|---:|
| cycling_forest | F01/P1、F02/P1、M01/P2、M02/P2 | 7 | 685 | 301 |
| urban_street_assets | F01/P2、F02/P2、M01/P1、M02/P1 | 6 | 694 | 293 |
| 总计 | 八组合各出现一次世界实例 | 13 | 1379 | 594 |

沿用各场景原生光照和阴影。鱼眼为 2880×2880；本次源 ERP 和展开输出均为 1536×768，
不是实用真实数据的 5760×2880。LUT、投影和 mask 按实际尺寸生成；鱼眼内参保持其原始像素单位。
每场景 `rig.json` 保存 20260320 Double Sphere 全部内参及矩阵，没有使用 031 标定。
双相机物理轴方向相反、等高，合成基线 0.18 m（不是实测外参）。街景相机中心为
`(0,+.09,1.15)` / `(0,-.09,1.15)`；森林另叠加场景 origin/yaw，详见其 rig。
`camera_to_world` 为共同 ERP 朝向，`physical_ds_camera_to_world` 才是相反的物理镜头朝向。

沿用显式 ERP mask 与真实鱼眼参考 `02214.jpg` 的镜头轮廓 mask。
后者来源算法为灰度阈值、闭运算、最大外轮廓填充；这是既有实测遮罩估计的限制。
标签和 oracle mask 使用已保存的有效域，未根据新渲染 RGB 是否为黑色逐点判无效。
新衣物参与实际深度/实例渲染后重新计算可见性；沿用 5×5 邻域和 4 cm 深度容差。

监督仍为 `in_view & inside_full_crop`，遮挡不自动取消可靠 GT 的监督。
13 行实际 loader 均校验完整 256-square crop 往返和监督 mask；target 第三列为显式二值监督，
不是 DenseKP prediction 的 keypoint_value 置信度。实例 mask 受有效域约束，框严格包围保存的实例 mask。
森林 cam4/person4、街景 cam4/person3/person4 不在有效区，无框；未运行检测器，不能算检测失败。
`diagnostic_loader_identity.json` 给出每行的角色、姿态、相机、图像与 integration_only 标记。

| 检查 | 沿用门槛 | 本次最大误差 / 结果 |
|---|---:|---:|
| 参数还原人体 | 2e-5 m | 0 |
| pelvis 居中 | 2e-6 m | 通过 |
| 原始网格 R @ V | 1e-6 m | 5.96046e-8 m |
| 世界点目标变换 | 2e-6 m | 通过 |
| Blender 实际网格及其 R @ V | 5e-6 m | R @ V 最大 8.24520e-7 m |
| 实际衣物共用世界变换 | 5e-6 m | 3.97069e-7 m |
| 投影 / crop 往返 | 2e-4 px | crop 最大 1.22786e-5 px；投影通过 |
| DS 光线往返 | 1e-8 | 通过 |
| 人体含衣物 AABB 间隔 | 至少 5 cm | 通过 |
| 衣物最大局部偏移 | 1 cm | 6.083 mm |

衣物求解目标仍为局部外侧间距 0.4 mm，求解容差 1e-8 m。
新增 float32 保存后复查采用 2e-7 m 舍入容差，实际最小读回值约 0.399932 mm；
不把该值写成严格 ≥0.4 mm，也不将读回容差当作修改原有投影、世界网格或 crop 门槛的理由。

## 耗时、代码与边界

已计时阶段合计 **1139.13 秒（18 分 59 秒）**，包含失败版本预览，不包含编码、审查和暂停。
几何 2.59 s、衣物修正 11.77 s、最终 48 张棚拍 254.18 s、DS 完整管线 433.37 s、最终数值复查 1.56 s。
其余初版/眼部试验预览分别 255.45 / 119.20 / 61.02 s。DS 场景记录的 render_seconds
还包括图像后处理，不能当作精确 GPU kernel 时间；也不能再与 DS 总时长重复相加。
首次 GPU 检查至报告落盘跨约 10 小时 16 分，包含跨夜中断，不是持续计算耗时。
9 月 30 日复查 GPU0 空闲、18 MiB，自己的渲染进程已退出；GPU1 原有 Blender GUI 未动。

新增入口为 `prepare_cycling_characters.py`、`revise_cycling_eye_materials.py`、
`refine_cycling_character_clothing.py`、`freeze_cycling_characters.py`、
`render_cycling_characters_ds.py`、`finalize_cycling_characters.py`（均在 tools/experiments），
共享材质在 `tools/render/character_materials.py`，独立核验在 `tools/validate/validate_cycling_characters.py`。
保留旧渲染材质分支，新角色走逐人物配置；冻结审查通过后才允许 DS 渲染。
精确 Blender 命令见各场景 `render_command.json`，后续重跑必须另建实验目录。

测试涵盖衣物裁切/重心映射、几何、DS 和遮挡监督，共 16 项通过。
没有下载、重建、训练、推理、计算 SHA256、提交 commit 或修改外部项目。
原重建保留检查记录为文件路径、大小及 mtime 早于任务开始；不是内容哈希证明。

本轮无未完成的接入项。已拒收版本保留，最终版本通过有限视图视觉及数值验收。
尚未解决且属于下一阶段：独立三维发型、通勤衣着、更多独立姿态来源、正式渲染噪声质量标准、
正式数据划分/规模及训练效果。32-sample DS 图仍有可见噪声，不能替代正式图像质量验收。
四角色与两个姿态组成连通来源图，当前八组合不能宣称是按身份且按姿态来源隔离的训练/验证集。
