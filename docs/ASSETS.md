# 资产、来源与复现边界

源码不包含运行所需的全部模型、人物、纹理、groom、场景或数据。外部参考仓库
`/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch` 只读；其路径不能作为源码交付自带资产的承诺。
本页归并本地来源记录，不重新下载或确认网站当前许可。“公开可下载/免费”不等于无限制授权；实际用途应逐资产核查原许可及附件。

## 来源与许可落点

| 来源 | 本地记录与外部原始来源 | 限制 |
| --- | --- | --- |
| 独立骑行照片 | `outputs/motion_inputs/cycling_images_20260929/manifest.json`：12张COCO train2014/V-COCO骑车照片的 image/person/bicycle、下载与原图URL、逐图许可；上游 `https://cocodataset.org/`、`https://github.com/s-gupta/v-coco` | 逐图 CC 条款不同，含 NC/ND/SA，不能合并为统一授权 |
| 照片重建 | `outputs/experiments/cycling_hmr2_20260929/` 与 `cycling_smplx_20260929/`；HMR2资产来源 `https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz`，本地 `outputs/downloads/cycling_image_pilot_20260929/hmr2_assets/` | 模型预训练含COCO；这里不是独立泛化评估。外部4D-Humans代码不由本仓库改写 |
| SMPL-X | 角色manifest中的模型路径；`https://smpl-x.is.tue.mpg.de/modellicense.html` | 模型需另行取得授权；SMPL参数不能直接当SMPL-X参数或DenseKP索引 |
| BEDLAM/BEDLAM2 | `assets_external/bedlam2_pilot/LICENSE.txt`、`b2_groom_abc.tar.gz.source.json`、`inventory.json`；`https://bedlam2.is.tuebingen.mpg.de/license.html` | 许可、来源附件保留；不将代码许可推广为模型/皮肤/动作许可 |
| 来源隔离行人动作 | `outputs/experiments/source_isolated_pilot_20260930/sources/` 的 `acquisition.json`、`bedlam2_license.html`、`motion_download_description.txt`、`us_2495_L_2200.npz` 和映射CSV | 原100fps序列与本地30fps片段没有完整逐帧对应证明；约3倍索引关系是推断，不是已验证精确映射 |
| 皮肤/十成人角色 | `outputs/experiments/cycling_ten_adults_candidates_20260930/character_manifest.json` | 记录 Meshcapade CC BY-NC 4.0、version20220207；角色shape、皮肤供体和外观版本不是同一身份概念 |
| 场景/HDRI | `outputs/scenes/urban_street_assets/asset_manifest.json`；Poly Haven `https://polyhaven.com/license`，Hidden Alley原始说明 `https://blog.polyhaven.com/hidden-alley/` | CC0记录限对应Poly Haven资产，不泛化到Fab、BEDLAM或其它模型；材质/贴图/HDRI需连同场景准备 |
| 已退役BVH支线 | `outputs/audits/dedicated_cycling_mocap_audit/zenodo_record.json` 等原审计，Zenodo记录10668611 | 原CC BY 4.0记录与制品保留；当前不再支持该检索/拟合输入制作CLI |

仓库 LICENSE 及所有第三方许可、来源附件均保留。本轮没有修改许可或资产manifest；上表URL是已保存的来源定位，不是本轮联网核验或法律结论。

## 当前完整生产与消费链

1. **姿态与shape**：每张照片独立重建，两个核心来源为 COCO292226/person1330527、COCO306800/person225160。HMR2 的 SMPL 72维姿态/10维beta经过现有适配，不直接当作165维SMPL-X姿态。`body.parameters.parameters_from_pose` 按原七块组装；目标角色使用自己的shape，行人动作中的beta不直接覆盖它。
2. **人体几何**：`body.parameters.evaluate` 与专用准备配方保存顶点、faces、138点及reference/rest/joints。当前简单衣物用 `clothing.preparation.prepare_simple_clothing` 和 `clothing.shells`，十套 STYLES 留在 `tools/experiments/adult_candidate_recipe.py`。坐标、材质ID、绑定及来源manifest随制品保留；这不是布料物理仿真。
3. **UV与皮肤**：`tools/experiments/audit_skin_texture.py` 是仍保留的UV制品生产者，生成 `outputs/body_texture_audit/body_uv.npz` 等。原10475顶点/20908面与11313个UV的不同拓扑通过 faces_uv 对应，不能只按顶点索引贴图。`prepare_procedural_eyes.py` 是按需眼部资产工具；皮肤图、UV与眼部方案需匹配版本。
4. **groom导出**：`tools/experiments/inspect_bedlam2_groom.py` 在Blender中读取Alembic并导出points/offsets/imported_radius；已知记录 `outputs/experiments/cycling_commuter_pilot_20260930/groom_import_audit_v2/report.json`。随后 `bind_commuter_groom.py`、`bind_adult_grooms.py` 建立绑定；不能省略导出步骤或把原groom坐标直接当米。
5. **groom转移**：`clothing.groom_transfer.transfer_groom` 消费已有body_faces、重心权重、curve_offsets、frames、local_points、collision_correction、active mask与根部offset；保留float64/float32 frames差异和原运算顺序，不重筛毛、不修头型或穿插。
6. **场景与成像**：人体、衣物、groom及语义点共用世界变换；渲染、DS/Omni后处理和标签消费这些已准备资产。入口分工见 [场景索引](rendering/SCENE_RECIPE_INDEX.md)，几何/单位见 [契约](CONTRACTS.md)。

原旧衣物motion cache 依赖其body、garment及帧映射，不能把对应关系迁到任意新shape。当前接受的简单衣物链保留；复杂衣物修复、union、settling、polo等制作能力已退役，失败资产仍在。

## 仍需如实保留的限制

- 十成人记录为5女5男；12个已做组合、8个未尝试组合，不是全部配方通过。儿童与鞋袜配套未完成；F04外观家族不进入来源隔离划分。
- 两份历史 [人物试验](appearance/COMMUTER_TWO_CHARACTER_PILOT_20260930.md) 和 [衣物失败记录](appearance/COMMUTER_CLOTHING_REPAIR_20260930.md) 仍被组合manifest生产者写作来源引用，因此保留。它们不表示当前接受复杂衣物方案。
- 固定 mixed 制作脚本已退役；已有27行混合回归及source-isolated对原 report/manifest/geometry的读取继续保留，不承诺当前代码重制整个旧固定配方。
- 样本旧canonical路径可能缺失；现存迁移位置和缺口见 [样本导读](SAMPLE_WALKTHROUGH.md)。掩膜、外观、HDRI和原始渲染权重未在所有旧制品中完整冻结，路径存在不等于可位级复现。
- 保存的reference/rest作为迁移对照共同输入，只证明该输入下的公共计算一致，不证明即时 `rest_body` 重算位级一致。
- 场景/人物外观和完整制作质量并未因小数组或Blender合成测试通过而获得认可。当前默认outdoor_courtyard场景仍有缺失路径风险，需显式选已有资产。

历史工具可从 `git show ada2295:tools/experiments/repair_commuter_top.py` 等固定提交查看；完整旧文档从 `git show d8bd387:<文档路径>` 取回。Git仅保存已跟踪源码，**outputs和资产没有因此获得备份**。

旧衣物示例为13027顶点/26328面，使用face-corner UV、原不翻V约定及贴图色彩规则；这些只适用于对应缓存，不是新shell的统一格式。HDRI基线使用abandoned_church_4k、线性色彩/AgX，地面与背景接缝和眼部UV/视觉逼真度限制仍在。Unreal uasset不能直接当Blender衣物缓存。

BEDLAM透视适配仍在 `tools/annotation/build_bedlam_tiny_frame_manifest.py` → `tools/annotation/generate_bedlam_densekp.py` → `tools/validate/validate_bedlam_densekp_tiny.py`。原 `datasets/densekp_bedlam_tiny/` 当前缺失；[帧映射原证据](data/bedlam/BEDLAM_TINY_FRAME_MAPPING.md) 中历史186人/157帧及neutral问题限原输入，不代表当前真实数据可运行。非线性鱼眼GT须使用同几何同投影，不能靠warp旧透视标签替代。
