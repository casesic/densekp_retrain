# DS源ERP分辨率对照实验（2026-10-08）

## 结论

本次在同一 heavy 街景、同一人物/相机/固定框、原始 DenseKP 权重下，重新渲染了6张源ERP。提高源分辨率确实改善了**最终图局部和模型实际256-square输入**中的人体轮廓、衣褶与背景边缘，不只是展示时放大更清楚；但定位收益不单调，也不是所有样本都改善。

- 共同GT支持集486点的裁剪误差均值：A/B/C为 **14.910/14.493/14.740 px**。B比A改善2.80%，C比A仅改善1.14%，C比B退化1.71%。
- 重遮挡主目标 cam2/person3：**16.263/13.117/12.974 px**，B、C相对A分别改善19.34%、20.22%。
- 完整人物对照 cam2/person4：**4.122/3.773/3.649 px**，改善8.46%、11.46%。
- 不是所有样本受益：cam2/person2到C退化6.83%；完全遮挡的cam4/person3从91.627恶化到97.560/100.890 px。
- 当前证据支持把3072×1536（B）作为有限后续复核候选，不支持直接把5760×2880（C）扩展为默认批量源渲染。C相对B没有总体定位收益，存储和渲染成本继续增加。
- 本实验仍经过“源ERP→DS鱼眼→normal ERP”，没有验证DS直接渲染。直接渲染仍可作为独立研究方向，用来判断能否避免中间ERP采样及降低存储成本，但本结果不构成立即开发或效果承诺。

## 范围与输入

代码HEAD为`d0926fa`，实验在保留用户已有`docs/TODO.md`修改的脏工作树中执行；没有提交commit、计算SHA256、训练、更新权重或修改外部pano-stitch。

输出根：[ds_source_resolution_20261008](../../outputs/experiments/ds_source_resolution_20261008/)。实验计划在渲染前写入[experiment_plan.json](../../outputs/experiments/ds_source_resolution_20261008/experiment_plan.json)，固定内容包括：

| 项目 | 实际值 |
| --- | --- |
| 布局 | 旧`ds_occlusion_control_20260928/heavy`的4人布局 |
| 相机 | cam2、cam4；20260320-riding-dataset3 Double Sphere参数 |
| A/B/C源ERP | 1536×768 / 3072×1536 / 5760×2880 |
| DS鱼眼 | 三组均为2880×2880 |
| 最终normal ERP | 三组均为5760×2880 |
| 渲染 | Blender 5.1.1，Cycles，64 samples，seed 0，animated seed关闭，denoising关闭 |
| 场景 | `assets/scenes/photoreal/urban_street/urban_street_assets.blend` |
| 人体/衣物 | `outputs/baselines/temporal_case_aaron_1095/{geometry,clothing_geometry}` |
| 身体纹理 | BEDLAM `skin_m_white_01_ALB.png`，由冻结appearance manifest解析 |
| 镜头轮廓 | 真实20260320 riding-dataset3的`cam{2,4}/02214.jpg`，仅提取二值轮廓 |
| 模型 | `assets_external/densekp.ckpt`原始权重，严格Lightning加载 |
| 推理 | GPU0，FP32，无autocast，TF32关闭；完整256-square输入，模型内部规则不变 |

场景色彩管理只读复核为AgX、look=None、exposure=-0.65、gamma=1、display=sRGB。渲染器写32-bit multilayer EXR的Combined/Depth/IndexOB，同时写经过场景色彩管理的preview PNG；当前DS路径按既有行为用preview PNG作RGB，用EXR作原始射线距离和IndexOB。RGB没有改成从线性EXR读取。

历史旧图只作来源参考，没有直接复用为A。旧源渲染没有冻结全部外部纹理/场景字节，而且早于当前完整镜头mask路径，不能证明今天能逐像素重现。因此A/B/C全部在本轮同一代码、资产和设置下重新渲染；这保证本轮内部可比，不声称重现2026-09-28旧图。

## 最小实现修改

[prepare_ds_domain](../../src/densekp_retrain/imaging/ds.py#L81)和[process_scene](../../src/densekp_retrain/imaging/ds.py#L263)新增可选`output_size=(width,height)`：

- 不传时，最终ERP继续使用源ERP尺寸，保留旧入口默认行为。
- 传入5760×2880时，最终RGB、实例、有效mask、标签和框均使用最终栅格。
- [project_ds_scene](../../src/densekp_retrain/imaging/ds.py#L112)另按源深度shape计算`source_xy`，只用它在原始源ERP的射线距离和实例图上查询可见性；最终`xy`仍来自5760×2880标签栅格。
- [remap_ds_images](../../src/densekp_retrain/imaging/ds.py#L148)保持既有插值：源RGB周期线性采样到DS，实例ID先`rint`再最近邻；DS到最终ERP时RGB线性、实例最近邻；显式mask最近邻。没有把RGB、深度和ID统一成一种插值。

新增有界入口：

- [run_ds_source_resolution.py](../../tools/experiments/run_ds_source_resolution.py)：固定三组、双相机渲染和后处理，记录耗时、采样显存和失败。
- [evaluate_ds_source_resolution.py](../../tools/experiments/evaluate_ds_source_resolution.py)：先冻结框/支持集，再加载原权重推理，保存逐例NPZ和横向对照图。

针对性CPU检查：先运行新增的源/最终栅格解耦测试与镜头mask测试，2项通过；一次组合测试因启动时没有设置`PYTHONPATH=src:.`而导入失败，修正启动环境后相关`test_imaging_pipeline`和`test_ds_scene_pair`共11项全部通过。没有运行整个测试库。

## 坐标、mask、标签与固定框检查

评估前冻结结果见[evaluation_manifest.json](../../outputs/experiments/ds_source_resolution_20261008/evaluation/evaluation_manifest.json)：

- 三组双相机最终图均为5760×2880。
- `world_densekp`、最终ERP `xy`、`in_view`、最终`valid_mask`逐值一致。
- 五行固定框的裁剪affine和裁剪GT逐值一致。
- 框来自旧clear固定框，x/y及边长按5760/1536=2880/768=3.75映射；没有针对新图重新求框。
- 五行全部评价：cam2/person2、person3、person4，以及cam4/person2、person3。person4只有cam2存在预先冻结的有效框，未补造cam4框。
- 固定共同监督共486点；跨三组始终可见200点、始终遮挡256点、可见性分类变化30点。主指标用相同共同支持集；另报告各组自身可见/遮挡分组。

源ERP可见性继续使用周期5×5源像素邻域和严格小于0.04m的射线距离容差。5×5没有随分辨率扩大：它在A/B/C对应不同角度范围，不能称为相同角度邻域。这造成了小幅可见性变化，例如cam2/person3为22/21/21点；主分组用共同可见20点、共同遮挡100点，另外3点只进入固定总支持和自身分类报告。

cam4轴向继续由现有`CAMERA_EXTRINSIC["cam4"]`处理，没有额外翻转。ERP横向周期采样、显式有效域mask、最终预测周期误差均沿用现有规则。预测第三值原样保存，不作置信度、筛选条件或指标权重。

## 图像细节

以下每张横向图均按A/B/C排列：

- 主目标局部：[detail compare](../../outputs/experiments/ds_source_resolution_20261008/evaluation/row_1_cam2_person_3_detail_compare.jpg)
- 主目标实际模型输入：[model input compare](../../outputs/experiments/ds_source_resolution_20261008/evaluation/row_1_cam2_person_3_model_input_compare.png)
- 主目标GT/预测：[overlay compare](../../outputs/experiments/ds_source_resolution_20261008/evaluation/row_1_cam2_person_3_overlay_compare.jpg)
- 完整人物实际模型输入：[person4 model input](../../outputs/experiments/ds_source_resolution_20261008/evaluation/row_2_cam2_person_4_model_input_compare.png)

`detail_compare`从统一5760×2880最终图按同一空间框取局部，再仅为展示缩放；`model_input_compare`来自Dataset实际保存的256×256 `img_disp`，只用nearest放大2倍显示。后者证明更多边缘/衣褶信息确实进入模型输入，而不只是展示端把低分辨率图放大。

对五个实际256输入计算的Laplacian方差均值为A/B/C = **696/1301/1641**，平均绝对Laplacian为 **14.27/20.46/22.25**。人工图中B/C的轮廓、手部和衣物边缘也更清楚。不过该指标会同时响应Cycles采样噪声，不能单独当作“有效语义细节”或模型效果指标。

## DenseKP结果

完整机器可读结果见[evaluation/report.json](../../outputs/experiments/ds_source_resolution_20261008/evaluation/report.json)，逐点预测、GT、affine、误差和mask位于`evaluation/{A,B,C}/*.npz`。15个预测共6210个输出值全部有限。

### 固定共同支持集汇总

| 分组 | 点数 | A crop / ERP px | B crop / ERP px | C crop / ERP px | B相对A | C相对A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 全部共同监督 | 486 | 14.910 / 45.349 | **14.493 / 43.742** | 14.740 / 44.469 | -2.80% / -3.54% | -1.14% / -1.94% |
| 共同可见 | 200 | 4.836 / 19.930 | **4.392 / 18.427** | 4.503 / 18.917 | -9.17% / -7.54% | -6.87% / -5.09% |
| 共同遮挡 | 256 | 23.970 / 68.477 | **23.525 / 66.559** | 23.918 / 67.582 | -1.86% / -2.80% | -0.22% / -1.31% |

这是逐点评价汇总，不是等人物加权。B在三个固定分组都最好；C相对A仍小幅改善，但没有延续B的总体收益。

### 逐样本全部共同监督点

| 相机/人物 | 角色 | 点数 | A crop px | B crop px | C crop px | A/B/C ERP周期px |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cam2/person2 | 边缘人物 | 90 | **3.414** | 3.453 | 3.647 | 20.583 / 20.818 / 21.989 |
| cam2/person3 | 重遮挡主目标 | 123 | 16.263 | 13.117 | **12.974** | 50.885 / 41.042 / 40.595 |
| cam2/person4 | 完整人物对照 | 138 | 4.122 | 3.773 | **3.649** | 8.043 / 7.362 / 7.121 |
| cam4/person2 | 边缘人物 | 93 | 5.609 | 5.388 | **5.361** | 33.819 / 32.483 / 32.321 |
| cam4/person3 | 完全遮挡目标 | 42 | **91.627** | 97.560 | 100.890 | 230.320 / 245.234 / 253.605 |

主目标cam2/person3的共同可见误差为7.382/6.072/7.334 px：B改善，但C回到接近A；共同遮挡误差为18.293/14.657/14.201 px，B/C均改善。因此主目标总改善主要来自遮挡点推断，不是所有可见点随分辨率单调变好。

预测落到显式有效域外仍保留在误差分母。cam2/person2的A组有1个、cam4/person2三组各有4个监督点的预测不在mask内；报告只另记retention，不删除这些错误。

## 耗时、显存与存储

| 组 | 源尺寸 | 双相机渲染 | 后处理 | 采样GPU峰值/增量 | 后处理进程累计RSS峰值 | 组目录字节数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 1536×768 | 69.1s | 28.2s | 3311 / 3293 MiB | 828 MiB | 120,432,141（约115 MiB） |
| B | 3072×1536 | 79.4s | 24.8s | 3395 / 3377 MiB | 850 MiB | 319,812,438（约305 MiB） |
| C | 5760×2880 | 119.3s | 23.0s | 3439 / 3421 MiB | 1194 MiB | 949,289,767（约905 MiB） |

渲染显存每0.5秒采样一次，可能低估瞬时峰值；GPU增量以启动前18 MiB为基线。后处理RSS是同一长期Python进程的累计`ru_maxrss`，不能解释为各组独立增量。后处理耗时还受LUT/文件缓存和执行顺序影响，不能据表断言高分辨率后处理更快。

B/C相对A的组目录存储约为2.66×/7.88×；双相机渲染墙钟约为1.15×/1.73×。C的两份源EXR各约380 MiB，B约109 MiB，A约28 MiB。评价目录另约4.3 MiB；整个实验根最终约1.4 GiB。

原始模型加载47.79s；15个裁剪的FP32推理合计4.77s；PyTorch记录的推理峰值显存为2,775,959,552 bytes（约2.59 GiB）。该推理成本基本不随源分辨率变化，因为最终模型输入始终是256×256。

## 执行次数与失败记录

[execution.json](../../outputs/experiments/ds_source_resolution_20261008/execution.json)保留全部记录：

- 两次渲染前失败，均未启动Blender、未产生源图：第一次是Python沙箱内`nvidia-smi -i`资源采样失败；第二次是首次续跑对已有空`A/source`目录使用了非`exist_ok`创建。两项均修正后从零渲染状态续跑，没有删除旧输出。
- 正式成功启动3次Blender，每次顺序渲染cam2/cam4；实际成功源图6张，失败源图0张。
- 后处理3组全部完成；推理15个固定裁剪全部完成；没有OOM、非有限预测或自动降级尺寸。
- GPU1已有他人的Blender/桌面进程。本实验没有终止它，正式渲染和推理均用`CUDA_VISIBLE_DEVICES=0`限制在当时空闲的GPU0。

## 评审建议与未覆盖项

代码已确认：提高源ERP分辨率能把更多高频信息带入最终256输入；B在本组固定共同支持上取得小幅总体收益，并对重遮挡主目标和完整人物对照有明显收益。代码和结果同样确认：收益不单调、存在样本退化，完全遮挡视角尤其不能靠更多像素解决。

建议停在当前评审点：

1. 不把C设为默认，不批量重制数据集。其总体指标差于B，存储约为A的7.88倍。
2. 若评审认为值得继续，只把B用于更多**预先冻结**的小规模场景确认，重点检查不同人体尺度、遮挡类型和真实框；不能据本单场景直接推广。
3. 保留DS直接渲染议题，但作为单独实验：它可能减少中间重采样和源EXR存储，本轮没有实现或验证，不能用本结果宣称会改善DenseKP。

未覆盖：真实图GT、检测框分布、更多身份/姿态/场景、训练收益、直接DS渲染，以及源分辨率与固定5×5可见性角度支持解耦。此次结果也不改变此前同图换框实验的结论。
