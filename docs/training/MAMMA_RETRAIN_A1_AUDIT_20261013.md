# MAMMA 重训 A1 只读核查与首轮方案

日期：2026-10-13。状态：**待评审；未适配生产代码、未构造或运行模型、未训练、未渲染**。本报告的目标是 2D MammaNet 微调准备，不涉及分割器重训、`ma_3d` 算法或最终合图。MAMMA 路径 `/home/yy/Repos/mamma` 实际解析到 `/mnt/data1/yy/home/yy/Repos/mamma`。工作树已有大量修改；本批只新增本报告和 `outputs/audits/mamma_retrain_a1_20261013/` 内的小文件。

## 阅读入口及活动对象

先读本仓库 `docs/training/DUAL_VIEW_TEMPORAL_PLAN_20261008.md`、`DUAL_VIEW_TEMPORAL_AUDIT_20261008.md`、`ASSET_SOURCE_POOLS_A2_20261010.md`、`FIRST_DATASET_PREPARATION_A1_20261011.md`、`FIRST_FOUR_EPISODES_A1_20261011.md`、`docs/rendering/ROI_ONLY_RENDER_A1_20261009.md`、`PERSISTENT_ROI_CONSUMER_A1_20261009.md`、`docs/CONTRACTS.md` 和 `docs/tools/SCENE_EDITOR_A3_20261013.md`。随后查 MAMMA `README.md`、`landmarks/README.md`、`landmarks/train.py`、`landmarks/configs/train/models_2d/config_mammanet_mask_512.yaml`、`model/pose_mammanet.yaml`、`data/BEDLAM_WD.yaml`、`landmarks/lib/datasets/{BEDLAM_WD,vitdet_dataset,augmentations,utils_eval}.py`、`landmarks/lib/models/models_2d/{denseldmks2d_vit,mvhead,loss}.py`、`landmarks/run_ma_2d.py`。部署以 `riding-data/run_model_comparison.py`、`mamma_mask_roi.py`、`export_mamma_trt.py`、`mamma_trt_runtime.py` 为当前代码入口，并对照 `docs/mamma/{README,YOLO_TRIAL}.md`、`riding-data/MAMMA_OPTIMIZATION.md`；旧报告中的实验参数不自动视为活动默认值。没有发现适用的 `AGENTS.md`。未读取凭据。

`run_model_comparison.py:164-311` 的活动比较入口默认使用 COCO Mask R-CNN 人物框和 mask，置信度 ≥0.8、至多四人，默认提前筛选 mask；无 `--mamma-trt-engine` 时使用 PyTorch MammaNet，显式传 engine 才切 TensorRT。`--mamma-max-person-box-area` 默认未启用。YOLO26s-seg 是单独试验，文档明确未替换正式检测器。MAMMA 原 `ma_2d` 管线和此比较入口并不完全相同；本批原权重基线须固定其中一种路线和输入。

| 对应项 | 部署输入/输出与现状 | 训练实现 | 现有数据与缺口 |
| --- | --- | --- | --- |
| 权重/版本 | PyTorch 路线严格加载 `mamma/data/weights/ma_2d/mamma_mask_full_cvpr.ckpt` 的 `state_dict` (`run_model_comparison.py:199-206`)；本地文件约 1.6 GB。导出脚本同一 checkpoint 严格加载，再导出 ONNX 与 FP16/FP32 TensorRT；本地 ONNX、FP32 engine 存在。engine 是推理二进制，不能作为训练 checkpoint。 | `DenseLdmks2DViT` + `pose_mammanet`，ViT-B 风格骨干、6层 MammaNet decoder。`train.py:131-135` 的 `model_checkpoint` 仅 `strict=False` 初始化模型参数；`resume_from` 字段未传给 `trainer.fit`，不恢复 optimizer/scheduler。ViTPose 文件路径 `data/weights/vitpose/vitpose-b-multi-coco.pth` 本机缺失，推理构造时会跳过，再由完整 MammaNet checkpoint 覆盖；训练新模型不能把它当已加载。 | 原 checkpoint 与部署加载路径已对应，但**未按本批禁令加载 1.6 GB 权重或比较 state_dict key/版本元数据**。下一批记录 checkpoint 内配置、epoch/step、strict key 对照及导出物的实际生成记录。不能凭文件名证明所有历史 engine 与当前 checkpoint 字节相同。 |
| 图像/框/mask | 检测输入 BGR 原图缩至最长边 1600、转 RGB；MammaNet 裁剪从原图 BGR 由 `ViTDetDataset` 构造。Mask R-CNN 概率 mask 在局部放大，先 0.45 找局部，再 0.5 二值化；`person_batch` 以同一仿射采成一通道 mask。 | `config_mammanet_mask_512.yaml` 的 `image_size=[384,512]` 是宽高，`img_size=[512,384]` 是高宽；网络 `img [B,3,512,384]`、`mask [B,1,512,384]`。训练 WebDataset `.decode('rgb8')`、RGB 均值/方差为 ImageNet 数值乘 255；部署 BGR 裁剪后转 RGB 用同值。 | `roi.png`、局部人物 mask、局部 physical mask 可引用；框来自**渲染实例 oracle**，分布不同于预测 mask/框。需定向检验 mask 二值化和训练 GT mask 对应，基线固定预测或 oracle 输入。 |
| 裁剪与还原 | `ViTDetDataset:35-100` 用 `xyxy` 中心、原宽高乘 1.2，再扩成 384:512 宽高比；高斯预滤波由下采样比决定，`fast_inference` 只取带 `ceil(4σ)+3` 边界的源局部。仿射 `generate_image_patch_cv2`，常数边界填充。`run_model_comparison.py:267-270` 将预测 `[-1,1]` xy 先转 `[0,1]`，乘 `box_size` 加中心减半，回完整 ERP。 | `BEDLAM_WD:232-356` 用保存的 center/scale，`scale/1.2*scale_crop`、仿射到 384×512、RGB 归一化、点归一到 `[-1,1]`；BEDLAM square 特例 `scale_crop=1.3`。训练极端裁剪、缩放、平移、旋转、颜色与 mask 增强另行生效，与部署输入不完全一致。 | DenseKP 是 256 方形、不同仿射。需新的矩形 adapter；完整 ERP xy、局部 ROI xy 和模型 xy 独立存/换，不在每个样本重建全 ERP 空画布。 |
| 输出头 | `mvhead.py:125-186` 各 512 个 `joints2d[xy,third]`、visibility、contact、floor_contact。比较部署 ONNX/TRT **仅导出** joints2d `[1,512,3]` 和 visibility `[1,512,1]` (`export_mamma_trt.py:47-83`)；原 `run_ma_2d.py:223-239` 可输出四头。 | `pose_mammanet.yaml` 四头均启用；`denseldmks2d_vit.py:181-199` 中坐标 NLL、visibility BCE、contact 与 floor contact focal loss 相加。 | 现有 138 点与 mask 不能直接给四头标签。辅助标签未知须用独立 loss-valid mask 或冻结相应头，绝不能填零。 |

## 512 模板与身体兼容

`verts_512.pkl` 只读有界检查结果记录在 [`template_check.json`](../../outputs/audits/mamma_retrain_a1_20261013/template_check.json)：本地对象是 float32 `[512,10475]` one-hot 矩阵，每行恰一个 1、行和为 1，512 个选中顶点互异，索引范围 0–10070。`BEDLAM_WD:67-73` 取前半与后半各 `num_joints/2` 行再 `argmax`，当前 `num_joints=512` 时顺序就是全 512 行。`_smplx_512_body_parts.json` 的分区计数为 body190、左右手各63、左右脚各44、左右头各54，覆盖 512 个编号。Dataset 声明翻转对 `i ↔ i+256`；当前数据配置 `flip_prob=0`，尚未独立做镜像顶点坐标验收。`MammaNetDecoder` 的 512 query 顺序直接输出，未做后置身体部位重排；Dataset 的 `parts2body_idx` 在此路径未使用。`DrawUV` 也读取模板矩阵，但其 UV 对齐不替代几何核证。

这一定义是**有序 SMPL-X 顶点选择**，不是 R138 插值。不能从 R138 插值出 MAMMA 点；同为 10475 顶点也不足以判定编号语义。当前 e01/e02/e03 所用五个资产引用（M01、M02 两次、M03、M04）的身体是 `[T,10475,3]`，有序面 `[20908,3]` 在五份文件间逐元素一致，512 索引均在范围内。`docs/data/bedlam/BEDLAM_R138_COMPATIBILITY_A2_20261009.md` 给出了本地 standard 与 neutral locked-head 的有序面、顶点组及 R138 对应审计，但该结论**没有检查 MAMMA 512 模板**。MAMMA README 指向 locked-head 身体，然而 `mamma/data/body_models/smplx_locked_head/` 本机不存在；模板所依据的模型版本、每个选中顶点在 standard/当前骑行/locked-head 中的同一语义，仍是 pending。下一批从可信来源拿模板对应的有序面或少量编号顶点/UV，逐资产核实，不重新拟合身体。当前仅能说“可尝试按 index 重算投影，正式监督资格待证”。

## 标签与 loss 的实际语义

`BEDLAM_WD:205-367` 将完整网格的 `vertices2d`、`vertex_visibility`、`floor_contact_mask` 按所选顶点取样；若 `sdf_vertices` 是数组，contact 为 SDF `<0.01`，并乘 `(1-visibility)`。若缺 SDF，原 Dataset 把 contact 置零；这只是历史实现，**本数据未知 contact 不能照搬为零**。`floor_contact_mask` 缺失时仅打印信息，后续变量可能未定义，故不是可安全省略字段。`split_data` 还剔除 `person_visitiblity_rate<0.15` 的人物，不能把该策略无条件用于边缘/遮挡任务。

坐标 `target_weight` 依据裁剪后是否在输入范围：内点 1，外点按距离指数衰减，手/脚加倍、手乘 `hand_weight`；**不因 `vertex_visibility=0` 而把坐标权重清零**。遮挡但投影在裁剪内且有可靠几何的点仍受坐标监督，符合目标方向，但域外点的非零衰减与本仓库 DenseKP `in_view & inside_full_crop` 二元权重不同。MAMMA 原目标保持现状记录；下一批另设计显式 `geometry_defined ∧ in_physical_domain ∧ in_crop ∧ crop_supported` 及可选遮挡点策略，需评审后实施，不静默改变原损失。

`loss.py:37-73` 对第三值命名 `pred_log_sigmas`，取 `sigma=exp(third)`，坐标项 `weighted squared error/(2 sigma²)`，另有加权 `2*third`，坐标总项乘 `loss_weights.joints2d=100` 与 `total=1`，误差项上限 25；visibility BCE、两个 focal loss 以系数 1 直接加在总 loss 上。**按训练 loss，第三值是 log sigma，不是 log variance，也不是 DenseKP 的第三列权重。** 但 MAMMA `run_ma_2d.py:251-253` 的可视化用了 `sqrt(exp(third))`，`optimization/utils/triangulation_functions.py:37` 在 `channel_is_logvar=True` 时也用 `exp(0.5*third)`；后处理对其意义有不一致证据。当前比较入口将第三值原样保留，跨视角评估须把它与真实几何误差分开，并在下一批查明实际 `ma_3d` 参数及下游用法；本批不修改拟合算法。

`train.py` 默认 4 GPU、`max_steps=10,000,000`、每 1000 step checkpoint，不适合小试。当前 `configure_optimizers` 仅把 decoder 和可训练 backbone 加进 AdamW，**mask_downscaling 参数未加入 optimizer**；配置中的 `lr_backbone` 没用于该函数的 param group。下一批要显式列 key 匹配检查、参数组与 optimizer 恢复测试，不能借 `strict=False` 让不匹配权重默默通过；不可调用默认训练入口。

## 已有三段数据的实际复用边界

[`reuse_inventory.json`](../../outputs/audits/mamma_retrain_a1_20261013/reuse_inventory.json) 从 `common_parent_rows.json`、五份 `sequence_index.json` 和实际文件做了只读核算：e01 62 条，e02 背/正 67+106 条，e03 背/正 101+99 条，共 435 条合法父行；e00 仅几何拒绝，没有图像或训练行。e02 有 55+16 `person_absent` 与各 2 `empty`，e03 有 21+21 `person_absent` 与 2+4 `empty`，不能补框替代。报告中的 train/val 只是候选池，未形成正式训练集。

各段预选两条、共六条观察槽位：e01 首帧双相机 p1；e02 正面 rig 首帧 cam2 行人 p2 有框、cam4 同人处于物理域外无框；e03 背面 rig 中帧 cam2 行人 p2 与正面 rig 中帧 cam4 骑行 p1。五个有框槽位实际 `roi.png`、局部人物/valid mask、`annotations.npz`、`raster_storage.json` 均存在；无框槽位仍有相机 ROI/valid/annotation，但无该人局部 mask。所见图像和 mask 的存储宽高一致；六槽的实际尺寸、mask 正像素数、引用路径、源整数帧及矩阵行见 inventory。小范围额外核对 e02 正面 rig 的 f15、f17、f25、f30 行人共同 138 点：f15 cam4 为 0 个域内点；f17 两路共同物理 5、共同可见 0；f25 为 135/22，f30 为 138/45。它可作为后段任务样例，不能因首帧不完整整段删除；这些仍是 **R138** 证据，不可冒充 MAMMA 512 对应数。

可直接引用的是 ROI RGB、局部 oracle 人物实例 mask、局部 valid mask、完整图逻辑尺寸/ROI offset/coverage、紧实例框、相机 rig/逐帧变换、源资产和精确逐人物时间映射。`raster_storage.json` 明示 ROI 外是 unknown；`physical_mask_ref` 指向完整物理有效域，不应由 RGB 黑色猜测。`annotations.npz` 保存 R138 的 `xy/in_view/visible`，不是 MAMMA 标签。实例 ID 源于渲染者 `pass_index`：`render_road_pass_sources.py:89-90` 给身体、衣物同一人物 ID，`groom.py:59` 给头发同 ID，`raster_storage.py:43-50` 在 physical mask 内形成该人可见局部 mask。因此 GT mask 含可见衣/发像素，遮挡处不恢复不可见表面；部署 mask 是 Mask R-CNN 预测阈值后的近似目标，误差分布不同。当前局部 mask 与 MAMMA 输入的精确 nearest/阈值采样一致性仍需小试。

从 `body_vertices[asset_frame_index]` 取经**确认兼容**的 512 个点，再用同一逐帧 `scene_from_model`（或 legacy `model_to_world`）、相机 CV `camera_to_world`、ERP 公式投影，原则上可重算完整 ERP 512 xy。须用 `person_timeline_ref` 的整数 source/asset frame、`scene_matrices_ref` 的人序号和相机 rig 一一核对，不能假设合成 10Hz 等于 KIT 或骑行源帧率。只有 R138 或参数并不足以跳过模板/根所有权检查。compact 已清理部分源 EXR/depth/instance；投影可从身体缓存重算，但 512 点被衣服、头发、邻人和场景遮挡的准确标签**不能从 138 点 visibility 或最终 RGB 推导**。最小补充方案是在未来公共消费者已有深度/实例缓冲仍活着时，按已验证模板投影 512 点并保存逐点 known/in-domain/occluded 的小数组或证据；无需长期留全尺寸 EXR。旧 compact 若无法重播缓存，就把 visibility 标 unknown、屏蔽其 loss，或经授权做小范围补采。

已渲染 ROI 对**435/435 当前原生 oracle 框**的部署矩形支持经解析核算通过：`1.2×` 框、384:512 扩张和 `ceil(4σ)+3` 源采样边界都在 ROI 和完整 ERP 内；每 rig 计数在 inventory。该检查依据 `vitdet_dataset.py:35-86`，未运行像素仿射，也**不证明任意扰动框或 MAMMA 原训练旋转/极端裁剪均被覆盖**。原先 435/435 DenseKP 方形 `bounded_native_box_v1` 整族支持只是另一裁剪合同。后续每个新框先求实际 MAMMA 采样矩形及滤波 halo，超出 ROI/ERP 明确拒绝或补渲；不补黑、不静默缩框。生成端应预留一个审核后的矩形框族，现有原生框可先做接口小试。

`FIRST_FOUR_EPISODES_A1_20261011.md` 记录 e01/e02/e03 视频技术生产完成但观感待评；e02 `M04_KIT425_walk_replay_v2` 是新缓存，source10–100，旧已展示片段不是同缓存且 source9–99，不能转移人工接受范围。KIT11/KIT425 仅已确认录制路径不同，演员独立与预训练暴露未知。旧多人、路口和 A3 新产物也不能自动扩大本批候选的正式资格。

## 与 A3 的交接和文件所有权

机器可读的 [`interface_draft.json`](../../outputs/audits/mamma_retrain_a1_20261013/interface_draft.json) 将现有字段、MAMMA 新增必需、可选诊断、compact 单独不可恢复项分开。A3 当前 `tools/scene_editor/` 已能保存道路控制点、rig/相机、人物组件引用、整数源帧、单时刻预览及完整 episode，seed 目前只记录，组件重抽未接通；这些功能不被本报告扩成生成改造任务。A3 拥有编辑器、其 recipe/服务和当前产物；未来明确分工后的公共生成端负责人拥有 live buffer 512 visibility、模板拓扑证据与 MAMMA 框族 ROI 预留；A1 拥有新 Dataset adapter、点投影校验、loss mask 与评价。双方不同时改 renderer/consumer/packing。交接顺序是 A1 冻结模板/小样本判据和框政策，再由指定生成端负责人加最小模型无关点级证据，A1 回读验证。生成端宜保留完整身体几何及来源，分别导出 R138 与有序 512 点，不把模型私有数组强塞进编辑器配方。

## 推荐下一批最小路线及通过标准

1. **接口小试，尚不生产适配。** 先选六槽中五个有框样本加 e02 无框分母；验模板相对资产的有序语义及逐点 ERP 投影，核对 mask ID 含衣发、物理/coverage/遮挡三状态，使用 ROI offset 做 384×512 仿射与完整 ERP 逆变换。通过标准：点序、源整数帧、相机、mask、原图回投及 native crop 逐项可追溯；像素与可信全图/官方 Dataset 定向对照在明确容差内；无框保持 null。若模板对应不成立，停在数据资格门槛。
2. **原权重基线。** 固定同一批父样本、oracle/预测 mask 标记、原生框及冻结挑战框；检查 PyTorch 严格权重加载和活动 TensorRT 导出路线的同输入数值可比性。所有结果记录后端、mask 来源和检测器，不把其变化算成微调收益。真实数据无 GT 时仅测同图换框漂移、两路共同判可见点一致性及可配对率；不报真值精度。
3. **有界微调建议。** 在数据来源资格及投影通过后，首轮 100–200 optimizer 更新，优先只训练 `decoder.landmarks` 坐标/uncertainty MLP，冻结 ViT、decoder 共享层、mask encoder、visibility/contact/floor-contact 头；这样缺失辅助标签不被伪造，且避免未知可见性被训练成零。若已补得可信逐点可见性，第二个独立受控阶段才考虑共享 decoder 与 visibility 头，并用显式 loss-valid mask；contact/floor-contact 继续冻结，除非取得真实标签。参数组应显式列出、记录实际 trainable/key count；若开放 mask encoder，必须加入 optimizer（原函数漏掉）。AdamW 起点可试坐标头 `1e-5`，其它组暂冻结；这是工程起点，不是已验最优值。按 435 行全部可用的理论上界、batch4–8，100–200 更新对应 400–1600 次样本曝光、约 0.9–3.7 个 435 行等效遍历；实际合格 train 集目前只有候选 e02 的 173 行且来源未批准，所以训练量与 GPU wall 无可靠估计。先做 10–20 step 内存/吞吐测量再冻结计算预算，不能套默认四 GPU 大步数。
4. **框增强独立 off/on。** 先冻结同一父样本、scene/来源 split、原权重、总更新数与评估框；off 为原生框，on 只改变有支持的 MAMMA 矩形框采样。固定 MAMMA 自带 scale/translation/rotation/extreme-crop/随机 mask 增强的开关及随机流，不让它们成为两组额外差异；最终策略待小试支持统计决定，不照搬 DenseKP 方形幅度。每个训练框失败须计分母和原因，绝不重抽到看似成功。

评价以完整 ERP 的 GT 定位误差、共同**几何有效** 512 编号的跨视角对应/重投影、同图换框漂移和可见/遮挡/物理边缘/完整身体分组为主，分别报人、点、相机、episode 分母。预测落域外仍记录误差与失败；网络第三值另做 calibration/排序分析，不替代真值误差。连续数据首轮逐帧训练，可离线评估时间稳定性，不引入时序网络。同场景多 rig、同动作邻帧、不同衣发版本同侧；当前 e00 缺失、环境分侧混杂、KIT 演员关系和预训练暴露未知，因此本数据只能先做流程小试，不能宣称来源独立的泛化改善。

## 评审点与下一批待办

本批停在评审点。下一批先核实模板与五份有序身体的语义、checkpoint 内版本/严格 key、实际 `ma_3d` uncertainty 使用，再批准小范围点级 visibility 补充和 adapter。需决定原生/预测框 mask 的基线口径、正式来源资格与 e01/e02/e03 视频观感，随后冻结同一父样本及评价表。只有这些门槛通过才运行模型基线、参数更新或新增渲染。本报告的 CPU 模板和五份面序核对约数秒，全部 CPU 数组检查远低于 10 分钟；没有读取大型 checkpoint、GPU 或完整资产库。
