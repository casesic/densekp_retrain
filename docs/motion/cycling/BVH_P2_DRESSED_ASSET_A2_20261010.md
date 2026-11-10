# participant_2 直臂动作的 M01 穿衣资产

**状态：源身体的 120 帧动态和穿衣观感均已获用户接受；衣物与头发的全段数值检查通过。** 本轮仅使用 `cycling_M01_v2` 自身已有的 simple 衣物和 groom，没有修改已接受的 BVH→SMPL-X 身体、重拟合或加入自行车/道路场景。全部产物为 `integration_only`。用户随后表示“我觉得效果可以”，本次决定仅将穿衣动态人工状态更新为 accepted，见[接受记录](../../../outputs/experiments/bvh_p2_dressed_asset_a2_20261010/dressed_acceptance_20261010.json)；场景消费仍未完成，旧验证和预览字节不变。

观看入口：[120 帧穿衣双视角 HTML](../../../outputs/experiments/bvh_p2_dressed_asset_a2_20261010/preview/index.html) · [MP4](../../../outputs/experiments/bvh_p2_dressed_asset_a2_20261010/preview/comparison.mp4) · [已接受身体对照](../../../outputs/experiments/bvh_p2_smplx_pilot_a2_20261010/preview/index.html)。下文“待用户观看”记载的是制作当时的状态；用户现在已表示效果可接受。助理查看了代表静帧，未把它当作连续人工验收。

## 权威输入与时间

- 身体只读引用 [fit_attempt_1/parameters.npz](../../../outputs/experiments/bvh_p2_smplx_pilot_a2_20261010/fit_attempt_1/parameters.npz)；身体缓存、faces 和由项目原 `R[138,10475]` 生成的 138 点保持原字节。原报告与 [用户身体接受记录](../../../outputs/experiments/bvh_p2_smplx_pilot_a2_20261010/body_acceptance_20261010.json) 未改写。
- BVH 源帧 `[5700,5820)` → asset `[0,120)`。逐条保留整数 `time_base=1/10000000`、源/相对时间步长 `333333 ticks`，即发布 BVH 头的 `Frame Time=0.0333333s`；这不是实际曝光 PTS。[时间表](../../../outputs/experiments/bvh_p2_dressed_asset_a2_20261010/time_mapping.csv) 与身体原表一致，没有改为 1/30、插帧、变速或循环。
- 身体已为 M01 标准 male SMPL-X 30 维固定 shape、模型 Y-up、pelvis 居中。原 BVH 的固定轴/尺度适配已在权威身体内；本次不再应用、不取用源 root，也不逐帧按脚底贴地。未来道路行进、道路 yaw 和一次固定骑行高度由 scene 决定。

## 绑定与逐帧生成

[冻结配方](../../../outputs/experiments/bvh_p2_dressed_asset_a2_20261010/frozen_recipe.json) 指向 M01 本人的 reference，不从 APD 复制逐帧衣物或 head 矩阵。公共 `body.sequence_binding.bound_garment` 使用原 `simple_clothing_v1` 的身体三角形重心绑定/权重，沿当前帧身体法线作原有 `4 mm` 偏移；衣物顶点、面集、绑定和程序化 GarmentRest 材质固定。每帧服装几何保存于新资产的 `full/clothing_vertices.npy`。不存在逐帧删面、换 UV、重做衣型或布料模拟。

头发引用 M01 的同一 groom_v1：原 63656 条曲线，固定 active 63652 条、1591300 个有效点。参考空间是 M01 shaped-rest、pelvis-centered 模型。公共 `head_transform` 逐帧从**本次**身体的 global/body 旋转与 head15 关节求得 `H=[Q, J_head(t)−Q J_head(rest)]`，故身体整体朝向只进入一次。只保存一份原参考引用和 120 个小型矩阵；未来身体/衣物/R138 用 `scene @ model`，头发用 `scene @ head`，不复制 120 份毛发。无头发物理或逐帧重新筛曲线。

## 检查结果与限制

[全段验证](../../../outputs/experiments/bvh_p2_dressed_asset_a2_20261010/validation.json) 显示：重新求值的身体对权威缓存最大差约 `3.58e−7 m`，R138 最大差约 `1.19e−7 m`；源姿态、betas、transl 和时间数组未改。原 reference 身体与衣物绑定配方回读通过，服装固定 2612 顶点/5018 面，全部值有限。head 矩阵刚性、固定 active 集合及任意场景矩阵的 `scene @ head` 复合代数通过；头皮锚点最大近似差约 `8.63 mm`。衣物相对绑定基面的局部反向面最多 33 个；这是问题定位量，不是完整人体相交统计。轻微局部穿插、少量褶皱可以由用户评审接受；明显拉飞、脱体或几何闪烁仍需否决。

代表探针冻结为 asset `0/35/60/90/119`：首末、中段、膝部极值和另一晚段时刻。CPU 全段准备后渲染 5 帧两方向，静帧检查未见明显错坐标、衣物大面积脱体或头发留在原地。完整预览保持 512×512 每方向、Cycles 64 samples、无降噪、固定 seed/灯光/色彩与相机视野，复用 5 帧探针。画面标签里的 front/side 是 studio 展示方向，以渲染 manifest 中的 camera_from_model 为实际相机依据，不将其解释为标定的真实源摄像机视角。编码沿原 BVH 标称时间，容器 PTS 舍入单独记录。穿衣动态尚待用户实际观看。

## A1 读取与来源边界

[组合交付](../../../outputs/experiments/bvh_p2_dressed_asset_a2_20261010/delivery_manifest.json) 保存父身体、M01 角色家族、原 session `2_RB_M27_20230809_PA`、模型/shape、原时间、body/cloth/R138、材质/groom 以及几何范围与身体最低点。当前身体最低点在模型 Y 轴上约为 `−1.012…−0.876 m`，供 A1 选择整段固定高度。A1 的现有 `dynamic_scene.adapt_proposal`、`geometry_paths`、`select_timeline` 已只读读取 [组合提案](../../../outputs/experiments/bvh_p2_dressed_asset_a2_20261010/a1_proposal.json) 的 120 条原时间与全部几何路径；`person_time.map_timeline` 在与本资产时间完全相同的合成校验时钟上选回全部原帧，量化误差为 0，记录在 [接口检查](../../../outputs/experiments/bvh_p2_dressed_asset_a2_20261010/a1_adapter_check.json)。**这只是资产适配器读取通过，不是道路场景消费成功**：尚须 A1 为这个动作选一次固定骑行高度、明确 scene 矩阵/episode 与人物时间映射，再做相机、ROI、观测及打包检查。本轮没有伪造场景矩阵、图像或 packed rows，也没有把 APD 的 `1.2 m` 高度冒充本动作验收值。

源来自 [Kyriakou 等的 *Dataset of Motion Capture of Cyclists*（Zenodo 10668611）](https://zenodo.org/records/10668611)，本地官方元数据记录 CC BY 4.0；SMPL-X 模型和 M01 角色/材质各有自身许可。来源组 `zenodo_10668611_participant_2`，M01 家族也出现在 APD：不同 motion、换衣、邻段、变速和 seed 不自动形成独立 train/val。本组合只追加到 A2 组合目录，不改变旧条目的验收事实。

## 资源和保护

准备和渲染前分别记录 GPU、显存/进程、内存及磁盘。GPU0 后来出现其它任务占用，实际低成本 studio 预览改用当时空闲的 GPU1（`GPU-6f046d56-acbe-d49b-774a-9d463e768cac`），未干预任何进程。沙箱内 Blender 不能枚举 CUDA，因此一次未渲染即失败的目录和[失败记录](../../../outputs/experiments/bvh_p2_dressed_asset_a2_20261010/render_failure_record.json)保留，随后在批准的 GPU 执行环境中完成探针/全段。[资源记录](../../../outputs/experiments/bvh_p2_dressed_asset_a2_20261010/resource_after.json)：CPU 准备 `6.87 s`，探针 `37.03 s`，补渲其余 115 帧 `497.98 s`，封装/PTS 检查 `2.53 s`，实测计算合计约 `544.41 s`（未计未出图的失败启动），低于 1800 秒预算。新增目录约 `52.45 MiB`，没有复制模型、源 BVH、整套角色或逐帧完整毛发；未降低 64 samples 或缩短 120 帧。

复用命令（输出已存在时准备脚本拒绝覆盖，不可直接重跑本次产物）：

```bash
/home/yuyue/anaconda3/envs/camerahmr/bin/python tools/experiments/prepare_bvh_p2_dressed_a2.py
CUDA_VISIBLE_DEVICES=GPU-6f046d56-acbe-d49b-774a-9d463e768cac /mnt/data1/khb/app/blender-5.1.1-linux-x64/blender -b -t 4 --python tools/render/preview_apd_6526_dressed.py -- --stage probe --asset-root outputs/experiments/bvh_p2_dressed_asset_a2_20261010
CUDA_VISIBLE_DEVICES=GPU-6f046d56-acbe-d49b-774a-9d463e768cac /mnt/data1/khb/app/blender-5.1.1-linux-x64/blender -b -t 4 --python tools/render/preview_apd_6526_dressed.py -- --stage full --asset-root outputs/experiments/bvh_p2_dressed_asset_a2_20261010
/home/yuyue/anaconda3/envs/camerahmr/bin/python tools/visualize/package_bvh_p2_dressed_a2.py
/home/yuyue/anaconda3/envs/camerahmr/bin/python tools/experiments/package_bvh_p2_dressed_asset_a2.py
```

实际渲染使用 `--render-folder renders_gpu_retry`（探针）和 `renders_gpu`（全段）；GPU 可见性受执行环境控制，上述命令只是复用入口，不应覆盖已保存输出。衣物与头发动态评审后停止，不自动进入道路/participant_1/训练。
