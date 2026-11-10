# A2：冻结连续骑行资产、BEDLAM 衣物试制与公共接口交接

日期：2026-10-09。全部 **integration_only**。已接受的76帧简单衣物版本完成资产交付和CPU回读；BEDLAM已实际完成一套衣物的初始适配及一次局部修正，现有5姿态双视角对照；用户最终决定 **“先保留简单衣物，不推进 BEDLAM”**，因此不生成新衣物76帧预览。骑行公共road_pass尚缺接口能力，本轮没有执行骑行DS。

入口：[同姿态衣物对照](../../../outputs/experiments/continuous_cycling_asset_a2_20261009/review_for_user/index.html) · [已接受的原76帧播放](../../../outputs/experiments/apd_groom_flicker_20261009/full_preview/frames/index.html) · [接受版资产manifest](../../../outputs/experiments/continuous_cycling_asset_a2_20261009/accepted/manifest.json) · [给A1的接口交接](../../data/CYCLING_SCENE_INTERFACE_A2_20261009.md)。

## 1. 用户决定与保护范围

用户已观看并接受64 samples穿衣预览，包括顶部亮度、左臂快摆、轻微局部穿模。接受范围只覆盖 cycling_M01_v2、APD6526 源 `[30,106)`、asset `[0,76)` 这一样例；不代表角色库、真实重建或正式数据集通过。原身体参数、shape、root、groom绑定、材质和原时间不动，不再研究亮度/采样/降噪。旧报告的待评审状态作为历史证据保留，本轮另存最新决定。

执行期间用户补充：**“其实需要我人眼判断，如果看的过去就可以。”** 本轮据此撤回仅由助理作出的衣物质量否决：局部边长/折转是诊断提示，不能单凭数值替代用户决定。随后用户明确选择保留简单衣物、不推进BEDLAM。本轮据此停止衣物试制；这不等于用边长比自动否决整个BEDLAM。当前状态以 `review_status.json` 为准；`pre_user_triage.json`、`bedlam_candidates_pre_user_review.json`、两版原 `probe_decision.json` 保留为被新决定取代的中间判断，不能当作用户拒绝记录。

已读身体、穿衣、闪烁报告/manifest/配方/time、A1旁表、共享计划、ASSETS/CONTRACTS/DS模式和新行人交付报告。检查Git与适用AGENTS路径，未发现AGENTS。保护TODO、README、prompt、A1文件/测试/行人输出、外部pano-stitch与历史资产；仅新增A2独立脚本、文档和本轮目录。没有修改公共生产模块、角色源、原fixture，没有SHA256、commit、下载或安装。

最终size/mtime检查发现并行期间 `docs/history/prompt.md`、共享总计划、A1的 `sequence_packing.py` 有外部变化；A2没有写入这些文件，也未回退。已重读新版packer，其固定tick/30Hz源时间假设仍在，因此接口缺口结论不变。TODO和其它已记录的旧输入无变化。详细差异见 `protected_after.json`。

## 2. 已接受版：可复用资产交付

资产ID：`cycling_M01_v2_APD6526_30_106_simple_v1`。权威仍是 `outputs/experiments/apd_6526_smplx_pilot_20261008/fit_attempt_2/parameters.npz`。固定 male、30维betas、10475身体顶点/20908面、原顶点顺序。模型具体语义发行号unknown，引用实际本地文件和旧model_reference，不编造版本。

`accepted/manifest.json` 沿用A1已有的 asset_id、character_manifest、character_family、geometry、clothing_reference、groom_reference、parameters、source_ref、available_source_frames等名称；时间/root/材质扩展采用明确提案schema，`extensions_effective_in_A1=false`，不能视为A1已实现。

| 组件 | 权威／消费表示 |
| --- | --- |
| 身体 | 原参数为权威；geometry中的身体、joints、faces、R138是指向稳定旧缓存的链接 |
| 衣物 | 原simple_clothing_v1，2612顶点/5018面；固定重心绑定＋4mm归一化法线偏移，原面集不删改 |
| 衣物材质 | 原GarmentRest连续shader与rest坐标；没有独立UV，不伪称验证了UV |
| Groom | 原1591400参考点/63656曲线；固定active集合63652曲线/1591300有效点；只引用一份points/radius/offsets和76个head矩阵 |
| 时间 | 原整数timestamp、source_frame、asset_frame、episode_tick、time_base与单位推断状态；没有改名冒充均匀帧率 |
| 接受记录 | 最新用户接受64版studio；局部问题继承旧穿衣记录，不反复渲染旧完整视频 |

原时间首末 `1476871677560014`–`1476871680560014`，跨度3s；间隔39949–40010 ticks，微秒依据APD25Hz推断，仍非官方单位确认。原演员身份unknown，目标角色不声称还原APD源人物。

世界变换由场景拥有：身体/衣物/138用同一 `scene_from_model[t]`；头发用同一scene乘已包含global/root的H。源髋平移已移除，pelvis固定，不叠头部路径、不逐帧min-Y贴地。未来道路高度必须显式固定给出。详见接口交接。

## 3. 全76帧实际CPU验证

新脚本直接只读复用A1 `body.sequence_binding.bound_garment/head_transform`，并重新加载冻结SMPL-X求值；没有优化器或逐帧shape变化。结果见 [accepted/validation.json](../../../outputs/experiments/continuous_cycling_asset_a2_20261009/accepted/validation.json)、逐帧记录及独立check-only回读。

| 检查 | 本轮实际结果 |
| --- | --- |
| 参数→身体对旧权威缓存 | 最大2.3841858e-7m，低于原2e-5m容差 |
| 原R[138,10475]→138对旧结果 | 最大1.1920929e-7m，低于原2e-6m容差 |
| 公共衣物函数→旧76帧缓存 | 最大差0 |
| 公共head函数→旧head矩阵 | 最大7.5731712e-8 |
| 固定任意scene矩阵下R与body变换一致 | 最大1.4305115e-7m；只是代数测试，不是道路投影 |
| Groom | 全76时刻变换参考点有限；固定曲线集合，scene@head复合等式通过，无重复root/head |
| 时间/shape/拓扑 | 原76条逐项对应，固定betas、faces与模型一致；所有求值有限；不新增贴地 |

继承已接受的毫米级局部穿插、小面积负面法向和左臂快摆，不以零相交为门槛。身体标签仍由R@body产生，衣物不改变标签。CPU成功不声称完成DS视域、像素投影、框或Dataset回读。

可复跑只读验证：

```bash
python tools/experiments/prepare_cycling_asset_a2.py accepted --check-only \
  --validation-output /tmp/cycling_a2_readback_new.json
```

目标输出必须是新路径。已实际运行该入口，76帧再次通过。

## 4. BEDLAM：实际适配及用户决定暂不采用

本轮只检查并选择**1个具体衣物候选**，没有为凑数扩到3个。由既有P0 manifest定位：

`/mnt/data1/yy/home/yy/datasets/BEDLAM/clothing_data/tiny/rp_aaron_posed_009/clothing_simulations/1095/1095.npz`

它确有可读网格，不只是RGB或uasset：39帧、6Hz；13027顶点、26328面、14776 UV点和26328 face-corner UV面。对应源body为male，192帧/30Hz；衣物0对应body local0、原source101，元数据start101/end292；缓存39帧每5个body帧采样，末对应291。实际复核原NPZ第0帧与旧提取衣物逐值相同，对应body参数用当前同拓扑模型回读通过。原texture_01 diffuse/normal均本地存在，保留UV不翻V，sRGB/Non-Color、normal strength1、roughness0.58、metallic0。

造型是带帽拉链上衣＋长裤，非简单shell改色。没有另开Unreal、下载或解压大包。查询范围只含已知docs/manifests、third_party引用、BEDLAM tiny具体目录和两个archive目录的顶层清单、BEDLAM2 groom inventory；未扫描磁盘。

### 来源许可

已读取[官方BEDLAM数据许可](https://bedlam.is.tuebingen.mpg.de/license.html)：明确覆盖衣物网格和纹理，限定非商业研究等用途并限制传播；不是代码MIT。许可对修改/向第三方提供有约束，本轮现有本地研究资源的内部适配不构成派生物公开发布许可。保留该不确定性，商业、传播或正式训练用途不在此批准。[官方衣物入口](https://bedlam.is.tuebingen.mpg.de/clothingsim.php)由本地官方downloader指明，本轮网页工具未能读取其内容，未伪称核实下载页细节。

署名：Black, Patel, Tesch and Yang, *BEDLAM*, CVPR2023。APD CC BY4.0与SMPL-X、目标shape、皮肤、groom许可分别保留；见 `ATTRIBUTION.md`、`license_evidence.json`。

### 初始绑定

在计算前固定源缓存第0帧，不按适配结果挑选source参考。先在对应已变形body上找衣物近表面三角形及重心，转移原body的骨骼权重；将cloth−body_anchor偏移通过源混合骨骼旋转逆变换，再用目标固定shape身体锚点和目标每帧骨骼旋转变形。

```text
offset = inverse(sum(skin_weight * source_global_rotation)) @ (source_cloth-source_anchor)
target_cloth[t] = target_body_anchor[t] + sum(skin_weight * target_global_rotation[t]) @ offset
```

这保留源服装细节/褶皱，属于从非rest缓存得到的近似绑定，**不是官方rest恢复、物理布料或源动作原样套用**。body不改。绑定矩阵可逆性、源参考重建及固定faces/UV检查通过。

固定5探针：asset `[0,6,16,38,75]`／source `[30,36,46,68,105]`，覆盖首末与左臂快摆。初版5姿态有限、拓扑固定；99%边长比约1.78–2.01，少数边最高10.779倍；近表面法向诊断最小约−44.3mm，低于−2cm的点最多约0.207%。这些不是精确碰撞证明，也不是贴身shell毫米门槛，需结合服装轮廓看。

### 唯一局部修正

针对初版可见袖部锯齿及少数>4倍边，冻结五探针异常边端点＋两圈邻接，共671/13027顶点。仅对该局部衣物变形场采用固定线性邻域算子（权重0.5、12次），保持近似目标rest和全段同一集合；不改body、UV、拓扑，不按帧裁面，不做时间平滑。原版保留。修正最大位移约10.19cm，5帧最大边比反增至15.410；助理静态观察认为袖肘局部外翻更突出，但**用户观感才是最终选择依据**。

两版最初标记 `candidate_visual_review_pending`，用户选择后标记 `not_selected_by_user`；已呈现相同姿态/视角/灯光/64samples的5帧对照。网页明确是离散探针，不能冒充连续播放；尚未生成新衣物76帧动画，也没有动态人工验收。用户现已决定不推进BEDLAM，因此本轮不生成完整新衣物预览。局部修正额度已用完，不追加第三版或身体优化。

元数据执行错误单列：初次texture路径层级写错，被存在性assert捕获；保存初始日志/绑定后，仅修正路径解析并复用同一binding恢复，未重新绑定。它不等于衣物资源缺失。两版binding与探针现场均保留。

## 5. 公共生成器实际完成情况

已只读核实A1最新行人报告、实际manifest/源代码：行人已完成3时刻/6张C分辨率ERP、compact与6条Dataset回读；尚未完整两段或训练。这些是**A1行人证据，不是本轮骑行证据**。

对APD尚缺：native不均匀时间表贯通；固定pelvis时由场景逐时刻负责前进；骑行固定放置高度及不同于步行脚底的地面检查；consumer/packer真实源tick传递。A1旁表schema已经允许非均匀时间，不需另造索引。BEDLAM真实UV材质入口是可选后续能力，不阻塞已接受shell版本。

因此，本轮没有调用road_pass做虚假的“原地穿行”，没有绕过脚底检查、伪造walking参数/100fps映射或复制DS链。实际骑行scene数=0、最终ERP=0、DS/packed/Dataset回读=0。准确文件/调用点、建议字段和后续3时刻预算见[接口交接](../../data/CYCLING_SCENE_INTERFACE_A2_20261009.md)。这属于明确等待A1的接入项，不以manifest代替实际消费。

## 6. 资源、存储与交付边界

计算前通过主机查询GPU/进程/磁盘（沙箱内驱动不可读，使用只读主机查询），GPU0 18MiB/0%，GPU1已有Blender PID43491，不干预。两次预览前刷新检查，使用 GPU0 UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`，CUDA只暴露这一卡；CPU求值2线程并隐藏CUDA。结束GPU0恢复18MiB/0%，GPU1占用变化未干预。repo约12TiB、/tmp约333GiB可用。

| 阶段 | 实测 |
| --- | --- |
| 接受版76帧首次CPU核查 | 6.91s，峰值RSS671768KiB |
| 独立只读76帧复查 | 6.85s |
| 初始绑定后元数据恢复＋5姿态核查 | 2.34s，峰值RSS720076KiB；初次失败wall未单独保存，不虚构为0 |
| BEDLAM初版5时刻×2方向 | 36.42s，setup14.57s，峰值RSS2438216KiB |
| 局部修正版5时刻×2方向 | 36.71s，setup15.14s，峰值RSS2458288KiB |
| 角色预览实际渲染合计 | 73.14s，20张512×512；没有重渲染原接受版 |
| 全76新衣物预览 | 未运行；按每版5帧最慢双方向实测×76×1.5＋setup＋60s估算，初版678.03s／修正版663.16s，均低于1800s；用户选择停止，与预算不足无关 |
| 骑行DS | 未运行；预算900s尚未使用 |

RGB旧接受版直接引用；两版新探针、源绑定、一次局部算子、少量对照PNG及执行记录保留。没有76份groom或角色整套副本。两版未选用探针现场和所有旧文件均保留，本轮无清理。新增outputs约14.6MB，精确统计与保护结果见 `delivery_validation.json`／`resource_record.json`，链接目标大小不计作复制量。RSS不是VRAM，未连续采样GPU峰值，不编造显存峰值。

新增代码只有 `prepare_cycling_asset_a2.py`、`preview_cycling_bedlam_a2.py`、`package_cycling_asset_a2.py`。验证覆盖原参数/R回读、公共绑定函数、两版body/groom只读、UV/拓扑/相机对齐、路径/时间与保护检查；没有借本轮运行训练/推理或不相关全库测试。

## 7. 下一步与停止点

用户已选择保留简单衣物、不推进BEDLAM，无需再次确认。两版BEDLAM只保留为未选用试制，不宣称动态通过；其支线结束不阻塞已接受资产交付。A1应先合入接口交接列出的能力，再按同一公共链执行3连续原时刻、6最终ERP的有界试验。

来源组APD_5724486_v1_identity_session_unknown、角色家族cycling_M01保持；BEDLAM服装只增加派生关系，不新增独立人物/验证来源。换seed、衣服、邻段、循环或重采样不能拆成独立train/val。没有移入datasets、双路扩渲染或训练；停在本批资产和用户观感评审点。
