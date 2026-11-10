# A1：高分辨率框接口与历史分辨率兼容

日期：2026-10-08。状态：本批实施及有界 CPU 验收通过，停止于 A1 评审点。

## 范围与结论

三个历史调用方已显式固定最终 ERP 为 1536×768；新生成入口仍默认 C 分辨率和 compact。新增显式尺寸的框增强接口，可以消费已经清理临时源的 5760×2880 compact 数据。旧框算法的缺省尺寸、浮点运算顺序及历史 trial 1.2 扩张保持。

真实消费者验收只读复用 `outputs/acceptance/ds_output_modes_20261008/attempt_4/compact/`：6 个父框、每框 4 个候选，共 24 个新框，全部通过支持集检查、打包和实际 Dataset 回读。图像与 138 点使用同一 affine；遮挡点监督规则保持。没有渲染、动作制作、模型构造、权重加载、推理、训练、参数更新、下载、安装、commit 或 SHA256。

已读取总计划、审计、CONTRACTS、DS 输出模式及接口草案；草案仅作提案参考，没有实施序列、动作、周期裁剪等其他部分。仓库及相关目录未发现适用的 AGENTS.md。保留初始 `docs/README.md`、`docs/TODO.md` 修改和规划/审计未跟踪文件；A2 新文件未修改。未编辑共享总计划、导航、README、body/transforms、衣物/groom、动作实现或序列 driver。

## 改动清单

| 文件 | 修改 |
| --- | --- |
| `tools/experiments/run_source_isolated_batch.py` | 仅追加 `--final-width 1536 --final-height 768` |
| `tools/experiments/run_source_occlusion_pairs.py` | 同上 |
| `tools/experiments/render_source_ground_repairs.py` | 同上 |
| `src/densekp_retrain/data/boxes.py` | 共用原扰动算子；新增强制尺寸/策略接口、框类型检查、可序列化拒绝记录；具名保留历史像素门槛 |
| `tools/experiments/prepare_box_robustness.py` | 新增 `prepare_sized_derivatives` 实际消费者；旧 CLI 保持历史配方并显式检查 1536×768；紧框断言按类型生效 |
| `tests/test_high_res_boxes.py` | 命令构造、尺寸/边界/框类型/拒绝、真实 Dataset affine 和打包定向测试 |
| `tests/fixtures/historical_box_commands_v1.json` | 修改前保存的三个命令表达式 AST fixture |
| `tools/validate/validate_high_res_boxes_a1.py` | 最多 6×4 个框的 compact CPU 验收入口 |

Dataset、公共打包器、监督模块没有改动。新框元数据保存在与 NPZ 行序对应的 `index.json`；未扩展通用序列索引协议。

## 历史兼容证据

修改前保存独立源码与原 fixture 到 [before](../../outputs/acceptance/high_res_boxes_a1_20261008/before/)，原 9 项 CPU 测试通过。原 `tests/fixtures/source_packing_v1.json` 未重写，继续逐值验证历史框数组、dtype、shape、裁边及异常结果。

`change_box` 仍按如下顺序计算：float64 框 → 中心与原始宽高 → `center += [dx,dy] * max(size) * 1.2` → 尺度扰动 → 按原始边长单侧截断 → 裁到 1536×768。`trial_crop_from_box` 仍由调用方传 float32 框，执行中心计算和 `np.float32(maxside * 1.2 / 200)`；乘除和中间舍入顺序保持。`box_rejection` 的 `<2` 源栅格像素门槛具名为 `HISTORICAL_MIN_SIDE_PX`，等于 2 仍可接受，历史 reason 仍为 `invalid_box`。

三个历史命令仅追加两个最终尺寸选项。测试实际求值命令表达式，并在移除新增四个 token 后与修改前 AST 完全比较；采样 24、源宽 1536、full、人物、场景、布局、CUDA 字符串及所有其他参数表达式保持。**没有执行业务子进程**，命令里的 CUDA 不代表本次使用 GPU。

[historical_compatibility.json](../../outputs/acceptance/high_res_boxes_a1_20261008/historical_compatibility.json) 保存构造结果。三个活动生成入口及 `run_ds_source_resolution.py` 与 HEAD 文件逐字节一致；新默认仍为源/最终 5760×2880、compact，A/B/C 实验定义未改。

原 `prepare_box_robustness.py` CLI 仍服务于冻结 400 exposure 历史 trial，不接收新的高分辨率配方。对新数据应使用下列显式接口；错误把高分辨率数据送到旧 CLI 会拒绝尺寸不匹配，不能静默生成旧尺度裁边结果。历史 RNG、配额、扰动范围、fallback、阈值和已有结果未改写。

## 新接口与规则

```python
from densekp_retrain.data.boxes import change_box_for_image, crop_from_box

box = change_box_for_image(
    [3887, 995, 4527, 2127],
    image_size_wh=(5760, 2880),
    boundary_policy="erp_reject_seam",
    min_side_px=2,
    shift_reference_expansion=1.2,
    dx=0.03,
)
center, scale = crop_from_box(
    box,
    image_size_wh=(5760, 2880),
    boundary_policy="erp_reject_seam",
    min_side_px=2,
    crop_policy="deployment",
)
```

`image_size_wh` 没有缺省值。`prepare_sized_derivatives` 从真实 PNG 读取尺寸，与 manifest 的 `final_erp_resolution`、valid/person mask 的实际 shape 核对。新路径不推断 1536×768，不对图像缩放补齐，不根据黑色 RGB 裁边。

新框为 float32、有序 xyxy、右下端 exclusive。检查 shape、有限值、正宽高、最小边长、裁边后的空框/小框。扰动参数必须有限，factor 和平移归一化倍率须正，edge 为 0–3 整数，fraction 在 [0,1]。非法几何使用 `BoxRejected` reason code。

| 边界策略 | 行为 |
| --- | --- |
| `raster_clip` | 普通矩形图像，四侧裁到 `[0,W]×[0,H]`，裁后再检查 |
| `erp_reject_seam` | 输入或扰动结果 x0<0 / x1>W 立即拒绝；纵向仍按矩形边界裁边 |

ERP 使用有序展开坐标；例如 `[5750,y0,5770,y1]` 在 W=5760 时拒绝。反向端点也拒绝，不能将两端分别 mod、交换或求 min/max 作为修复。未实现周期裁剪。即使框本身不跨缝，若扩张后的实际 square crop 跨缝，也拒绝 `erp_crop_seam_crossing`；目标 mask 同时触及左右两条接缝边时保守拒绝 `erp_target_support_at_both_seam_edges`。因此本版本可能拒绝可被未来周期实现正确消费的样本。

框的矩形限制与有效域支持分开：先核对矩形几何，再对 `person_mask & valid_mask` 检查目标支持，实际 affine 后以 nearest 插值计算完整 256-square 中目标像素支持。GT 域检查继续沿用渲染器的 floor/横向 modulo 规则；这一标签域检查不是对框端点执行 modulo。

### 框类型

| `bbox_kind` | 特有规则 |
| --- | --- |
| `visible_instance_tight` | 必须精确等于有效域内可见实例 mask 紧框 |
| `fixed_reference` | 固定参考框，可与可见紧框不同；须声明参考来源，不自动当 amodal GT |
| `detection_tracking` | 检测/跟踪框，可与可见紧框不同；来源和目标 ID 是输入声明，不声称完成身份关联 |
| `manual` | 人工框，须声明来源；本次仅定向 fixture 测试，没有消费真实人工标注 |
| `synthetic_perturbation` | 新衍生框的明确类型；保存父框类型、父框坐标、父 loader、参数和策略 |

所有类型均检查实际图像/mask/manifest 尺寸、有限正框、身份与标注行/person mask 声明相符、非空来源、有效目标 mask 非空以及框与目标支持相交；实际裁剪还检查监督点与目标像素门槛。不能靠换类型关闭检查。真实 compact 只消费 `visible_instance_tight` 父框；其他类型的规则通过小型 fixture 检查，尚未验证真实检测/跟踪或人工框数据。

### 扰动与扩张不是同一参数

- `dx/dy` 位移单位为 **父框最大边长（源栅格像素）× `shift_reference_expansion`**。历史系数具名为 1.2；新接口要求明确传入。它不是 final crop 的倍率。
- `factor` 只改变扰动框宽高。单侧 fraction 乘原始对应边长，保留原算子语义。
- `crop_policy="deployment"`：最终正方形边长按框最大边 ×1.0；`"historical_trial"`：×1.2。两种策略独立记录于 index，并核对实际 Dataset affine，没有统一默认值或宣称两者等价。
- 本次还只读核对外部 `src/densekp_delivery/runners/pytorch_runner.py:95` 的框宽高 `/200`，未修改或运行外部代码。本报告中的 1.0 策略复现这里的扩张定义，不宣称整个外部推理链已验收。
- 新验收显式配置：最小边 2 **源 ERP 像素**，至少 10 个监督点，完整 256-square 内至少 16 个有效可见目标像素。这不是不同分辨率下相同角度门槛，也不是已确定的训练增强范围。

`prepare_sized_derivatives(parents, variants, output, *, min_side_px, min_points, min_target_crop_pixels, shift_reference_expansion, boundary_policy)` 是可复用实际消费者。父记录要求 image、manifest、valid/person mask、mask/person ID、camera、bbox_kind/source、given_bbox_xyxy、loader_input/row；variant 要求 name、parameters、crop_policy。输出目录必须不存在。输出 `derivatives.npz`、再次经现有 packer 打包的 `packed.npz`、`index.json`、逐框 geometry、拒绝计数；实际回读 packed Dataset。

新路径不把失败候选静默换成父框。`rejection_record` 将诊断中 NaN/Inf 转为 JSON null，保留 reason 与参数；strict JSON 可保存。旧结果不修写。

## CPU 定向测试

修改前运行 9 项；实现后第一轮 18 项通过；收紧 edge 整数检查并补拒绝计数后，组合 18 项通过。最终复核修正跨缝检查中多余的 side float32 转换后，新接口模块 9 项再次通过。共四次定向测试进程、54 次 test case 执行；没有全库 discover 或参数更新测试。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/densekp-a1-mpl \
python -m unittest tests.test_box_robustness \
  tests.test_source_packing_contract tests.test_high_res_boxes
```

[tests_final.log](../../outputs/acceptance/high_res_boxes_a1_20261008/tests_final.log) 记录 18/18 通过。覆盖历史三入口及新默认、原 fixture 数值、右/下高坐标、非法尺寸/NaN/Inf/反向/空框、2 px 边界、普通矩形裁边、输入/扰动/扩张 crop 跨缝、mask 与 manifest 尺寸不一致、各框类型的支持/来源/身份、JSON 拒绝记录与计数、两种扩张的实际 affine、黑色 RGB 不影响 mask 支持、全部遮挡 GT 仍监督，以及 NPZ 打包和 Dataset 回读。

[tests_boundary_final.log](../../outputs/acceptance/high_res_boxes_a1_20261008/tests_boundary_final.log) 记录最终边界补测 9/9 通过。另以已保存 24 行做只读复核，当前接口产生的 center/scale 与实际验收 NPZ 逐值相同，未增加新候选；`final_policy_check.json` 同时记录 `[30,0,50,80]` 在 80×100 栅格上因 Dataset 实际 side=80.0000011921 而跨缝拒绝的舍入边界用例。

本次专属文件的 `git diff --check` 通过。全工作区检查会报告原 `docs/TODO.md` 的既有尾空格；按文件边界保留，没有顺手修正。

## 真实 compact 消费者验收

运行入口（重新执行时必须使用新的不存在目录，且需重新授权框数量预算）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/densekp-a1-mpl \
python tools/validate/validate_high_res_boxes_a1.py \
  --output outputs/acceptance/high_res_boxes_a1_20261008/<new_run>
```

本次仅执行一次主验收，产物：[compact_run_1/acceptance.json](../../outputs/acceptance/high_res_boxes_a1_20261008/compact_run_1/acceptance.json)。父框为 cam2 的 person 1/2/3/4 与 cam4 的 person 1/2；均来自原 diagnostic_loader 的有效行。比如 cam2/person1 原框 `[3887,995,4527,2127]` 两轴均已超过旧上限，实际保留正确高坐标。

| 每个父框的候选 | 参数 | 裁剪策略 |
| --- | --- | --- |
| center_x | dx=0.03，按原 maxside×1.2 归一化 | deployment 1.0 |
| center_y | dy=-0.02，同上 | historical_trial 1.2 |
| scale | factor=0.9 | deployment 1.0 |
| single_edge | 右侧 edge=2，fraction=0.15 | historical_trial 1.2 |

这些只用于接口验证，没有测量 YOLO 误差或确定训练范围。共 24 候选、24 接受、0 拒绝；没有挑模型结果、增加候选、fallback 或二次生成。fixture 拒绝测试独立于这个真实数据预算。

- 24 行先实际 Dataset 检查，再打包后回读，最后独立核对图像与点；共 72 次真实数据 Dataset item 读取，只有 24 个新框。
- 每个图像裁剪逐像素等于原 PNG 使用返回 affine 的 OpenCV warp；全部 138 点逐值等于同一矩阵的齐次变换，保留 float32 中间赋值。
- 总监督点 2443，其中遮挡但参与监督 1420。仍为 `in_view & inside_full_crop`，visible 不作为可靠 GT 的排除条件。
- 最大逆 affine 误差 `0.000069428701 px`，低于本次源像素容差 `max(0.0002,5760*float32_eps)=0.000686645508 px`。
- 独立解析式也核对了实际 affine 与保存的 center/scale，以及两种 crop_policy。
- 输入树中 source、临时 source、EXR、鱼眼图均不存在；只依赖长期 PNG、mask、manifest、NPZ。原 compact 文件的大小/mtime_ns 清单前后完全一致。该证据未使用哈希，不将它声称为文件内容哈希验证。
- 新 NPZ/index/geometry/24 张 crop PNG 全在独立 outputs；没有覆盖旧文件或写入 datasets。父来源记录保存在本次 `config.json`；NPZ 索引旁表保留对应关系。

## 拒绝与限制

新路径按原因计数，见 `consumer/rejections.json`。本次真实样本没有拒绝；定向测试确实触发并保存了非法扰动和 `erp_seam_crossing` 各一次，NaN 父框也能够保存 null 诊断。其他拒绝代码包括非有限/形状错误、反向/空框、过小边、尺寸不一致、身份/来源缺失、紧框不一致、无目标支持、ERP 扩张 crop 跨缝和不足监督/目标像素支持。

未实现或未覆盖：周期 roll/unwrap 裁剪、跨缝可用率、真实检测误差/身份关联、动态序列、多场景泛化、完全没有可见目标的框、真实人工框、正式 train/val 划分及部署模型效果。固定参考/检测框仍需可靠标签与显式目标 mask 才能进入此合成监督消费者。缺文件、损坏 archive、缺必需字段或 Dataset 契约异常会终止任务并保留现场，不伪装为成功拒绝样本。

本实现记录 bbox 类型和策略不代表批准接口草案全部字段，也不改变训练配方。后续规模和增强范围、其他来源框、ERP 周期支持均需另批评审。

## 计算范围与资源

| 项目 | 实测 |
| --- | --- |
| 主验收墙钟，不含 Python 导入 | 28.0652 s |
| 主验收 CPU 时间，不含导入 | 28.0558 s |
| 主进程总墙钟（`/usr/bin/time -v`） | 32.04 s |
| 主进程 user / system | 28.53 / 2.79 s |
| 主进程峰值 RSS | 724020 KiB，约 707.05 MiB |
| 组合 18 项测试进程总墙钟 / 峰值 RSS | 3.06 s / 564744 KiB |
| 最终边界 9 项补测进程总墙钟 / 峰值 RSS | 2.98 s / 563944 KiB |
| 最终 18 项测试计时 | 0.123 s，不含导入 |
| 主验收目录最终文件字节数 | 4,444,133 B，约 4.24 MiB |
| CPU 限制 | OpenCV=1，OMP=1，OpenBLAS=1 |
| 渲染/模型/推理/optimizer update | 均为 0 |

峰值 RSS 由 Linux `getrusage` 与 `/usr/bin/time` 一致记录；最终文件字节数不是临时磁盘峰值，未连续采集磁盘峰值。资源文件见 `consumer_resources.txt`、`test_resources.txt`、`test_boundary_resources.txt`。其余计算为有界文件阅读、命令构造、原 fixture 测试和清单统计；无视频解码、GPU 任务或全库测试。

本轮停止于 A1 评审，未启动下一阶段动作接入、渲染、批量生成或训练。
