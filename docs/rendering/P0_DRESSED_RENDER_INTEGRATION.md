# P0 穿衣渲染接入

## 结果

Phase 7 已使用 BEDLAM 官方匹配组合 `rp_aaron_posed_009 / 1095` 生成 16 帧最小穿衣合成数据集：

```ini
P0_DRESSED_RENDER_INTEGRATION = PASS
OUTPUT_DIR = outputs/baselines/p0_dressed_dataset
```

衣物来自 `1095.npz` 的逐帧 `vertices_seq`，拓扑来自 `faces`；没有依赖 Alembic。衣物只作为渲染几何和遮挡物，DenseKP 仍严格为 `R @ smplx_body_vertices`。

## 帧映射

选定 clothing frame：

```text
0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 38
```

映射公式经 metadata 与数组长度验证：

```text
local_motion_index = 5 × clothing_frame_index
source_frame_30fps = 101 + local_motion_index
```

对应 local motion frame 为 `0, 10, 25, 35, 50, 60, 75, 85, 100, 110, 125, 135, 150, 160, 175, 190`。完整逐帧记录见 `geometry/dressed_frame_mapping.csv` 和 `manifest.jsonl`。

## 坐标与对齐

body forward 使用 motion 中的 male、10 betas、完整 165 维 poses、global orientation 和原始 `trans`。这是必要的：clothing vertices 已处于包含该平移的 BEDLAM 米制 Y-up 坐标中。body 与 clothing 共享同一 Blender world transform，再整体落地。

16 帧合并统计中，衣物顶点到最近 body vertex 的距离中位数为 0.0148 m，95 分位为 0.0622 m；逐帧统计稳定。人工检查 contact sheet 未发现整体漂移、尺度错误、明显帧偏移或严重穿模，故 `BODY_CLOTHING_ALIGNMENT = PASS`。

## 渲染 pass 与标注

每帧使用两个 pass：

1. 隐藏地面、透明世界，渲染 body+clothing，生成 person mask；
2. 开启地面和背景，渲染最终 RGB 与包含 body+clothing+scene 的 depth。

visibility 使用第二个 pass 的 full-scene depth。当前状态为 `VISIBLE / OCCLUDED / OUT_OF_FRAME / BEHIND_CAMERA`，不区分 body self-occlusion 和 clothing occlusion。`densekp_valid` 与 `densekp_visible` 分开保存，所有 138 点的 valid 均为 1，训练 loss 没有修改。

## 几何与投影回归

```ini
BODY_GEOMETRY_FINGERPRINT_BEFORE = 00dd073faa497d91ab55de498e6af6e7
BODY_GEOMETRY_FINGERPRINT_AFTER  = 00dd073faa497d91ab55de498e6af6e7
BODY_GEOMETRY_UNCHANGED = YES
DENSEKP_3D_RECOMPUTE_MAX_DIFF = 0
PROJECTION_MEAN_ERROR_PX = 0.000010634
PROJECTION_MAX_ERROR_PX = 0.000033464
PROJECTION_REGRESSION = PASS
```

16 帧平均双 pass 时间为 310.672 ms/frame。机器验收见 `outputs/baselines/p0_dressed_dataset/dressed_validation.json`，人工验收图见 `outputs/baselines/p0_dressed_dataset/contact_sheet.png`。

## 输出

- `rgb/`：最终穿衣 RGB；
- `depth/`：full-scene camera-space Z；
- `masks/person/`：body+clothing union；
- `annotations/`：DenseKP 2D、depth、valid、visible；
- `overlay/`：逐帧 DenseKP overlay；
- `manifest.jsonl`：配对、帧映射、相机、几何和路径 metadata；
- `contact_sheet.png`：16 帧人工验收图。

下一阶段应先做 synthetic source dataset 的人工确认，再设计独立 training NPZ adapter；不要把 adapter 耦合进 renderer。
