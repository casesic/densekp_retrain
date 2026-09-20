# FISHEYE CAMERA P0 — Omni-Radtan Validation Report (Phase 17)

Status: **PASS**（冻结 baseline，commit `1ad4259 synthetic-render-baseline-before-cycling`）

本报告冻结 Phase 17 的 Omni-Radtan fisheye camera P0 验证结果。所有数值由
`tools/fisheye/run_fisheye_camera_p0.py` 生成，机器可读结果见
`outputs/fisheye_camera_p0/regression_report.json` 与 `outputs/fisheye_camera_p0/report.json`。

## 1. Camera Parameter

独立于 Blender 的纯 NumPy 实现：`tools/fisheye/omni_radtan.py`。

Camera model: **Unified Camera Model (UCM) + radial distortion (Omni-Radtan)**

| 参数 | 值 |
|---|---|
| resolution | 3000 x 3000 |
| cx, cy | 1500.0, 1500.0 |
| xi | 3.485643 |
| f | 4495.392 |
| k1 | 7.048e-1 |
| k2 | 7.548 |

投影公式（入射角 theta，方位角 phi）：

```
m   = sin(theta) / (cos(theta) + xi)
r   = f * (m + k1*m^3 + k2*m^5)
u,v = cx + r*cos(phi), cy + r*sin(phi)
```

模型有效域：`theta <= arccos(-1/xi)`（xi > 1 时为 FOV 上限），monotone branch 由
二分法（64 次迭代）数值反演。

## 2. Coordinate Convention

- 相机系：CV 约定，**+X 右，+Y 下，+Z 光轴向前**。
- 位姿：`camera_to_world`（4x4），世界点先平移到相机系再归一化为单位射线。
- 像素坐标：top-left origin。
- 世界系：与 Phase 16 temporal pipeline 一致的 Y-up 世界系，DenseKP 3D 点经
  `(x, -z, y + lift)` 提升到地面。
- fisheye 图像不使用 Blender 相机渲染：Blender 仅以 equirectangular
  (2048x1024) 渲染 per-camera 全景源（`tools/fisheye/render_equirectangular_source.py`），
  Omni-Radtan 成像由 `cv2.remap` 按 unproject 得到的射线表重采样完成。
  相机外参在 `camera_config.json` 中定义，Blender 与 NumPy 共用同一份。

## 3. Center Case (`outputs/fisheye_camera_p0/center/`)

单相机 `fisheye_center`，位于光轴正对人体（相机在人体前方 3.2 m，
光轴穿过人体包围盒中心）。

- 人体成像位于图像中心区域（finite 像素范围约 [1272, 1757]^2）。
- DenseKP：138/138 finite，138/138 in-frame，86 visible（深度可见性阈值 4 cm）。
- 结果：**PASS**

## 4. Seam Case (`outputs/fisheye_camera_p0/seam/`)

两个共心（同位置）fisheye camera 绕世界竖直轴旋转：

- `cam_left`：+75 degree
- `cam_right`：-75 degree

人体位于两幅图像边缘 overlap 区域（cam_left 中人体位于左边缘
x∈[100, 454]，cam_right 中位于右边缘 x∈[2556, 2904]，两机朝向相差
150 degree，overlap 关系已人工检查确认）。

- cam_left：138/138 finite，138/138 in-frame，85 visible
- cam_right：138/138 finite，138/138 in-frame，87 visible
- 结果：**PASS**

## 5. Projection Validation

| 项目 | 结果 |
|---|---|
| optical axis projection | pixel = (1500, 1500)，误差 < 1e-12 |
| rotational symmetry | 4 轴对称点，radius 一致（1e-10） |
| pixel -> ray -> pixel roundtrip（1000 random rays） | max angular error < 3e-8 rad |
| DenseKP reprojection vs scalar reference | max 7.2e-13 px（center 2.3e-13, left 7.2e-13, right 4.5e-13） |
| warp 后 roundtrip angular error | max 2.6e-8 rad |

单元测试：`tools/fisheye/test_omni_radtan.py`（3 tests, OK），
运行方式 `PYTHONPATH=. python tools/fisheye/test_omni_radtan.py`。

## 6. DenseKP Validation

Annotation 保存于各 case 目录，格式：

| 文件 | shape | 说明 |
|---|---|---|
| `densekp_fisheye_2d.npy` | [T=1, 138, 2] float32 | fisheye 像素坐标（top-left origin） |
| `densekp_valid.npy` | [T=1, 138] bool | 投影落在图像内 |
| `densekp_visible.npy` | [T=1, 138] bool | 深度可见（\|surface_depth − keypoint_range\| < 4 cm） |
| `densekp_depth_error.npy` | [T=1, 138] float32 | 深度误差（m） |

3D source 一致性证明（`densekp_3d_fingerprint`）：

- 源：`geometry/densekp_3d.npy` frame 0（138 点），blake2b-128 指纹
  `f4cdc111e679b88d7779b6c44db5a06f`。
- 指纹在保存 `densekp_3d_source.npy` 后、三次 case 投影后均逐字节复核，
  `consistent_across_cases = true`：即 fisheye projection 前后的 3D source
  完全一致，projection 为纯函数，不修改输入。

统计汇总：

| case | finite | valid (in-frame) | visible |
|---|---|---|---|
| center | 138/138 | 138/138 | 86 |
| seam/cam_left | 138/138 | 138/138 | 85 |
| seam/cam_right | 138/138 | 138/138 | 87 |

## 7. Limitations

1. **单帧验证**：当前仅验证 geometry frame 0（T=1），未做多帧 temporal fisheye 序列。
2. **成像方式**：Omni-Radtan 图像由 equirectangular 源重采样得到，非 Blender 原生
   fisheye 光线追踪；中间存在一次 2048x1024 等距柱状重采样，深度掩码用
   INTER_NEAREST，遮挡边缘有约一个全景像素的量化误差（深度误差 p95 ≈ 0.25 m
   主要来自 5x5 邻域取 min 的表面采样，非模型误差）。
3. **深度可见性阈值**为 4 cm 的固定绝对阈值，未按距离自适应。
4. **valid ≠ visible**：`valid` 只表示投影在图像内（138/138），`visible` 才是
   经过自遮挡/衣物遮挡判定的可见点数（85–87）。
5. **域限制**：xi > 1 的 UCM 存在 FOV 上限 `arccos(-1/xi)`，超域射线返回 NaN，
   训练侧需依赖 `valid` mask 过滤。
6. 未引入 Unreal / BEDLAM2 scene / 新 renderer，未修改 Blender 相机；本阶段不
   做 fisheye training 与骑行场景。

## 复现

```bash
# 全量（含 Blender equirectangular 源渲染）
BLENDER_BIN=/path/to/blender python tools/fisheye/run_fisheye_camera_p0.py
# 仅 warp + 注解 + 报告（跳过 Blender 渲染）
python tools/fisheye/run_fisheye_camera_p0.py --skip-render
python -m unittest tools.fisheye.test_omni_radtan
```
