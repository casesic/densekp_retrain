# 坐标系文档（COORDINATE SYSTEMS）

> 所有公式均从现有代码原样提取，未做任何改写或"优化"。
> 与既有文档 `P0_COORDINATE_CONVENTION.md`、`P0_DEPTH_VISIBILITY.md` 交叉核对过；结论一致处注明出处。
> 无法从代码确认之处明确标注 **UNCERTAIN**，不做猜测。

## 0. 总览：坐标系链条

```text
SMPL-X local (米, Y-up, 右手)
   ↓ tools/render/render_p0_blender.py:133 / tools/fisheye/render_equirectangular_scene_source.py:60
Blender world (米, Z-up)
   ↓ tools/render/render_p0_blender.py:197-199 (perspective)
   ↓ tools/fisheye/run_fisheye_camera_p0.py:77 (fisheye，NumPy 等价变换)
camera coordinate（两种：Blender 相机局部 / CV 相机坐标）
   ↓ 投影
image coordinate（左上原点像素）
   ↓ bbox / center / scale → crop trans
densekp/core/datasets/utils.py :: gen_trans_from_patch_cv
bbox/crop coordinate（原始图像像素域内的仿射）
   ↓ xy/256-0.5
DenseKP network input coordinate（中心原点归一化）
```

---

## 1. SMPL-X local（几何制品坐标系）

| 项 | 值 | 依据 |
|---|---|---|
| 单位 | 米 | P0_COORDINATE_CONVENTION.md；smplx 模型输出 |
| 轴 | +X 右，+Y 上，+Z 前后 | 同上；代码中 `conv` 与 lift 用法一致 |
| 手性 | 右手 | conv 矩阵 det=+1，变换不改变手性 |
| shape | `vertices [T,10475,3] float32`，`densekp_3d [T,138,3] float32` | render_p0_blender.py:68 的 shape 断言 |
| 原点 | SMPL-X 模型定义（pelvis 附近） | 模型内部；代码未做平移（transl 通常为 0，`source_translation_used` 可选） |
| 生成 | tools/generate/generate_real_motion_geometry.py :: main（smplx 前向）；`densekp_3d = einsum('kv,tvc->tkc', R, V)` | — |

注意：`geometry/densekp_3d.npy` 与 `vertices.npy` **同一坐标系**（SMPL-X local），渲染阶段才做 world 变换。

## 2. SMPL-X local → Blender world

实现：`tools/render/render_p0_blender.py:133`

```python
conv = Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1)))
ob.matrix_world = Matrix.Translation((0,0,-float(vs[0,:,1].min()))) @ conv
```

即 `(x_b, y_b, z_b) = (x_s, -z_s, y_s)`，再加世界 Z 平移使 frame 0 最低点 `z_b=0`。
fisheye 路径的 NumPy 等价：`tools/fisheye/run_fisheye_camera_p0.py:77`

```python
lift = -float(vertices[:,1].min())
world = np.column_stack((x, -z, y + lift))
```

（Phase 18 再叠加场景 yaw 旋转与 body_location：tools/fisheye/run_fisheye_scene_p0.py:47-51。）

| 项 | 值 |
|---|---|
| Blender world | +X 右，+Y 后，+Z 上；米；右手（Z-up） |
| 逐帧矩阵 | `manifest.jsonl::object_matrix_world`（完整 4×4，可独立复算） |

## 3. Perspective camera 坐标系（两条并存的表示）

### 3.1 Blender 相机局部系（渲染器内部）

- 相机局部约定：观察方向 `-Z`，上 `+Y`，右 `+X`（Blender 官方约定）。
- `p_cam = cam.matrix_world.inverted() @ p_world`（render_p0_blender.py:197-198）。
- `annotations/densekp_camera_xyz.npy` 存的就是这个 `q`（Blender 相机局部系，**不是** CV 系）。

### 3.2 像素投影（左上原点）

实现：`tools/render/render_p0_blender.py:165,199`

```python
fx = fy = cd.lens / cd.sensor_width * w     # 50/36*512 = 711.111… px
cx = cy = w / 2                              # 256.0
z = -float(q.z)                              # Blender -Z 是前向 → 深度取负
an = [fx*q.x/z + cx, cy - fy*q.y/z]
```

- 图像原点：左上；u 向右，v 向下；无 0.5 px 偏移。
- 与 Blender API 交叉验证：`world_to_camera_view` 返回左下原点 NDC，比较式 `(ndc.x*w, (1-ndc.y)*h)`（render_p0_blender.py:199-200）。
- 实测最大差异 4.1e-5 px（P0_COORDINATE_CONVENTION.md §4）。
- manifest 记录 `projection='perspective_pinhole'` 与 `intrinsics{fx,fy,cx,cy}`。

**Depth 语义（实测，不得更改）**：Eevee `Depth` pass = camera-space Z（即 `-q.z`），不是欧氏距离。依据：`render_p0_blender.py::plane_test` 已知平面测试 + `finalize_p0_outputs.py` 自动判定 + `P0_DEPTH_VISIBILITY.md`。

## 4. ERP（equirectangular）源坐标

- 渲染：Blender 相机 `type='PANO'`，`panorama_type='EQUIRECTANGULAR'`，经度 [-π,π]，纬度 [-π/2,π/2]
  （tools/fisheye/render_equirectangular_source.py / render_equirectangular_scene_source.py:65-67）。
- 相机朝向转换（Blender 局部 → CV）：`render_equirectangular_scene_source.py:71-72`

```python
# Blender local axes: X right, Y up, -Z forward. CV: X right, Y down, Z forward.
matrix[:3,0] = r[:,0]; matrix[:3,1] = -r[:,1]; matrix[:3,2] = -r[:,2]
```

- `camera_to_world` 采用 **CV 约定**（+X right, +Y down, +Z forward），由
  `tools/fisheye/run_fisheye_camera_p0.py::pose(target, origin, yaw_deg)` 构造（列：right/down/forward）。
- ERP 像素映射（warp 采样）：`tools/fisheye/run_fisheye_camera_p0.py:47`

```python
lon = arctan2(rays[:,0], rays[:,2]); lat = arcsin(-rays[:,1])
mapx = (lon/(2π)+0.5)*w ; mapy = (0.5 - lat/π)*h
```

  即 x 向东经、y 从上（北极）向下，左上原点。**UNCERTAIN**：Blender EQUIRECTANGULAR 相机自身的经纬起点与该映射的严格一致性未从代码单独验证，仅由 warp 后 mask/深度 gate 与 roundtrip 测试间接覆盖（fisheye_camera_p0 报告 pass）。

## 5. Omni-Radtan fisheye（src/densekp_retrain/camera/omni_radtan.py）

代码 docstring（omni_radtan.py:1-6）明确：**CV-style：+X right，+Y down，+Z optical forward；`camera_pose` 为 camera-to-world**。

前向（project_points_omni_radtan，omni_radtan.py:27-38）：

```text
q   = (p - pose[:3,3]) @ pose[:3,:3]        # world → camera（列向量右乘 Rᵀ）
ray = q / ||q||
rho = ||ray[:2]||；theta = atan2(rho, ray.z)；phi = atan2(ray.y, ray.x)
m   = sin(theta) / (cos(theta) + xi)
r   = f * (m + k1*m³ + k2*m⁵)
pixel = (cx + r·cos(phi), cy + r·sin(phi))
model_domain: theta ≤ arccos(-1/xi)（xi>1 时），m ≥ 0
```

默认内参（frozen）：`3000×3000, cx=cy=1500, xi=3.485643, f=4495.392, k1=0.7048, k2=7.548`。

反向（unproject_pixels_omni_radtan，omni_radtan.py:40-51）：对 `m` 二分求逆（r(m) 单调），再

```text
theta = atan(m) + asin( xi·m / sqrt(1+m²) )
ray = (sinθ·cosφ, sinθ·sinφ, cosθ)
```

roundtrip 单测：`tests/test_omni_radtan.py`（<3e-8 rad）。**不改投影定义。**

## 6. bbox / center / scale

两套生成端（数学一致，来源不同）：

1. **BEDLAM 官方**：`tools/annotation/generate_bedlam_densekp.py::official_bbox`
   - 前 22 个 joints2d → 图内过滤（0≤x≤W，0≤y≤H，且 <8 个有效则丢弃）
   - center = bbox 中心（原图像素）；`scale = max(w,h)/200 × 1.2`
2. **相机压力 solver**：`tools/render/solve_camera_stress_placement.py::bbox_metrics`（记录 full/in-frame bbox、截断比例等，用于 gate，不进训练 NPZ）。

消费端（训练）：`densekp/core/datasets/dataset_keypoints.py:59`

```python
bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()
```

即 `scale` 存的是 `bbox_max_edge/200`（含 1.2 扩张与否取决于生成端）；`BBOX_SHAPE` 未配置时（None）直接取 max。**UNCERTAIN**：官方历史 DenseKP NPZ 的 scale 是否已含 1.2 扩张无法从本仓库证明（ANNOTATION_AUDIT.md Unknowns）。

## 7. crop transform（bbox/crop → 256×256）

实现：`densekp/core/datasets/utils.py`

- 仿射矩阵：`gen_trans_from_patch_cv(c_x, c_y, src_w, src_h, dst_w, dst_h, scale, rot)`
  - 3 点对应（center/downdir/rightdir）→ `cv2.getAffineTransform`；
  - `rot` 为度，`rot_rad = π·rot/180`，作用于 downdir/rightdir（图像逆时针方向定义见 rotate_2d）。
- 裁剪：`generate_image_patch_cv2` → `cv2.warpAffine(..., (256,256))`；flip 时 `img[:, ::-1]` 且 `c_x = W - c_x - 1`。
- 坐标变换：`trans_points2d_parallel(keypoints, trans)`（3×3 齐次），然后（utils.py:620-621）

```python
proj_verts[:, :2] = trans_points2d_parallel(proj_verts[:, :2], trans)
proj_verts[:, :-1] = proj_verts[:, :-1] / patch_width - 0.5
```

注意第三处写法 `[:, :-1]` 作用在前两列（x,y），**第三通道不被归一化**，原样保留（对训练 GT 为常数 1.0）。
- 归一化后：中心原点，x 右、y 下，名义范围 [-0.5, 0.5]。

## 8. DenseKP network input coordinate

- 图像：`img_patch` [3,256,256]，ImageNet mean/std（constants.py:23-24），BGR→RGB（utils.py:605）。
- backbone 输入：`x[:,:,:,32:-32]` → [B,3,256,192]（densekp_trainer.py:79）。**预测坐标仍定义在完整 256 宽的 crop 坐标系**（训练损失直接用 256 域归一化 GT；宽度裁剪只影响特征图）。
- 输出 `pred_keypoints [B,138,3]`：0:2 = 归一化 xy（中心原点）；2 = `with_var=True` 时 log σ（densekp_trainer.py:105），否则未监督。
- val 分支：`batch['proj_verts']`（原图像素）经 `batch['_trans']` 变换后 `/IMAGE_SIZE - 0.5`（densekp_trainer.py:160-161），与训练 crop 数学一致（val 无增强）。

## 9. fisheye 域的注记坐标（现状）

`warp_case` 输出（tools/fisheye/run_fisheye_camera_p0.py:63-64）：
`densekp_fisheye_2d [T,138,2] float32`：左上原点像素，投影自 `project_points_omni_radtan`；
`densekp_valid [T,138]`（在图像内）、`densekp_visible [T,138]`（深度差 <4cm）。
可见性深度用 `point_range = ||camera_points||`（欧氏距离）与 warped depth 比较 —— 注意这与 perspective 域 camera-space-Z 语义不同，是每个域单独验证的（P0_DEPTH_VISIBILITY.md §3 明确警告不得跨域沿用）。

## 10. 速查表

| 坐标系 | 原点 | 轴 | 单位 | 代码 |
|---|---|---|---|---|
| SMPL-X local | 模型 pelvis | X右 Y上 Z前 | 米 | generate_real_motion_geometry.py |
| Blender world | 原点+Z 抬升 | X右 Y后 Z上 | 米 | render_p0_blender.py:133 |
| Blender camera local | 相机 | -Z前 +Y上 +X右 | 米 | render_p0_blender.py:197 |
| Perspective image | 左上 | u右 v下 | px | render_p0_blender.py:199 |
| ERP source | 左上 | x=经度向右, y=北极向下 | px | run_fisheye_camera_p0.py:47 |
| Omni-Radtan camera | 相机 | CV: X右 Y下 Z前 | 米/无量纲 | omni_radtan.py:27 |
| Fisheye image | 左上 | u右 v下（omni_radtan 像素约定：cx 向右, cy 向下） | px | omni_radtan.py:33 |
| bbox/center/scale | 图像左上 | 像素 | px / 200px | generate_bedlam_densekp.py:46 |
| Crop 256 | crop 左上 → 归一化后 crop 中心 | x右 y下 | [-0.5,0.5] | utils.py:615-621 |
| Network pred | crop 中心 | x右 y下（第 3 通道 log σ） | [-0.5,0.5] | densekp_trainer.py:96-112 |
