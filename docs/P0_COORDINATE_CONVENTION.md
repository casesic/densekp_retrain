# P0 坐标与透视投影约定

## 1. 单一几何源

唯一几何制品为 `outputs/p0_perspective/geometry/vertices.npy`，shape 为 `[16,10475,3]`、dtype 为 `float32`。Blender 每帧用 `foreach_set("co", vertices[t])` 原样更新 mesh；DenseKP 使用同一数组计算：

```python
densekp_3d[t] = densekp_regressor_R @ vertices[t]
```

自动回读 Blender mesh 顶点的 `max_abs_diff` 为 0；重新计算 DenseKP 3D 相对落盘数组的 `max_abs_diff` 也为 0。P0 mesh 没有 Armature、Subdivision、shape key、cloth、modifier 或 parent deformation。

## 2. 坐标系

SMPL-X 模型输出单位为米，使用右手坐标系：

- `+X`：人体右/图像横向；
- `+Y`：上；
- `+Z`：前后深度轴。

Blender 使用米制 scene unit 的右手坐标系：

- `+X`：右；
- `+Y`：后；
- `+Z`：上。

P0 的 SMPL-X local 到 Blender object/world 旋转为：

```text
                  [ 1  0  0  0 ]
T_smpl_to_blender=[ 0  0 -1  0 ]
                  [ 0  1  0  0 ]
                  [ 0  0  0  1 ]
```

即 `(x_b, y_b, z_b) = (x_s, -z_s, y_s)`，determinant 为 `+1`，未改变手性。随后仅增加世界 `Z` 平移，使 frame 0 最低点落在 `Z=0`；完整矩阵逐帧写入 `manifest.jsonl` 的 `object_matrix_world`。

## 3. Blender camera

相机固定为：

```text
location_world = [0, -3.2, 1.0]
rotation_euler = [pi/2, 0, 0]
```

Blender camera local 坐标仍遵循官方约定：观察方向是 local `-Z`，图像上方向是 local `+Y`，local `+X` 向右。世界点到相机点：

```text
p_world = object_matrix_world @ p_smpl
p_camera_blender = inverse(camera_matrix_world) @ p_world
z_forward = -p_camera_blender.z
```

物体在镜头前要求 `z_forward > 0`。

## 4. 相机内参与像素坐标

P0 显式配置：

```text
resolution             = 512 x 512
resolution_percentage  = 100
pixel_aspect            = 1
lens                    = 50 mm
sensor_width            = 36 mm
sensor_height           = 36 mm
sensor_fit              = HORIZONTAL
shift_x / shift_y       = 0 / 0
clip_start / clip_end   = 0.05 / 100 m
```

因正方形分辨率、正方形 sensor 和 pixel aspect 1：

```text
fx = fy = lens_mm / sensor_width_mm * resolution_width
   = 711.1111111111 px
cx = cy = 256.0 px
```

解析投影采用图像左上角为原点、`u` 向右、`v` 向下：

```text
u = fx * X_camera / (-Z_camera) + cx
v = cy - fy * Y_camera / (-Z_camera)
```

Blender `world_to_camera_view` 返回左下原点的归一化坐标，因此比较式为：

```text
u_blender = ndc.x * width
v_blender = (1 - ndc.y) * height
```

没有人为加入或减去 0.5 px。16×138 个点的解析投影与 Blender API 对比：

```text
mean   = 0.000010915 px
median = 0.000010426 px
p95    = 0.000021209 px
max    = 0.000040889 px
```

误差远低于 0.1/0.5 px 验收线，属于 float32 输入和 Blender mathutils 浮点计算差异，不存在系统性半像素偏移。

## 5. manifest 复现要求

`manifest.jsonl` 每帧记录 geometry frame、avatar/motion ID、分辨率、lens、sensor、fit、pixel aspect、shift、near/far、camera/object pose 与 4×4 matrix、`fx/fy/cx/cy` 以及 RGB/depth/mask/annotation 路径。只依赖该记录和 geometry artifact，可以复算世界点、相机点和 DenseKP 像素坐标。
