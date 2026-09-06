# Perspective P0 closed loop

## 结论

```ini
P0_PERSPECTIVE_CLOSED_LOOP = PASS
```

本轮没有下载 BEDLAM/AMASS motion、没有使用 Unreal、没有生成正式训练数据。所有输出仅用于 renderer/GT 管线验证。

## 已实现链路

```text
deterministic procedural SMPL-X parameters
  -> external Python smplx
  -> vertices.npy [16,10475,3]
       + faces.npy [20908,3]
       + densekp_3d.npy [16,138,3]
  -> one Blender 5.1.1 headless process
       -> 16 RGB PNG
       -> 16 multilayer EXR Depth/Mask
  -> external OpenEXR postprocess
       -> 16 float32 depth NPY
       -> 16 binary mask PNG
       -> densekp_2d/depth/visible NPY
       -> 16 overlay + contact sheet
  -> pyrender EGL cross-check at frame 0/8/15
```

## Toy motion

`tools/generate_p0_toy_motion.py` 使用 neutral SMPL-X v1.1、10 betas 全零、trans 全零，生成 16 帧平滑确定性动作：global yaw 从 -10° 过渡到 +10°，left shoulder 抬起至 35°，left elbow 弯曲至 -55°。参数保存在 `motion_params.npz`，fps 记录为 30。

首尾最大顶点变化为 0.322611 m；frame 1/8/15 相对 frame 0 的最大变化分别为 0.004658/0.217385/0.322611 m。RGB frame 0 与 15 不同，contact sheet 中动作连续。

## 输出目录

```text
outputs/p0_perspective/
├── geometry/
│   ├── vertices.npy
│   ├── faces.npy
│   ├── densekp_3d.npy
│   ├── motion_params.npz
│   └── geometry_manifest.json
├── rgb/                 # 000000.png ... 000015.png
├── depth/               # float32 NPY
├── depth_raw/           # Blender multilayer EXR
├── mask/                # binary PNG
├── annotations/
│   ├── densekp_2d.npy
│   ├── densekp_camera_xyz.npy
│   ├── densekp_depth.npy
│   ├── densekp_depth_error.npy
│   └── densekp_visible.npy
├── overlay/             # 16 frames + contact_sheet.png
├── crosscheck/          # pyrender frame 0/8/15 + report
├── manifest.jsonl       # 16 records
├── p0_render_report.json
├── p0_output_report.json
└── depth_semantics.json
```

总输出约 57 MiB，主要是未压缩的 32-bit multilayer EXR；这只是 16 帧验证产物。

## 验收表

| 条件 | 结果 | 证据 |
|---|---|---|
| 16 帧 SMPL-X vertices | PASS | `(16,10475,3)`，finite |
| 单进程 Blender batch | PASS | 一次 `blender -b` 连续写 16 帧 |
| RGB | PASS | 16 PNG，逐帧 motion 有变化 |
| depth | PASS | 16 float32 NPY；已知平面确认 camera-space Z |
| mask | PASS | 16 二值 PNG，与同次渲染 Alpha 同源 |
| DenseKP projection | PASS | `(16,138,2)`，全部 finite |
| Blender vs analytic | PASS | mean `1.09e-5 px`，max `4.09e-5 px` |
| overlay | PASS | 16 张及 contact sheet 视觉检查通过 |
| visibility | PASS | 1229 visible / 979 self-occluded，视觉基本合理 |
| manifest | PASS | 16 行，包含 camera/object matrix、intrinsics 与路径 |
| single geometry source | PASS | mesh 回读 diff=0，DenseKP 重算 diff=0 |
| pyrender cross-check | PASS | 三帧 silhouette mean IoU=0.999124 |

## 性能基线

| 阶段 | 实测 |
|---|---:|
| geometry generation | 0.332 s / 16 帧 |
| Blender script 内总耗时 | 3.450 s，含已知平面和 scene setup |
| Blender 16 帧 render 调用总计 | 2.275 s |
| Blender mean render | 142.21 ms/frame |
| Blender process peak RSS | 564.7 MiB |
| 外部 depth/visibility/overlay | 约 1.4 s |
| pyrender frame 0/8/15 | 通过；本轮不作为正式吞吐 benchmark |

这里的 `render` 时间含 compositor EXR 写出；RGB 的最终 `save` 日志也在同一调用内。外部命令启动 Blender 到 Python 脚本开始的耗时未被 Blender 内部计时器单独分离，因此不伪造一个精确 startup 数值。单进程策略已把启动成本摊到 16 帧。

宿主 `nvidia-smi` 在其他任务同时占用两张 GPU 时进行了观察，无法把显存增量可靠归因给本次 Eevee P0；因此只记录进程 CPU RSS，不报告不可信的 GPU memory delta。

## 独立环境

创建了 `/home/yy/conda_envs/densekp_render`。该 venv 通过 `--system-site-packages` 只读复用现有 CameraHMR 包，并只在自身安装了小型 `OpenEXR 3.4.15`；没有修改 CameraHMR 环境。正式长期使用前仍应生成独立依赖锁定文件，避免共享基础环境版本漂移。

## 下一步

用一段已授权、定义明确的真实 SMPL-X `motion_seq.npz` 替换 toy 参数，保持 geometry artifact、Blender reader、manifest、projection/depth/visibility tests 完全不变，渲染 8--32 个分散帧做 P0-real-motion 验证。通过后才进入 near-camera、edge 和 truncation。
