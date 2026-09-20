# Headless 渲染环境审计

审计日期：2026-09-05。所有测试均未安装软件、未下载资产，并显式移除 `DISPLAY`。

## 系统与 GPU

| 项目 | 实测结果 |
|---|---|
| Linux | Ubuntu 22.04.5 LTS，kernel 6.8.0-136-generic，x86_64 |
| GPU | 2 × NVIDIA GeForce RTX 3090，单卡 24 GiB |
| NVIDIA driver | 555.52.04 |
| driver CUDA capability | 12.5；本机 PATH 中没有 `nvcc`，不影响 EGL 或 Blender OptiX 渲染 |
| DISPLAY | 未设置；`XDG_SESSION_TYPE=tty` |
| X11 工具 | 存在 `Xvfb`、`glxinfo`，但 smoke test 未使用 X/Xvfb |
| Vulkan | 有 loader 与 Mesa ICD libraries；没有 `vulkaninfo`，本轮未验证 Vulkan device |
| EGL | `libEGL.so`、NVIDIA 与 Mesa EGL libraries 均存在 |
| OpenGL | `libOpenGL.so`、NVIDIA/Mesa GLX libraries 均存在 |
| OSMesa | 系统和 `/usr/local/lib` 均存在 `libOSMesa.so.8`；未执行 OSMesa smoke test |

受限沙箱不暴露 `/dev/nvidia*`，其中 `nvidia-smi` 和 EGL 会失败或段错误；宿主权限下 GPU 正常。后续自动任务必须显式申请 GPU device passthrough，不能把沙箱失败误诊为驱动损坏。

## 软件

| 组件 | 状态 |
|---|---|
| Blender | 已安装便携版 5.1.1：`/mnt/data1/khb/app/blender-5.1.1-linux-x64/blender`，未加入 PATH |
| pyrender | 0.1.45，位于既有只读 CameraHMR 环境 |
| trimesh | 4.4.7 |
| PyOpenGL | 3.1.0 |
| moderngl | 5.12.0 |
| smplx | 0.1.28 |
| torch | 2.1.1 |
| numpy | 1.26.4 |

本轮没有修改 `/home/yy/conda_envs/camerahmr`。正式 renderer 应建立独立环境，不应长期借用训练/推理环境。

## Blender 无界面实测

命令入口为 `tools/blender_headless_smoke_test.py`。

- `BLENDER_HEADLESS_CPU = PASS`：`blender -b`、无 `DISPLAY`、Eevee、256×256、单帧成功，约 18.0 秒。此数字包含进程启动与首次引擎初始化，不能当作稳定逐帧吞吐。
- `BLENDER_HEADLESS_GPU = PASS`：Cycles、OptiX、16 samples、256×256、单帧成功，渲染调用约 2.38 秒；Blender 枚举到两张 RTX 3090 的 CUDA 与 OptiX device。
- 输出：`outputs/smoke/headless/blender_rgb.png`。当前文件是最后一次 Cycles 测试结果。

结论：当前 Linux 服务器可以在完全没有 `DISPLAY`、没有 Xvfb 的条件下直接执行 Blender Cycles GPU render。Eevee 也通过，但后续复杂场景、驱动升级和长序列仍需稳定性测试。

Blender 启动开销显著。批量生成应一次加载模型/材质/场景后循环多帧，或维护常驻 worker；禁止每帧启动一个 Blender 进程。

## pyrender EGL 实测

命令入口为 `tools/pyrender_headless_smoke_test.py`，设置 `PYOPENGL_PLATFORM=egl` 后再导入 OpenGL/pyrender。

结果：`PYRENDER_EGL = PASS`。

```text
DISPLAY                  = null
GL_VENDOR                = NVIDIA Corporation
GL_RENDERER              = NVIDIA GeForce RTX 3090/PCIe/SSE2
GL_VERSION               = 4.1.0 NVIDIA 555.52.04
resolution               = 256 x 256
frames                   = 10
EGL renderer startup     = 0.243 s
mean render              = 0.00219 s/frame
min/max render           = 0.00112 / 0.01047 s
process max RSS          = 226.7 MiB
```

这是一个低面数球体、单灯光的小测试，只能说明数量级和 EGL 稳定性，不能外推到高分辨率多人物 PBR 场景。已验证输出：

- `pyrender_rgb.png`：RGB；
- `pyrender_depth.npy`：`float32` depth，全部 finite，有效深度范围约 2.252--2.827；
- `pyrender_mask.png`：`RenderFlags.SEG` 生成的二值 object mask。

官方 pyrender 文档也将 EGL 定义为无 display manager 的 GPU offscreen backend，OSMesa 为 CPU software backend。当前机器无需 OSMesa fallback。

## SMPL-X / DenseKP 实测

已有合法 SMPL-X v1.1 male/female/neutral 模型位于：

```text
/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools/smpl_models/smplx/
```

`tools/smplx_densekp_render_smoke_test.py` 只读加载 neutral 模型，zero pose / zero betas 得到 `(10475,3)` 顶点。同一个 `vertices` 数组同时：

1. 与 `model.faces` 组成 pyrender mesh；
2. 进入 `densekp_regressor_R.npy @ vertices`；
3. 使用与 pyrender 相同的透视内参和 OpenGL-to-image 轴约定投影。

结果：

```text
SMPLX_RENDER_SMOKE       = PASS
DENSEKP_OVERLAY_SMOKE    = PASS
vertices                 = (10475, 3), float32
densekp_2d               = (138, 2), float32, all finite
u range                  = [127.28, 385.02]
v range                  = [123.81, 393.93]
```

输出为 `smplx_rgb.png`、`smplx_densekp_overlay.png`、`smplx_vertices.npy`、`smplx_densekp_2d.npy` 与 `smplx_depth.npy`。视觉检查确认 138 点贴合人体表面。该测试证明“单一几何真值源”在当前服务器成立，但尚未验证 motion、纹理和 visibility 分类。
