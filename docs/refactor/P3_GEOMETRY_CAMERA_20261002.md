# P3.1/P3.2：人体、衣物与坐标相机计算

完成日期：2026-10-02。P2 已通过用户整体评审；本批 P3.1/P3.2 实现与本地验收完成，待评审。
P3.3 成像后处理、P3.4 Blender 边界及 P3.5 场景配方仍未实施。

## 1. 独立基线与执行范围

分支 `master`，HEAD `a3a5e7bb641bf1dee774d015127fee6b39c28bda`。
`starting_state.json` 和迁移前 Python 源码快照位于
[refactor_geometry_camera_20261002](../../outputs/acceptance/refactor_geometry_camera_20261002/)。
本批开始后保留 P1/P2 实现、用户修改及所有历史输出；只向这个新验收目录写几何数组。
没有训练、下载、渲染、新增角色/姿态资产、修改外部 pano-stitch、覆盖原几何或提交 commit。

先执行活动旧实现得到 `before/data/`，再迁函数。选择四组已有资产：

| 人物 | 模型 | 两个独立骑行姿态 |
| --- | --- | --- |
| cycling_F01_v1 | 本地 SMPLX_FEMALE.npz，10 betas | coco_292226_person_1330527、coco_306800_person_225160 |
| cycling_M01_v1 | 本地 SMPLX_MALE.npz，10 betas | 同上 |

参数/历史几何来自 `outputs/experiments/cycling_four_characters_20260929/derivatives/`，
模型路径取已有 character manifest，读取两个明确的本地模型文件，没有递归加载资产。
CPU 重算身体、关节、138 点、rest pose、两种衣物外壳、绑定、关节旋转、世界坐标及保存恢复。
没有拟合新 betas，没有选择或生成新姿态，也没有重新解决衣物质量。

主 before 在旧脚本函数尚未替换时捕获。额外的原始 EXR 可见性核查，使用迁移前快照中独立编译的旧函数，
在迁移后执行，保存到 `before/source_visibility/`；不调用新包装生成旧期望值。
正常模型基线没有依靠兼容导出对照来证明等价。

## 2. 迁移结果与保留原位的代码

| 公共位置 | 迁入内容 |
| --- | --- |
| [body/parameters.py](../../src/densekp_retrain/body/parameters.py) | `PARAMETERS`、`evaluate`、`zero_pose`、`rest_body`，明确 CPU、axis-angle 弧度和米 |
| [body/measurements.py](../../src/densekp_retrain/body/measurements.py) | `measurement_rows`、`measure`，保留固定解剖带规则 |
| [body/transforms.py](../../src/densekp_retrain/body/transforms.py) | `world_points`，人体/衣物/语义点共用身体最低点与变换顺序 |
| [clothing/shells.py](../../src/densekp_retrain/clothing/shells.py) | `vertex_normals`、`outfit_regions`、`shell`、`clipped_shell`；两种历史方案不合并 |
| [clothing/binding.py](../../src/densekp_retrain/clothing/binding.py) | `clip_surface`、`closest_surface`、`bind_new_points`、`pose_rotations`、`CONV`、`frame`、`triangle_frame` |
| [io/geometry.py](../../src/densekp_retrain/io/geometry.py) | `save_geometry` 保存参数/几何与回读门槛；JSON 使用既有公共 writer |
| [camera/erp.py](../../src/densekp_retrain/camera/erp.py) | `project_erp`、历史 `erp_points`，各自表达式顺序保留 |
| [camera/rig.py](../../src/densekp_retrain/camera/rig.py) | `rig_poses`，区分 normal ERP 基底与物理 DS 光轴 |
| [camera/domain.py](../../src/densekp_retrain/camera/domain.py) | `point_in_view`，DS 域、图内范围及显式 ERP mask |
| [camera/visibility.py](../../src/densekp_retrain/camera/visibility.py) | ERP 深度/实例可见性与旧 DS 实例邻域可见性，保持不同算法 |

共迁移 25 个定义（23 个函数、2 个常量）。38 个入口文件改变导入或移除原定义，
剩余计算 AST 一致；迁移计算 AST 忽略文档字符串后逐项一致。
证据分别为 `entry_structure.json`、`moved_computations_final.json`。
活动调用方直接导入公共模块，包括旧 DS 遮挡入口通过另一脚本间接获得的 world_points。
原脚本保留导入名兼容；公共模块不反向导入 tools。

具体 `POSES`、角色/材质/颜色、shape fit 目标、衣物款式 profile、场景 SCENES 保留实验层。
`rest_garment`、领口造型及修补/碰撞配方也保留各实验，公共化仅提取实际复用的计算。
旧 cycling 的 `save_body_and_clothing` 仍组合其衣物与 manifest；没有与 commuter 的 `save_geometry`
强行统一。例如前者保存 int32 faces，后者保留 model.faces 原 dtype，两个序列化契约继续独立。
`body/geometry.py` 的模型构造与 regressor、DS/Omni 的投影模型和标定、EXR 读取实现没有改算法。
`process_scene`、`warp_case`、Blender renderer 仍是后续任务；已有入口只更新调用和格式。

## 3. 坐标与算法约定

- 人体与衣物输入为 SMPL-X 模型坐标，+Y 向上，单位米。`world_points` 先做 `(x,-z,y)`，
  使用 **身体 vertices 的最低 Y** 做地面抬升，再绕世界 +Z 旋转 yaw，最后加 location。
  衣物不能用自身最低点计算 lift，138 点也不能单独抬升。
- `camera[4,4]` 是 CV camera-to-world，旋转矩阵列为右/下/前轴。
  行向量变换为 `q=(world-t)@R`；`distance=norm(q)` 是光心到点的射线距离，**不是 optical-Z 深度**。
- ERP 横轴为经度，`+/-pi` 可分别落在 x=width 和 x=0。投影函数不统一做取模，
  每个消费方保留自己的周期/边界策略；零长度射线仍产生原 NaN，不新增修复或偏移。
- normal ERP 两相机基底相同、光心不同；物理 DS cam4 在投影轴中翻转，两个物理前向轴相反。
  rig 中 0.18 m 等参数是合成配置，不宣称真实设备基线。保持 20260320 标定，不混入 031/0311。
- DS 真实鱼眼为 2880×2880；本批已有场景的 ERP 为 1536×768。
  真实 ERP 5760×2880 的单位不能直接用于这个合成栅格。
- `point_in_view` 保留 floor、横向 modulo、纵向 clip，再与 DS 域/鱼眼边界/显式 mask 相交。
  不从 RGB 黑色推断有效区，也不改成 `points_in_mask` 的 rint 规则。
- `erp_visibility` 在原始源 ERP 的 5×5 邻域内比较正射线距离和 instance ID，横向周期采样，严格 `<0.04 m`。
  `visible_at_instance_points` 是旧 DS 的四舍五入中心、7×7 非周期实例邻域，无距离比较。
  Omni `warp_case` 的鱼眼 5×5 深度判断仍在原后处理函数，留 P3.3 拆解；没有统一这三种规则。
- `shell` 整面选择与 `clipped_shell` 三角形裁剪保持独立；clipped shell 的 9 位去重与
  `clip_surface` 的 8 位去重也不合并。24 个最近面中心的绑定搜索是近似方法，不是穷尽碰撞证明。

## 4. 已执行验收

| 检查 | 实测结果 |
| --- | --- |
| 四组 CPU SMPL-X/衣物/相机前后回放 | `after_comparison.json`：409 数组、12 JSON 精确一致，含 dtype、shape、字段/参数顺序、返回类型、异常及 CPU RNG；仅输出根路径允许变化 |
| 与原几何/衣物资产关系 | 四组身体及 138 点最大差 0；面索引一致；六种 clothing_v2 数组最大差均 0，含 barycentric IDs/weights、reference/rest、faces 和衣物顶点；详情在各组合 records.json |
| 写出及重载 | 旧、新 save_geometry 在各自验收目录写 vertices/faces/138 点/motion_params/manifest；全部逐项一致；原 2e-5 m export、1e-6 m operator、2e-6 m pelvis gate 保持 |
| 代表合成数组 | seam 两侧、极点、零/非有限射线、负/NaN/零 baseline、全空有效 mask、错 instance、区域过小、非法 shell offset、退化法线、衣物 profile 等独立旧结果保持 |
| 既有 normal-ERP buffer | `saved_scene/` 保存 val_000 两相机的投影/有效域及对现有 buffer 的旧、新函数结果；没有用它冒充原始 source-ERP 可见性 |
| 原始 EXR 可见性 | `source_visibility_comparison.json`：20 数组、1 JSON 精确；读取 val_000 两个原始 source EXR，无 warp/render。xy、in_view、surface_visible、visible、occluded_in_view 均与原 annotations 精确一致 |
| CPU 回归 | `cpu_tests_final.json`：92 项通过，0 失败/跳过。含原混合监督、Dataset、几何、衣物测试及新 5 项便携测试 |
| 兼容与结构 | 旧名 identity 检查独立于 before 验收；25 定义、38 入口的计算 AST 一致；公共库无 tools 依赖 |
| 命令入口 | `cli_checks.json`：15 个入口从 `/tmp`、无 PYTHONPATH 运行 --help；包括渲染命令的参数解析，但没有启动渲染 |

原始 EXR 的每相机 414 点分组：cam2 为 in_view 388、visible 193、遮挡 195；
cam4 为 in_view 147、visible 57、遮挡 90。surface_visible 未与 in_view 相交时分别是 214、190，
不能把它直接当作监督集合。分组与旧实现、已保存标注均一致。

纯数组前后比较没有浮点容差；历史 writer 的 gate 保留用于其本来的恢复检查。
本批只实测上述两种性别、两种姿态、10-betas 模型及一个已有双相机场景，不宣称全部角色/资产组合均重算。
没有实际重训、真实神经网络预测或 Blender 验收，不能以这些结果宣布 P3.3/P3.4 完成。

最初两项 CPU mock 测试仅替换了旧 pilot 的 DS 标定，迁移后 domain 模块仍用实际 2880 标定，
导致小尺寸 mock 的 roundtrip 集合为空。已同步 mock 两处读标定的位置，预期值、生产规则没有变；
`cpu_tests.json`/log 的失败记录保留，最终结果另存。原始 EXR 首次读取因缺少搜索路径未找到 OpenEXR，
使用已有 `assets_external/python_deps` 后通过；未安装新包，初次日志保留。

便携独立基线在 [geometry_camera_v1.json](../../tests/fixtures/geometry_camera_v1.json)，
[测试](../../tests/test_geometry_camera_contract.py) 不依赖本地 SMPL-X 资产或 outputs；
它依赖现有 NumPy/Torch/SMPL-X/SciPy/trimesh 环境。

## 5. 复核、兼容与后续

主复核从仓库运行，输出目录必须是新目录；捕获模式拒绝已迁移的 evaluate：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/densekp-refactor-mpl PYTHONPATH=src:. /home/yy/conda_envs/camerahmr/bin/python -B tools/validate/verify_geometry_camera.py --before outputs/acceptance/refactor_geometry_camera_20261002/before/data --output /tmp/densekp-p3-geometry-review-new
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. /home/yy/conda_envs/camerahmr/bin/python -B -m unittest tests.test_geometry_camera_contract -v
```

原始 EXR 检查脚本及精确路径保存在证据目录 `source_visibility_check.py`；需既有本地
`assets_external/python_deps` 提供 OpenEXR。运行环境、源码及对应记录均保存，不复制模型或纹理资产。

本次另核查到：既有 `.gitignore` 的 `data/` 规则会忽略 **P2 的 `src/densekp_retrain/data/`**。
这些代码实际存在、已执行，但不能假定普通 git add 会收录；此前“docs/测试不被忽略”的检查不涵盖该包。
本批遵守不改 `.gitignore`、不提交的边界，新 P3 模块和本文/便携 fixture 均不被忽略。
P4/最终入库需显式核对这个既有忽略问题，避免只收录调用方而漏掉 P2 实现；本轮没有改路径或暂存文件。

无阻碍 P3.1/P3.2 本地验收的运行问题。后续按 P3.3→P3.4 推进：先独立冻结 process_scene/warp_case
已有输出、区分原始深度与重映射 buffer，再迁图像后处理；Blender 的对象/材质/linked/native 边界须单独实测。
GPU 计算仍须先检查设备和进程，不假定 GPU 0 恢复。
旧函数导出与目录暂保留；回退以本批快照为界，不能覆盖已评审 P1/P2 或删除历史输出。
