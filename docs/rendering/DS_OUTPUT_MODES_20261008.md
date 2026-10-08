# DS 输出模式与有界验收（2026-10-08）

## 结论

已实现 `compact|full` 两种输出模式。后续三个活动 DS 生成入口默认 compact，源 ERP 与最终
normal ERP 均为 5760×2880，DS 栅格仍为 2880×2880。公共后处理默认仍是 full，历史流程
显式固定 full；A/B/C 分辨率实验及“B 总体最好、C 并非全面改善”的结论没有改写。

C 组双相机真实后处理验收通过。两模式的 normal ERP、标签、mask、框、配对数组、共享世界点
和 Dataset loader 逐值一致。compact 清理临时源后仍可完成 Dataset 回读、索引打包和逐人
mask 框读取；不包含源 EXR/预览、鱼眼 RGB、reference lens mask、overlay 或 review 图。

## 入口与兼容

| 入口 | 模式默认 | 源 ERP | 最终 ERP | 说明 |
| --- | --- | ---: | ---: | --- |
| `imaging.ds.process_scene` | full | 调用者输入 | 未传时等于源尺寸 | 保持旧调用行为 |
| `render_cycling_ds.py` | compact | 5760×2880 | 5760×2880 | 后续普通双场景/骑行入口 |
| `render_cycling_characters_ds.py` | compact | 5760×2880 | 5760×2880 | 后续多角色骑行入口 |
| `render_adult_candidates_ds.py` | compact | 5760×2880 | 5760×2880 | 后续来源隔离布局入口 |
| 旧 scene-pair、occlusion control | 显式 full | 原 1536×768 | 原配方 | 历史证据不变 |
| 旧来源隔离 batch/control/repair | 显式 full | 原 1536×768 | 原配方 | 仍可读取 source 世界数组 |
| `run_ds_source_resolution.py` | 显式 full | A/B/C 各自尺寸 | 5760×2880 | A/B 没有改成 C |

`render_equirectangular_source.py` 的通用分辨率默认值未改变；新渲染会在源 manifest 记录 Blender
版本。Omni、透视和外部 pano-stitch 未修改。

## 文件与消费者依据

compact 长期保留：

- `cam2|cam4/normal_erp.png`：Dataset 图像；PNG 无损，色彩路径不变。
- `annotations.npz`：人物 ID、138 点 `xy`、`in_view/visible` 及现有域、相机和世界点字段。
- `valid_mask.png`、`person_<id>_mask.png`、`boxes.json`：有效域、框增强现有 mask 接口和框。
- `pair_annotations.npz`、`shared_world_densekp.npy`：双相机配对和必要小型几何。
- `people.json`、`rig.json`、`report.json`、`source_recipe.json`、`generation_status.json`：
  来源、相机、分辨率、samples/seed、代码/软件版本、执行及清理状态。
- 活动入口回读生成的 `diagnostic_loader.npz`、`validation.json`：打包和实际 Dataset 证据。

full 保留以上内容，另保留源 EXR/预览、源世界人体/衣物/groom 数组及元数据、每相机
`fisheye.png`、`reference_lens_mask.png`、`overlay.png`、`pair_review.jpg`、`crop_review.jpg`；
成人入口还保留既有 actual-input 调试制品。compact 在保存函数内直接跳过调试图，并非生成后删除。
鱼眼 RGB/实例仍在内存按同一路径计算；源 RGB、射线距离和实例 ID 仍用于相同的可见性计算。

## 临时目录生命周期

compact 默认使用 `<output>/_temporary_ds_source/source`，也可通过 `--temporary-root` 指定尚不
存在的独占目录。父目录 marker 绑定最终 output 绝对路径和随机 token。流程为：创建并登记 →
渲染/复制 → 后处理 → 源几何与最终文件回读 → Dataset/打包检查 → 写成功状态 → 校验 marker
后清理 → 单独记录 cleanup 状态。

目录碰撞直接拒绝；异常写 `generation_status.json` 并标记 `preserved_after_failure`，不在 finally
清理。缺 marker 或 owner/output/path 不匹配时拒绝删除。最终 `source_recipe.json` 用占位符替换
临时输出参数，只把 EXR 文件名记为 ephemeral，不声明已清理路径为输入。重新计算 DS 成像、
源栅格遮挡或可见性仍需按人物/场景/相机配方重新渲染。

若数据回读已经通过但清理失败，状态单独记录 `data_validation=passed`、`cleanup=failed`、检查结果、
清理异常和仍实际存在的临时路径；外层异常记录器识别该状态并保持原记录，不降级覆盖成普通生成失败。

## CPU 检查

`PYTHONPATH=src:. python -m unittest tests.test_imaging_pipeline tests.test_ds_scene_pair` 共 14 项通过。
新增覆盖非法模式、两模式相同数值产物、compact 不生成调试图、配对不依赖 overlay、成功清理、
失败保留及历史目录保护。三个活动入口的 `--help` 已实际读取；未运行全库 discover。

5760 宽回读还暴露了旧固定 `2e-4 px` 验证容差过小：float32 坐标实测最大舍入
`0.000244044 px`。验证门槛改为 `max(0.0002, width * float32_eps)`，本次为
`0.000686646 px`；这不改变投影、图像或标签值。

## 双相机验收结果

成功目录为 `outputs/acceptance/ds_output_modes_20261008/attempt_4/`。只读复用现有
`ds_source_resolution_20261008/C/source`，没有新渲染；两模式均重新执行真实 EXR 读取与 DS
后处理。源设置为 Cycles、samples=64、seed=0、animated_seed=false，cam2/cam4，源/最终
5760×2880。

| 项目 | full | compact |
| --- | ---: | ---: |
| 最终保留 | 949,396,400 B（905.42 MiB） | 27,049,906 B（25.80 MiB） |
| 源复制 | 0.371 s | 0.376 s |
| CPU 后处理 | 21.774 s | 12.891 s |
| 回读/消费者检查 | 4.322 s | 3.149 s |
| 清理 | 不适用 | 0.097 s |
| compact 临时 source 实测峰值 | 不适用 | 867,778,918 B（827.58 MiB） |

总墙钟 46.74 s。临时峰值是对 compact 独占目录的实际遍历；加上同时存在的最终 compact 文件，
任务峰值约 894.83 MB（十进制）。内存峰值未可靠采集，不作估计。

逐值比较覆盖 2 张 normal ERP、2 张 valid mask、8 张逐人 mask、2 份 annotations、pair arrays、
shared world points 和 diagnostic loader。每种模式实际回读/打包 6 行，并由 mask 重算核对 6 个框。
监督点 622，其中遮挡但参与监督 358；两模式一致。full 中源和调试文件均存在可读，compact
禁止项为空，最终 recipe 不含已清理临时路径。

`prepare_box_robustness.change_box` 明确保留历史 1536×768 裁边，本次没有冒充 5760 算法；
验收覆盖其真实前置接口（逐人/valid mask、boxes、Dataset、`trial_crop_from_box`）。未来若在 C
尺寸制作新框增强集，仍需先定义分辨率感知的裁边策略。

## 失败记录与边界

实际启动 4 次，前三次现场保留，第四次通过：

1. 首次把公共 geometry 错传为 `None`，在 full 后处理入口失败。
2. `attempt_2/full` 使用默认 Python，缺 OpenEXR；未安装依赖。
3. `attempt_3/full` 完成后处理，在旧高分辨率固定容差处失败。
4. `attempt_4` 使用已有 `densekp_render` 环境，full/compact 全部通过。

失败目录按授权未删除，合计占用较大。历史 C 组、TODO、其他输出和外部资产均未修改或删除。
复用的旧 C manifest 未记录 Blender 版本；新 renderer 已补记。因没有启动 Blender/GPU，本轮无需
查询 GPU；没有训练、推理、权重加载、下载、安装、提交或 SHA256。

compact 将本样例长期保留量降至 full 的约 2.85%，且不改变模型输入或监督数组，适合作为后续
新数据默认模式。full 仍用于渲染/遮挡复核和分辨率实验。本结论只验证保存策略，不重做分辨率
收益结论，也不授权批量生成或 DS 直接渲染。
