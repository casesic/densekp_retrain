# 服装接入约束与方案

## 本轮结论

当前机器没有服装 Alembic 文件，也没有能够证明与现有 `smplx_animation.npz` 匹配的服装 metadata。因此禁止将静态 garment 或任意缓存套到当前人体上。

现有示例 motion 的可验证信息为：

- 126 帧，30 fps，male；
- 来源字段指向 KIT `314/bend_left01_stageii.npz`；
- 10 维 betas 已保存在 NPZ；
- 当前 P0 选取 source frame：`0, 8, 17, 25, 33, 42, 50, 58, 63, 75, 83, 92, 100, 108, 117, 125`。

这些字段没有 BEDLAM subject/body asset ID、garment ID 或 clothing cache path，仓库也没有对应关系表。因此：

```ini
MATCHING_CLOTHING_AVAILABLE = NO
BODY_CLOTHING_PAIR_VERIFIED = NO
FRAME_MAPPING_VERIFIED = NO
```

## 官方服装格式

`third_party/bedlam2_render/unreal/import/import_abc_clothing.py` 将 `.abc` 以 `AlembicImportType.GEOMETRY_CACHE`、`PER_FRAME` sampling 导入。源文件为 Blender 米制坐标，导入 Unreal 时才执行米到厘米以及轴向转换。目录示例为 `<subject>/<animation>/<animation>.abc`，说明依赖关系是：

```text
garment × body shape/subject × motion → per-frame simulated cache
```

官方 BEDLAM2 Starter Pack 含 51 个 body shape 与 150 个 matching motion/clothing 组合，但组件为 Unreal 5.3.2 `.uasset`，不能直接作为当前 Blender 源 Alembic 使用。源缓存在 BEDLAM2 受许可下载区的 `Downloads > Clothing > ABC format`，文件模式为 `b2_clothing_abc*.tar`。官方文档提示完整源缓存需约 3 TB。

## 所需用户操作

访问官方 [BEDLAM 下载与注册页面](https://bedlam.is.tue.mpg.de/index.html)，登录并接受数据许可；随后在 BEDLAM2 下载区查看：

1. `Clothing > ABC format` 下全部 `b2_clothing_abc*.tar` 的准确文件名与大小；
2. 每个分片对应的 subject/body 与 animation 清单；
3. 同一 pair 的 motion NPZ，通常来自 `b2_motions_npz_training.tar`；
4. 若选择 Starter Pack，确认是否同时提供原始 `.abc` 和 motion `.npz`，而不仅是 Unreal `.uasset`。

公开页面没有暴露分片索引和大小，因此当前不能准确指定一个低于 2 GB 的安全下载。不要下载完整约 3 TB 服装集。建议用户把下载区的文件列表及大小提供给本项目，再只选 `whitelist_animations_starterpack.json` 中一个 body/animation pair 对应的最小 ABC 分片与 motion。

建议放置位置：

```text
assets/bedlam2_clothing/<subject>/<animation>/<animation>.abc
assets/bedlam2_motion/<subject>/<animation>.npz
assets/bedlam2_clothing/<subject>/<animation>/pair_metadata.json
```

`pair_metadata.json` 至少记录 subject、body shape、gender、animation、garment、fps、frame range、单位和坐标约定。

## 获得资产后的接入门槛

在渲染前先只读审计 Alembic object、frame range、fps、拓扑、transform 与单位，并建立显式表：

```text
render_frame ↔ source_motion_frame ↔ clothing_frame
```

只有 body identity/betas、motion、服装缓存、fps、offset、坐标和缩放全部有官方 metadata 支撑，才设置 `BODY_CLOTHING_PAIR_VERIFIED = YES`。衣物只作为渲染 geometry/occluder；DenseKP 始终为：

```python
densekp_3d = R @ smplx_body_vertices
```

最终 scene depth 应包含 body、clothing 和 scene；`densekp_valid` 保持 1，`densekp_visible` 独立反映遮挡，不修改训练 loss。
