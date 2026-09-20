# 外观资产审计

## 结论

本机没有可直接接入当前 Blender 流程的 HDRI、皮肤纹理、头发、鞋或动作兼容服装缓存。本轮没有下载资产。现有资源足以用程序化材质、世界背景、地面和灯光建立最小外观基线，但还不足以生成穿衣人物。

| 层级 | 内容 | 本机状态 | 说明 |
|---|---|---|---|
| Level 0 | 人体材质/肤色 | 可用 | Principled BSDF 程序化生成 |
| Level 1 | 灯光、背景、简单场景 | 可用 | 程序化生成；没有合法 HDRI |
| Level 2 | 皮肤纹理 | 不可用 | 相关 `.uasset` 是 Unreal 资源，不是 Blender 原始纹理 |
| Level 3 | 头发、鞋 | 不可用 | 没有 groom/cache 或鞋资产；`shoeTexture.blend` 只是制作模板 |
| Level 4 | 服装纹理 | 不可用 | 没有可直接使用的原始贴图 |
| Level 5 | 动画服装几何 | 不可用 | 没有与当前人体和动作匹配的 Alembic 缓存 |

```ini
HDRI_AVAILABLE = NO
SKIN_TEXTURE_AVAILABLE = NO
HAIR_ASSET_AVAILABLE = NO
SHOE_ASSET_AVAILABLE = NO
```

## BEDLAM 服装类型与依赖

BEDLAM/BEDLAM2 的 `unreal/import/import_abc_clothing.py` 会递归查找 `.abc`，按 Geometry Cache 导入并逐帧采样。代码中的目录把缓存放在具体 subject 与 animation 下。这是按动作模拟的 Alembic/Geometry Cache，不是可随意换动作的通用贴身网格。

```text
garment × body shape/subject × motion → cloth simulation/cache
```

因此不能假定“同一 garment × 任意 avatar × 任意 motion”自由组合。

```ini
CLOTHING_ASSET_AVAILABLE = NO
CLOTHING_TYPE = CODE_ONLY；预期为 per-motion simulated Alembic Geometry Cache
CLOTHING_MOTION_COMPATIBLE = NO
```

推荐路线是选项 C：后续只获取一组与目标人体和动作明确匹配的 BEDLAM/BEDLAM2 服装缓存。若只有静态 garment mesh，则必须另做 skinning 或 cloth simulation。

## DenseKP 定义不随衣物改变

```python
densekp_3d = R @ smplx_body_vertices
```

禁止对 clothing vertices 使用 `R`。当前 138 点全部参与 loss，且没有 visibility weighting；本阶段不改变 valid policy 或 loss。未来可额外保存 body visibility 与 clothing occlusion 供分析。

## 后续场景接入

优先支持 Blender native scene、HDRI，以及 glTF/FBX/OBJ。若 SimWorld 能导出这些标准 3D 资产，就作为独立场景层导入 Blender，并保持人体顶点、相机和投影分支不变；若只能导出 Unreal level，届时再评估 Unreal，不猜测其内部实现。
