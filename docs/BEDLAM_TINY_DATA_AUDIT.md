# BEDLAM v1 Tiny 数据审计

## Scene 与下载范围

本轮只处理：

```text
20221024_3-10_100_batch01handhair_static_highSchoolGym
```

下载根目录：

```text
/home/yy/datasets/BEDLAM/bedlam_data/images/
```

实际下载文件：

| 相对路径 | 字节数 |
|---|---:|
| `scene/ground_truth/scene_gt.tar.gz` | 90,170 |
| `scene/ground_truth/scene_gt.tar.gz.xxh128` | 99 |
| `scene/png/scene_png.0.tar` | 3,642,112,000 |
| `scene/png/scene_png.0.tar.xxh128` | 99 |

表中的 `scene` 代表本文开头的完整 scene 名称。未下载 `png.1.tar`～`png.4.tar`、masks、mp4 或 dataset snapshot。

## Ground-truth archive

只使用 `tar -tzf` 列出内容，没有解压或修改 archive。成员包括：

- scene 根目录下的 `be_seq.csv`；
- `ground_truth/camera/seq_000000_camera.csv` 到 `seq_000099_camera.csv`，共 100 个 camera CSV。

因此 sequence mapping 和按 sequence 命名的 camera ground truth 文件可用。由于本轮按要求只查看成员名，没有读取 CSV 内容；相机 CSV 是否逐字段包含 intrinsics、extrinsics、translation 及其具体 convention，仍需下一阶段解析后确认。

该 archive 的成员列表中没有 SMPL-X motion/body 参数，也没有 SMPL-X body model。

## PNG shard 结构

`png.0.tar` 可以完整遍历，包含 2,645 张 PNG、20 个 sequence：`seq_000000`～`seq_000019`。成员结构为：

```text
scene/png/seq_xxxxxx/seq_xxxxxx_ffff.png
```

各 sequence 的图片数和帧范围：

| sequence | 数量 | frame 范围 |
|---|---:|---|
| `seq_000000` | 118 | `0000–0117` |
| `seq_000001` | 137 | `0000–0136` |
| `seq_000002` | 123 | `0000–0122` |
| `seq_000003` | 117 | `0000–0116` |
| `seq_000004` | 140 | `0000–0139` |
| `seq_000005` | 137 | `0000–0136` |
| `seq_000006` | 139 | `0000–0138` |
| `seq_000007` | 153 | `0000–0152` |
| `seq_000008` | 125 | `0000–0124` |
| `seq_000009` | 153 | `0000–0152` |
| `seq_000010` | 136 | `0000–0135` |
| `seq_000011` | 164 | `0000–0163` |
| `seq_000012` | 131 | `0000–0130` |
| `seq_000013` | 124 | `0000–0123` |
| `seq_000014` | 121 | `0000–0120` |
| `seq_000015` | 137 | `0000–0136` |
| `seq_000016` | 122 | `0000–0121` |
| `seq_000017` | 117 | `0000–0116` |
| `seq_000018` | 117 | `0000–0116` |
| `seq_000019` | 134 | `0000–0133` |

## Tiny 候选

从完整 PNG 成员序列中均匀选择 24 个位置，覆盖全部 20 个 sequence，避免只取连续相邻帧。只将下列成员解压到：

```text
/home/yy/datasets/BEDLAM/bedlam_data/tiny/
20221024_3-10_100_batch01handhair_static_highSchoolGym/images/
```

选中成员：

```text
scene/png/seq_000000/seq_000000_0000.png
scene/png/seq_000000/seq_000000_0115.png
scene/png/seq_000001/seq_000001_0112.png
scene/png/seq_000002/seq_000002_0090.png
scene/png/seq_000003/seq_000003_0082.png
scene/png/seq_000004/seq_000004_0080.png
scene/png/seq_000005/seq_000005_0055.png
scene/png/seq_000006/seq_000006_0033.png
scene/png/seq_000007/seq_000007_0009.png
scene/png/seq_000007/seq_000007_0124.png
scene/png/seq_000008/seq_000008_0086.png
scene/png/seq_000009/seq_000009_0076.png
scene/png/seq_000010/seq_000010_0037.png
scene/png/seq_000011/seq_000011_0016.png
scene/png/seq_000011/seq_000011_0131.png
scene/png/seq_000012/seq_000012_0082.png
scene/png/seq_000013/seq_000013_0066.png
scene/png/seq_000014/seq_000014_0057.png
scene/png/seq_000015/seq_000015_0051.png
scene/png/seq_000016/seq_000016_0029.png
scene/png/seq_000017/seq_000017_0022.png
scene/png/seq_000018/seq_000018_0020.png
scene/png/seq_000019/seq_000019_0018.png
scene/png/seq_000019/seq_000019_0133.png
```

解压后实际得到 24 张图片；24 张均可解码，shape 均为 `[720,1280,4]`。第四通道为 alpha，后续生成流程需要明确以 RGB/BGR 三通道方式读取，不能误把 alpha 当成监督或可见性。

## 运行 generator 前仍缺少的输入

- 与上述 sequence/frame 对齐的 SMPL-X motion/body 参数，或直接提供完整 `[10475,3]` 顶点和至少前 22 个三维 joints；
- SMPL-X body model。有限搜索未找到 `SMPLX_NEUTRAL/MALE/FEMALE` 的 NPZ 或 PKL；
- 对 `be_seq.csv` 和 camera CSV 的实际字段解析，以及每张 tiny PNG 到相机记录的精确映射；
- 将这些数据整理为 `tools/generate_bedlam_densekp.py` 所需的小型几何 NPZ。

本轮未运行 DenseKP generator、DatasetKeypoints 验证或训练。

```ini
GT_ARCHIVE_OK = YES
PNG_ARCHIVE_OK = YES
PNG_COUNT_IN_SHARD0 = 2645
TINY_PNG_EXTRACTED = 24
CAMERA_GT_AVAILABLE = YES
NEED_SMPLX_GT = YES
```
