# BEDLAM Tiny 帧映射审计

## 结论

选定 scene 的 GT 已按 archive 内部结构解压。24 张 tiny 图片全部能够按真实文件名映射到 camera CSV 的同名行，并按照 BEDLAM 官方 `df_full_body.py` 状态机映射到 person、motion 文件和 SMPL-X 参数索引。

```ini
TINY_IMAGES = 24
TINY_PERSON_SAMPLES = 186
TINY_SEQUENCES = 20
REQUIRED_CAMERA_FILES = 20
REQUIRED_MOTION_FILE_COUNT = 157

CAMERA_MAPPING_VERIFIED = YES
FRAME_TO_MOTION_MAPPING_VERIFIED = YES

NEED_SMPLX_MOTION_GT = YES
NEED_SMPLX_BODY_MODEL = NO
```

`NEED_SMPLX_BODY_MODEL=NO` 表示本机只读旧项目目录中已有 male、female、neutral 三种可加载 SMPL-X 文件，并不授权复制或修改它们。neutral 模型兼容性仍有一项需在 motion GT 到位后复核，见本文末尾。

## 已生成文件

- `outputs/audits/bedlam_tiny_mapping/frame_manifest.csv`：186 行 person sample；
- `outputs/audits/bedlam_tiny_mapping/required_motion_files.txt`：157 条去重后的精确相对路径，每行只有一个路径；
- `outputs/audits/bedlam_tiny_mapping/required_motion_indices.txt`：逐 motion 文件列出实际需要的 `smplx_param_ind`；
- `tools/build_bedlam_tiny_frame_manifest.py`：可复现本次映射的脚本。

## Tiny image naming

真实 archive member 形式为：

```text
20221024_3-10_100_batch01handhair_static_highSchoolGym/
png/seq_xxxxxx/seq_xxxxxx_ffff.png
```

官方 `get_params()` 使用 `sorted(glob(...))` 后的 `enumerate` 索引作为 `img_ind`。本 scene 每个 sequence 的文件名从 `0000` 连续递增，camera CSV 的 `name` 也逐行完全对应，因此：

```text
image_name = seq_xxxxxx_ffff.png
img_ind = int(ffff)
camera_row = img_ind
```

24 张的完整 archive member 和 frame ID 已记录在 `docs/BEDLAM_TINY_DATA_AUDIT.md`；相同信息也写入 manifest 的 `archive_member、seq_name、image_name、img_ind` 列。

## be_seq mapping

实际 `be_seq.csv`：

- columns：`Index, Type, Body, X, Y, Z, Yaw, Pitch, Roll, Comment`；
- data rows：771；
- sequence 示例：`sequence_name=seq_000000;frames=118;camera_hfov=52.0;...`；
- Body 示例：`rp_henry_posed_001_1084`；
- start-frame 示例：`start_frame=1;texture_body=...`；
- 对应 body transform 示例：`X=-124.264522...`、`Y=171.731711...`、`Z=0`、`Yaw=257.521868...`。

状态机严格按官方顺序遍历 `Comment`：

```text
sequence_name 行：
  seq_name = comment.split(';')[0].split('=')[-1]
  camera_csv = ground_truth/camera/<seq_name>_camera.csv

start_frame 行：
  start_frame = int(comment.split(';')[0].split('=')[-1])
  body = Body
  parts = body.split('_')
  person_id = '_'.join(parts[:-1])
  sequence_id = parts[-1]
  motion_relpath = <person_id>/<sequence_id>/motion_seq.npz
```

注意官方源码使用下划线连接 `person_id`，不是星号连接。

一张图会关联该 `sequence_name` 状态下的每个 Body 行，因此 manifest 是 person-level，共 186 行，而不是 24 行 image-only 表。

## 帧索引

官方对 6 fps 和 30 fps 都使用：

```text
smplx_param_ind = img_ind + start_frame
cam_ind = img_ind
```

`--fps 6` 仅跳过 `img_ind % 5 != 0` 的图片，不会除以 5 或重新编号。本轮保留所有 30 fps tiny 图片：24 张中 18 张不是 5 的倍数。

实测范围：

- tiny 最大 `img_ind`：133；
- `start_frame`：1～112；
- 所需 `smplx_param_ind`：1～215；
- 186/186 行满足参数索引公式；
- 186/186 行满足 `camera_row == img_ind`。

## Camera mapping

实际 camera columns：

```text
name, x, y, z, yaw, pitch, roll,
focal_length, sensor_width, sensor_height, hfov
```

20 个所需 camera CSV 的 row count：

```text
seq_000000=118  seq_000001=137  seq_000002=123  seq_000003=117
seq_000004=140  seq_000005=137  seq_000006=139  seq_000007=153
seq_000008=125  seq_000009=153  seq_000010=136  seq_000011=164
seq_000012=131  seq_000013=124  seq_000014=121  seq_000015=137
seq_000016=122  seq_000017=117  seq_000018=117  seq_000019=134
```

每张 tiny 图片均满足 `img_ind < camera row count`，且 `camera_rows[img_ind]['name'] == image_name`。不修改单位的整体范围：

| 字段 | min | max |
|---|---:|---:|
| `focal_length` | 36.905 | 36.905 |
| `x` | -955.192993 | 983.550659 |
| `y` | -904.501953 | 968.723877 |
| `z` | 165.173645 | 174.273743 |
| `pitch` | -9.877667 | 1.538141 |
| `yaw` | -168.736069 | 161.770905 |
| `roll` | -2.915687 | 2.938174 |

## Required motion files

精确的 157 条相对路径保存在 `outputs/audits/bedlam_tiny_mapping/required_motion_files.txt`。该文件可直接相对于官方默认目录解析：

```text
bedlam_data/smplx_gt/neutral_ground_truth_motioninfo/
<person_id>/<sequence_id>/motion_seq.npz
```

每个文件所需的去重参数索引保存在 `required_motion_indices.txt`，没有按 6 fps 改写索引。

## 官方 download_utils.sh 审计

本轮只读取官方脚本，没有执行：

1. 从 SMPL-X 下载 `models_smplx_v1_1.zip`，解压到 `bedlam_data/body_models/smplx`；
2. 从 BEDLAM 下载单个 `neutral_ground_truth.zip`，解压到 `bedlam_data/smplx_gt`；
3. 两个下载站分别要求注册，并由脚本交互读取各自 username/password；
4. motion GT 在脚本中是一个整体 ZIP；
5. 脚本不支持按 person 或 sequence 选择性下载，且没有对 ZIP 内单个成员做远端 range extraction；
6. 因而该脚本无法只获取 `required_motion_files.txt` 中的 157 个文件；若服务器没有额外的单文件接口，最小仍是完整 `neutral_ground_truth.zip`。

本轮没有运行脚本，也没有请求、读取、打印或保存任何密码。

## SMPL-X body model audit

在只读路径中找到：

```text
/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools/smpl_models/smplx/
SMPLX_MALE.npz
SMPLX_FEMALE.npz
SMPLX_NEUTRAL.npz
```

三者的 `v_template` 均为 `[10475,3]`、faces 均为 `[20908,3]`，且都可通过当前 `smplx` 包实例化并输出 `[1,10475,3]` vertices 和 `[1,127,3]` joints。没有复制或修改旧路径。

官方 `df_full_body.py` 的配置为：

| gender | ext | num_betas | flat_hand_mean | use_pca |
|---|---|---:|---|---|
| male | `npz` | 10 | `True` | `False` |
| female | `npz` | 10 | `True` | `False` |
| neutral | `npz` | 11 | `True` | `False` |

现有 male/female 实际加载 10 个 shape directions，符合官方配置。现有 neutral 虽能合法实例化，但当前 `smplx` 版本在请求 11 时只暴露 10 个 shape directions，并给出 warning。motion GT 尚未存在，无法检查其 `betas` 实际长度；若所需 motion 中存在 neutral 且 `betas` 为 11 维，必须先解决该兼容性问题。当前不另行下载模型。

## 尚缺资源

- `neutral_ground_truth_motioninfo` 中 manifest 指定的 157 个 `motion_seq.npz`；
- motion 文件到位后的 `poses/trans/betas/gender/gender_sub` schema、长度和参数索引上界验证；
- neutral motion 与现有 neutral body model 的 beta 维度兼容性确认。

本轮没有运行 DenseKP generator、训练或下载其他数据。
