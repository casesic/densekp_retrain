# 俯视场景编辑器 A3

当前推荐入口已迁至 [human_data_generator](../../../human_data_generator/README.md)。以下命令与输出路径是旧仓库历史入口，旧服务不会自动加载新项目；请勿为迁移而停止现有进程。

从仓库根目录启动：

```bash
python tools/scene_editor/server.py --port 8765
```

浏览器打开 `http://127.0.0.1:8765`。远程开发时在 VS Code 的“端口”面板转发 8765，再打开转发地址；服务只监听本机 `127.0.0.1`。

1. 选城市或河岸示例。新场景中每个人先有起点、终点两个控制点，可再插入转弯点；已保存配方保留原有全部点。绿色是已核查的道路候选矩形，灰色是未知区域。地图坐标为世界 XY 米，向上是 +Y。
2. 点人物卡，选中并拖动轨迹点；右侧可按编号选择任一点并输入精确 X/Y 米制坐标。先点击“添加控制点”，再点击屏幕上可见的曲线附近（20 像素内）即可插入；点在已有控制点上也会插入邻近的新点。删除使用右侧按钮。密集路线默认只显示部分控制点，当前选中点始终显示。切换机位模式后，点中双机组可拖动；点击地图其他位置会放置所选双机组。右侧可填写 yaw、X、Y。滚轮缩放，平移模式拖地图。
   已生成的视频默认收起；点击底部“查看视频”在当前页面打开，不改变地图高度。生成完成时会自动打开新视频。
3. 拖时间轴选择 0–30 帧（10 Hz），右侧显示所选人物当帧位置与路线行进方向。先点“检查当前时刻”或“全段 CPU 检查”：单帧会提示附近的突然转向，整段会阻止 0.1 秒内超过 60° 的朝向跳变。保存修订会新建 `<场景编号>_r<修订>.json`，已有版本不会覆盖。
4. 选一组 rig 点“预览当前时刻”。只渲染该帧该组 cam2/cam4；预览不会纳入正式打包。完全相同的配方、帧和 rig 会直接复用已完成预览。不同预览在 3 分钟内共用常驻 Blender 场景，换场景或人物资产时会重新装配。通过全段预检后点“生成整段”，同一冻结配方重新渲染 31 帧。改动后旧预览会标“已过期”。
5. 整段完成并打包成功后，在页面下方“已生成整段”选择任务和机位组，直接播放或下载 MP4。两组视频分别位于 `outputs/experiments/scene_editor_a3_20261013/episodes/<task_id>/run/rigs/<rig_id>/pack/dual_roi.mp4`；刷新页面仍可从已完成任务列表打开。任务仅完成渲染、打包失败时不会显示为可播放视频。

命令行重放：

```bash
python tools/scene_editor/cli.py check --recipe outputs/experiments/scene_editor_a3_20261013/recipes/city_example_r3.json --output /tmp/a3-check
python tools/scene_editor/cli.py run --recipe outputs/experiments/scene_editor_a3_20261013/recipes/city_example_r3.json --output outputs/experiments/scene_editor_a3_20261013/cli_replay --gpu-uuid GPU-UUID --budget-seconds 2400
```

指定帧预览在 `run` 后增加 `--frame 15 --rig-id rig_back --budget-seconds 300`；运行前应先用 `nvidia-smi` 核对空闲 GPU。CLI 的 `--output` 必须是新的目录。

当前已制作资产的编辑器适配器限定为已注册的诊断来源。城市默认两人，可再加入第三个已注册身份，其初始路线同样只有两个端点；河岸只有一个已注册身份。两场景都可复制注册 rig 至最多三组，但本轮实际生产只验收城市两组。修改 seed 会保留当前冻结人物，不会抽出新身份。地图未完成建筑、护栏等障碍测绘，灰色不能视为通行。详细验收与限制见 `docs/tools/SCENE_EDITOR_A3_20261013.md`。

MAMMA 512 正常生成、只读导出与 A1 mask 读取入口见 [A3 主流程](../../docs/data/MAMMA_GENERATION_REFACTOR_A3_20261022.md)。已有编辑器服务需用户稍后重启，才会加载本批 Python 后端改动。
