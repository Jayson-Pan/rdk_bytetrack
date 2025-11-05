## 参数调节指南（靠近/远离趋势识别）

本指南说明如何调节 `python/ultralytics_YOLO_ByteTrack.py` 中的趋势识别参数，以更稳定地判断交通车辆相对骑手的“靠近（approaching）/远离（receding）”。当前实现基于：
- 背景特征点 + RANSAC 估计相机的全局仿射运动（相机运动补偿）
- 将上帧目标框投影到当前帧得到“预测框”，与“实际检测框”比较，形成尺度与底边的残差信号
- 对残差做 EMA 平滑，并使用滞回与连帧确认，仅在置信时打出 approaching/receding 标签（不显示 stable）

---

### 一、快速运行示例

- 摄像头：
```
python3 python/ultralytics_YOLO_ByteTrack.py --input /dev/video0
```

- 指定视频（默认会在 `source/video` 下搜索文件名）：
```
python3 python/ultralytics_YOLO_ByteTrack.py --input vehicle_track.mp4
```

- 更敏感（更容易触发靠近/远离）：
```
python3 python/ultralytics_YOLO_ByteTrack.py \
  --trend-scale-high 0.08 --trend-scale-low 0.04 \
  --trend-bottom-high 1.0 --trend-bottom-low 0.5 \
  --trend-consistent-k 1 --trend-ema-alpha 0.45
```

---

### 二、可调参数说明

- `--trend-scale-high`（默认 0.12）/ `--trend-scale-low`（默认 0.06）
  - 含义：相机运动补偿后，“实际框面积”相对“预测框面积”的相对残差阈值（近似比例）。
  - 用途：面积扩张（>0）代表靠近，收缩（<0）代表远离；采用高/低双阈值形成滞回。
  - 建议范围：0.06 ~ 0.18；越小越敏感，但可能更易误报。

- `--trend-bottom-high`（默认 1.5）/ `--trend-bottom-low`（默认 0.8）
  - 含义：相机运动补偿后，bbox 底边 y 的残差（px/帧）。底边 y 上升通常代表目标靠近。
  - 建议范围：0.5 ~ 3.0 px/帧；头盔摄像场景可从 1.0 ~ 1.5 起调。

- `--trend-consistent-k`（默认 2）
  - 含义：同一趋势（靠近/远离）需要连续满足的最少帧数。
  - 越小越敏感；1 表示只要当前帧满足即可出标签，但稳定性下降。

- `--trend-ema-alpha`（默认 0.4）
  - 含义：EMA 平滑系数，越大越“灵敏”，越小越“平滑”。
  - 建议范围：0.2 ~ 0.6。

---

### 三、推荐预设

- 均衡（默认）
  - `--trend-scale-high 0.12 --trend-scale-low 0.06`
  - `--trend-bottom-high 1.5 --trend-bottom-low 0.8`
  - `--trend-consistent-k 2 --trend-ema-alpha 0.4`

- 保守（更稳更少误报）
  - `--trend-scale-high 0.18 --trend-scale-low 0.09`
  - `--trend-bottom-high 2.0 --trend-bottom-low 1.0`
  - `--trend-consistent-k 3 --trend-ema-alpha 0.3`

- 敏感（更早提示）
  - `--trend-scale-high 0.08 --trend-scale-low 0.04`
  - `--trend-bottom-high 1.0 --trend-bottom-low 0.5`
  - `--trend-consistent-k 1 --trend-ema-alpha 0.45`

---

### 四、现场调参流程（建议）

1) 固定 `--trend-consistent-k` 与 `--trend-ema-alpha`（如 `K=2, alpha=0.4`）。
2) 先调 `--trend-bottom-high`（如 1.5 → 1.2 → 1.0），使明显靠近能稳定触发。
3) 再调 `--trend-scale-high`（如 0.12 → 0.10 → 0.08），弥补底边不明显场景（侧向/不居中）。
4) 视抖动情况微调低阈值（`--trend-bottom-low`、`--trend-scale-low`）以优化滞回与回落。
5) 若仍有跳变误报：增大 `K` 或减小 `alpha`；若仍不敏感：减小高阈值或缩短 `K`。

---

### 五、与 ByteTrack 参数的关系（可选项）

- `--track-thresh`（默认 0.3）：降低会增加低分目标参与跟踪，可能引入轻微抖动；一般保持默认或略升。
- `--match-thresh`（默认 0.8）：关联阈值，过低会造成误匹配；建议默认。
- `--track-buffer`（默认 60）：轨迹丢失缓冲帧数，也用于清理趋势状态的“老化窗口”。

---

### 六、常见问题与建议

- “靠近”触发太慢：
  - 降低 `--trend-bottom-high`（例如 1.5 → 1.0），或降低 `--trend-scale-high`（0.12 → 0.08）
  - 降低 `--trend-consistent-k`（2 → 1），或增大 `--trend-ema-alpha`（0.4 → 0.5）

- 易误报（路面颠簸/头盔抖动）：
  - 提高高阈值（如 bottom 1.5 → 2.0；scale 0.12 → 0.15）
  - 提高 `--trend-consistent-k`（2 → 3），或降低 `--trend-ema-alpha`（0.4 → 0.3）

- 仅显示“靠近/远离”，不显示“稳定”
  - 代码已按此策略实现：不满足置信条件时不渲染趋势文字。

---

### 七、输入与运行小贴士

- 指定摄像头：`--input /dev/video0`
- 指定视频（自动在 `source/video` 下搜索）：`--input attach_intention.mp4`
- 退出实时窗口：按 `q`


