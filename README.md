# RDK ByteTrack

在 D-Robotics RDK 板端运行的车辆检测与跟踪 Demo。项目基于 YOLO11 BPU 量化模型和 ByteTrack，支持图片、视频和摄像头输入，并输出车辆的 `approaching`、`receding` 和 `warning` 标签。

## Features

- 运行在 RDK 板端 `hobot_dnn` 环境
- 支持图片、视频文件和摄像头输入
- 只跟踪交通工具类别
- 基于 ByteTrack 生成稳定目标 ID
- 支持靠近/远离趋势识别
- 支持基于警戒线的简单告警

## Installation

### Prerequisites

- RDK 板端 Python 3 环境
- `hobot_dnn`
- `numpy`
- `scipy`
- `opencv-python` 或系统 OpenCV

安装基础依赖：

```bash
pip install numpy scipy opencv-python
```

说明：

- `hobot_dnn` 需要使用 RDK BSP 提供的 Python 环境。
- `matching.py` 已内置回退逻辑，没有 `lap` 和 `cython_bbox` 也可以运行。

## Quick Start

运行检测脚本：

```bash
python3 python/YOLO_Detect.py
```

运行检测 + 跟踪 + 趋势识别：

```bash
python3 python/ultralytics_YOLO_ByteTrack.py
```

使用摄像头：

```bash
python3 python/ultralytics_YOLO_ByteTrack.py --input /dev/video0
```

## Usage

### 1. Detection only

默认视频：

```bash
python3 python/YOLO_Detect.py
```

指定图片：

```bash
mkdir -p outputs
python3 python/YOLO_Detect.py --input demo.jpg --output outputs/result.jpg
```

指定摄像头：

```bash
python3 python/YOLO_Detect.py --input camera
```

### 2. Tracking and trend analysis

默认视频：

```bash
python3 python/ultralytics_YOLO_ByteTrack.py
```

指定视频：

```bash
python3 python/ultralytics_YOLO_ByteTrack.py --input vehicle_track.mp4
```

更敏感的趋势参数：

```bash
python3 python/ultralytics_YOLO_ByteTrack.py \
  --trend-scale-high 0.08 \
  --trend-scale-low 0.04 \
  --trend-bottom-high 1.0 \
  --trend-bottom-low 0.5 \
  --trend-consistent-k 1 \
  --trend-ema-alpha 0.45
```

### 3. Compare two videos

```bash
python3 python/compare.py video1.mp4 video2.mp4 output.mp4
```

## Scripts

| Script | Purpose |
|------|------|
| `python/YOLO_Detect.py` | 只做目标检测与可视化 |
| `python/ultralytics_YOLO_ByteTrack.py` | 检测、跟踪、趋势识别和告警 |
| `python/compare.py` | 横向拼接两个视频做效果对比 |

## Configuration

常用参数：

| Parameter | Description |
|------|------|
| `--model-path` | BPU 量化模型路径 |
| `--input` | 图片、视频路径，或 `camera` / `/dev/video0` |
| `--output` | 图片模式输出路径 |
| `--score-thres` | 检测置信度阈值 |
| `--nms-thres` | NMS IoU 阈值 |
| `--track-thresh` | ByteTrack 跟踪阈值 |
| `--track-buffer` | 丢失轨迹保留帧数 |
| `--match-thresh` | 关联匹配阈值 |
| `--trend-*` | 趋势识别参数 |
| `--warn-*` | 告警判定参数 |

默认模型路径：

```text
source/reference_hbm_models/yolo11n_detect_bayese_640x640_nv12_int16softmax_modified.bin
```

详细调参说明见 `TUNING_GUIDE.md`。

## Runtime Behavior

- 图片输入会保存结果到 `--output`
- 视频和摄像头输入默认只实时显示，不保存输出视频
- 有显示环境时可弹窗预览
- 没有 `DISPLAY` 时会自动跳过 `imshow`
- 按 `q` 退出实时窗口

## Troubleshooting

### Error: `No module named hobot_dnn`

原因：你不在 RDK 板端 Python 环境中。

解决：

- 使用板端系统自带 `python3`
- 确认 BSP 环境已正确安装

### Error: output image cannot be saved

原因：输出目录不存在。

解决：

```bash
mkdir -p outputs
```

### No preview window

原因：当前环境没有 GUI 或没有 `DISPLAY`。

解决：

- 在桌面环境中运行
- 或直接查看日志输出

## Limitations

- 当前只对交通工具类别做跟踪和趋势判断
- 视频模式默认不保存结果文件
- 趋势识别对抖动、遮挡和小目标较敏感
- 项目依赖 RDK 板端推理环境

