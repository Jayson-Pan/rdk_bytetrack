#!/user/bin/env python
 
# Copyright (c) 2025, MaChao D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 
# 注意: 此程序在RDK板端端运行
# Attention: This program runs on RDK board.
 
# pip install scipy
 
import os
import cv2
import numpy as np
from collections import deque
# scipy
try:
    from scipy.special import softmax
    from scipy.optimize import linear_sum_assignment # ByteTrack 使用
except ImportError:
    print("scipy 未安装或不完整，正在安装/升级。")
    os.system("pip install -U scipy") # 确保已安装并更新
    from scipy.special import softmax
    from scipy.optimize import linear_sum_assignment
 
 
# hobot_dnn
try:
    from hobot_dnn import pyeasy_dnn as dnn  # BSP Python API
except ImportError:
    print("您的 Python 环境未准备好，请使用系统 python3 运行此程序。")
    exit()
 
from time import time
import argparse
import logging
import sys
 
# 导入 ByteTrack
try:
    from tracker.byte_tracker import BYTETracker
except ImportError:
    print("未找到 ByteTrack。请确保已安装并添加到 PYTHONPATH 中。")
    print("您可以从以下地址克隆: https://github.com/ifzhang/ByteTrack")
    exit()
 
 
# 日志模块配置
logging.basicConfig(
    level=logging.INFO, # 修改为 INFO 级别，以减少默认日志的冗余信息
    format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO_ByteTrack")

VIDEO_DIRECTORY = os.path.join('source', 'video')
DEFAULT_VIDEO_NAME = 'vehicle_track.mp4'
# 交通工具类别（COCO 数据集）: bicycle, car, motorcycle, airplane, bus, train, truck, boat
VEHICLE_CLASS_IDS = {1, 2, 3, 4, 5, 6, 7, 8}
TREND_HISTORY_SIZE = 10
TREND_COMPARE_WINDOW = 3
TREND_APPROACH_THRESHOLD = 0.25

# 运动补偿与趋势判定参数
MAX_BG_FEATURES = 300
GFTT_QUALITY = 0.01
GFTT_MIN_DIST = 8
LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
AFFINE_RANSAC_THRESH = 3.0
MIN_BG_MATCHES = 30

# 平滑与滞回
EMA_ALPHA = 0.4
SCALE_HIGH = 0.12
SCALE_LOW = 0.06
BOTTOM_HIGH = 1.5
BOTTOM_LOW = 0.8
CONSISTENT_K = 2

def tlwh_to_corners(tlwh):
    x, y, w, h = tlwh
    x2, y2 = x + w, y + h
    return np.array([[x, y], [x2, y], [x2, y2], [x, y2]], dtype=np.float32)

def transform_points_affine(M, pts):
    if M is None:
        return pts.copy()
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    out = (M @ pts_h.T).T
    return out

def rect_area_from_corners(pts):
    xs = pts[:, 0]
    ys = pts[:, 1]
    w = max(1.0, float(xs.max() - xs.min()))
    h = max(1.0, float(ys.max() - ys.min()))
    return w * h

def mean_bottom_y_from_corners(pts):
    # 取 y 最大的两个点的平均值
    idx = np.argsort(pts[:, 1])
    bottom_two = pts[idx[-2:], 1]
    return float(bottom_two.mean())
 
# IoU 计算的辅助函数 (如果不直接使用 ByteTrack 中的函数)
def calculate_iou(box1, box2):
    """
    计算两个边界框之间的 IoU。
    边界框格式: [x1, y1, x2, y2]
    """
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])
 
    inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
 
    if union_area == 0:
        return 0
    return inter_area / union_area
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='source/reference_hbm_models/yolo11n_detect_bayese_640x640_nv12_int16softmax_modified.bin',
                        help="""BPU 量化 *.bin 模型路径。
                                 RDK X3(模块): Bernoulli2.
                                 RDK Ultra: Bayes.
                                 RDK X5(模块): Bayes-e.
                                 RDK S100: Nash-e.
                                 RDK S100P: Nash-m.""")
    parser.add_argument('--input', type=str, default=os.path.join(VIDEO_DIRECTORY, DEFAULT_VIDEO_NAME),
                        help='要加载的测试图像或视频路径。使用 "camera" 表示摄像头。')
    parser.add_argument('--output', type=str, default='outputs/result.jpg',
                        help='处理结果的保存路径（仅图像输入时保存到文件）。')
    parser.add_argument('--classes-num', type=int, default=80, help='检测的类别数量。')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='YOLO NMS 的 IoU 阈值。') 
    parser.add_argument('--score-thres', type=float, default=0.2, help='YOLO 检测的置信度阈值。') 
    parser.add_argument('--reg', type=int, default=16, help='DFL 回归层。')
    
    # ByteTrack 特定参数 
    parser.add_argument('--track-thresh', type=float, default=0.3, help='ByteTrack 跟踪置信度阈值')
    parser.add_argument('--track-buffer', type=int, default=60, help='ByteTrack 丢失轨迹的缓冲帧数')
    parser.add_argument('--match-thresh', type=float, default=0.8, help='ByteTrack 第二次关联的匹配阈值')
    # 趋势参数（可调）
    parser.add_argument('--trend-scale-high', type=float, default=0.12, help='尺度残差高阈值（靠近/远离判定）')
    parser.add_argument('--trend-scale-low', type=float, default=0.06, help='尺度残差低阈值（滞回解除）')
    parser.add_argument('--trend-bottom-high', type=float, default=1.5, help='底边残差高阈值(px/帧)')
    parser.add_argument('--trend-bottom-low', type=float, default=0.8, help='底边残差低阈值(px/帧)')
    parser.add_argument('--trend-consistent-k', type=int, default=2, help='连帧确认阈值')
    parser.add_argument('--trend-ema-alpha', type=float, default=0.4, help='EMA 平滑系数')
    # Warning 判定参数（不使用 TTC）
    parser.add_argument('--warn-guard-frac', type=float, default=0.75, help='警戒线相对高度比例（0~1，越大越靠近底部）')
    parser.add_argument('--warn-guard-hys', type=float, default=0.02, help='警戒线滞回（相对高度）')
    parser.add_argument('--warn-min-area', type=float, default=900, help='参与告警的最小目标面积（像素）')
    parser.add_argument('--warn-enter-k', type=int, default=2, help='进入告警的连续帧数')
    parser.add_argument('--warn-exit-m', type=int, default=5, help='退出告警的连续帧数')
    
    opt = parser.parse_args()
    logger.info(opt)

    # 应用可调参数到全局配置
    global SCALE_HIGH, SCALE_LOW, BOTTOM_HIGH, BOTTOM_LOW, CONSISTENT_K, EMA_ALPHA
    SCALE_HIGH = opt.trend_scale_high
    SCALE_LOW = opt.trend_scale_low
    BOTTOM_HIGH = opt.trend_bottom_high
    BOTTOM_LOW = opt.trend_bottom_low
    CONSISTENT_K = opt.trend_consistent_k
    EMA_ALPHA = opt.trend_ema_alpha
    logger.info(f"趋势参数: scale_high={SCALE_HIGH}, scale_low={SCALE_LOW}, bottom_high={BOTTOM_HIGH}, bottom_low={BOTTOM_LOW}, K={CONSISTENT_K}, ema_alpha={EMA_ALPHA}")
    logger.info(f"告警参数: guard_frac={opt.warn_guard_frac}, guard_hys={opt.warn_guard_hys}, min_area={opt.warn_min_area}, enter_k={opt.warn_enter_k}, exit_m={opt.warn_exit_m}")
 
    # 标准化输入路径，便于直接使用视频文件名
    input_lower = opt.input.lower() if isinstance(opt.input, str) else ''
    if input_lower in ('video', 'default'):
        opt.input = os.path.join(VIDEO_DIRECTORY, DEFAULT_VIDEO_NAME)
    elif not os.path.isfile(opt.input):
        candidate_path = os.path.join(VIDEO_DIRECTORY, opt.input)
        if os.path.isfile(candidate_path):
            opt.input = candidate_path

    if not os.path.exists(opt.model_path):
        logger.error("未找到模型文件: %s", opt.model_path)
        logger.error("请将 yolo11n_detect_bayese_640x640_nv12_int16softmax_modified.bin 放置到指定路径。")
        exit(1)
 
    model = YOLO11_x5_Detect(opt)
 
    # 初始化 ByteTrack
    # 对于 ByteTrack，frame_rate 很重要。我们将在获取视频源后设置它。
    # 对于单张图像，使用默认值即可。
    class ByteTrackArgs:
        def __init__(self):
            self.track_thresh = opt.track_thresh 
            self.track_buffer = opt.track_buffer
            self.match_thresh = opt.match_thresh
            self.mot20 = False  # MOT20 数据集
            self.frame_rate = 30 # 默认值，对于视频会进行更新
 
    bytetrack_args = ByteTrackArgs()
    tracker = BYTETracker(bytetrack_args)
    
    cap = None
    is_video_input = False
    is_camera = False
    vehicle_class_ids = set(VEHICLE_CLASS_IDS)
 
    if opt.input.lower() == 'camera' or opt.input == '/dev/video0':
        # 优先使用显式设备路径与 V4L2 后端
        cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
        if not cap.isOpened():
            # 回退到索引 0
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("无法打开摄像头 /dev/video0。请检查权限与设备连接。")
        logger.info("正在从摄像头 /dev/video0 读取视频流...")
        is_video_input = True
        is_camera = True
        # 使用正确的帧率重新初始化
        fps_read = cap.get(cv2.CAP_PROP_FPS)
        bytetrack_args.frame_rate = int(fps_read) if fps_read and fps_read > 0 else 30
        tracker = BYTETracker(bytetrack_args)
    elif os.path.isfile(opt.input):
        if opt.input.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img = cv2.imread(opt.input)
            if img is None:
                raise ValueError(f"加载图像失败: {opt.input}")
 
            logger.info(f"处理单张图像: {opt.input}")
            input_tensor = model.preprocess_yuv420sp(img) # 在模型中设置 self.img_h, self.img_w 等属性
            outputs = model.c2numpy(model.forward(input_tensor))
            yolo_results = model.postProcess(outputs) # 格式: (class_id, score, x1, y1, x2, y2) 列表
 
            vehicle_results = [r for r in yolo_results if int(r[0]) in vehicle_class_ids]
            logger.info(f"检测到 {len(yolo_results)} 个物体, 其中交通工具类别 {len(vehicle_results)} 个。")
 
            detections_for_bytetrack = []
            if len(vehicle_results) > 0:
                detections_for_bytetrack = np.array(
                    [[r[2], r[3], r[4], r[5], r[1]] for r in vehicle_results] # x1,y1,x2,y2,score
                )
            
            frame_height, frame_width = model.img_h, model.img_w
            online_targets = []
            if len(detections_for_bytetrack) > 0:
                # 更新 ByteTrack
                # 由于 detections_for_bytetrack 中的坐标已经是相对于预处理后的图像尺寸，所以传入的 img_info 和 img_size 都应该是原始的图像尺寸，避免在tracker.update 中进行错误的缩放。
                online_targets = tracker.update(detections_for_bytetrack, 
                                            (frame_height, frame_width), 
                                            (frame_height, frame_width))
            
            logger.info("\033[1;32m" + "绘制跟踪结果: " + "\033[0m")
            for target in online_targets:
                tlwh = target.tlwh
                track_id = target.track_id
                score = target.score 
                # class_id = target.cls
                
                matched_class_id = -1
                best_iou = 0.01 # 设置一个小的阈值以确保有一定的重叠
                
                track_bbox_xyxy = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]
 
                for yolo_det in vehicle_results: # 使用 vehicle_results 进行匹配
                    yolo_cls, yolo_score, y_x1, y_y1, y_x2, y_y2 = yolo_det
                    yolo_bbox_xyxy = [y_x1, y_y1, y_x2, y_y2]
                    
                    current_iou = calculate_iou(track_bbox_xyxy, yolo_bbox_xyxy)
                    if current_iou > best_iou and abs(score - yolo_score) < 0.1:
                        best_iou = current_iou
                        matched_class_id = int(yolo_cls)
                
                # 仅当我们找到对应的类别时才绘制（单张图像不输出趋势标签）
                trend_label = ""
                if matched_class_id in vehicle_class_ids:
                    draw_track(img, track_bbox_xyxy, score, matched_class_id, track_id, trend_label)
                else:
                    logger.debug(f"轨迹 ID {track_id} 未能匹配到交通工具类别。IoU: {best_iou:.2f}")
 
            cv2.imwrite(opt.output, img)
            logger.info("\033[1;32m" + f"结果已保存到: \"./{opt.output}\"" + "\033[0m")
            return
        else:
            cap = cv2.VideoCapture(opt.input)
            if not cap.isOpened():
                raise IOError(f"无法打开视频文件: {opt.input}")
            logger.info(f"正在从视频文件 {opt.input} 读取帧...")
            is_video_input = True
            # 使用正确的帧率重新初始化
            bytetrack_args.frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
            tracker = BYTETracker(bytetrack_args) 
    else:
        raise ValueError(f"无效的输入路径或类型: {opt.input}. 请提供图像文件、视频文件路径或 'camera'。")
 
    # 单张图片逻辑应该已经处理了这种情况
    if not is_video_input: 
        return
 
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps 已在 bytetrack_args.frame_rate 中设置

    if not is_camera:
        logger.info("文件输入: 仅实时显示处理画面，不保存视频文件。")

    # 趋势检测状态
    track_last_seen = {}
    track_prev_boxes = {}
    track_states = {}
    prev_gray = None

    # 告警状态（per track）与阈值
    warn_states = {}
    # 警戒线（含滞回）
    y_guard = int(frame_height * opt.warn_guard_frac)
    y_guard_up = int(frame_height * min(1.0, opt.warn_guard_frac + opt.warn_guard_hys))
    y_guard_down = int(frame_height * max(0.0, opt.warn_guard_frac - opt.warn_guard_hys))
    logger.info(f"Guard line y={y_guard} (up={y_guard_up}, down={y_guard_down})")
 
    frame_count = 0
    prev_time = time()
    can_show = bool(os.environ.get('DISPLAY'))
    if can_show:
        try:
            cv2.namedWindow("RDK YOLO ByteTrack", cv2.WINDOW_NORMAL)
        except Exception:
            can_show = False
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("视频流结束或无法读取帧。")
            break
 
        frame_count += 1
        # 计算 FPS
        now_time = time()
        fps = 1.0 / max(now_time - prev_time, 1e-6)
        prev_time = now_time
        # logger.debug(f"正在处理第 {frame_count} 帧...")
        input_tensor = model.preprocess_yuv420sp(frame)
        outputs_raw = model.forward(input_tensor)
        outputs_np = model.c2numpy(outputs_raw)
        yolo_results_all = model.postProcess(outputs_np)
 
        # --- 只筛选交通工具类别 ---
        vehicle_results = [r for r in yolo_results_all if int(r[0]) in vehicle_class_ids]
        if frame_count % 30 == 0: # 每30帧打印一次数量信息，避免日志过多
             logger.info(f"帧 {frame_count}: 检测到 {len(yolo_results_all)} 个物体, 其中交通工具类别 {len(vehicle_results)} 个。")

        # 背景特征点与全局仿射（相机运动）估计
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        M_global = None
        if prev_gray is not None:
            # 生成背景掩膜（排除车辆区域）
            bg_mask = np.full(prev_gray.shape, 255, dtype=np.uint8)
            for det in vehicle_results:
                _, _, x1, y1, x2, y2 = det
                x1c = max(0, int(x1)); y1c = max(0, int(y1)); x2c = min(int(x2), bg_mask.shape[1]-1); y2c = min(int(y2), bg_mask.shape[0]-1)
                if x2c > x1c and y2c > y1c:
                    bg_mask[y1c:y2c, x1c:x2c] = 0
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=MAX_BG_FEATURES, qualityLevel=GFTT_QUALITY, minDistance=GFTT_MIN_DIST, mask=bg_mask)
            if prev_pts is not None and len(prev_pts) >= MIN_BG_MATCHES:
                curr_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, winSize=LK_WIN_SIZE, maxLevel=LK_MAX_LEVEL, criteria=LK_CRITERIA)
                st = st.reshape(-1) if st is not None else None
                if curr_pts is not None and st is not None:
                    p0 = prev_pts.reshape(-1, 2)[st == 1]
                    p1 = curr_pts.reshape(-1, 2)[st == 1]
                    if len(p0) >= MIN_BG_MATCHES:
                        M_global, inliers = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=AFFINE_RANSAC_THRESH, confidence=0.99, maxIters=1000)
                        if M_global is None and len(p0) > 0:
                            # 回退到平移
                            delta = np.median(p1 - p0, axis=0)
                            M_global = np.array([[1.0, 0.0, float(delta[0])],[0.0, 1.0, float(delta[1])]], dtype=np.float32)
            if M_global is None:
                M_global = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0]], dtype=np.float32)
        else:
            # 第一帧无运动估计
            M_global = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0]], dtype=np.float32)

        detections_for_bytetrack = []
        if len(vehicle_results) > 0: # 使用筛选后的 vehicle_results
            detections_for_bytetrack = np.array(
                [[r[2], r[3], r[4], r[5], r[1]] for r in vehicle_results] # x1,y1,x2,y2,score
            )
        
        online_targets = []
        if len(detections_for_bytetrack) > 0:
            t1 = time() # 记录更新前的时间
            # 更新 ByteTrack
            # 由于 detections_for_bytetrack 中的坐标已经是相对于预处理后的图像尺寸，所以传入的 img_info 和 img_size 都应该是原始的图像尺寸，避免在tracker.update 中进行错误的缩放。
            online_targets = tracker.update(detections_for_bytetrack, 
                                            (frame_height, frame_width), 
                                            (frame_height, frame_width))
            t2 = time()
            logger.info(f"ByteTrack 更新耗时 = {1000 * (t2 - t1):.2f} ms, 当前在线目标数: {len(online_targets)}")
 
        for target in online_targets:
            tlwh = target.tlwh
            track_id = target.track_id
            score = target.score
 
            matched_class_id = -1
            best_iou = 0.01
            track_bbox_xyxy = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]
 
            for yolo_det in vehicle_results: # 使用 vehicle_results 进行匹配
                yolo_cls, yolo_score, y_x1, y_y1, y_x2, y_y2 = yolo_det
                yolo_bbox_xyxy = [y_x1, y_y1, y_x2, y_y2]
                
                current_iou = calculate_iou(track_bbox_xyxy, yolo_bbox_xyxy)
                if current_iou > best_iou and abs(score - yolo_score) < 0.1:
                    best_iou = current_iou
                    matched_class_id = int(yolo_cls)
            
            trend_label = ""
            if matched_class_id in vehicle_class_ids: # 确保匹配到的确实是交通工具
                # 使用全局仿射进行相机运动补偿的尺度/底边趋势估计
                prev_box = track_prev_boxes.get(track_id)
                if prev_box is not None and prev_gray is not None:
                    prev_corners = tlwh_to_corners(prev_box)
                    pred_corners = transform_points_affine(M_global, prev_corners)
                    pred_area = rect_area_from_corners(pred_corners)
                    curr_area = max(1.0, float(tlwh[2] * tlwh[3]))
                    scale_resid = (curr_area - pred_area) / max(pred_area, 1e-6)

                    pred_bottom_y = mean_bottom_y_from_corners(pred_corners)
                    curr_bottom_y = float(tlwh[1] + tlwh[3])
                    bottom_resid = curr_bottom_y - pred_bottom_y

                    state = track_states.get(track_id)
                    if state is None:
                        state = {
                            'ema_scale': scale_resid,
                            'ema_bottom': bottom_resid,
                            'label': '',
                            'prop_type': None,
                            'prop_count': 0,
                            'clear_count': 0
                        }
                    else:
                        state['ema_scale'] = EMA_ALPHA * scale_resid + (1.0 - EMA_ALPHA) * state['ema_scale']
                        state['ema_bottom'] = EMA_ALPHA * bottom_resid + (1.0 - EMA_ALPHA) * state['ema_bottom']

                    # 提议类型：仅靠近/远离；否则 None（不显示）
                    proposed = None
                    if state['ema_scale'] > SCALE_HIGH or state['ema_bottom'] > BOTTOM_HIGH:
                        proposed = 'approaching'
                    elif state['ema_scale'] < -SCALE_HIGH or state['ema_bottom'] < -BOTTOM_HIGH:
                        proposed = 'receding'

                    # 更新滞回/连帧确认
                    if proposed is not None:
                        if proposed == state['label']:
                            state['prop_type'] = None
                            state['prop_count'] = 0
                            state['clear_count'] = 0
                        else:
                            if proposed == state['prop_type']:
                                state['prop_count'] += 1
                            else:
                                state['prop_type'] = proposed
                                state['prop_count'] = 1
                            if state['prop_count'] >= CONSISTENT_K:
                                state['label'] = proposed
                                state['prop_type'] = None
                                state['prop_count'] = 0
                                state['clear_count'] = 0
                    else:
                        # 判断是否清除已有标签
                        if state['label']:
                            if (abs(state['ema_scale']) < SCALE_LOW) and (abs(state['ema_bottom']) < BOTTOM_LOW):
                                state['clear_count'] += 1
                                if state['clear_count'] >= CONSISTENT_K:
                                    state['label'] = ''
                                    state['clear_count'] = 0
                            else:
                                state['clear_count'] = 0
                        state['prop_type'] = None
                        state['prop_count'] = 0

                    trend_label = state['label']
                    track_states[track_id] = state

                # warning 判定（不使用 TTC）
                # 目标几何信息
                width = float(max(1.0, tlwh[2]))
                height = float(max(1.0, tlwh[3]))
                area_px = width * height
                bottom_y = float(tlwh[1] + tlwh[3])
                center_x = float(tlwh[0] + tlwh[2] * 0.5)

                area_ok = (area_px >= float(opt.warn_min_area))
                within_guard_enter = (bottom_y >= float(y_guard_up))
                within_guard_exit = (bottom_y <= float(y_guard_down))

                state = track_states.get(track_id)
                ema_scale = state['ema_scale'] if state is not None else 0.0
                ema_bottom = state['ema_bottom'] if state is not None else 0.0
                curr_trend = state['label'] if state is not None else ''
                approaching_gate = (curr_trend == 'approaching') and (ema_scale > SCALE_LOW or ema_bottom > BOTTOM_LOW)

                wstate = warn_states.get(track_id)
                if wstate is None:
                    wstate = {'active': False, 'confirm': 0, 'clear': 0}

                if not wstate['active']:
                    if approaching_gate and area_ok and within_guard_enter and (ema_bottom >= BOTTOM_HIGH or ema_scale >= SCALE_HIGH):
                        wstate['confirm'] += 1
                        if wstate['confirm'] >= int(opt.warn_enter_k):
                            wstate['active'] = True
                            wstate['confirm'] = 0
                            wstate['clear'] = 0
                    else:
                        wstate['confirm'] = 0
                else:
                    # 退出条件
                    if (curr_trend == 'receding') or within_guard_exit or ((abs(ema_bottom) < BOTTOM_LOW) and (abs(ema_scale) < SCALE_LOW)):
                        wstate['clear'] += 1
                        if wstate['clear'] >= int(opt.warn_exit_m):
                            wstate['active'] = False
                            wstate['clear'] = 0
                            wstate['confirm'] = 0
                    else:
                        wstate['clear'] = 0

                warn_states[track_id] = wstate

                # 构建标签文本：approaching/receding + warning（若触发）
                trend_text = trend_label
                if wstate['active']:
                    trend_text = (trend_text + " | warning") if trend_text else "warning"

                track_last_seen[track_id] = frame_count
                draw_track(frame, track_bbox_xyxy, score, matched_class_id, track_id, trend_text)
            else:
                # 可选：绘制未匹配的轨迹或记录日志
                # cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])), (0,0,255), 2)
                # cv2.putText(frame, f"ID:{track_id} (Unmatched)", (int(tlwh[0]), int(tlwh[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                logger.debug(f"轨迹 ID {track_id} 在第 {frame_count} 帧未能可靠匹配到交通工具类别。")

        # 记录上一帧 box
        for target in online_targets:
            track_prev_boxes[target.track_id] = target.tlwh.copy()

        # 清理长时间未出现的轨迹状态，防止内存增长
        stale_ids = [tid for tid, last_seen in track_last_seen.items() if frame_count - last_seen > opt.track_buffer]
        for tid in stale_ids:
            track_last_seen.pop(tid, None)
            track_prev_boxes.pop(tid, None)
            track_states.pop(tid, None)
 
 
        # 警戒线可视化（加粗并加标签，增强可见性）
        cv2.line(frame, (0, y_guard), (frame_width - 1, y_guard), (0, 255, 255), 3)
        cv2.putText(frame, f"guard {opt.warn_guard_frac:.2f}", (10, max(20, y_guard - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        # 叠加 FPS 与帧号
        cv2.putText(frame, f"FPS: {fps:.1f}  Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 实时显示（若环境支持 GUI）
        if can_show:
            try:
                cv2.imshow("RDK YOLO ByteTrack", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as _:
                # 在无显示环境时禁用显示
                can_show = False
 
        # 更新上一帧灰度图
        prev_gray = gray

    cap.release()
    if can_show:
        cv2.destroyAllWindows()
class YOLO11_x5_Detect():
    def __init__(self, opt):
        # 加载BPU的bin模型, 打印相关参数
        # Load the quantized *.bin model and print its parameters
        try:
            begin_time = time()
            self.quantize_model = dnn.load(opt.model_path)
            logger.debug("\033[1;31m" + "Load D-Robotics Quantize model time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")
        except Exception as e:
            logger.error("❌ Failed to load model file: %s"%(opt.model_path))
            logger.error("You can download the model file from the following docs: ./models/download.md") 
            logger.error(e)
            exit(1)
 
        logger.info("\033[1;32m" + "-> input tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")
 
        logger.info("\033[1;32m" + "-> output tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].outputs):
            logger.info(f"output[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")
 
        # 将反量化系数准备好, 只需要准备一次
        # prepare the quantize scale, just need to generate once
        self.s_bboxes_scale = self.quantize_model[0].outputs[1].properties.scale_data[np.newaxis, :]
        self.m_bboxes_scale = self.quantize_model[0].outputs[3].properties.scale_data[np.newaxis, :]
        self.l_bboxes_scale = self.quantize_model[0].outputs[5].properties.scale_data[np.newaxis, :]
        logger.info(f"{self.s_bboxes_scale.shape=}, {self.m_bboxes_scale.shape=}, {self.l_bboxes_scale.shape=}")
 
        # DFL求期望的系数, 只需要生成一次
        # DFL calculates the expected coefficients, which only needs to be generated once.
        self.weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, np.newaxis, :]
        logger.info(f"{self.weights_static.shape = }")
 
        # anchors, 只需要生成一次
        self.s_anchor = np.stack([np.tile(np.linspace(0.5, 79.5, 80), reps=80), 
                            np.repeat(np.arange(0.5, 80.5, 1), 80)], axis=0).transpose(1,0)
        self.m_anchor = np.stack([np.tile(np.linspace(0.5, 39.5, 40), reps=40), 
                            np.repeat(np.arange(0.5, 40.5, 1), 40)], axis=0).transpose(1,0)
        self.l_anchor = np.stack([np.tile(np.linspace(0.5, 19.5, 20), reps=20), 
                            np.repeat(np.arange(0.5, 20.5, 1), 20)], axis=0).transpose(1,0)
        logger.info(f"{self.s_anchor.shape = }, {self.m_anchor.shape = }, {self.l_anchor.shape = }")
 
        # 输入图像大小, 一些阈值, 提前计算好
        self.input_image_size = 640
        self.SCORE_THRESHOLD = opt.score_thres
        self.NMS_THRESHOLD = opt.nms_thres
        self.CONF_THRES_RAW = -np.log(1/self.SCORE_THRESHOLD - 1)
        logger.info("SCORE_THRESHOLD  = %.2f, NMS_THRESHOLD = %.2f"%(self.SCORE_THRESHOLD, self.NMS_THRESHOLD))
        logger.info("CONF_THRES_RAW = %.2f"%self.CONF_THRES_RAW)
 
        self.input_H, self.input_W = self.quantize_model[0].inputs[0].properties.shape[2:4]
        logger.info(f"{self.input_H = }, {self.input_W = }")
 
        self.REG = opt.reg
        print(f"{self.REG = }")
 
        self.CLASSES_NUM = opt.classes_num
        print(f"{self.CLASSES_NUM = }")
 
    def preprocess_yuv420sp(self, img):
        RESIZE_TYPE = 0
        LETTERBOX_TYPE = 1
        PREPROCESS_TYPE = LETTERBOX_TYPE
        logger.info(f"PREPROCESS_TYPE = {PREPROCESS_TYPE}")
 
        begin_time = time()
        self.img_h, self.img_w = img.shape[0:2]
        if PREPROCESS_TYPE == RESIZE_TYPE:
            # 利用resize的方式进行前处理, 准备nv12的输入数据
            begin_time = time()
            input_tensor = cv2.resize(img, (self.input_W, self.input_H), interpolation=cv2.INTER_NEAREST) # 利用resize重新开辟内存节约一次
            input_tensor = self.bgr2nv12(input_tensor)
            self.y_scale = 1.0 * self.input_H / self.img_h
            self.x_scale = 1.0 * self.input_W / self.img_w
            self.y_shift = 0
            self.x_shift = 0
            logger.info("\033[1;31m" + f"pre process(resize) time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        elif PREPROCESS_TYPE == LETTERBOX_TYPE:
            # 利用 letter box 的方式进行前处理, 准备nv12的输入数据
            begin_time = time()
            self.x_scale = min(1.0 * self.input_H / self.img_h, 1.0 * self.input_W / self.img_w)
            self.y_scale = self.x_scale
            
            if self.x_scale <= 0 or self.y_scale <= 0:
                raise ValueError("Invalid scale factor.")
            
            new_w = int(self.img_w * self.x_scale)
            self.x_shift = (self.input_W - new_w) // 2
            x_other = self.input_W - new_w - self.x_shift
            
            new_h = int(self.img_h * self.y_scale)
            self.y_shift = (self.input_H - new_h) // 2
            y_other = self.input_H - new_h - self.y_shift
            
            input_tensor = cv2.resize(img, (new_w, new_h))
            input_tensor = cv2.copyMakeBorder(input_tensor, self.y_shift, y_other, self.x_shift, x_other, cv2.BORDER_CONSTANT, value=[127, 127, 127])
            input_tensor = self.bgr2nv12(input_tensor)
            logger.info("\033[1;31m" + f"pre process(letter box) time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        else:
            logger.error(f"illegal PREPROCESS_TYPE = {PREPROCESS_TYPE}")
            exit(-1)
 
        logger.debug("\033[1;31m" + f"pre process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        logger.info(f"y_scale = {self.y_scale:.2f}, x_scale = {self.x_scale:.2f}")
        logger.info(f"y_shift = {self.y_shift:.2f}, x_shift = {self.x_shift:.2f}")
        return input_tensor
 
    def bgr2nv12(self, bgr_img):
        begin_time = time()
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        logger.debug("\033[1;31m" + f"bgr8 to nv12 time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return nv12
 
    def forward(self, input_tensor):
        begin_time = time()
        quantize_outputs = self.quantize_model[0].forward(input_tensor)
        logger.debug("\033[1;31m" + f"forward time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return quantize_outputs
 
    def c2numpy(self, outputs):
        begin_time = time()
        outputs = [dnnTensor.buffer for dnnTensor in outputs]
        logger.debug("\033[1;31m" + f"c to numpy time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return outputs
 
    def postProcess(self, outputs):
        begin_time = time()
        # reshape
        s_clses = outputs[0].reshape(-1, self.CLASSES_NUM)
        s_bboxes = outputs[1].reshape(-1, self.REG * 4)
        m_clses = outputs[2].reshape(-1, self.CLASSES_NUM)
        m_bboxes = outputs[3].reshape(-1, self.REG * 4)
        l_clses = outputs[4].reshape(-1, self.CLASSES_NUM)
        l_bboxes = outputs[5].reshape(-1, self.REG * 4)
 
        # classify: 利用numpy向量化操作完成阈值筛选(优化版 2.0)
        s_max_scores = np.max(s_clses, axis=1)
        s_valid_indices = np.flatnonzero(s_max_scores >= self.CONF_THRES_RAW)  # 得到大于阈值分数的索引，此时为小数字
        s_ids = np.argmax(s_clses[s_valid_indices, : ], axis=1)
        s_scores = s_max_scores[s_valid_indices]
 
        m_max_scores = np.max(m_clses, axis=1)
        m_valid_indices = np.flatnonzero(m_max_scores >= self.CONF_THRES_RAW)  # 得到大于阈值分数的索引，此时为小数字
        m_ids = np.argmax(m_clses[m_valid_indices, : ], axis=1)
        m_scores = m_max_scores[m_valid_indices]
 
        l_max_scores = np.max(l_clses, axis=1)
        l_valid_indices = np.flatnonzero(l_max_scores >= self.CONF_THRES_RAW)  # 得到大于阈值分数的索引，此时为小数字
        l_ids = np.argmax(l_clses[l_valid_indices, : ], axis=1)
        l_scores = l_max_scores[l_valid_indices]
 
        # 3个Classify分类分支：Sigmoid计算
        s_scores = 1 / (1 + np.exp(-s_scores))
        m_scores = 1 / (1 + np.exp(-m_scores))
        l_scores = 1 / (1 + np.exp(-l_scores))
 
        # 3个Bounding Box分支：反量化
        s_bboxes_float32 = s_bboxes[s_valid_indices,:].astype(np.float32) * self.s_bboxes_scale
        m_bboxes_float32 = m_bboxes[m_valid_indices,:].astype(np.float32) * self.m_bboxes_scale
        l_bboxes_float32 = l_bboxes[l_valid_indices,:].astype(np.float32) * self.l_bboxes_scale
 
        # 3个Bounding Box分支：dist2bbox (ltrb2xyxy)
        s_ltrb_indices = np.sum(softmax(s_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        s_anchor_indices = self.s_anchor[s_valid_indices, :]
        s_x1y1 = s_anchor_indices - s_ltrb_indices[:, 0:2]
        s_x2y2 = s_anchor_indices + s_ltrb_indices[:, 2:4]
        s_dbboxes = np.hstack([s_x1y1, s_x2y2])*8
 
        m_ltrb_indices = np.sum(softmax(m_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        m_anchor_indices = self.m_anchor[m_valid_indices, :]
        m_x1y1 = m_anchor_indices - m_ltrb_indices[:, 0:2]
        m_x2y2 = m_anchor_indices + m_ltrb_indices[:, 2:4]
        m_dbboxes = np.hstack([m_x1y1, m_x2y2])*16
 
        l_ltrb_indices = np.sum(softmax(l_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        l_anchor_indices = self.l_anchor[l_valid_indices,:]
        l_x1y1 = l_anchor_indices - l_ltrb_indices[:, 0:2]
        l_x2y2 = l_anchor_indices + l_ltrb_indices[:, 2:4]
        l_dbboxes = np.hstack([l_x1y1, l_x2y2])*32
 
        # 大中小特征层阈值筛选结果拼接
        dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0)
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)
 
        # xyxy 2 xyhw
        # xy = (dbboxes[:,2:4] + dbboxes[:,0:2])/2.0
        hw = (dbboxes[:,2:4] - dbboxes[:,0:2])
        # xyhw = np.hstack([xy, hw])
 
        xyhw2 = np.hstack([dbboxes[:,0:2], hw])
 
        # 分类别nms
        results = []
        for i in range(self.CLASSES_NUM):
            id_indices = ids==i
            indices = cv2.dnn.NMSBoxes(xyhw2[id_indices,:], scores[id_indices], self.SCORE_THRESHOLD, self.NMS_THRESHOLD)
            if len(indices) == 0:
                continue
            for indic in indices:
                x1, y1, x2, y2 = dbboxes[id_indices,:][indic]
                x1 = int((x1 - self.x_shift) / self.x_scale)
                y1 = int((y1 - self.y_shift) / self.y_scale)
                x2 = int((x2 - self.x_shift) / self.x_scale)
                y2 = int((y2 - self.y_shift) / self.y_scale)
 
                x1 = x1 if x1 > 0 else 0
                x2 = x2 if x2 > 0 else 0
                y1 = y1 if y1 > 0 else 0
                y2 = y2 if y2 > 0 else 0
                x1 = x1 if x1 < self.img_w else self.img_w
                x2 = x2 if x2 < self.img_w else self.img_w
                y1 = y1 if y1 < self.img_h else self.img_h
                y2 = y2 if y2 < self.img_h else self.img_h
 
                results.append((i, scores[id_indices][indic], x1, y1, x2, y2))
 
        logger.debug("\033[1;31m" + f"Post Process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
 
        return results
class YOLO11_Detect():
    def __init__(self, opt):
        try:
            begin_time = time()
            self.quantize_model = dnn.load(opt.model_path)
            logger.debug("\033[1;31m" + "加载 D-Robotics 量化模型耗时 = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")
        except Exception as e:
            logger.error("❌ 加载模型文件失败: %s"%(opt.model_path))
            logger.error(e) # 打印原始错误信息
            exit(1)
 
        logger.info("\033[1;32m" + "-> 输入张量" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(f"输入[{i}], 名称={quantize_input.name}, 类型={quantize_input.properties.dtype}, 形状={quantize_input.properties.shape}")
 
        logger.info("\033[1;32m" + "-> 输出张量" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].outputs): # 这里应该是 quantize_output
            logger.info(f"输出[{i}], 名称={quantize_input.name}, 类型={quantize_input.properties.dtype}, 形状={quantize_input.properties.shape}")
 
        # 存储原始图像高宽和模型输入高宽
        self.img_h, self.img_w = 0, 0 
        self.input_H, self.input_W = self.quantize_model[0].inputs[0].properties.shape[1:3]
        logger.info(f"模型输入高度 = {self.input_H}, 模型输入宽度 = {self.input_W}")
        
        self.s_bboxes_scale = self.quantize_model[0].outputs[1].properties.scale_data[np.newaxis, :]
        self.m_bboxes_scale = self.quantize_model[0].outputs[3].properties.scale_data[np.newaxis, :]
        self.l_bboxes_scale = self.quantize_model[0].outputs[5].properties.scale_data[np.newaxis, :]
        
        self.weights_static = np.array([i for i in range(opt.reg)]).astype(np.float32)[np.newaxis, np.newaxis, :] # 使用 opt.reg
        
        # 确保使用整数除法 //
        s_feat_h, s_feat_w = self.input_H // 8, self.input_W // 8
        m_feat_h, m_feat_w = self.input_H // 16, self.input_W // 16
        l_feat_h, l_feat_w = self.input_H // 32, self.input_W // 32
 
        self.s_anchor = np.stack([np.tile(np.linspace(0.5, s_feat_w - 0.5, s_feat_w), reps=s_feat_h),
                                  np.repeat(np.arange(0.5, s_feat_h, 1), s_feat_w)], axis=0).transpose(1,0)
        self.m_anchor = np.stack([np.tile(np.linspace(0.5, m_feat_w - 0.5, m_feat_w), reps=m_feat_h),
                                  np.repeat(np.arange(0.5, m_feat_h, 1), m_feat_w)], axis=0).transpose(1,0)
        self.l_anchor = np.stack([np.tile(np.linspace(0.5, l_feat_w - 0.5, l_feat_w), reps=l_feat_h),
                                  np.repeat(np.arange(0.5, l_feat_h, 1), l_feat_w)], axis=0).transpose(1,0)
 
 
        self.SCORE_THRESHOLD = opt.score_thres
        self.NMS_THRESHOLD = opt.nms_thres
        self.CONF_THRES_RAW = -np.log(1/self.SCORE_THRESHOLD - 1) if self.SCORE_THRESHOLD > 0 and self.SCORE_THRESHOLD < 1 else self.SCORE_THRESHOLD 
 
        self.REG = opt.reg
        self.CLASSES_NUM = opt.classes_num
        
        # 用于将坐标缩放回原始图像
        self.x_scale, self.y_scale = 1.0, 1.0
        self.x_shift, self.y_shift = 0, 0
 
 
    def preprocess_yuv420sp(self, img):
        # 确保 self.img_h, self.img_w 设置为原始图像维度
        # 并且 self.x_scale, self.y_scale, self.x_shift, self.y_shift 计算正确
        # 以便 postProcess 将坐标转换回原始图像空间。
        RESIZE_TYPE = 0
        LETTERBOX_TYPE = 1
        PREPROCESS_TYPE = LETTERBOX_TYPE # 或根据您的模型选择 RESIZE_TYPE
        # logger.info(f"预处理类型 = {PREPROCESS_TYPE}") # 此日志可能比较多余
 
        #局部计时器
        # begin_time = time() 
        self.img_h, self.img_w = img.shape[0:2] # 存储原始维度
        
        if PREPROCESS_TYPE == RESIZE_TYPE:
            self.y_scale = 1.0 * self.input_H / self.img_h
            self.x_scale = 1.0 * self.input_W / self.img_w
            self.y_shift = 0
            self.x_shift = 0
            input_tensor = cv2.resize(img, (self.input_W, self.input_H), interpolation=cv2.INTER_NEAREST)
            input_tensor = self.bgr2nv12(input_tensor)
 
        elif PREPROCESS_TYPE == LETTERBOX_TYPE:
            self.x_scale = min(1.0 * self.input_H / self.img_h, 1.0 * self.input_W / self.img_w)
            self.y_scale = self.x_scale # letterbox 通常 x, y 轴使用相同缩放比例
 
            if self.x_scale <= 0 or self.y_scale <= 0: # 确保缩放有效
                logger.warning(f"预处理中出现无效的缩放因子 ({self.x_scale}, {self.y_scale})，图像尺寸: {self.img_w}x{self.img_h}，模型输入: {self.input_W}x{self.input_H}")
                self.x_scale = 1.0
                self.y_scale = 1.0
 
 
            new_w = int(self.img_w * self.x_scale)
            self.x_shift = (self.input_W - new_w) // 2
            x_other = self.input_W - new_w - self.x_shift
 
            new_h = int(self.img_h * self.y_scale)
            self.y_shift = (self.input_H - new_h) // 2
            y_other = self.input_H - new_h - self.y_shift
            
            # 确保 new_w 和 new_h > 0
            if new_w <=0 or new_h <=0:
                new_w = self.img_w
                new_h = self.img_h
                self.x_shift = (self.input_W - new_w) // 2
                x_other = self.input_W - new_w - self.x_shift
                self.y_shift = (self.input_H - new_h) // 2
                y_other = self.input_H - new_h - self.y_shift
                resized_img = img # 不进行resize
            else:
                 resized_img = cv2.resize(img, (new_w, new_h))
 
            input_tensor = cv2.copyMakeBorder(resized_img, self.y_shift, y_other, self.x_shift, x_other, cv2.BORDER_CONSTANT, value=[127, 127, 127])
            input_tensor = self.bgr2nv12(input_tensor)
        else:
            logger.error(f"非法的预处理类型 = {PREPROCESS_TYPE}")
            exit(-1)
        # logger.debug(f"y_scale = {self.y_scale:.2f}, x_scale = {self.x_scale:.2f}, y_shift = {self.y_shift}, x_shift = {self.x_shift}")
        return input_tensor
 
    def bgr2nv12(self, bgr_img):
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width        
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        return nv12
        
    def forward(self, input_tensor):
        begin_time = time()
        quantize_outputs = self.quantize_model[0].forward(input_tensor)
        logger.debug("\033[1;31m" + f"推理耗时 = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return quantize_outputs
 
    def c2numpy(self, outputs):
        begin_time = time()
        outputs_np = [dnnTensor.buffer for dnnTensor in outputs] # 重命名以避免冲突
        logger.debug("\033[1;31m" + f"C 结构转 NumPy 数组耗时 = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return outputs_np
 
    def postProcess(self, outputs_np):
        begin_time = time()
        s_clses = outputs_np[0].reshape(-1, self.CLASSES_NUM)
        s_bboxes = outputs_np[1].reshape(-1, self.REG * 4)
        m_clses = outputs_np[2].reshape(-1, self.CLASSES_NUM)
        m_bboxes = outputs_np[3].reshape(-1, self.REG * 4)
        l_clses = outputs_np[4].reshape(-1, self.CLASSES_NUM)
        l_bboxes = outputs_np[5].reshape(-1, self.REG * 4)
 
        s_max_scores = np.max(s_clses, axis=1)
        s_valid_indices = np.flatnonzero(s_max_scores >= self.CONF_THRES_RAW)
        s_ids = np.argmax(s_clses[s_valid_indices, : ], axis=1)
        s_scores = s_max_scores[s_valid_indices]
 
        m_max_scores = np.max(m_clses, axis=1)
        m_valid_indices = np.flatnonzero(m_max_scores >= self.CONF_THRES_RAW)
        m_ids = np.argmax(m_clses[m_valid_indices, : ], axis=1)
        m_scores = m_max_scores[m_valid_indices]
 
        l_max_scores = np.max(l_clses, axis=1)
        l_valid_indices = np.flatnonzero(l_max_scores >= self.CONF_THRES_RAW)
        l_ids = np.argmax(l_clses[l_valid_indices, : ], axis=1)
        l_scores = l_max_scores[l_valid_indices]
 
        s_scores = 1 / (1 + np.exp(-s_scores))
        m_scores = 1 / (1 + np.exp(-m_scores))
        l_scores = 1 / (1 + np.exp(-l_scores))
 
        s_bboxes_float32 = s_bboxes[s_valid_indices,:].astype(np.float32) * self.s_bboxes_scale
        m_bboxes_float32 = m_bboxes[m_valid_indices,:].astype(np.float32) * self.m_bboxes_scale
        l_bboxes_float32 = l_bboxes[l_valid_indices,:].astype(np.float32) * self.l_bboxes_scale
        
        s_stride, m_stride, l_stride = 8, 16, 32
 
        s_ltrb_indices = np.sum(softmax(s_bboxes_float32.reshape(-1, 4, self.REG), axis=2) * self.weights_static, axis=2)
        s_anchor_indices = self.s_anchor[s_valid_indices, :]
        s_x1y1 = s_anchor_indices - s_ltrb_indices[:, 0:2]
        s_x2y2 = s_anchor_indices + s_ltrb_indices[:, 2:4]
        s_dbboxes = np.hstack([s_x1y1, s_x2y2]) * s_stride
 
        m_ltrb_indices = np.sum(softmax(m_bboxes_float32.reshape(-1, 4, self.REG), axis=2) * self.weights_static, axis=2)
        m_anchor_indices = self.m_anchor[m_valid_indices, :]
        m_x1y1 = m_anchor_indices - m_ltrb_indices[:, 0:2]
        m_x2y2 = m_anchor_indices + m_ltrb_indices[:, 2:4]
        m_dbboxes = np.hstack([m_x1y1, m_x2y2]) * m_stride
 
        l_ltrb_indices = np.sum(softmax(l_bboxes_float32.reshape(-1, 4, self.REG), axis=2) * self.weights_static, axis=2)
        l_anchor_indices = self.l_anchor[l_valid_indices,:]
        l_x1y1 = l_anchor_indices - l_ltrb_indices[:, 0:2]
        l_x2y2 = l_anchor_indices + l_ltrb_indices[:, 2:4]
        l_dbboxes = np.hstack([l_x1y1, l_x2y2]) * l_stride
        
        dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0)
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)
 
        # x1,y1,w,h for NMS
        xyhw2 = np.hstack([dbboxes[:,0:2], dbboxes[:,2:4] - dbboxes[:,0:2]]) 
 
        results = []
        # 对所有检测结果进行一次性阈值筛选
        score_mask_all = scores >= self.SCORE_THRESHOLD
        
        # 获取通过分数阈值的检测的索引、边界框、分数和类别ID
        # 注意：这里的索引是相对于 dbboxes, scores, ids 这些拼接后数组的
        indices_passed_score = np.where(score_mask_all)[0]
        
        # 如果没有检测结果通过分数阈值，则直接返回空列表
        if len(indices_passed_score) == 0:
            return results
 
        # 获取通过分数阈值的检测框、分数和类别
        dbboxes_passed_score = dbboxes[indices_passed_score, :]
        scores_passed_score = scores[indices_passed_score]
        ids_passed_score = ids[indices_passed_score]
        xyhw2_passed_score = xyhw2[indices_passed_score, :]
 
        # 对通过分数阈值的检测结果进行NMS (OpenCV NMSBoxes要求 list of bboxes 和 list of scores)
        # cv2.dnn.NMSBoxesBoxes 对每个类别是分开做的，但这里我们先对所有类别一起做，然后再按类别分开，或者，如果您想严格按类别NMS，则需要循环每个类别
        # 为了简化，这里演示一个更直接的方式，先获取所有高分框，然后NMS
        # 注意：标准的NMS通常是按类别进行的。如果模型输出很多重叠的不同类别的框，这里的全局NMS可能不是最优选择。
 
        for i in range(self.CLASSES_NUM):
            # 1. 筛选出当前类别的检测结果
            class_specific_mask_initial = (ids == i)
            if not np.any(class_specific_mask_initial):
                continue
            
            # 2. 对当前类别的检测结果应用分数阈值
            # class_scores_current_cat = scores[class_specific_mask_initial]
            # class_dbboxes_current_cat = dbboxes[class_specific_mask_initial, :]
            # class_xyhw2_current_cat = xyhw2[class_specific_mask_initial, :]
 
            # 合并类别筛选和分数筛选
            combined_mask = class_specific_mask_initial & (scores >= self.SCORE_THRESHOLD)
            if not np.any(combined_mask):
                continue
                
            current_cat_scores_for_nms = scores[combined_mask]
            current_cat_xyhw2_for_nms = xyhw2[combined_mask, :]
            current_cat_dbboxes_for_output = dbboxes[combined_mask, :] # 用于后续输出坐标
 
            # 3. 对筛选后的结果执行NMS
            # tolist() 是为了兼容OpenCV NMSBoxes有时对numpy数组的处理问题
            indices_after_nms = cv2.dnn.NMSBoxes(current_cat_xyhw2_for_nms.tolist(), 
                                                 current_cat_scores_for_nms.tolist(), 
                                                 self.SCORE_THRESHOLD, # NMSBoxes内部也可能用score_threshold，但我们已经筛选过了
                                                 self.NMS_THRESHOLD)
 
            if len(indices_after_nms) == 0:
                continue
            
            # NMSBoxes 返回的是被选中框在输入列表中的索引
            # 如果 indices_after_nms 是 (N,1) 的形状, 展平它
            if isinstance(indices_after_nms, np.ndarray) and indices_after_nms.ndim > 1:
                 indices_after_nms = indices_after_nms.flatten()
 
            for indic_in_nms_input in indices_after_nms:
                # indic_in_nms_input 是 current_cat_dbboxes_for_output 中的索引
                x1_model, y1_model, x2_model, y2_model = current_cat_dbboxes_for_output[indic_in_nms_input]
                final_score = current_cat_scores_for_nms[indic_in_nms_input]
 
                # 缩放到原始图像尺寸
                # self.x_scale, self.y_scale, self.x_shift, self.y_shift 在 preprocess 中设置
                x1 = int((x1_model - self.x_shift) / self.x_scale)
                y1 = int((y1_model - self.y_shift) / self.y_scale)
                x2 = int((x2_model - self.x_shift) / self.x_scale)
                y2 = int((y2_model - self.y_shift) / self.y_scale)
 
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, self.img_w -1) # 确保在原始图像边界内
                y2 = min(y2, self.img_h -1) # 确保在原始图像边界内
                
                # 确保是有效的边界框
                if x1 < x2 and y1 < y2: 
                     results.append((i, final_score, x1, y1, x2, y2))
                    #  results.append((i, final_score, x1_model, y1_model, x2_model, y2_model))
        
        logger.debug("\033[1;31m" + f"后处理耗时 = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return results
 
 
coco_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
 
rdk_colors = [ # 确保有足够的颜色，或者对 class_id 使用取模运算
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),(49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),(147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)
] * 4 # 如果类别数超过20，重复颜色列表
 
# 修改后的绘制函数，包含 track_id
def draw_track(img, bbox_xyxy, score, class_id, track_id, trend_label="stable") -> None:
    x1, y1, x2, y2 = map(int, bbox_xyxy) # 确保绘制时使用整数坐标
    
    # 按 track_id 着色以便于区分
    color_index = int(track_id) % len(rdk_colors) 
    # 或者按 class_id 着色: color_index = int(class_id) % len(rdk_colors)
    color = rdk_colors[color_index]
 
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    label = f"ID:{track_id} {coco_names[int(class_id)]}: {score:.2f}"
    if trend_label:
        label += f" | {trend_label}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    label_y_pos = y1 - 10 if y1 - 10 > label_height else y1 + label_height + 10
    # 如果可能，确保标签背景在图像边界内
    label_bg_x2 = x1 + label_width
    label_bg_y1 = label_y_pos - label_height
    label_bg_y2 = label_y_pos + (label_height // 4) # 调整以更好地适应文本
 
    cv2.rectangle(
        img, (x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, cv2.FILLED
    )
    cv2.putText(img, label, (x1, label_y_pos - (label_height//2) + (label_height//4) ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
 
 
if __name__ == "__main__":
    main()
