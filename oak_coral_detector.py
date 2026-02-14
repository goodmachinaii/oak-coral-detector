#!/usr/bin/env python3
# TAG: OAK_CORAL_DETECTOR
"""OAK Coral Detector
Detección de objetos + profundidad con Luxonis OAK.
Motor principal: Coral EdgeTPU (SSDLite MobileDet) vía Docker/local.
Fallback automático: YOLOv4-tiny en CPU.
"""
from __future__ import annotations
import os
import time
from pathlib import Path
import json
from urllib import request as urlrequest
from urllib import error as urlerror
import cv2
import depthai as dai
import numpy as np

cv2.setUseOptimized(True)
cv2.setNumThreads(2)

BASE_DIR = Path(os.environ.get('OAK_CORAL_BASE_DIR', Path(__file__).resolve().parent))
STOP_FILE = BASE_DIR / 'STOP_OAK_CORAL_DETECTOR.flag'
LOG_FILE = BASE_DIR / 'oak_coral_detector_runtime.log'

# Fallback CPU YOLO files
MODELS_DIR = Path(os.environ.get('OAK_CORAL_MODELS_DIR', BASE_DIR / 'models'))
YOLO_CFG = MODELS_DIR / 'cpu/yolov4-tiny.cfg'
YOLO_WEIGHTS = MODELS_DIR / 'cpu/yolov4-tiny.weights'
YOLO_NAMES = MODELS_DIR / 'cpu/coco.names'

# Coral files
CORAL_MODEL = MODELS_DIR / 'ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite'
CORAL_LABELS = MODELS_DIR / 'coco_labels.txt'
CORAL_DOCKER_URL = os.environ.get('CORAL_DOCKER_URL', 'http://127.0.0.1:8765')
CORAL_HTTP_TIMEOUT = float(os.environ.get('CORAL_HTTP_TIMEOUT', '0.6'))

RGB_PREVIEW_SIZE = tuple(map(int, os.environ.get('RGB_PREVIEW_SIZE', '640,360').split(',')))
RGB_FPS = int(os.environ.get('RGB_FPS', '15'))
CONF_TH = float(os.environ.get('CONF_THRESHOLD', '0.35'))
NMS_TH = float(os.environ.get('NMS_THRESHOLD', '0.40'))


def log(msg: str) -> None:
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {msg}'
    print(line, flush=True)
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


class CPUDetector:
    def __init__(self):
        with YOLO_NAMES.open('r', encoding='utf-8') as f:
            self.labels = [x.strip() for x in f if x.strip()]
        net = cv2.dnn.readNetFromDarknet(str(YOLO_CFG), str(YOLO_WEIGHTS))
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(416, 416), scale=1/255.0, swapRB=True)

    def detect(self, frame):
        class_ids, scores, boxes = self.model.detect(frame, confThreshold=CONF_TH, nmsThreshold=NMS_TH)
        if len(class_ids):
            class_ids = class_ids.flatten().tolist()
            scores = scores.flatten().tolist()
            boxes = [tuple(map(int, b)) for b in boxes]
        else:
            class_ids, scores, boxes = [], [], []
        return class_ids, scores, boxes, self.labels


class DockerCoralDetector:
    def __init__(self):
        with urlrequest.urlopen(f"{CORAL_DOCKER_URL}/health", timeout=CORAL_HTTP_TIMEOUT) as r:
            if r.status != 200:
                raise RuntimeError('coral docker health failed')
            _ = json.loads(r.read().decode('utf-8') or '{}')

    def detect(self, frame):
        ok, enc = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return [], [], [], {}
        req = urlrequest.Request(
            f"{CORAL_DOCKER_URL}/infer?threshold={CONF_TH}",
            data=enc.tobytes(),
            headers={'Content-Type': 'application/octet-stream'},
            method='POST'
        )
        try:
            with urlrequest.urlopen(req, timeout=CORAL_HTTP_TIMEOUT) as resp:
                data = json.loads(resp.read().decode('utf-8'))
        except (urlerror.URLError, TimeoutError) as e:
            raise RuntimeError(f'coral docker timeout/error: {e}')

        detections = data.get('detections', [])
        in_w, in_h = data.get('input_size', [300, 300])
        h, w = frame.shape[:2]
        sx, sy = w / in_w, h / in_h

        class_ids, scores, boxes = [], [], []
        labels = {}
        for d in detections:
            cid = int(d.get('id', -1))
            labels[cid] = d.get('label', str(cid))
            bb = d.get('bbox', {})
            x1, y1 = float(bb.get('xmin', 0)), float(bb.get('ymin', 0))
            x2, y2 = float(bb.get('xmax', 0)), float(bb.get('ymax', 0))
            x, y = int(x1 * sx), int(y1 * sy)
            bw, bh = int((x2 - x1) * sx), int((y2 - y1) * sy)
            class_ids.append(cid)
            scores.append(float(d.get('score', 0)))
            boxes.append((x, y, bw, bh))
        return class_ids, scores, boxes, labels


class CoralDetector:
    def __init__(self):
        from pycoral.adapters import common, detect
        from pycoral.utils.dataset import read_label_file
        from pycoral.utils.edgetpu import make_interpreter

        self.common = common
        self.detect_mod = detect
        self.labels = read_label_file(str(CORAL_LABELS))
        self.interpreter = make_interpreter(str(CORAL_MODEL))
        self.interpreter.allocate_tensors()
        self.in_w, self.in_h = common.input_size(self.interpreter)

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.in_w, self.in_h))
        self.common.set_input(self.interpreter, resized)
        self.interpreter.invoke()
        objs = self.detect_mod.get_objects(self.interpreter, score_threshold=CONF_TH)

        h, w = frame.shape[:2]
        class_ids, scores, boxes = [], [], []
        sx, sy = w / self.in_w, h / self.in_h
        for o in objs:
            x1, y1, x2, y2 = o.bbox.xmin, o.bbox.ymin, o.bbox.xmax, o.bbox.ymax
            x, y = int(x1 * sx), int(y1 * sy)
            bw, bh = int((x2 - x1) * sx), int((y2 - y1) * sy)
            class_ids.append(int(o.id))
            scores.append(float(o.score))
            boxes.append((x, y, bw, bh))
        return class_ids, scores, boxes, self.labels


def make_detector():
    try:
        det = DockerCoralDetector()
        log('Detector: CORAL Docker activo')
        return det, 'coral-docker'
    except Exception as e:
        log(f'Coral Docker no disponible ({e})')

    try:
        if CORAL_MODEL.exists() and CORAL_LABELS.exists():
            det = CoralDetector()
            log('Detector: CORAL local activo')
            return det, 'coral-local'
        raise RuntimeError('Modelos Coral no encontrados')
    except Exception as e:
        log(f'Detector Coral local no disponible ({e}); usando fallback CPU YOLO')
        return CPUDetector(), 'cpu'


def build_pipeline():
    p = dai.Pipeline()
    cam_rgb = p.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setPreviewSize(*RGB_PREVIEW_SIZE)
    cam_rgb.setFps(RGB_FPS)

    xout_rgb = p.create(dai.node.XLinkOut)
    xout_rgb.setStreamName('rgb')
    xout_rgb.input.setBlocking(False)
    xout_rgb.input.setQueueSize(1)
    cam_rgb.preview.link(xout_rgb.input)

    mono_l = p.create(dai.node.MonoCamera)
    mono_r = p.create(dai.node.MonoCamera)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_l.setFps(RGB_FPS)
    mono_r.setFps(RGB_FPS)

    stereo = p.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    xout_depth = p.create(dai.node.XLinkOut)
    xout_depth.setStreamName('depth')
    xout_depth.input.setBlocking(False)
    xout_depth.input.setQueueSize(1)
    stereo.depth.link(xout_depth.input)
    return p


def run_once(detector, mode):
    clicked = {'stop': False, 'exit': False}
    btn_stop = [0, 0, 0, 0]
    btn_exit = [0, 0, 0, 0]

    def on_mouse(event, x, y, flags, param):
        _ = (flags, param)
        if event == cv2.EVENT_LBUTTONDOWN:
            if btn_stop[0] <= x <= btn_stop[2] and btn_stop[1] <= y <= btn_stop[3]:
                clicked['stop'] = True
            if btn_exit[0] <= x <= btn_exit[2] and btn_exit[1] <= y <= btn_exit[3]:
                clicked['exit'] = True

    pipeline = build_pipeline()
    try:
        device_cm = dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH)
    except TypeError:
        device_cm = dai.Device(pipeline, usb2Mode=True)

    infer_ms = 0.0
    try:
        with device_cm as device:
            q_rgb = device.getOutputQueue(name='rgb', maxSize=1, blocking=False)
            q_depth = device.getOutputQueue(name='depth', maxSize=1, blocking=False)

            cv2.namedWindow('OAK Coral Detector - RGB+Depth')
            cv2.setMouseCallback('OAK Coral Detector - RGB+Depth', on_mouse)
            cv2.namedWindow('OAK Coral Detector - Depth')

            last_frame, last_depth = None, None
            last_frame_ts = time.monotonic()
            frame_idx, detect_every_n = 0, 2
            cached = ([], [], [], [])
            start_t = time.monotonic()
            fps_counter = 0
            fps = 0.0

            while True:
                if STOP_FILE.exists() or clicked['stop']:
                    return 'stop'
                if clicked['exit']:
                    return 'exit'

                mr = q_rgb.tryGet()
                md = q_depth.tryGet()
                if mr is not None:
                    last_frame = mr.getCvFrame()
                    last_frame_ts = time.monotonic()
                    fps_counter += 1
                if md is not None:
                    last_depth = md.getFrame()

                now = time.monotonic()
                if now - start_t >= 1.0:
                    fps = fps_counter / (now - start_t)
                    fps_counter = 0
                    start_t = now
                if now - last_frame_ts > 6.0:
                    raise RuntimeError('No llegan frames RGB >6s (reinicio automático)')
                if last_frame is None:
                    if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                        return 'exit'
                    continue

                frame = last_frame.copy()
                h, w = frame.shape[:2]
                frame_idx += 1
                if frame_idx % detect_every_n == 0:
                    t0 = time.monotonic()
                    cached = detector.detect(frame)
                    infer_ms = (time.monotonic() - t0) * 1000
                class_ids, scores, boxes, labels = cached

                for cid, score, box in zip(class_ids, scores, boxes):
                    x, y, bw, bh = [int(v) for v in box]
                    label = labels.get(cid, str(cid)) if isinstance(labels, dict) else (labels[cid] if cid < len(labels) else str(cid))
                    z_text = ''
                    if last_depth is not None and last_depth.size > 0:
                        dh, dw = last_depth.shape[:2]
                        rx1 = int((x + bw * 0.35) * dw / w)
                        ry1 = int((y + bh * 0.35) * dh / h)
                        rx2 = int((x + bw * 0.65) * dw / w)
                        ry2 = int((y + bh * 0.65) * dh / h)
                        rx1, ry1 = max(0, rx1), max(0, ry1)
                        rx2, ry2 = min(dw - 1, rx2), min(dh - 1, ry2)
                        if ry2 >= ry1 and rx2 >= rx1:
                            roi = last_depth[ry1:ry2 + 1, rx1:rx2 + 1]
                            valid = roi[roi > 0]
                            if valid.size > 0:
                                z_mm = int(np.median(valid))
                                z_text = f' | Z:{z_mm/10:.1f}cm'
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {score*100:.0f}%{z_text}', (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                margin = 12
                btn_w = 118
                btn_h = 32
                gap = 10
                y1 = h - margin - btn_h
                y2 = h - margin
                btn_stop[:] = [margin, y1, margin + btn_w, y2]
                btn_exit[:] = [margin + btn_w + gap, y1, margin + btn_w + gap + btn_w, y2]
                cv2.rectangle(frame, (btn_stop[0], btn_stop[1]), (btn_stop[2], btn_stop[3]), (0, 140, 255), -1)
                cv2.putText(frame, 'STOP', (btn_stop[0] + 24, btn_stop[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
                cv2.rectangle(frame, (btn_exit[0], btn_exit[1]), (btn_exit[2], btn_exit[3]), (0, 0, 255), -1)
                cv2.putText(frame, 'EXIT', (btn_exit[0] + 28, btn_exit[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                if mode.startswith('coral'):
                    inf_text = 'INFERENCE: CORAL'
                    inf_color = (0, 220, 0)
                    model_name = 'SSDLite MobileDet'
                else:
                    inf_text = 'INFERENCE: CPU FALLBACK'
                    inf_color = (0, 0, 255)
                    model_name = 'YOLOv4-tiny'

                cv2.putText(frame, inf_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, inf_color, 2)
                cv2.putText(frame, f'detections: {len(boxes)} | mode: {mode}', (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
                cv2.putText(frame, f'FPS host: {fps:.1f} | infer: {infer_ms:.0f}ms', (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(frame, f'model: {model_name}', (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                cv2.imshow('OAK Coral Detector - RGB+Depth', frame)
                if last_depth is not None:
                    dv = cv2.normalize(last_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imshow('OAK Coral Detector - Depth', cv2.applyColorMap(dv, cv2.COLORMAP_TURBO))

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    return 'exit'
    finally:
        cv2.destroyAllWindows()


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    STOP_FILE.unlink(missing_ok=True)
    detector, mode = make_detector()
    log(f'Iniciando OAK Coral Detector (mode={mode})')
    while True:
        if STOP_FILE.exists():
            break
        try:
            a = run_once(detector, mode)
            if a in ('stop', 'exit'):
                break
        except Exception as e:
            log(f'Reinicio de pipeline por excepción: {e}')
            if STOP_FILE.exists():
                break
            if mode.startswith('coral'):
                log('Coral inestable: cambio automático a CPU fallback')
                detector, mode = CPUDetector(), 'cpu'
            time.sleep(1)
    log('OAK Coral Detector detenido')


if __name__ == '__main__':
    main()
