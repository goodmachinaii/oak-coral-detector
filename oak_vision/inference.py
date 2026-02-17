from __future__ import annotations
import json
from urllib import request as urlrequest
from urllib import error as urlerror
import cv2
from .config import Settings, log


class CPUDetector:
    def __init__(self, settings: Settings):
        self.settings = settings
        with settings.yolo_names.open('r', encoding='utf-8') as f:
            self.labels = [x.strip() for x in f if x.strip()]
        net = cv2.dnn.readNetFromDarknet(str(settings.yolo_cfg), str(settings.yolo_weights))
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(416, 416), scale=1/255.0, swapRB=True)

    def detect(self, frame):
        class_ids, scores, boxes = self.model.detect(
            frame,
            confThreshold=self.settings.conf_th,
            nmsThreshold=self.settings.nms_th,
        )
        if len(class_ids):
            class_ids = class_ids.flatten().tolist()
            scores = scores.flatten().tolist()
            boxes = [tuple(map(int, b)) for b in boxes]
        else:
            class_ids, scores, boxes = [], [], []
        return class_ids, scores, boxes, self.labels


class DockerCoralDetector:
    def __init__(self, settings: Settings):
        self.settings = settings
        with urlrequest.urlopen(f"{settings.coral_docker_url}/health", timeout=settings.coral_http_timeout) as r:
            if r.status != 200:
                raise RuntimeError('coral docker health failed')
            _ = json.loads(r.read().decode('utf-8') or '{}')

    def detect(self, frame):
        ok, enc = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return [], [], [], {}
        req = urlrequest.Request(
            f"{self.settings.coral_docker_url}/infer?threshold={self.settings.conf_th}",
            data=enc.tobytes(),
            headers={'Content-Type': 'application/octet-stream'},
            method='POST',
        )
        try:
            with urlrequest.urlopen(req, timeout=self.settings.coral_http_timeout) as resp:
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
    def __init__(self, settings: Settings):
        from pycoral.adapters import common, detect
        from pycoral.utils.dataset import read_label_file
        from pycoral.utils.edgetpu import make_interpreter

        self.settings = settings
        self.common = common
        self.detect_mod = detect
        self.labels = read_label_file(str(settings.coral_labels))
        self.interpreter = make_interpreter(str(settings.coral_model))
        self.interpreter.allocate_tensors()
        self.in_w, self.in_h = common.input_size(self.interpreter)

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.in_w, self.in_h))
        self.common.set_input(self.interpreter, resized)
        self.interpreter.invoke()
        objs = self.detect_mod.get_objects(self.interpreter, score_threshold=self.settings.conf_th)

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


def make_detector(settings: Settings):
    try:
        det = DockerCoralDetector(settings)
        log(settings, 'Detector: CORAL Docker activo')
        return det, 'coral-docker'
    except Exception as e:
        log(settings, f'Coral Docker no disponible ({e})')

    try:
        if settings.coral_model.exists() and settings.coral_labels.exists():
            det = CoralDetector(settings)
            log(settings, 'Detector: CORAL local activo')
            return det, 'coral-local'
        raise RuntimeError('Modelos Coral no encontrados')
    except Exception as e:
        log(settings, f'Detector Coral local no disponible ({e}); usando fallback CPU YOLO')
        return CPUDetector(settings), 'cpu'
