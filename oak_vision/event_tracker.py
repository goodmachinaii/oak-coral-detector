from __future__ import annotations
from dataclasses import dataclass, field
import time
import numpy as np

try:
    from norfair import Detection, Tracker
    NORFAIR_AVAILABLE = True
except Exception:
    NORFAIR_AVAILABLE = False


@dataclass
class TrackMetadata:
    label: str
    score_history: list[float] = field(default_factory=list)
    confirmed: bool = False
    event_created: bool = False
    first_seen: float = 0.0
    depth_history: list[float] = field(default_factory=list)

    @property
    def computed_score(self) -> float:
        padded = self.score_history + [0.0] * max(0, 3 - len(self.score_history))
        return sorted(padded)[len(padded) // 2]

    @property
    def avg_depth_cm(self):
        valid = [d for d in self.depth_history if d is not None]
        return sum(valid) / len(valid) if valid else None

    @property
    def min_depth_cm(self):
        valid = [d for d in self.depth_history if d is not None]
        return min(valid) if valid else None


def _iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def iou_distance(detection, tracked_object) -> float:
    det_box = detection.points
    trk_box = tracked_object.estimate
    ix1 = max(det_box[0][0], trk_box[0][0])
    iy1 = max(det_box[0][1], trk_box[0][1])
    ix2 = min(det_box[1][0], trk_box[1][0])
    iy2 = min(det_box[1][1], trk_box[1][1])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_det = (det_box[1][0] - det_box[0][0]) * (det_box[1][1] - det_box[0][1])
    area_trk = (trk_box[1][0] - trk_box[0][0]) * (trk_box[1][1] - trk_box[0][1])
    union = area_det + area_trk - inter
    iou = inter / union if union > 0 else 0.0
    return 1.0 - iou


class EventTracker:
    def __init__(
        self,
        score_threshold: float = 0.7,
        distance_threshold: float = 0.7,
        initialization_delay: int = 3,
        hit_counter_max: int = 15,
    ):
        self.score_threshold = score_threshold
        self.distance_threshold = distance_threshold
        self.initialization_delay = initialization_delay
        self.hit_counter_max = hit_counter_max
        self._meta: dict[int, TrackMetadata] = {}

        self._use_norfair = NORFAIR_AVAILABLE
        self._next_id = 1
        self._tracks = {}  # fallback: id -> {'bbox','misses','hits','label','last_det'}

        if self._use_norfair:
            self.tracker = Tracker(
                distance_function=iou_distance,
                distance_threshold=distance_threshold,
                initialization_delay=initialization_delay,
                hit_counter_max=hit_counter_max,
            )

    def _append_meta(self, track_id: int, det: dict, now: float):
        if track_id not in self._meta:
            self._meta[track_id] = TrackMetadata(label=det['label'], first_seen=now)
        meta = self._meta[track_id]
        meta.score_history.append(float(det['confidence']))
        if det.get('depth_cm') is not None:
            meta.depth_history.append(float(det['depth_cm']))
        if not meta.confirmed and meta.computed_score >= self.score_threshold:
            meta.confirmed = True
        is_new = meta.confirmed and not meta.event_created
        if is_new:
            meta.event_created = True
        return meta, is_new

    def _update_norfair(self, detections: list[dict]):
        now = time.monotonic()
        norfair_dets = []
        det_map = {}
        for det in detections:
            x, y, w, h = det['bbox']
            points = np.array([[x, y], [x + w, y + h]], dtype=np.float32)
            nd = Detection(points=points, scores=np.array([det['confidence'], det['confidence']]), label=det['label'])
            norfair_dets.append(nd)
            det_map[id(nd)] = det

        tracked_objects = self.tracker.update(detections=norfair_dets)
        out = []
        for obj in tracked_objects:
            det = det_map.get(id(obj.last_detection)) if obj.last_detection is not None else None
            if det is None:
                continue
            track_id = obj.id
            meta, is_new = self._append_meta(track_id, det, now)
            est = obj.estimate
            bbox = (int(est[0][0]), int(est[0][1]), int(est[1][0] - est[0][0]), int(est[1][1] - est[0][1]))
            out.append({**det, 'bbox': bbox, 'tracker_track_id': track_id, 'is_new_event': is_new, 'confirmed': meta.confirmed})
        return out

    def _update_fallback(self, detections: list[dict]):
        now = time.monotonic()
        matched = set()
        out = []

        for tid, tr in list(self._tracks.items()):
            best_iou, best_idx = 0.0, None
            for i, det in enumerate(detections):
                if i in matched or det['label'] != tr['label']:
                    continue
                iou = _iou_xywh(tr['bbox'], det['bbox'])
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            if best_idx is not None and best_iou >= (1.0 - self.distance_threshold):
                det = detections[best_idx]
                matched.add(best_idx)
                tr['bbox'] = det['bbox']
                tr['misses'] = 0
                tr['hits'] += 1
                meta, is_new = self._append_meta(tid, det, now)
                out.append({**det, 'tracker_track_id': tid, 'is_new_event': is_new, 'confirmed': meta.confirmed})
            else:
                tr['misses'] += 1

        for i, det in enumerate(detections):
            if i in matched:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {'bbox': det['bbox'], 'misses': 0, 'hits': 1, 'label': det['label']}
            if self.initialization_delay <= 1:
                meta, is_new = self._append_meta(tid, det, now)
                out.append({**det, 'tracker_track_id': tid, 'is_new_event': is_new, 'confirmed': meta.confirmed})

        return out

    def update(self, detections: list[dict]) -> list[dict]:
        if self._use_norfair:
            return self._update_norfair(detections)
        return self._update_fallback(detections)

    def get_ended_tracks(self) -> list[dict]:
        if self._use_norfair:
            active_ids = {obj.id for obj in self.tracker.tracked_objects}
        else:
            active_ids = {tid for tid, tr in self._tracks.items() if tr['misses'] < self.hit_counter_max}
            for tid in list(self._tracks.keys()):
                if self._tracks[tid]['misses'] >= self.hit_counter_max:
                    del self._tracks[tid]

        ended = []
        dead = []
        for tid, meta in self._meta.items():
            if tid not in active_ids and meta.event_created:
                now = time.monotonic()
                ended.append({
                    'track_id': tid,
                    'label': meta.label,
                    'duration_sec': now - meta.first_seen,
                    'frame_count': len(meta.score_history),
                    'max_confidence': max(meta.score_history) if meta.score_history else 0.0,
                    'avg_depth_cm': meta.avg_depth_cm,
                    'min_depth_cm': meta.min_depth_cm,
                })
                dead.append(tid)
            elif tid not in active_ids:
                dead.append(tid)

        for tid in dead:
            self._meta.pop(tid, None)
        return ended

    def get_active_summary(self) -> dict[str, int]:
        if self._use_norfair:
            active_ids = {obj.id for obj in self.tracker.tracked_objects}
        else:
            active_ids = {tid for tid, tr in self._tracks.items() if tr['misses'] < self.hit_counter_max}
        out: dict[str, int] = {}
        for tid, meta in self._meta.items():
            if tid in active_ids and meta.confirmed:
                out[meta.label] = out.get(meta.label, 0) + 1
        return out
