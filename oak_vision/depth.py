from __future__ import annotations
import numpy as np


def depth_cm_for_box(depth_frame, frame_shape, box):
    if depth_frame is None or depth_frame.size == 0:
        return None

    h, w = frame_shape[:2]
    x, y, bw, bh = [int(v) for v in box]
    dh, dw = depth_frame.shape[:2]
    rx1 = int((x + bw * 0.35) * dw / w)
    ry1 = int((y + bh * 0.35) * dh / h)
    rx2 = int((x + bw * 0.65) * dw / w)
    ry2 = int((y + bh * 0.65) * dh / h)
    rx1, ry1 = max(0, rx1), max(0, ry1)
    rx2, ry2 = min(dw - 1, rx2), min(dh - 1, ry2)
    if ry2 < ry1 or rx2 < rx1:
        return None

    roi = depth_frame[ry1:ry2 + 1, rx1:rx2 + 1]
    valid = roi[roi > 0]
    if valid.size == 0:
        return None

    z_mm = int(np.median(valid))
    return round(z_mm / 10.0, 1)


def depth_text(depth_cm):
    if depth_cm is None:
        return ''
    return f' | Z:{depth_cm:.1f}cm'
