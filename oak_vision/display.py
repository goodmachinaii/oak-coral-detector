from __future__ import annotations
import os
import time
import cv2
import numpy as np


def gui_enabled(headless_mode: str) -> bool:
    has_display = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
    return has_display if headless_mode == 'auto' else headless_mode not in ('1', 'true', 'yes', 'on')


def draw_hud(frame, mode: str, infer_ms: float, fps: float, boxes_count: int):
    if mode.startswith('coral'):
        inf_text = 'INFERENCE: CORAL'
        inf_color = (0, 220, 0)
        model_name = 'SSDLite MobileDet'
    else:
        inf_text = 'INFERENCE: CPU FALLBACK'
        inf_color = (0, 0, 255)
        model_name = 'YOLOv4-tiny'

    cv2.putText(frame, inf_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, inf_color, 2)
    cv2.putText(frame, f'detections: {boxes_count} | mode: {mode}', (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
    cv2.putText(frame, f'FPS host: {fps:.1f} | infer: {infer_ms:.0f}ms', (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f'model: {model_name}', (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def draw_buttons(frame, btn_stop: list[int], btn_exit: list[int]):
    h = frame.shape[0]
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


def show_frames(gui_on: bool, frame, depth_frame):
    if gui_on:
        cv2.imshow('OAK Coral Detector - RGB+Depth', frame)
        if depth_frame is not None:
            dv = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow('OAK Coral Detector - Depth', cv2.applyColorMap(dv, cv2.COLORMAP_TURBO))
            

def idle_sleep(gui_on: bool):
    if gui_on:
        return cv2.waitKey(1) & 0xFF
    time.sleep(0.005)
    return None
