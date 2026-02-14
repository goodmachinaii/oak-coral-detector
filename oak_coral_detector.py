#!/usr/bin/env python3
# TAG: OAK_CORAL_DETECTOR
"""OAK Coral Detector orchestrator.

Fase 1 (modular): mantiene comportamiento actual importando módulos
(en vez de concentrar toda la lógica en un solo archivo).
"""
from __future__ import annotations
import time
import cv2
import depthai as dai

from oak_vision.config import load_settings, log
from oak_vision.capture import build_pipeline
from oak_vision.inference import make_detector, CPUDetector
from oak_vision.depth import depth_text_for_box
from oak_vision.display import gui_enabled, draw_buttons, draw_hud, show_frames, idle_sleep
from oak_vision.storage import DetectionStorage

cv2.setUseOptimized(True)
cv2.setNumThreads(2)


def run_once(settings, detector, mode, storage: DetectionStorage):
    clicked = {'stop': False, 'exit': False}
    btn_stop = [0, 0, 0, 0]
    btn_exit = [0, 0, 0, 0]
    gui_on = gui_enabled(settings.headless)

    def on_mouse(event, x, y, flags, param):
        _ = (flags, param)
        if event == cv2.EVENT_LBUTTONDOWN:
            if btn_stop[0] <= x <= btn_stop[2] and btn_stop[1] <= y <= btn_stop[3]:
                clicked['stop'] = True
            if btn_exit[0] <= x <= btn_exit[2] and btn_exit[1] <= y <= btn_exit[3]:
                clicked['exit'] = True

    pipeline = build_pipeline(settings)
    try:
        device_cm = dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH)
    except TypeError:
        device_cm = dai.Device(pipeline, usb2Mode=True)

    infer_ms = 0.0
    try:
        with device_cm as device:
            q_rgb = device.getOutputQueue(name='rgb', maxSize=1, blocking=False)
            q_depth = device.getOutputQueue(name='depth', maxSize=1, blocking=False)

            if gui_on:
                cv2.namedWindow('OAK Coral Detector - RGB+Depth')
                cv2.setMouseCallback('OAK Coral Detector - RGB+Depth', on_mouse)
                cv2.namedWindow('OAK Coral Detector - Depth')
            else:
                log(settings, 'Modo headless activo: sin ventanas OpenCV (usar stop_coral_stack.sh para detener).')

            last_frame, last_depth = None, None
            last_frame_ts = time.monotonic()
            frame_idx, detect_every_n = 0, 2
            cached = ([], [], [], [])
            start_t = time.monotonic()
            fps_counter = 0
            fps = 0.0
            coral_timeout_streak = 0

            while True:
                if settings.stop_file.exists() or clicked['stop']:
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
                    key = idle_sleep(gui_on)
                    if key in (27, ord('q')):
                        return 'exit'
                    continue

                frame = last_frame.copy()
                frame_idx += 1
                if frame_idx % detect_every_n == 0:
                    t0 = time.monotonic()
                    try:
                        cached = detector.detect(frame)
                        infer_ms = (time.monotonic() - t0) * 1000
                        coral_timeout_streak = 0
                    except Exception as e:
                        if mode.startswith('coral') and 'timeout' in str(e).lower():
                            coral_timeout_streak += 1
                            log(settings, f'Coral timeout {coral_timeout_streak}/{settings.coral_max_timeouts}: {e}')
                            if coral_timeout_streak >= settings.coral_max_timeouts:
                                raise RuntimeError(f'coral timeout consecutivo ({coral_timeout_streak})')
                            continue
                        raise

                class_ids, scores, boxes, labels = cached
                stored_rows = []
                for cid, score, box in zip(class_ids, scores, boxes):
                    x, y, bw, bh = [int(v) for v in box]
                    label = labels.get(cid, str(cid)) if isinstance(labels, dict) else (labels[cid] if cid < len(labels) else str(cid))
                    z_text = depth_text_for_box(last_depth, frame.shape, box)
                    depth_cm = None
                    if 'Z:' in z_text:
                        try:
                            depth_cm = float(z_text.split('Z:')[1].replace('cm', '').strip())
                        except Exception:
                            depth_cm = None
                    stored_rows.append({
                        'label': label,
                        'confidence': float(score),
                        'bbox': (x, y, bw, bh),
                        'depth_cm': depth_cm,
                    })
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {score*100:.0f}%{z_text}', (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                storage.store(stored_rows, mode=mode, infer_ms=infer_ms)

                draw_buttons(frame, btn_stop, btn_exit)
                draw_hud(frame, mode, infer_ms, fps, len(boxes))
                show_frames(gui_on, frame, last_depth)

                key = idle_sleep(gui_on)
                if key in (27, ord('q')):
                    return 'exit'
    finally:
        cv2.destroyAllWindows()


def main():
    settings = load_settings()
    settings.base_dir.mkdir(parents=True, exist_ok=True)
    settings.stop_file.unlink(missing_ok=True)

    detector, mode = make_detector(settings)
    storage = DetectionStorage(
        db_path=settings.db_path,
        retention_days=settings.db_retention_days,
        prune_every_sec=settings.db_prune_every_sec,
    )
    log(settings, f'Iniciando OAK Coral Detector (mode={mode})')

    try:
        while True:
            if settings.stop_file.exists():
                break
            try:
                action = run_once(settings, detector, mode, storage)
                if action in ('stop', 'exit'):
                    break
            except Exception as e:
                log(settings, f'Reinicio de pipeline por excepción: {e}')
                if settings.stop_file.exists():
                    break
                if mode.startswith('coral'):
                    log(settings, 'Coral inestable: cambio automático a CPU fallback')
                    detector, mode = CPUDetector(settings), 'cpu'
                time.sleep(1)
    finally:
        storage.close()

    log(settings, 'OAK Coral Detector detenido')


if __name__ == '__main__':
    main()
