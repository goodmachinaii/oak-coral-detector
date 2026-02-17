#!/usr/bin/env python3
# TAG: OAK_CORAL_DETECTOR
from __future__ import annotations
import time
import cv2
import depthai as dai

from oak_vision.config import load_settings, log
from oak_vision.capture import build_pipeline
from oak_vision.inference import make_detector, CPUDetector
from oak_vision.depth import depth_cm_for_box, depth_text
from oak_vision.display import gui_enabled, draw_buttons, draw_hud, show_frames, idle_sleep
from oak_vision.storage import DetectionStorage
from oak_vision.api import ApiServer
from oak_vision.hardening import ExponentialBackoff
from oak_vision.event_tracker import EventTracker

cv2.setUseOptimized(True)
cv2.setNumThreads(2)


def run_once(settings, detector, mode, storage: DetectionStorage, state: dict, tracker: EventTracker, track_to_event: dict):
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
                    state['oak_connected'] = True
                if md is not None:
                    last_depth = md.getFrame()

                now = time.monotonic()
                if now - start_t >= 1.0:
                    fps = fps_counter / (now - start_t)
                    fps_counter = 0
                    start_t = now
                    state['fps'] = fps

                if now - last_frame_ts > settings.oak_frame_timeout:
                    storage.log_system_event('oak_timeout', 'oak', 'error', f'No RGB > {settings.oak_frame_timeout}s')
                    raise RuntimeError(f'No llegan frames RGB >{settings.oak_frame_timeout}s (reinicio automático)')

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
                        state['infer_ms'] = infer_ms
                        state['mode'] = mode
                        coral_timeout_streak = 0
                    except Exception as e:
                        if mode.startswith('coral') and 'timeout' in str(e).lower():
                            coral_timeout_streak += 1
                            storage.log_system_event('coral_timeout', 'coral', 'warning', str(e))
                            log(settings, f'Coral timeout {coral_timeout_streak}/{settings.coral_max_timeouts}: {e}')
                            if coral_timeout_streak >= settings.coral_max_timeouts:
                                raise RuntimeError(f'coral timeout consecutivo ({coral_timeout_streak})')
                            continue
                        raise

                class_ids, scores, boxes, labels = cached
                raw_rows = []
                for cid, score, box in zip(class_ids, scores, boxes):
                    x, y, bw, bh = [int(v) for v in box]
                    label = labels.get(cid, str(cid)) if isinstance(labels, dict) else (labels[cid] if cid < len(labels) else str(cid))
                    d_cm = depth_cm_for_box(last_depth, frame.shape, box)
                    z_text = depth_text(d_cm)
                    raw_rows.append({'label': label, 'confidence': float(score), 'bbox': (x, y, bw, bh), 'depth_cm': d_cm})
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {score*100:.0f}%{z_text}', (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                filtered = [d for d in raw_rows if d['confidence'] >= settings.min_score]
                enriched = tracker.update(filtered)

                for det in enriched:
                    tid = det['tracker_track_id']
                    if det['is_new_event'] and det['confirmed']:
                        db_event_id = storage.create_event(tid, det['label'], det['bbox'], det['confidence'], det.get('depth_cm'))
                        track_to_event[tid] = db_event_id
                        det['event_id'] = db_event_id
                    elif det['confirmed'] and tid in track_to_event:
                        db_event_id = track_to_event[tid]
                        storage.update_event(db_event_id, det['bbox'], det['confidence'], det.get('depth_cm'))
                        det['event_id'] = db_event_id
                    else:
                        det['event_id'] = None

                storage.store(enriched, mode=mode, infer_ms=infer_ms)

                for ended in tracker.get_ended_tracks():
                    tid = ended['track_id']
                    if tid in track_to_event:
                        storage.close_event(track_to_event.pop(tid), ended['duration_sec'])

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

    started = time.monotonic()
    detector, mode = make_detector(settings)
    state = {'mode': mode, 'fps': 0.0, 'infer_ms': 0.0, 'running': True, 'oak_connected': True, 'uptime_sec': 0}

    storage = DetectionStorage(
        db_path=settings.db_path,
        retention_days=settings.db_retention_days,
        prune_every_sec=settings.db_prune_every_sec,
    )
    storage.log_system_event('startup', 'orchestrator', 'info', 'startup')

    tracker = EventTracker(
        score_threshold=settings.score_threshold,
        distance_threshold=settings.tracker_distance_threshold,
        initialization_delay=settings.tracker_initialization_delay,
        hit_counter_max=settings.tracker_hit_counter_max,
    )
    track_to_event: dict[int, int] = {}

    api = ApiServer(storage, state, host=settings.api_host, port=settings.api_port)
    ok, backend = api.start()
    if ok:
        log(settings, f'API local activa en http://{settings.api_host}:{settings.api_port} ({backend})')
    else:
        log(settings, f'API local no iniciada ({backend})')

    log(settings, f'Iniciando OAK Coral Detector (mode={mode})')
    backoff = ExponentialBackoff(initial=1.0, maximum=30.0)

    try:
        while True:
            state['uptime_sec'] = int(time.monotonic() - started)
            if settings.stop_file.exists():
                break
            try:
                action = run_once(settings, detector, mode, storage, state, tracker, track_to_event)
                backoff.reset()
                if action in ('stop', 'exit'):
                    break
            except Exception as e:
                storage.log_system_event('pipeline_restart', 'orchestrator', 'error', str(e))
                log(settings, f'Reinicio de pipeline por excepción: {e}')
                if settings.stop_file.exists():
                    break
                if mode.startswith('coral'):
                    storage.log_system_event('coral_fallback', 'coral', 'warning', 'fallback coral->cpu')
                    log(settings, 'Coral inestable: cambio automático a CPU fallback')
                    detector, mode = CPUDetector(settings), 'cpu'
                    state['mode'] = mode
                state['oak_connected'] = False
                backoff.wait()
                storage.log_system_event('oak_reconnect', 'oak', 'info', 'retry after backoff')
    finally:
        state['running'] = False
        api.stop()
        storage.log_system_event('shutdown', 'orchestrator', 'info', 'shutdown')
        storage.close()

    log(settings, 'OAK Coral Detector detenido')


if __name__ == '__main__':
    main()
