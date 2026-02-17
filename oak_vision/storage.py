from __future__ import annotations
import os
import sqlite3
import time
import threading
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS detections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT    NOT NULL,
    mode       TEXT    NOT NULL,
    infer_ms   REAL,
    label      TEXT    NOT NULL,
    confidence REAL    NOT NULL,
    bbox_x     INTEGER,
    bbox_y     INTEGER,
    bbox_w     INTEGER,
    bbox_h     INTEGER,
    depth_cm   REAL,
    event_id   INTEGER REFERENCES events(event_id)
);
CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_label     ON detections(label);
CREATE INDEX IF NOT EXISTS idx_depth     ON detections(depth_cm);

CREATE TABLE IF NOT EXISTS events (
    event_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tracker_track_id  INTEGER,
    label             TEXT NOT NULL,
    first_seen        TEXT NOT NULL,
    last_seen         TEXT NOT NULL,
    duration_sec      REAL,
    frame_count       INTEGER DEFAULT 1,
    max_confidence    REAL,
    min_depth_cm      REAL,
    avg_depth_cm      REAL,
    last_bbox_x       INTEGER,
    last_bbox_y       INTEGER,
    last_bbox_w       INTEGER,
    last_bbox_h       INTEGER,
    status            TEXT DEFAULT 'active'
);
CREATE INDEX IF NOT EXISTS idx_events_label  ON events(label);
CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
CREATE INDEX IF NOT EXISTS idx_events_first  ON events(first_seen);

CREATE TABLE IF NOT EXISTS system_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT NOT NULL,
    event_type TEXT NOT NULL,
    component  TEXT,
    severity   TEXT,
    detail     TEXT
);
CREATE INDEX IF NOT EXISTS idx_sysev_type ON system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_sysev_ts   ON system_events(timestamp);
"""


class DetectionStorage:
    def __init__(self, db_path: Path, retention_days: int = 7, prune_every_sec: int = 300):
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self.prune_every_sec = prune_every_sec
        self._last_prune = 0.0

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode = WAL;')
        self.conn.execute('PRAGMA synchronous = NORMAL;')
        self.conn.execute('PRAGMA cache_size = -2000;')
        self.conn.execute('PRAGMA temp_store = MEMORY;')
        self.conn.executescript(SCHEMA_SQL)
        self._ensure_column('detections', 'event_id', 'INTEGER')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_event_id ON detections(event_id)')
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, col_type: str):
        with self._lock:
            cols = {r[1] for r in self.conn.execute(f'PRAGMA table_info({table})').fetchall()}
            if column not in cols:
                self.conn.execute(f'ALTER TABLE {table} ADD COLUMN {column} {col_type}')

    def close(self):
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                pass

    def _query(self, sql: str, params=()):
        with self._lock:
            self.conn.row_factory = sqlite3.Row
            cur = self.conn.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def store(self, detections, mode: str, infer_ms: float):
        now_iso = datetime.now(timezone.utc).isoformat(timespec='seconds')
        rows = []
        for d in detections:
            rows.append(
                (
                    now_iso,
                    mode,
                    infer_ms,
                    d['label'],
                    d['confidence'],
                    d['bbox'][0],
                    d['bbox'][1],
                    d['bbox'][2],
                    d['bbox'][3],
                    d.get('depth_cm'),
                    d.get('event_id'),
                )
            )
        if rows:
            with self._lock:
                with self.conn:
                    self.conn.executemany(
                    """
                    INSERT INTO detections (
                        timestamp, mode, infer_ms, label, confidence,
                        bbox_x, bbox_y, bbox_w, bbox_h, depth_cm, event_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        self._maybe_prune()

    def create_event(self, tracker_track_id: int, label: str, bbox: tuple, confidence: float, depth_cm):
        now_iso = datetime.now(timezone.utc).isoformat(timespec='seconds')
        with self._lock:
            with self.conn:
                cur = self.conn.execute(
                """
                INSERT INTO events (
                    tracker_track_id, label, first_seen, last_seen,
                    frame_count, max_confidence, min_depth_cm, avg_depth_cm,
                    last_bbox_x, last_bbox_y, last_bbox_w, last_bbox_h, status
                ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, 'active')
                """,
                (tracker_track_id, label, now_iso, now_iso, confidence, depth_cm, depth_cm, *bbox),
            )
            return int(cur.lastrowid)

    def update_event(self, event_id: int, bbox: tuple, confidence: float, depth_cm):
        now_iso = datetime.now(timezone.utc).isoformat(timespec='seconds')
        with self._lock:
            with self.conn:
                if depth_cm is not None:
                    self.conn.execute(
                        """
                        UPDATE events
                        SET last_seen=?, frame_count=frame_count+1,
                            max_confidence=MAX(max_confidence, ?),
                            min_depth_cm=CASE WHEN min_depth_cm IS NULL THEN ? ELSE MIN(min_depth_cm, ?) END,
                            avg_depth_cm=CASE
                                WHEN avg_depth_cm IS NULL THEN ?
                                ELSE ((avg_depth_cm * (frame_count - 1)) + ?) / frame_count
                            END,
                            last_bbox_x=?, last_bbox_y=?, last_bbox_w=?, last_bbox_h=?
                        WHERE event_id=?
                        """,
                        (now_iso, confidence, depth_cm, depth_cm, depth_cm, depth_cm, *bbox, event_id),
                    )
                else:
                    self.conn.execute(
                        """
                        UPDATE events
                        SET last_seen=?, frame_count=frame_count+1,
                            max_confidence=MAX(max_confidence, ?),
                            last_bbox_x=?, last_bbox_y=?, last_bbox_w=?, last_bbox_h=?
                        WHERE event_id=?
                        """,
                        (now_iso, confidence, *bbox, event_id),
                    )

    def close_event(self, event_id: int, duration_sec: float):
        with self._lock:
            with self.conn:
                self.conn.execute("UPDATE events SET status='ended', duration_sec=? WHERE event_id=?", (duration_sec, event_id))

    def log_system_event(self, event_type: str, component: str, severity: str, detail: str = ''):
        now_iso = datetime.now(timezone.utc).isoformat(timespec='seconds')
        with self._lock:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO system_events (timestamp, event_type, component, severity, detail) VALUES (?, ?, ?, ?, ?)",
                    (now_iso, event_type, component, severity, detail),
                )

    def get_latest(self, limit=20):
        return self._query(
            """
            SELECT timestamp, mode, infer_ms, label, confidence, bbox_x, bbox_y, bbox_w, bbox_h, depth_cm, event_id
            FROM detections ORDER BY id DESC LIMIT ?
            """,
            (int(limit),),
        )

    def get_history(self, minutes=60, label=None):
        sql = "SELECT timestamp, mode, infer_ms, label, confidence, depth_cm, event_id FROM detections WHERE timestamp >= datetime('now', ?)"
        params = [f'-{int(minutes)} minutes']
        if label:
            sql += ' AND label = ?'
            params.append(label)
        sql += ' ORDER BY id DESC LIMIT 500'
        return self._query(sql, tuple(params))

    def get_events(self, hours=24, label=None, status=None):
        sql = "SELECT * FROM events WHERE first_seen >= datetime('now', ?)"
        params = [f'-{int(hours)} hours']
        if label:
            sql += ' AND label = ?'
            params.append(label)
        if status:
            sql += ' AND status = ?'
            params.append(status)
        sql += ' ORDER BY event_id DESC LIMIT 500'
        return self._query(sql, tuple(params))

    def get_event_detail(self, event_id: int):
        ev = self._query('SELECT * FROM events WHERE event_id=? LIMIT 1', (int(event_id),))
        if not ev:
            return None
        dets = self._query(
            'SELECT timestamp, confidence, depth_cm, bbox_x, bbox_y, bbox_w, bbox_h FROM detections WHERE event_id=? ORDER BY id DESC LIMIT 500',
            (int(event_id),),
        )
        out = ev[0]
        out['detections'] = dets
        return out

    def get_closest(self, label=None):
        sql = "SELECT label, MIN(min_depth_cm) AS min_depth_cm FROM events WHERE min_depth_cm IS NOT NULL"
        params = []
        if label:
            sql += ' AND label=?'
            params.append(label)
        sql += ' GROUP BY label ORDER BY min_depth_cm ASC'
        return self._query(sql, tuple(params))

    def get_stats(self, hours=24):
        h = int(hours)
        by_event = self._query(
            "SELECT label, COUNT(*) AS event_count, COALESCE(SUM(duration_sec),0) AS total_duration_sec FROM events WHERE first_seen >= datetime('now', ?) GROUP BY label ORDER BY event_count DESC",
            (f'-{h} hours',),
        )
        by_detection = self._query(
            "SELECT label, COUNT(*) AS n FROM detections WHERE timestamp >= datetime('now', ?) GROUP BY label ORDER BY n DESC",
            (f'-{h} hours',),
        )
        active_rows = self._query("SELECT label, COUNT(*) AS n FROM events WHERE status='active' GROUP BY label")
        active_now = {r['label']: r['n'] for r in active_rows}
        return {
            'hours': h,
            'by_event': by_event,
            'by_detection': by_detection,
            'active_now': active_now,
            'closest': self.get_closest(),
        }

    def get_db_size_mb(self):
        total = 0
        base = str(self.db_path)
        for path in (base, base + '-wal', base + '-shm'):
            if os.path.exists(path):
                total += os.path.getsize(path)
        return round(total / (1024 * 1024), 1)

    def get_last_errors(self, limit=10):
        return self._query(
            "SELECT timestamp, event_type, detail FROM system_events WHERE severity IN ('error','warning') ORDER BY id DESC LIMIT ?",
            (int(limit),),
        )

    def _maybe_prune(self):
        now = time.monotonic()
        if now - self._last_prune < self.prune_every_sec:
            return
        self._last_prune = now
        with self._lock:
            with self.conn:
                self.conn.execute("DELETE FROM detections WHERE timestamp < datetime('now', ?)", (f'-{self.retention_days} days',))
                self.conn.execute("DELETE FROM events WHERE first_seen < datetime('now', ?)", (f'-{self.retention_days} days',))
                self.conn.execute("DELETE FROM system_events WHERE timestamp < datetime('now', '-30 days')")
