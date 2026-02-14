from __future__ import annotations
import sqlite3
import time
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
    depth_cm   REAL
);
CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_label     ON detections(label);
CREATE INDEX IF NOT EXISTS idx_depth     ON detections(depth_cm);
"""


class DetectionStorage:
    def __init__(self, db_path: Path, retention_days: int = 7, prune_every_sec: int = 300):
        self.db_path = db_path
        self.retention_days = retention_days
        self.prune_every_sec = prune_every_sec
        self._last_prune = 0.0

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), timeout=5.0)
        self.conn.execute('PRAGMA journal_mode = WAL;')
        self.conn.execute('PRAGMA synchronous = NORMAL;')
        self.conn.execute('PRAGMA cache_size = -2000;')
        self.conn.execute('PRAGMA temp_store = MEMORY;')
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

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
                )
            )

        if rows:
            with self.conn:
                self.conn.executemany(
                    """
                    INSERT INTO detections (
                        timestamp, mode, infer_ms, label, confidence,
                        bbox_x, bbox_y, bbox_w, bbox_h, depth_cm
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

        self._maybe_prune()

    def _maybe_prune(self):
        now = time.monotonic()
        if now - self._last_prune < self.prune_every_sec:
            return
        self._last_prune = now
        with self.conn:
            self.conn.execute(
                "DELETE FROM detections WHERE timestamp < datetime('now', ?)",
                (f'-{self.retention_days} days',),
            )
