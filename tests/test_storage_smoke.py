import tempfile
import unittest
from pathlib import Path

from oak_vision.storage import DetectionStorage


class TestStorageSmoke(unittest.TestCase):
    def test_event_lifecycle_and_health_queries(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / 'oak.db'
            s = DetectionStorage(db_path=db, retention_days=7, prune_every_sec=999999)
            try:
                event_id = s.create_event(1, 'person', (10, 20, 30, 40), 0.9, 120.0)
                s.update_event(event_id, (11, 21, 30, 40), 0.95, 110.0)
                s.store([
                    {'label': 'person', 'confidence': 0.95, 'bbox': (11, 21, 30, 40), 'depth_cm': 110.0, 'event_id': event_id}
                ], mode='cpu', infer_ms=12.3)
                s.log_system_event('startup', 'orchestrator', 'info', 'ok')
                s.close_event(event_id, 5.0)

                events = s.get_events(hours=24)
                self.assertGreaterEqual(len(events), 1)
                self.assertIn('min_depth_cm', events[0])

                closest = s.get_closest()
                self.assertTrue(isinstance(closest, list))

                health_errors = s.get_last_errors(limit=5)
                self.assertEqual(len(health_errors), 0)
            finally:
                s.close()


if __name__ == '__main__':
    unittest.main()
