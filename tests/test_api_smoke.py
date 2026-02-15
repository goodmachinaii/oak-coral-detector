import tempfile
import unittest
from pathlib import Path

from oak_vision.api import create_app
from oak_vision.storage import DetectionStorage


class TestApiSmoke(unittest.TestCase):
    def test_core_endpoints(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / 'oak.db'
            storage = DetectionStorage(db_path=db, retention_days=7, prune_every_sec=999999)
            try:
                event_id = storage.create_event(99, 'person', (10, 20, 30, 40), 0.9, 123.4)
                storage.store([
                    {'label': 'person', 'confidence': 0.9, 'bbox': (10, 20, 30, 40), 'depth_cm': 123.4, 'event_id': event_id}
                ], mode='cpu', infer_ms=10.5)

                state = {
                    'mode': 'cpu',
                    'fps': 7.5,
                    'infer_ms': 10.5,
                    'running': True,
                    'oak_connected': True,
                    'uptime_sec': 5,
                }
                app = create_app(storage, state)
                client = app.test_client()

                r = client.get('/status')
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json['mode'], 'cpu')

                r = client.get('/events?hours=24')
                self.assertEqual(r.status_code, 200)
                self.assertTrue(isinstance(r.json, list))

                r = client.get(f'/events/{event_id}')
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json['event_id'], event_id)

                r = client.get('/stats?hours=24')
                self.assertEqual(r.status_code, 200)
                self.assertIn('by_event', r.json)

                r = client.get('/health')
                self.assertEqual(r.status_code, 200)
                self.assertIn('db_size_mb', r.json)
            finally:
                storage.close()


if __name__ == '__main__':
    unittest.main()
