import unittest

from oak_vision.event_tracker import EventTracker


class TestEventTrackerSmoke(unittest.TestCase):
    def test_confirms_after_consistent_scores(self):
        t = EventTracker(score_threshold=0.7, initialization_delay=1, hit_counter_max=3)
        det = {'label': 'person', 'bbox': (100, 100, 50, 100), 'confidence': 0.9, 'depth_cm': 150.0}

        r1 = t.update([det])
        self.assertTrue(len(r1) >= 1)

        det2 = {'label': 'person', 'bbox': (102, 101, 50, 100), 'confidence': 0.92, 'depth_cm': 148.0}
        r2 = t.update([det2])
        self.assertTrue(any(x.get('confirmed') for x in r2))


if __name__ == '__main__':
    unittest.main()
