import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from common import write_new,read
from label_guard_queue import run_queue,sessions


class QueueTests(unittest.TestCase):
    def test_quit_stops_queue_and_resume_skips_completed_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name,n in [('a',2),('b',1)]:
                session=root/'sessions'/name;session.mkdir(parents=True)
                write_new(session/'session.json',dict(bc_source_session=name,split='train',frames=n))
                write_new(session/'import_manifest.json',dict(frames=[]))
                for i in range(n):cv2.imwrite(str(session/f'{i}.png'),np.zeros((20,30,3),np.uint8))
            with patch('rs_v4_screen_states.cv2.imshow'),patch('rs_v4_screen_states.cv2.destroyAllWindows'),patch('rs_v4_screen_states.cv2.waitKey',side_effect=[ord('g'),ord('q')]):
                run_queue(root)
            self.assertFalse(read(root/'sessions/a/label_queue_progress.json')['complete'])
            self.assertFalse((root/'sessions/b/labels.csv').exists())
            with patch('rs_v4_screen_states.cv2.imshow'),patch('rs_v4_screen_states.cv2.destroyAllWindows'),patch('rs_v4_screen_states.cv2.waitKey',side_effect=[ord('g'),ord('m')]) as keys:
                run_queue(root)
                self.assertEqual(keys.call_count,2)
            self.assertTrue(all(done for _,_,done in sessions(root)))
            with patch('rs_v4_screen_states.cmd_label') as label:
                run_queue(root);label.assert_not_called()


if __name__=='__main__':unittest.main()
