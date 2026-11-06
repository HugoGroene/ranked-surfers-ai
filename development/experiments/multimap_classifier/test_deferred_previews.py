import json
from pathlib import Path
import tempfile
import threading
import unittest
import cv2
import numpy as np
from deferred_previews import PreviewWriter


class PreviewTests(unittest.TestCase):
    def test_submission_returns_while_writer_busy_and_pixels_are_owned(self):
        entered=threading.Event();release=threading.Event()
        def write(path,image):
            entered.set()
            if not release.wait(2):raise TimeoutError('test did not release writer')
            return cv2.imwrite(path,image)
        with tempfile.TemporaryDirectory() as folder:
            writer=PreviewWriter(folder,write)
            image=np.full((20,30,3),74,np.uint8)
            path=Path(folder)/'canvas_v1_native.png'
            try:
                self.assertTrue(writer.submit(path,image))
                self.assertTrue(entered.wait(1))
                self.assertFalse(path.exists())
                image.fill(0)
            finally:
                release.set();writer.close()
            self.assertTrue(np.all(cv2.imread(str(path))==74))
            self.assertEqual(json.loads((Path(folder)/'startup_previews.json').read_text())['status'],'COMPLETE')
            self.assertTrue(all(not thread.is_alive() for thread in writer.executor._threads))

    def test_failed_save_is_reported_at_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            writer=PreviewWriter(folder,lambda *args:False)
            writer.submit(Path(folder)/'canvas_v1_policy.png',np.zeros((2,2,3),np.uint8))
            with self.assertRaises(OSError):writer.close()
            report=json.loads((Path(folder)/'startup_previews.json').read_text())
            self.assertEqual(report['status'],'FAILED')
            self.assertIn('canvas_v1_policy.png',report['errors'])

    def test_only_three_unique_diagnostics_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            writer=PreviewWriter(folder,lambda *args:True)
            image=np.zeros((2,2,3),np.uint8)
            try:
                for name in writer.NAMES:writer.submit(Path(folder)/name,image)
                with self.assertRaises(FileExistsError):writer.submit(Path(folder)/'canvas_v1_native.png',image)
                with self.assertRaises(ValueError):writer.submit(Path(folder)/'extra.png',image)
                with self.assertRaises(ValueError):writer.submit(Path(folder).parent/'canvas_v1_policy.png',image)
            finally:writer.close()


if __name__=='__main__':unittest.main()
