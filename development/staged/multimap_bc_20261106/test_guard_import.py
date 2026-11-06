import tempfile
from pathlib import Path
import unittest
import cv2
import numpy as np
from common import write_new,read,digest
from import_guard_warnings import prepare,warning_indices


class GuardImportTests(unittest.TestCase):
    def test_all_warnings_includes_every_uncertain_and_menu_frame(self):
        rows=[dict(at=i*.02,state='UNCERTAIN' if i<30 else 'MENU') for i in range(33)]
        self.assertEqual(warning_indices(rows,all_warnings=True),list(range(33)))
    def test_warning_burst_sampling_keeps_boundaries_and_context(self):
        rows=[dict(at=i*.05,state='UNCERTAIN' if 8<=i<=30 else 'GAMEPLAY') for i in range(40)]
        ids=warning_indices(rows)
        self.assertLess(len(ids),20)
        for i in (8,30):self.assertIn(i,ids)
        self.assertTrue(any(i<8 for i in ids));self.assertTrue(any(i>30 for i in ids))
        self.assertEqual(warning_indices([dict(at=0,state='GAMEPLAY')]),[])

    def test_exact_guard_padding_source_unchanged_unlabelled_and_split_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);session=root/'bc_session';run=session/'run_0001';frames=run/'frames'
            frames.mkdir(parents=True)
            write_new(session/'session.json',dict(map='halloween',split='train',map_release='fixture'))
            native=np.full((945,1920,3),123,np.uint8);source=frames/'000000.png'
            cv2.imwrite(str(source),native);sha=digest(source)
            write_new(run/'frames.json',[dict(at=5,state='UNCERTAIN',file=source.name,sha256=sha)])
            dest=prepare(session,root/'classifier');meta=read(dest/'session.json')
            self.assertEqual(meta['split'],'train');self.assertFalse((dest/'labels.csv').exists())
            row=read(dest/'import_manifest.json')['frames'][0]
            image=cv2.imread(str(dest/row['file']))
            self.assertEqual(image.shape,(1080,1920,3))
            np.testing.assert_array_equal(image[87:1032],native)
            self.assertFalse(image[:87].any());self.assertFalse(image[1032:].any())
            self.assertEqual(digest(source),sha);self.assertIsNone(row['label'])
            with self.assertRaisesRegex(ValueError,'Already imported'):prepare(session,root/'classifier')
            self.assertIsNone(prepare(session,root/'classifier',all_warnings=True))


if __name__=='__main__':unittest.main()
