import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np

from archive import save_sources
from dataset import (review, export, catalogue, contact_sheet, read, digest,
                     policy_frame, segments, source_frame, source_index)


class NativeRecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/'run'; self.run.mkdir()
        self.rgb = np.zeros((30,60,3), np.uint8)
        self.rgb[:,15:45] = (240,110,20)
        payload = cv2.imencode('.png', cv2.cvtColor(self.rgb,cv2.COLOR_RGB2BGR))[1].tobytes()
        archive = save_sources(self.run, [{'at':100.+i*.05, 'canvas_jpeg':payload} for i in range(6)])
        np.save(self.run/'timestamps.npy', np.arange(6)*.05)
        np.save(self.run/'actions.npy', [0,1,0,3,0,4])
        np.save(self.run/'valid.npy', [1,1,0,1,1,1])
        np.save(self.run/'aux.npy', np.zeros((6,557),np.float32))
        (self.run/'events.jsonl').write_text('')
        (self.run/'manifest.json').write_text(json.dumps({
            'schema':1, 'source_archive':archive,'duration_seconds':.25,
            'status':'candidate','valid_action_samples':5,
            'capture':{'framing_profile':'canvas_v1','detector_mode':'off'}}))

    def approve(self):
        return review(self.run, status='approved', reviewed=True, intervals='0.05:0.2')

    def test_lossless_roundtrip(self):
        _, index = source_index(self.run)
        np.testing.assert_array_equal(source_frame(self.run,index,0),self.rgb)

    def test_jpeg_is_not_native_master(self):
        with self.assertRaises(ValueError):
            save_sources(self.root,[{'canvas_jpeg':b'jpeg','at':1}])

    def test_explicit_review_required(self):
        with self.assertRaises(ValueError): review(self.run,status='approved',intervals='0:0.2')
        with self.assertRaises(ValueError): review(self.run,status='approved',reviewed=True)

    def test_no_automatic_approval_from_duration(self):
        catalogue(self.root,self.root/'review.html')
        self.assertEqual(read(self.run/'manifest.json')['status'],'candidate')
        with self.assertRaises(ValueError): export(self.run,self.root/'export','road_wide',128)

    def test_trim_validation(self):
        for text in ('0:0.3','0.2:0.1','0:0.2,0.1:0.25','nan:0.1'):
            with self.assertRaises(ValueError): segments(text,.25)

    def test_no_movement_in_approved_range(self):
        with self.assertRaises(ValueError):
            review(self.run,status='approved',reviewed=True,intervals='0:0.01')

    def test_review_does_not_modify_original(self):
        before=digest(self.run/'manifest.json')
        self.approve()
        self.assertEqual(before,digest(self.run/'manifest.json'))

    def test_export_preserves_times_labels_and_mask(self):
        self.approve(); target=self.root/'export'
        export(self.run,target,'road_wide',128)
        self.assertEqual(np.load(target/'frames.npy').shape,(6,3,128,128))
        np.testing.assert_array_equal(np.load(target/'training_mask.npy'),[0,1,0,1,1,0])
        for name in ('actions.npy','timestamps.npy','valid.npy','events.jsonl'):
            self.assertEqual((target/name).read_bytes(),(self.run/name).read_bytes())
        self.assertEqual(read(target/'export.json')['status'],'COMPLETE')
        with self.assertRaises(FileExistsError): export(self.run,target,'road_wide',128)

    def test_changed_label_rejects_approval(self):
        self.approve(); np.save(self.run/'actions.npy',[1]*6)
        with self.assertRaises(ValueError): export(self.run,self.root/'export','full',128)

    def test_changed_pixels_no_completion_marker(self):
        self.approve()
        (self.run/'source_frames/000003.png').write_bytes(b'damaged')
        with self.assertRaises(ValueError): export(self.run,self.root/'export','full',128)
        self.assertFalse((self.root/'export/export.json').exists())

    def test_latest_rejection_overrides_previous_approval(self):
        self.approve(); review(self.run,status='rejected')
        with self.assertRaises(ValueError): export(self.run,self.root/'export','full',128)

    def test_sizes_and_crop_match_active_transform(self):
        from development.experiments.road_vision.road_crop import crop_rgb
        from dataset import CROPS
        for size in (128,192,256,384):
            expected=crop_rgb(self.rgb,{'roi':CROPS['road_wide'],'size':[size,size]})
            np.testing.assert_array_equal(policy_frame(self.rgb,'road_wide',size),expected)

    def test_preview_works_for_unapproved_candidate(self):
        contact_sheet(self.run,self.root/'preview.jpg','road_wide',128)
        self.assertTrue((self.root/'preview.jpg').exists())
        self.assertEqual(read(self.run/'manifest.json')['status'],'candidate')

    def test_snapshot_changes_only_intended_anchors(self):
        import build_snapshot
        provenance=read(build_snapshot.HERE/'snapshot_provenance.json')
        self.assertEqual(digest(build_snapshot.ROOT/'rs_v612_demo.py'),provenance['source_sha256'])
        self.assertEqual(digest(build_snapshot.HERE/'recorder_snapshot.py'),provenance['snapshot_sha256'])
        text=(build_snapshot.HERE/'recorder_snapshot.py').read_text(encoding='utf-8')
        compile(text,'recorder_snapshot.py','exec')
        self.assertIn('IMWRITE_PNG_COMPRESSION',text)

    def test_snapshot_label_alignment_and_130ms_bound(self):
        import recorder_snapshot as recorder
        times=[5.,5.05,5.10,5.30,5.35]
        events=[{'at':5.025,'type':'keydown','code':'ArrowLeft','repeat':False},
                {'at':5.125,'type':'keydown','code':'ArrowRight','repeat':False}]
        actions,valid=recorder.action_labels(times,events)
        self.assertEqual(actions.tolist(),[1,0,2,0,0])
        self.assertEqual(valid.tolist(),[True,True,False,True,False])

    def test_snapshot_save_end_to_end_keeps_native_and_legacy(self):
        import recorder_snapshot as recorder
        payload=(self.run/'source_frames/000000.png').read_bytes()
        samples=[{'at':100.+i*.05,'frame':np.zeros((3,288,512),np.uint8),
                  'canvas_jpeg':payload,'detection':None} for i in range(6)]
        target=self.root/'saved'
        with patch.object(recorder,'make_contact_sheet'), patch.object(recorder,'make_preview_video',return_value=None):
            result=recorder.save_replay(samples,[],target,status='candidate',
                capture_info={'framing_profile':'canvas_v1','detector_mode':'off'})
        self.assertEqual(result['status'],'candidate')
        self.assertEqual(result['source_archive']['count'],6)
        self.assertEqual(np.load(target/'frames.npz')['frames'].shape,(6,3,288,512))
        source_index(target)
        with self.assertRaises(ValueError): recorder.curate(target,'approved','0.01-0.2')


if __name__=='__main__': unittest.main()
