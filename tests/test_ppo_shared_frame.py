import mmap
import struct
import unittest
import numpy as np
from tools.ppo_shared_frame import read_shared_frame
from rs_v612_gameplay_guard import GuardUncertainError


class SharedFrameTests(unittest.TestCase):
    def test_header_validation_and_owned_pixels_survive_next_publish(self):
        with mmap.mmap(-1,72) as mapping:
            header=struct.pack('<IIIIQQ',0x52534631,1,2,1,7,8)
            mapping[:32]=header;mapping[64:72]=bytes(range(8))
            image=read_shared_frame(mapping,{'serial':7},2,1)
            mapping[64:72]=bytes([99])*8
            self.assertEqual(image.tolist(),list(range(8)))
            for serial,w,h in [(8,2,1),(7,1,2),(7,1,1)]:
                with self.assertRaises(GuardUncertainError):read_shared_frame(mapping,{'serial':serial},w,h)

    def test_changed_publication_during_copy_is_rejected(self):
        class ChangedMap(mmap.mmap):
            def __getitem__(self,key):
                if key==slice(None,32):
                    self.reads+=1
                    if self.reads==2:return bytes(32)
                return super().__getitem__(key)
        with ChangedMap(-1,72) as mapping:
            mapping.reads=0
            mapping[:32]=struct.pack('<IIIIQQ',0x52534631,1,2,1,7,8)
            with self.assertRaisesRegex(GuardUncertainError,'overwritten'):
                read_shared_frame(mapping,{'serial':7},2,1)
