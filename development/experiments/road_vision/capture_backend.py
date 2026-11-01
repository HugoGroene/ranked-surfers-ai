"""Select full-pixel Electron paths without changing the policy input contract."""
import sys
from pathlib import Path

BACKENDS=('native_bitmap','bitmap_100','gpu_100')

def install_backend(name):
    if name not in BACKENDS:raise ValueError('Unsupported road capture backend')
    if name=='native_bitmap':return
    folder=Path(__file__).resolve().parent.parent/'capture_pipeline'
    sys.path.insert(0,str(folder))
    try:
        from adapter import install_capture
        install_capture(name)
    finally:sys.path.remove(str(folder))
