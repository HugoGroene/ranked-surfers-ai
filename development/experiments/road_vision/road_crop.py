"""Policy-only crop; never modify the full-resolution guard image."""
import cv2
import numpy as np
from road_config import CROPS

def crop_rgb(rgb,config):
    from rs_v5_segmentation_live_benchmark import fit_rgb_to_target
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:raise ValueError('Expected uint8 RGB')
    h,w=rgb.shape[:2];x0,y0,x1,y1=config['roi']
    if not (0<=x0<x1<=1 and 0<=y0<y1<=1):raise ValueError('Invalid ROI')
    left,top,right,bottom=round(x0*w),round(y0*h),round(x1*w),round(y1*h)
    if right<=left or bottom<=top:raise ValueError('Empty crop')
    small=fit_rgb_to_target(rgb[top:bottom,left:right],tuple(config['size']),cv2.INTER_AREA)
    return np.ascontiguousarray(small.transpose(2,0,1))

def observation(bgra,config):
    if bgra.shape!=(945,1920,4) or bgra.dtype!=np.uint8:
        raise ValueError('Road crop requires unchanged 1920x945 BGRA canvas')
    rgb=cv2.cvtColor(bgra,cv2.COLOR_BGRA2RGB)
    return crop_rgb(rgb,config),rgb  # Full native view is still available to diagnostics.

def recorded_observation(chw,config):
    """Explicit low-resolution conversion; it cannot recover discarded detail."""
    if chw.shape!=(3,288,512) or chw.dtype!=np.uint8:raise ValueError('Unexpected recorded image')
    # canvas_v1 letterboxes 1920x945 into 512x288: content is 512x252, y=18.
    return crop_rgb(chw[:,18:270,:].transpose(1,2,0),config)

def preview(rgb,config,path,*,writer=None):
    """Save a diagnostic overlay on a copy; never consumed by a policy."""
    h,w=rgb.shape[:2];x0,y0,x1,y1=config['roi']
    out=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    cv2.rectangle(out,(round(x0*w),round(y0*h)),(round(x1*w)-1,round(y1*h)-1),(0,255,0),3)
    if not (writer or cv2.imwrite)(str(path),out):raise OSError('Preview write failed')
