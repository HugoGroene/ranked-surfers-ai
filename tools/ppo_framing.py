"""Versioned policy/detector/BC preprocessing; guard sources are separate."""
import cv2
import numpy as np
from rs_v5_segmentation_live_benchmark import fit_rgb_to_target

CANVAS_WH = (1920,945)


def canvas_observation(bgra, target=(512,288), interpolation=cv2.INTER_LINEAR):
    if bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError('canvas_v1 requires BGRA canvas pixels')
    h,w = bgra.shape[:2]
    if w < 640 or h < 320 or abs((w/h)/(1920/945)-1) > .005:
        raise ValueError('canvas_v1 requires calibrated 1920:945 aspect ratio (0.5% tolerance)')
    native = cv2.cvtColor(bgra,cv2.COLOR_BGRA2RGB)
    small = fit_rgb_to_target(native,target,interpolation=interpolation)
    return np.ascontiguousarray(small.transpose(2,0,1)),native


def manifest_profile(manifest):
    return manifest.get('capture',{}).get('framing_profile','legacy')
