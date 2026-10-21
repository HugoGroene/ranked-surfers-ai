#!/usr/bin/env python3
"""Nine-class V5 YOLO segmentation, exact-source timing and preview helpers.

Shadow benchmarking only. No V5 information is fed to V4's three-class PPO
auxiliary vector. No collision or action rules are encoded here.
"""
from __future__ import annotations

from collections import Counter
import time

import cv2
import numpy as np

V5_CLASSES = (
    'coin', 'train_front', 'train_side', 'train_roof', 'ramp',
    'barrier_jump', 'barrier_roll', 'barrier_either', 'tunnel_entrance',
)


def check_model_names(model):
    names = model.names
    actual = (tuple(str(names[i]) for i in sorted(names))
              if isinstance(names, dict) else tuple(str(n) for n in names))
    if actual != V5_CLASSES:
        raise ValueError('Expected V5 nine-class segmentation checkpoint in this order: '
                         f'{V5_CLASSES}. Model reports: {actual}')
    if getattr(model, 'task', None) != 'segment':
        raise ValueError(f'Expected YOLO segmentation model, got task={model.task!r}')


def distance_band(y2):
    """Image-height region only, NOT a calibrated physical distance or risk."""
    if y2 < .42:
        return 'upper_frame'
    if y2 < .73:
        return 'middle_frame'
    return 'lower_frame'


def detect_v5(model, frame_rgb, captured_at, episode, request_id, imgsz, conf):
    """Called from existing V4 single-worker asynchronous inference pool."""
    began = time.perf_counter()
    # Ultralytics takes OpenCV BGR numpy images; V4 stores native frames as RGB.
    # Keep the original exact-source RGB frame untouched for mask overlays.
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    converted = time.perf_counter()
    result = model.predict(source=bgr, imgsz=imgsz, conf=conf, verbose=False)[0]
    predicted = time.perf_counter()
    h, w = frame_rgb.shape[:2]
    objects = []
    # Accessing Ultralytics masks.xy converts model masks to original-image
    # polygon coordinates; measure that cost separately from model.predict.
    polygons = result.masks.xy if result.masks is not None else []
    if result.boxes is not None and len(result.boxes):
        coords = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        for idx, (xy, confidence, class_id) in enumerate(zip(coords, scores, class_ids)):
            name = result.names[int(class_id)]
            if name not in V5_CLASSES:
                raise ValueError(f'Unexpected V5 class {name!r} (id={class_id})')
            x1, y1, x2, y2 = [float(n) for n in xy]
            bounds = [float(np.clip(x1/w, 0, 1)), float(np.clip(y1/h, 0, 1)),
                      float(np.clip(x2/w, 0, 1)), float(np.clip(y2/h, 0, 1))]
            polygon = []
            if idx < len(polygons):
                points = np.asarray(polygons[idx], dtype=np.float32)
                if len(points) >= 3:
                    # Downsample only the visualization polygon, not inference.
                    stride = max(1, (len(points)+95)//96)
                    points = points[::stride]
                    polygon = [[float(np.clip(px/w,0,1)), float(np.clip(py/h,0,1))]
                               for px,py in points]
            objects.append({'class':str(name), 'conf':float(confidence),
                            'box':bounds, 'polygon':polygon,
                            'image_region':distance_band(bounds[3])})
    finished = time.perf_counter()
    return {
        'episode':int(episode), 'request_id':int(request_id),
        'captured_at':float(captured_at), 'finished_at':finished,
        'inference_ms':round((finished-began)*1000,2),
        'input_rgb_to_bgr_ms':round((converted-began)*1000,2),
        'predict_wall_ms':round((predicted-converted)*1000,2),
        'mask_and_box_extract_ms':round((finished-predicted)*1000,2),
        'detections':objects, 'source_frame':frame_rgb,
        'mask_count':sum(bool(item['polygon']) for item in objects),
    }


def shadow_features(snapshot, now=None, max_age_s=.6):
    """Preserve V4 aux shape but do not misrepresent train roofs as hazards.

    Returns an all-zero legacy 3-class vector. V5 detections are recorded
    separately in report/overlays and are NOT PPO inputs yet.
    """
    from rs_v2_live_hazards_v2 import FEATURE_COUNT
    if now is None:
        now = time.perf_counter()
    blank = np.zeros(FEATURE_COUNT, dtype=np.float32)
    if snapshot is None:
        return blank, None, False
    age_ms = max(0.0, now-float(snapshot['captured_at']))*1000
    fresh = age_ms <= 1000*max_age_s
    return blank, age_ms, fresh


def draw_mask_overlay(snapshot):
    """Draw on the exact frame the segmentation actually analyzed."""
    base = cv2.cvtColor(snapshot['source_frame'], cv2.COLOR_RGB2BGR)
    h,w = base.shape[:2]
    mask_layer = base.copy()
    for obj in snapshot['detections']:
        polygon = obj.get('polygon',[])
        if len(polygon) >= 3:
            coords=np.array([[int(x*w),int(y*h)] for x,y in polygon],dtype=np.int32)
            color = _color(obj['class'])
            cv2.fillPoly(mask_layer,[coords],color)
    vis=cv2.addWeighted(mask_layer,.3,base,.7,0)
    for obj in snapshot['detections']:
        x1,y1,x2,y2=obj['box']
        p1=(int(x1*w),int(y1*h));p2=(int(x2*w),int(y2*h))
        color=_color(obj['class'])
        cv2.rectangle(vis,p1,p2,color,2)
        tag=f"{obj['class']} {obj['conf']:.2f}"
        cv2.putText(vis,tag,(p1[0],max(14,p1[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    return vis


def _color(name):
    palette = [(255,204,0),(60,70,245),(245,100,55),(55,210,95),
               (225,85,215),(25,220,255),(190,145,255),(0,150,250),(160,220,65)]
    return palette[V5_CLASSES.index(name)]


def class_and_region_counts(snapshots):
    classes=Counter(); regions={region:Counter() for region in
             ('upper_frame','middle_frame','lower_frame')}
    for snap in snapshots:
        for d in snap['detections']:
            classes[d['class']]+=1
            regions[d['image_region']][d['class']]+=1
    return {'classes':dict(classes),'image_regions':{k:dict(v) for k,v in regions.items()}}
