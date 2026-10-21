#!/usr/bin/env python3
"""Make labeled previews of V5 LabelMe polygon annotations; no edits to source images."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from rs_v5_export_labelme import CLASSES, IMG_EXTS, load_annotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=Path('ai_data/v5_hazards/raw'))
    p.add_argument('--output',type=Path,default=Path('ai_data/v5_hazards/annotation_previews'))
    p.add_argument('--max-images',type=int,default=24)
    p.add_argument('--width',type=int,default=1280)
    a=p.parse_args()
    if not a.raw.is_dir():p.error(f'Missing {a.raw}')
    images=[im for session in sorted(a.raw.iterdir()) if session.is_dir()
            for im in sorted(session.iterdir()) if im.suffix.lower() in IMG_EXTS and im.name.startswith('frame_')
            and im.with_suffix('.json').is_file()]
    if not images:p.error('No labeled images (JPG + LabelMe JSON) found')
    a.output.mkdir(parents=True, exist_ok=True)
    for image in images[:a.max_images]:
        ann=load_annotation(image)
        frame=cv2.imread(str(image))
        h,w=frame.shape[:2]
        shrink=min(1.,a.width/w)
        vis=cv2.resize(frame,(round(w*shrink),round(h*shrink)),interpolation=cv2.INTER_AREA)
        for cls,pts in ann:
            i=CLASSES.index(cls)
            color=(int((i*97+45)%255), int((i*73+110)%255), int((i*59+170)%255))
            coords=np.asarray([[round(x*(vis.shape[1]-1)),round(y*(vis.shape[0]-1))] for x,y in pts],dtype=np.int32)
            cv2.polylines(vis,[coords],True,color,2,lineType=cv2.LINE_AA)
            tx,ty=coords.min(axis=0)
            cv2.putText(vis,cls,(int(tx),max(18,int(ty)-4)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),4,cv2.LINE_AA)
            cv2.putText(vis,cls,(int(tx),max(18,int(ty)-4)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2,cv2.LINE_AA)
        dest=a.output/f'{image.parent.name}__{image.stem}.jpg'
        cv2.imwrite(str(dest),vis,[cv2.IMWRITE_JPEG_QUALITY,90])
    print(f'[V5] Wrote {min(len(images),a.max_images)} annotated previews into {a.output}')

if __name__=='__main__':main()
