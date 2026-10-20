#!/usr/bin/env python3
"""Visual preprocessing and *diagnostic* candidate coin detections for RS v2.

The detector is deliberately not used as a training reward until it is
validated on actual gameplay. Candidate detections are not ground truth.
"""
from pathlib import Path
import cv2
import numpy as np

WIDTH, HEIGHT = 256, 144
STACK = 4
CHANNELS = STACK * 3
ACTIONS = ('NONE', 'LEFT', 'RIGHT', 'JUMP', 'ROLL')


def bgra_to_chw(bgra, width=WIDTH, height=HEIGHT):
    image = cv2.resize(bgra, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def rgb_to_chw(rgb, width=WIDTH, height=HEIGHT):
    image = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image.transpose(2, 0, 1))


def frames_to_obs(frames):
    assert len(frames) == STACK
    return np.ascontiguousarray(np.stack(frames).reshape(CHANNELS, HEIGHT, WIDTH))


def latest_rgb(obs):
    assert obs.shape == (CHANNELS, HEIGHT, WIDTH), obs.shape
    return np.moveaxis(obs[-3:], 0, 2).copy()


def candidate_coins(rgb):
    """Diagnostic-only yellow/orange circular components, normalized coords.

    Not sufficient to distinguish every coin from signs, pumpkins, or lighting.
    Only count features in perspective roadway; always visually inspect overlays.
    """
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, (9, 135, 177), (37, 255, 255))
    road = np.zeros((h, w), dtype=np.uint8)
    # At horizon, roadway is very narrow; it widens toward the camera.
    for yy in range(round(.12*h), round(.86*h)):
        frac = yy / h
        half = .075 + .31 * max(0, (frac-.12)/.74)
        cx = .5*w
        road[yy, max(0, round(cx-half*w)):min(w, round(cx+half*w))] = 255
    mask = cv2.bitwise_and(yellow, road)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    num, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    result = []
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if not (3 <= area <= .008*w*h and 2 <= bw <= .09*w and 2 <= bh <= .09*h):
            continue
        aspect = bw / max(1, bh)
        if not (.46 <= aspect <= 2.15):
            continue
        cx, cy = centers[i]
        result.append((float(cx/w), float(cy/h), float(max(bw/w, bh/h)), int(area)))
    return result


def annotate(rgb, coins=None, text=None):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = img.shape[:2]
    for x, y, r, _ in (coins if coins is not None else candidate_coins(rgb)):
        cv2.circle(img, (round(x*w), round(y*h)), max(3, round(r*max(w, h)*.6)), (0, 255, 255), 1)
    if text:
        cv2.putText(img, text[:105], (8, min(23, h-4)), cv2.FONT_HERSHEY_SIMPLEX,
                    .48 if w > 500 else .35, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_supplied_screenshots(paths, output):
    images = []
    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tiny = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        marked = annotate(tiny, text=Path(path).name[:19])
        images.append(cv2.resize(marked, (WIDTH*3, HEIGHT*3), interpolation=cv2.INTER_NEAREST))
    if not images:
        raise ValueError('No usable input screenshots')
    cv2.imwrite(str(output), np.vstack(images))
    return len(images)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--screenshots', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('rs_v2_screenshot_diagnostics.jpg'))
    args = p.parse_args()
    print(f'Saved {render_supplied_screenshots(args.screenshots,args.output)} panels to {args.output}')
