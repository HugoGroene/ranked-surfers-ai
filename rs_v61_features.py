#!/usr/bin/env python3
"""V6.1 observations: preserve all V6 RGB+segmentation inputs, add screen-column
context and *uncertain* commanded-lane/action timing. No obstacle->action rules.

Screen columns are image-space thirds, not calibrated track lanes or a safety map.
The optional commanded-lane odometry is not visually verified; confidence must
never be interpreted as evidence that a keypress changed the game state.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from rs_v5_shadow_segmentation import V5_CLASSES
from rs_v6_ppo_features import (AUX_DIM as V6_AUX_DIM,
                                FEATURE_VERSION as V6_FEATURE_VERSION,
                                encode_aux as encode_v6_aux)

COLUMNS = 3
METRICS = 3  # maximum weighted confidence, max box-bottom y, max width coverage
SCREEN_COLUMN_DIM = len(V5_CLASSES)*COLUMNS*METRICS
LANE_DIM = 4  # three one-hot *tentative* commanded positions and confidence
ACTION_TIMING_DIM = 5  # seconds since each of five actions, bounded and normalized
AUX_DIM = V6_AUX_DIM + SCREEN_COLUMN_DIM + LANE_DIM + ACTION_TIMING_DIM
FEATURE_VERSION = 'v61_screen_columns_command_odometry_timing_v1'
ACTION_NAMES = ('NONE', 'LEFT', 'RIGHT', 'JUMP', 'ROLL')


def _valid_detection(snapshot, episode_tag, now, max_stale_ms):
    if not snapshot or int(snapshot.get('episode', -1)) != int(episode_tag):
        return False
    timestamp = float(snapshot.get('captured_at', float('nan')))
    age_ms = (float(now)-timestamp)*1000.
    return np.isfinite(age_ms) and -5. <= age_ms <= max_stale_ms


def screen_columns(snapshot, *, episode_tag, now, max_stale_ms=300.):
    """9 classes x 3 horizontal IMAGE columns x (confidence,bottom,coverage).

    A detection may occupy more than one screen column. Values are zero if
    missing/stale/other episode. Not collision probabilities, calibrated lanes,
    distance-to-collision, nor action recommendations.
    """
    out = np.zeros((len(V5_CLASSES), COLUMNS, METRICS), dtype=np.float32)
    if not _valid_detection(snapshot, episode_tag, now, max_stale_ms):
        return out.ravel()
    cls_index = {name:i for i,name in enumerate(V5_CLASSES)}
    for obj in snapshot.get('detections', []):
        name = obj.get('class')
        if name not in cls_index:
            raise ValueError('Unknown V5 class: '+str(name))
        try:
            x1,y1,x2,y2 = [float(x) for x in obj['box']]
            conf = float(obj.get('conf', 0.))
        except (KeyError,ValueError,TypeError):
            continue
        if not np.isfinite([x1,y1,x2,y2,conf]).all():
            continue
        x1,y1,x2,y2,conf = np.clip([x1,y1,x2,y2,conf],0.,1.)
        if x2<=x1 or y2<=y1:
            continue
        for col in range(COLUMNS):
            left,right=col/COLUMNS,(col+1)/COLUMNS
            overlap=max(0.,min(x2,right)-max(x1,left))
            if overlap<=0:
                continue
            coverage=min(1.,overlap/(right-left))
            out[cls_index[name],col,0]=max(out[cls_index[name],col,0],conf*coverage)
            out[cls_index[name],col,1]=max(out[cls_index[name],col,1],y2)
            out[cls_index[name],col,2]=max(out[cls_index[name],col,2],coverage)
    return out.ravel()


@dataclass
class ActionContext:
    """Command history, with *unverified* lane belief for debugging.

    The start lane is only assumed to be center when opt-in enabled. After
    commands, belief uncertainty increases because no visual motion tracking
    is yet calibrated. Disabled by default, so false lane labels are not fed
    into the network.
    """
    assume_center: bool = False

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.commanded_lane = 1
        self.lane_confidence = .25 if self.assume_center else 0.
        self.last_action_times = [None]*5
        self.command_counts = [0]*5

    def note_action(self, action, at):
        action=int(action)
        if not 0 <= action < 5:
            raise ValueError(action)
        self.last_action_times[action] = float(at)
        self.command_counts[action] += 1
        if action in (1,2):
            direction = -1 if action==1 else 1
            self.commanded_lane = max(0,min(2,self.commanded_lane+direction))
            # A keypress is NOT proof of in-game movement.
            self.lane_confidence *= .55

    def vector(self, now):
        lane=np.zeros(4,dtype=np.float32)
        if self.lane_confidence > 0:
            lane[self.commanded_lane]=self.lane_confidence
        lane[3] = self.lane_confidence
        timings = np.ones(5,dtype=np.float32)
        for i, t in enumerate(self.last_action_times):
            if t is not None:
                timings[i]=float(np.clip((float(now)-t)/1.0,0.,1.))
        return np.concatenate([lane,timings]).astype(np.float32)


def encode_aux(hazards, action_history, episode_seconds, frame_intervals,
               detection_age_ms, *, snapshot, episode_tag, now, action_context,
               max_stale_ms=300.):
    base=encode_v6_aux(hazards,action_history,episode_seconds,frame_intervals,detection_age_ms)
    columns=screen_columns(snapshot,episode_tag=episode_tag,now=now,max_stale_ms=max_stale_ms)
    result=np.concatenate([base,columns,action_context.vector(now)]).astype(np.float32)
    if result.shape != (AUX_DIM,) or not np.isfinite(result).all() or np.any(result<0) or np.any(result>1):
        raise RuntimeError(f'Invalid V6.1 feature shape/range: {result.shape}')
    return result


def shape_description():
    return {'feature_version':FEATURE_VERSION, 'v6_base_version':V6_FEATURE_VERSION,
            'classes':list(V5_CLASSES),'image_frames':4,'image_channels':12,
            'v6_aux_dim':V6_AUX_DIM,'screen_column_dim':SCREEN_COLUMN_DIM,
            'action_timing_dim':ACTION_TIMING_DIM, 'commanded_lane_dim':LANE_DIM,
            'aux_dim':AUX_DIM,
            'warning':'Screen columns are NOT track lanes; commanded lane is unverified and disabled by default.'}
