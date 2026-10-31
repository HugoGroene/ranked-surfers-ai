"""Cache object geometry, while recomputing freshness for every observation."""
import math
import numpy as np
from rs_v6_ppo_features import encode_hazards
from rs_v61_features import screen_columns


class DetectorFeatureCache:
    def __init__(self):
        self.snapshot=None
        self.key=None
        self.hits=self.misses=0

    def read(self,snapshot,*,now,episode_tag,max_stale_ms):
        timestamp=float(snapshot['captured_at']) if snapshot else None
        if (not snapshot or not math.isfinite(timestamp) or
                snapshot.get('episode')!=episode_tag):
            hazards,age,fresh=encode_hazards(snapshot,now=now,episode_tag=episode_tag,max_stale_ms=max_stale_ms)
            return hazards,age,fresh,screen_columns(snapshot,now=now,episode_tag=episode_tag,max_stale_ms=max_stale_ms)
        key=(timestamp,episode_tag)
        if self.snapshot is not snapshot or self.key!=key:
            self.snapshot,self.key=snapshot,key
            self.hazards=encode_hazards(snapshot,now=timestamp,episode_tag=episode_tag,max_stale_ms=max_stale_ms)[0]
            self.columns=screen_columns(snapshot,now=timestamp,episode_tag=episode_tag,max_stale_ms=max_stale_ms)
            self.misses+=1
        else:self.hits+=1
        age=max(0.,now-timestamp)*1000
        fresh=age<=max_stale_ms
        columns_valid=fresh and now-timestamp>=-.005
        return (self.hazards.copy() if fresh else np.zeros_like(self.hazards),age,fresh,
                self.columns.copy() if columns_valid else np.zeros_like(self.columns))
