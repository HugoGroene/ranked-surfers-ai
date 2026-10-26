#!/usr/bin/env python3
"""No-browser/no-GPU synthetic tests for V6 nine-class features and reward logic.

This is not the SB3 PPO gradient check; train_rs_v6_ppo.py --mode check runs that.
"""
import numpy as np
from rs_v6_ppo_features import (AUX_DIM, FEATURE_VERSION, HAZARD_DIM,
                                IMAGE_BANDS, SLOTS_PER_BAND, VALUES_PER_SLOT,
                                V5_CLASSES, encode_aux, encode_hazards)
from rs_v6_rewards import SurvivalRewards


def main():
    objs=[]
    for ci, name in enumerate(V5_CLASSES):
        for band_index, y2 in enumerate((.39,.60,.82)):
            for i in range(3):
                bottom = y2 - i*.01
                x1 = .08 + i*.09
                obj = {'class':name,'conf':.7 + i*.04,
                       'box':[x1,.05,x1+.09,bottom],
                       'polygon':[[x1,.05],[x1+.09,.05],
                                  [x1+.09,bottom],[x1,bottom]]}
                objs.append(obj)
    snap={'episode':7,'captured_at':500.,'detections':objs}
    hazard, age, fresh = encode_hazards(snap, now=500.065, episode_tag=7)
    assert fresh and 64. < age < 66.
    assert hazard.shape == (HAZARD_DIM,)
    assert float(hazard[-1]) == 1
    for ci in range(len(V5_CLASSES)):
        for band_idx in range(len(IMAGE_BANDS)):
            offset=((ci * len(IMAGE_BANDS) + band_idx)*SLOTS_PER_BAND*VALUES_PER_SLOT)
            one=hazard[offset:offset+VALUES_PER_SLOT]
            two=hazard[offset+VALUES_PER_SLOT:offset+2*VALUES_PER_SLOT]
            assert one[-1] == 1 and two[-1] == 1
            assert one[3] >= two[3]
            assert one[5] > 0 and one[6] > 0
    # Three raw detections per (class,image-band) -> 9 per class.
    count_offset=len(V5_CLASSES)*len(IMAGE_BANDS)*SLOTS_PER_BAND*VALUES_PER_SLOT
    np.testing.assert_allclose(hazard[count_offset:-1],np.full(len(V5_CLASSES),.75))
    aux=encode_aux(hazard,[1,3,4],1.4,[.03,.05],age)
    assert aux.shape==(AUX_DIM,) and np.isfinite(aux).all()
    assert aux.min()>=0 and aux.max()<=1
    stale, stale_age, valid=encode_hazards(snap,now=500.301,episode_tag=7,max_stale_ms=300)
    assert not valid and not stale.any() and stale_age > 300
    wrong,_,valid=encode_hazards(snap,now=500.050,episode_tag=9)
    assert not valid and not wrong.any()
    none,_,valid=encode_hazards(None,now=500.)
    assert not valid and not none.any()
    # No detection means a missing representation, not a safe path.
    missing_aux=encode_aux(none,[],0.,[],None)
    assert missing_aux[-1]==1 and missing_aux[HAZARD_DIM-1]==0

    reward=SurvivalRewards(per_second=.4,milestone_bonus=.5,
                           milestone_seconds=2.,death_penalty=1.2)
    r,p=reward.step(.05,False)
    assert abs(r-.02)<1e-5 and p['death']==0
    r2,p=reward.step(.05,False)
    assert r2==0
    r3,p=reward.step(2.04,False)
    assert abs(p['milestones']-.5)<1e-5 and p['survival']<=.060001
    r4,p=reward.step(2.06,True)
    assert abs(p['death']+1.2)<1e-5
    reward.reset()
    r5,p=reward.step(.05,False)
    assert abs(r5-.02)<1e-5 and p['milestones']==0
    print(f'[SYNTHETIC CHECK] PASS | schema={FEATURE_VERSION} '
          f'| hazard_dim={HAZARD_DIM} | aux_dim={AUX_DIM} '
          '| freshness, episode gating, nine classes, action history, rewards', flush=True)

if __name__=='__main__':
    main()
