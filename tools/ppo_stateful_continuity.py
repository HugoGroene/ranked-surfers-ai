"""Short evidence-backed gameplay continuity; no reward or terminal decisions."""
import math

class Continuity:
    def __init__(self):
        self.until=-1.
        self.last=-1.
        self.accepted=0

    def observe(self, *, now, fresh, focused, paused, state, score, threshold,
                hud_present, advancing=False, confident=False):
        valid=all(math.isfinite(x) for x in (now,score,threshold))
        if (not valid or not fresh or not focused or paused or now<self.last or
            state in ('MENU','LOADING','OTHER') or score>=threshold*.30 or not hud_present):
            self.until=-1.;self.last=now
            return False
        self.last=now
        if confident or advancing:self.until=now+.75
        allowed=now<=self.until
        if allowed and not confident:self.accepted+=1
        return allowed

async def verify(raw,state,probs,score,*,resync=False,views=None):
    import time
    import cv2
    import numpy as np
    if not getattr(raw,'_stateful_episode_verified',False):return False
    if not hasattr(raw,'_stateful_continuity'):raw._stateful_continuity=Continuity()
    now=time.perf_counter()
    # Confident gameplay already follows the existing guard path. Cache its
    # already-computed image evidence, but never add browser RPCs to that path.
    # Returning False here means no override of the original guard verdict.
    if state!='UNCERTAIN':
        if (state=='GAMEPLAY' and probs.get('GAMEPLAY',0)>=raw.state_confidence
                and math.isfinite(score) and 0<=score<raw.args.threshold*.30
                and 0<=now-float(raw.last_capture_end)<.1):
            # Resync has already checked focus and the DOM/rendered pause panel.
            # Unlike normal steps it has no precomputed HUD/track views. Seed
            # evidence here too, so a later ambiguous frame can use that proof.
            if views is None and resync:
                views=(raw._meter_mask(raw._v2_last_bgra),
                       raw._motion_view(raw._v2_last_bgra,raw.crop))
            if views is not None:
                hud,track=views
                if np.count_nonzero(hud)>=6:
                    raw._stateful_continuity.until=now+.75
                    raw._stateful_continuity.last=now
                    previous=getattr(raw,'_stateful_evidence',None)
                    if previous is None or now-previous[0]>.5:
                        raw._stateful_evidence=(now,hud.copy(),track.copy())
        else:
            raw._stateful_continuity=Continuity()
            raw._stateful_evidence=None
        return False
    frame=raw._v2_last_bgra
    hud,track=views if views is not None else (raw._meter_mask(frame),raw._motion_view(frame,raw.crop))
    previous=getattr(raw,'_stateful_evidence',None)
    advancing=False
    if previous is not None and 0<now-previous[0]<.75:
        advancing=(np.count_nonzero(cv2.absdiff(hud,previous[1]))>=3 and
                   float(np.mean(cv2.absdiff(track,previous[2])))>=raw.startup_motion_threshold and
                   float(track.std())>=12.)
    # Keep a baseline long enough for distance digits to change.
    if previous is None or advancing or now-previous[0]>.5:
        raw._stateful_evidence=(now,hud.copy(),track.copy())
    focused=bool(await raw.page.evaluate('document.hasFocus() && document.visibilityState === "visible"'))
    paused=bool(await raw.page.locator('.game-pause-panel[aria-label="Game paused"]').is_visible())
    # RPCs take time: freshness must be checked AFTER they complete.
    now=time.perf_counter()
    age=now-float(raw.last_capture_end)
    allowed=raw._stateful_continuity.observe(now=now,fresh=0<=age<.1,focused=focused,
        paused=paused,state=state,score=float(score),threshold=float(raw.args.threshold),
        hud_present=np.count_nonzero(hud)>=6,advancing=advancing,
        confident=state=='GAMEPLAY' and probs.get('GAMEPLAY',0)>=raw.state_confidence)
    if allowed and state=='UNCERTAIN':
        count=raw._stateful_continuity.accepted
        if count<=10 or count%100==0:
            print(f'[STATEFUL GUARD] Continued verified episode; count={count}; advancing={advancing}; resync={resync}',flush=True)
    return allowed
