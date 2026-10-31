"""One exact-capture classification cache; never an independent-frame vote."""
def install_frame_state_cache(raw):
    original=raw._read_screen_state
    cache=[None]
    raw.frame_state_cache_hits=raw.frame_state_cache_misses=0
    def read():
        image=getattr(raw,'_v2_last_bgra',None)
        stamp=getattr(raw,'last_capture_end',None)
        classifier=getattr(raw,'screen_classifier',None)
        confidence=getattr(raw,'state_confidence',None)
        serial=getattr(raw,'_learning_capture_serial',None)
        last=cache[0]
        if (image is not None and stamp is not None and last is not None and
            image is last[0] and stamp==last[1] and classifier is last[2] and
            confidence==last[3] and serial==last[4]):
            raw.frame_state_cache_hits+=1
            return last[5],dict(last[6])
        state,probs=original()
        raw.frame_state_cache_misses+=1
        cache[0]=(image,stamp,classifier,confidence,serial,state,dict(probs))
        return state,dict(probs)
    raw._read_screen_state=read
