"""Opt-in exact delivered-frame crop reuse; no capture or guard age changes."""
import json


def matching_crop(raw):
    pump=getattr(raw,'pump',None)
    entry=getattr(pump,'delivered_prepared',None)
    timing=getattr(raw,'_headless_capture_timing',{})
    if entry is None or entry[2] is None:return None
    stamp,serial,prepared=entry
    if stamp!=timing.get('request_started_at') or serial!=timing.get('electron_serial'):return None
    # A newer headless delivery must not stand in for the image packed by v2.
    shot=getattr(raw,'_v2_last_bgra',None)
    if shot is None or shot is not getattr(getattr(raw,'grabber',None),'image',None):return None
    if getattr(raw,'_background_geometry_invalid',None):return None
    if getattr(raw,'_framing_profile',None)!='canvas_v1':return None
    # Let the original path validate geometry and submit initial crop evidence.
    # Optional asynchronous writes are checked/joined at cleanup, not as guards.
    if not getattr(raw,'_canvas_profile_preview_saved',False) or not getattr(raw,'_road_preview_saved',False):return None
    small,native=prepared
    w,h=raw.target_wh
    if small.shape!=(3,h,w) or native.shape!=(945,1920,3):return None
    return prepared


def install(trainer):
    if getattr(trainer,'_pump_crop_reuse_installed',False):return
    trainer._pump_crop_reuse_installed=True
    original=trainer.create_live_env
    def create(*args,**kwargs):
        monitored,raw=original(*args,**kwargs)
        native_frames=raw._native_frames
        counts=dict(hits=0,misses=0)
        raw.crop_reuse_counts=counts
        def frames():
            prepared=matching_crop(raw)
            if prepared is None:
                counts['misses']+=1
                return native_frames()
            counts['hits']+=1
            small,native=prepared
            from tools.ppo_background_display import canvas_from_display
            # Preserve the original geometry checks (this only creates a view).
            canvas_from_display(raw._v2_last_bgra,raw.region,raw._background_canvas)
            if counts['hits']==1:
                print('[CROP REUSE] Exact delivered-frame crop reused; guard geometry and freshness unchanged.',flush=True)
            raw.native_image_wh=(native.shape[1],native.shape[0])
            raw.last_source_native=native
            return small,native
        raw._native_frames=frames
        close=raw._close_browser
        async def finish():
            try:return await close()
            finally:
                report=dict(counts,reset_wall_ms=list(getattr(raw,'reset_times_ms',[])),
                    reset_note='Actual environment reset durations, in order; first reset includes initial readiness. Browser launch precedes reset. Recorded at cleanup, not in the action loop.')
                (raw.output_dir/'crop_reuse.json').write_text(json.dumps(report,indent=2))
        raw._close_browser=finish
        return monitored,raw
    trainer.create_live_env=create
