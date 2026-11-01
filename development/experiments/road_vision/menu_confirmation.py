"""Experiment-only eligibility for more movement-free MENU captures.

Never accepts a terminal or permits movement. The production six-capture tail
confirmation and all terminal/freshness thresholds remain authoritative.
"""
def candidate(readings):
    window=readings[-6:]
    return (len(window)==6 and
            all(r.ml_menu and not r.visual_gameplay for r in window) and
            any(r.menu_evidence for r in window) and
            all(r.ml_template_corroborated for r in window[-3:]))

def install():
    import tools.ppo_stream_menu_tail as tail
    original=tail.pending_menu_tail
    if getattr(original,'_road_confirmation',False):return
    def pending(readings):return original(readings) or candidate(readings)
    pending._road_confirmation=True
    tail.pending_menu_tail=pending
