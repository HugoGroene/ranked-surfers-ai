"""Finish a nearly corroborated death transition; can never return PLAY."""
from rs_v612_gameplay_guard import sequence_decision


def pending_menu_tail(readings):
    tail=readings[-5:]
    if (len(readings)<6 or len(tail)!=5 or
            not all(r.ml_menu and not r.visual_gameplay for r in tail)):
        return False
    strong=sum(r.menu_evidence for r in tail)
    corroborated=sum(r.ml_template_corroborated for r in tail)
    # Keep the existing route. Episode 0112 also had five confident MENU
    # readings and two strong template hits, but two animation dips left
    # only three moderate corroborations. Let that candidate request fresh
    # evidence only when BOTH final frames have template support again.
    # This predicate never confirms death: confirm_menu_tail still requires
    # a new capture and the unchanged sequence_decision terminal rules.
    return ((strong>=1 and corroborated>=4) or
            (strong>=2 and corroborated>=3 and
             all(r.ml_template_corroborated for r in tail[-2:])) or
            # One further independent moderate vote can complete the existing
            # four-consecutive rule after a menu animation dip. Repeated
            # moderate template evidence is required even without a strong hit.
            (corroborated>=4 and
             all(r.ml_template_corroborated for r in tail[-3:])))


async def confirm_menu_tail(readings,sample,wait):
    frame=None
    if not pending_menu_tail(readings):
        return frame,'ABORT'
    for _ in range(6):
        await wait(.045)
        frame,reading=await sample()
        readings.append(reading)
        if not reading.ml_menu or reading.visual_gameplay:
            return frame,'ABORT'
        if sequence_decision(readings[-6:],max_frames=6)=='TERMINAL':
            return frame,'TERMINAL'
    return frame,'ABORT'
