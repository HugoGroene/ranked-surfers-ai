"""Finish a nearly corroborated death transition; can never return PLAY."""
from rs_v612_gameplay_guard import sequence_decision


def _menu_suffix(readings):
    """Evidence after the most recent conflicting state, capped at six frames."""
    tail=[]
    for reading in reversed(readings[-6:]):
        if reading.state!='MENU' or not reading.ml_menu or reading.visual_gameplay:
            break
        tail.append(reading)
    return list(reversed(tail))


def pending_menu_tail(readings):
    # A transition can start partway through the original capture budget.
    # Count only the uninterrupted MENU suffix, never evidence across gameplay.
    # This admits observation only: a fresh capture must still satisfy the
    # unchanged terminal rules. One final animation dip is permitted.
    tail=_menu_suffix(readings)
    if len(readings)<6 or len(tail)<4:
        return False
    strong=sum(r.menu_evidence for r in tail)
    corroborated=sum(r.ml_template_corroborated for r in tail)
    return (any(r.ml_template_corroborated for r in tail[-2:]) and
            ((strong>=1 and corroborated>=3) or corroborated>=4))


async def confirm_menu_tail(readings,sample,wait):
    frame=None
    if not pending_menu_tail(readings):
        return frame,'ABORT'
    evidence=_menu_suffix(readings)
    for _ in range(6):
        await wait(.045)
        frame,reading=await sample()
        readings.append(reading)
        if reading.state!='MENU' or not reading.ml_menu or reading.visual_gameplay:
            return frame,'ABORT'
        evidence.append(reading)
        if sequence_decision(evidence[-6:],max_frames=6)=='TERMINAL':
            return frame,'TERMINAL'
    return frame,'ABORT'
