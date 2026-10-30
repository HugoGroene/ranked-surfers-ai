"""Finish a nearly corroborated death transition; can never return PLAY."""
from rs_v612_gameplay_guard import sequence_decision


def pending_menu_tail(readings):
    tail=readings[-5:]
    return (len(readings)>=6 and len(tail)==5 and
            all(r.ml_menu and not r.visual_gameplay for r in tail) and
            sum(r.menu_evidence for r in tail)>=1 and
            sum(r.ml_template_corroborated for r in tail)>=4)


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
