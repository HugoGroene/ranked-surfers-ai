"""Demand-acknowledged latest-frame stream; no frame FIFO or cached-frame reads."""
import asyncio
import base64
import math
import time
from rs_v612_gameplay_guard import GuardUncertainError


def presentation_time(metadata,received,wall,width,height):
    if (metadata.get('deviceWidth')!=width or metadata.get('deviceHeight')!=height or
        metadata.get('pageScaleFactor')!=1 or metadata.get('offsetTop',0)!=0 or
        metadata.get('scrollOffsetX',0)!=0 or metadata.get('scrollOffsetY',0)!=0):
        raise GuardUncertainError('Stream viewport geometry changed')
    epoch=metadata.get('timestamp')
    if not isinstance(epoch,(float,int)) or not math.isfinite(epoch):
        raise GuardUncertainError('Stream lacks a valid frame timestamp')
    age=wall-epoch
    if not -.003 <= age <= .5:
        raise GuardUncertainError('Stream frame clock is invalid or too old')
    converted=received-max(0.,age)
    mono=metadata.get('monotonicTimestamp')
    # Use shared monotonic ticks only when independently consistent with epoch.
    if isinstance(mono,(int,float)) and math.isfinite(mono) and abs(mono-converted)<.02 and mono<=received:
        return float(mono)
    return converted


class LatestFrameStream:
    def __init__(self,session,fallback,width,height):
        self.session,self.fallback,self.width,self.height=session,fallback,width,height
        self.event=asyncio.Event()
        self.latest=None
        self.pending=[]
        self.serial=self.used=0
        self.error=None
        self.last_timing={}
        self.stream_reads=self.fallback_reads=0

    def on_frame(self,event):
        try:
            now=time.perf_counter()
            presented=presentation_time(event['metadata'],now,time.time(),self.width,self.height)
            self.serial+=1
            self.latest=(self.serial,presented,event['data'])
            self.pending.append(event['sessionId'])
            if len(self.pending)>8:
                raise GuardUncertainError('Unexpected unbounded stream acknowledgements')
        except (KeyError,TypeError,GuardUncertainError) as exc:
            self.error=GuardUncertainError(f'Stream frame rejected: {exc}')
        self.event.set()

    async def start(self):
        self.session.on('Page.screencastFrame',self.on_frame)
        await self.session.send('Page.startScreencast',dict(format='jpeg',quality=90,
            maxWidth=self.width,maxHeight=self.height,everyNthFrame=1,maxFramesInFlight=1))

    async def acknowledge(self):
        pending,self.pending=self.pending,[]
        for session_id in pending:
            await self.session.send('Page.screencastFrameAck',dict(sessionId=session_id))

    async def capture(self):
        began=time.perf_counter()
        deadline=began+.15
        while True:
            if self.error:
                raise self.error
            self.event.clear()
            if self.latest is not None:
                serial,presented,data=self.latest
                if serial>self.used and presented>=began and 0<=time.perf_counter()-presented<=.05:
                    try:
                        decoded=base64.b64decode(data,validate=True)
                    except (ValueError,TypeError) as exc:
                        raise GuardUncertainError('Invalid stream image') from exc
                    if not decoded:
                        raise GuardUncertainError('Empty stream image')
                    self.used=serial
                    # Permit at most one frame ahead while inference/inputs run;
                    # the next read still rejects swaps older than its request.
                    await self.acknowledge()
                    if time.perf_counter()-presented>.05:
                        continue
                    self.stream_reads+=1
                    self.last_timing=dict(browser_frame_at=presented,stream_serial=serial,capture_kind='stream')
                    return decoded
            await self.acknowledge()
            remaining=deadline-time.perf_counter()
            if remaining<=0:
                break
            try:
                await asyncio.wait_for(self.event.wait(),timeout=remaining)
            except asyncio.TimeoutError:
                break
        # A paused/static UI need not produce a new swap. Perform an independent
        # surface read instead of reusing cached stream pixels. All guards remain.
        self.fallback_reads+=1
        self.last_timing=dict(capture_kind='independent_surface_fallback')
        return await self.fallback.capture()

    async def close(self):
        self.session.remove_listener('Page.screencastFrame',self.on_frame)
        await self.session.send('Page.stopScreencast')
