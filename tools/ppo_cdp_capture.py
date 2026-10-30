"""Direct Chromium surface capture with a validated, cached canvas clip."""
import asyncio
import base64
from rs_v612_gameplay_guard import GuardUncertainError


class CdpCanvasCapture:
    def __init__(self,session,clip):
        self.session,self.clip=session,dict(clip,scale=1)

    @classmethod
    async def create(cls,session,canvas,width,height):
        box=await canvas.bounding_box()
        if box is None or any(abs(box[key]-value)>.5 for key,value in
            [('x',0),('y',0),('width',width),('height',height)]):
            raise GuardUncertainError('Direct capture requires the verified full-viewport canvas')
        return cls(session,box)

    async def capture(self):
        try:
            response=await asyncio.wait_for(self.session.send('Page.captureScreenshot',dict(
                format='jpeg',quality=90,clip=self.clip,fromSurface=True,
                captureBeyondViewport=False,optimizeForSpeed=True)),timeout=5)
            data=base64.b64decode(response['data'],validate=True)
            if not data:
                raise ValueError('Empty image')
            return data
        except (TimeoutError,asyncio.TimeoutError,KeyError,ValueError,TypeError) as exc:
            raise GuardUncertainError(f'Direct canvas capture failed: {exc}') from exc
