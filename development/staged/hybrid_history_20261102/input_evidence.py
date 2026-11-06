"""Bounded copies of actual actor rollout observations; disk export after gameplay."""
import json
from pathlib import Path
import numpy as np


class InputEvidence:
    def __init__(self,frames,age_scale):
        self.frames=frames;self.age_scale=age_scale;self.samples=[];self.rollouts=0

    def collect(self,buffer):
        if len(self.samples)>=6:return
        self.rollouts+=1
        count=buffer.buffer_size if buffer.full else buffer.pos
        # The settled callback may run before its final transition is inserted.
        # Only inspect populated rows that actually trained the actor.
        actors=np.flatnonzero(buffer.actor_masks[:count,0]>0)
        if not len(actors):return
        valid=buffer.observations['aux'][actors,0,-self.frames:].sum(axis=1)
        choices=list(dict.fromkeys([int(actors[0]),int(actors[np.argmax(valid)]),
                                    int(actors[len(actors)//2]),int(actors[-1])]))[:3]
        for index in choices:
            if len(self.samples)>=6:break
            self.samples.append(dict(image=buffer.observations['image'][index,0].copy(),
                aux=buffer.observations['aux'][index,0].copy(),rollout=self.rollouts,index=index,
                action=int(buffer.actions[index,0].reshape(-1)[0]),
                episode_start=bool(buffer.episode_starts[index,0])))

    def write(self,folder):
        from PIL import Image,ImageDraw
        folder=Path(folder);folder.mkdir(exist_ok=False)
        entries=[]
        for number,sample in enumerate(self.samples,1):
            name=f'{number:02d}_rollout{sample["rollout"]}_index{sample["index"]}'
            image=sample['image'];aux=sample['aux'];n=self.frames
            if image.shape[0]!=3*n:raise ValueError('Preview image contract mismatch')
            if not np.isfinite(image).all() or (image<0).any() or (image>255).any() or not np.equal(image,np.floor(image)).all():
                raise ValueError('Expected raw uint8-valued policy pixels')
            np.savez_compressed(folder/f'{name}.npz',image=image,aux=aux)
            ages=aux[-2*n:-n];valid=aux[-n:]
            meta={k:v for k,v in sample.items() if k not in ('image','aux')}
            meta.update(name=name,image_shape=list(image.shape),dtype=str(image.dtype),
                aux_shape=list(aux.shape),age_scale_seconds=self.age_scale,
                slot_ages_ms=[float(ages[i]*self.age_scale*1000) if valid[i] else None for i in range(n)],
                age_clipped=[bool(valid[i] and ages[i]>=1) for i in range(n)],
                valid=valid.tolist(),ordering='oldest to newest',
                source='Exact stored actor rollout observation before CNN normalization; includes all auxiliary inputs in NPZ')
            h,w=image.shape[1:];cols=min(n,4);rows=(n+cols-1)//cols
            sheet=Image.new('RGB',(cols*w,rows*(h+40)),(30,30,30));draw=ImageDraw.Draw(sheet)
            for i in range(n):
                rgb=Image.fromarray(image[i*3:(i+1)*3].transpose(1,2,0).astype(np.uint8))
                rgb.save(folder/f'{name}_slot{i+1:02d}.png')
                x=(i%cols)*w;y=(i//cols)*(h+40)
                sheet.paste(rgb,(x,y+40))
                label=(f'{meta["slot_ages_ms"][i]:.1f} ms old' if valid[i] else 'MISSING (invalid)')
                draw.text((x+4,y+4),f'Slot {i+1}: {label}',fill='white')
            sheet.save(folder/f'{name}_sheet.png')
            (folder/f'{name}.json').write_text(json.dumps(meta,indent=2))
            entries.append(meta)
        (folder/'index.json').write_text(json.dumps(dict(samples=entries,
            note='Smoke-only evidence, no additional captures. Raw PNGs unchanged; labels appear only outside contact-sheet pixels. Captures can include startup and mature history; inspect validity flags.'),indent=2))
