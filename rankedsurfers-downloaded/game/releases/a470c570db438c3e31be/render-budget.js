// Optional visual-only adaptation. Simulation speed, inputs and browser frame
// scheduling are untouched; lower the canvas workload after sustained stalls.
export class RenderBudget {
  constructor(){this.enabled=false;this.ceiling=1;this.scale=1;this.reset();}
  reset(){this.last=0;this.elapsed=0;this.frames=0;this.good=0;this.settle=2000;}
  configure(enabled,ceiling=1){
    const max=Number.isFinite(ceiling)?Math.max(.5,Math.min(1.5,ceiling)):1;
    if(this.enabled===enabled&&this.ceiling===max)return;
    this.enabled=enabled;this.ceiling=max;this.scale=max;this.reset();
  }
  sample(now,visible=true){
    if(!this.enabled||!visible){this.reset();return this.scale;}
    if(!this.last){this.last=now;return this.scale;}
    const delta=now-this.last;this.last=now;
    if(delta<=0||delta>250){this.elapsed=0;this.frames=0;this.good=0;return this.scale;}
    if(this.settle>0){this.settle-=delta;return this.scale;}
    this.elapsed+=delta;this.frames++;
    if(this.elapsed<2000)return this.scale;
    const ms=this.elapsed/this.frames;this.elapsed=0;this.frames=0;
    if(ms>22){this.scale=Math.max(.5,Math.round((this.scale-.1)*100)/100);this.good=0;}
    else if(ms<18){if(++this.good>=4){this.scale=Math.min(this.ceiling,Math.round((this.scale+.05)*100)/100);this.good=0;}}
    else this.good=0;
    return this.scale;
  }
}
