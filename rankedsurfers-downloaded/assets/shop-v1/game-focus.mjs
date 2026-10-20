// Focus the Unity canvas after a real transition. Do not steal focus from a modal or chat.
export function resumeGame(options={}){
 let cancelled=false,first=0,second=0;
 const focus=()=>{
  if(cancelled||document.hidden||document.querySelector('dialog[open],[role="dialog"][aria-modal="true"]'))return;
  const active=document.activeElement;
  if(options.defer&&active?.matches('input,textarea,[contenteditable="true"]'))return;
  const frame=document.querySelector('iframe[title="Ranked Surfers game"]');
  if(!frame)return;
  active?.blur?.();frame.focus({preventScroll:true});
  frame.contentWindow?.postMessage({source:'ranked-site',type:'focus-game'},location.origin);
 };
 first=requestAnimationFrame(()=>{second=requestAnimationFrame(focus);});
 return()=>{cancelled=true;cancelAnimationFrame(first);cancelAnimationFrame(second);};
}
