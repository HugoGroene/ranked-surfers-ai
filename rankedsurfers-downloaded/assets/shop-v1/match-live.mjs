// Bind Claude's original preview screens to real match state/actions.
import {r as React,j as jsx} from '/assets/client-Pp6hiLUC.js';
import {MatchFound,RoundResult,MatchResult,ChatBar,useChatMute} from './screens.mjs';
import {matchPhase,completedRounds,playerDistance,chatAllowed} from './match-state.mjs';
import {readEquipment} from './rs-shop.mjs';
import {cosmeticBack} from './cosmetic-slot.mjs';
import {resumeGame} from './game-focus.mjs';
const h=jsx.jsx;
export function MatchExperience(props){
 const {match:m,me,now=Date.now(),busy,gameLoaded,ready,decline,skipWait,onChat,onDismiss,onReplay,onQueue,renderName}=props;
 const phase=matchPhase(m,now);
 const [pending,setPending]=React.useState(false),[error,setError]=React.useState(''),inFlight=React.useRef(false);
 const [muted,toggleMute]=useChatMute();
 React.useEffect(()=>{setError('');},[m.id,phase]);
 React.useEffect(()=>{document.body.classList.toggle('rs-round-playing',phase==='live');return()=>document.body.classList.remove('rs-round-playing');},[phase]);
 React.useEffect(()=>{if(phase==='live'&&gameLoaded&&!props.localFinished)return resumeGame({defer:true});},[m.id,phase,gameLoaded,props.localFinished]);
 // No score, chat button, HUD, or match overlay while the round is running.
 if(phase==='live'||phase==='cancelled'||!m.players?.some(p=>p.id===me?.id)||m.players.length!==2)return null;
 const act=fn=>async(...args)=>{if(!fn||inFlight.current)return;inFlight.current=true;setPending(true);setError('');try{await fn(...args);}catch(e){setError(e.message||'Please try again.');}finally{inFlight.current=false;setPending(false);}};
 const rounds=completedRounds(m).map(r=>({...r,meters:Object.fromEntries(m.players.map(p=>[p.id,playerDistance(r,p.id)??0]))}));
 const match={...m,players:m.players.map(p=>p.id===me.id?{...p,skin:props.localSkin??p.skin,cosmetics:{back:cosmeticBack(readEquipment())}}:p),ready:m.ready||[],deadline:m.ready_deadline,starts:m.started,rounds,delta:m.deltas?.[me.id]??0};
 const chat={};if(!muted)for(const msg of m.chat||[])chat[msg.player_id||msg.playerId]=msg;
 const chatBar=chatAllowed(m,now)&&onChat?h(ChatBar,{onSend:act(text=>onChat({id:crypto.randomUUID(),text})),muted,onMute:toggleMute,disabled:pending,error}):error?h('p',{role:'alert',children:error}):null;
 const shared={match,me:me.id,chat,chatBar,renderName,now,busy:busy||pending};
 if(phase==='ready'||phase==='between'&&!rounds.length)return h(MatchFound,{...shared,onAccept:act(ready),onDecline:act(decline),acceptDisabled:!gameLoaded||now>=m.ready_deadline,onSkip:act(skipWait),showSkip:phase==='between'&&!!m.skipAvailable,skipDisabled:!gameLoaded||m.skipVotes?.includes(me.id)});
 if(phase==='between')return h(RoundResult,{...shared,onSkip:act(skipWait),showSkip:!!m.skipAvailable,skipDisabled:!gameLoaded||m.skipVotes?.includes(me.id)});
 if(phase==='complete')return h(MatchResult,{...shared,onQueue:act(onQueue),onReplay:act(onReplay),onClose:act(onDismiss),onReport:props.onReport});
 return h('div',{className:'gx-overlay',role:'dialog','aria-modal':true,'aria-label':'Match ended',children:h('section',{className:'gx-sheet',children:[props.nativeOutcome,h('button',{className:'gx-btn gx-primary',onClick:onDismiss,children:'Back to lobby'})]})});
}
