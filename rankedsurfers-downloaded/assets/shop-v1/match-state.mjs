// Shared state selectors; real match clocks and callbacks remain owned by the site.
export function matchPhase(m,now){return m.status==='running'?(now>=m.started?'live':'between'):m.status;}
export function completedRounds(m){return (m.rounds||[]).filter(r=>(m.status!=='running'||r.id!==m.id)&&(r.winner!==undefined||r.status==='complete'||r.status==='void'));}
export function playerDistance(round,id){return round?.results?.[id]?.meters??round?.meters?.[id];}
export function chatAllowed(m,now){return m.status==='running'?(now<m.started||m.liveChat===true):['complete','void'].includes(m.status)&&(!m.finished||now<m.finished+86400000);}
