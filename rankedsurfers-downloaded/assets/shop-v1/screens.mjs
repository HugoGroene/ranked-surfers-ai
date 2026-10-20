import {avatarEquipment} from './cosmetic-slot.mjs';
// Match + rank-up screens in the liquid-glass style (local preview). Designs follow the owner's picks: characters face
// the camera wearing their gear and don't spin, no cosmetic labels, round dots instead of R1/R2 chips, "Find match"
// instead of "Rematch" (no win trading). The overlay is the glass layer; what's on it is plain content.
import { r as React, j as jsx } from '/assets/client-Pp6hiLUC.js';
const mountAvatar = (...a) => import('./products.mjs').then((m) => m.mountAvatar(...a));
const h = jsx.jsx, hs = jsx.jsxs, fmt = (n) => Math.round(n).toLocaleString('en-US');
const RANKS = [{ name: 'Bronze', min: 0 }, { name: 'Silver', min: 900 }, { name: 'Gold', min: 1100 }, { name: 'Diamond', min: 1400 }, { name: 'Champion', min: 1800 }, { name: 'Grand Champion', min: 2000 }];
const rankOf = (r) => { let i = 0; RANKS.forEach((x, k) => { if (r >= x.min) i = k; }); return { ...RANKS[i], next: RANKS[i + 1] }; };
export const badge = (rank, top) => top ? '/ranks/number-one-v1.webp' : '/ranks/' + (rank === 'Grand Champion' ? 'grand-champion-ruby-v1.webp' : rank + '.png');
const bigBadge = (rank, top) => rank === 'Champion' && !top ? '/ranks/Champion-hd.png' : badge(rank, top);
const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

function Character({ p, side, dim, mystery }) {
  const ref = React.useRef(null);
  React.useEffect(() => { let ctl, dead = false; const g = p.cosmetics || {}; if(!Number.isInteger(p.skin))return;
    mountAvatar(ref.current, { skin: p.skin, ...avatarEquipment(g),
      interactive: false, start: side === 'left' ? 0.35 : -0.35, zoom: 1.02, shadow: true }).then((c) => dead ? c.dispose() : (ctl = c)).catch(() => {});
    return () => { dead = true; ctl?.dispose(); }; }, [p.skin, side, p.cosmetics?.back, p.cosmetics?.cape, p.cosmetics?.wing]);
  if(!Number.isInteger(p.skin))return h('img',{className:'gx-rank-fallback',src:badge(p.rank),alt:p.rank+' rank'});
  return h('canvas', { ref, className: 'gx-char' + (dim ? ' dim' : '') + (mystery ? ' mystery' : ''), 'aria-hidden': true });
}
// Chat shows as speech bubbles above whoever said it (newest only, fades after 6 s), plus a slim bar to reply.
function Bubble({ msg }) {
  const [shown, setShown] = React.useState(msg);
  React.useEffect(() => { setShown(msg); if (!msg) return; const t = setTimeout(() => setShown(null), 6000); return () => clearTimeout(t); }, [msg?.id]);
  return shown ? h('span', { className: 'gx-bubble', role: 'status', children: shown.text }, shown.id) : null;
}
const ICON = { chat: 'M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12Z', off: 'M4 4l16 16M9.5 5.4A8 8 0 0 1 20.4 15M17.6 18.3A8 8 0 0 1 9.4 19.1L4 20l1-4.6A8 8 0 0 1 6 7.4', send: 'M5 12h14M13 6l6 6-6 6' };
const Svg = ({ d }) => h('svg', { viewBox: '0 0 24 24', width: 18, height: 18, fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true, children: h('path', { d }) });
// Mute is remembered on this device until it is turned off again.
export function useChatMute() {
  const [muted, set] = React.useState(() => { try { return localStorage.getItem('rs-chat-muted') === '1'; } catch { return false; } });
  const toggle = () => set((m) => { try { localStorage.setItem('rs-chat-muted', m ? '0' : '1'); } catch {} return !m; });
  return [muted, toggle];
}
export function ChatBar({ onSend, muted, onMute, disabled = false, error }) {
  const [text, setText] = React.useState('');
  const send = (t) => { t = t.trim(); if (!t || disabled) return; onSend(t); setText(''); };
  return hs('form', { className: 'gx-chatbar' + (muted ? ' is-muted' : ''), onSubmit: (e) => { e.preventDefault(); send(text); }, children: [
    h('button', { type: 'button', className: 'gx-mute', 'aria-pressed': muted, title: muted ? 'Chat muted: tap to turn chat back on' : 'Mute chat', 'aria-label': muted ? 'Unmute chat' : 'Mute chat', onClick: onMute, children: h(Svg, { d: muted ? ICON.off : ICON.chat }) }),
    h('input', { value: text, maxLength: 240, placeholder: muted ? 'Chat muted' : 'Say something…', disabled: muted, 'aria-label': 'Message your opponent', 'aria-invalid': !!error, title: error || undefined, onChange: (e) => setText(e.target.value), onKeyDown: (e) => e.stopPropagation(), onKeyUp: (e) => e.stopPropagation() }),
    text.trim() ? h('button', { type: 'submit', className: 'gx-send', 'aria-label': 'Send', disabled, children: h(Svg, { d: ICON.send }) })
      : !muted && h('div', { className: 'gx-quicks', children: ['gg', 'gr', 'gl', 'wp'].map((q) => h('button', { type: 'button', className: 'gx-quick', disabled, onClick: () => send(q), children: q }, q)) })] });
}
export function Player({ p, side, dim, say, hidden, renderName, children }) {
  return hs('div', { className: 'gx-player gx-' + side + (dim ? ' is-dim' : '') + (hidden ? ' is-hidden' : ''), children: [
    hs('div', { className: 'gx-stage', children: [hidden ? hs(React.Fragment, { children: [h(Character, { p: { skin: 0 }, side, mystery: true }), h('span', { className: 'gx-mystery', 'aria-label': 'Opponent hidden until you accept', children: '?' })] }) : h(Character, { p, side, dim }), !hidden && h(Bubble, { msg: say })] }),
    h('strong', { className: 'gx-name', children: hidden ? '???' : renderName ? renderName(p) : p.name }),
    hs('span', { className: 'gx-rank', children: hidden ? [h('img', { src: '/ranks/anonymous.png', alt: '' }), '??? · ???? Elo'] : [h('img', { src: badge(p.rank), alt: '' }), p.rank + ' · ' + fmt(p.rating) + ' Elo'] }),
    children] });
}
const Dots = ({ rounds, me }) => h('ol', { className: 'gx-dots', 'aria-label': 'Rounds', children: rounds.map((r, i) => h('li', { className: r.winner === me ? 'w' : r.winner ? 'l' : 'd', title: 'Round ' + (i + 1) + ' · ' + fmt(r.meters[me]) + ' m – ' + fmt(Object.entries(r.meters).find(([k]) => k !== me)[1]) + ' m' }, i)) });
const useClock = (on = true) => { const [now, set] = React.useState(Date.now()); React.useEffect(() => { if (!on) return; const t = setInterval(() => set(Date.now()), 100); return () => clearInterval(t); }, [on]); return now; };
const Overlay = ({ tone, label, children }) => h('div', { className: 'gx-overlay ' + (tone || ''), role: 'dialog', 'aria-modal': true, 'aria-label': label, children: h('section', { className: 'gx-sheet', children }) });

// ---- match found → accept → 3-2-1 ----
export function MatchFound({ match, me, onAccept, onDone, onDecline = onDone, onSkip, showSkip = false, skipDisabled = false, chat = {}, chatBar, now: serverNow, busy = false, acceptDisabled = false, renderName }) {
  const clock = useClock(serverNow === undefined), now = serverNow ?? clock, you = match.players.find((p) => p.id === me), opp = match.players.find((p) => p.id !== me);
  const accepted = match.ready.includes(me), both = match.ready.length === 2, left = Math.max(0, Math.ceil((match.deadline - now) / 1000));
  const count = match.starts ? Math.ceil((match.starts - now) / 1000) : null;
  React.useEffect(() => { if (count !== null && count <= -1) onDone?.(); }, [count]);
  const chip = (p) => h('span', { className: 'gx-ready' + (match.ready.includes(p.id) ? ' on' : ''), children: match.ready.includes(p.id) ? 'Ready' : 'Accepting…' });
  return h(Overlay, { label: 'Match found', children: [
    hs('header', { className: 'gx-head', children: [h('span', { className: 'gx-sub', children: 'Ranked duel · first to ' + match.winsRequired }), h('h2', { className: 'gx-title', children: both ? 'Get ready' : 'Match found' })] }, 'h'),
    hs('div', { className: 'gx-duel', children: [h(Player, { renderName, p: you, side: 'left', say: accepted ? chat[you.id] : null, children: chip(you) }), h('span', { className: 'gx-vs', 'aria-hidden': true, children: 'VS' }), h(Player, { renderName, p: opp, side: 'right', hidden: !accepted, say: accepted ? chat[opp.id] : null, children: chip(opp) }, accepted ? 'shown' : 'hidden')] }, 'd'),
    h('footer', { className: 'gx-foot', children: both ? hs(React.Fragment, { children: [h('span', { className: 'gx-wait', children: 'Round 1 starts in' }), h('b', { className: 'gx-count' + (count <= 0 ? ' go' : ''), children: count > 0 ? count : 'Go' }, count), count > 0 && showSkip && h('button', { className: 'gx-btn gx-primary', onClick: onSkip, disabled: busy || skipDisabled, children: 'Skip' })] })
      : accepted ? h('span', { className: 'gx-wait', children: 'Waiting for ' + opp.name + '…' })
      : hs(React.Fragment, { children: [
          hs('div', { className: 'gx-accept-time', children: [h('span', { children: 'Accept in ' + left + 's' }), h('i', { children: h('b', { style: { width: Math.max(0, Math.min(100, (match.deadline - now) / 150)) + '%' } }) })] }),
          h('button', { className: 'gx-btn gx-primary', onClick: onAccept, disabled: busy || acceptDisabled, children: 'Accept' }), h('button', { className: 'gx-btn gx-quiet', onClick: onDecline, disabled: busy, children: 'Decline' })] }) }, 'f'), accepted && chatBar] });
}

// ---- between rounds ----
export function RoundResult({ match, me, onSkip, onDone, chat = {}, chatBar, now: serverNow, busy = false, showSkip = true, skipDisabled = false, renderName }) {
  const clock = useClock(serverNow === undefined), now = serverNow ?? clock, you = match.players.find((p) => p.id === me), opp = match.players.find((p) => p.id !== me), last = match.rounds.at(-1);
  const won = last.winner === me, n = match.rounds.length, left = Math.max(0, Math.ceil((match.starts - now) / 1000)), wins = (id) => match.rounds.filter((r) => r.winner === id).length;
  React.useEffect(() => { if (match.starts - now <= 0) onDone?.(); }, [now]);
  const dist = (p, best) => h('b', { className: 'gx-dist' + (best ? ' best' : ''), children: fmt(last.meters[p.id]) + ' m' });
  return h(Overlay, { tone: won ? 'is-win' : 'is-loss', label: 'Round ' + n + (won ? ' won' : ' lost'), children: [
    hs('header', { className: 'gx-head', children: [h('span', { className: 'gx-sub', children: 'Ranked duel · first to ' + match.winsRequired }), h('h2', { className: 'gx-title gx-outcome', children: 'Round ' + n + (won ? ' won' : ' lost') })] }, 'h'),
    hs('div', { className: 'gx-duel', children: [h(Player, { renderName, p: you, side: 'left', say: chat[you.id], children: dist(you, won) }),
      hs('div', { className: 'gx-score', children: [hs('b', { children: [wins(me), h('i', { children: '–' }), wins(opp.id)] }), h(Dots, { rounds: match.rounds, me })] }),
      h(Player, { renderName, p: opp, side: 'right', say: chat[opp.id], children: dist(opp, !won) })] }, 'd'),
    hs('footer', { className: 'gx-foot', children: [h('span', { className: 'gx-wait', children: 'Round ' + (n + 1) + ' starts in' }), h('b', { className: 'gx-count small', children: left }, left), showSkip && h('button', { className: 'gx-btn gx-primary', onClick: onSkip, disabled: busy || skipDisabled, children: 'Skip' })] }, 'f'), chatBar] });
}

// ---- end of match ----
function useCount(target, delay = 650, ms = 950) {
  const [v, set] = React.useState(reduced() ? target : 0);
  React.useEffect(() => { if (reduced()) return; let raf, t0; const id = setTimeout(() => { const step = (t) => { t0 ??= t; const k = Math.min(1, (t - t0) / ms); set(Math.round(target * (1 - Math.pow(1 - k, 3)))); if (k < 1) raf = requestAnimationFrame(step); }; raf = requestAnimationFrame(step); }, delay); return () => { clearTimeout(id); cancelAnimationFrame(raf); }; }, [target]);
  return v;
}
export function MatchResult({ match, me, onQueue, onReplay, onClose, onReport, chat = {}, chatBar, busy = false, renderName }) {
  const you = match.players.find((p) => p.id === me), opp = match.players.find((p) => p.id !== me), won = match.winner === me;
  const delta = match.delta, shown = useCount(delta), wins = (id) => match.rounds.filter((r) => r.winner === id).length;
  const now = you.rating + shown, after = rankOf(you.rating + delta), span = after.next ? after.next.min - after.min : 1, fill = after.next ? Math.max(0, Math.min(100, (now - after.min) / span * 100)) : 100;
  return h(Overlay, { tone: won ? 'is-win' : 'is-loss', label: won ? 'Victory' : 'Defeat', children: [
    hs('header', { className: 'gx-head', children: [h('span', { className: 'gx-sub', children: 'Ranked duel' }), h('h2', { className: 'gx-title gx-outcome big', children: won ? 'Victory' : 'Defeat' })] }, 'h'),
    hs('div', { className: 'gx-duel', children: [h(Player, { renderName, p: you, side: 'left', dim: !won, say: chat[you.id] }),
      hs('div', { className: 'gx-score', children: [hs('b', { children: [wins(me), h('i', { children: '–' }), wins(opp.id)] }), h(Dots, { rounds: match.rounds, me })] }),
      h(Player, { renderName, p: opp, side: 'right', dim: won, say: chat[opp.id] })] }, 'd'),
    hs('div', { className: 'gx-elo', children: [h('img', { src: badge(after.name), alt: '' }), hs('div', { className: 'gx-elo-main', children: [
      hs('div', { className: 'gx-elo-row', children: [h('strong', { children: after.name + ' · ' + fmt(now) }), h('span', { className: 'gx-delta ' + (delta >= 0 ? 'up' : 'down'), children: (delta >= 0 ? '+' : '−') + Math.abs(shown) + ' Elo' })] }),
      h('div', { className: 'gx-bar', children: h('i', { style: { width: fill + '%' } }) }),
      h('small', { children: after.next ? fmt(after.next.min - (you.rating + delta)) + ' Elo to ' + after.next.name : 'Top rank' })] })] }, 'e'),
    hs('footer', { className: 'gx-foot', children: [h('button', { className: 'gx-btn gx-primary', onClick: onQueue, disabled: busy || !onQueue, children: 'Find match' }), h('button', { className: 'gx-btn', onClick: onReplay, disabled: busy || !onReplay, children: 'Watch replay' }), h('button', { className: 'gx-btn', onClick: onClose, disabled: busy || !onClose, children: 'Back to lobby' }), onReport && match.status === 'complete' && h('button', { className: 'gx-btn gx-quiet', onClick: onReport, children: 'Report' })] }, 'f'), chatBar] });
}

// ---- rank up ----
const TIER = {
  Champion: { tone: 'tier-champion', line: 'You reached Champion', from: 'Diamond' },
  'Grand Champion': { tone: 'tier-gc', line: 'Top 100 in the world', from: 'Champion' },
  'Grand Master': { tone: 'tier-gm', line: '#1 in the world', from: 'Grand Champion', top: true },
};
export function RankUp({ rank, rating, onClose }) {
  const t = TIER[rank];
  React.useEffect(() => { const k = (e) => e.key === 'Escape' && onClose(); addEventListener('keydown', k); return () => removeEventListener('keydown', k); }, []);
  return h('div', { className: 'gx-overlay gx-rankup ' + t.tone, role: 'dialog', 'aria-modal': true, 'aria-label': 'New rank: ' + rank, children: hs('section', { className: 'gx-rank-sheet', children: [
    hs('div', { className: 'gx-badge-wrap', children: [h('img', { className: 'gx-badge', src: bigBadge(rank === 'Grand Master' ? 'Grand Champion' : rank, t.top), alt: rank }), h('span', { className: 'gx-shine', style: { WebkitMaskImage: `url(${bigBadge(rank === 'Grand Master' ? 'Grand Champion' : rank, t.top)})`, maskImage: `url(${bigBadge(rank === 'Grand Master' ? 'Grand Champion' : rank, t.top)})` }, 'aria-hidden': true })] }),
    h('span', { className: 'gx-sub', children: 'New rank' }),
    h('h2', { className: 'gx-rank-name', children: rank }),
    h('p', { className: 'gx-rank-line', children: t.line + ' · ' + fmt(rating) + ' Elo' }),
    hs('div', { className: 'gx-climb', 'aria-label': t.from + ' to ' + rank, children: [h('img', { src: badge(t.from), alt: t.from }), h('span', { className: 'gx-climb-line', children: h('i', {}) }), h('img', { src: badge(rank === 'Grand Master' ? 'Grand Champion' : rank, t.top), alt: rank })] }),
    h('button', { className: 'gx-btn gx-primary', onClick: onClose, children: 'Continue' })] }) });
}
