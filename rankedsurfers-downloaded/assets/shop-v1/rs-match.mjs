// Live match screens: Codex's MatchExperience (real match state, clocks and actions) with each player's character
// and worn item looked up on the server, the signed-in player always on the left, and rank-up screens.
import { r as React, j as jsx } from '/assets/client-Pp6hiLUC.js';
import { MatchExperience } from './match-live.mjs';
import { RankUp } from './screens.mjs';
const h = jsx.jsx;
const looks = new Map();
const look = (key, id) => {
  if (!looks.has(key)) {
    if (looks.size >= 64) looks.delete(looks.keys().next().value);
    looks.set(key, fetch('/api/shop/look/' + encodeURIComponent(id)).then((r) => r.ok ? r.json() : null).catch(() => null).then((v) => { if (!v) looks.delete(key); return v; }));
  }
  return looks.get(key);
};

export function RSMatch(props) {
  const m = props.match, me = props.me, [found, setFound] = React.useState({});
  // Accept reveals the real opponent ID without changing the round ID.
  const identities = JSON.stringify((m.players || []).filter((p) => p.id !== 'opponent' && !p.anonymous).map((p) => p.id));
  React.useEffect(() => {
    let alive = true;
    Promise.all(JSON.parse(identities).map((id) => look(m.id + ':' + id, id).then((v) => [id, v]))).then((e) => alive && setFound(Object.fromEntries(e.filter(([, v]) => v))));
    return () => { alive = false; };
  }, [m.id, identities]);
  const players = (m.players || []).map((p) => found[p.id] ? { ...p, skin: found[p.id].skin, cosmetics: { back: found[p.id].back } } : p)
    .sort((a, b) => (a.id === me?.id ? -1 : b.id === me?.id ? 1 : 0));
  return h(MatchExperience, { ...props, match: { ...m, players } });
}

// New rank screens for Champion, Grand Champion and Grand Master (#1). Remembers the last rank seen per account.
const ORDER = ['Unranked', 'Bronze', 'Silver', 'Gold', 'Diamond', 'Champion', 'Grand Champion', 'Grand Master'];
export function RSRankWatch({ player }) {
  const [show, setShow] = React.useState(null);
  const label = player?.id ? (player.rank === 'Grand Champion' && player.position === 1 ? 'Grand Master' : player.rank) : null;
  React.useEffect(() => {
    if (!label || !ORDER.includes(label)) return;
    const key = 'rs-rank-seen:' + player.id;
    let before = null; try { before = localStorage.getItem(key); localStorage.setItem(key, label); } catch {}
    if (before && ORDER.indexOf(label) > ORDER.indexOf(before) && ['Champion', 'Grand Champion', 'Grand Master'].includes(label)) setShow({ rank: label, rating: player.rating });
  }, [player?.id, label]);
  return show ? h(RankUp, { rank: show.rank, rating: show.rating, onClose: () => setShow(null) }) : null;
}
