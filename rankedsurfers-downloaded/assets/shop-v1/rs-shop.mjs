// Ranked Surfers shop + "Look" (worn item). Served from /assets/shop-v1/. Prices, ownership and payments are all
// server-side (/api/shop/*); this file only shows them. Stripe = card / Apple Pay / Google Pay; PayPal opens the
// shop checkout page; crypto (Solana) is paid inline.
import { r as React, j as jsx } from '/assets/client-Pp6hiLUC.js';
// three.js (about 2 MB) loads only when a 3D item is shown, not with every page
const mountItem = (...a) => import('./products.mjs').then((m) => m.mountItem(...a));
const h = jsx.jsx, hs = jsx.jsxs;
const RELEASE = 'site-ranked-auto-20261012-v2';
async function api(path, body) {
  const r = await fetch('/api' + path, { method: body ? 'POST' : 'GET', credentials: 'same-origin',
    headers: { 'X-Ranked-Release': RELEASE, ...(body ? { 'Content-Type': 'application/json' } : {}) }, body: body ? JSON.stringify(body) : undefined });
  let d = {}; try { d = await r.json(); } catch {}
  if (!r.ok) throw Object.assign(Error(d.error || 'Something went wrong. Try again.'), { status: r.status });
  return d;
}
const money = (c) => '$' + (c / 100).toFixed(2);
const ART = { cape: [['cape', {}]], eclipse: [['wings', { wingStyle: 6 }]], bundle: [['cape', {}], ['wings', { wingStyle: 6 }]] };

// ---- shared state: what you own and wear ------------------------------------------------------------------
const store = { inv: null, listeners: new Set() };
export async function refreshInventory() {
  try { store.inv = await api('/shop/inventory'); } catch (e) { store.inv = e.status === 401 ? null : store.inv; }
  store.listeners.forEach((f) => f(store.inv)); sendLook(); attachLook();
  return store.inv;
}
function useInventory() {
  const [inv, set] = React.useState(store.inv);
  React.useEffect(() => { store.listeners.add(set); return () => store.listeners.delete(set); }, []);
  return inv;
}
async function equip(item) {
  store.inv = await api('/shop/equip', { item });
  store.listeners.forEach((f) => f(store.inv)); sendLook();
}

// ---- the game: show what you wear on your runner ------------------------------------------------------------
const gameFrame = () => document.querySelector('iframe[title="Ranked Surfers game"]');
function sendLook() {
  const back = store.inv?.equipped || 'none';
  // The older owner game also reads this preview setting while loading. Keep it
  // aligned with server equipment so an old try-on cannot reapply blue wings.
  try {
    const key = 'rs-owner-equipment-v1';
    if (store.inv && localStorage.getItem(key)) localStorage.setItem(key, JSON.stringify({ accessory: back === 'cape' ? 'cape' : back === 'eclipse' ? 'wings' : 'none', wingDesign: 'classic' }));
  } catch {}
  const w = gameFrame()?.contentWindow; if (!w) return;
  w.postMessage({ source: 'ranked-site', type: 'wing-style', style: 6 }, location.origin);
  w.postMessage({ source: 'ranked-site', type: 'accessory-preview', accessory: back === 'cape' ? 'cape' : back === 'eclipse' ? 'wings' : 'none' }, location.origin);
}
let lastSkin = null;
// the match screens read what you wear from here (server state, not this device)
export const readEquipment = () => ({ back: store.inv?.equipped || 'none' });
addEventListener('message', (e) => {
  if (e.origin !== location.origin || e.data?.source !== 'ranked-game') return;
  const ev = e.data.event || {};
  if (ev.type === 'loaded') sendLook();
  if (ev.type === 'skin-selected' && Number.isInteger(ev.skin) && ev.skin >= 0 && ev.skin <= 15 && ev.skin !== lastSkin && store.inv) { lastSkin = ev.skin; api('/shop/skin', { skin: ev.skin }).catch(() => {}); }
});

// ---- 3D art -------------------------------------------------------------------------------------------------
function Art({ part, opts }) {
  const ref = React.useRef(null);
  React.useEffect(() => { let stopped = false, ctl = null; mountItem(ref.current, part, { ...opts, sway: false }).then((c) => { if (stopped) c.dispose(); else ctl = c; }).catch(() => {}); return () => { stopped = true; ctl?.dispose(); }; }, [part, opts.wingStyle]);
  return h('canvas', { ref, className: 'rs-render', 'aria-hidden': true });
}
const Arts = ({ id, className }) => h('div', { className: className + (ART[id].length > 1 ? ' duo' : ''), 'aria-hidden': true, children: ART[id].map(([p, o]) => h(Art, { part: p, opts: o }, p)) });

// ---- paying -------------------------------------------------------------------------------------------------
function PayChoice({ item, config, setError, onPaid }) {
  const [step, setStep] = React.useState('choose'), [busy, setBusy] = React.useState(false), [crypto, setCrypto] = React.useState(null), [note, setNote] = React.useState(''), [copied, setCopied] = React.useState(false);
  const requestId = React.useRef(null); requestId.current ??= globalThis.crypto.randomUUID();
  async function stripe() { setError(''); setBusy(true); try { const o = await api('/shop/orders', { item: item.id, provider: 'stripe', requestId: requestId.current }); const u = new URL(o.url); if (u.origin !== 'https://checkout.stripe.com') throw Error('Invalid checkout address.'); location.assign(u.href); } catch (e) { setError(e.message); setBusy(false); } }
  async function sol() { setError(''); setStep('crypto'); try { setCrypto(await api('/shop/orders', { item: item.id, provider: 'crypto', requestId: requestId.current })); } catch (e) { setError(e.message); setStep('choose'); } }
  async function check() { setError(''); setBusy(true); try { const v = await api('/shop/orders/' + crypto.id + '/check', {}); if (v.status === 'paid') { setNote('Payment confirmed.'); onPaid(v); } else setNote(v.status === 'confirming' ? 'Payment seen. Waiting for confirmation…' : 'Not received yet. It can take a minute after you send it.'); } catch (e) { setError(e.message); } finally { setBusy(false); } }
  if (step === 'crypto') return hs('div', { className: 'rs-pay rs-pay-crypto', children: [
    h('b', { children: 'Pay with Solana' }),
    crypto ? hs(React.Fragment, { children: [
      hs('p', { className: 'rs-pay-amount', children: ['Send exactly ', h('strong', { children: crypto.cryptoAmount + ' SOL' })] }),
      hs('div', { className: 'rs-pay-address', children: [h('code', { children: crypto.recipient }), h('button', { type: 'button', className: 'text-button', onClick: () => { navigator.clipboard?.writeText(crypto.recipient); setCopied(true); }, children: copied ? 'Copied' : 'Copy' })] }),
      hs('div', { className: 'rs-pay-row', children: [h('a', { className: 'button light', href: crypto.paymentUri, children: 'Open wallet' }), h('button', { type: 'button', className: 'button yellow', disabled: busy, onClick: check, children: 'Check payment' })] }),
      h('p', { className: 'rs-modal-note', children: 'Send the exact amount: it identifies your order. Quote valid for 60 minutes.' })] }) : h('p', { className: 'rs-modal-note', children: 'Getting a quote…' }),
    note && h('p', { className: 'rs-modal-note', role: 'status', children: note }),
    h('button', { type: 'button', className: 'text-button', onClick: () => { setStep('choose'); setNote(''); }, children: 'Other payment methods' })] });
  return hs('div', { className: 'rs-pay', children: [
    config?.stripe?.enabled && h('button', { type: 'button', className: 'button yellow', disabled: busy, onClick: stripe, children: busy ? 'Opening checkout…' : 'Card · Apple Pay · Google Pay' }),
    config?.paypal?.enabled && h('a', { className: 'button light rs-pay-link', href: '/shop-checkout.html?item=' + encodeURIComponent(item.id), children: 'PayPal' }),
    config?.crypto?.enabled && h('button', { type: 'button', className: 'button light', onClick: sol, children: 'Crypto (Solana)' }),
    !config?.stripe?.enabled && !config?.paypal?.enabled && !config?.crypto?.enabled && h('p', { className: 'rs-modal-note', children: 'Checkout is not open yet.' })] });
}

// ---- the Shop page -------------------------------------------------------------------------------------------
export default function Shop({ player, signIn }) {
  const inv = useInventory(), [config, setConfig] = React.useState(null), [item, setItem] = React.useState(null), [error, setError] = React.useState(''), [status, setStatus] = React.useState('');
  const modal = React.useRef(null), trigger = React.useRef(null);
  React.useEffect(() => { api('/shop/config').then(setConfig).catch(() => setConfig({ items: [] })); }, [player?.id]);
  React.useEffect(() => { if (player?.id) refreshInventory(); }, [player?.id]);
  // returning from Stripe: confirm with the server (it asks Stripe), then clean the address bar
  React.useEffect(() => {
    const q = new URLSearchParams(location.search), id = q.get('shop_order');
    if (q.get('checkout') === 'cancelled') setStatus('Checkout cancelled. No payment was made.');
    if (!id || !player?.id) return;
    let tries = 0, alive = true; setStatus('Confirming your payment…');
    const tick = async () => { if (!alive) return; try { const v = await api('/shop/orders/' + encodeURIComponent(id) + '/check', {}); if (v.status === 'paid') { setStatus('Payment confirmed. Your item is ready in Look.'); refreshInventory(); history.replaceState(null, '', '/?view=shop'); return; } } catch {} if (++tries < 15) setTimeout(tick, 2000); else setStatus('Payment not confirmed yet. If you paid, it will appear shortly; reopen the Shop to check.'); };
    tick(); return () => { alive = false; };
  }, [player?.id]);
  const owns = (it) => !!inv && it.grants.every((g) => inv.owned.includes(g));
  const close = () => { setItem(null); setError(''); requestAnimationFrame(() => trigger.current?.focus()); };
  React.useEffect(() => { if (!item) return; modal.current?.querySelector('button,a')?.focus(); const key = (e) => { if (e.key === 'Escape') { e.preventDefault(); close(); } }; document.addEventListener('keydown', key); return () => document.removeEventListener('keydown', key); }, [item]);
  const items = config?.items || [];
  const full = (it) => it.grants.length > 1 ? items.filter((x) => x.grants.length === 1 && it.grants.includes(x.id)).reduce((a, x) => a + x.cents, 0) : 0;
  const wearable = (it) => it.id === 'bundle' ? 'cape' : it.id;
  return hs('section', { className: 'page-surface rs-shop', children: [
    h('h1', { className: 'rs-shop-title', children: 'Shop' }),
    status && h('p', { className: 'rs-shop-status', role: 'status', children: status }),
    h('div', { className: 'rs-grid', children: items.map((p) => hs('button', { className: 'rs-tile', onClick: (e) => { trigger.current = e.currentTarget; setItem(p); setError(''); }, children: [
      h(Arts, { id: p.id, className: 'rs-tile-art' }),
      h('span', { className: 'rs-tile-name', children: p.name }),
      owns(p) ? h('span', { className: 'rs-tile-price owned', children: 'Owned' }) : hs('span', { className: 'rs-tile-price', children: [money(p.cents), full(p) > p.cents && h('s', { children: money(full(p)) })] })] }, p.id)) }),
    item && h('div', { className: 'rs-modal-backdrop', onClick: close, children: hs('section', { ref: modal, className: 'rs-modal', role: 'dialog', 'aria-modal': true, 'aria-label': item.name, onClick: (e) => e.stopPropagation(), children: [
      h('button', { className: 'rs-modal-close', 'aria-label': 'Close', onClick: close, children: '×' }),
      h(Arts, { id: item.id, className: 'rs-modal-art' }),
      hs('div', { className: 'rs-modal-side', children: [
        h('h2', { children: item.name }),
        owns(item) ? h('div', { className: 'rs-modal-price owned', children: 'Owned' }) : hs('div', { className: 'rs-modal-price', children: [money(item.cents), full(item) > item.cents && h('s', { children: money(full(item)) })] }),
        item.id === 'bundle' && h('p', { className: 'rs-modal-note', children: 'Both items. Wear one at a time.' }),
        !player?.id ? h('button', { className: 'button yellow', onClick: signIn, children: 'Sign in to buy' })
          : owns(item) ? h('button', { className: 'button yellow', disabled: inv?.equipped === wearable(item), onClick: () => equip(wearable(item)).catch((e) => setError(e.message)), children: inv?.equipped === wearable(item) ? 'Wearing' : 'Wear it' })
          : h(PayChoice, { item, config, setError, onPaid: () => refreshInventory() }, item.id),
        error && h('p', { className: 'rs-modal-note', role: 'alert', children: error })] })] }) })] });
}

// ---- "Look" button next to Skins: only what you own ----------------------------------------------------------
const LOOK = [{ id: 'none', name: 'None' }, { id: 'cape', name: 'RS Cape' }, { id: 'eclipse', name: 'Eclipse Wings' }];
let panel = null, thumbs = [];
function renderLook() {
  if (!panel) return;
  for (const t of panel.querySelectorAll('.rs-look-tile')) { const on = t.dataset.id === (store.inv?.equipped || 'none'); t.setAttribute('aria-pressed', String(on)); t.querySelector('.rs-look-state').textContent = on ? 'Wearing' : ''; }
}
function closeLook() { if (!panel) return; thumbs.forEach((t) => t.dispose()); thumbs = []; panel.remove(); panel = null; document.querySelector('.rs-look-btn')?.setAttribute('aria-expanded', 'false'); }
function openLook() {
  if (panel) return closeLook();
  const owned = new Set(['none', ...(store.inv?.owned || [])]);
  panel = document.createElement('section'); panel.className = 'rs-look'; panel.setAttribute('role', 'dialog'); panel.setAttribute('aria-label', 'Your look');
  panel.innerHTML = `<header><h2>Your look</h2><button type="button" class="rs-look-x" aria-label="Close">×</button></header><div class="rs-look-grid" role="radiogroup" aria-label="Back accessory">${LOOK.filter((i) => owned.has(i.id)).map((i) => `<button type="button" class="rs-look-tile" role="radio" data-id="${i.id}"><span class="rs-look-art">${i.id === 'none' ? '<svg class="rs-look-none" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="24" r="20"/><path d="M10 38 38 10"/></svg>' : '<canvas aria-hidden="true"></canvas>'}</span><b>${i.name}</b><span class="rs-look-state"></span></button>`).join('')}</div>`;
  document.body.append(panel); document.querySelector('.rs-look-btn')?.setAttribute('aria-expanded', 'true');
  panel.querySelector('.rs-look-x').onclick = closeLook;
  for (const t of panel.querySelectorAll('.rs-look-tile')) {
    t.onclick = () => equip(t.dataset.id).then(renderLook).catch(() => {});
    const c = t.querySelector('canvas'); if (c) mountItem(c, t.dataset.id === 'cape' ? 'cape' : 'wings', t.dataset.id === 'eclipse' ? { wingStyle: 6, sway: false } : { sway: false }).then((x) => { if (panel) thumbs.push(x); else x.dispose(); }).catch(() => {});
  }
  renderLook(); panel.querySelector('[aria-pressed="true"]')?.focus();
}
store.listeners.add(renderLook);
addEventListener('keydown', (e) => { if (e.key === 'Escape' && panel) closeLook(); });
addEventListener('pointerdown', (e) => { if (panel && !panel.contains(e.target) && !e.target.closest('.rs-look-btn')) closeLook(); });
function attachLook() {
  const row = document.querySelector('.play-actions'); if (!row) return;
  const has = row.querySelector('.rs-look-btn');
  if (!store.inv || store.inv.owned.length === 0) { has?.remove(); return; }
  if (has) return;
  const skins = [...row.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Skins'); if (!skins) return;
  const b = document.createElement('button'); b.type = 'button'; b.className = 'button light rs-look-btn'; b.textContent = 'Look';
  b.setAttribute('aria-haspopup', 'dialog'); b.setAttribute('aria-expanded', 'false'); b.onclick = openLook; skins.after(b);
}
new MutationObserver(attachLook).observe(document.getElementById('root') || document.body, { childList: true, subtree: true });
refreshInventory();
setInterval(() => { if (!document.hidden) refreshInventory(); }, 120000);
