// Liquid glass behaviour for the local preview: the selected tab's pill glides between tabs (the one signature
// motion), and on Chromium the top bar bends the light behind it with an SVG displacement lens. Pure decoration:
// it never touches the site's state or events.
const chromium = !!navigator.userAgentData?.brands?.some((b) => /Chromium/.test(b.brand));
if (chromium) {
  document.documentElement.classList.add('rs-lens');
  document.body.insertAdjacentHTML('beforeend', `<svg width="0" height="0" style="position:absolute" aria-hidden="true"><filter id="rs-lens" x="0" y="0" width="100%" height="100%">
    <feTurbulence type="fractalNoise" baseFrequency="0.006 0.012" numOctaves="2" seed="7" result="noise"/>
    <feGaussianBlur in="noise" stdDeviation="3" result="soft"/>
    <feDisplacementMap in="SourceGraphic" in2="soft" scale="22" xChannelSelector="R" yChannelSelector="G"/></filter></svg>`);
}

function pill() {
  const nav = document.querySelector('.topbar nav');
  if (!nav) return;
  let el = nav.querySelector('.rs-pill');
  if (!el) { el = document.createElement('span'); el.className = 'rs-pill'; el.setAttribute('aria-hidden', 'true'); nav.prepend(el); }
  const sel = nav.querySelector('button.selected');
  if (!sel) { el.style.opacity = '0'; return; }
  el.style.opacity = '1';
  el.style.width = sel.offsetWidth + 'px';
  el.style.transform = `translateX(${sel.offsetLeft}px)`;
}
const watch = new MutationObserver(() => requestAnimationFrame(pill));
function attach() {
  const nav = document.querySelector('.topbar nav');
  if (!nav) return setTimeout(attach, 200);
  watch.observe(nav, { subtree: true, attributes: true, attributeFilter: ['class'], childList: true });
  addEventListener('resize', pill);
  pill();
}
attach();
// the app re-renders the whole header when signing in/out; re-attach if the nav is replaced
new MutationObserver(() => { const nav = document.querySelector('.topbar nav'); if (nav && !nav.querySelector('.rs-pill')) { watch.disconnect(); attach(); } })
  .observe(document.getElementById('root') || document.body, { childList: true, subtree: true });

// Grand Champion corner: show your place big and the Elo needed to pass the player one place above
// (like the #1 bar's "+61 ahead of #2"), instead of a progress bar to World #1. Uses the public leaderboard page.
async function placeCorner() {
  const bar = document.querySelector('.lobby-bar .rank-progress--summit');
  if (!bar) return;
  const pos = +(bar.title.match(/standing is #(\d+)/) || [])[1];
  const elo = +((document.querySelector('.lobby-bar .player-rank div span')?.textContent || '').replace(/\D/g, ''));
  const key = pos + ':' + elo;
  if (!pos || bar.dataset.rsPlace === key) return;
  bar.dataset.rsPlace = key;
  let line = 'Grand Champion';
  try {
    const page = await (await fetch('/api/leaderboard/page', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ offset: Math.max(0, pos - 2), limit: 100 }) })).json();
    const above = (page.players || []).find((p) => p.position === pos - 1);
    if (above) line = Math.max(1, above.rating - elo + 1).toLocaleString('en-US') + ' Elo to #' + (pos - 1);
  } catch {}
  let el = bar.querySelector('.rs-place');
  if (!el) { el = document.createElement('div'); el.className = 'rs-place'; bar.prepend(el); }
  el.innerHTML = '<div class="big">#' + pos + '</div><div class="lead"></div>';
  el.querySelector('.lead').textContent = line;
}
new MutationObserver(() => placeCorner()).observe(document.getElementById('root') || document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['title'] });
setInterval(() => { const b = document.querySelector('.lobby-bar .rank-progress--summit'); if (b) { delete b.dataset.rsPlace; placeCorner(); } }, 60000);

// icon-only nav on mid-size windows: keep each page's name as a hover tooltip
new MutationObserver(() => { for (const b of document.querySelectorAll('.topbar nav button:not([title])')) { const t = b.textContent.trim(); if (t) b.title = t; } })
  .observe(document.getElementById('root') || document.body, { childList: true, subtree: true });
