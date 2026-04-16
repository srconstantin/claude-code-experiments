import { AXES, QUESTIONS, ANSWERS } from './data.js';

const root = document.getElementById('app');

// Load image manifest: { slug: filename }. Missing manifest is fine.
let IMAGES = {};
fetch('./images/manifest.json')
  .then((r) => (r.ok ? r.json() : {}))
  .then((m) => {
    IMAGES = m || {};
  })
  .catch(() => {});

function slugify(s) {
  return s
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '') // strip diacritics
    .replace(/['’`]/g, '')
    .replace(/&/g, 'and')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

const state = {
  view: 'start',
  current: 0,
  answers: new Array(QUESTIONS.length).fill(null),
  result: null,
  seed: null,
};

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on') && typeof v === 'function') {
      node.addEventListener(k.slice(2).toLowerCase(), v);
    } else if (v !== undefined && v !== null && v !== false) {
      node.setAttribute(k, v);
    }
  }
  for (const c of children) {
    if (c == null || c === false) continue;
    node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return node;
}

function shell(inner) {
  return el(
    'div',
    { class: 'max-w-3xl mx-auto px-6 sm:px-10 py-10 sm:py-16 fade' },
    inner,
  );
}

function ornament({ width = 140, extraClass = '' } = {}) {
  return el('div', {
    class: 'flex items-center justify-center gap-3 ornament ' + extraClass,
    style: `width: 100%; max-width: ${width + 60}px; margin-left: auto; margin-right: auto;`,
    html: `
      <span class="rule" style="flex: 0 1 ${width / 2}px; max-width: ${width / 2}px;"></span>
      <svg viewBox="0 0 14 14" width="11" height="11" aria-hidden="true" fill="currentColor">
        <path d="M7 0 L8.3 5.7 L14 7 L8.3 8.3 L7 14 L5.7 8.3 L0 7 L5.7 5.7 Z" />
      </svg>
      <span class="rule" style="flex: 0 1 ${width / 2}px; max-width: ${width / 2}px;"></span>
    `,
  });
}

function render() {
  root.innerHTML = '';
  if (state.view === 'start') root.appendChild(renderStart());
  else if (state.view === 'quiz') root.appendChild(renderQuiz());
  else if (state.view === 'result') root.appendChild(renderResult());
  else if (state.view === 'stats') root.appendChild(renderStats());
}

function renderStart() {
  return shell(
    el(
      'div',
      { class: 'text-center' },
      el('div', { class: 'mb-7' }, ornament({ width: 120 })),
      el(
        'p',
        { class: 'eyebrow mb-6' },
        'An Aesthetic Profile',
      ),
      el(
        'h1',
        {
          class:
            'text-5xl sm:text-6xl font-medium text-paper mb-5 leading-[1.02]',
        },
        'The Ultimate Perfume Quiz',
      ),
      el(
        'p',
        {
          class:
            'text-lg sm:text-xl text-paper/75 mb-10 max-w-xl mx-auto leading-relaxed italic',
        },
        'Find the perfume that suits your aesthetic.',
      ),
      el('div', { class: 'mb-10' }, ornament({ width: 60 })),
      el(
        'button',
        {
          class:
            'option-btn cta-btn px-10 py-3.5 border border-accent-400 text-accent-300 hover:text-paper rounded-full text-base tracking-wide',
          onclick: () => {
            state.view = 'quiz';
            state.current = 0;
            state.answers = new Array(QUESTIONS.length).fill(null);
            render();
          },
        },
        'Begin',
      ),
      el(
        'p',
        { class: 'text-sm text-paper/50 mt-8 italic' },
        `${QUESTIONS.length} questions · eight axes · about ten minutes`,
      ),
      el(
        'button',
        {
          class: 'text-xs text-paper/45 hover:text-accent-300 mt-10 tracking-wider uppercase',
          style: 'letter-spacing: 0.18em;',
          onclick: () => {
            state.view = 'stats';
            render();
          },
        },
        'See what others chose →',
      ),
    ),
  );
}

function renderQuiz() {
  const i = state.current;
  const q = QUESTIONS[i];
  const selected = state.answers[i];

  const optionNodes = q.options.map((label, idx) => {
    const score = idx - 2;
    const isSel = selected === score;
    return el(
      'button',
      {
        class:
          'option-btn w-full text-left px-5 py-3 border hairline rounded-lg text-paper/90 ' +
          (isSel ? 'selected' : ''),
        onclick: () => {
          state.answers[i] = score;
          if (i < QUESTIONS.length - 1) {
            state.current = i + 1;
          } else {
            finish();
          }
          render();
        },
      },
      el('span', { class: 'text-base' }, label),
    );
  });

  const progress = ((i + 1) / QUESTIONS.length) * 100;

  return shell(
    el(
      'div',
      {},
      el(
        'div',
        { class: 'mb-8' },
        el(
          'div',
          { class: 'flex justify-between items-baseline mb-2' },
          el(
            'span',
            { class: 'text-xs text-paper/50' },
            `Question ${i + 1} of ${QUESTIONS.length}`,
          ),
        ),
        el(
          'div',
          {
            class: 'h-1 bg-paper/10 w-full relative overflow-hidden rounded-full',
          },
          el('div', {
            class: 'absolute inset-y-0 left-0 bg-accent-400 rounded-full',
            style: `width: ${progress}%; transition: width 250ms ease`,
          }),
        ),
      ),
      el(
        'h2',
        {
          class:
            'text-2xl sm:text-3xl font-medium text-paper mb-8 leading-snug',
        },
        q.text,
      ),
      el('div', { class: 'space-y-2' }, ...optionNodes),
      el(
        'div',
        { class: 'flex justify-between items-center mt-8' },
        el(
          'button',
          {
            class:
              'text-sm text-paper/60 hover:text-accent-300 ' +
              (i === 0 ? 'invisible' : ''),
            onclick: () => {
              if (i > 0) {
                state.current = i - 1;
                render();
              }
            },
          },
          '← Back',
        ),
        el(
          'button',
          {
            class: 'text-sm text-paper/40 hover:text-paper/80',
            onclick: () => {
              state.view = 'start';
              render();
            },
          },
          'Start over',
        ),
      ),
    ),
  );
}

function computeAxisMeans() {
  const sums = new Array(8).fill(0);
  const counts = new Array(8).fill(0);
  for (let i = 0; i < QUESTIONS.length; i++) {
    const ans = state.answers[i];
    if (ans == null) continue;
    const axis = QUESTIONS[i].axis;
    sums[axis] += ans;
    counts[axis] += 1;
  }
  return sums.map((s, i) => (counts[i] ? s / counts[i] : 0));
}

// Seeded RNG so zero-mean tie-breaks are stable for a given run.
function mulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function meansToKey(means, rand) {
  // positive mean → 0, negative mean → 1, zero → random
  return means
    .map((m) => {
      if (m > 0) return '0';
      if (m < 0) return '1';
      return rand() < 0.5 ? '0' : '1';
    })
    .join('');
}

function hamming(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d++;
  return d;
}

function lookupAnswer(key) {
  const direct = ANSWERS.find((a) => a.key === key);
  if (direct) return { entry: direct, exact: true };
  let best = ANSWERS[0];
  let bestD = Infinity;
  for (const a of ANSWERS) {
    const d = hamming(a.key, key);
    if (d < bestD) {
      bestD = d;
      best = a;
    }
  }
  return { entry: best, exact: false };
}

function finish() {
  const seed = Date.now() & 0xffffffff;
  const rand = mulberry32(seed);
  const means = computeAxisMeans();
  const key = meansToKey(means, rand);
  const { entry, exact } = lookupAnswer(key);
  state.result = { means, key, entry, exact };
  state.view = 'result';

  const slug = slugify(entry.name);
  fetch('/api/submit', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ slug }),
  }).catch(() => {});
}

function renderAxisChart(means) {
  // For each axis, render a horizontal bar with poles on either side,
  // a center line at 0, and a marker at the user's mean score (-2..+2).
  const rows = AXES.map((axis, i) => {
    const m = means[i];
    const pct = ((m + 2) / 4) * 100; // -2 → 0%, +2 → 100%
    const toward = m < 0 ? axis.negative : m > 0 ? axis.positive : 'neutral';
    const magnitude = Math.abs(m);
    const intensity =
      magnitude < 0.25
        ? 'neutral'
        : magnitude < 0.75
        ? 'slight'
        : magnitude < 1.5
        ? 'clear'
        : 'strong';

    // Fill portion from center toward the side of the mean
    const fillStyle =
      m === 0
        ? ''
        : m < 0
        ? `right: 50%; width: ${(Math.abs(m) / 2) * 50}%;`
        : `left: 50%; width: ${(m / 2) * 50}%;`;

    return el(
      'div',
      { class: 'mb-5' },
      el(
        'div',
        { class: 'flex justify-between items-baseline mb-2' },
        el(
          'span',
          {
            class:
              'text-sm ' + (m < 0 ? 'text-accent-300' : 'text-paper/50'),
          },
          axis.negative,
        ),
        el(
          'span',
          { class: 'text-xs text-paper/40' },
          intensity === 'neutral'
            ? 'neutral'
            : `${intensity} ${toward.toLowerCase()}`,
        ),
        el(
          'span',
          {
            class:
              'text-sm ' + (m > 0 ? 'text-accent-300' : 'text-paper/50'),
          },
          axis.positive,
        ),
      ),
      el(
        'div',
        { class: 'relative h-1.5 axis-track rounded-full' },
        el('div', {
          class: 'absolute inset-y-0 left-1/2 w-px bg-paper/20',
        }),
        m !== 0
          ? el('div', {
              class:
                'absolute inset-y-0 ' +
                (m < 0 ? 'axis-fill-neg' : 'axis-fill-pos') +
                ' rounded-full',
              style: fillStyle,
            })
          : null,
        el('div', {
          class:
            'marker absolute top-1/2 w-3 h-3 rounded-full bg-accent-300 -translate-y-1/2 -translate-x-1/2',
          style: `left: ${pct}%;`,
        }),
      ),
    );
  });

  return el(
    'div',
    {},
    el(
      'h3',
      { class: 'text-xl text-paper/90 mb-5 text-center' },
      'Your profile',
    ),
    el('div', {}, ...rows),
  );
}

function renderResult() {
  const { entry, means, key, exact } = state.result;
  const parts = [];

  const slug = slugify(entry.name);
  const imageFile = IMAGES[slug];

  parts.push(
    el(
      'div',
      { class: 'text-center mb-12' },
      el('div', { class: 'mb-5' }, ornament({ width: 60 })),
      el(
        'p',
        { class: 'eyebrow mb-6' },
        'Your perfume',
      ),
      imageFile
        ? el(
            'div',
            { class: 'flex justify-center mb-8' },
            el(
              'div',
              { class: 'bottle-frame' },
              el('img', {
                src: `./images/${imageFile}`,
                alt: entry.name,
                class: 'max-h-80 max-w-full rounded-sm shadow-2xl block',
                onerror: function () {
                  this.style.display = 'none';
                },
              }),
            ),
          )
        : null,
      el(
        'h1',
        {
          class:
            'text-3xl sm:text-4xl font-medium text-paper leading-tight mb-5',
        },
        entry.name,
      ),
      el(
        'p',
        {
          class:
            'text-base sm:text-lg text-paper/85 leading-relaxed max-w-2xl mx-auto italic',
        },
        entry.description,
      ),
      !exact
        ? el(
            'p',
            { class: 'text-xs text-paper/45 mt-5' },
            'Nearest match — your exact profile had no listing.',
          )
        : null,
    ),
  );

  parts.push(
    el(
      'div',
      { class: 'max-w-xl mx-auto mt-6 mb-10' },
      renderAxisChart(means),
    ),
  );

  parts.push(
    el(
      'div',
      {
        class:
          'flex flex-col sm:flex-row gap-4 justify-center items-center mt-12',
      },
      el(
        'button',
        {
          class:
            'option-btn cta-btn px-8 py-3 border border-accent-400 text-accent-300 hover:text-paper rounded-full text-base',
          onclick: () => {
            state.view = 'start';
            render();
          },
        },
        'Take the quiz again',
      ),
      el(
        'button',
        {
          class: 'text-xs text-paper/45 hover:text-accent-300 tracking-wider uppercase',
          style: 'letter-spacing: 0.18em;',
          onclick: () => {
            state.view = 'stats';
            render();
          },
        },
        'See what others chose →',
      ),
    ),
  );

  parts.push(
    el(
      'div',
      { class: 'text-center mt-10' },
      el(
        'details',
        { class: 'text-xs text-paper/40 cursor-pointer inline-block' },
        el('summary', {}, `Profile key: ${key}`),
        el(
          'p',
          { class: 'mt-2 max-w-md mx-auto' },
          AXES.map((a, i) => (key[i] === '1' ? a.negative : a.positive)).join(
            ' · ',
          ),
        ),
      ),
    ),
  );

  return shell(el('div', {}, ...parts));
}

function renderStats() {
  const container = el('div', {});
  const body = el(
    'div',
    {},
    el('div', { class: 'text-center mb-7' }, ornament({ width: 100 })),
    el(
      'p',
      { class: 'eyebrow text-center mb-5' },
      'The ledger',
    ),
    el(
      'h1',
      {
        class:
          'text-4xl sm:text-5xl font-medium text-paper mb-4 leading-[1.05] text-center',
      },
      'Results so far',
    ),
    el(
      'p',
      { class: 'text-base text-paper/70 mb-10 text-center italic max-w-xl mx-auto' },
      'How the fragrances have fallen across everyone who has taken the quiz.',
    ),
    el('div', { id: 'stats-body', class: 'text-sm text-paper/60' }, 'Loading…'),
    el(
      'div',
      { class: 'mt-12 text-center' },
      el(
        'button',
        {
          class:
            'option-btn cta-btn px-8 py-3 border border-accent-400 text-accent-300 hover:text-paper rounded-full text-base',
          onclick: () => {
            state.view = 'start';
            render();
          },
        },
        '← Back',
      ),
    ),
  );
  container.appendChild(body);

  fetch('/api/stats', { cache: 'no-store' })
    .then((r) => (r.ok ? r.json() : {}))
    .then((data) => {
      const host = container.querySelector('#stats-body');
      if (!host || state.view !== 'stats') return;
      host.innerHTML = '';
      host.appendChild(renderStatsTable(data));
    })
    .catch(() => {
      const host = container.querySelector('#stats-body');
      if (host) host.textContent = 'Could not load stats.';
    });

  return shell(container);
}

function renderStatsTable(data) {
  const entries = Object.entries(data || {}).filter(
    ([, n]) => typeof n === 'number' && n > 0,
  );
  const total = entries.reduce((s, [, n]) => s + n, 0);

  if (total === 0) {
    return el(
      'p',
      { class: 'text-paper/60' },
      'No results recorded yet. Be the first!',
    );
  }

  const nameBySlug = Object.fromEntries(
    ANSWERS.map((a) => [slugify(a.name), a.name]),
  );

  entries.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const max = entries[0][1];

  const rows = entries.map(([slug, count]) => {
    const name = nameBySlug[slug] || slug;
    const pct = (count / total) * 100;
    const barPct = (count / max) * 100;
    return el(
      'div',
      { class: 'mb-3' },
      el(
        'div',
        { class: 'flex justify-between items-baseline mb-1' },
        el('span', { class: 'text-paper/90 text-sm' }, name),
        el(
          'span',
          { class: 'text-paper/50 text-xs' },
          `${count} · ${pct.toFixed(1)}%`,
        ),
      ),
      el(
        'div',
        { class: 'h-1.5 axis-track rounded-full relative overflow-hidden' },
        el('div', {
          class: 'absolute inset-y-0 left-0 bg-accent-400/60 rounded-full',
          style: `width: ${barPct}%;`,
        }),
      ),
    );
  });

  return el(
    'div',
    {},
    el(
      'p',
      { class: 'text-sm text-paper/50 mb-6' },
      `${total.toLocaleString()} total ${total === 1 ? 'result' : 'results'} · ${entries.length} distinct ${entries.length === 1 ? 'perfume' : 'perfumes'}`,
    ),
    el('div', {}, ...rows),
  );
}

render();
