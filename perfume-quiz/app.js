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

function render() {
  root.innerHTML = '';
  if (state.view === 'start') root.appendChild(renderStart());
  else if (state.view === 'quiz') root.appendChild(renderQuiz());
  else if (state.view === 'result') root.appendChild(renderResult());
}

function renderStart() {
  return shell(
    el(
      'div',
      { class: 'text-center' },
      el(
        'h1',
        {
          class:
            'text-5xl sm:text-6xl font-medium text-paper mb-5 leading-[1.05]',
        },
        'The Ultimate Perfume Quiz',
      ),
      el(
        'p',
        { class: 'text-lg sm:text-xl text-paper/70 mb-12' },
        'Find the perfume that suits your aesthetic.',
      ),
      el(
        'button',
        {
          class:
            'option-btn px-8 py-3 border border-accent-400 text-accent-300 hover:text-paper rounded-full text-base',
          onclick: () => {
            state.view = 'quiz';
            state.current = 0;
            state.answers = new Array(QUESTIONS.length).fill(null);
            render();
          },
        },
        'Start',
      ),
      el(
        'p',
        { class: 'text-sm text-paper/40 mt-8' },
        `${QUESTIONS.length} questions · about 10 minutes`,
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
      el(
        'p',
        { class: 'text-sm text-accent-400 mb-4' },
        'Your perfume',
      ),
      imageFile
        ? el(
            'div',
            { class: 'flex justify-center mb-8' },
            el('img', {
              src: `./images/${imageFile}`,
              alt: entry.name,
              class:
                'max-h-80 max-w-full rounded-md shadow-2xl border hairline',
              onerror: function () {
                this.style.display = 'none';
              },
            }),
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
            'text-base sm:text-lg text-paper/80 leading-relaxed max-w-2xl mx-auto',
        },
        entry.description,
      ),
      !exact
        ? el(
            'p',
            { class: 'text-xs text-paper/40 mt-5' },
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
          'flex flex-col sm:flex-row gap-4 justify-center items-center mt-10',
      },
      el(
        'button',
        {
          class:
            'option-btn px-8 py-3 border border-accent-400 text-accent-300 hover:text-paper rounded-full text-base',
          onclick: () => {
            state.view = 'start';
            render();
          },
        },
        'Take the quiz again',
      ),
      el(
        'details',
        { class: 'text-xs text-paper/40 cursor-pointer' },
        el('summary', {}, `Profile key: ${key}`),
        el(
          'p',
          { class: 'mt-2 max-w-md' },
          AXES.map((a, i) => (key[i] === '1' ? a.negative : a.positive)).join(
            ' · ',
          ),
        ),
      ),
    ),
  );

  return shell(el('div', {}, ...parts));
}

render();
