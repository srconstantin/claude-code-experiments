(() => {
  'use strict';

  // ============ STATE ============

  const STORAGE_KEY = 'colors-palettes';
  const FORMAT_KEY = 'colors-display-format';
  const VALID_FORMATS = ['hsl', 'rgb', 'hex', 'hsv'];

  const DEFAULT_PALETTE = () => ({
    title: '',
    colors: [
      { h: 0, s: 60, l: 50 },
      { h: 120, s: 60, l: 50 },
      { h: 240, s: 60, l: 50 },
    ],
  });

  let active = DEFAULT_PALETTE();
  let selectedIndex = 0;
  let saved = loadSaved();
  let displayFormat = loadDisplayFormat();

  function loadSaved() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || []; }
    catch { return []; }
  }
  function persistSaved() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(saved));
  }
  function loadDisplayFormat() {
    const v = localStorage.getItem(FORMAT_KEY);
    return VALID_FORMATS.includes(v) ? v : 'hsl';
  }
  function persistDisplayFormat() {
    localStorage.setItem(FORMAT_KEY, displayFormat);
  }

  const clone = o => structuredClone(o);

  // ============ COLOR MATH ============

  // Wikipedia "Alternative HSL to RGB" formula.
  function hslToRgb(h, s, l) {
    s /= 100; l /= 100;
    const k = n => (n + h / 30) % 12;
    const a = s * Math.min(l, 1 - l);
    const f = n => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
    return {
      r: Math.round(255 * f(0)),
      g: Math.round(255 * f(8)),
      b: Math.round(255 * f(4)),
    };
  }

  function rgbToHex(r, g, b) {
    const hex = n => n.toString(16).padStart(2, '0').toUpperCase();
    return '#' + hex(r) + hex(g) + hex(b);
  }

  function rgbToHsv(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const d = max - min;
    const s = max === 0 ? 0 : d / max;
    const v = max;
    let h;
    if (d === 0) h = 0;
    else if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h = Math.round(h * 60);
    if (h < 0) h += 360;
    return { h, s: Math.round(s * 100), v: Math.round(v * 100) };
  }

  const hslString = c => `hsl(${c.h}, ${c.s}%, ${c.l}%)`;

  function colorDisplay(c) {
    const { r, g, b } = hslToRgb(c.h, c.s, c.l);
    switch (displayFormat) {
      case 'rgb':
        return { lines: [`R ${r}`, `G ${g}`, `B ${b}`], compact: `${r}, ${g}, ${b}` };
      case 'hex': {
        const hex = rgbToHex(r, g, b);
        return { lines: [hex], compact: hex };
      }
      case 'hsv': {
        const v = rgbToHsv(r, g, b);
        return { lines: [`H ${v.h}`, `S ${v.s}`, `V ${v.v}`], compact: `${v.h}, ${v.s}, ${v.v}` };
      }
      case 'hsl':
      default:
        return { lines: [`H ${c.h}`, `S ${c.s}`, `L ${c.l}`], compact: `${c.h}, ${c.s}, ${c.l}` };
    }
  }

  // ============ DOM SHORTCUTS ============

  const $ = id => document.getElementById(id);

  function escapeHTML(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  }

  // ============ RENDER ============

  function render() {
    renderPalette();
    renderDetail();
    if ($('titleInput') !== document.activeElement) {
      $('titleInput').value = active.title;
    }
  }

  function renderPalette() {
    const row = $('paletteRow');
    row.innerHTML = '';
    const N = active.colors.length;
    for (let i = 0; i < N; i++) {
      row.appendChild(makeGap(i));
      row.appendChild(makeCell(i));
    }
    row.appendChild(makeGap(N));
  }

  function makeCell(i) {
    const c = active.colors[i];
    const cell = document.createElement('div');
    cell.className = 'color-cell' + (i === selectedIndex ? ' selected' : '');
    cell.draggable = true;
    cell.dataset.idx = String(i);

    const swatch = document.createElement('div');
    swatch.className = 'swatch';
    swatch.style.background = hslString(c);

    const label = document.createElement('div');
    label.className = 'label';
    label.innerHTML = colorDisplay(c).lines.join('<br>');

    const del = document.createElement('button');
    del.className = 'delete';
    del.title = 'Delete color';
    del.textContent = '×';
    del.draggable = false;
    del.addEventListener('click', e => {
      e.stopPropagation();
      if (active.colors.length === 1) {
        alert('Palette must have at least one color.');
        return;
      }
      active.colors.splice(i, 1);
      if (selectedIndex >= active.colors.length) selectedIndex = active.colors.length - 1;
      render();
    });

    cell.append(swatch, del, label);

    cell.addEventListener('click', e => {
      if (e.target.closest('.delete')) return;
      selectedIndex = i;
      render();
    });

    cell.addEventListener('dragstart', e => {
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', String(i));
      cell.classList.add('dragging');
      $('paletteRow').classList.add('has-dragging');
    });
    cell.addEventListener('dragend', () => {
      document.querySelectorAll('.color-cell').forEach(el => {
        el.classList.remove('dragging', 'drop-before', 'drop-after');
      });
      $('paletteRow').classList.remove('has-dragging');
    });
    cell.addEventListener('dragover', e => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      const rect = cell.getBoundingClientRect();
      const before = e.clientX < rect.left + rect.width / 2;
      document.querySelectorAll('.color-cell').forEach(el => {
        if (el !== cell) el.classList.remove('drop-before', 'drop-after');
      });
      cell.classList.toggle('drop-before', before);
      cell.classList.toggle('drop-after', !before);
    });
    cell.addEventListener('drop', e => {
      e.preventDefault();
      const from = parseInt(e.dataTransfer.getData('text/plain'), 10);
      if (isNaN(from)) return;
      const rect = cell.getBoundingClientRect();
      const before = e.clientX < rect.left + rect.width / 2;
      const to = before ? i : i + 1;
      moveColor(from, to);
    });
    return cell;
  }

  function moveColor(from, to) {
    if (from === to || from + 1 === to) return;
    const [moved] = active.colors.splice(from, 1);
    const adjustedTo = from < to ? to - 1 : to;
    active.colors.splice(adjustedTo, 0, moved);
    selectedIndex = adjustedTo;
    render();
  }

  function moveSaved(from, to) {
    if (from === to || from + 1 === to) return;
    const [moved] = saved.splice(from, 1);
    const adjustedTo = from < to ? to - 1 : to;
    saved.splice(adjustedTo, 0, moved);
    persistSaved();
    renderLibrary();
  }

  function makeGap(i) {
    const gap = document.createElement('div');
    gap.className = 'gap-zone';
    const btn = document.createElement('button');
    btn.textContent = '+';
    btn.title = `Insert color at position ${i + 1}`;
    btn.addEventListener('click', () => {
      active.colors.splice(i, 0, inheritColor(i));
      selectedIndex = i;
      render();
    });
    gap.appendChild(btn);
    return gap;
  }

  function inheritColor(index) {
    if (index > 0) return clone(active.colors[index - 1]);
    if (index < active.colors.length) return clone(active.colors[index]);
    return { h: 200, s: 50, l: 50 };
  }

  function renderDetail() {
    const c = active.colors[selectedIndex];
    if (!c) return;
    $('detailHeader').textContent = `Selected color: ${selectedIndex + 1} of ${active.colors.length}`;
    $('largeSwatch').style.background = hslString(c);

    $('hSlider').style.background =
      'linear-gradient(to right, hsl(0,100%,50%), hsl(60,100%,50%), hsl(120,100%,50%), hsl(180,100%,50%), hsl(240,100%,50%), hsl(300,100%,50%), hsl(360,100%,50%))';
    $('sSlider').style.background =
      `linear-gradient(to right, hsl(${c.h}, 0%, ${c.l}%), hsl(${c.h}, 100%, ${c.l}%))`;
    $('lSlider').style.background =
      `linear-gradient(to right, hsl(${c.h}, ${c.s}%, 0%), hsl(${c.h}, ${c.s}%, 50%), hsl(${c.h}, ${c.s}%, 100%))`;

    setIfDifferent('hSlider', c.h);
    setIfDifferent('sSlider', c.s);
    setIfDifferent('lSlider', c.l);
    setIfDifferent('hInput', c.h);
    setIfDifferent('sInput', c.s);
    setIfDifferent('lInput', c.l);

    const { r, g, b } = hslToRgb(c.h, c.s, c.l);
    const hex = rgbToHex(r, g, b);
    const hsv = rgbToHsv(r, g, b);
    $('formats').innerHTML = `
      <div class="lab">HSL</div>
      <div class="val" data-copy="hsl(${c.h}, ${c.s}%, ${c.l}%)">hsl(${c.h}, ${c.s}%, ${c.l}%)</div>
      <div class="lab">RGB</div>
      <div class="val" data-copy="rgb(${r}, ${g}, ${b})">rgb(${r}, ${g}, ${b})</div>
      <div class="lab">Hex</div>
      <div class="val" data-copy="${hex}">${hex}</div>
      <div class="lab">HSV</div>
      <div class="val" data-copy="hsv(${hsv.h}, ${hsv.s}%, ${hsv.v}%)">hsv(${hsv.h}, ${hsv.s}%, ${hsv.v}%)</div>
    `;
  }

  function setIfDifferent(id, value) {
    const el = $(id);
    if (el === document.activeElement) {
      if (el.tagName === 'INPUT' && el.type === 'range') {
        if (parseInt(el.value, 10) !== value) el.value = value;
      }
      return;
    }
    if (parseInt(el.value, 10) !== value) el.value = value;
  }

  function copyText(el) {
    const text = el.getAttribute('data-copy');
    navigator.clipboard.writeText(text).then(() => {
      el.classList.add('copied');
      setTimeout(() => el.classList.remove('copied'), 700);
    }).catch(() => {
      const range = document.createRange();
      range.selectNodeContents(el);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    });
  }

  function renderLibrary() {
    const lib = $('library');
    lib.innerHTML = '';
    $('libEmpty').classList.toggle('hidden', saved.length > 0);
    saved.forEach((p, i) => {
      const card = document.createElement('div');
      card.className = 'lib-card';
      card.dataset.idx = String(i);
      card.draggable = true;
      const previewSwatches = p.colors.map(c => `<div style="background:${hslString(c)}"></div>`).join('');
      const previewHexes = p.colors.map(c => `<div>${colorDisplay(c).compact}</div>`).join('');
      card.innerHTML = `
        <div class="lib-card-header">
          <div>
            <h3>${escapeHTML(p.title || '(untitled)')}</h3>
            <div class="lib-meta">${p.colors.length} color${p.colors.length === 1 ? '' : 's'}</div>
          </div>
          <div class="lib-actions">
            <button class="btn" data-act="load">Edit</button>
            <button class="btn" data-act="invert">Invert</button>
            <button class="btn" data-act="download">PNG</button>
            <button class="btn" data-act="delete">Delete</button>
          </div>
        </div>
        <div class="lib-preview">${previewSwatches}</div>
        <div class="lib-preview-hex">${previewHexes}</div>
      `;
      lib.appendChild(card);
    });
  }

  // ============ ACTIONS ============

  function invertPalette(p) {
    return {
      title: 'Inverted ' + (p.title || 'untitled'),
      colors: p.colors.map(c => ({
        h: (c.h + 180) % 360,
        s: 100 - c.s,
        l: 100 - c.l,
      })),
    };
  }

  function savePaletteToLibrary(p) {
    const idx = saved.findIndex(x => x.title === p.title);
    if (idx >= 0) {
      if (!confirm(`A palette named "${p.title}" already exists. Overwrite?`)) return false;
      saved[idx] = p;
    } else {
      saved.push(p);
    }
    persistSaved();
    return true;
  }

  function downloadPaletteAsPNG(p) {
    const W = 160;
    const H = 220;
    const TITLE_H = 56;
    const LABEL_H = 44;
    const cols = p.colors;
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(W * cols.length, 320);
    canvas.height = TITLE_H + H + LABEL_H;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#000';
    ctx.font = 'bold 22px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(p.title || 'Untitled palette', 16, TITLE_H / 2);

    const cellW = canvas.width / cols.length;
    cols.forEach((c, i) => {
      const x = i * cellW;
      ctx.fillStyle = hslString(c);
      ctx.fillRect(x, TITLE_H, cellW, H);
      const { r, g, b } = hslToRgb(c.h, c.s, c.l);
      const hex = rgbToHex(r, g, b);

      ctx.fillStyle = '#000';
      ctx.font = 'bold 14px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(hex, x + cellW / 2, TITLE_H + H + 8);

      ctx.fillStyle = '#666';
      ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillText(`H ${c.h}  S ${c.s}  L ${c.l}`, x + cellW / 2, TITLE_H + H + 26);
    });

    canvas.toBlob(blob => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = (p.title || 'palette').replace(/[^a-z0-9_\-]+/gi, '_') + '.png';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(a.href);
    });
  }

  function switchTab(which) {
    const isActive = which === 'active';
    $('tab-active').classList.toggle('active', isActive);
    $('tab-library').classList.toggle('active', !isActive);
    $('view-active').classList.toggle('hidden', !isActive);
    $('view-library').classList.toggle('hidden', isActive);
    if (!isActive) renderLibrary();
  }

  // ============ EVENT WIRING ============

  function bindSliderAndInput(sliderId, inputId, field) {
    const onChange = e => {
      let v = parseInt(e.target.value, 10);
      if (isNaN(v)) return;
      const max = field === 'h' ? 360 : 100;
      v = Math.max(0, Math.min(max, v));
      if (active.colors[selectedIndex][field] === v) return;
      active.colors[selectedIndex][field] = v;
      render();
    };
    $(sliderId).addEventListener('input', onChange);
    $(inputId).addEventListener('input', onChange);
  }

  function refreshFormatToggleHighlight() {
    document.querySelectorAll('#formatToggle button').forEach(b => {
      b.classList.toggle('active', b.dataset.fmt === displayFormat);
    });
  }

  function init() {
    bindSliderAndInput('hSlider', 'hInput', 'h');
    bindSliderAndInput('sSlider', 'sInput', 's');
    bindSliderAndInput('lSlider', 'lInput', 'l');

    $('titleInput').addEventListener('input', e => {
      active.title = e.target.value;
    });

    $('saveBtn').addEventListener('click', () => {
      if (!active.title.trim()) {
        alert('Give the palette a title before saving.');
        $('titleInput').focus();
        return;
      }
      if (savePaletteToLibrary(clone(active))) {
        alert(`Saved "${active.title}".`);
      }
    });

    $('invertBtn').addEventListener('click', () => {
      active = invertPalette(active);
      selectedIndex = 0;
      render();
    });

    $('newBtn').addEventListener('click', () => {
      if (!confirm('Discard the current palette and start a new one?')) return;
      active = DEFAULT_PALETTE();
      selectedIndex = 0;
      render();
    });

    $('downloadBtn').addEventListener('click', () => downloadPaletteAsPNG(active));

    // Click-to-copy on format values (delegated, since formats panel re-renders).
    $('formats').addEventListener('click', e => {
      const val = e.target.closest('.val');
      if (val) copyText(val);
    });

    // Library card drag-and-drop reordering (delegated).
    const libEl = $('library');
    libEl.addEventListener('dragstart', e => {
      // suppress drag when initiated from a button inside a card
      if (e.target.closest('button')) { e.preventDefault(); return; }
      const card = e.target.closest('.lib-card');
      if (!card || !libEl.contains(card)) return;
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', card.dataset.idx);
      card.classList.add('dragging');
    });
    libEl.addEventListener('dragend', () => {
      libEl.querySelectorAll('.lib-card').forEach(el => {
        el.classList.remove('dragging', 'drop-before', 'drop-after');
      });
    });
    libEl.addEventListener('dragover', e => {
      const card = e.target.closest('.lib-card');
      if (!card) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      const rect = card.getBoundingClientRect();
      const before = e.clientY < rect.top + rect.height / 2;
      libEl.querySelectorAll('.lib-card').forEach(el => {
        if (el !== card) el.classList.remove('drop-before', 'drop-after');
      });
      card.classList.toggle('drop-before', before);
      card.classList.toggle('drop-after', !before);
    });
    libEl.addEventListener('drop', e => {
      const card = e.target.closest('.lib-card');
      if (!card) return;
      e.preventDefault();
      const from = parseInt(e.dataTransfer.getData('text/plain'), 10);
      if (isNaN(from)) return;
      const targetIdx = parseInt(card.dataset.idx, 10);
      const rect = card.getBoundingClientRect();
      const before = e.clientY < rect.top + rect.height / 2;
      moveSaved(from, before ? targetIdx : targetIdx + 1);
    });

    // Library card actions (delegated, since cards are re-rendered).
    $('library').addEventListener('click', e => {
      const btn = e.target.closest('button[data-act]');
      if (!btn) return;
      const card = btn.closest('.lib-card');
      const i = parseInt(card.dataset.idx, 10);
      const p = saved[i];
      if (!p) return;
      switch (btn.dataset.act) {
        case 'load':
          active = clone(p);
          selectedIndex = 0;
          switchTab('active');
          render();
          break;
        case 'invert':
          if (savePaletteToLibrary(invertPalette(p))) renderLibrary();
          break;
        case 'download':
          downloadPaletteAsPNG(p);
          break;
        case 'delete':
          if (!confirm(`Delete palette "${p.title || '(untitled)'}"?`)) return;
          saved.splice(i, 1);
          persistSaved();
          renderLibrary();
          break;
      }
    });

    // Format toggle.
    document.querySelectorAll('#formatToggle button').forEach(b => {
      b.addEventListener('click', () => {
        displayFormat = b.dataset.fmt;
        persistDisplayFormat();
        refreshFormatToggleHighlight();
        render();
        renderLibrary();
      });
    });
    refreshFormatToggleHighlight();

    $('tab-active').addEventListener('click', () => switchTab('active'));
    $('tab-library').addEventListener('click', () => switchTab('library'));

    render();
  }

  init();
})();
