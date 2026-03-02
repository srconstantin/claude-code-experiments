// ─── Bond Math ───────────────────────────────────────────────────────────────

function bondPrice(face, couponRate, yieldRate, maturity, freq = 2) {
  const periods = maturity * freq;
  const coupon = face * couponRate / freq;
  const r = yieldRate / freq;
  if (r === 0) return coupon * periods + face;
  let price = 0;
  for (let t = 1; t <= periods; t++) price += coupon / Math.pow(1 + r, t);
  price += face / Math.pow(1 + r, periods);
  return price;
}

function macaulayDuration(face, couponRate, yieldRate, maturity, freq = 2) {
  const periods = maturity * freq;
  const coupon = face * couponRate / freq;
  const r = yieldRate / freq;
  const price = bondPrice(face, couponRate, yieldRate, maturity, freq);
  let num = 0;
  for (let t = 1; t <= periods; t++) {
    const cf = t === periods ? coupon + face : coupon;
    num += (t / freq) * (cf / Math.pow(1 + r, t));
  }
  return num / price;
}

function modDuration(face, couponRate, yieldRate, maturity, freq = 2) {
  const mac = macaulayDuration(face, couponRate, yieldRate, maturity, freq);
  return mac / (1 + yieldRate / freq);
}

function convexity(face, couponRate, yieldRate, maturity, freq = 2) {
  const periods = maturity * freq;
  const coupon = face * couponRate / freq;
  const r = yieldRate / freq;
  const price = bondPrice(face, couponRate, yieldRate, maturity, freq);
  let conv = 0;
  for (let t = 1; t <= periods; t++) {
    const cf = t === periods ? coupon + face : coupon;
    conv += t * (t + 1) * (cf / Math.pow(1 + r, t));
  }
  return conv / (price * Math.pow(1 + r, 2) * freq * freq);
}

// Nelson-Siegel yield curve model
function nelsonSiegel(t, beta0, beta1, beta2, tau) {
  if (t < 0.001) return beta0 + beta1;
  const f = (1 - Math.exp(-t / tau)) / (t / tau);
  return beta0 + beta1 * f + beta2 * (f - Math.exp(-t / tau));
}

function fmt(n, dec = 2) { return n.toFixed(dec); }
function fmtPct(n, dec = 2) { return fmt(n * 100, dec) + '%'; }

// ─── Chart Registry ──────────────────────────────────────────────────────────

const charts = {};

function makeChart(id, config) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), config);
  return charts[id];
}

// ─── Section 1: Bond Basics ──────────────────────────────────────────────────

function initBasics() {
  const inputs = ['b-face', 'b-coupon', 'b-ytm', 'b-maturity'];
  inputs.forEach(id => {
    const el = document.getElementById(id);
    el.addEventListener('input', updateBasics);
    document.getElementById(id + '-val').textContent = el.value;
    el.addEventListener('input', () => {
      document.getElementById(id + '-val').textContent = el.value;
    });
  });

  // Cash flow chart
  makeChart('cf-chart', {
    type: 'bar',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      plugins: { legend: { display: false },
        title: { display: true, text: 'Cash Flow Timeline', color: '#a0aec0' } },
      scales: {
        x: { ticks: { color: '#a0aec0' }, grid: { color: '#2d3748' } },
        y: { ticks: { color: '#a0aec0' }, grid: { color: '#2d3748' } }
      }
    }
  });

  updateBasics();
}

function updateBasics() {
  const face = parseFloat(document.getElementById('b-face').value);
  const couponRate = parseFloat(document.getElementById('b-coupon').value) / 100;
  const ytm = parseFloat(document.getElementById('b-ytm').value) / 100;
  const maturity = parseInt(document.getElementById('b-maturity').value);
  const freq = 2;

  const price = bondPrice(face, couponRate, ytm, maturity, freq);
  const mac = macaulayDuration(face, couponRate, ytm, maturity, freq);
  const md = modDuration(face, couponRate, ytm, maturity, freq);
  const couponAmt = face * couponRate / freq;
  const currentYield = (couponAmt * freq) / price;

  document.getElementById('b-price').textContent = '$' + fmt(price, 2);
  document.getElementById('b-mac-dur').textContent = fmt(mac, 2) + ' yrs';
  document.getElementById('b-mod-dur').textContent = fmt(md, 2);
  document.getElementById('b-cur-yield').textContent = fmtPct(currentYield);

  // Color price relative to par
  const priceEl = document.getElementById('b-price');
  if (price > face + 0.5) priceEl.style.color = '#68d391';
  else if (price < face - 0.5) priceEl.style.color = '#fc8181';
  else priceEl.style.color = '#f6e05e';

  // Premium/discount label
  const prem = price - face;
  const premEl = document.getElementById('b-premium');
  if (Math.abs(prem) < 0.5) {
    premEl.textContent = 'Trading at PAR';
    premEl.style.color = '#f6e05e';
  } else if (prem > 0) {
    premEl.textContent = `Premium of $${fmt(prem, 2)} (coupon > market yield)`;
    premEl.style.color = '#68d391';
  } else {
    premEl.textContent = `Discount of $${fmt(-prem, 2)} (coupon < market yield)`;
    premEl.style.color = '#fc8181';
  }

  // Build cash flow chart
  const periods = maturity * freq;
  const labels = [];
  const cashflows = [];
  for (let t = 1; t <= periods; t++) {
    labels.push(`${t / freq}Y`);
    cashflows.push(t === periods ? couponAmt + face : couponAmt);
  }

  const c = charts['cf-chart'];
  c.data.labels = labels;
  c.data.datasets = [{
    label: 'Cash Flow',
    data: cashflows,
    backgroundColor: cashflows.map((v, i) => i === cashflows.length - 1 ? '#4299e1' : '#00d4aa'),
    borderRadius: 3,
  }];
  c.update();
}

// ─── Section 2: Price-Yield Relationship ─────────────────────────────────────

function initPriceYield() {
  ['py-coupon', 'py-maturity', 'py-ytm'].forEach(id => {
    const el = document.getElementById(id);
    el.addEventListener('input', () => {
      document.getElementById(id + '-val').textContent = el.value;
      updatePriceYield();
    });
  });

  makeChart('py-chart', {
    type: 'line',
    data: { datasets: [] },
    options: {
      responsive: true,
      animation: false,
      plugins: {
        legend: { display: false },
        title: { display: true, text: 'Bond Price vs Yield', color: '#a0aec0' },
        tooltip: {
          callbacks: {
            label: ctx => `Price: $${fmt(ctx.parsed.y)} at yield ${fmt(ctx.parsed.x)}%`
          }
        }
      },
      scales: {
        x: {
          type: 'linear', title: { display: true, text: 'Yield (%)', color: '#a0aec0' },
          ticks: { color: '#a0aec0' }, grid: { color: '#2d3748' }
        },
        y: {
          title: { display: true, text: 'Price ($)', color: '#a0aec0' },
          ticks: { color: '#a0aec0' }, grid: { color: '#2d3748' }
        }
      }
    }
  });

  updatePriceYield();
}

function updatePriceYield() {
  const couponRate = parseFloat(document.getElementById('py-coupon').value) / 100;
  const maturity = parseInt(document.getElementById('py-maturity').value);
  const currentYtm = parseFloat(document.getElementById('py-ytm').value) / 100;
  const face = 1000;

  const yields = [];
  const prices = [];
  for (let y = 0.5; y <= 15; y += 0.1) {
    yields.push(y);
    prices.push(bondPrice(face, couponRate, y / 100, maturity));
  }

  const currentPrice = bondPrice(face, couponRate, currentYtm, maturity);
  const md = modDuration(face, couponRate, currentYtm, maturity);
  const conv = convexity(face, couponRate, currentYtm, maturity);

  document.getElementById('py-price').textContent = '$' + fmt(currentPrice);
  document.getElementById('py-moddur').textContent = fmt(md, 2);
  document.getElementById('py-convexity').textContent = fmt(conv, 2);

  // Duration approximation for 100bps move
  const approxUp = -md * 0.01 * currentPrice;
  const actualUp = bondPrice(face, couponRate, currentYtm + 0.01, maturity) - currentPrice;
  const approxDown = md * 0.01 * currentPrice;
  const actualDown = bondPrice(face, couponRate, currentYtm - 0.01, maturity) - currentPrice;
  document.getElementById('py-dur-approx').innerHTML =
    `+100bps → price change: <b style="color:#fc8181">~${fmt(approxUp, 2)}</b> (actual: ${fmt(actualUp, 2)})<br>` +
    `-100bps → price change: <b style="color:#68d391">~+${fmt(approxDown, 2)}</b> (actual: +${fmt(actualDown, 2)})`;

  const c = charts['py-chart'];
  c.data.datasets = [
    {
      label: 'Price-Yield Curve',
      data: yields.map((y, i) => ({ x: y, y: prices[i] })),
      borderColor: '#00d4aa', borderWidth: 2, pointRadius: 0, fill: false, tension: 0.4
    },
    {
      label: 'Current Point',
      data: [{ x: currentYtm * 100, y: currentPrice }],
      borderColor: '#f6e05e', backgroundColor: '#f6e05e',
      pointRadius: 8, pointHoverRadius: 10, type: 'scatter'
    }
  ];
  c.update();
}

// ─── Section 3: Duration ─────────────────────────────────────────────────────

function initDuration() {
  ['dur-shock'].forEach(id => {
    const el = document.getElementById(id);
    el.addEventListener('input', () => {
      document.getElementById(id + '-val').textContent = el.value;
      updateDuration();
    });
  });

  makeChart('dur-chart', {
    type: 'bar',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#a0aec0' } },
        title: { display: true, text: 'Price % Change by Rate Shock', color: '#a0aec0' }
      },
      scales: {
        x: { ticks: { color: '#a0aec0' }, grid: { color: '#2d3748' } },
        y: {
          title: { display: true, text: '% Change in Price', color: '#a0aec0' },
          ticks: { color: '#a0aec0' }, grid: { color: '#2d3748' }
        }
      }
    }
  });

  updateDuration();
}

const DUR_BONDS = [
  { label: '2Y, 5% coupon', maturity: 2, coupon: 0.05 },
  { label: '5Y, 5% coupon', maturity: 5, coupon: 0.05 },
  { label: '10Y, 5% coupon', maturity: 10, coupon: 0.05 },
  { label: '30Y, 5% coupon', maturity: 30, coupon: 0.05 },
  { label: '10Y, 2% coupon', maturity: 10, coupon: 0.02 },
  { label: '10Y, 8% coupon', maturity: 10, coupon: 0.08 },
];

function updateDuration() {
  const shockBps = parseFloat(document.getElementById('dur-shock').value);
  const shockDecimal = shockBps / 10000;
  const face = 1000;
  const ytm = 0.05;

  const results = DUR_BONDS.map(b => {
    const p0 = bondPrice(face, b.coupon, ytm, b.maturity);
    const p1 = bondPrice(face, b.coupon, ytm + shockDecimal, b.maturity);
    const md = modDuration(face, b.coupon, ytm, b.maturity);
    const pctChange = (p1 - p0) / p0 * 100;
    return { label: b.label, pctChange, md };
  });

  // Update table
  const tbody = document.getElementById('dur-table-body');
  tbody.innerHTML = results.map(r => {
    const color = r.pctChange > 0.001 ? '#68d391' : r.pctChange < -0.001 ? '#fc8181' : '#a0aec0';
    return `<tr>
      <td>${r.label}</td>
      <td>${fmt(r.md, 2)}</td>
      <td style="color:${color}">${fmt(r.pctChange, 2)}%</td>
    </tr>`;
  }).join('');

  const c = charts['dur-chart'];
  c.data.labels = results.map(r => r.label);
  c.data.datasets = [{
    label: `Price change for ${shockBps > 0 ? '+' : ''}${shockBps}bps shock`,
    data: results.map(r => r.pctChange),
    backgroundColor: results.map(r => r.pctChange >= 0 ? '#68d391aa' : '#fc8181aa'),
    borderColor: results.map(r => r.pctChange >= 0 ? '#68d391' : '#fc8181'),
    borderWidth: 1, borderRadius: 4
  }];
  c.update();
}

// ─── Section 4: Yield Curve ──────────────────────────────────────────────────

const CURVE_PRESETS = {
  normal:   { beta0: 4.5, beta1: -2.5, beta2: 1.5, tau: 2 },
  inverted: { beta0: 3.5, beta1: 2.0,  beta2: -1.5, tau: 1.5 },
  flat:     { beta0: 4.0, beta1: 0.1,  beta2: 0.1, tau: 2 },
  humped:   { beta0: 4.0, beta1: -1.0, beta2: 3.0, tau: 1.5 },
};

const CURVE_MATURITIES = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30];
const CURVE_LABELS = ['3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y'];

function initYieldCurve() {
  ['yc-beta0', 'yc-beta1', 'yc-beta2', 'yc-tau'].forEach(id => {
    const el = document.getElementById(id);
    el.addEventListener('input', () => {
      document.getElementById(id + '-val').textContent = parseFloat(el.value).toFixed(1);
      updateYieldCurve();
    });
  });

  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const preset = CURVE_PRESETS[btn.dataset.preset];
      document.getElementById('yc-beta0').value = preset.beta0;
      document.getElementById('yc-beta1').value = preset.beta1;
      document.getElementById('yc-beta2').value = preset.beta2;
      document.getElementById('yc-tau').value = preset.tau;
      ['yc-beta0', 'yc-beta1', 'yc-beta2', 'yc-tau'].forEach(id => {
        document.getElementById(id + '-val').textContent =
          parseFloat(document.getElementById(id).value).toFixed(1);
      });
      updateYieldCurve();
    });
  });

  makeChart('yc-chart', {
    type: 'line',
    data: { datasets: [] },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#a0aec0' } },
        title: { display: true, text: 'Treasury Yield Curve', color: '#a0aec0' }
      },
      scales: {
        x: { ticks: { color: '#a0aec0' }, grid: { color: '#2d3748' } },
        y: {
          title: { display: true, text: 'Yield (%)', color: '#a0aec0' },
          ticks: { color: '#a0aec0' }, grid: { color: '#2d3748' },
          min: 0, max: 8
        }
      }
    }
  });

  // Load normal preset
  const p = CURVE_PRESETS.normal;
  document.getElementById('yc-beta0').value = p.beta0;
  document.getElementById('yc-beta1').value = p.beta1;
  document.getElementById('yc-beta2').value = p.beta2;
  document.getElementById('yc-tau').value = p.tau;
  ['yc-beta0', 'yc-beta1', 'yc-beta2', 'yc-tau'].forEach(id => {
    document.getElementById(id + '-val').textContent =
      parseFloat(document.getElementById(id).value).toFixed(1);
  });

  updateYieldCurve();
}

function updateYieldCurve() {
  const beta0 = parseFloat(document.getElementById('yc-beta0').value);
  const beta1 = parseFloat(document.getElementById('yc-beta1').value);
  const beta2 = parseFloat(document.getElementById('yc-beta2').value);
  const tau = parseFloat(document.getElementById('yc-tau').value);

  const yields = CURVE_MATURITIES.map(t => nelsonSiegel(t, beta0, beta1, beta2, tau));

  // Curve analysis
  const shortEnd = yields[2]; // 1Y
  const longEnd = yields[7];  // 10Y
  const spread = longEnd - shortEnd;
  // True hump: yields in the middle (2Y–7Y) exceed BOTH endpoints.
  // Compare mid-range max against the higher of the two endpoints.
  const midMax = Math.max(...yields.slice(3, 7)); // 2Y, 3Y, 5Y, 7Y
  const trueHump = midMax - Math.max(shortEnd, longEnd);

  let shape = 'Normal (upward sloping)';
  let shapeColor = '#68d391';
  let interpretation = 'Markets expect moderate growth and stable rates. Most common curve shape.';
  if (spread < -0.3) {
    shape = 'Inverted (downward sloping)';
    shapeColor = '#fc8181';
    interpretation = 'Short rates > long rates. Historically signals recession risk within ~12-18 months.';
  } else if (Math.abs(spread) < 0.3) {
    shape = 'Flat';
    shapeColor = '#f6e05e';
    interpretation = 'Transition state — often between normal and inverted. Uncertainty about growth outlook.';
  } else if (trueHump > 0.1) {
    shape = 'Humped';
    shapeColor = '#b794f4';
    interpretation = 'Medium-term yields peak then fall. Mixed signals about near vs long-term rates.';
  }

  document.getElementById('yc-shape').textContent = shape;
  document.getElementById('yc-shape').style.color = shapeColor;
  document.getElementById('yc-interpretation').textContent = interpretation;
  // HTML spans already contain the "2s10s:" / "3m10y:" label, so just set the value.
  document.getElementById('yc-2s10s').textContent = fmt(yields[7] - yields[3], 2) + '%';
  document.getElementById('yc-3m10y').textContent = fmt(yields[7] - yields[0], 2) + '%';

  const c = charts['yc-chart'];
  if (!c) return;
  c.data.labels = CURVE_LABELS;
  c.data.datasets = [{
    label: 'Yield Curve',
    data: yields,
    borderColor: '#00d4aa',
    backgroundColor: 'rgba(0, 212, 170, 0.1)',
    borderWidth: 3,
    pointRadius: 5,
    pointBackgroundColor: '#00d4aa',
    fill: true,
    tension: 0.4
  }];
  c.update();
}

// ─── Section 5: Credit Spreads ───────────────────────────────────────────────

const RATING_SPREADS = {
  'AAA': 0.20,
  'AA':  0.40,
  'A':   0.80,
  'BBB': 1.40,
  'BB':  2.50,
  'B':   4.50,
  'CCC': 8.00,
};

function initCredit() {
  ['cr-maturity', 'cr-coupon'].forEach(id => {
    const el = document.getElementById(id);
    el.addEventListener('input', () => {
      document.getElementById(id + '-val').textContent = el.value;
      updateCredit();
    });
  });

  document.getElementById('cr-rating').addEventListener('change', updateCredit);

  makeChart('cr-chart', {
    type: 'bar',
    data: { datasets: [] },
    options: {
      responsive: true,
      indexAxis: 'y',
      plugins: {
        legend: { display: false },
        title: { display: true, text: 'Bond Price by Credit Rating', color: '#a0aec0' }
      },
      scales: {
        x: {
          title: { display: true, text: 'Price ($)', color: '#a0aec0' },
          ticks: { color: '#a0aec0' }, grid: { color: '#2d3748' }
        },
        y: { ticks: { color: '#a0aec0' }, grid: { color: '#2d3748' } }
      }
    }
  });

  updateCredit();
}

function updateCredit() {
  const riskFreeYield = 0.045;
  const maturity = parseInt(document.getElementById('cr-maturity').value);
  const couponRate = parseFloat(document.getElementById('cr-coupon').value) / 100;
  const selectedRating = document.getElementById('cr-rating').value;
  const face = 1000;

  const selectedSpread = RATING_SPREADS[selectedRating] / 100;
  const selectedYtm = riskFreeYield + selectedSpread;
  const selectedPrice = bondPrice(face, couponRate, selectedYtm, maturity);

  document.getElementById('cr-rf-yield').textContent = fmtPct(riskFreeYield);
  document.getElementById('cr-spread').textContent = fmt(RATING_SPREADS[selectedRating] * 100, 0) + ' bps';
  document.getElementById('cr-total-yield').textContent = fmtPct(selectedYtm);
  document.getElementById('cr-price').textContent = '$' + fmt(selectedPrice);

  const ratingNames = Object.keys(RATING_SPREADS);
  const prices = ratingNames.map(r => {
    const ytm = riskFreeYield + RATING_SPREADS[r] / 100;
    return bondPrice(face, couponRate, ytm, maturity);
  });

  const ratingColors = {
    'AAA': '#68d391', 'AA': '#68d391', 'A': '#f6e05e',
    'BBB': '#f6e05e', 'BB': '#ed8936', 'B': '#fc8181', 'CCC': '#e53e3e'
  };

  const c = charts['cr-chart'];
  c.data.labels = ratingNames;
  c.data.datasets = [{
    data: prices,
    backgroundColor: ratingNames.map(r =>
      r === selectedRating ? ratingColors[r] : ratingColors[r] + '55'),
    borderColor: ratingNames.map(r => ratingColors[r]),
    borderWidth: ratingNames.map(r => r === selectedRating ? 3 : 1),
    borderRadius: 4
  }];
  c.update();
}

// ─── Section 6: Quiz ─────────────────────────────────────────────────────────

const QUESTIONS = [
  {
    q: "The Fed raises rates by 75 basis points (0.75%). You hold a 10-year Treasury with 5% coupon and 5% yield. What happens to your bond's price?",
    options: ['Price rises', 'Price falls', 'Price stays the same', 'Price falls, then rises'],
    answer: 1,
    explanation: 'When market yields rise, existing bond prices fall — they must offer a higher return to compete with new bonds. With a modified duration of ~7.5 years, a 75bps rise means roughly a 5.6% price drop.'
  },
  {
    q: "Which bond has GREATER interest rate sensitivity (higher duration)?",
    options: [
      '30-year bond with 2% coupon',
      '5-year bond with 6% coupon',
      '10-year bond with 5% coupon',
      '2-year bond with 2% coupon'
    ],
    answer: 0,
    explanation: 'Duration increases with maturity and decreases with coupon rate. The 30Y/2% bond has the highest duration (~22 years) because it\'s long-dated AND low-coupon (most cash flows are far in the future).'
  },
  {
    q: "The yield curve inverts (short-term rates higher than long-term). What does this typically signal?",
    options: [
      'Strong economic growth ahead',
      'High inflation expectations',
      'Elevated recession risk',
      'Currency devaluation risk'
    ],
    answer: 2,
    explanation: 'An inverted yield curve (especially the 2s10s spread) has historically preceded US recessions. Markets price in Fed rate CUTS in the future (long-term yields fall), reflecting expectations of economic weakness.'
  },
  {
    q: "A bond's coupon is 5%, but it trades at $950 (below par of $1,000). What must be true?",
    options: [
      'Market yield < 5%',
      'Market yield > 5%',
      'Market yield = 5%',
      'The issuer is about to default'
    ],
    answer: 1,
    explanation: 'A bond trading at a discount means investors demand a higher return than the coupon offers. Market yield > coupon rate → price < par. The bond adjusts in price so that its total return (coupon + capital gain at maturity) equals the market yield.'
  },
  {
    q: "You buy a 30-year bond today. Over the next year, you earn the coupon AND the bond 'rolls down' the yield curve. What does roll-down mean?",
    options: [
      'The bond gets riskier over time',
      'You refinance the bond at a lower rate',
      'A 30Y bond becomes a 29Y bond, typically with a lower yield in an upward curve',
      'The bond price falls as it approaches maturity'
    ],
    answer: 2,
    explanation: 'In a normal (upward-sloping) yield curve, a 30Y bond becomes a 29Y bond after a year. 29Y yields are typically lower, so the bond price RISES as it rolls down the curve. This "carry + rolldown" is a key source of bond returns.'
  },
  {
    q: "The 'Z-spread' of a corporate bond is 200bps over Treasuries. What does this represent?",
    options: [
      'The bond pays 2% above Treasury yield',
      'The constant spread added to each point on the spot curve to match the bond\'s market price',
      'The bond\'s credit rating score',
      'The difference between coupon and yield'
    ],
    answer: 1,
    explanation: 'The Z-spread (zero-volatility spread) is added to the entire Treasury spot (zero-coupon) curve to discount the bond\'s cash flows to its market price. It represents the total credit/liquidity premium above risk-free rates.'
  },
  {
    q: "Duration approximation says price change ≈ -ModDur × ΔYield × Price. The actual price change is slightly LESS negative than the approximation for a yield increase. Why?",
    options: [
      'Duration overestimates risk for small moves',
      'Convexity — the price-yield relationship is curved, so price falls less than the linear approximation',
      'The coupon payments offset the yield increase',
      'Market makers add a liquidity premium'
    ],
    answer: 1,
    explanation: 'The price-yield relationship is CONVEX (curved), not linear. Duration is a first-order (linear) approximation. Convexity is the second-order correction. Because of convexity, bond prices fall LESS than duration predicts when yields rise, and rise MORE than predicted when yields fall. Convexity is always positive for plain bonds — it\'s good for bondholders!'
  },
  {
    q: "A repo (repurchase agreement) lets a bond dealer sell a bond today and buy it back tomorrow. What is this primarily used for?",
    options: [
      'Speculating on bond prices going up',
      'Short-term financing — using bonds as collateral to borrow cash overnight',
      'Hedging against interest rate risk',
      'Converting coupon bonds to zero-coupon bonds'
    ],
    answer: 1,
    explanation: 'Repos are the plumbing of the bond market. A dealer "repos out" bonds to borrow cash overnight at the repo rate. The buyer of the repo earns the repo rate. It\'s effectively a collateralized loan, and the repo market enables huge amounts of bond market leverage.'
  },
  {
    q: "A bond has Modified Duration = 8 and Convexity = 90. Yields spike up 2% (200 bps). What is the approximate price change?",
    options: [
      '-16%',
      '-14.2%',
      '-8%',
      '-17.8%'
    ],
    answer: 1,
    explanation: 'Price change ≈ -ModDur × ΔY + 0.5 × Convexity × ΔY² = -8 × 0.02 + 0.5 × 90 × 0.02² = -0.16 + 0.018 = -0.142 = -14.2%. The convexity term saves you about 1.8% compared to the duration-only approximation of -16%.'
  },
  {
    q: "The 5-year forward rate 5 years from now (5y5y) is 4.5%. The 5-year spot rate is 3.5% and the 10-year spot rate is 4.0%. Is this consistent?",
    options: [
      'No — forward rates must always equal spot rates',
      'Yes — forward rates connect spot rates via the no-arbitrage condition',
      'No — the 5y5y should equal the 10Y spot rate',
      'Yes — but only if the curve is flat'
    ],
    answer: 1,
    explanation: 'Forward rates are DERIVED from spot rates via no-arbitrage: (1 + r10)^10 = (1 + r5)^5 × (1 + f5y5y)^5. Here: 1.04^10 = 1.035^5 × 1.045^5. Let\'s check: 1.4802 ≈ 1.1877 × 1.2462 ≈ 1.4803. ✓ Spot-forward relationships must hold or there\'s a free arbitrage!'
  }
];

let quizState = { current: 0, score: 0, answered: false };

function initQuiz() {
  renderQuestion();
  document.getElementById('quiz-next').addEventListener('click', nextQuestion);
  document.getElementById('quiz-restart').addEventListener('click', restartQuiz);
}

function renderQuestion() {
  const q = QUESTIONS[quizState.current];
  document.getElementById('quiz-progress').textContent =
    `Question ${quizState.current + 1} of ${QUESTIONS.length} | Score: ${quizState.score}`;
  document.getElementById('quiz-question').textContent = q.q;

  const optDiv = document.getElementById('quiz-options');
  optDiv.innerHTML = q.options.map((opt, i) =>
    `<button class="quiz-option" data-index="${i}">${String.fromCharCode(65 + i)}. ${opt}</button>`
  ).join('');

  optDiv.querySelectorAll('.quiz-option').forEach(btn => {
    btn.addEventListener('click', () => selectAnswer(parseInt(btn.dataset.index)));
  });

  document.getElementById('quiz-explanation').style.display = 'none';
  document.getElementById('quiz-next').style.display = 'none';
  document.getElementById('quiz-restart').style.display = 'none';
  quizState.answered = false;
}

function selectAnswer(i) {
  if (quizState.answered) return;
  quizState.answered = true;

  const q = QUESTIONS[quizState.current];
  const buttons = document.querySelectorAll('.quiz-option');

  buttons.forEach((btn, idx) => {
    btn.disabled = true;
    if (idx === q.answer) btn.classList.add('correct');
    else if (idx === i) btn.classList.add('wrong');
  });

  if (i === q.answer) {
    quizState.score++;
    document.getElementById('quiz-result').textContent = '✓ Correct!';
    document.getElementById('quiz-result').style.color = '#68d391';
  } else {
    document.getElementById('quiz-result').textContent = '✗ Not quite.';
    document.getElementById('quiz-result').style.color = '#fc8181';
  }

  document.getElementById('quiz-explanation-text').textContent = q.explanation;
  document.getElementById('quiz-explanation').style.display = 'block';

  if (quizState.current < QUESTIONS.length - 1) {
    document.getElementById('quiz-next').style.display = 'inline-block';
  } else {
    showFinalScore();
  }
}

function nextQuestion() {
  quizState.current++;
  renderQuestion();
}

function showFinalScore() {
  const pct = Math.round(quizState.score / QUESTIONS.length * 100);
  let grade, color;
  if (pct >= 90) { grade = 'Excellent! You\'re ready for the bond desk.'; color = '#68d391'; }
  else if (pct >= 70) { grade = 'Good work! Review duration and convexity.'; color = '#f6e05e'; }
  else { grade = 'Keep studying! Go back through the interactive sections.'; color = '#fc8181'; }

  document.getElementById('quiz-final').style.display = 'block';
  document.getElementById('quiz-final').innerHTML =
    `<h3 style="color:${color}">Final Score: ${quizState.score}/${QUESTIONS.length} (${pct}%)</h3>
     <p style="color:${color}">${grade}</p>`;
  document.getElementById('quiz-restart').style.display = 'inline-block';
}

function restartQuiz() {
  quizState = { current: 0, score: 0, answered: false };
  document.getElementById('quiz-final').style.display = 'none';
  renderQuestion();
}

// ─── Navigation ──────────────────────────────────────────────────────────────

function initNav() {
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.section).classList.add('active');
      // Charts initialized in hidden sections have 0×0 size; resize them when shown.
      Object.values(charts).forEach(c => { try { c.resize(); } catch(e) {} });
    });
  });
}

// ─── Init ─────────────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
  initNav();
  initBasics();
  initPriceYield();
  initDuration();
  initYieldCurve();
  initCredit();
  initQuiz();
});
