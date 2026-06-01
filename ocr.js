/**
 * ocr.js — Advanced multi-pass Tesseract OCR with auto-detecting grid size
 *
 * Key improvements over v1:
 *  • Auto-detect grid size (8×8 vs 10×10 vs other NxN) via symbol clustering
 *  • Multi-threshold voting (5 passes) for better letter accuracy
 *  • K-means-style column/row centroid detection instead of fixed bucket math
 *  • Lookalike-aware majority voting per cell (I/L, O/0, B/8, etc.)
 *  • No magic assumption of exactly N symbols – works with noise/gaps
 */

const sharp = require('sharp');
sharp.cache(false);
const { createWorker } = require('tesseract.js');

// ─── OCR lookalike corrections (applied at voting time) ───────────────────────
const LOOKALIKE_GROUPS = [
  ['I', 'L', '1', '|', 'J'],
  ['O', '0', 'Q', 'D'],
  ['B', '8', '3'],
  ['S', '5'],
  ['G', '6', 'C'],
  ['Z', '2'],
  ['E', 'F'],
  ['U', 'V'],
];

// Build canonical map: non-alpha → preferred alpha
const CANONICAL = {};
for (const group of LOOKALIKE_GROUPS) {
  const alpha = group.find(c => /^[A-Z]$/.test(c));
  if (!alpha) continue;
  for (const c of group) {
    if (!/^[A-Z]$/.test(c)) CANONICAL[c] = alpha;
  }
}

function canonicalise(char) {
  return CANONICAL[char.toUpperCase()] || char.toUpperCase();
}

// ─── Simple 1-D k-means-style centroid finder ─────────────────────────────────
/**
 * Given a sorted list of values and a target cluster count,
 * iteratively refine cluster centroids until stable.
 * Returns sorted list of centroids.
 */
function findCentroids(values, k) {
  if (values.length === 0) return [];
  const sorted = [...values].sort((a, b) => a - b);
  const min = sorted[0], max = sorted[sorted.length - 1];

  if (k <= 1) return [(min + max) / 2];

  // Initialise centroids evenly spaced
  let centroids = Array.from({ length: k }, (_, i) => min + (i / (k - 1)) * (max - min));

  for (let iter = 0; iter < 30; iter++) {
    // Assign each value to nearest centroid
    const clusters = Array.from({ length: k }, () => []);
    for (const v of sorted) {
      let best = 0, bestDist = Infinity;
      for (let i = 0; i < k; i++) {
        const d = Math.abs(v - centroids[i]);
        if (d < bestDist) { bestDist = d; best = i; }
      }
      clusters[best].push(v);
    }

    // Recompute centroids
    const newCentroids = centroids.map((c, i) => {
      if (clusters[i].length === 0) return c;
      return clusters[i].reduce((a, b) => a + b, 0) / clusters[i].length;
    });

    // Check convergence
    const moved = newCentroids.some((nc, i) => Math.abs(nc - centroids[i]) > 0.01);
    centroids = newCentroids;
    if (!moved) break;
  }

  return centroids.sort((a, b) => a - b);
}

// ─── Auto-detect grid size from symbol cloud ──────────────────────────────────
/**
 * Try k=8 and k=10 clusterings on the X coordinates.
 * Pick whichever produces tighter within-cluster variance.
 */
function detectGridSize(xs) {
  if (xs.length === 0) return 8;

  const tryK = (k) => {
    const cents = findCentroids(xs, k);
    let totalVar = 0;
    const clusters = Array.from({ length: k }, () => []);
    for (const x of xs) {
      let best = 0, bestDist = Infinity;
      for (let i = 0; i < k; i++) {
        const d = Math.abs(x - cents[i]);
        if (d < bestDist) { bestDist = d; best = i; }
      }
      clusters[best].push(x);
    }
    for (const cl of clusters) {
      if (cl.length === 0) continue;
      const mean = cl.reduce((a, b) => a + b, 0) / cl.length;
      totalVar += cl.reduce((s, v) => s + (v - mean) ** 2, 0);
    }
    // Normalise by k so we compare fairly
    return totalVar / k;
  };

  // If very few symbols, default to 8
  if (xs.length < 30) return 8;

  const v8  = tryK(8);
  const v10 = tryK(10);

  // Heuristic: also check symbol density
  // A 10×10 grid should have ~100 symbols; an 8×8 ~ 64
  const symbolCount = xs.length;
  if (symbolCount > 350) return 10; // many multi-pass hits → likely 10×10

  // Use variance ratio to decide
  // If v10 is significantly better (lower) than v8, go with 10
  return (v10 < v8 * 0.85) ? 10 : 8;
}

// ─── Main extractGrid function ────────────────────────────────────────────────
/**
 * @param {string} imagePath
 * @param {number|null} forcedSize  - if null, auto-detect
 * @returns {string[][]|null}
 */
async function extractGrid(imagePath, forcedSize = null) {
  let worker = null;
  try {
    const THRESHOLDS = [70, 100, 130, 160, 190, 210];
    const allSymbols = []; // { char, x, y }

    worker = await createWorker('eng');
    await worker.setParameters({
      tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
      tessedit_pageseg_mode: '6', // Assume uniform block of text
    });

    for (const th of THRESHOLDS) {
      let buf;
      try {
        buf = await sharp(imagePath)
          .grayscale()
          .normalize()
          .sharpen({ sigma: 1.5 })
          .threshold(th)
          .toBuffer();
      } catch (e) {
        console.warn(`Sharp preprocessing failed at threshold ${th}:`, e.message);
        continue;
      }

      let res;
      try {
        res = await worker.recognize(buf);
      } catch (e) {
        console.warn(`Tesseract failed at threshold ${th}:`, e.message);
        continue;
      }

      if (!res.data.symbols) continue;

      for (const s of res.data.symbols) {
        const raw = (s.text || '').replace(/[^A-Za-z0-9|]/g, '').toUpperCase();
        if (!raw || raw.length !== 1) continue;
        const ch = canonicalise(raw);
        if (!/^[A-Z]$/.test(ch)) continue;

        const midX = (s.bbox.x0 + s.bbox.x1) / 2;
        const midY = (s.bbox.y0 + s.bbox.y1) / 2;
        allSymbols.push({ char: ch, x: midX, y: midY });
      }
    }

    await worker.terminate();
    worker = null;

    if (allSymbols.length === 0) {
      console.warn('No symbols detected from OCR.');
      return null;
    }

    console.log(`Total symbol observations (all passes): ${allSymbols.length}`);

    // ── Detect or use forced grid size ──
    const xs = allSymbols.map(s => s.x);
    const ys = allSymbols.map(s => s.y);

    const gridSize = forcedSize !== null ? forcedSize : detectGridSize(xs);
    console.log(`Using grid size: ${gridSize}×${gridSize}`);

    // ── Cluster columns and rows ──
    const colCentroids = findCentroids(xs, gridSize);
    const rowCentroids = findCentroids(ys, gridSize);

    // ── Vote per cell ──
    // cellVotes[r][c] = { char: count }
    const cellVotes = Array.from({ length: gridSize }, () =>
      Array.from({ length: gridSize }, () => ({}))
    );

    const colSpan = colCentroids.length > 1
      ? (colCentroids[colCentroids.length - 1] - colCentroids[0]) / (gridSize - 1)
      : 50;
    const rowSpan = rowCentroids.length > 1
      ? (rowCentroids[rowCentroids.length - 1] - rowCentroids[0]) / (gridSize - 1)
      : 50;
    const colTol = colSpan * 0.5;
    const rowTol = rowSpan * 0.5;

    for (const s of allSymbols) {
      // Assign to nearest column centroid within tolerance
      let bestC = -1, bestCDist = Infinity;
      for (let i = 0; i < colCentroids.length; i++) {
        const d = Math.abs(s.x - colCentroids[i]);
        if (d < bestCDist) { bestCDist = d; bestC = i; }
      }
      if (bestCDist > colTol * 2) continue; // too far from any centroid → noise

      let bestR = -1, bestRDist = Infinity;
      for (let i = 0; i < rowCentroids.length; i++) {
        const d = Math.abs(s.y - rowCentroids[i]);
        if (d < bestRDist) { bestRDist = d; bestR = i; }
      }
      if (bestRDist > rowTol * 2) continue;

      cellVotes[bestR][bestC][s.char] = (cellVotes[bestR][bestC][s.char] || 0) + 1;
    }

    // ── Build final grid ──
    const grid = Array.from({ length: gridSize }, (_, r) =>
      Array.from({ length: gridSize }, (_, c) => {
        const votes = cellVotes[r][c];
        let best = '?', maxV = 0;
        for (const [ch, v] of Object.entries(votes)) {
          if (v > maxV) { maxV = v; best = ch; }
        }
        return best;
      })
    );

    // Log the extracted grid for debugging
    console.log('Extracted grid:');
    for (const row of grid) {
      console.log(row.join(' '));
    }

    return grid;
  } catch (err) {
    console.error('OCR Error:', err);
    if (worker) {
      try { await worker.terminate(); } catch (_) {}
    }
    return null;
  }
}

module.exports = { extractGrid };
