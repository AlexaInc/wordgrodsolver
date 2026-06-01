/**
 * ocr.js — Dual-pass Tesseract OCR for word grid images
 *
 * Supports BOTH colour schemes automatically:
 *   • Light background, dark letters  (white/grey BG, black letters)
 *   • Dark background, light letters  (black BG, white letters)
 *
 * Strategy:
 *   Pass A — Full-image PSM 6 (uniform text block), multiple thresholds
 *     • Crops the border first (removes outer frame noise)
 *     • Maps each detected symbol bbox-centre to its grid cell
 *     • Weight 1 per vote
 *
 *   Pass B — Cell-by-cell PSM 10 (single character), multiple thresholds
 *     • Extracts each cell individually, 3× upscaled
 *     • Weight 2 per vote (cell-level is more reliable)
 *
 *   Final — majority vote per cell across both passes
 *
 * Background detection:
 *   Samples the mean pixel value of the border region.
 *   If mean < 128 → dark background → negate before thresholding
 *   so Tesseract always receives black-text-on-white.
 */

'use strict';

const sharp            = require('sharp');
sharp.cache(false);
const { createWorker } = require('tesseract.js');

// ─── Character normalisation ───────────────────────────────────────────────────
// Map digits/symbols Tesseract sometimes emits to their closest letter.
// We never remap one letter to another — that is the solver's job.
const CHAR_MAP = {
  '0': 'O', '1': 'I', '2': 'Z', '3': 'B',
  '4': 'A', '5': 'S', '6': 'G', '7': 'T',
  '8': 'B', '9': 'G', '|': 'I',
};

function clean(ch) {
  const u = (ch || '').toUpperCase();
  if (/^[A-Z]$/.test(u)) return u;
  return CHAR_MAP[u] || null;
}

// ─── Vote helpers ──────────────────────────────────────────────────────────────
function mergeVotes(a, b) {
  const out = { ...a };
  for (const [ch, v] of Object.entries(b)) out[ch] = (out[ch] || 0) + v;
  return out;
}

function pickWinner(votes) {
  let best = '?', maxV = 0;
  for (const [ch, v] of Object.entries(votes)) {
    if (v > maxV) { maxV = v; best = ch; }
  }
  return best;
}

// ─── Background detection ──────────────────────────────────────────────────────
/**
 * Detect whether the image has a dark background.
 * Samples a thin ring just inside the border region and computes mean luminance.
 * Returns true if background is dark (mean < 128) → need to negate for Tesseract.
 *
 * @param {string} imgPath
 * @param {number} border   – border thickness in pixels
 * @returns {Promise<boolean>}
 */
async function isDarkBackground(imgPath, border) {
  try {
    const meta  = await sharp(imgPath).metadata();
    const W = meta.width, H = meta.height;

    // Sample the four corner cells of the grid border area
    // Use a small strip just inside the outer border
    const sampleSize = Math.max(4, Math.round(border * 0.8));

    // Top-left corner sample
    const sample = await sharp(imgPath)
      .extract({
        left:   Math.max(0, border - sampleSize),
        top:    Math.max(0, border - sampleSize),
        width:  sampleSize * 2,
        height: sampleSize * 2,
      })
      .grayscale()
      .raw()
      .toBuffer();

    const mean = sample.reduce((s, v) => s + v, 0) / sample.length;
    const dark = mean < 128;
    console.log(`[OCR] Background mean luminance: ${mean.toFixed(1)} → ${dark ? 'DARK (will negate)' : 'LIGHT'}`);
    return dark;
  } catch (e) {
    console.warn('[OCR] Background detection failed, assuming light:', e.message);
    return false;
  }
}

// ─── Sharp pipeline builder ────────────────────────────────────────────────────
/**
 * Build a preprocessed image buffer from a region of the source image.
 * Handles both light and dark backgrounds:
 *   - Dark BG: negate BEFORE threshold so letters become dark on light BG
 *   - Light BG: threshold directly
 *
 * @param {string}  imgPath
 * @param {object}  region      – { left, top, width, height }
 * @param {number}  threshold   – binarisation threshold (0-255)
 * @param {boolean} darkBg      – true if image has dark background
 * @param {number}  scale       – upscale factor (1 = no scaling)
 * @returns {Promise<Buffer>}
 */
async function buildBuf(imgPath, region, threshold, darkBg, scale = 1) {
  let pipeline = sharp(imgPath).extract(region).grayscale().normalize();

  if (darkBg) {
    // Negate so white letters become black — Tesseract needs black text on white
    pipeline = pipeline.negate();
  }

  pipeline = pipeline.sharpen({ sigma: 1.2 });

  if (scale > 1) {
    pipeline = pipeline.resize(
      region.width  * scale,
      region.height * scale,
      { kernel: 'lanczos3' }
    );
  }

  pipeline = pipeline.threshold(threshold);
  return pipeline.toBuffer();
}

// ─── Pass A: full-image OCR (PSM 6) ───────────────────────────────────────────
async function passA(worker, imgPath, gridSize, border, darkBg) {
  const meta = await sharp(imgPath).metadata();
  const W = meta.width, H = meta.height;

  const cropL = border, cropT = border;
  const cropW = W - 2 * border;
  const cropH = H - 2 * border;
  const cellW = cropW / gridSize;
  const cellH = cropH / gridSize;

  const votes = Array.from({ length: gridSize }, () =>
    Array.from({ length: gridSize }, () => ({}))
  );

  // Use thresholds on the light side — after negate (dark BG) or direct (light BG)
  const THRESHOLDS = [80, 110, 140, 170];

  for (const th of THRESHOLDS) {
    let buf;
    try {
      buf = await buildBuf(
        imgPath,
        { left: cropL, top: cropT, width: cropW, height: cropH },
        th, darkBg, 1
      );
    } catch (e) {
      console.warn(`[PassA] preprocess th=${th}: ${e.message}`);
      continue;
    }

    let res;
    try {
      res = await worker.recognize(buf);
    } catch (e) {
      console.warn(`[PassA] tesseract th=${th}: ${e.message}`);
      continue;
    }

    if (!res.data.symbols) continue;

    for (const s of res.data.symbols) {
      const ch = clean(s.text);
      if (!ch) continue;
      const mx = (s.bbox.x0 + s.bbox.x1) / 2;
      const my = (s.bbox.y0 + s.bbox.y1) / 2;
      const c  = Math.min(gridSize - 1, Math.max(0, Math.floor(mx / cellW)));
      const r  = Math.min(gridSize - 1, Math.max(0, Math.floor(my / cellH)));
      votes[r][c][ch] = (votes[r][c][ch] || 0) + 1;
    }
  }

  return votes;
}

// ─── Pass B: cell-by-cell OCR (PSM 10) ────────────────────────────────────────
async function passB(worker, imgPath, gridSize, border, darkBg) {
  const meta = await sharp(imgPath).metadata();
  const W = meta.width, H = meta.height;

  const innerW = W - 2 * border;
  const innerH = H - 2 * border;
  const cellW  = innerW / gridSize;
  const cellH  = innerH / gridSize;

  const PAD    = 0.10; // 10% inset from each cell edge
  const SCALE  = 3;    // upscale for sharper OCR
  const WEIGHT = 2;    // cell-level votes count double

  const THRESHOLDS = [80, 110, 140, 170, 200];

  const votes = Array.from({ length: gridSize }, () =>
    Array.from({ length: gridSize }, () => ({}))
  );

  for (let r = 0; r < gridSize; r++) {
    for (let c = 0; c < gridSize; c++) {
      const left   = Math.round(border + c * cellW + cellW * PAD);
      const top    = Math.round(border + r * cellH + cellH * PAD);
      const width  = Math.max(3, Math.round(cellW * (1 - 2 * PAD)));
      const height = Math.max(3, Math.round(cellH * (1 - 2 * PAD)));

      for (const th of THRESHOLDS) {
        let buf;
        try {
          buf = await buildBuf(imgPath, { left, top, width, height }, th, darkBg, SCALE);
        } catch (e) {
          continue;
        }

        let res;
        try {
          res = await worker.recognize(buf);
        } catch (e) {
          continue;
        }

        const rawCh = (res.data.text || '').replace(/[^A-Za-z0-9|]/g, '').charAt(0);
        const ch    = clean(rawCh);
        if (ch && res.data.confidence > 15) {
          votes[r][c][ch] = (votes[r][c][ch] || 0) + WEIGHT;
        }
      }
    }
  }

  return votes;
}

// ─── Auto-detect grid size ─────────────────────────────────────────────────────
async function autoDetectSize(imgPath, border, darkBg) {
  const meta = await sharp(imgPath).metadata();
  const W = meta.width, H = meta.height;

  let buf;
  try {
    buf = await buildBuf(
      imgPath,
      { left: border, top: border, width: W - 2 * border, height: H - 2 * border },
      130, darkBg, 1
    );
  } catch (e) {
    console.warn('[OCR] autoDetect preprocess failed:', e.message);
    return 8;
  }

  const worker = await createWorker('eng');
  await worker.setParameters({
    tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
    tessedit_pageseg_mode:   '6',
  });
  const res = await worker.recognize(buf);
  await worker.terminate();

  const count = (res.data.symbols || []).filter(s => /^[A-Z]$/i.test(s.text)).length;
  const size  = count > 160 ? 10 : 8;
  console.log(`[OCR] Auto-detect: ${count} symbols → ${size}×${size}`);
  return size;
}

// ─── Main ─────────────────────────────────────────────────────────────────────
/**
 * @param {string}      imagePath
 * @param {number|null} forcedSize   – 8 or 10; null = auto-detect
 * @returns {Promise<string[][]|null>}
 */
async function extractGrid(imagePath, forcedSize = null) {
  let workerA = null;
  let workerB = null;

  try {
    const meta   = await sharp(imagePath).metadata();
    const minDim = Math.min(meta.width, meta.height);
    const border = Math.round(minDim * 0.055);

    console.log(`[OCR] Image ${meta.width}×${meta.height}, border=${border}px`);

    // ── Detect background colour scheme ──────────────────────────────────────
    const darkBg = await isDarkBackground(imagePath, border);

    // ── Determine grid size ───────────────────────────────────────────────────
    const gridSize = (forcedSize !== null)
      ? forcedSize
      : await autoDetectSize(imagePath, border, darkBg);

    console.log(`[OCR] Grid size: ${gridSize}×${gridSize}`);

    // ── Pass A: full-image PSM 6 ──────────────────────────────────────────────
    workerA = await createWorker('eng');
    await workerA.setParameters({
      tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
      tessedit_pageseg_mode:   '6',
    });
    const votesA = await passA(workerA, imagePath, gridSize, border, darkBg);
    await workerA.terminate();
    workerA = null;

    // ── Pass B: cell-by-cell PSM 10 ───────────────────────────────────────────
    workerB = await createWorker('eng');
    await workerB.setParameters({
      tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
      tessedit_pageseg_mode:   '10',
    });
    const votesB = await passB(workerB, imagePath, gridSize, border, darkBg);
    await workerB.terminate();
    workerB = null;

    // ── Merge votes ───────────────────────────────────────────────────────────
    const mergedVotes = Array.from({ length: gridSize }, (_, r) =>
      Array.from({ length: gridSize }, (_, c) =>
        mergeVotes(votesA[r][c], votesB[r][c])
      )
    );

    // ── Pass C: rescue unknown cells ──────────────────────────────────────────
    // For any cell that is still '?' after the two main passes, run an
    // aggressive extra-contrast re-try with more threshold variants.
    // This handles thin letters like I/L on dark backgrounds.
    const meta2   = await sharp(imagePath).metadata();
    const innerW2 = meta2.width  - 2 * border;
    const innerH2 = meta2.height - 2 * border;
    const cellW2  = innerW2 / gridSize;
    const cellH2  = innerH2 / gridSize;

    let rescueWorker = null;
    const unknownCells = [];
    for (let r = 0; r < gridSize; r++) {
      for (let c = 0; c < gridSize; c++) {
        if (pickWinner(mergedVotes[r][c]) === '?') unknownCells.push({ r, c });
      }
    }

    if (unknownCells.length > 0) {
      console.log(`[OCR] PassC: rescuing ${unknownCells.length} unknown cell(s)...`);

      // Try multiple PSM modes — thin letters like I/L need PSM 7 or 8
      const RESCUE_PSM_MODES = ['10', '7', '8', '13'];
      const RESCUE_THRESHOLDS = [50, 70, 90, 110, 130, 150, 170, 190, 210, 230];
      const RESCUE_SCALE = 6; // larger upscale for thin single-stroke characters

      for (const psmMode of RESCUE_PSM_MODES) {
        rescueWorker = await createWorker('eng');
        await rescueWorker.setParameters({
          tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
          tessedit_pageseg_mode:   psmMode,
        });

        for (const { r, c } of unknownCells) {
          // Skip if already resolved in a previous PSM pass
          if (pickWinner(mergedVotes[r][c]) !== '?') continue;

          // Use full cell (minimal padding) for rescue to capture thin strokes
          const left   = Math.round(border + c * cellW2 + cellW2 * 0.02);
          const top    = Math.round(border + r * cellH2 + cellH2 * 0.02);
          const width  = Math.max(3, Math.round(cellW2 * 0.96));
          const height = Math.max(3, Math.round(cellH2 * 0.96));

          for (const th of RESCUE_THRESHOLDS) {
            try {
              const buf = await buildBuf(
                imagePath, { left, top, width, height },
                th, darkBg, RESCUE_SCALE
              );
              const res = await rescueWorker.recognize(buf);
              const rawCh = (res.data.text || '').replace(/[^A-Za-z0-9|]/g, '').charAt(0);
              const ch = clean(rawCh);
              // Only accept high-confidence votes in rescue pass to avoid noise
              if (ch && res.data.confidence > 40) {
                mergedVotes[r][c][ch] = (mergedVotes[r][c][ch] || 0) + 1;
              }
            } catch (_) {}
          }
        }

        await rescueWorker.terminate();
        rescueWorker = null;
      }

      for (const { r, c } of unknownCells) {
        console.log(`[OCR] PassC cell[${r}][${c}]: votes=${JSON.stringify(mergedVotes[r][c])} → ${pickWinner(mergedVotes[r][c])}`);
      }
    }

    // ── Build final grid ──────────────────────────────────────────────────────
    const grid = Array.from({ length: gridSize }, (_, r) =>
      Array.from({ length: gridSize }, (_, c) =>
        pickWinner(mergedVotes[r][c])
      )
    );

    console.log('[OCR] Extracted grid:');
    for (const row of grid) console.log('  ' + row.join(' '));

    // Sanity check: if more than 40% of cells are '?' → likely failed
    const totalCells = gridSize * gridSize;
    const unknowns   = grid.flat().filter(c => c === '?').length;
    if (unknowns > totalCells * 0.4) {
      console.error(`[OCR] Too many unknown cells (${unknowns}/${totalCells}) — extraction unreliable`);
      return null;
    }

    return grid;

  } catch (err) {
    console.error('[OCR] Fatal error:', err);
    for (const w of [workerA, workerB]) {
      if (w) try { await w.terminate(); } catch (_) {}
    }
    return null;
  }
}

module.exports = { extractGrid };
