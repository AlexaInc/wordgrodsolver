/**
 * solver.js — Word Grid solver
 *
 * Key design decisions:
 *  • LOOKALIKES are ONE-WAY only: "OCR may read char X as char Y" means
 *    when we're looking for X, we also accept Y in the grid.
 *    It does NOT mean when looking for Y we accept X — that caused false matches.
 *  • '?' wildcard: OCR-uncertain cells match any target letter.
 *  • Pattern search collects every N-char path starting at cells that
 *    match the pattern's first letter (exact or lookalike).
 *  • Deduplication by match string before returning.
 */

'use strict';

// ─── Directions ───────────────────────────────────────────────────────────────
const DIRECTIONS = [
  { r:  0, c:  1, name: 'LtoR'  },
  { r:  0, c: -1, name: 'RtoL'  },
  { r:  1, c:  0, name: 'UtoD'  },
  { r: -1, c:  0, name: 'DtoU'  },
  { r:  1, c:  1, name: 'diagDR'},
  { r: -1, c: -1, name: 'diagUL'},
  { r:  1, c: -1, name: 'diagDL'},
  { r: -1, c:  1, name: 'diagUR'},
];

// ─── Lookalike table (ONE-WAY) ────────────────────────────────────────────────
// "OCR may misread target letter X as one of these grid characters."
// When searching for X, we also accept any char listed here.
// This is intentionally NOT symmetric — visual similarity is asymmetric.
// e.g. OCR misreads I as L (thin vertical), but does NOT misread L as I.
const LOOKALIKES = {
  'A': new Set(['4']),
  'B': new Set(['8', '3']),
  'C': new Set(['G', 'O', 'Q']),
  'D': new Set(['O', 'Q', '0']),
  'E': new Set(['F']),
  'F': new Set(['E']),
  'G': new Set(['C', '6', 'Q']),
  'H': new Set([]),
  'I': new Set(['L', '1', '|', 'J']),
  'J': new Set(['I']),
  'K': new Set(['X']),
  'L': new Set(['I', '1', '|']),
  'M': new Set(['N']),
  'N': new Set(['M']),
  'O': new Set(['0', 'Q', 'D']),
  'P': new Set([]),
  'Q': new Set(['O', 'G', '0']),
  'R': new Set([]),
  'S': new Set(['5', '8']),
  'T': new Set(['7']),
  'U': new Set(['V']),
  'V': new Set(['U']),
  'W': new Set([]),
  'X': new Set(['K']),
  'Y': new Set([]),
  'Z': new Set(['2', '7']),
};

/**
 * Does gridChar match targetChar?
 *  - Exact match always wins.
 *  - '?' in grid = OCR-uncertain = wildcard, matches any target.
 *  - One-way lookalike: target's known OCR substitutes are checked.
 */
function charMatch(target, gridChar) {
  if (!gridChar || gridChar === ' ') return false;
  if (gridChar === '?') return true;             // OCR uncertain → wildcard
  const t = target.toUpperCase();
  const g = gridChar.toUpperCase();
  if (t === g) return true;                      // exact match
  const alts = LOOKALIKES[t];
  return !!(alts && alts.has(g));                // known OCR substitute
}

function inBounds(grid, r, c) {
  return r >= 0 && r < grid.length &&
         c >= 0 && c < (grid[r] ? grid[r].length : 0);
}

// ─── Solver ───────────────────────────────────────────────────────────────────
/**
 * @param {string[][]} grid
 * @param {{ pattern?: string, word?: string }[]} words
 * @returns {Object}
 */
function solve(grid, words) {
  const results = {};
  const rows = grid.length;
  if (rows === 0) return results;

  for (const wordObj of words) {
    const isExact   = wordObj.word && !wordObj.word.includes('-');
    const isPattern = !!wordObj.pattern;

    if (isExact) {
      // ── Exact word search ────────────────────────────────────────────────
      const target = wordObj.word.toUpperCase();
      const len    = target.length;
      let found    = false;

      outer:
      for (let r = 0; r < rows && !found; r++) {
        for (let c = 0; c < grid[r].length && !found; c++) {
          if (!charMatch(target[0], grid[r][c])) continue;
          for (const dir of DIRECTIONS) {
            const er = r + dir.r * (len - 1);
            const ec = c + dir.c * (len - 1);
            if (!inBounds(grid, er, ec)) continue;
            let ok = true, candidate = '';
            for (let i = 0; i < len; i++) {
              const nr = r + dir.r * i, nc = c + dir.c * i;
              if (!inBounds(grid, nr, nc) || !charMatch(target[i], grid[nr][nc])) {
                ok = false; break;
              }
              candidate += grid[nr][nc];
            }
            if (ok) {
              results[wordObj.word] = { r, c, dir: dir.name, match: candidate };
              found = true; break;
            }
          }
        }
      }

    } else if (isPattern) {
      // ── Pattern search ───────────────────────────────────────────────────
      const pattern   = wordObj.pattern.toUpperCase();
      const startChar = pattern[0];
      const len       = pattern.length;
      const hits      = [];

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < grid[r].length; c++) {
          if (!charMatch(startChar, grid[r][c])) continue;

          for (const dir of DIRECTIONS) {
            const er = r + dir.r * (len - 1);
            const ec = c + dir.c * (len - 1);
            if (!inBounds(grid, er, ec)) continue;

            let ok = true, candidate = '';
            for (let i = 0; i < len; i++) {
              const nr = r + dir.r * i, nc = c + dir.c * i;
              if (!inBounds(grid, nr, nc)) { ok = false; break; }
              const ch = grid[nr][nc];
              if (!ch || ch === ' ') { ok = false; break; }
              candidate += ch;
            }
            if (ok && candidate.length === len) {
              hits.push({ r, c, dir: dir.name, match: candidate });
            }
          }
        }
      }

      // Deduplicate by match string
      const seen   = new Set();
      const unique = hits.filter(h => !seen.has(h.match) && seen.add(h.match));
      if (unique.length > 0) results[pattern] = unique;
    }
  }

  return results;
}

// ─── Leaderboard ──────────────────────────────────────────────────────────────
let leaderboard = [];
const getWordScore  = word  => word.length * 10;
const getLeaderboard = ()  => leaderboard;
function recordScore(userName, score) {
  leaderboard.push({ name: userName, score, date: new Date().toISOString() });
  leaderboard.sort((a, b) => b.score - a.score);
  leaderboard = leaderboard.slice(0, 10);
}

module.exports = { solve, charMatch, getWordScore, recordScore, getLeaderboard };
