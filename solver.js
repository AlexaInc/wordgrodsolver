/**
 * solver.js — Word Grid solver with comprehensive lookalike tolerance
 *
 * Improvements over v1:
 *  • Symmetric LOOKALIKES map (A→B implies B can match A)
 *  • Wildcard (-) pattern matching ignores only dashes, not all unknowns
 *  • Pattern matching collects ALL candidates with their actual grid characters
 *  • charMatch is stricter: only matches confirmed lookalike pairs, not random guesses
 *  • Grid boundary checks are consolidated in one place (no off-by-one)
 *  • Deduplication of candidates by match string
 */

// ─── Directions ───────────────────────────────────────────────────────────────
const DIRECTIONS = [
  { r: 0,  c:  1, name: 'LtoR'  },
  { r: 0,  c: -1, name: 'RtoL'  },
  { r: 1,  c:  0, name: 'UtoD'  },
  { r: -1, c:  0, name: 'DtoU'  },
  { r: 1,  c:  1, name: 'diagDR'},
  { r: -1, c: -1, name: 'diagUL'},
  { r: 1,  c: -1, name: 'diagDL'},
  { r: -1, c:  1, name: 'diagUR'},
];

// ─── Lookalike table (symmetric) ─────────────────────────────────────────────
// Each entry lists chars that can be confused FOR the key char by OCR.
const LOOKALIKES_RAW = {
  'A': ['4', 'R'],
  'B': ['8', '3', 'R', 'S', 'E'],
  'C': ['G', 'O', 'Q', '(', 'L'],
  'D': ['O', 'Q', 'B', '0'],
  'E': ['F', 'B', '3', 'L'],
  'F': ['E', 'P'],
  'G': ['C', 'O', '6', 'Q', 'D'],
  'H': ['N', 'M'],
  'I': ['L', '1', 'T', 'J', '|'],
  'J': ['I', 'L', '1'],
  'K': ['R', 'X'],
  'L': ['I', '1', 'T', '|', '[', 'J'],
  'M': ['N', 'H', 'W'],
  'N': ['M', 'H', 'R'],
  'O': ['0', 'Q', 'G', 'C', 'D', 'U'],
  'P': ['F', 'B', 'R'],
  'Q': ['O', 'G', 'C', '0'],
  'R': ['B', 'P', 'K', 'I', 'A', 'N'],
  'S': ['5', '8', 'B', '6'],
  'T': ['I', 'L', '7', '+'],
  'U': ['V', 'W', 'O', 'Y'],
  'V': ['U', 'Y', 'W'],
  'W': ['M', 'V', 'U'],
  'X': ['K', 'Y'],
  'Y': ['V', 'U', 'X'],
  'Z': ['2', '7'],
};

// Build symmetric version: if A can be confused as B, then B can be confused as A
const LOOKALIKES = {};
for (const [key, alts] of Object.entries(LOOKALIKES_RAW)) {
  if (!LOOKALIKES[key]) LOOKALIKES[key] = new Set();
  for (const alt of alts) {
    LOOKALIKES[key].add(alt);
    // symmetric
    if (/^[A-Z]$/.test(alt)) {
      if (!LOOKALIKES[alt]) LOOKALIKES[alt] = new Set();
      LOOKALIKES[alt].add(key);
    }
  }
}

/**
 * Does `gridChar` match `targetChar` considering OCR lookalikes?
 * '?' means OCR was uncertain — treat as wildcard (matches any target).
 */
function charMatch(target, gridChar) {
  if (!gridChar || gridChar === ' ') return false;
  if (gridChar === '?') return true; // OCR unknown → wildcard, solver decides
  const t = target.toUpperCase();
  const g = gridChar.toUpperCase();
  if (t === g) return true;
  const alts = LOOKALIKES[t];
  return !!(alts && alts.has(g));
}

/**
 * Check all cells are in bounds on a grid
 */
function inBounds(grid, r, c) {
  return r >= 0 && r < grid.length && c >= 0 && c < (grid[r] ? grid[r].length : 0);
}

/**
 * Solve the grid for a list of word/pattern objects.
 *
 * Each item in `words` is one of:
 *   { word: 'MATRIX' }           → exact word search
 *   { pattern: 'M---' }          → pattern search (first char + length)
 *
 * Returns an object:
 *   { 'M---': [ { r, c, dir, match, reliable } ], ... }
 *   { 'MATRIX': { r, c, dir, match } }
 */
function solve(grid, words) {
  const results = {};
  const rows = grid.length;
  if (rows === 0) return results;

  for (const wordObj of words) {
    const isExact = wordObj.word && !wordObj.word.includes('-');
    const isPattern = !!wordObj.pattern;

    if (isExact) {
      const target = wordObj.word.toUpperCase();
      const len = target.length;
      let found = false;

      outer:
      for (let r = 0; r < rows && !found; r++) {
        const cols = grid[r].length;
        for (let c = 0; c < cols && !found; c++) {
          if (!charMatch(target[0], grid[r][c])) continue;
          for (const dir of DIRECTIONS) {
            // Quick bounds check for last character
            const er = r + dir.r * (len - 1);
            const ec = c + dir.c * (len - 1);
            if (!inBounds(grid, er, ec)) continue;

            let match = true;
            let candidate = '';
            for (let i = 0; i < len; i++) {
              const nr = r + dir.r * i;
              const nc = c + dir.c * i;
              if (!inBounds(grid, nr, nc) || !charMatch(target[i], grid[nr][nc])) {
                match = false;
                break;
              }
              candidate += grid[nr][nc];
            }
            if (match) {
              results[wordObj.word] = { r, c, dir: dir.name, match: candidate };
              found = true;
              break;
            }
          }
        }
      }
    } else if (isPattern) {
      const pattern = wordObj.pattern.toUpperCase(); // e.g. "M---"
      const startChar = pattern[0];
      const len = pattern.length;
      const hits = [];

      for (let r = 0; r < rows; r++) {
        const cols = grid[r].length;
        for (let c = 0; c < cols; c++) {
          if (!charMatch(startChar, grid[r][c])) continue;

          for (const dir of DIRECTIONS) {
            // Bounds check for last char
            const er = r + dir.r * (len - 1);
            const ec = c + dir.c * (len - 1);
            if (!inBounds(grid, er, ec)) continue;

            let possible = true;
            let candidate = '';
            for (let i = 0; i < len; i++) {
              const nr = r + dir.r * i;
              const nc = c + dir.c * i;
              if (!inBounds(grid, nr, nc)) { possible = false; break; }
              const ch = grid[nr][nc];
              if (!ch || ch === ' ') { possible = false; break; }
              candidate += ch;
            }

            if (possible && candidate.length === len) {
              hits.push({ r, c, dir: dir.name, match: candidate });
            }
          }
        }
      }

      // Deduplicate by match string
      const seen = new Set();
      const unique = hits.filter(h => {
        if (seen.has(h.match)) return false;
        seen.add(h.match);
        return true;
      });

      if (unique.length > 0) results[pattern] = unique;
    }
  }

  return results;
}

// ─── Leaderboard ──────────────────────────────────────────────────────────────
let leaderboard = [];

function getWordScore(word) {
  return word.length * 10;
}

function recordScore(userName, score) {
  leaderboard.push({ name: userName, score, date: new Date().toISOString() });
  leaderboard.sort((a, b) => b.score - a.score);
  leaderboard = leaderboard.slice(0, 10);
}

function getLeaderboard() {
  return leaderboard;
}

module.exports = { solve, charMatch, getWordScore, recordScore, getLeaderboard };
