/**
 * bot.js — Word Grid Solver Bot (GramJS / pure MTProto)
 *
 * Uses the `telegram` npm package (GramJS) with a BOT_TOKEN for pure MTProto.
 * No Telegraf, no HTTP Bot API polling — raw MTProto layer.
 *
 * Required env vars:
 *   BOT_TOKEN   – Telegram bot token (from @BotFather)
 *   API_ID      – Telegram API ID   (from https://my.telegram.org/apps)
 *   API_HASH    – Telegram API hash (from https://my.telegram.org/apps)
 *
 * Optional:
 *   PORT        – HTTP dashboard port (default 7860)
 */

'use strict';

require('dotenv').config();

const { TelegramClient } = require('telegram');
const { StringSession }  = require('telegram/sessions');
const { NewMessage }     = require('telegram/events');

const express = require('express');
const fs      = require('fs');
const path    = require('path');

const { extractGrid } = require('./ocr');
const { solve }       = require('./solver');

// ─── Environment ──────────────────────────────────────────────────────────────
const BOT_TOKEN = process.env.BOT_TOKEN;
const API_ID    = parseInt(process.env.API_ID  || '0', 10);
const API_HASH  = process.env.API_HASH || '';

if (!BOT_TOKEN) {
  console.error('[FATAL] BOT_TOKEN is required.');
  process.exit(1);
}
if (!API_ID || !API_HASH) {
  console.error('[FATAL] API_ID and API_HASH are required for GramJS MTProto.');
  console.error('        Get them from https://my.telegram.org/apps');
  process.exit(1);
}

// ─── Dictionary ───────────────────────────────────────────────────────────────
const dictionaryPath = path.join(__dirname, 'node_modules/check-word/words/en.txt');
const dictionary = new Set();
try {
  const data = fs.readFileSync(dictionaryPath, 'utf8');
  for (const line of data.split('\n')) {
    const w = line.trim().toLowerCase();
    if (w.length >= 3) dictionary.add(w);
  }
  console.log(`[Dict] Loaded ${dictionary.size} words.`);
} catch (err) {
  console.error('[Dict] Failed to load dictionary:', err.message);
}

function isWord(w) {
  return dictionary.has((w || '').toLowerCase());
}

// ─── Stats ────────────────────────────────────────────────────────────────────
const stats = {
  imagesProcessed: 0,
  wordsFound:      0,
  startTime:       Date.now(),
  botUsername:     'loading...',
};

// ─── Utility helpers ──────────────────────────────────────────────────────────
function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function deleteFile(filePath) {
  for (let i = 0; i < 5; i++) {
    try {
      if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
      return;
    } catch (_) {
      await sleep(800);
    }
  }
}

// ─── Caption parsers ──────────────────────────────────────────────────────────
/**
 * Parse word patterns from caption text — single left-to-right pass.
 * Supports all of:
 *   M--- (4)          → pattern + explicit length hint
 *   M----             → standalone dashes (2+ required)
 *   W---- H--- (4)    → multiple patterns on one line
 *   Find M--- and P-- → mixed prose + patterns
 */
function parsePatterns(text) {
  const results = [];
  const seen    = new Set();

  // Single combined regex: captures "X---" optionally followed by " (N)"
  // Minimum 2 dashes (so 3-letter minimum words).
  const re = /([A-Z])(-+)(?:\s*\(\d+\))?/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m[2].length < 2) continue; // skip single-dash noise
    const pattern = (m[1] + m[2]).toUpperCase();
    if (!seen.has(pattern)) {
      seen.add(pattern);
      results.push({ pattern });
    }
  }

  return results;
}

/**
 * Detect grid size hint from caption text.
 * Returns a number (forced size) or null (let OCR auto-detect).
 */
function detectGridSizeFromCaption(text) {
  const upper = text.toUpperCase();

  // Explicit NxN e.g. "10x10" or "8x8"
  const sizeMatch = text.match(/\b(\d+)\s*[xX×]\s*\1\b/);
  if (sizeMatch) {
    const n = parseInt(sizeMatch[1], 10);
    if (n >= 4 && n <= 15) {
      console.log(`[Grid] Caption explicit size: ${n}×${n}`);
      return n;
    }
  }

  if (
    upper.includes('HARD MODE') ||
    upper.includes('10X10') ||
    upper.includes('10 X 10')
  ) {
    console.log('[Grid] Caption phrase → 10×10');
    return 10;
  }

  return null; // let OCR auto-detect
}

// ─── Result formatter ─────────────────────────────────────────────────────────
function formatResults(results, grid, patterns) {
  let msg = '🎯 <b>WORD GRID RESULTS</b>\n\n';
  let foundAny = false;

  for (const p of patterns) {
    const key = p.pattern || p.word;
    if (!key) continue;

    const entry = results[key];
    if (!entry) {
      msg += `❓ ${key}: not found in grid\n`;
      continue;
    }

    if (Array.isArray(entry)) {
      // Pattern search — filter candidates against dictionary
      const startChar = key[0].toUpperCase();
      const wordMatches = new Set();

      for (const hit of entry) {
        const raw = hit.match.toUpperCase();

        if (isWord(raw)) {
          wordMatches.add(raw);
          continue;
        }

        // OCR may have mis-read the first character — force the known start char
        const forced = startChar + raw.slice(1);
        if (isWord(forced)) {
          wordMatches.add(forced);
          continue;
        }

        console.log(`[Debug] Rejected non-word for "${key}": ${raw}`);
      }

      if (wordMatches.size > 0) {
        const list = [...wordMatches].map(w => `<code>${w}</code>`).join(', ');
        msg += `✅ ${key}: ${list}\n`;
        stats.wordsFound += wordMatches.size;
        foundAny = true;
      } else {
        msg += `❓ ${key}: no dictionary words matched\n`;
      }
    } else {
      // Exact word search result
      msg += `✅ <code>${entry.match}</code> @ [${entry.r},${entry.c}] ${entry.dir}\n`;
      foundAny = true;
    }
  }

  if (!foundAny) {
    msg += '😔 No real word matches found for these patterns.\n';
    msg += 'Tip: check that your caption patterns match the grid letters.\n';
  }

  // Always show the extracted grid for verification
  msg += '\n🔍 <b>Extracted Grid:</b>\n';
  msg += '<pre>' + grid.map(row => row.join(' ')).join('\n') + '</pre>';

  return msg;
}

// ─── GramJS Bot ───────────────────────────────────────────────────────────────
async function startBot() {
  const session = new StringSession(''); // blank = new session each time (fine for bots)

  const client = new TelegramClient(session, API_ID, API_HASH, {
    connectionRetries: 10,
    retryDelay:        2000,
    autoReconnect:     true,
    useWSS:            false,
  });

  console.log('[GramJS] Connecting via MTProto...');

  await client.start({
    botAuthToken: BOT_TOKEN,
  });

  console.log('[GramJS] Connected successfully.');

  const me = await client.getMe();
  stats.botUsername = me.username || 'bot';
  console.log(`[Bot] Running as @${stats.botUsername}`);

  // ── Message handler ──────────────────────────────────────────────────────────
  client.addEventHandler(async (event) => {
    const msg = event.message;
    if (!msg) return;

    const chatId  = msg.peerId;
    const caption = (msg.message || '').trim();

    // /start command
    if (caption === '/start' || caption.startsWith('/start ')) {
      await client.sendMessage(chatId, {
        message: [
          '👋 <b>Word Grid Solver Bot</b>',
          '',
          'Send me a word grid image with patterns in the caption.',
          '',
          '<b>Caption format:</b>',
          '<code>M--- (4) P------- (8) S----- (6)</code>',
          '',
          '📐 Supports <b>8×8 and 10×10</b> grids (auto-detected)',
          'Override with: <code>10x10</code> in caption',
        ].join('\n'),
        parseMode: 'html',
      });
      return;
    }

    // Only handle messages that carry a photo
    const hasPhoto = msg.media && (
      msg.media.className === 'MessageMediaPhoto' ||
      (msg.media.document &&
       msg.media.document.mimeType &&
       msg.media.document.mimeType.startsWith('image/'))
    );

    if (!hasPhoto) return;

    // ── Download image to disk via MTProto ─────────────────────────────────────
    // GramJS downloadMedia accepts a STRING path as outputFile → writes file,
    // returns the path. Do NOT pass Buffer constructor — that causes the
    // "writer.write is not a function" error.
    const imagePath = path.join(
      __dirname,
      `grid_${Date.now()}_${Math.random().toString(36).slice(2)}.jpg`
    );

    try {
      await client.sendMessage(chatId, {
        message: '🔍 Downloading and processing your grid image...',
      });

      // Pass the destination path string — GramJS streams the file there via MTProto
      const result = await client.downloadMedia(msg, {
        outputFile: imagePath,
      });

      // result is the path string when outputFile is a string path
      if (!result) throw new Error('downloadMedia returned null/undefined');
      if (!fs.existsSync(imagePath)) throw new Error('File not written to disk');

      console.log(`[Download] Saved to ${imagePath} (${fs.statSync(imagePath).size} bytes)`);

    } catch (dlErr) {
      console.error('[Download] Failed:', dlErr.message);
      await client.sendMessage(chatId, {
        message: '❌ Failed to download the image. Please try again.',
      });
      await deleteFile(imagePath);
      return;
    }

    // ── Process the image ──────────────────────────────────────────────────────
    try {
      stats.imagesProcessed++;

      const forcedSize = detectGridSizeFromCaption(caption);

      const grid = await extractGrid(imagePath, forcedSize);

      if (!grid || grid.length === 0) {
        await client.sendMessage(chatId, {
          message: '❌ Could not extract a grid from the image.\nMake sure the grid letters are clearly visible.',
        });
        return;
      }

      const patterns = parsePatterns(caption);

      if (patterns.length === 0) {
        const noPatMsg =
          '📋 <b>Grid extracted</b> (no patterns in caption):\n\n' +
          '<pre>' + grid.map(r => r.join(' ')).join('\n') + '</pre>\n\n' +
          'Add patterns like <code>M--- (4)</code> to find words!';
        await client.sendMessage(chatId, { message: noPatMsg, parseMode: 'html' });
        return;
      }

      const results = solve(grid, patterns);
      const reply   = formatResults(results, grid, patterns);

      await client.sendMessage(chatId, { message: reply, parseMode: 'html' });

    } catch (err) {
      console.error('[Handler] Processing error:', err);
      await client.sendMessage(chatId, {
        message: '🚨 An error occurred while processing. Please try again.',
      });
    } finally {
      await deleteFile(imagePath);
    }

  }, new NewMessage({}));

  console.log('[Bot] Listening for messages...');

  // Graceful shutdown
  const shutdown = async (sig) => {
    console.log(`[Bot] ${sig} received — disconnecting...`);
    try { await client.disconnect(); } catch (_) {}
    process.exit(0);
  };
  process.once('SIGINT',  () => shutdown('SIGINT'));
  process.once('SIGTERM', () => shutdown('SIGTERM'));
}

// ─── Express dashboard ────────────────────────────────────────────────────────
const app  = express();
const PORT = parseInt(process.env.PORT || '7860', 10);

app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/stats', (_req, res) => {
  res.json({
    ...stats,
    uptime: Math.floor((Date.now() - stats.startTime) / 1000),
  });
});

app.listen(PORT, () => {
  console.log(`[Dashboard] Running on port ${PORT}`);
});

// ─── Boot ─────────────────────────────────────────────────────────────────────
startBot().catch(err => {
  console.error('[FATAL] Bot startup failed:', err);
  process.exit(1);
});
