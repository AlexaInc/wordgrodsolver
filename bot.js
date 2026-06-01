/**
 * bot.js — Word Grid Solver Bot (pure GramJS MTProto, zero Bot API HTTP calls)
 *
 * Download strategy (MTProto only, two attempts):
 *   1. Re-fetch message via client.getMessages() → fresh fileReference
 *      → downloadFileV2 with explicit InputPhotoFileLocation + correct dcId
 *   2. client.downloadMedia() on the re-fetched message (GramJS full flow)
 *
 * Both private chats and groups are handled identically — GramJS resolves
 * the entity and handles DC auth export automatically.
 *
 * Required env vars:
 *   BOT_TOKEN   – Telegram bot token  (from @BotFather)
 *   API_ID      – Telegram API ID     (from https://my.telegram.org/apps)
 *   API_HASH    – Telegram API hash   (from https://my.telegram.org/apps)
 *
 * Optional:
 *   PORT        – HTTP dashboard port (default 7860)
 */

'use strict';

require('dotenv').config();

const { TelegramClient } = require('telegram');
const { StringSession }  = require('telegram/sessions');
const { NewMessage }     = require('telegram/events');
const { Api }            = require('telegram');
const bigInt             = require('big-integer');

const express = require('express');
const fs      = require('fs');
const path    = require('path');

const { extractGrid } = require('./ocr');
const { solve }       = require('./solver');

// ─── downloadFileV2 from GramJS internals ─────────────────────────────────────
const { downloadFileV2 } = require('./node_modules/telegram/client/downloads');

// ─── Environment ──────────────────────────────────────────────────────────────
const BOT_TOKEN = process.env.BOT_TOKEN;
const API_ID    = parseInt(process.env.API_ID  || '0', 10);
const API_HASH  = process.env.API_HASH || '';

if (!BOT_TOKEN) { console.error('[FATAL] BOT_TOKEN is required.'); process.exit(1); }
if (!API_ID || !API_HASH) {
  console.error('[FATAL] API_ID and API_HASH are required (https://my.telegram.org/apps)');
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

function isWord(w) { return dictionary.has((w || '').toLowerCase()); }

// ─── Stats ────────────────────────────────────────────────────────────────────
const stats = {
  imagesProcessed: 0,
  wordsFound:      0,
  startTime:       Date.now(),
  botUsername:     'loading...',
};

// ─── Utilities ────────────────────────────────────────────────────────────────
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function deleteFile(p) {
  for (let i = 0; i < 5; i++) {
    try { if (fs.existsSync(p)) fs.unlinkSync(p); return; } catch (_) { await sleep(500); }
  }
}

// ─── Photo size picker ────────────────────────────────────────────────────────
/**
 * Return the best (largest, non-stripped, non-progressive, non-empty) photo size.
 * Telegram quality tiers: w > y > d > x > c > m > b > a > s
 */
function pickBestSize(sizes) {
  if (!sizes || sizes.length === 0) return null;
  for (const t of ['w', 'y', 'd', 'x', 'c', 'm', 'b', 'a', 's']) {
    const s = sizes.find(sz =>
      sz.type === t &&
      !(sz instanceof Api.PhotoStrippedSize) &&
      !(sz instanceof Api.PhotoSizeEmpty) &&
      !(sz instanceof Api.PhotoSizeProgressive)
    );
    if (s) return s;
  }
  return sizes.find(sz =>
    !(sz instanceof Api.PhotoStrippedSize) &&
    !(sz instanceof Api.PhotoSizeEmpty)
  ) || null;
}

// ─── MTProto image download ───────────────────────────────────────────────────
/**
 * Get the entity for a peer — works for PeerUser, PeerChat, PeerChannel.
 * GramJS handles all three cases when you pass the peerId directly.
 */
async function getEntitySafe(client, peerId) {
  try {
    return await client.getEntity(peerId);
  } catch (e) {
    // Last resort: use the peer object directly (works for most cases)
    console.warn(`[DL] getEntity failed (${e.message}), using peerId directly`);
    return peerId;
  }
}

/**
 * Re-fetch the message to get a fresh fileReference, then download via
 * GramJS downloadFileV2 with an explicit InputPhotoFileLocation.
 *
 * This avoids two bugs:
 *   a) Stale fileReference in the event message → AUTH_BYTES_INVALID
 *   b) downloadMedia picking PhotoStrippedSize → 0-byte file
 */
async function downloadViaMTProto(client, originalMsg, destPath) {
  console.log('[DL-1] Re-fetching message for fresh fileReference...');

  const entity = await getEntitySafe(client, originalMsg.peerId);

  // Re-fetch to get fresh fileReference
  const msgs = await client.getMessages(entity, { ids: [originalMsg.id] });
  const freshMsg = msgs && msgs[0];

  if (!freshMsg || !freshMsg.media) {
    throw new Error('Re-fetched message has no media');
  }

  const photo = freshMsg.media.photo;
  if (!photo || photo instanceof Api.PhotoEmpty) {
    throw new Error('Re-fetched message has no valid photo');
  }

  const size = pickBestSize(photo.sizes);
  if (!size) throw new Error('Photo has no usable size');

  console.log(`[DL-1] Downloading: type=${size.type} dcId=${photo.dcId}`);

  const location = new Api.InputPhotoFileLocation({
    id:            photo.id,
    accessHash:    photo.accessHash,
    fileReference: photo.fileReference,
    thumbSize:     size.type,
  });

  const fileSizeBi = 'size' in size ? bigInt(size.size) : bigInt(512 * 1024);

  await downloadFileV2(client, location, {
    outputFile: destPath,
    fileSize:   fileSizeBi,
    dcId:       photo.dcId,
  });

  const bytes = fs.existsSync(destPath) ? fs.statSync(destPath).size : 0;
  if (bytes === 0) throw new Error('downloadFileV2 produced 0 bytes');
  console.log(`[DL-1] Success: ${bytes} bytes`);
}

/**
 * Fallback: use client.downloadMedia() on the re-fetched message.
 * GramJS handles DC export auth internally.
 */
async function downloadViaDownloadMedia(client, originalMsg, destPath) {
  console.log('[DL-2] Trying client.downloadMedia() on re-fetched message...');

  const entity  = await getEntitySafe(client, originalMsg.peerId);
  const msgs    = await client.getMessages(entity, { ids: [originalMsg.id] });
  const freshMsg = msgs && msgs[0];

  if (!freshMsg || !freshMsg.media) {
    throw new Error('Re-fetched message has no media');
  }

  const result = await client.downloadMedia(freshMsg, { outputFile: destPath });

  const bytes = fs.existsSync(destPath) ? fs.statSync(destPath).size : 0;
  if (bytes === 0) throw new Error('downloadMedia produced 0 bytes');
  console.log(`[DL-2] Success: ${bytes} bytes`);
}

/**
 * Master download: try downloadFileV2 first, fall back to downloadMedia.
 * Both are pure MTProto — no HTTP, no Bot API.
 */
async function downloadImage(client, msg, destPath) {
  if (!msg.media) throw new Error('Message has no media');

  // Attempt 1: downloadFileV2 with explicit location (avoids stripped-size bug)
  try {
    await downloadViaMTProto(client, msg, destPath);
    return;
  } catch (e1) {
    console.warn(`[DL-1] Failed: ${e1.message}`);
    try { fs.unlinkSync(destPath); } catch (_) {}
  }

  // Attempt 2: GramJS downloadMedia on re-fetched message
  try {
    await downloadViaDownloadMedia(client, msg, destPath);
    return;
  } catch (e2) {
    console.error(`[DL-2] Failed: ${e2.message}`);
    try { fs.unlinkSync(destPath); } catch (_) {}
    throw new Error(`All MTProto download attempts failed. Last: ${e2.message}`);
  }
}

// ─── Caption helpers ──────────────────────────────────────────────────────────
/**
 * Returns { gridSize: 8|10 } if caption contains a recognised trigger phrase,
 * otherwise null (message is silently ignored).
 *   "WORD GRID CHALLENGE" → 8×8
 *   "HARD MODE CHALLENGE" → 10×10
 */
function getChallengeInfo(text) {
  const u = text.toUpperCase();
  if (u.includes('WORD GRID CHALLENGE')) {
    console.log('[Trigger] WORD GRID CHALLENGE → 8×8');
    return { gridSize: 8 };
  }
  if (u.includes('HARD MODE CHALLENGE')) {
    console.log('[Trigger] HARD MODE CHALLENGE → 10×10');
    return { gridSize: 10 };
  }
  return null;
}

/**
 * Extract word patterns from caption text — left-to-right, single pass.
 * Supports: "M--- (4)", "M----", "W---- H--- (4)", mixed prose.
 */
function parsePatterns(text) {
  const results = [], seen = new Set();
  const re = /([A-Z])(-+)(?:\s*\(\d+\))?/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m[2].length < 2) continue;
    const pattern = (m[1] + m[2]).toUpperCase();
    if (!seen.has(pattern)) { seen.add(pattern); results.push({ pattern }); }
  }
  return results;
}

// ─── Result formatter ─────────────────────────────────────────────────────────
function formatResults(results, grid, patterns) {
  let msg = '🎯 <b>WORD GRID RESULTS</b>\n\n';
  let foundAny = false;

  for (const p of patterns) {
    const key   = p.pattern || p.word;
    if (!key) continue;
    const entry = results[key];

    if (!entry) {
      msg += `❓ ${key}: not found\n`;
      continue;
    }

    if (Array.isArray(entry)) {
      const startChar   = key[0].toUpperCase();
      const wordMatches = new Set();

      for (const hit of entry) {
        const raw = hit.match.toUpperCase();
        if (isWord(raw)) { wordMatches.add(raw); continue; }
        const forced = startChar + raw.slice(1);
        if (isWord(forced)) { wordMatches.add(forced); continue; }
        console.log(`[Debug] Rejected: ${key} → ${raw}`);
      }

      if (wordMatches.size > 0) {
        msg += `✅ ${key}: ${[...wordMatches].map(w => `<code>${w}</code>`).join(', ')}\n`;
        stats.wordsFound += wordMatches.size;
        foundAny = true;
      } else {
        msg += `❓ ${key}: no dictionary words found\n`;
      }
    } else {
      msg += `✅ <code>${entry.match}</code> @ [${entry.r},${entry.c}] ${entry.dir}\n`;
      foundAny = true;
    }
  }

  if (!foundAny) {
    msg += '😔 No real word matches found.\n';
    msg += 'Tip: verify caption patterns match the grid letters.\n';
  }

  msg += '\n🔍 <b>Extracted Grid:</b>\n';
  msg += '<pre>' + grid.map(row => row.join(' ')).join('\n') + '</pre>';
  return msg;
}

// ─── GramJS Bot ───────────────────────────────────────────────────────────────
async function startBot() {
  const session = new StringSession('');
  const client  = new TelegramClient(session, API_ID, API_HASH, {
    connectionRetries: 10,
    retryDelay:        2000,
    autoReconnect:     true,
    useWSS:            false,
  });

  console.log('[GramJS] Connecting via MTProto...');
  await client.start({ botAuthToken: BOT_TOKEN });
  console.log('[GramJS] Connected.');

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
          'Send a word grid image with the challenge caption and word patterns.',
          '',
          '<b>Triggers (case-insensitive):</b>',
          '• <code>WORD GRID CHALLENGE</code> → solves 8×8 grid',
          '• <code>HARD MODE CHALLENGE</code>  → solves 10×10 grid',
          '',
          '<b>Example caption:</b>',
          '<code>WORD GRID CHALLENGE\nM--- (4) P------- (8) C----- (6)</code>',
          '',
          'The bot only processes images with these exact trigger phrases.',
        ].join('\n'),
        parseMode: 'html',
      });
      return;
    }

    // ── Gate: only act on challenge captions ───────────────────────────────────
    const challenge = getChallengeInfo(caption);
    if (!challenge) return; // silently ignore everything else

    // ── Must carry a photo ─────────────────────────────────────────────────────
    const hasPhoto = msg.media && (
      msg.media.className === 'MessageMediaPhoto' ||
      (msg.media.document &&
       msg.media.document.mimeType &&
       msg.media.document.mimeType.startsWith('image/'))
    );

    if (!hasPhoto) {
      await client.sendMessage(chatId, {
        message: '⚠️ Please attach a grid image along with the challenge caption.',
      });
      return;
    }

    // ── Download image ─────────────────────────────────────────────────────────
    const imagePath = path.join(
      __dirname,
      `grid_${Date.now()}_${Math.random().toString(36).slice(2)}.jpg`
    );

    try {
      await client.sendMessage(chatId, {
        message: `🔍 Processing your ${challenge.gridSize}×${challenge.gridSize} word grid...`,
      });

      await downloadImage(client, msg, imagePath);

    } catch (dlErr) {
      console.error('[Download] Failed:', dlErr.message);
      console.error(dlErr.stack);
      await client.sendMessage(chatId, {
        message: `❌ Could not download image: ${dlErr.message}`,
      });
      await deleteFile(imagePath);
      return;
    }

    // ── OCR + Solve ────────────────────────────────────────────────────────────
    try {
      stats.imagesProcessed++;

      const grid = await extractGrid(imagePath, challenge.gridSize);

      if (!grid || grid.length === 0) {
        await client.sendMessage(chatId, {
          message: '❌ Could not read the grid from the image.\nMake sure the letters are clearly visible.',
        });
        return;
      }

      const patterns = parsePatterns(caption);

      if (patterns.length === 0) {
        await client.sendMessage(chatId, {
          message:
            `📋 <b>${challenge.gridSize}×${challenge.gridSize} grid extracted</b> (no patterns found):\n\n` +
            '<pre>' + grid.map(r => r.join(' ')).join('\n') + '</pre>\n\n' +
            'Add patterns like <code>M--- (4)</code> to find words!',
          parseMode: 'html',
        });
        return;
      }

      const results = solve(grid, patterns);
      await client.sendMessage(chatId, {
        message:   formatResults(results, grid, patterns),
        parseMode: 'html',
      });

    } catch (err) {
      console.error('[Handler] Error:', err);
      await client.sendMessage(chatId, {
        message: '🚨 Processing error. Please try again.',
      });
    } finally {
      await deleteFile(imagePath);
    }

  }, new NewMessage({}));

  console.log('[Bot] Listening for messages...');

  const shutdown = async sig => {
    console.log(`[Bot] ${sig} — disconnecting...`);
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
  res.json({ ...stats, uptime: Math.floor((Date.now() - stats.startTime) / 1000) });
});
app.listen(PORT, () => console.log(`[Dashboard] Running on port ${PORT}`));

// ─── Boot ─────────────────────────────────────────────────────────────────────
startBot().catch(err => { console.error('[FATAL]', err); process.exit(1); });
