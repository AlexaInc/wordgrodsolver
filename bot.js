/**
 * bot.js — Word Grid Solver Bot (GramJS MTProto + Bot API file download)
 *
 * Architecture:
 *   • GramJS (MTProto) handles ALL message events — no HTTP polling
 *   • Image download: GramJS downloadFileV2 via InputPhotoFileLocation
 *     (passing the full Message to downloadMedia silently returns 0 bytes
 *      for bots because stripped sizes; we build the location manually)
 *   • Final fallback: Bot API getFile → HTTPS stream (always works)
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

const { TelegramClient }    = require('telegram');
const { StringSession }     = require('telegram/sessions');
const { NewMessage }        = require('telegram/events');
const { Api }               = require('telegram');
const { downloadFileV2 }    = require('telegram/client/downloads');
const bigInt                = require('big-integer');

const express = require('express');
const fs      = require('fs');
const https   = require('https');
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

// ─── Utilities ────────────────────────────────────────────────────────────────
function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function deleteFile(filePath) {
  for (let i = 0; i < 5; i++) {
    try {
      if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
      return;
    } catch (_) {
      await sleep(500);
    }
  }
}

// ─── Image download ───────────────────────────────────────────────────────────

/**
 * Pick the largest non-stripped, non-empty PhotoSize from photo.sizes.
 * Returns the size object or null.
 */
function pickBestPhotoSize(sizes) {
  if (!sizes || sizes.length === 0) return null;
  // Prefer types in descending quality order
  const preferOrder = ['w', 'y', 'd', 'x', 'c', 'm', 'b', 'a', 's'];
  for (const t of preferOrder) {
    const s = sizes.find(sz =>
      sz.type === t &&
      !(sz instanceof Api.PhotoStrippedSize) &&
      !(sz instanceof Api.PhotoSizeEmpty) &&
      !(sz instanceof Api.PhotoSizeProgressive)
    );
    if (s) return s;
  }
  // Fallback: any non-stripped, non-empty
  return sizes.find(sz =>
    !(sz instanceof Api.PhotoStrippedSize) &&
    !(sz instanceof Api.PhotoSizeEmpty)
  ) || null;
}

/**
 * Attempt 1 — GramJS downloadFileV2 via InputPhotoFileLocation.
 * Builds the TL location manually to avoid the stripped-size bug in downloadMedia.
 */
async function downloadViaMTProto(client, photo, destPath) {
  const size = pickBestPhotoSize(photo.sizes);
  if (!size) throw new Error('No usable photo size in TL photo object');

  console.log(`[DL-1] MTProto InputPhotoFileLocation type=${size.type} dcId=${photo.dcId}`);

  const fileLocation = new Api.InputPhotoFileLocation({
    id:            photo.id,
    accessHash:    photo.accessHash,
    fileReference: photo.fileReference,
    thumbSize:     size.type,
  });

  const fileSize = 'size' in size
    ? bigInt(size.size)
    : bigInt(512 * 1024); // safe fallback estimate

  await downloadFileV2(client, fileLocation, {
    outputFile: destPath,
    fileSize,
    dcId: photo.dcId,
  });

  if (!fs.existsSync(destPath)) throw new Error('File not written');
  const bytes = fs.statSync(destPath).size;
  if (bytes === 0) throw new Error('Downloaded file is 0 bytes');
  console.log(`[DL-1] MTProto success — ${bytes} bytes`);
}

/**
 * Attempt 2 — Bot API getFile → HTTPS stream.
 * Uses Bot API to resolve a file_id to a download URL, then streams it.
 * This always works for bots regardless of DC or file reference freshness.
 *
 * To get the Bot API file_id we call Bot API getUpdates is NOT available
 * during long-polling conflicts — instead we use a trick:
 * forward the message to ourselves to get a fresh file_id from Bot API.
 *
 * Simpler approach: call Bot API sendDocument/getFile with the message's
 * photo directly. Since we're a bot, we can use the message_id + chat_id
 * to call Bot API copyMessage and get file_id — but that's wasteful.
 *
 * THE REAL TRICK: GramJS photo.id is NOT the Bot API file_id.
 * But we can encode a Bot API file_id from the TL photo using the
 * standard Telegram encoding scheme (type 2 = photo).
 * Format: pack(type, dc_id, id, access_hash, file_reference) → base64url
 *
 * This is what python-telegram-bot, aiogram etc. all do internally.
 */

/**
 * Encode a Bot API file_id from a TL Photo object.
 * Telegram Bot API file_id encoding for photos (type_id = 2):
 *   byte  0:     file_type (2 = photo)
 *   byte  1:     dc_id
 *   bytes 2-9:   id (int64 LE)
 *   bytes 10-17: access_hash (int64 LE)
 *   byte  18:    len(file_reference)
 *   bytes 19+:   file_reference bytes
 *   byte  19+N:  thumbnail type byte (e.g. 'y'.charCodeAt(0))
 * Then base64url encode the whole thing.
 */
function encodePhotoFileId(photo, thumbType) {
  const fileRef  = Buffer.isBuffer(photo.fileReference)
    ? photo.fileReference
    : Buffer.from(photo.fileReference);

  // Pack id and access_hash as signed int64 LE (BigInt → Buffer)
  function bigIntToLE8(bi) {
    // Handle both native BigInt and big-integer library
    const hex = (typeof bi === 'bigint' ? bi : BigInt(bi.toString()))
      .toString(16)
      .replace('-', '');
    const padded = hex.padStart(16, '0');
    const buf = Buffer.from(padded, 'hex');
    buf.reverse();
    return buf;
  }

  const typeFlag   = Buffer.from([2, photo.dcId]);             // type=photo, dcId
  const idBuf      = bigIntToLE8(photo.id);
  const hashBuf    = bigIntToLE8(photo.accessHash);
  const refLen     = Buffer.from([fileRef.length]);
  const thumbBuf   = Buffer.from([thumbType.charCodeAt(0)]);

  const combined = Buffer.concat([typeFlag, idBuf, hashBuf, refLen, fileRef, thumbBuf]);
  return combined.toString('base64')
    .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

/**
 * Download via Bot API: encode file_id → getFile → HTTPS stream.
 */
async function downloadViaBotApi(photo, thumbType, destPath) {
  console.log(`[DL-2] Bot API getFile...`);

  let fileId;
  try {
    fileId = encodePhotoFileId(photo, thumbType);
  } catch (e) {
    throw new Error(`file_id encoding failed: ${e.message}`);
  }

  // Call Bot API getFile
  const fileInfo = await new Promise((resolve, reject) => {
    const body = JSON.stringify({ file_id: fileId });
    const req = https.request({
      hostname: 'api.telegram.org',
      path:     `/bot${BOT_TOKEN}/getFile`,
      method:   'POST',
      headers:  { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
    }, (res) => {
      let d = '';
      res.on('data', c => d += c);
      res.on('end', () => {
        try {
          const p = JSON.parse(d);
          if (p.ok) resolve(p.result);
          else reject(new Error(`getFile API error: ${p.description}`));
        } catch (e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.setTimeout(10000, () => req.destroy(new Error('getFile timeout')));
    req.write(body);
    req.end();
  });

  if (!fileInfo.file_path) throw new Error('getFile returned no file_path');

  const downloadUrl = `https://api.telegram.org/file/bot${BOT_TOKEN}/${fileInfo.file_path}`;
  console.log(`[DL-2] Streaming from Bot API URL...`);

  // Stream to disk
  await new Promise((resolve, reject) => {
    const fileStream = fs.createWriteStream(destPath);
    https.get(downloadUrl, (res) => {
      if (res.statusCode !== 200) {
        reject(new Error(`HTTP ${res.statusCode} downloading file`));
        return;
      }
      res.pipe(fileStream);
      fileStream.on('finish', () => fileStream.close(resolve));
      fileStream.on('error', reject);
    }).on('error', reject)
      .setTimeout(30000, function() { this.destroy(new Error('Download timeout')); });
  });

  const bytes = fs.existsSync(destPath) ? fs.statSync(destPath).size : 0;
  if (bytes === 0) throw new Error('Bot API download produced 0 bytes');
  console.log(`[DL-2] Bot API success — ${bytes} bytes`);
}

/**
 * Master download function — tries MTProto first, Bot API second.
 */
async function downloadImage(client, msg, destPath) {
  // Extract the TL Photo object
  const media = msg.media;
  if (!media) throw new Error('Message has no media');

  let photo = null;
  if (media instanceof Api.MessageMediaPhoto) {
    photo = media.photo;
  } else if (media instanceof Api.Photo) {
    photo = media;
  }

  if (!photo || photo instanceof Api.PhotoEmpty) {
    throw new Error('Message media contains no valid photo');
  }

  const bestSize = pickBestPhotoSize(photo.sizes);
  if (!bestSize) throw new Error('Photo has no usable sizes');
  const thumbType = bestSize.type || 'y';

  // Attempt 1: MTProto
  try {
    await downloadViaMTProto(client, photo, destPath);
    return;
  } catch (e1) {
    console.warn(`[DL-1] MTProto failed: ${e1.message} — trying Bot API...`);
    try { fs.unlinkSync(destPath); } catch (_) {}
  }

  // Attempt 2: Bot API
  try {
    await downloadViaBotApi(photo, thumbType, destPath);
    return;
  } catch (e2) {
    console.error(`[DL-2] Bot API failed: ${e2.message}`);
    throw new Error(`All download methods failed. MTProto: ${e2.message}`);
  }
}

// ─── Caption helpers ──────────────────────────────────────────────────────────
/**
 * Returns { gridSize: 8|10 } if caption contains a recognised challenge phrase,
 * or null if the message should be ignored.
 *   "WORD GRID CHALLENGE" → 8×8
 *   "HARD MODE CHALLENGE" → 10×10
 */
function getChallengeInfo(text) {
  const upper = text.toUpperCase();
  if (upper.includes('WORD GRID CHALLENGE')) {
    console.log('[Trigger] WORD GRID CHALLENGE → 8×8');
    return { gridSize: 8 };
  }
  if (upper.includes('HARD MODE CHALLENGE')) {
    console.log('[Trigger] HARD MODE CHALLENGE → 10×10');
    return { gridSize: 10 };
  }
  return null;
}

/**
 * Parse word patterns from caption — single left-to-right pass.
 * Handles: "M--- (4)", "M----", "W---- H--- (4)", mixed prose.
 */
function parsePatterns(text) {
  const results = [];
  const seen    = new Set();
  const re      = /([A-Z])(-+)(?:\s*\(\d+\))?/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m[2].length < 2) continue;
    const pattern = (m[1] + m[2]).toUpperCase();
    if (!seen.has(pattern)) {
      seen.add(pattern);
      results.push({ pattern });
    }
  }
  return results;
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
        console.log(`[Debug] Rejected non-word for "${key}": ${raw}`);
      }

      if (wordMatches.size > 0) {
        msg += `✅ ${key}: ${[...wordMatches].map(w => `<code>${w}</code>`).join(', ')}\n`;
        stats.wordsFound += wordMatches.size;
        foundAny = true;
      } else {
        msg += `❓ ${key}: no dictionary words matched\n`;
      }
    } else {
      msg += `✅ <code>${entry.match}</code> @ [${entry.r},${entry.c}] ${entry.dir}\n`;
      foundAny = true;
    }
  }

  if (!foundAny) {
    msg += '😔 No real word matches found.\n';
    msg += 'Tip: verify caption patterns match grid letters.\n';
  }

  msg += '\n🔍 <b>Extracted Grid:</b>\n';
  msg += '<pre>' + grid.map(row => row.join(' ')).join('\n') + '</pre>';
  return msg;
}

// ─── GramJS Bot ───────────────────────────────────────────────────────────────
async function startBot() {
  const session = new StringSession('');

  const client = new TelegramClient(session, API_ID, API_HASH, {
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

    // /start
    if (caption === '/start' || caption.startsWith('/start ')) {
      await client.sendMessage(chatId, {
        message: [
          '👋 <b>Word Grid Solver Bot</b>',
          '',
          'Send a word grid image with the challenge caption and word patterns.',
          '',
          '<b>Triggers:</b>',
          '• <code>WORD GRID CHALLENGE</code> → solves 8×8 grid',
          '• <code>HARD MODE CHALLENGE</code>  → solves 10×10 grid',
          '',
          '<b>Add word patterns in the caption:</b>',
          '<code>WORD GRID CHALLENGE\nM--- (4) P------- (8) S----- (6)</code>',
        ].join('\n'),
        parseMode: 'html',
      });
      return;
    }

    // ── Gate: only recognised challenge captions ───────────────────────────────
    const challenge = getChallengeInfo(caption);
    if (!challenge) return; // silently ignore

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

    // ── Download ───────────────────────────────────────────────────────────────
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
          message: '❌ Could not read the grid from the image. Make sure letters are clearly visible.',
        });
        return;
      }

      const patterns = parsePatterns(caption);

      if (patterns.length === 0) {
        const noPatMsg =
          `📋 <b>${challenge.gridSize}×${challenge.gridSize} grid extracted</b> ` +
          `(no word patterns in caption):\n\n` +
          '<pre>' + grid.map(r => r.join(' ')).join('\n') + '</pre>\n\n' +
          'Add patterns like <code>M--- (4)</code> to find words!';
        await client.sendMessage(chatId, { message: noPatMsg, parseMode: 'html' });
        return;
      }

      const results = solve(grid, patterns);
      const reply   = formatResults(results, grid, patterns);
      await client.sendMessage(chatId, { message: reply, parseMode: 'html' });

    } catch (err) {
      console.error('[Handler] Error:', err);
      await client.sendMessage(chatId, { message: '🚨 Processing error. Please try again.' });
    } finally {
      await deleteFile(imagePath);
    }

  }, new NewMessage({}));

  console.log('[Bot] Listening for messages...');

  const shutdown = async (sig) => {
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
startBot().catch(err => {
  console.error('[FATAL]', err);
  process.exit(1);
});
