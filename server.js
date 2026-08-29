#!/usr/bin/env node
/**
 * PixelScribe — static dev server for www/
 * ════════════════════════════════════════════════════════════════════
 * The app must be served over http:// rather than opened as a file://
 * URL. Three things break under file://
 *
 *   • Web Workers (Tesseract.js, PDF.js) cannot be constructed
 *   • WebAssembly.instantiateStreaming needs a real Content-Type
 *   • fetch() — used to probe which vendor assets exist — is blocked
 *
 * Usage:  node server.js [--port 5173] [--root www] [--host 0.0.0.0]
 */

'use strict';

const http = require('http');
const fs   = require('fs');
const path = require('path');
const url  = require('url');

/* ── args ────────────────────────────────────────────────────────── */
function arg(name, fallback) {
  const i = process.argv.indexOf('--' + name);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}
const PORT = parseInt(process.env.PORT || arg('port', '5173'), 10);
const HOST = process.env.HOST || arg('host', '0.0.0.0');
const ROOT = path.resolve(__dirname, arg('root', 'www'));

/* ── MIME map ────────────────────────────────────────────────────────
   .mjs and .wasm are the ones that matter: ONNX Runtime dynamically
   imports ort-wasm-simd-threaded.mjs and streams the .wasm beside it.
   Serving either as text/plain or octet-stream makes the import fail. */
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'text/javascript; charset=utf-8',
  '.mjs':  'text/javascript; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif':  'image/gif',
  '.webp': 'image/webp',
  '.bmp':  'image/bmp',
  '.svg':  'image/svg+xml',
  '.ico':  'image/x-icon',
  '.pdf':  'application/pdf',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf':  'font/ttf',
  '.onnx': 'application/octet-stream',
  '.traineddata': 'application/octet-stream',
  '.gz':   'application/gzip',
  '.txt':  'text/plain; charset=utf-8',
  '.map':  'application/json; charset=utf-8',
};

function mimeFor(file) {
  return MIME[path.extname(file).toLowerCase()] || 'application/octet-stream';
}

function send(res, status, body, headers) {
  res.writeHead(status, Object.assign({ 'Content-Type': 'text/plain; charset=utf-8' }, headers || {}));
  res.end(body);
}

const server = http.createServer((req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    return send(res, 405, 'Method Not Allowed', { Allow: 'GET, HEAD' });
  }

  let pathname;
  try {
    pathname = decodeURIComponent(url.parse(req.url).pathname);
  } catch (_) {
    return send(res, 400, 'Bad Request');
  }
  if (pathname.endsWith('/')) pathname += 'index.html';

  // Resolve inside ROOT and reject anything that escapes it (../, symlink, …)
  const filePath = path.resolve(ROOT, '.' + path.posix.normalize(pathname));
  if (filePath !== ROOT && !filePath.startsWith(ROOT + path.sep)) {
    return send(res, 403, 'Forbidden');
  }

  fs.stat(filePath, (err, stat) => {
    if (err || !stat.isFile()) {
      console.warn(`  ${req.method} ${pathname} → 404`);
      return send(res, 404, 'Not Found: ' + pathname);
    }

    const headers = {
      'Content-Type':   mimeFor(filePath),
      'Content-Length': stat.size,
      // Dev server: never let a stale vendor blob survive an edit.
      'Cache-Control':  'no-cache, no-store, must-revalidate',
      // Vendor probes use fetch(); keep them unblocked if the page is
      // ever hosted on a different origin than these assets.
      'Access-Control-Allow-Origin': '*',
    };

    if (req.method === 'HEAD') {         // used by the local-vs-CDN probes
      res.writeHead(200, headers);
      return res.end();
    }

    res.writeHead(200, headers);
    const stream = fs.createReadStream(filePath);
    stream.on('error', () => res.destroy());
    stream.pipe(res);
  });
});

if (!fs.existsSync(path.join(ROOT, 'index.html'))) {
  console.error(`✗  No index.html under ${ROOT}`);
  process.exit(1);
}

server.listen(PORT, HOST, () => {
  console.log('');
  console.log('  PixelScribe — offline content-aware text editor');
  console.log('  ─────────────────────────────────────────────────');
  console.log(`  serving   ${ROOT}`);
  console.log(`  local     http://localhost:${PORT}`);
  console.log(`  demo      http://localhost:${PORT}/?demo=1`);
  console.log('  Ctrl+C to stop');
  console.log('');
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`✗  Port ${PORT} is already in use. Try: node server.js --port 5174`);
    process.exit(1);
  }
  throw err;
});
