from __future__ import annotations

import json
import os
import platform
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import app

SESSION_IDLE_SEC = 15 * 60
_SESSIONS: dict[str, tuple[app.StreamSession, float]] = {}
_SESSIONS_LOCK = threading.Lock()


def _session(sid: str, create: bool = True) -> app.StreamSession | None:
    with _SESSIONS_LOCK:
        item = _SESSIONS.get(sid)
        if item is None:
            if not create:
                return None
            item = (app.StreamSession(), time.time())
        _SESSIONS[sid] = (item[0], time.time())
        return item[0]


def _janitor():
    while True:
        time.sleep(60)
        now = time.time()
        with _SESSIONS_LOCK:
            for sid, (s, seen) in list(_SESSIONS.items()):
                if now - seen > SESSION_IDLE_SEC:
                    s.close()
                    del _SESSIONS[sid]


PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phụ đề realtime</title>
<style>
:root{--bg:#fafafa;--fg:#171717;--muted:#737373;--card:#fff;--border:#e5e5e5;--accent:#ea580c;
      --body-text-color-subdued:#8a8a8a;--border-color-primary:#e5e5e5}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#ededed;--muted:#a3a3a3;--card:#1f1f1f;--border:#333;
      --body-text-color-subdued:#8a8a8a;--border-color-primary:#333}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:1.15rem;margin:0 0 4px} .sub{color:var(--muted);margin:0 0 16px;font-size:.92rem}
.bar{display:flex;flex-wrap:wrap;align-items:center;gap:10px 18px;margin:14px 0 6px}
button{font:inherit;border:1px solid var(--border);background:var(--card);color:var(--fg);padding:8px 16px;border-radius:10px;cursor:pointer}
button.rec{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600;min-width:150px}
button.rec.on{background:#dc2626;border-color:#dc2626}
label{display:flex;gap:6px;align-items:center;color:var(--fg);cursor:pointer}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#fff;margin-right:8px;vertical-align:middle}
.on .dot{animation:blink 1s infinite} @keyframes blink{50%{opacity:.2}}
#stats,#client{color:var(--muted);font-size:.85rem;min-height:1.3em;white-space:pre-wrap;word-break:break-word}
.meterbox{position:relative;height:6px;background:var(--border);border-radius:3px;margin:8px 0 6px;overflow:hidden}
#meter{height:100%;width:0;background:#16a34a;transition:width .15s}
#thr{position:absolute;top:0;bottom:0;width:2px;background:#dc2626;left:12%}
#err{color:#dc2626;font-size:.9rem;white-space:pre-wrap}
details{margin-top:14px;background:var(--card);border:1px solid var(--border);border-radius:10px;padding:10px 14px}
summary{cursor:pointer;font-weight:600}
textarea{width:100%;min-height:90px;margin-top:10px;font:inherit;background:var(--bg);color:var(--fg);border:1px solid var(--border);border-radius:8px;padding:8px}
/*CSS*/
</style></head>
<body><main>
<h1>🎙️ Phụ đề tiếng Việt realtime</h1>
<p class="sub">Dòng <i>nghiêng</i> là câu đang nói. Câu có gạch chấm đang chờ Gemini sửa. Thông tin cá nhân hiện thành nhãn như [SĐT], [TÊN].</p>
<div id="stage"><div class="stage"><div class="line hint">Bấm Bắt đầu ghi âm và nói…</div></div></div>
<div class="bar">
  <button id="rec" class="rec"><span class="dot"></span><span id="recl">Bắt đầu ghi âm</span></button>
  <label><input type="checkbox" id="correct" checked> Gemini sửa câu</label>
  <label><input type="checkbox" id="raw"> So sánh với chữ Whisper</label>
  <button id="clear">Xoá phụ đề</button>
</div>
<div class="meterbox" title="Mức âm micro; vạch đỏ là ngưỡng coi là có tiếng nói"><div id="meter"></div><div id="thr"></div></div>
<div id="client"></div>
<div id="stats"></div><div id="err"></div>
<details><summary>Toàn bộ phụ đề</summary><div id="transcript" style="margin-top:10px"></div>
<textarea id="final" readonly placeholder="Văn bản để copy"></textarea></details>
</main>
<script>
const sid = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2) + Date.now());
const $ = id => document.getElementById(id);
let ctx = null, src = null, proc = null, stream = null, pending = [], recording = false, sendTimer = null, busy = false;

function showErr(msg){ $("err").textContent = msg || ""; }

async function post(path, body){
  const r = await fetch(`${path}?sid=${sid}` + (body && body.qs ? body.qs : ""), {
    method: "POST", headers: {"Content-Type": "application/octet-stream"}, body: body && body.data ? body.data : null});
  if (!r.ok) { let t = ""; try { t = (await r.text()).slice(0, 300); } catch (e) {} throw new Error(`${path}: HTTP ${r.status} ${t}`); }
  return r.json();
}

async function sendPending(){
  if (busy || !pending.length) return;
  busy = true;
  const chunks = pending; pending = [];
  let n = 0; for (const c of chunks) n += c.length;
  const pcm = new Int16Array(n); let o = 0;
  for (const c of chunks) for (let i = 0; i < c.length; i++) {
    const v = Math.max(-1, Math.min(1, c[i])); pcm[o++] = v < 0 ? v * 0x8000 : v * 0x7FFF;
  }
  try {
    await post("chunk", {data: pcm.buffer, qs: `&sr=${ctx.sampleRate}&correct=${$("correct").checked ? 1 : 0}`});
    sent++; showErr("");
  } catch (e) { sendErr++; showErr("Không gửi được audio lên server: " + e.message); }
  busy = false;
}

const AC = window.AudioContext || window.webkitAudioContext;
let level = 0, frames = 0, sent = 0, sendErr = 0, startedAt = 0;

function newContext(rate){
  try { return rate ? new AC({sampleRate: rate}) : new AC(); } catch (e) { return new AC(); }
}

async function start(){
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showErr("Trình duyệt không cho dùng micro ở trang này (cần HTTPS hoặc localhost)."); return;
  }
  ctx = newContext(16000);
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true}});
  } catch (e) {
    showErr("Không mở được micro: " + e.message + ". Bấm biểu tượng ổ khoá trên thanh địa chỉ để cho phép micro.");
    ctx.close(); return;
  }
  try { src = ctx.createMediaStreamSource(stream); }
  catch (e) { ctx.close(); ctx = newContext(0); src = ctx.createMediaStreamSource(stream); }
  if (ctx.state !== "running") { try { await ctx.resume(); } catch (e) {} }
  proc = ctx.createScriptProcessor(4096, 1, 1);
  proc.onaudioprocess = e => {
    if (!recording) return;
    const d = e.inputBuffer.getChannelData(0);
    let s = 0; for (let i = 0; i < d.length; i++) s += d[i] * d[i];
    level = Math.max(Math.sqrt(s / d.length), level * 0.85); frames++;
    pending.push(new Float32Array(d));
  };
  src.connect(proc); proc.connect(ctx.destination);
  recording = true; frames = 0; sent = 0; sendErr = 0; startedAt = Date.now();
  sendTimer = setInterval(sendPending, 500);
  $("rec").classList.add("on"); $("recl").textContent = "Dừng";
  showErr("");
}

function clientDiag(){
  if (!recording) { $("client").textContent = ""; return; }
  const pct = Math.min(100, Math.round(level / 0.1 * 100));
  $("meter").style.width = pct + "%";
  const thr = Math.min(100, Math.round(vad / 0.1 * 100));
  $("thr").style.left = thr + "%";
  let msg = `Trình duyệt: âm thanh ${ctx.state}, ${ctx.sampleRate} Hz · mức mic ${level.toFixed(3)} · đã gửi ${sent} gói` + (sendErr ? `, lỗi ${sendErr}` : "");
  if (ctx.state !== "running") msg += " · ⚠️ âm thanh bị trình duyệt tạm dừng, bấm Dừng rồi bấm lại";
  else if (Date.now() - startedAt > 2000 && frames === 0) msg += " · ⚠️ micro không trả dữ liệu (kiểm tra mic đang chọn trong trình duyệt)";
  $("client").textContent = msg;
}
setInterval(clientDiag, 250);

async function stop(){
  recording = false; clearInterval(sendTimer);
  while (busy) await new Promise(r => setTimeout(r, 50));
  await sendPending();
  try { proc.disconnect(); src.disconnect(); stream.getTracks().forEach(t => t.stop()); await ctx.close(); } catch (e) {}
  $("rec").classList.remove("on"); $("recl").textContent = "Bắt đầu ghi âm";
  try { await post("stop"); } catch (e) { showErr(e.message); }
}

$("rec").onclick = () => recording ? stop() : start();
$("clear").onclick = async () => { pending = []; try { await post("clear"); } catch (e) { showErr(e.message); } };

let lastStage = "", vad = 0.012;
async function poll(){
  try {
    const r = await fetch(`state?sid=${sid}&raw=${$("raw").checked ? 1 : 0}`, {cache: "no-store"});
    if (r.ok) {
      const j = await r.json();
      vad = j.vad || vad;
      if (j.stage !== lastStage) { $("stage").innerHTML = j.stage; lastStage = j.stage; }
      $("transcript").innerHTML = j.transcript; $("stats").innerHTML = j.stats;
      if (document.activeElement !== $("final")) $("final").value = j.final;
    }
  } catch (e) {}
  setTimeout(poll, 500);
}
poll();
</script></body></html>"""


def _page() -> bytes:
    return PAGE.replace("/*CSS*/", app.CSS).encode("utf-8")


def _state(sess: app.StreamSession | None, raw: bool) -> dict:
    stage, transcript, final, stats = app.render(sess, raw)
    return {"stage": stage, "transcript": transcript, "final": final,
            "stats": stats.replace("  \n", "<br>"), "vad": app.SILENCE_RMS}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "phude/1.0"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _route(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        return u.path.rstrip("/").rsplit("/", 1)[-1], q

    def do_GET(self):
        name, q = self._route()
        if name == "state":
            self._json(_state(_session(q.get("sid", ""), create=False), q.get("raw") == "1"))
        elif name == "health":
            self._json({"ok": True})
        else:
            self._send(200, _page(), "text/html; charset=utf-8")

    def do_POST(self):
        name, q = self._route()
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        sid = q.get("sid", "")
        if not sid:
            return self._json({"error": "thiếu sid"}, 400)
        try:
            if name == "chunk":
                sess = _session(sid)
                sess.correct_enabled = q.get("correct", "1") == "1"
                pcm = np.frombuffer(body[: len(body) // 2 * 2], dtype="<i2")
                if len(pcm):
                    sess.push(app.to_mono_16k(int(float(q.get("sr", "48000"))), pcm))
                self._json({"ok": True})
            elif name == "stop":
                sess = _session(sid, create=False)
                if sess:
                    sess.flush()
                self._json({"ok": True})
            elif name == "clear":
                sess = _session(sid, create=False)
                if sess:
                    sess.reset()
                self._json({"ok": True})
            else:
                self._json({"error": "không có đường dẫn này"}, 404)
        except Exception as e:
            app.log(f"Lỗi xử lý {name}:\n{traceback.format_exc()}")
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


_SERVER: ThreadingHTTPServer | None = None


def start(port: int = 8000, background: bool = False) -> ThreadingHTTPServer:
    global _SERVER
    stop()
    app.models()
    ThreadingHTTPServer.allow_reuse_address = True
    _SERVER = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    _SERVER.daemon_threads = True
    threading.Thread(target=_janitor, daemon=True).start()
    app.log(f"Server phụ đề chạy ở cổng {port} | Python {platform.python_version()} | numpy {np.__version__} | "
            f"BACKEND={app.BACKEND} | Whisper={app.WHISPER_ENDPOINT} ({app.WHISPER_INPUT_FORMAT}) | "
            f"Gemini={app.GEMINI_ENDPOINT} | SILENCE_RMS={app.SILENCE_RMS}")
    if background:
        threading.Thread(target=_SERVER.serve_forever, daemon=True).start()
    else:
        _SERVER.serve_forever()
    return _SERVER


def stop():
    global _SERVER
    if _SERVER is not None:
        _SERVER.shutdown()
        _SERVER.server_close()
        _SERVER = None


if __name__ == "__main__":
    start(int(os.getenv("DATABRICKS_APP_PORT", "8000")))
