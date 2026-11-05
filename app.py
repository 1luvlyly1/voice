from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import threading
import time
import traceback
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from pii import mask_pii, placeholders_in


def log(msg: str):
    print(f"[phude {time.strftime('%H:%M:%S')}] {msg}", flush=True)

BACKEND = os.getenv("BACKEND", "gateway")
WHISPER_ENDPOINT = os.getenv("WHISPER_ENDPOINT", "ai_training_whisper_endpoint")
WHISPER_INPUT_FORMAT = os.getenv("WHISPER_INPUT_FORMAT", "auto")
WHISPER_PARAMS = json.loads(os.getenv("WHISPER_PARAMS", "{}") or "{}")
GEMINI_ENDPOINT = os.getenv("GEMINI_ENDPOINT", "gemini-3-8-flash-ai_training")
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "512"))
GLOSSARY = os.getenv("GLOSSARY", "Databricks, Unity Catalog, AI Gateway, Delta Lake")

TARGET_SR = 16_000
STREAM_EVERY = float(os.getenv("STREAM_EVERY", "0.5"))
PARTIAL_STEP_SEC = float(os.getenv("PARTIAL_STEP_SEC", "0.8"))
SILENCE_SEC = float(os.getenv("SILENCE_SEC", "0.6"))
MAX_SEGMENT_SEC = float(os.getenv("MAX_SEGMENT_SEC", "12"))
SILENCE_RMS = float(os.getenv("SILENCE_RMS", "0.012"))
MIN_SPEECH_SEC = 0.25
WHISPER_CONCURRENCY = int(os.getenv("WHISPER_CONCURRENCY", "8"))
GEMINI_CONCURRENCY = int(os.getenv("GEMINI_CONCURRENCY", "4"))

FRAME = int(0.03 * TARGET_SR)


def to_mono_16k(sr: int, data: np.ndarray) -> np.ndarray:
    x = np.asarray(data)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float32)
    if int(sr) != TARGET_SR:
        x = resample(x, int(sr), TARGET_SR)
    return x


def resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if len(x) == 0:
        return x.astype(np.float32)
    if sr > target:
        fc = 0.45 * target / sr
        n_taps = int(8 * sr / target) * 2 + 1
        n = np.arange(n_taps) - n_taps // 2
        h = 2 * fc * np.sinc(2 * fc * n) * np.hamming(n_taps)
        x = np.convolve(x, h / h.sum(), mode="same")
    n_out = int(round(len(x) * target / sr))
    t = np.arange(n_out) * (sr / target)
    return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def frame_rms(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    f = x[: n * FRAME].reshape(n, FRAME)
    return np.sqrt((f * f).mean(axis=1) + 1e-12)


def speech_seconds(rms: np.ndarray) -> float:
    return float((rms > SILENCE_RMS).sum()) * FRAME / TARGET_SR


def trailing_silence_seconds(rms: np.ndarray) -> float:
    loud = np.nonzero(rms > SILENCE_RMS)[0]
    last = loud[-1] + 1 if len(loud) else 0
    return (len(rms) - last) * FRAME / TARGET_SR


def quietest_cut(rms: np.ndarray, search_sec: float = 3.0) -> int:
    k = max(1, int(search_sec * TARGET_SR / FRAME))
    start = max(0, len(rms) - k)
    idx = start + int(np.argmin(rms[start:])) if len(rms) else 0
    return max(FRAME, idx * FRAME)


def trim_silence(seg: np.ndarray) -> np.ndarray:
    rms = frame_rms(seg)
    loud = np.nonzero(rms > SILENCE_RMS)[0]
    if not len(loud):
        return seg
    start = max(0, (loud[0] - 10) * FRAME)
    end = min(len(seg), (loud[-1] + 1 + 6) * FRAME)
    return seg[start:end]


def wav_b64(audio: np.ndarray) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def load_audio(path: str) -> tuple[int, np.ndarray]:
    if path.lower().endswith(".wav"):
        with wave.open(path, "rb") as w:
            sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        dtype = {1: np.uint8, 2: "<i2", 4: "<i4"}[width]
        data = np.frombuffer(raw, dtype=dtype)
        if width == 1:
            data = (data.astype(np.float32) - 128) / 128
        return sr, data.reshape(-1, ch) if ch > 1 else data
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32")
    return sr, data


HALLUCINATION_RE = re.compile(
    r"(hãy\s+subscribe|đăng\s+ký\s+kênh|cảm\s+ơn\s+các\s+bạn\s+đã\s+(?:xem|theo\s+dõi|lắng\s+nghe)"
    r"|hẹn\s+gặp\s+lại\s+các\s+bạn|ghiền\s+mì\s+gõ|vietsub|subtitles?\s+by|like\s+và\s+share|bấm\s+chuông)",
    re.IGNORECASE,
)


def drop_hallucinations(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(p for p in parts if p and not HALLUCINATION_RE.search(p)).strip()


_API = None
_API_LOCK = threading.Lock()


def api():
    global _API
    if _API is None:
        with _API_LOCK:
            if _API is None:
                from databricks.sdk import WorkspaceClient

                _API = WorkspaceClient().api_client
    return _API


def invoke(endpoint: str, body: dict) -> dict:
    return api().do("POST", f"/serving-endpoints/{endpoint}/invocations", body=body,
                    headers={"Content-Type": "application/json", "Accept": "application/json"})


def extract_text(resp) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, list):
        return extract_text(resp[0]) if resp else ""
    if isinstance(resp, dict):
        for k in ("text", "transcription", "transcript", "content"):
            if k in resp and isinstance(resp[k], (str, list)):
                return extract_text(resp[k]) if isinstance(resp[k], list) else resp[k]
        if "choices" in resp:
            return chat_content(resp)
        for k in ("predictions", "outputs", "output", "result", "results", "data"):
            if k in resp:
                return extract_text(resp[k])
    return str(resp)


def chat_content(resp: dict) -> str:
    content = resp["choices"][0]["message"].get("content", "")
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type", "text") == "text")
    return content or ""


class WhisperClient:

    AUTO_FORMATS = ["dataframe:audio", "dataframe:audio_b64", "dataframe:audio_base64", "dataframe:input",
                    "inputs_list", "inputs_str", "inputs_dict", "instances"]

    def __init__(self, endpoint: str, fmt: str):
        self.endpoint = endpoint
        self.fmt = None if fmt == "auto" else fmt
        self.detect_lock = threading.Lock()
        self.sem = threading.Semaphore(WHISPER_CONCURRENCY)
        self.last_raw = ""
        self.calls = 0

    def _call(self, fmt: str, b64: str) -> str:
        resp = invoke(self.endpoint, self.build(fmt, b64))
        self.calls += 1
        try:
            self.last_raw = json.dumps(resp, ensure_ascii=False)[:300]
        except (TypeError, ValueError):
            self.last_raw = str(resp)[:300]
        if self.calls <= 3:
            log(f"Whisper response mẫu (lần {self.calls}): {self.last_raw}")
        return extract_text(resp).strip()

    @staticmethod
    def build(fmt: str, b64: str) -> dict:
        if fmt.startswith("dataframe:"):
            body = {"dataframe_records": [{fmt.split(":", 1)[1]: b64}]}
        elif fmt == "inputs_list":
            body = {"inputs": [b64]}
        elif fmt == "inputs_str":
            body = {"inputs": b64}
        elif fmt == "inputs_dict":
            body = {"inputs": {"audio": b64}}
        elif fmt == "instances":
            body = {"instances": [b64]}
        else:
            raise ValueError(f"WHISPER_INPUT_FORMAT không hợp lệ: {fmt}")
        if WHISPER_PARAMS:
            body["params"] = WHISPER_PARAMS
        return body

    def __call__(self, audio: np.ndarray) -> str:
        b64 = wav_b64(audio)
        with self.sem:
            if self.fmt:
                return self._call(self.fmt, b64)
            with self.detect_lock:
                if self.fmt:
                    return self._call(self.fmt, b64)
                return self._detect(b64)

    def _detect(self, b64: str) -> str:
        from databricks.sdk.errors import BadRequest, InternalError

        errors = []
        for fmt in self.AUTO_FORMATS:
            try:
                text = self._call(fmt, b64)
            except (BadRequest, InternalError) as e:
                errors.append(f"{fmt}: {str(e)[:160]}")
                log(f"Whisper không nhận định dạng {fmt}: {str(e)[:300]}")
                continue
            self.fmt = fmt
            log(f"[whisper] định dạng đầu vào phù hợp: {fmt}")
            return text
        raise RuntimeError("Không dò được định dạng đầu vào của Whisper endpoint. "
                           "Đặt WHISPER_INPUT_FORMAT thủ công. Chi tiết: " + " | ".join(errors))

    @property
    def label(self):
        return self.fmt or "đang dò"


SYSTEM_PROMPT = """Bạn là biên tập viên phụ đề tiếng Việt. Đầu vào là một câu do hệ thống nhận dạng giọng nói tạo ra, có thể sai chính tả, sai dấu thanh, thiếu dấu câu hoặc nghe nhầm từ.
Nhiệm vụ: sửa thành câu tiếng Việt đúng và dễ đọc, giữ nguyên ý người nói.
Quy tắc:
1. Chỉ sửa chính tả, dấu thanh, dấu câu, viết hoa, và từ bị nghe nhầm khi chắc chắn dựa vào ngữ cảnh hoặc bảng thuật ngữ.
2. Không thêm ý, không bỏ ý, không tóm tắt, không dịch, không trả lời hay bình luận nội dung câu.
3. Giữ nguyên mọi nhãn trong ngoặc vuông như [SĐT], [TÊN], [EMAIL], [ĐỊA_CHỈ]... đúng số lượng và vị trí. Không đoán hay điền thông tin thật vào.
4. Nếu không chắc một từ, giữ nguyên từ gốc.
5. Chỉ trả về đúng câu đã sửa, không giải thích, không ngoặc kép."""


class GeminiCorrector:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def __call__(self, text: str, context: list[str]) -> str:
        ctx = "\n".join(context[-3:]) or "(không có)"
        user = (f"Bảng thuật ngữ: {GLOSSARY or '(không có)'}\n"
                f"Ngữ cảnh - các câu trước, chỉ để tham khảo, KHÔNG sửa:\n{ctx}\n\n"
                f"Câu cần sửa:\n{text}")
        body = {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": GEMINI_MAX_TOKENS,
        }
        return chat_content(invoke(self.endpoint, body))


def sanitize_correction(raw: str, out: str) -> tuple[str, str]:
    out = (out or "").strip().strip('"“”`').strip()
    out = re.sub(r"^(?:câu\s+đã\s+sửa|kết\s+quả)\s*:\s*", "", out, flags=re.IGNORECASE).strip()
    if not out:
        return raw, "kept"
    if placeholders_in(out) != placeholders_in(raw):
        return raw, "kept"
    ratio = len(out) / max(1, len(raw))
    lo, hi = (0.6, 1.6) if len(raw) >= 15 else (0.3, 2.5)
    if not lo <= ratio <= hi:
        return raw, "kept"
    return out, "done"


class MockWhisper:
    label = "mock"

    def __call__(self, audio):
        time.sleep(0.2)
        return f"đoạn nói dài {len(audio) / TARGET_SR:.1f} giây số của tôi là không chín một hai ba bốn năm sáu bảy tám"


class MockGemini:
    def __call__(self, text, context):
        time.sleep(0.6)
        return text[:1].upper() + text[1:].replace("đoạn nói", "Đoạn nói") + "."


_WHISPER = None
_GEMINI = None
_INIT_LOCK = threading.Lock()
_GEMINI_POOL = ThreadPoolExecutor(max_workers=GEMINI_CONCURRENCY, thread_name_prefix="gemini")


def models():
    global _WHISPER, _GEMINI
    if _WHISPER is None:
        with _INIT_LOCK:
            if _WHISPER is None:
                if BACKEND == "mock":
                    _GEMINI, _WHISPER = MockGemini(), MockWhisper()
                else:
                    _GEMINI = GeminiCorrector(GEMINI_ENDPOINT)
                    _WHISPER = WhisperClient(WHISPER_ENDPOINT, WHISPER_INPUT_FORMAT)
    return _WHISPER, _GEMINI


@dataclass
class Segment:
    raw: str
    text: str
    status: str = "pending"
    pii: list[str] = field(default_factory=list)


class StreamSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.buf = np.zeros(0, dtype=np.float32)
        self.segments: list[Segment] = []
        self.partial = ""
        self.last_partial_len = 0
        self.correct_enabled = True
        self.t_whisper = 0.0
        self.t_gemini = 0.0
        self.audio_total_sec = 0.0
        self.error = ""
        self.chunks = 0
        self.level = 0.0
        self.speech_total = 0.0
        self.whisper_calls = 0
        self.whisper_empty = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def push(self, audio: np.ndarray):
        rms = frame_rms(audio)
        with self.lock:
            self.buf = np.concatenate([self.buf, audio])
            self.audio_total_sec += len(audio) / TARGET_SR
            self.chunks += 1
            self.level = float(rms.max()) if len(rms) else 0.0
            self.speech_total += speech_seconds(rms)
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def flush(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        with self.lock:
            seg, self.buf = self.buf, np.zeros(0, dtype=np.float32)
        self._commit(seg)
        with self.lock:
            self.partial, self.last_partial_len = "", 0

    def reset(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        enabled = self.correct_enabled
        self.__init__()
        self.correct_enabled = enabled

    def close(self):
        self._stop.set()

    def snapshot(self):
        with self.lock:
            return list(self.segments), self.partial, len(self.buf) / TARGET_SR

    def _whisper(self, audio: np.ndarray) -> tuple[str, list[str]]:
        whisper, _ = models()
        t0 = time.perf_counter()
        self.whisper_calls += 1
        text = whisper(audio)
        self.t_whisper = time.perf_counter() - t0
        self.error = ""
        if not text.strip():
            self.whisper_empty += 1
        return mask_pii(drop_hallucinations(text))

    def _fail(self, where: str, e: Exception):
        self.error = f"{where} lỗi: {type(e).__name__}: {e}"
        log(f"{where} lỗi:\n{traceback.format_exc()}")

    def _commit(self, seg: np.ndarray):
        if speech_seconds(frame_rms(seg)) < MIN_SPEECH_SEC:
            return
        try:
            text, pii = self._whisper(trim_silence(seg))
        except Exception as e:
            self._fail("Whisper", e)
            return
        if not text:
            return
        s = Segment(raw=text, text=text, pii=pii, status="pending" if self.correct_enabled else "off")
        with self.lock:
            context = [x.text for x in self.segments[-3:]]
            self.segments.append(s)
        if self.correct_enabled:
            _GEMINI_POOL.submit(self._correct, s, context)

    def _correct(self, s: Segment, context: list[str]):
        _, gemini = models()
        t0 = time.perf_counter()
        try:
            out = gemini(s.raw, context)
        except Exception as e:
            s.status = "error"
            self._fail("Gemini", e)
            return
        self.t_gemini = time.perf_counter() - t0
        text, status = sanitize_correction(s.raw, out)
        text, pii2 = mask_pii(text)
        s.pii += pii2
        s.text, s.status = text, status

    def _run(self):
        while not self._stop.is_set():
            with self.lock:
                buf = self.buf.copy()
                last_partial_len = self.last_partial_len
            n = len(buf)
            dur = n / TARGET_SR
            rms = frame_rms(buf)
            has_speech = speech_seconds(rms) >= MIN_SPEECH_SEC

            if has_speech and (trailing_silence_seconds(rms) >= SILENCE_SEC or dur >= MAX_SEGMENT_SEC):
                cut = n if dur < MAX_SEGMENT_SEC else quietest_cut(rms)
                self._commit(buf[:cut])
                with self.lock:
                    self.buf = self.buf[cut:]
                    self.partial, self.last_partial_len = "", 0
                continue

            if has_speech and n - last_partial_len >= PARTIAL_STEP_SEC * TARGET_SR:
                try:
                    text, _ = self._whisper(trim_silence(buf))
                except Exception as e:
                    self._fail("Whisper", e)
                    time.sleep(1)
                    continue
                with self.lock:
                    if len(self.buf) >= n:
                        self.partial, self.last_partial_len = text, n
                continue

            if not has_speech and dur > 1.5:
                keep = int(0.5 * TARGET_SR)
                with self.lock:
                    self.buf = self.buf[n - keep:]
                    self.last_partial_len = 0

            time.sleep(0.05)


def _seg_html(s: Segment) -> str:
    return f"<span class='seg {s.status}'>{html.escape(s.text)}</span>"


def diagnose(sess: StreamSession) -> str:
    whisper, _ = models()
    if sess.error:
        return sess.error
    if sess.chunks == 0:
        return "Server chưa nhận được audio từ trình duyệt."
    if sess.speech_total < MIN_SPEECH_SEC and sess.audio_total_sec > 3:
        return (f"Có nhận audio nhưng mic quá nhỏ: mức âm {sess.level:.3f} < ngưỡng {SILENCE_RMS}. "
                "Nói to/gần mic hơn hoặc giảm SILENCE_RMS.")
    if sess.whisper_calls and sess.whisper_empty == sess.whisper_calls:
        raw = getattr(whisper, "last_raw", "")
        return (f"Whisper được gọi {sess.whisper_calls} lần nhưng đều trả về rỗng. "
                f"Có thể sai WHISPER_INPUT_FORMAT. Response gần nhất: {raw}")
    return ""


def render(sess: StreamSession | None, show_raw: bool = False):
    if sess is None:
        stage = "<div class='stage'><div class='line hint'>Bấm Bắt đầu ghi âm và nói…</div></div>"
        return stage, "", "", "Chưa nhận được audio nào từ trình duyệt."
    segs, partial, pending = sess.snapshot()
    problem = diagnose(sess)

    lines = [f"<div class='line old'>{_seg_html(s)}</div>" for s in segs[-2:]]
    if partial:
        lines.append(f"<div class='line partial'>{html.escape(partial)}</div>")
    elif not segs:
        lines.append("<div class='line hint'>đang nghe…</div>")
    if lines and not partial:
        lines[-1] = lines[-1].replace("line old", "line")
    lines = lines[-3:]
    if problem and not partial:
        lines.append(f"<div class='line err'>⚠️ {html.escape(problem[:300])}</div>")
    stage = "<div class='stage'>" + "".join(lines) + "</div>"

    if show_raw:
        rows = "".join(
            f"<tr><td>{html.escape(s.raw)}</td><td>{_seg_html(s)}</td></tr>" for s in segs)
        transcript = (f"<table class='cmp'><tr><th>Whisper (đã che PII)</th><th>Hiển thị</th></tr>{rows}</table>"
                      if segs else "")
    else:
        transcript = "<div class='transcript'>" + " ".join(_seg_html(s) for s in segs) + "</div>"

    whisper, _ = models()
    n_pii = sum(len(s.pii) for s in segs)
    waiting = sum(s.status == "pending" for s in segs)
    stats = (f"Nhận {sess.chunks} gói, {sess.audio_total_sec:.0f}s audio, có tiếng nói {sess.speech_total:.1f}s "
             f"(mức âm {sess.level:.3f}, ngưỡng {SILENCE_RMS}) · "
             f"Whisper {sess.whisper_calls} lần, rỗng {sess.whisper_empty}, {sess.t_whisper:.2f}s "
             f"({getattr(whisper, 'label', '')}) · "
             f"Gemini {sess.t_gemini:.2f}s, chờ {waiting} câu · che {n_pii} PII · audio chờ {pending:.1f}s")
    if problem:
        stats += f"  \n⚠️ {html.escape(problem[:600])}"
    return stage, transcript, " ".join(s.text for s in segs), stats


def on_chunk(chunk, sess, correct, show_raw):
    if sess is None:
        sess = StreamSession()
    sess.correct_enabled = bool(correct)
    if chunk is not None:
        sr, data = chunk
        if data is not None and len(data):
            sess.push(to_mono_16k(sr, data))
    return (sess, *render(sess, show_raw))


def on_stop(sess, show_raw):
    if sess is not None:
        sess.flush()
    return (sess, *render(sess, show_raw))


def on_clear(sess, show_raw):
    if sess is not None:
        sess.reset()
    return (sess, *render(sess, show_raw))


def on_tick(sess, show_raw):
    return render(sess, show_raw)


CSS = """
.stage {min-height: 190px; display: flex; flex-direction: column; justify-content: flex-end; gap: 6px;
        padding: 20px 28px; border-radius: 14px; background: #0b0b0c; text-align: center;}
.stage .line {color: #fff; font-size: 1.7rem; line-height: 1.35; font-weight: 500;}
.stage .line.old {color: #a3a3a3; font-size: 1.25rem;}
.stage .line.partial {color: #d4d4d4; font-style: italic;}
.stage .line.hint {color: #737373; font-size: 1.1rem;}
.stage .line.err {color: #fca5a5; font-size: 1rem; font-weight: 400; word-break: break-word;}
.stage .line * {color: inherit !important;}
.seg.pending {border-bottom: 1px dotted currentColor;}
.transcript {font-size: 1.05rem; line-height: 1.7;}
.transcript .seg.pending, .cmp .seg.pending {color: var(--body-text-color-subdued);}
.cmp {width: 100%; border-collapse: collapse; font-size: .95rem;}
.cmp td, .cmp th {border-bottom: 1px solid var(--border-color-primary); padding: 6px 8px; vertical-align: top; text-align: left;}
"""


def build_ui():
    import gradio as gr

    with gr.Blocks(title="Phụ đề realtime", css=CSS) as demo:
        gr.Markdown("### 🎙️ Phụ đề tiếng Việt realtime\nDòng *nghiêng* là câu đang nói. Câu có gạch chấm đang chờ Gemini sửa. "
                    "Thông tin cá nhân được thay bằng nhãn như [SĐT], [TÊN].")
        sess = gr.State(None, delete_callback=lambda s: s and s.close())
        stage = gr.HTML(render(None)[0])
        with gr.Row():
            mic = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Micro", scale=3)
            with gr.Column(scale=1, min_width=200):
                correct = gr.Checkbox(True, label="Gemini sửa câu")
                show_raw = gr.Checkbox(False, label="So sánh với chữ Whisper")
                clear = gr.Button("Xoá phụ đề")
        stats = gr.Markdown()
        with gr.Accordion("Toàn bộ phụ đề", open=False):
            transcript = gr.HTML()
            final = gr.Textbox(lines=4, show_copy_button=True, label="Văn bản để copy")

        outs = [sess, stage, transcript, final, stats]
        mic.stream(on_chunk, [mic, sess, correct, show_raw], outs, stream_every=STREAM_EVERY,
                   time_limit=60 * 60, show_progress="hidden", concurrency_limit=None)
        mic.stop_recording(on_stop, [sess, show_raw], outs, show_progress="hidden")
        clear.click(on_clear, [sess, show_raw], outs)
        gr.Timer(0.5).tick(on_tick, [sess, show_raw], outs[1:], show_progress="hidden", concurrency_limit=None)
    return demo


def launch(port: int, root_path: str | None = None, block: bool = True):
    models()
    return build_ui().queue(default_concurrency_limit=None).launch(
        server_name="0.0.0.0", server_port=port, root_path=root_path,
        prevent_thread_lock=not block, inline=False, share=False,
    )


def simulate_file(path: str, speed: float = 1.0, correct: bool = True, on_update=None, every: float = 0.5):
    sr, data = load_audio(path)
    audio = to_mono_16k(sr, data)
    sess = StreamSession()
    sess.correct_enabled = correct
    step = int(STREAM_EVERY * TARGET_SR)
    t_next = time.perf_counter()
    for i in range(0, len(audio), step):
        sess.push(audio[i:i + step])
        t_next += STREAM_EVERY / speed
        time.sleep(max(0.0, t_next - time.perf_counter()))
        if on_update:
            stage, _, _, stats = render(sess)
            on_update(stage, stats)
    sess.flush()
    deadline = time.time() + 30
    while any(s.status == "pending" for s in sess.segments) and time.time() < deadline:
        time.sleep(every)
        if on_update:
            stage, _, _, stats = render(sess)
            on_update(stage, stats)
    if on_update:
        stage, _, _, stats = render(sess)
        on_update(stage, stats)
    return sess


if __name__ == "__main__":
    launch(int(os.getenv("DATABRICKS_APP_PORT", "8000")))
