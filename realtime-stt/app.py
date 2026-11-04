"""
POC phụ đề tiếng Việt near real-time trên Databricks Apps.

  Micro (trình duyệt) --0.5s--> on_chunk(): nối audio vào bộ đệm phiên, trả phụ đề mới nhất
                                   |
                       worker nền của phiên (VAD theo năng lượng)
                         - có thêm ~0.8s audio : Whisper cả câu đang nói -> che PII -> dòng tạm (nghiêng)
                         - gặp khoảng lặng     : Whisper lần cuối -> che PII -> chốt câu
                                                    |
                                       hàng đợi Gemini (chạy nền, không chặn Whisper)
                                         sửa câu -> kiểm tra nhãn PII -> che PII lần 2 -> thay câu trên phụ đề

Hai model gọi qua AI Gateway / Model Serving:
  WHISPER_ENDPOINT = ai_training_whisper_endpoint
  GEMINI_ENDPOINT  = gemini-3-8-flash-ai_training
BACKEND=mock để thử giao diện khi chưa có quyền gọi endpoint.
"""

from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import gcd

import gradio as gr
import numpy as np
from scipy.signal import resample_poly

from pii import mask_pii, placeholders_in

# ----------------------------------------------------------------------------
# Cấu hình (đổi trong app.yaml)
# ----------------------------------------------------------------------------
BACKEND = os.getenv("BACKEND", "gateway")  # gateway | mock
WHISPER_ENDPOINT = os.getenv("WHISPER_ENDPOINT", "ai_training_whisper_endpoint")
WHISPER_INPUT_FORMAT = os.getenv("WHISPER_INPUT_FORMAT", "auto")  # auto | dataframe:<cột> | inputs_list | inputs_str | inputs_dict | instances
WHISPER_PARAMS = json.loads(os.getenv("WHISPER_PARAMS", "{}") or "{}")  # vd {"language": "vi"} nếu endpoint hỗ trợ
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
WHISPER_CONCURRENCY = int(os.getenv("WHISPER_CONCURRENCY", "8"))  # tổng số lệnh gọi Whisper song song cho cả app
GEMINI_CONCURRENCY = int(os.getenv("GEMINI_CONCURRENCY", "4"))

FRAME = int(0.03 * TARGET_SR)  # khung 30 ms cho VAD


# ----------------------------------------------------------------------------
# Âm thanh + VAD
# ----------------------------------------------------------------------------
def to_mono_16k(sr: int, data: np.ndarray) -> np.ndarray:
    x = np.asarray(data)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float32)
    if sr != TARGET_SR:
        g = gcd(int(sr), TARGET_SR)
        x = resample_poly(x, TARGET_SR // g, int(sr) // g).astype(np.float32)
    return x


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
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, TARGET_SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


# Câu Whisper hay "bịa" khi gặp tiếng ồn / im lặng (học từ phụ đề YouTube)
HALLUCINATION_RE = re.compile(
    r"(hãy\s+subscribe|đăng\s+ký\s+kênh|cảm\s+ơn\s+các\s+bạn\s+đã\s+(?:xem|theo\s+dõi|lắng\s+nghe)"
    r"|hẹn\s+gặp\s+lại\s+các\s+bạn|ghiền\s+mì\s+gõ|vietsub|subtitles?\s+by|like\s+và\s+share|bấm\s+chuông)",
    re.IGNORECASE,
)


def drop_hallucinations(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(p for p in parts if p and not HALLUCINATION_RE.search(p)).strip()


# ----------------------------------------------------------------------------
# Gọi endpoint qua Databricks SDK
# ----------------------------------------------------------------------------
_API = None
_API_LOCK = threading.Lock()


def api():
    """api_client của WorkspaceClient. Trong Databricks Apps tự xác thực bằng service principal của app."""
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
    """Lấy chữ từ nhiều kiểu response: predictions/outputs/text/choices, str hoặc dict hoặc list."""
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
    if isinstance(content, list):  # một số model trả về danh sách khối (text / reasoning)
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type", "text") == "text")
    return content or ""


class WhisperClient:
    """Tự dò định dạng đầu vào của endpoint lần gọi đầu tiên, sau đó dùng cố định."""

    AUTO_FORMATS = ["dataframe:audio", "dataframe:audio_b64", "dataframe:audio_base64", "dataframe:input",
                    "inputs_list", "inputs_str", "inputs_dict", "instances"]

    def __init__(self, endpoint: str, fmt: str):
        self.endpoint = endpoint
        self.fmt = None if fmt == "auto" else fmt
        self.detect_lock = threading.Lock()
        self.sem = threading.Semaphore(WHISPER_CONCURRENCY)

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
                return extract_text(invoke(self.endpoint, self.build(self.fmt, b64))).strip()
            with self.detect_lock:
                if self.fmt:  # luồng khác vừa dò xong
                    return extract_text(invoke(self.endpoint, self.build(self.fmt, b64))).strip()
                return self._detect(b64)

    def _detect(self, b64: str) -> str:
        from databricks.sdk.errors import BadRequest, InternalError

        errors = []
        for fmt in self.AUTO_FORMATS:
            try:
                text = extract_text(invoke(self.endpoint, self.build(fmt, b64))).strip()
            except (BadRequest, InternalError) as e:  # sai định dạng -> thử kiểu khác
                errors.append(f"{fmt}: {str(e)[:160]}")
                continue
            self.fmt = fmt
            print(f"[whisper] định dạng đầu vào phù hợp: {fmt}", flush=True)
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
    """Kiểm tra đầu ra của Gemini. Trả (câu dùng để hiển thị, trạng thái)."""
    out = (out or "").strip().strip('"“”`').strip()
    out = re.sub(r"^(?:câu\s+đã\s+sửa|kết\s+quả)\s*:\s*", "", out, flags=re.IGNORECASE).strip()
    if not out:
        return raw, "kept"
    if placeholders_in(out) != placeholders_in(raw):
        return raw, "kept"  # Gemini làm mất / thêm / sửa nhãn PII -> không tin
    ratio = len(out) / max(1, len(raw))
    lo, hi = (0.6, 1.6) if len(raw) >= 15 else (0.3, 2.5)
    if not lo <= ratio <= hi:
        return raw, "kept"  # viết lại quá nhiều -> có thể đã "sáng tác"
    return out, "done"


# ----------------------------------------------------------------------------
# Giả lập (BACKEND=mock)
# ----------------------------------------------------------------------------
class MockWhisper:
    label = "mock"

    def __call__(self, audio):
        time.sleep(0.2)
        return f"đoạn nói dài {len(audio) / TARGET_SR:.1f} giây số của tôi là không chín một hai ba bốn năm sáu bảy tám"


class MockGemini:
    def __call__(self, text, context):
        time.sleep(0.6)
        return text[:1].upper() + text[1:].replace("đoạn nói", "Đoạn nói") + "."


# ----------------------------------------------------------------------------
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
    raw: str                     # chữ Whisper, đã che PII
    text: str                    # chữ hiển thị (sau Gemini nếu có)
    status: str = "pending"      # pending | done | kept | off | error
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
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- luồng Gradio ----
    def push(self, audio: np.ndarray):
        with self.lock:
            self.buf = np.concatenate([self.buf, audio])
            self.audio_total_sec += len(audio) / TARGET_SR
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

    # ---- nhận dạng ----
    def _whisper(self, audio: np.ndarray) -> tuple[str, list[str]]:
        whisper, _ = models()
        t0 = time.perf_counter()
        text = whisper(audio)
        self.t_whisper = time.perf_counter() - t0
        self.error = ""
        return mask_pii(drop_hallucinations(text))

    def _commit(self, seg: np.ndarray):
        if speech_seconds(frame_rms(seg)) < MIN_SPEECH_SEC:
            return
        try:
            text, pii = self._whisper(trim_silence(seg))
        except Exception as e:  # noqa: BLE001
            self.error = f"Whisper lỗi: {e}"
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
        except Exception as e:  # noqa: BLE001
            s.status, self.error = "error", f"Gemini lỗi: {e}"
            return
        self.t_gemini = time.perf_counter() - t0
        text, status = sanitize_correction(s.raw, out)
        text, pii2 = mask_pii(text)  # lần 2: bắt PII mà Gemini "làm lộ" (vd sửa chữ số thành số)
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
                except Exception as e:  # noqa: BLE001
                    self.error = f"Whisper lỗi: {e}"
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


# ----------------------------------------------------------------------------
# Giao diện phụ đề
# ----------------------------------------------------------------------------
def _seg_html(s: Segment) -> str:
    return f"<span class='seg {s.status}'>{html.escape(s.text)}</span>"


def render(sess: StreamSession | None, show_raw: bool = False):
    if sess is None:
        stage = "<div class='stage'><div class='line hint'>Bấm Record và bắt đầu nói…</div></div>"
        return stage, "", "", ""
    segs, partial, pending = sess.snapshot()

    lines = [f"<div class='line old'>{_seg_html(s)}</div>" for s in segs[-2:]]
    if partial:
        lines.append(f"<div class='line partial'>{html.escape(partial)}</div>")
    elif not segs:
        lines.append("<div class='line hint'>đang nghe…</div>")
    if lines and not partial:
        lines[-1] = lines[-1].replace("line old", "line")
    stage = "<div class='stage'>" + "".join(lines[-3:]) + "</div>"

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
    stats = (f"Whisper: {sess.t_whisper:.2f}s ({getattr(whisper, 'label', '')}) · "
             f"Gemini: {sess.t_gemini:.2f}s, đang chờ {waiting} câu · "
             f"đã che {n_pii} PII · audio chờ: {pending:.1f}s · đã nghe {sess.audio_total_sec:.0f}s")
    if sess.error:
        stats += f"  \n⚠️ {html.escape(sess.error[:400])}"
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
.stage .line * {color: inherit !important;}
.seg.pending {border-bottom: 1px dotted currentColor;}
.transcript {font-size: 1.05rem; line-height: 1.7;}
.transcript .seg.pending, .cmp .seg.pending {color: var(--body-text-color-subdued);}
.cmp {width: 100%; border-collapse: collapse; font-size: .95rem;}
.cmp td, .cmp th {border-bottom: 1px solid var(--border-color-primary); padding: 6px 8px; vertical-align: top; text-align: left;}
"""


def build_ui() -> gr.Blocks:
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
        # Cập nhật phụ đề đều đặn để câu Gemini sửa xong vẫn hiện ra cả khi đã dừng ghi âm
        gr.Timer(0.5).tick(on_tick, [sess, show_raw], outs[1:], show_progress="hidden", concurrency_limit=None)
    return demo


if __name__ == "__main__":
    models()
    build_ui().queue(default_concurrency_limit=None).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("DATABRICKS_APP_PORT", "8000")),
    )
