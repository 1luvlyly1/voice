"""
Che thông tin cá nhân (PII) trong văn bản tiếng Việt từ nhận dạng giọng nói.

Nguyên tắc: thà che nhầm còn hơn để lọt. Thứ tự che: email -> dãy số -> ngày sinh -> địa chỉ -> tên.
Hàm chính: mask_pii(text) -> (text_đã_che, danh_sách_loại_PII)

Có xử lý số đọc bằng chữ mà Whisper hay trả về, ví dụ "không chín một hai ba bốn năm sáu bảy tám".
Chỉ là heuristic (regex + ngữ cảnh), đủ cho POC; cần thử với dữ liệu thật và bổ sung luật.
"""

from __future__ import annotations

import re
import unicodedata

PLACEHOLDERS = ("SĐT", "CCCD", "CMND", "SỐ_TK", "EMAIL", "TÊN", "ĐỊA_CHỈ", "NGÀY_SINH")
PLACEHOLDER_RE = re.compile(r"\[(?:" + "|".join(PLACEHOLDERS) + r")\]")

# ----------------------------------------------------------------------------
# Email (viết hoặc đọc: "nguyenvana a còng gmail chấm com")
# ----------------------------------------------------------------------------
_TLD = r"(?:com|vn|net|org|edu|gov|io|info|biz|me|co)"
_DOT = r"\s*(?:\.|\bchấm\b|\bdot\b)\s*"
EMAIL_RE = re.compile(
    r"[\w.+\-]+\s*(?:@|\ba\s+còng\b|\ba\s+móc\b|\bat\b)\s*[\w\-]+(?:" + _DOT + r"[\w\-]+)*" + _DOT + _TLD + r"\b",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------------
# Dãy số (kể cả số đọc bằng chữ)
# ----------------------------------------------------------------------------
DIGIT_WORDS = {
    "không": "0", "linh": "0", "lẻ": "0",
    "một": "1", "mốt": "1", "hai": "2", "ba": "3", "bốn": "4", "tư": "4",
    "năm": "5", "lăm": "5", "sáu": "6", "bảy": "7", "bẩy": "7", "tám": "8", "chín": "9",
}
_TOKEN = r"(?:\+?\d+|" + "|".join(sorted(DIGIT_WORDS, key=len, reverse=True)) + r")"
NUMBER_RUN_RE = re.compile(rf"(?<!\w){_TOKEN}(?:[\s.,\-]+{_TOKEN})*(?!\w)", re.IGNORECASE)
_TOKEN_RE = re.compile(_TOKEN, re.IGNORECASE)
# Số tiền / số lượng thì không phải PII
MONEY_AFTER_RE = re.compile(r"\s*(?:đồng|đ\b|vnđ|vnd|usd|\$|triệu|tỷ|tỉ|nghìn|ngàn|k\b|%|phần\s+trăm)", re.IGNORECASE)


def _digits_of(run: str) -> str:
    out = []
    for tok in _TOKEN_RE.findall(run):
        t = tok.lower()
        out.append(DIGIT_WORDS.get(t, t.lstrip("+")))
    return "".join(out)


def _classify_number(d: str) -> str | None:
    n = len(d)
    if n < 9:
        return None
    if d.startswith("84") and n == 11:
        return "SĐT"
    if d.startswith("0") and n in (10, 11):
        return "SĐT"
    if n == 12:
        return "CCCD"
    if n == 9:
        return "CMND"
    return "SỐ_TK"


def _mask_numbers(text: str, found: list[str]) -> str:
    def repl(m: re.Match) -> str:
        if MONEY_AFTER_RE.match(text, m.end()):
            return m.group(0)
        kind = _classify_number(_digits_of(m.group(0)))
        if kind is None:
            return m.group(0)
        found.append(kind)
        return f"[{kind}]"

    return NUMBER_RUN_RE.sub(repl, text)


# ----------------------------------------------------------------------------
# Ngày sinh
# ----------------------------------------------------------------------------
_DATE = (
    r"(?:\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{2,4}"
    r"|\d{1,2}\s+tháng\s+\d{1,2}(?:\s+năm\s+\d{2,4})?"
    r"|(?:năm\s+)?\d{4})"
)
DOB_RE = re.compile(r"(\b(?:sinh\s+(?:vào\s+)?(?:ngày|năm)?|ngày\s+sinh(?:\s+(?:là|:))?|năm\s+sinh(?:\s+(?:là|:))?)\s*)" + _DATE,
                    re.IGNORECASE)

# ----------------------------------------------------------------------------
# Địa chỉ: che phần sau từ khoá cho tới hết câu
# ----------------------------------------------------------------------------
ADDRESS_RE = re.compile(
    r"(\b(?:địa\s+chỉ(?:\s+(?:của\s+)?(?:tôi|em|mình|tui|nhà|anh|chị|khách\s+hàng))?(?:\s+(?:là|ở|tại))?"
    r"|số\s+nhà|nhà\s+(?:tôi|em|mình|tui|anh|chị)\s+(?:ở|tại)(?:\s+số)?"
    r"|(?:tôi|em|mình|tui)\s+(?:sống|ở)\s+(?:tại\s+|ở\s+)?số)\s*:?\s*)"
    r"([^.!?;\n]{3,120})",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------------
# Tên người
# ----------------------------------------------------------------------------
_PRON = r"(?:tôi|em|mình|tui|tớ|cháu|con|anh|chị)"
# Sau các cụm này, che tên dù viết thường hay viết hoa
NAME_CUE_ANYCASE_RE = re.compile(
    r"\b(?:(?:họ\s+(?:và\s+)?)?tên\s+(?:(?:của\s+)?" + _PRON + r"\s+)?(?:là|:)"
    r"|họ\s+(?:và\s+)?tên(?:\s+(?:là|:))?"
    r"|tên\s+đầy\s+đủ(?:\s+(?:là|:))?"
    r"|" + _PRON + r"\s+tên(?:\s+là)?)\s+",
    re.IGNORECASE,
)
# Sau các cụm này, chỉ che nếu từ tiếp theo viết hoa ("tôi là Lan" nhưng không che "tôi là sinh viên")
NAME_CUE_CAPITAL_RE = re.compile(
    r"\b(?:(?:tôi|em|mình|tui|tớ|cháu|con)\s+là|gọi\s+(?:tôi|em|mình)\s+là"
    r"|anh|chị|ông|bà|cô|chú|bác|bạn|em|cháu|thầy|dì|cậu|mợ|dượng|ngài|khách\s+hàng)\s+",
    re.IGNORECASE,
)
SURNAMES = {
    "nguyễn", "trần", "lê", "phạm", "hoàng", "huỳnh", "phan", "vũ", "võ", "đặng", "bùi", "đỗ", "hồ",
    "ngô", "dương", "lý", "đinh", "trịnh", "đoàn", "lâm", "mai", "trương", "cao", "lương", "tạ", "hà",
    "tô", "châu", "quách", "lưu", "thái", "kiều", "từ", "la", "triệu", "khổng", "mạc", "tăng", "văn",
}
NOT_NAMES = {  # cụm viết hoa bắt đầu bằng họ nhưng là địa danh/tổ chức hay gặp
    "hà nội", "hà giang", "hà tĩnh", "hà nam", "hà đông", "hồ chí minh", "cao bằng", "lâm đồng",
    "thái bình", "thái nguyên", "mai châu", "hồ gươm", "hồ tây", "lý sơn", "tô lịch",
}
NAME_STOP = {
    "và", "ở", "nhà", "sinh", "số", "năm", "tuổi", "đang", "làm", "thì", "muốn", "cần", "hỏi", "gọi",
    "từ", "của", "có", "là", "đây", "ạ", "nha", "nhé", "đó", "nhưng", "mà", "với", "hiện", "hôm",
    "ngày", "bên", "tại", "đến", "nay", "rồi", "không", "đã", "sẽ", "được", "bị", "cho", "này", "ơi",
    "em", "anh", "chị", "tôi", "mình", "quê", "ạ", "vâng", "dạ", "thưa", "sống", "học", "người",
}
_WORD_RE = re.compile(r"[ \t]*([^\W\d_]+)")
_UP = "A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ"
_CW = rf"[{_UP}][^\W\d_]*"
CAP_SEQ_RE = re.compile(rf"(?<!\w){_CW}(?:[ \t]+{_CW})+")
_MAX_NAME_WORDS = 4


def _take_name(text: str, pos: int, need_capital: bool) -> int:
    """Trả vị trí kết thúc của tên bắt đầu tại pos (hoặc pos nếu không có tên)."""
    end, count = pos, 0
    while count < _MAX_NAME_WORDS:
        m = _WORD_RE.match(text, end)
        if not m:
            break
        w = m.group(1)
        if w.lower() in NAME_STOP or (need_capital and not w[0].isupper()):
            break
        end, count = m.end(), count + 1
    return end


def _mask_names(text: str, found: list[str]) -> str:
    spans: list[tuple[int, int]] = []
    for cue_re, need_cap in ((NAME_CUE_ANYCASE_RE, False), (NAME_CUE_CAPITAL_RE, True)):
        for m in cue_re.finditer(text):
            end = _take_name(text, m.end(), need_cap)
            if end > m.end():
                spans.append((m.end(), end))
    # "Nguyễn Văn An", "gặp Trần Thị B" ... không cần từ khoá (chỉ dạng viết hoa, bắt đầu bằng họ)
    for m in CAP_SEQ_RE.finditer(text):
        words = list(re.finditer(r"\S+", m.group(0)))
        for i, w in enumerate(words[:-1]):
            if w.group(0).lower() not in SURNAMES:
                continue
            tail = words[i: i + _MAX_NAME_WORDS]
            phrase = " ".join(x.group(0) for x in tail).lower()
            if not any(phrase.startswith(p) for p in NOT_NAMES):
                spans.append((m.start() + tail[0].start(), m.start() + tail[-1].end()))
            break
    if not spans:
        return text
    # gộp các khoảng chồng nhau rồi thay từ cuối lên đầu
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    for s, e in reversed(merged):
        found.append("TÊN")
        text = text[:s] + "[TÊN]" + text[e:]
    return text


# ----------------------------------------------------------------------------
def mask_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    if not text:
        return text, found
    text = unicodedata.normalize("NFC", text)

    def email_repl(m):
        found.append("EMAIL")
        return "[EMAIL]"

    text = EMAIL_RE.sub(email_repl, text)

    def dob_repl(m):
        found.append("NGÀY_SINH")
        return m.group(1) + "[NGÀY_SINH]"

    text = DOB_RE.sub(dob_repl, text)
    text = _mask_numbers(text, found)

    def addr_repl(m):
        if PLACEHOLDER_RE.fullmatch(m.group(2).strip()):
            return m.group(0)
        found.append("ĐỊA_CHỈ")
        return m.group(1) + "[ĐỊA_CHỈ]"

    text = ADDRESS_RE.sub(addr_repl, text)
    text = _mask_names(text, found)
    return text, found


def placeholders_in(text: str) -> list[str]:
    return sorted(PLACEHOLDER_RE.findall(text))
