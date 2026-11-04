# Phụ đề tiếng Việt realtime trên Databricks Apps

Nói đến đâu, phụ đề hiện đến đó. App chỉ dùng 2 endpoint qua AI Gateway, không chạy model nào bên trong.

```
Micro ─0,5s─▶ VAD ─▶ ai_training_whisper_endpoint ─▶ bỏ câu "bịa" ─▶ che PII ─▶ dòng tạm (nghiêng)
                              │ khi ngừng nói ~0,6s
                              ▼
                 chốt câu ─▶ gemini-3-8-flash-ai_training (chạy nền, kèm 3 câu trước làm ngữ cảnh)
                              ▼
                 kiểm tra nhãn PII còn đủ ─▶ che PII lần 2 ─▶ thay câu trên phụ đề
```

- Gemini chỉ được gọi cho **câu đã chốt**, không gọi cho dòng tạm, để giảm độ trễ và số lần gọi.
- Gemini **không bao giờ thấy PII thật**: chữ đã được che trước khi gửi. Nếu Gemini làm mất hoặc thêm nhãn như `[SĐT]`, hay viết lại câu quá nhiều, app bỏ kết quả của Gemini và giữ câu gốc.
- **Che PII lần 2** sau Gemini: ví dụ Whisper ra "nguyễn văn nam" viết thường nên lần đầu không bắt được, Gemini viết hoa thành "Nguyễn Văn Nam" thì lần 2 sẽ che.
- Không lưu gì cả, phụ đề chỉ nằm trong bộ nhớ của phiên đang mở.

## Các file

| File | Vai trò |
|---|---|
| `app.py` | Giao diện phụ đề, VAD, gọi Whisper và Gemini, luồng xử lý nền |
| `pii.py` | Che PII tiếng Việt: SĐT, CCCD/CMND, số tài khoản/thẻ, email (kể cả đọc "a còng… chấm com"), tên, địa chỉ, ngày sinh. Bắt được cả số đọc bằng chữ ("không chín một hai…") |
| `test_pii.py` | Bộ test PII. Thêm câu thật của bạn vào đây rồi chạy `python test_pii.py` |
| `app.yaml` | Tên endpoint, thuật ngữ, độ nhạy |
| `requirements.txt` | Thư viện cần cài |

## Deploy

1. Upload thư mục lên Workspace, ví dụ `/Workspace/Users/<bạn>/realtime-stt`.
2. Vào **Compute → Apps → Create app → Custom**, đặt tên `realtime-stt`.
3. Ở mục **App resources**, thêm 2 *Serving endpoint*: `ai_training_whisper_endpoint` và `gemini-3-8-flash-ai_training`, đều với quyền **Can query**. App gọi endpoint bằng service principal của chính nó, nên thiếu bước này sẽ bị lỗi 403.
4. Bấm **Deploy** và trỏ tới thư mục. Mở URL app, cho phép micro, bấm **Record**.

Có thể deploy bằng CLI:
```bash
databricks sync . /Workspace/Users/<bạn>/realtime-stt
databricks apps deploy realtime-stt --source-code-path /Workspace/Users/<bạn>/realtime-stt
```

## Định dạng đầu vào của Whisper endpoint

Mỗi endpoint Whisper tự đóng gói nhận dữ liệu theo một kiểu khác nhau. Với `WHISPER_INPUT_FORMAT=auto`, app tự thử lần lượt các kiểu dưới đây ở lần gọi đầu, mỗi kiểu gửi file WAV 16 kHz mã hoá base64:

| Giá trị | Body gửi đi |
|---|---|
| `dataframe:audio` | `{"dataframe_records": [{"audio": "<b64>"}]}`. Thay `audio` bằng tên cột thật nếu khác |
| `inputs_list` | `{"inputs": ["<b64>"]}` |
| `inputs_str` | `{"inputs": "<b64>"}` |
| `inputs_dict` | `{"inputs": {"audio": "<b64>"}}` |
| `instances` | `{"instances": ["<b64>"]}` |

Kiểu nào chạy được sẽ được in ra log app (`[whisper] định dạng đầu vào phù hợp: …`) và hiện ở dòng trạng thái. Khi đã biết, hãy điền cố định vào `app.yaml` để bỏ bước dò.

Nếu không kiểu nào chạy, dòng trạng thái sẽ hiện thông báo lỗi của từng kiểu. Thường thông báo đó nói rõ endpoint cần cột gì, bạn chỉ cần đặt `WHISPER_INPUT_FORMAT=dataframe:<tên_cột>`. Nếu endpoint cần dữ liệu không phải WAV base64, phải sửa hàm `WhisperClient.build` trong `app.py`.

## Tinh chỉnh (`app.yaml`)

| Biến | Mặc định | Khi nào chỉnh |
|---|---|---|
| `GLOSSARY` | vài thuật ngữ mẫu | Điền tên sản phẩm, thuật ngữ riêng để Gemini sửa đúng từ bị nghe nhầm |
| `SILENCE_RMS` | 0.012 | Phòng ồn mà câu không chốt → tăng lên 0.02–0.03. Nói nhỏ bị mất chữ → giảm |
| `SILENCE_SEC` | 0.6 | Câu bị cắt khi đang ngập ngừng → tăng lên 0.8–1.0 |
| `PARTIAL_STEP_SEC` | 0.8 | Gặp lỗi 429 (rate limit) của Whisper → tăng lên 1.5. Mỗi người nói tạo khoảng 60–80 lần gọi Whisper/phút |
| `WHISPER_PARAMS` | trống | Ví dụ `{"language": "vi"}` nếu endpoint nhận tham số này |
| `BACKEND` | gateway | Đặt `mock` để thử giao diện mà không gọi endpoint |

Ô **"So sánh với chữ Whisper"** trên giao diện hiện bảng đối chiếu từng câu: chữ Whisper và chữ hiển thị. Dùng bảng này để đánh giá Gemini có sửa đúng không. Ô **"Gemini sửa câu"** cho phép tắt Gemini để so sánh độ trễ.

## Cần kiểm tra với admin

- **Inference table / payload logging của AI Gateway:** nếu endpoint Whisper bật tính năng này, âm thanh và chữ **chưa che PII** sẽ được ghi vào bảng Delta. App không kiểm soát được việc này.
- **Rate limit** (QPM) của 2 endpoint so với số người dùng cùng lúc.

## Giới hạn của POC

- Bộ lọc PII dùng regex và ngữ cảnh, nên thiên về che nhầm hơn là để lọt. Nó **không** bắt được tên đứng một mình mà không có từ gợi ý, ví dụ "cô giáo Mai". Hãy thêm câu thật vào `test_pii.py` để mở rộng luật.
- VAD dựa trên năng lượng âm thanh, hợp với phòng yên tĩnh.
- Dòng tạm là kết quả Whisper nhận dạng lại cả câu đang nói, nên chi phí tăng theo độ dài câu (tối đa 12 giây).
