from pii import mask_pii

CASES = [
    ("số điện thoại của tôi là 0912 345 678 nhé", ["0912", "678"], ["[SĐT]"]),
    ("gọi cho em số 0912.345.678", ["0912"], ["[SĐT]"]),
    ("số của em là không chín một hai ba bốn năm sáu bảy tám", ["không chín một"], ["[SĐT]"]),
    ("liên hệ +84 912 345 678", ["912"], ["[SĐT]"]),
    ("căn cước công dân 001203004567", ["001203004567"], ["[CCCD]"]),
    ("chứng minh thư 123456789", ["123456789"], ["[CMND]"]),
    ("số tài khoản 1903 6789 1234 56 ngân hàng", ["1903"], ["[SỐ_TK]"]),
    ("email là nguyen.van.a@gmail.com", ["nguyen.van.a", "gmail"], ["[EMAIL]"]),
    ("mail của em là lananh a còng gmail chấm com", ["lananh", "gmail"], ["[EMAIL]"]),
    ("tên tôi là nguyễn văn an, tôi muốn hỏi", ["nguyễn văn an"], ["[TÊN]"]),
    ("họ và tên Trần Thị Bích Ngọc", ["Bích Ngọc"], ["[TÊN]"]),
    ("em tên là lan ạ", [" lan "], ["[TÊN]"]),
    ("xin chào, tôi là Phạm Minh Tuấn.", ["Tuấn"], ["[TÊN]"]),
    ("chị Hương ơi cho em hỏi", ["Hương"], ["chị [TÊN]"]),
    ("hôm qua gặp Lê Hoàng Nam ở công ty", ["Hoàng Nam"], ["[TÊN]"]),
    ("địa chỉ của tôi là số 12 ngõ 45 Nguyễn Trãi, Thanh Xuân. Cảm ơn", ["ngõ 45", "Thanh Xuân"], ["[ĐỊA_CHỈ]", "Cảm ơn"]),
    ("tôi sinh ngày 12/05/1990 tại Huế", ["12/05/1990"], ["[NGÀY_SINH]"]),
    ("ngày sinh là 3 tháng 4 năm 1985", ["1985"], ["[NGÀY_SINH]"]),
]

KEEP = [
    "tôi là sinh viên năm hai ở Hà Nội",
    "dự án này trị giá 150.000.000 đồng",
    "doanh thu tăng 12 phần trăm trong năm 2024",
    "chúng ta có một hai ba bốn bước cần làm",
    "Thành phố Hồ Chí Minh có mưa",
    "hệ thống xử lý được 250 yêu cầu mỗi giây",
    "anh ấy đang làm việc",
]


def main():
    bad = 0
    for text, gone, must in CASES:
        out, _ = mask_pii(text)
        ok = all(g not in out for g in gone) and all(m in out for m in must)
        bad += not ok
        print(("OK  " if ok else "FAIL"), f"{text!r}\n      -> {out!r}")
    for text in KEEP:
        out, found = mask_pii(text)
        ok = not found
        bad += not ok
        print(("OK  " if ok else "FAIL"), f"{text!r}\n      -> {out!r}")
    print(f"\n{len(CASES) + len(KEEP) - bad}/{len(CASES) + len(KEEP)} đạt")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
