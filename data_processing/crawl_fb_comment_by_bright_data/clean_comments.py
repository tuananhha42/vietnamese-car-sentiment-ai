# -*- coding: utf-8 -*-
"""
Script làm sạch dữ liệu bình luận Facebook ngành Ô tô:
- Chuẩn hóa Unicode NFC.
- Loại bỏ các comment rỗng hoặc chỉ có khoảng trắng.
- Loại bỏ các comment vô nghĩa (chỉ gồm dấu chấm '.' để hóng bài, dấu câu, ký tự đặc biệt lặp lại).
- Loại bỏ các comment spam bán xe / cò mồi (chứa số điện thoại kèm từ khóa: ib, lh e, tư vấn, lăn bánh, trả góp...).
- Giữ nguyên cấu trúc cột và chuẩn mã hóa UTF-8 (utf-8-sig).
- Thống kê chi tiết từng nhóm dữ liệu rác đã được làm sạch.
"""

import argparse
import csv
import os
import re
import shutil
import sys
import unicodedata

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_FILE = os.path.join(CURRENT_DIR, "facebook_car_comments.csv")
DEFAULT_OUTPUT_FILE = os.path.join(CURRENT_DIR, "facebook_car_comments_clean.csv")

PHONE_PATTERN = re.compile(r"(\+?84|0)(3|5|7|8|9)\d{8}\b|(\+?84|0)[.\s]?(3|5|7|8|9)\d{2}[.\s]?\d{3}[.\s]?\d{3}\b")
SALES_SPAM_KEYWORDS = [
    "ib e", "ib em", "inbox em", "inbox e", "lh e", "lh em", "liên hệ e", "liên hệ em",
    "check inbox", "check tin nhắn", "chiết khấu tốt", "báo giá lăn bánh", "hỗ trợ trả góp"
]

def is_spam_or_noise(comment: str) -> (bool, str):
    """
    Kiểm tra xem bình luận có phải là rác/spam không.
    Trả về (is_noise, reason).
    """
    text = unicodedata.normalize("NFC", comment).strip()
    if not text:
        return True, "empty"

    # 1. Kiểm tra comment chỉ chứa dấu câu / ký tự đặc biệt (VD: '.', '...', '?', '!', '@@@')
    non_punct = re.sub(r"[^\w\s]", "", text)
    if not non_punct.strip():
        return True, "only_punctuation_or_dots"

    # 2. Kiểm tra comment quá ngắn vô nghĩa (độ dài chữ <= 1 ký tự, VD: 'a', 'x', '1')
    if len(non_punct.strip()) <= 1 and not any(ch in text for ch in ["👍", "❤️", "🔥", "👎"]):
        return True, "too_short_single_char"

    # 3. Kiểm tra comment "chấm hóng" phổ biến trên Facebook (VD: "chấm", "cham", "hóng", "hong")
    clean_lower = text.lower().strip()
    if clean_lower in ["chấm", "cham", ".", "..", "...", "hóng", "hong", "hóng hớt", "chấm hóng"]:
        return True, "dot_or_hong_bump"

    # 4. Kiểm tra spam bán xe / cò mồi: chứa số điện thoại hoặc từ khóa mời gọi liên hệ
    has_phone = bool(PHONE_PATTERN.search(text))
    has_sales_keyword = any(kw in clean_lower for kw in SALES_SPAM_KEYWORDS)
    if has_phone and has_sales_keyword:
        return True, "sales_phone_spam"

    return False, "valid"


def clean_comments_csv(input_file=DEFAULT_INPUT_FILE, output_file=DEFAULT_OUTPUT_FILE, backup=True):
    if not os.path.exists(input_file):
        print(f"[!] Lỗi: Không tìm thấy file '{input_file}'")
        return 0, 0, 0

    print("=" * 65)
    print(f"[*] BẮT ĐẦU LÀM SẠCH DỮ LIỆU BÌNH LUẬN XE HƠI TỪ: {os.path.basename(input_file)}")
    print("=" * 65)

    if backup and os.path.abspath(input_file) == os.path.abspath(output_file):
        backup_file = input_file + ".bak"
        shutil.copyfile(input_file, backup_file)
        print(f"[*] Đã tạo bản sao lưu tại: {backup_file}")

    total_rows = 0
    valid_rows = []
    stats = {
        "empty": 0,
        "only_punctuation_or_dots": 0,
        "too_short_single_char": 0,
        "dot_or_hong_bump": 0,
        "sales_phone_spam": 0,
        "missing_url": 0
    }
    header = None

    csv.field_size_limit(sys.maxsize)

    with open(input_file, mode="r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("[!] File rỗng, không có dữ liệu.")
            return 0, 0, 0

        try:
            comment_text_idx = header.index("comment_text")
        except ValueError:
            print("[!] Cảnh báo: Không tìm thấy cột 'comment_text' trong header!")
            return 0, 0, 0

        url_idx = header.index("url") if "url" in header else None

        for row in reader:
            total_rows += 1
            if not row:
                stats["empty"] += 1
                continue

            comment_val = ""
            if comment_text_idx < len(row):
                comment_val = row[comment_text_idx].strip()

            url_val = ""
            if url_idx is not None and url_idx < len(row):
                url_val = row[url_idx].strip()

            if url_idx is not None and not url_val:
                stats["missing_url"] += 1
                continue

            is_noise, reason = is_spam_or_noise(comment_val)
            if is_noise:
                stats[reason] = stats.get(reason, 0) + 1
            else:
                valid_rows.append(row)

    # Ghi ra file kết quả
    temp_output = output_file + ".tmp"
    with open(temp_output, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(valid_rows)

    if os.path.exists(output_file) and os.path.abspath(output_file) != os.path.abspath(temp_output):
        os.remove(output_file)
    os.rename(temp_output, output_file)

    kept_rows = len(valid_rows)
    total_removed = total_rows - kept_rows
    print(f"[✔] THỐNG KÊ KẾT QUẢ TIỀN XỬ LÝ & LÀM SẠCH:")
    print(f"    - Tổng số bản ghi đầu vào:                    {total_rows:>5}")
    print(f"    - Bình luận rỗng / không nội dung:            {stats['empty']:>5}")
    print(f"    - Chỉ chứa dấu chấm / dấu câu ('...', '.'):   {stats['only_punctuation_or_dots']:>5}")
    print(f"    - Comment 'chấm hóng', 'hóng' vô nghĩa:       {stats['dot_or_hong_bump']:>5}")
    print(f"    - Chữ đơn lẻ không rõ nghĩa (<= 1 ký tự):     {stats['too_short_single_char']:>5}")
    print(f"    - Spam bán xe / số điện thoại cò mồi:         {stats['sales_phone_spam']:>5}")
    print(f"    -----------------------------------------------------")
    print(f"    - Tổng số bản ghi rác bị loại bỏ:             {total_removed:>5} ({total_removed/total_rows*100:.2f}%)")
    print(f"    - Số bản ghi chất lượng cao giữ lại:          {kept_rows:>5} ({kept_rows/total_rows*100:.2f}%)")
    print(f"    - File kết quả lưu tại: {output_file}")
    print("=" * 65)

    return total_rows, kept_rows, total_removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Làm sạch dữ liệu CSV bình luận Facebook ngành ô tô")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_FILE, help="Đường dẫn file CSV đầu vào")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE, help="Đường dẫn file CSV đầu ra")
    parser.add_argument("--no-backup", action="store_true", help="Không tạo file sao lưu .bak")
    args = parser.parse_args()

    clean_comments_csv(
        input_file=args.input,
        output_file=args.output,
        backup=not args.no_backup
    )
