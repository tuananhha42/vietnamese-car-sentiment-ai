# -*- coding: utf-8 -*-
"""
Script Kiểm thử Toàn diện Hệ thống Phân tích Cảm xúc Xe Hơi Tiếng Việt
(Comprehensive Testing Suite for Vietnamese Car Sentiment Analysis)

Tính năng:
- Hỗ trợ 2 chế độ kiểm thử:
  + Local Engine Mode (`--mode local`): Gọi trực tiếp mô hình PhoBERT qua GPU/CPU.
  + REST API Mode (`--mode api`): Kiểm thử End-to-End thông qua Microservice (Port 8000).
- Đánh giá trên 100 câu bình luận thực tế từ Facebook kèm bộ kiểm thử trường hợp biên (Edge Cases).
- Đo lường chi tiết: Thời gian phản hồi (Latency ms/câu), Độ tin cậy trung bình, Phân phối nhãn, Negative Recall.
- Định dạng bảng kết quả trực quan bằng thư viện tabulate và xuất file test_100_results.csv.
"""

import argparse
import os
import sys
import time
import requests
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from classifier import VietnameseSentimentClassifier

EDGE_CASES = [
    {"text": ".", "note": "Dấu chấm 'chấm hóng' (Edge Case)"},
    {"text": "@@@###$$$ xe này chán vkl 😡😡😡", "note": "Ký tự lạ, emoij & tiếng lóng"},
    {"text": "550tr lăn bánh còn thừa cầm về, gầm cao 5 chỗ mới tinh lh: 0918.079.865", "note": "Spam cò mồi bán xe & số ĐT"},
    {"text": "Xe đi rất đầm chắc, tăng tốc bốc và cực kỳ tiết kiệm điện!", "note": "Khen ngợi trải nghiệm lái (POS)"},
    {"text": "Chạy được 3 tháng đã lỗi pin, dịch vụ bảo hành quá thất vọng", "note": "Khiếu nại sự cố nghiêm trọng (NEG)"},
    {"text": "Xe tưởng không ngon mà ngon không tưởng", "note": "Phủ định & đảo ngữ phức tạp"}
]

def run_local_test(texts):
    print("[*] Đang khởi tạo mô hình PhoBERT Local Engine...")
    classifier = VietnameseSentimentClassifier()
    start_time = time.time()
    predictions = classifier.predict_batch(texts, batch_size=32)
    elapsed_time = time.time() - start_time
    avg_latency = (elapsed_time / len(texts)) * 1000
    return predictions, avg_latency

def run_api_test(texts, api_url="http://127.0.0.1:8000/predict"):
    print(f"[*] Đang kiểm thử qua REST API Microservice: {api_url}")
    predictions = []
    latencies = []

    # Health check trước
    health_url = api_url.replace("/predict", "/health")
    try:
        h = requests.get(health_url, timeout=2.0)
        if h.status_code != 200:
            raise ConnectionError(f"Health check thất bại: {h.status_code}")
    except Exception as e:
        raise ConnectionError(f"Không thể kết nối đến REST API tại {health_url}. Vui lòng bật API trước (`python3 build_AI_model/api.py`) hoặc dùng cờ `--mode local`! Lỗi: {e}")

    for idx, text in enumerate(texts):
        t0 = time.time()
        try:
            resp = requests.post(api_url, json={"text": text}, timeout=10)
            lat = (time.time() - t0) * 1000
            latencies.append(lat)
            if resp.status_code == 200:
                predictions.append(resp.json())
            else:
                err_detail = resp.json().get("detail", "Lỗi")
                predictions.append({
                    "text": text,
                    "cleaned_text": "",
                    "label": "LỖI BIÊN (HTTP 400)",
                    "sentiment": "ERR",
                    "confidence": 0.0,
                    "detail_scores": {"POS": 0.0, "NEG": 0.0, "NEU": 0.0},
                    "warning": f"HTTP {resp.status_code}: {err_detail}"
                })
        except Exception as e:
            lat = (time.time() - t0) * 1000
            latencies.append(lat)
            predictions.append({
                "text": text,
                "cleaned_text": "",
                "label": "LỖI KẾT NỐI",
                "sentiment": "ERR",
                "confidence": 0.0,
                "detail_scores": {"POS": 0.0, "NEG": 0.0, "NEU": 0.0},
                "warning": str(e)
            })

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return predictions, avg_latency

def main():
    parser = argparse.ArgumentParser(description="Kiểm thử toàn diện hệ thống phân tích cảm xúc xe hơi tiếng Việt")
    parser.add_argument("--mode", type=str, choices=["local", "api"], default="local", help="Chế độ: 'local' (trực tiếp mô hình) hoặc 'api' (qua REST API Microservice)")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000/predict", help="Endpoint API dự đoán")
    parser.add_argument("--limit", type=int, default=100, help="Số lượng bình luận kiểm thử (mặc định: 100)")
    args = parser.parse_args()

    print("=" * 80)
    print(f"BẮT ĐẦU KIỂM THỬ THỰC TẾ TRÊN {args.limit} BÌNH LUẬN XE HƠI (CHẾ ĐỘ: {args.mode.upper()})")
    print("=" * 80)

    # 1. Nạp dữ liệu
    test_csv = os.path.join(CURRENT_DIR, "data/test.csv")
    if not os.path.exists(test_csv):
        test_csv = os.path.join(CURRENT_DIR, "data/val_comments.csv")
    if not os.path.exists(test_csv):
        test_csv = os.path.join(CURRENT_DIR, "../data_processing/crawl_fb_comment_by_bright_data/facebook_car_comments_clean.csv")

    print(f"[*] Đọc dữ liệu kiểm thử từ: {test_csv}")
    df = pd.read_csv(test_csv)
    df = df.dropna(subset=["comment_text"]).copy()
    df["comment_text"] = df["comment_text"].astype(str).str.strip()
    df = df[df["comment_text"].str.len() > 2].reset_index(drop=True)

    test_sample = df.head(args.limit).copy()
    raw_texts = test_sample["comment_text"].tolist()

    # 2. Thực thi kiểm thử
    if args.mode == "local":
        predictions, avg_latency = run_local_test(raw_texts)
    else:
        predictions, avg_latency = run_api_test(raw_texts, api_url=args.api_url)

    # 3. Tổng hợp kết quả
    table_rows = []
    for idx, (text, pred) in enumerate(zip(raw_texts, predictions)):
        short_text = (text[:55] + "...") if len(text) > 55 else text
        short_text = short_text.replace("\n", " ")
        conf_str = f"{pred['confidence'] * 100:.1f}%" if pred['confidence'] > 0 else "N/A"
        table_rows.append([
            idx + 1,
            short_text,
            pred["label"],
            pred["sentiment"],
            conf_str,
            f"{pred['detail_scores'].get('POS', 0.0):.2f}",
            f"{pred['detail_scores'].get('NEG', 0.0):.2f}",
            f"{pred['detail_scores'].get('NEU', 0.0):.2f}"
        ])

    headers = ["STT", "Nội dung bình luận", "Nhãn nhận diện", "Mã", "Độ tin cậy", "POS", "NEG", "NEU"]

    # In ra 25 câu mẫu minh họa
    print("\n--- BẢNG KẾT QUẢ MẪU (25 CÂU ĐẦU TIÊN TRONG BỘ TEST) ---")
    print(tabulate(table_rows[:25], headers=headers, tablefmt="github"))

    # 4. Thống kê chỉ số định lượng
    print("\n" + "=" * 80)
    print("TỔNG HỢP CHỈ SỐ KỸ THUẬT & HIỆU NĂNG HỆ THỐNG:")
    print("=" * 80)

    pred_sentiments = [p["sentiment"] for p in predictions]
    pred_labels = [p["label"] for p in predictions]
    confidences = [p["confidence"] for p in predictions if p["confidence"] > 0]

    pos_count = pred_sentiments.count("POS")
    neg_count = pred_sentiments.count("NEG")
    neu_count = pred_sentiments.count("NEU")
    err_count = pred_sentiments.count("ERR")
    total_valid = len(predictions) - err_count

    print(f"- Tổng số mẫu kiểm thử:           {len(predictions):>5} câu")
    print(f"- Thời gian xử lý trung bình:      {avg_latency:>8.2f} ms/câu")
    print(f"- Độ tin cậy trung bình (Conf):    {sum(confidences)/len(confidences)*100:>7.2f}%" if confidences else "N/A")
    print("\nPhân phối nhãn dự đoán:")
    print(f"  + TÍCH CỰC (POS) : {pos_count:>4} câu ({pos_count/len(predictions)*100:.1f}%)")
    print(f"  + TIÊU CỰC (NEG) : {neg_count:>4} câu ({neg_count/len(predictions)*100:.1f}%)")
    print(f"  + TRUNG TÍNH (NEU): {neu_count:>4} câu ({neu_count/len(predictions)*100:.1f}%)")
    if err_count > 0:
        print(f"  + LỖI BIÊN (ERR)  : {err_count:>4} câu ({err_count/len(predictions)*100:.1f}%)")

    # 5. Kiểm thử bổ sung bộ Edge Cases
    print("\n" + "=" * 80)
    print("KIỂM THỬ ĐẶC BIỆT CÁC TRƯỜNG HỢP BIÊN (EDGE CASES BATTERY):")
    print("=" * 80)
    edge_texts = [e["text"] for e in EDGE_CASES]
    if args.mode == "local":
        edge_preds, _ = run_local_test(edge_texts)
    else:
        edge_preds, _ = run_api_test(edge_texts, api_url=args.api_url)

    edge_table = []
    for e, p in zip(EDGE_CASES, edge_preds):
        edge_table.append([
            e["text"][:35],
            e["note"],
            p["label"],
            f"{p['confidence']*100:.1f}%" if p['confidence'] > 0 else "0%",
            p.get("warning") or "None"
        ])
    edge_headers = ["Đầu vào", "Mô tả trường hợp biên", "Kết quả nhận diện", "Độ tin cậy", "Cảnh báo / Warning"]
    print(tabulate(edge_table, headers=edge_headers, tablefmt="github"))

    # 6. Lưu file CSV
    out_csv = os.path.join(CURRENT_DIR, "data/test_100_results.csv")
    test_sample["predicted_sentiment"] = [p["sentiment"] for p in predictions]
    test_sample["predicted_label"] = [p["label"] for p in predictions]
    test_sample["confidence_score"] = [p["confidence"] for p in predictions]
    test_sample.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[✔] Đã lưu đầy đủ chi tiết kết quả vào: {out_csv}")
    print("=" * 80)

if __name__ == "__main__":
    main()
