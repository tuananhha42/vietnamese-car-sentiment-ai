import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Thêm thư mục hiện tại vào sys.path để import classifier
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from classifier import VietnameseSentimentClassifier

def main():
    print("=" * 60)
    print("BẮT ĐẦU QUÁ TRÌNH TỰ ĐỘNG GÁN NHÃN VÀ PHÂN CHIA TẬP TRAIN / VAL")
    print("=" * 60)
    
    # Đường dẫn file
    input_csv = os.path.join(CURRENT_DIR, "../data_processing/crawl_fb_comment_by_bright_data/facebook_car_comments_clean.csv")
    output_dir = os.path.join(CURRENT_DIR, "data")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Không tìm thấy file: {input_csv}")
        
    print(f"1. Đang đọc dữ liệu từ: {input_csv}")
    df = pd.read_csv(input_csv)
    total_raw = len(df)
    print(f"   -> Tổng số dòng trong file gốc: {total_raw}")
    
    # Làm sạch cơ bản: loại bỏ NaN hoặc chuỗi chỉ có khoảng trắng
    df = df.dropna(subset=["comment_text"]).copy()
    df["comment_text"] = df["comment_text"].astype(str).str.strip()
    df = df[df["comment_text"].str.len() > 0].reset_index(drop=True)
    valid_count = len(df)
    print(f"   -> Số bình luận có nội dung chữ hợp lệ: {valid_count} (loại bỏ {total_raw - valid_count} dòng rỗng)")
    
    # Khởi tạo model PhoBERT Pre-trained
    print("\n2. Đang khởi tạo mô hình PhoBERT Sentiment...")
    classifier = VietnameseSentimentClassifier()
    
    # Dự đoán theo batch với GPU
    print(f"\n3. Đang tiến hành phân loại cảm xúc cho {valid_count} bình luận (Batch size = 64)...")
    start_time = time.time()
    
    texts = df["comment_text"].tolist()
    results = classifier.predict_batch(texts, batch_size=64)
    elapsed_time = time.time() - start_time
    print(f"   -> Hoàn thành gán nhãn sau: {elapsed_time:.2f} giây ({valid_count / elapsed_time:.1f} comments/giây)")
    
    # Gán kết quả vào DataFrame
    df["sentiment"] = [r["sentiment"] for r in results]
    df["sentiment_label"] = [r["label"] for r in results]
    df["confidence_score"] = [r["confidence"] for r in results]
    df["score_pos"] = [r["detail_scores"]["POS"] for r in results]
    df["score_neg"] = [r["detail_scores"]["NEG"] for r in results]
    df["score_neu"] = [r["detail_scores"]["NEU"] for r in results]
    
    # Lưu toàn bộ dữ liệu đã được gán nhãn
    full_labeled_path = os.path.join(output_dir, "facebook_car_comments_labeled.csv")
    df.to_csv(full_labeled_path, index=False, encoding="utf-8-sig")
    print(f"\n4. Đã lưu toàn bộ dữ liệu có nhãn vào:")
    print(f"   -> {full_labeled_path}")
    
    # 5. Phân chia Train 80% và Validation/Test 20%
    print("\n5. Đang phân chia tập Train (80%) và Validation (20%)...")
    # Stratified split theo sentiment để đảm bảo tỷ lệ các lớp đồng đều giữa 2 tập
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=df["sentiment"]
    )
    
    train_path = os.path.join(output_dir, "train_comments.csv")
    val_path = os.path.join(output_dir, "val_comments.csv")
    
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    
    print(f"   -> Tập Train (80%): {len(train_df)} dòng -> Đã lưu: {train_path}")
    print(f"   -> Tập Validation (20%): {len(val_df)} dòng -> Đã lưu: {val_path}")
    
    # Thống kê phân phối cảm xúc
    print("\n" + "=" * 60)
    print("THỐNG KÊ PHÂN PHỐI CẢM XÚC TRÊN TOÀN BỘ BỘ DỮ LIỆU:")
    print("=" * 60)
    dist = df["sentiment_label"].value_counts()
    dist_pct = df["sentiment_label"].value_counts(normalize=True) * 100
    avg_conf = df["confidence_score"].mean()
    
    for label in dist.index:
        print(f"- {label:<12}: {dist[label]:>5} bình luận ({dist_pct[label]:>6.2f}%)")
    print(f"- Điểm tin cậy trung bình (Avg Confidence): {avg_conf:.4f}")
    
    print("\nPhân phối trong tập Train:")
    for label, count in train_df["sentiment_label"].value_counts().items():
        print(f"  + {label:<12}: {count:>5} ({count/len(train_df)*100:.2f}%)")
        
    print("\nPhân phối trong tập Validation:")
    for label, count in val_df["sentiment_label"].value_counts().items():
        print(f"  + {label:<12}: {count:>5} ({count/len(val_df)*100:.2f}%)")
        
    print("\nHOÀN TẤT THÀNH CÔNG!")

if __name__ == "__main__":
    main()
