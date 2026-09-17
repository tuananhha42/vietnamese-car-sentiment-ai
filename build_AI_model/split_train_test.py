import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    labeled_csv = os.path.join(data_dir, "facebook_car_comments_labeled.csv")
    
    if not os.path.exists(labeled_csv):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu đã phân loại: {labeled_csv}")
        
    print(f"1. Đọc dữ liệu từ: {labeled_csv}")
    df = pd.read_csv(labeled_csv)
    print(f"   -> Tổng số dòng: {len(df)}")
    
    # Chỉ giữ lại các cột theo đúng yêu cầu:
    selected_cols = ["comment_text", "sentiment_label", "sentiment", "confidence_score", "url", "post_url"]
    
    # Kiểm tra cột tồn tại
    for col in selected_cols:
        if col not in df.columns:
            raise ValueError(f"Cột '{col}' không tồn tại trong file!")
            
    df_clean = df[selected_cols].copy()
    
    # Phân chia 80% Train và 20% Test theo phương pháp Stratified (giữ nguyên tỷ lệ các lớp)
    train_df, test_df = train_test_split(
        df_clean,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=df_clean["sentiment"]
    )
    
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
    
    print("\n2. HOÀN THÀNH PHÂN CHIA DỮ LIỆU:")
    print(f"   -> Tệp Train: {train_path} ({len(train_df)} dòng - 80%)")
    print(f"   -> Tệp Test : {test_path} ({len(test_df)} dòng - 20%)")
    print(f"   -> Các cột được lưu: {selected_cols}")
    
    print("\n3. Thống kê tập Train (80%):")
    for label, count in train_df["sentiment_label"].value_counts().items():
        print(f"   + {label:<12}: {count:>5} ({count/len(train_df)*100:.2f}%)")
        
    print("\n4. Thống kê tập Test (20%):")
    for label, count in test_df["sentiment_label"].value_counts().items():
        print(f"   + {label:<12}: {count:>5} ({count/len(test_df)*100:.2f}%)")

if __name__ == "__main__":
    split_data()
