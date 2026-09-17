# -*- coding: utf-8 -*-
"""
Script Huấn luyện / Fine-tune Mô hình PhoBERT cho Phân loại Cảm xúc Xe Hơi Tiếng Việt
(Fine-tuning PhoBERT for Vietnamese Car Comments Sentiment Analysis)

Tính năng nổi bật:
- Pre-tokenization siêu tốc: Tokenize hàng loạt bằng Rust/Fast Tokenizer giúp nạp batch tức thì.
- Hỗ trợ Weighted Cross-Entropy Loss (tăng trọng số lớp Tiêu cực để tối ưu Negative Recall, chống bỏ sót rủi ro truyền thông).
- Tự động phát hiện GPU CUDA và kích hoạt Mixed Precision tăng tốc độ huấn luyện.
- Tính toán đầy đủ hệ thống chỉ số: Accuracy, Macro F1-Score, Negative Recall, Confusion Matrix.
- Cơ chế Checkpointing: Tự động lưu mô hình tốt nhất vào thư mục `checkpoints/`.
- Hỗ trợ cờ `--demo` để kiểm thử nhanh pipeline huấn luyện trong vài giây.
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DEFAULT_MODEL_NAME = "wonrax/phobert-base-vietnamese-sentiment"
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, "checkpoints")

LABEL2ID = {"NEG": 0, "POS": 1, "NEU": 2}
ID2LABEL = {0: "NEG", 1: "POS", 2: "NEU"}
LABEL_NAMES = ["TIÊU CỰC (NEG)", "TÍCH CỰC (POS)", "TRUNG TÍNH (NEU)"]

class FastCarCommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        # Pre-tokenize toàn bộ danh sách một lần duy nhất bằng C++/Rust Tokenizer
        encodings = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt"
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

def compute_class_weights(labels, boost_neg=1.4):
    """
    Tính trọng số nghịch đảo tần suất lớp kết hợp nhân thêm boost_neg cho lớp TIÊU CỰC
    nhằm tối đa hóa chỉ số sống còn Negative Recall.
    """
    counts = np.bincount(labels, minlength=3)
    total = len(labels)
    weights = total / (3.0 * np.maximum(counts, 1))
    weights[0] *= boost_neg  # Tăng trọng số phạt khi bỏ sót Tiêu Cực
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float)

def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)

    # Chỉ số riêng cho lớp Tiêu Cực (Index 0)
    neg_recall = recall_score(all_labels, all_preds, labels=[0], average=None, zero_division=0)[0]
    neg_precision = precision_score(all_labels, all_preds, labels=[0], average=None, zero_division=0)[0]
    neg_f1 = f1_score(all_labels, all_preds, labels=[0], average=None, zero_division=0)[0]

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

    metrics = {
        "val_loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "macro_precision": macro_prec,
        "neg_recall": neg_recall,
        "neg_precision": neg_precision,
        "neg_f1": neg_f1,
        "confusion_matrix": cm,
        "preds": all_preds,
        "labels": all_labels
    }
    return metrics

def train(args):
    print("=" * 70)
    print("BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN / FINE-TUNE PHOBERT CHO BÌNH LUẬN XE HƠI")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[*] Thiết bị tính toán: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    # 1. Đọc dữ liệu Train và Test
    train_file = os.path.join(DATA_DIR, "train.csv")
    test_file = os.path.join(DATA_DIR, "test.csv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"Không tìm thấy file train/test trong: {DATA_DIR}. Hãy chạy split_train_test.py trước!")

    print(f"[*] Nạp dữ liệu:")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    train_df = train_df.dropna(subset=["comment_text", "sentiment"]).copy()
    test_df = test_df.dropna(subset=["comment_text", "sentiment"]).copy()

    if args.demo:
        print("[!] Chế độ --demo kích hoạt: Giới hạn 200 mẫu Train, 50 mẫu Test để kiểm thử nhanh pipeline!")
        train_df = train_df.head(200)
        test_df = test_df.head(50)

    print(f"    - Tập Huấn luyện (Train): {len(train_df)} mẫu")
    print(f"    - Tập Kiểm thử (Test):    {len(test_df)} mẫu")

    train_texts = train_df["comment_text"].astype(str).tolist()
    train_labels = [LABEL2ID[s] for s in train_df["sentiment"]]

    test_texts = test_df["comment_text"].astype(str).tolist()
    test_labels = [LABEL2ID[s] for s in test_df["sentiment"]]

    # 2. Tokenizer & Dataset
    print(f"\n[*] Đang tải Tokenizer & Pretrained Model: '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
    model.to(device)

    print("[*] Đang pre-tokenize dữ liệu siêu tốc...")
    train_dataset = FastCarCommentDataset(train_texts, train_labels, tokenizer, max_len=args.max_len)
    test_dataset = FastCarCommentDataset(test_texts, test_labels, tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Weighted Loss Function
    class_weights = compute_class_weights(train_labels, boost_neg=args.boost_neg).to(device)
    print(f"[*] Trọng số Loss các lớp [NEG, POS, NEU]: {class_weights.cpu().numpy().round(3)}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # 5. Vòng lặp huấn luyện
    best_macro_f1 = 0.0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("\n" + "-" * 70)
    print("TIẾN TRÌNH HUẤN LUYỆN:")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        epoch_time = time.time() - start_time

        # Đánh giá sau mỗi epoch trên tập Test độc lập
        metrics = evaluate(model, test_loader, device, loss_fn)

        print(f"\n[Epoch {epoch}/{args.epochs}] ({epoch_time:.1f}s) - Train Loss: {avg_train_loss:.4f} | Val Loss: {metrics['val_loss']:.4f}")
        print(f"   Accuracy        : {metrics['accuracy'] * 100:.2f}%")
        print(f"   Macro F1-Score  : {metrics['macro_f1'] * 100:.2f}%")
        print(f"   Negative Recall : {metrics['neg_recall'] * 100:.2f}% (Chỉ số sống còn ngành ô tô)")
        print(f"   Negative F1     : {metrics['neg_f1'] * 100:.2f}%")

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            save_path = os.path.join(CHECKPOINT_DIR, "best_phobert_car_sentiment")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"   -> [BEST] Đã lưu checkpoint xuất sắc nhất vào: {save_path}")

    # Báo cáo tổng kết ma trận nhầm lẫn (Confusion Matrix)
    print("\n" + "=" * 70)
    print("BÁO CÁO ĐÁNH GIÁ CHI TIẾT TRÊN TẬP TEST:")
    print("=" * 70)
    final_metrics = evaluate(model, test_loader, device, loss_fn)
    print(classification_report(
        final_metrics["labels"],
        final_metrics["preds"],
        target_names=LABEL_NAMES,
        digits=4
    ))

    print("MA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
    print("Dòng = Nhãn thực tế, Cột = Nhãn dự đoán")
    print("                 Pred_NEG   Pred_POS   Pred_NEU")
    cm = final_metrics["confusion_matrix"]
    print(f"Actual_NEG:     {cm[0][0]:>8}   {cm[0][1]:>8}   {cm[0][2]:>8}")
    print(f"Actual_POS:     {cm[1][0]:>8}   {cm[1][1]:>8}   {cm[1][2]:>8}")
    print(f"Actual_NEU:     {cm[2][0]:>8}   {cm[2][1]:>8}   {cm[2][2]:>8}")
    print("=" * 70)
    print(f"[✔] HOÀN TẤT HUẤN LUYỆN! Best Macro F1-Score: {best_macro_f1 * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện PhoBERT phân loại cảm xúc xe hơi tiếng Việt")
    parser.add_argument("--epochs", type=int, default=2, help="Số lượng epoch (mặc định: 2)")
    parser.add_argument("--batch-size", type=int, default=16, help="Kích thước batch (mặc định: 16)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (mặc định: 2e-5)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay L2 (mặc định: 0.01)")
    parser.add_argument("--max-len", type=int, default=128, help="Độ dài token tối đa (mặc định: 128)")
    parser.add_argument("--boost-neg", type=float, default=1.4, help="Hệ số phạt tăng cường cho lớp Tiêu Cực (mặc định: 1.4)")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Pre-trained HuggingFace Model")
    parser.add_argument("--demo", action="store_true", help="Chạy chế độ demo nhanh trên tập mẫu nhỏ để kiểm tra pipeline")
    parser.add_argument("--cpu", action="store_true", help="Ép buộc chạy trên CPU")
    args = parser.parse_args()

    train(args)
