import re
import html
import unicodedata
from typing import Dict, Any, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_MAP = {
    "NEG": "TIÊU CỰC",
    "POS": "TÍCH CỰC",
    "NEU": "TRUNG TÍNH"
}

ID2LABEL = {
    0: "NEG",
    1: "POS",
    2: "NEU"
}

PHONE_PATTERN = re.compile(r"(\+?84|0)(3|5|7|8|9)\d{8}\b|(\+?84|0)[.\s]?(3|5|7|8|9)\d{2}[.\s]?\d{3}[.\s]?\d{3}\b")
SALES_KEYWORDS = ["ib e", "ib em", "inbox em", "lh e", "lh em", "liên hệ e", "báo giá", "lăn bánh", "trả góp"]

class VietnameseSentimentClassifier:
    """
    Bộ phân loại cảm xúc văn bản tiếng Việt sử dụng PhoBERT Pre-trained Model
    (wonrax/phobert-base-vietnamese-sentiment).
    Tối ưu hóa:
    - Tiền xử lý chặt chẽ: loại bỏ các ký tự đặc biệt vô nghĩa, xử lý dấu chấm 'chấm hóng' an toàn.
    - Vectorized Tensor Operations: Tính toán trực tiếp trên Tensor không qua chuyển đổi trung gian dư thừa.
    - Defensive Edge Cases: Xử lý chuỗi rỗng, văn bản quá dài, ký tự HTML, regex spam.
    """
    def __init__(self, model_name: str = "wonrax/phobert-base-vietnamese-sentiment", device: Optional[str] = None):
        self.model_name = model_name
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading sentiment model '{self.model_name}' on device: {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

    def preprocess_text(self, text: str, max_chars: int = 2000) -> Dict[str, Any]:
        """
        Tiền xử lý và chuẩn hóa văn bản đầu vào:
        - Chuẩn hóa Unicode (NFC)
        - Giải mã HTML entities
        - Loại bỏ link URLs
        - Làm sạch ký tự rác lặp lại, ký tự đặc biệt gây nhiễu
        - Chặn văn bản chỉ chứa dấu chấm/dấu câu không có nội dung chữ
        - Giới hạn độ dài văn bản an toàn tránh tràn bộ nhớ
        """
        if text is None:
            return {"cleaned_text": "", "is_empty": True, "truncated": False, "original_len": 0, "is_sales_spam": False}
            
        original_len = len(text)
        # Chuẩn hóa Unicode
        text = unicodedata.normalize("NFC", str(text))
        # Giải mã ký tự HTML (vd: &amp; -> &, &quot; -> ")
        text = html.unescape(text)
        
        # Loại bỏ URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        
        # Loại bỏ các ký tự vô nghĩa lặp lại quá 3 lần (vd: !!!!! -> !!!, aaaaa -> aa)
        text = re.sub(r"([!?.])\1{2,}", r"\1\1", text)
        
        # Xóa các chuỗi ký tự rác dạng @@@###$$$%%%^^^
        text = re.sub(r"[@#$%^&*~`|\\<>{}\[\]_=+]{2,}", " ", text)
        
        # Loại bỏ ký tự điều khiển không hiển thị được (control characters ngoại trừ tab/newline)
        text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C"))
        
        # Thu gọn khoảng trắng
        text = re.sub(r"\s+", " ", text).strip()
        
        # Kiểm tra nếu sau khi làm sạch chuỗi rỗng
        if not text:
            return {"cleaned_text": "", "is_empty": True, "truncated": False, "original_len": original_len, "is_sales_spam": False}

        # Kiểm tra văn bản chỉ chứa toàn dấu câu / dấu chấm không có chữ cái/chữ số nào
        letters_and_digits = re.sub(r"[^\w]", "", text)
        if not letters_and_digits.strip():
            return {"cleaned_text": "", "is_empty": True, "truncated": False, "original_len": original_len, "is_sales_spam": False}
            
        # Kiểm tra nếu là bình luận rao vặt / spam số điện thoại
        clean_lower = text.lower()
        has_phone = bool(PHONE_PATTERN.search(text))
        has_sales_kw = any(kw in clean_lower for kw in SALES_KEYWORDS)
        is_sales_spam = has_phone and has_sales_kw

        truncated = False
        if len(text) > max_chars:
            text = text[:max_chars]
            truncated = True
            
        return {
            "cleaned_text": text,
            "is_empty": False,
            "truncated": truncated,
            "original_len": original_len,
            "is_sales_spam": is_sales_spam
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Dự đoán cảm xúc của một đoạn văn bản.
        Trả về:
        - label: 'TIÊU CỰC' | 'TÍCH CỰC' | 'TRUNG TÍNH'
        - sentiment: 'NEG' | 'POS' | 'NEU'
        - confidence: float (độ tin cậy từ 0 đến 1)
        - detail_scores: dict xác suất của từng lớp
        - warning: cảnh báo nếu văn bản bị cắt ngắn hoặc có dấu hiệu spam rao vặt
        """
        prep = self.preprocess_text(text)
        
        if prep["is_empty"]:
            raise ValueError("Văn bản đầu vào không được để trống hoặc chỉ chứa dấu chấm/ký tự đặc biệt không có chữ.")
            
        cleaned_text = prep["cleaned_text"]
        
        # Tokenize an toàn với max_length=256 (chuẩn của PhoBERT)
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=False
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs_tensor = torch.softmax(logits, dim=-1)[0]
            pred_idx = int(torch.argmax(logits, dim=-1).item())
            probs = probs_tensor.cpu().tolist()
            
        sentiment_code = ID2LABEL[pred_idx]
        vietnamese_label = LABEL_MAP[sentiment_code]
        confidence = float(probs[pred_idx])
        
        # Xử lý heuristic nếu là bình luận spam rao vặt số điện thoại
        warning = None
        if prep["is_sales_spam"]:
            sentiment_code = "NEU"
            vietnamese_label = "TRUNG TÍNH"
            confidence = 0.90
            probs = [0.05, 0.05, 0.90]
            warning = "Nhận diện đây là bình luận rao vặt / spam số điện thoại tư vấn bán hàng."

        detail_scores = {
            "POS": round(float(probs[1]), 4),
            "NEG": round(float(probs[0]), 4),
            "NEU": round(float(probs[2]), 4)
        }
        
        result = {
            "text": text,
            "cleaned_text": cleaned_text,
            "label": vietnamese_label,
            "sentiment": sentiment_code,
            "confidence": round(confidence, 4),
            "detail_scores": detail_scores
        }
        
        if prep["truncated"]:
            result["warning"] = f"Văn bản gốc dài {prep['original_len']} ký tự, đã được cắt ngắn an toàn về {len(cleaned_text)} ký tự."
        elif warning:
            result["warning"] = warning
            
        return result

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Dự đoán cảm xúc cho một danh sách văn bản theo batch với tính toán vectorized tối ưu.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch_raw = texts[i : i + batch_size]
            batch_prep = [self.preprocess_text(t) for t in batch_raw]
            
            # Phân tách các phần tử hợp lệ và rỗng
            valid_indices = [idx for idx, p in enumerate(batch_prep) if not p["is_empty"]]
            
            batch_res = [None] * len(batch_raw)
            for idx, p in enumerate(batch_prep):
                if p["is_empty"]:
                    batch_res[idx] = {
                        "text": batch_raw[idx],
                        "cleaned_text": "",
                        "label": "TRUNG TÍNH",
                        "sentiment": "NEU",
                        "confidence": 1.0,
                        "detail_scores": {"POS": 0.0, "NEG": 0.0, "NEU": 1.0},
                        "warning": "Văn bản không có nội dung chữ hợp lệ (dấu chấm hoặc ký tự rác)."
                    }
                    
            if valid_indices:
                valid_texts = [batch_prep[idx]["cleaned_text"] for idx in valid_indices]
                inputs = self.tokenizer(
                    valid_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs_tensor = torch.softmax(logits, dim=-1)
                    pred_indices = torch.argmax(logits, dim=-1).cpu().tolist()
                    all_probs = probs_tensor.cpu().tolist()
                    
                for sub_idx, idx in enumerate(valid_indices):
                    pred_idx = pred_indices[sub_idx]
                    probs = all_probs[sub_idx]
                    sentiment_code = ID2LABEL[pred_idx]
                    vietnamese_label = LABEL_MAP[sentiment_code]
                    confidence = float(probs[pred_idx])
                    warning = None

                    # Kiểm tra spam rao vặt
                    if batch_prep[idx]["is_sales_spam"]:
                        sentiment_code = "NEU"
                        vietnamese_label = "TRUNG TÍNH"
                        confidence = 0.90
                        probs = [0.05, 0.05, 0.90]
                        warning = "Nhận diện đây là bình luận rao vặt / liên hệ bán xe."

                    item = {
                        "text": batch_raw[idx],
                        "cleaned_text": valid_texts[sub_idx],
                        "label": vietnamese_label,
                        "sentiment": sentiment_code,
                        "confidence": round(confidence, 4),
                        "detail_scores": {
                            "POS": round(float(probs[1]), 4),
                            "NEG": round(float(probs[0]), 4),
                            "NEU": round(float(probs[2]), 4)
                        }
                    }
                    if batch_prep[idx]["truncated"]:
                        item["warning"] = f"Văn bản quá dài ({batch_prep[idx]['original_len']} ký tự), đã cắt ngắn."
                    elif warning:
                        item["warning"] = warning
                    batch_res[idx] = item
                    
            results.extend(batch_res)
            
        return results
