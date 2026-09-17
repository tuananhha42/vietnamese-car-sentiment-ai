import os
import json
import re
import hashlib
from typing import Dict, Any, List, Optional
import requests
from dotenv import load_dotenv

# Tự động nạp cấu hình từ .env nếu có
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_PROJECT_DIR, ".env"))

class GeminiSuggester:
    """
    Module phân tích chuyên sâu bình luận bằng Gemini Flash Lite với cơ chế:
    - Multi-Key Auto-Rotation: Luân chuyển tự động 23 keys khi gặp Rate Limit/Quota Exceeded.
    - Fast Memory Cache: Cache kết quả bằng SHA256 hash của văn bản để phản hồi tức thì (<1ms) và tiết kiệm quota.
    - Flexible Config: Ưu tiên đọc biến môi trường (.env) trước khi đọc từ file.
    """
    def __init__(self, key_file_path: Optional[str] = None, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        if key_file_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            key_file_path = os.path.join(base_dir, "gemini_api_key")
            
        self.key_file_path = key_file_path
        self.keys = self._load_keys()
        self.current_key_idx = 0
        self._cache: Dict[str, Dict[str, Any]] = {}
        print(f"GeminiSuggester: Đã nạp {len(self.keys)} API keys. Model: '{self.model_name}'")

    def _load_keys(self) -> List[str]:
        keys = []
        # 1. Ưu tiên đọc từ biến môi trường GEMINI_API_KEYS
        env_keys = os.getenv("GEMINI_API_KEYS", "")
        if env_keys:
            keys = [k.strip() for k in env_keys.split(",") if k.strip()]

        # 2. Nếu chưa có trong env, đọc từ file key_file_path
        if not keys and os.path.exists(self.key_file_path):
            with open(self.key_file_path, "r", encoding="utf-8") as f:
                keys = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        if not keys:
            raise ValueError("Không tìm thấy Gemini API Key hợp lệ trong biến môi trường hoặc file gemini_api_key.")
        return keys

    def _get_next_key(self) -> str:
        return self.keys[self.current_key_idx]

    def _rotate_key(self, reason: str = ""):
        old_idx = self.current_key_idx
        self.current_key_idx = (self.current_key_idx + 1) % len(self.keys)
        print(f"[Gemini Key Rotation] Chuyển từ key #{old_idx + 1} sang key #{self.current_key_idx + 1}/{len(self.keys)}. Lý do: {reason}")

    def _get_cache_key(self, text: str, sentiment: str) -> str:
        raw = f"{text.strip()}:::{sentiment.strip()}:::{self.model_name}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def analyze_comment(self, comment_text: str, sentiment_label: str = "") -> Dict[str, Any]:
        """
        Gọi Gemini API để phân tích chuyên sâu:
        - Điểm khách hàng hài lòng (nếu có)
        - Điểm khách hàng chưa hài lòng / chê / thắc mắc
        - Đề xuất giải pháp hành động cho team Marketing & CSKH
        - Gợi ý câu phản hồi mẫu lịch sự, phù hợp tâm lý khách hàng
        """
        if not comment_text or not comment_text.strip():
            raise ValueError("Văn bản bình luận không được để trống.")

        # Kiểm tra Cache trước
        cache_key = self._get_cache_key(comment_text, sentiment_label)
        if cache_key in self._cache:
            cached_res = dict(self._cache[cache_key])
            cached_res["from_cache"] = True
            return cached_res

        prompt = f"""Bạn là chuyên gia tư vấn Marketing và Chăm sóc Khách hàng (CSKH) hàng đầu trong ngành ô tô tại Việt Nam.
Hãy phân tích bình luận sau đây của người dùng trên mạng xã hội Facebook về xe hơi:

Nội dung bình luận: "{comment_text}"
Nhãn cảm xúc đã nhận diện: {sentiment_label if sentiment_label else "Chưa xác định"}

Hãy phân tích thật sắc bén, thực tế và trả về định dạng JSON thuần túy theo đúng cấu trúc sau (không kèm ký tự markdown ```json):
{{
  "hai_long": [
    "liệt kê ngắn gọn các điểm khách hàng khen ngợi, hài lòng hoặc ấn tượng tốt (nếu không có thì trả về mảng rỗng [])"
  ],
  "chua_hai_long": [
    "liệt kê các điểm khách hàng chưa hài lòng, chê bai, lo ngại, hoặc thắc mắc cần giải đáp"
  ],
  "de_xuat_marketing": [
    "các đề xuất hành động cụ thể, khả thi cho team Marketing / Truyền thông / CSKH để xử lý vấn đề hoặc tận dụng điểm khen"
  ],
  "cau_phan_hoi_mau": "một câu trả lời mẫu hoàn chỉnh, lịch sự, thể hiện sự đồng cảm, chuyên nghiệp để Admin Fanpage có thể copy trả lời khách hàng ngay lập tức"
}}
"""

        endpoint_template = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "responseMimeType": "application/json"
            }
        }

        max_attempts = len(self.keys)
        attempts = 0
        last_error = None

        while attempts < max_attempts:
            current_key = self._get_next_key()
            url = f"{endpoint_template}?key={current_key}"
            attempts += 1

            try:
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=15
                )

                if response.status_code == 200:
                    res_json = response.json()
                    raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                    
                    # Làm sạch nếu dính markdown backticks
                    raw_text = re.sub(r"^```json\s*", "", raw_text)
                    raw_text = re.sub(r"^```\s*", "", raw_text)
                    raw_text = re.sub(r"\s*```$", "", raw_text)
                    
                    parsed_result = json.loads(raw_text)
                    parsed_result["key_used"] = f"Key #{self.current_key_idx + 1}"
                    parsed_result["model"] = self.model_name
                    parsed_result["from_cache"] = False
                    
                    # Lưu vào cache (giới hạn tối đa 500 phần tử)
                    if len(self._cache) > 500:
                        self._cache.pop(next(iter(self._cache)))
                    self._cache[cache_key] = parsed_result

                    return parsed_result

                elif response.status_code in [429, 403, 400]:
                    # Hết quota hoặc key không hợp lệ -> Xoay sang key tiếp theo
                    err_msg = f"HTTP {response.status_code}: {response.text[:100]}"
                    self._rotate_key(reason=err_msg)
                    last_error = err_msg
                    continue
                else:
                    err_msg = f"HTTP {response.status_code}: {response.text[:100]}"
                    self._rotate_key(reason=err_msg)
                    last_error = err_msg
                    continue

            except Exception as e:
                err_msg = f"Exception: {str(e)}"
                self._rotate_key(reason=err_msg)
                last_error = err_msg
                continue

        # Nếu đã xoay hết toàn bộ keys mà vẫn lỗi
        raise RuntimeError(f"Đã thử qua toàn bộ {len(self.keys)} API keys nhưng đều thất bại. Lỗi cuối cùng: {last_error}")

# Singleton instance
_suggester_instance = None

def get_gemini_suggester() -> GeminiSuggester:
    global _suggester_instance
    if _suggester_instance is None:
        _suggester_instance = GeminiSuggester()
    return _suggester_instance

if __name__ == "__main__":
    suggester = get_gemini_suggester()
    test_comment = "Vf3 đi chán ngắt _ mua kia moning lại ngon"
    print("Test 1 (API call):", test_comment)
    res1 = suggester.analyze_comment(test_comment, sentiment_label="TIÊU CỰC")
    print("From cache:", res1.get("from_cache"))
    print("\nTest 2 (Cache call):", test_comment)
    res2 = suggester.analyze_comment(test_comment, sentiment_label="TIÊU CỰC")
    print("From cache:", res2.get("from_cache"))
