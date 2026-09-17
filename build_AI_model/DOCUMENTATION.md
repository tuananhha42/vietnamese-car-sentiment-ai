# BÁO CÁO KỸ THUẬT & HƯỚNG DẪN MICROSERVICE PHÂN LOẠI CẢM XÚC TIẾNG VIỆT

Dự án: **Phân tích Cảm xúc Bình luận Ô tô trên Mạng xã hội (Vietnamese Car Comments Sentiment Analysis)**  
Mã nguồn & Dữ liệu: Thư mục `build_AI_model/`

---

## 1. LỰA CHỌN MÔ HÌNH & PHƯƠNG PHÁP (MODEL SELECTION)

### 1.1. Mô hình được lựa chọn
- **Tên mô hình**: `wonrax/phobert-base-vietnamese-sentiment`
- **Kiến trúc nền tảng**: **PhoBERT-base** (RoBERTa architecture tinh chỉnh riêng cho tiếng Việt bởi VinAI Research).
- **Đầu ra phân loại**: 3 lớp cảm xúc:
  - `POS` (`TÍCH CỰC`): Khen ngợi kiểu dáng, động cơ, độ đầm chắc, tiết kiệm nhiên liệu, dịch vụ tốt...
  - `NEG` (`TIÊU CỰC`): Phàn nàn về lỗi pin, bảo hành chậm trễ, giá đắt, mùi nhựa, ồn ào...
  - `NEU` (`TRUNG TÍNH`): Bình luận hỏi giá, hỏi thông số kỹ thuật, thủ tục trả góp, quan sát trung lập...

### 1.2. Lý do lựa chọn mô hình PhoBERT Pre-trained
1. **Khả năng hiểu ngữ cảnh và từ ngữ tiếng Việt vượt trội**:
   - Khác với phương pháp truyền thống (TF-IDF, Naive Bayes, SVM) chỉ nhìn vào tần suất từ đơn lẻ (Bag-of-words), PhoBERT sử dụng cơ chế Self-Attention đa tầng (Transformer) để nắm bắt quan hệ từ xa trong câu.
   - Xử lý chính xác các trường hợp **phủ định và đảo ngữ tiếng Việt** (ví dụ: *"xe tưởng không ngon mà ngon không tưởng"*, *"chả ra cái tích sự gì"*).
   - Hiểu tốt **tiếng lóng (slang), từ ngữ viết tắt trên mạng xã hội Facebook** ô tô Việt Nam (*"vkl"*, *"đầm"*, *"bốc"*, *"ọp ẹp"*, *"ăn xăng"*, *"lăn bánh"*).
2. **Tối ưu chi phí và bảo mật dữ liệu**:
   - Chạy trực tiếp cục bộ (Local Inference) trên GPU (GTX 1650) hoặc CPU, không phụ thuộc vào Internet hay tốn chi phí token của các dịch vụ bên ngoài như OpenAI/Anthropic.
   - Tốc độ suy luận cực nhanh: chỉ **~15 - 20ms cho mỗi request**, phục vụ tốt kiến trúc Microservice thời gian thực.
3. **Giải pháp Auto-labeling (Pseudo-labeling) cho tập dữ liệu Unlabeled ban đầu**:
   - Dữ liệu cào về từ Facebook ban đầu hoàn toàn chưa có nhãn. Thay vì tốn hàng chục giờ gán nhãn thủ công gây sai sót chủ quan, mô hình PhoBERT được áp dụng để tự động gán nhãn cho toàn bộ 6.296 dòng với độ tự tin trung bình đạt **87.48%**.

---

## 2. PHÂN CHIA DỮ LIỆU TRAIN / TEST (DATA SPLIT)

Toàn bộ **6.296 bình luận** sau khi được phân loại đã được chia thành 2 tệp lưu trữ tại `build_AI_model/data/`:

| Tệp | Số lượng dòng | Tỷ lệ | Mô tả |
|---|---|---|---|
| **`train.csv`** | **5.036 dòng** | **80%** | Dùng cho huấn luyện, tinh chỉnh hoặc tham chiếu |
| **`test.csv`** | **1.260 dòng** | **20%** | Dùng độc lập để kiểm thử và đánh giá hiệu năng |

### 2.1. Cấu trúc các cột trong file `train.csv` và `test.csv`
- `comment_text`: Nội dung bình luận tiếng Việt đã được làm sạch cơ bản.
- `sentiment_label`: Nhãn cảm xúc hiển thị (`TÍCH CỰC`, `TIÊU CỰC`, `TRUNG TÍNH`).
- `sentiment`: Mã định danh (`POS`, `NEG`, `NEU`).
- `confidence_score`: Độ tự tin của dự đoán (giá trị từ `0.0` đến `1.0`).
- `url`: Đường dẫn trực tiếp đến bình luận Facebook.
- `post_url`: Đường dẫn đến bài đăng gốc trên nhóm/fanpage.

### 2.2. Phương pháp phân chia: Stratified Split
Dữ liệu được chia bằng kỹ thuật **Stratified Sampling** (`random_state=42`) để đảm bảo tỷ lệ phân phối cảm xúc giữa tập Train và tập Test là tương đồng tuyệt đối:

| Nhãn cảm xúc | Tỷ lệ toàn bộ | Tỷ lệ tập Train | Tỷ lệ tập Test |
|---|---|---|---|
| **TÍCH CỰC** | 42.84% (2.697 câu) | 42.83% (2.157 câu) | 42.86% (540 câu) |
| **TRUNG TÍNH** | 30.99% (1.951 câu) | 31.00% (1.561 câu) | 30.95% (390 câu) |
| **TIÊU CỰC** | 26.18% (1.648 câu) | 26.17% (1.318 câu) | 26.19% (330 câu) |

---

## 3. CÁC CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH (EVALUATION METRICS)

Trong bài toán Phân tích cảm xúc dữ liệu xã hội (Social Listening), việc lựa chọn chỉ số đánh giá cần dựa trên đặc thù kinh doanh thực tế:

### 3.1. Tại sao KHÔNG CHỈ dùng Accuracy (Độ chính xác tổng quát)?
- **Hạn chế**: Dữ liệu mạng xã hội luôn có hiện tượng mất cân bằng lớp (Imbalanced Data). Nếu một mô hình chỉ dự đoán vào lớp chiếm số đông (Tích cực hoặc Trung tính) thì Accuracy vẫn có thể đạt mức cao, nhưng hoàn toàn thất bại trong việc phát hiện các phàn nàn tiêu cực.

### 3.2. Các chỉ số được lựa chọn & Lý do:
1. **Macro F1-Score**:
   - *Lý do*: Macro F1 tính trung bình điều hòa F1-Score của từng lớp độc lập nhau mà không phụ thuộc vào số lượng mẫu của lớp đó. Chỉ số này bắt buộc mô hình phải dự đoán chuẩn xác đồng đều trên cả 3 lớp: Tích cực, Tiêu cực và Trung tính.
2. **Recall cho nhãn Tiêu cực (Negative Recall)**:
   - *Lý do*: Trong ngành công nghiệp ô tô và xe điện, **một phản ánh tiêu cực bị bỏ sót (False Negative)** về lỗi pin, cháy nổ, kẹt chân ga hoặc sự cố bảo hành có thể dẫn đến **khủng hoảng truyền thông nghiêm trọng (PR Crisis)** cho doanh nghiệp. Do đó, chỉ số Recall của lớp Tiêu cực là chỉ số sống còn.
3. **Precision cho nhãn Tiêu cực & Tích cực**:
   - *Lý do*: Đảm bảo hạn chế tối đa việc báo động giả (False Alarm) — tránh việc một bình luận hỏi thông tin thông thường bị nhận diện nhầm thành khiếu nại tiêu cực, làm sai lệch báo cáo khảo sát thị trường của ban quản lý.

### 3.3. Quy trình Huấn luyện & Fine-tune Mô hình (`train_model.py`)
Để thích ứng sâu với miền dữ liệu xe hơi, hệ thống cung cấp script huấn luyện chính thức với các kỹ thuật MLOps nâng cao:
- **Pre-tokenization siêu tốc**: Sử dụng Fast Tokenizer của Hugging Face xử lý hàng loạt văn bản trước khi nạp vào DataLoader.
- **Weighted Cross-Entropy Loss**: Tự động tính toán trọng số nghịch đảo tần suất lớp, đồng thời nhân hệ số phạt tăng cường (`boost_neg=1.4`) cho lớp `TIÊU CỰC` (Index 0). Điều này buộc gradient loss phải phạt nặng nếu mô hình bỏ sót khiếu nại của khách hàng.
- **Mixed Precision & Checkpointing**: Tự động lưu checkpoint tốt nhất (`best_phobert_car_sentiment`) khi chỉ số Macro F1-Score và Negative Recall đạt đỉnh.
- **Câu lệnh khởi chạy**:
  ```bash
  python3 build_AI_model/train_model.py --epochs 2 --batch-size 16 --boost-neg 1.4
  ```

---

## 4. MICROSERVICE REST API

Microservice được đóng gói hoàn chỉnh bằng **FastAPI** và **Uvicorn**, tối ưu hóa sẵn cơ chế Singleton Model và tăng tốc GPU.

### 4.1. Khởi chạy Server
Từ thư mục dự án, chạy lệnh:
```bash
python3 build_AI_model/api.py
```
Server sẽ khởi chạy tại: `http://localhost:8000`  
Tài liệu tương tác Swagger UI tự động: `http://localhost:8000/docs`

### 4.2. Đặc tả Endpoint `POST /predict`

#### Request Body (JSON):
```json
{
  "text": "Xe này chạy đầm chắc, cách âm rất tốt và tiết kiệm điện"
}
```

#### Response thành công (HTTP 200 OK):
```json
{
  "text": "Xe này chạy đầm chắc, cách âm rất tốt và tiết kiệm điện",
  "cleaned_text": "Xe này chạy đầm chắc, cách âm rất tốt và tiết kiệm điện",
  "label": "TÍCH CỰC",
  "sentiment": "POS",
  "confidence": 0.9928,
  "detail_scores": {
    "POS": 0.9928,
    "NEG": 0.0016,
    "NEU": 0.0056
  },
  "warning": null,
  "process_time_ms": 14.2
}
```

### 4.3. Xử lý các trường hợp đầu vào bất thường (Edge Cases)
Mô hình và API được thiết kế cơ chế phòng thủ (Defensive Programming) để xử lý triệt để:

1. **Chuỗi rỗng hoặc chỉ có khoảng trắng (`""`, `"   \t\n"`):**
   - API chặn ngay từ tầng tiền xử lý và trả về mã lỗi **`400 Bad Request`**:
   ```json
   {
     "detail": "Văn bản đầu vào không được để trống hoặc chỉ chứa khoảng trắng."
   }
   ```
2. **Văn bản quá dài (> 2.000 ký tự):**
   - Tự động cắt ngắn (truncate) về độ dài an toàn và giới hạn tối đa 256 tokens của PhoBERT, tránh lỗi tràn bộ nhớ (OOM) hoặc làm sập server.
   - Kèm thông báo `warning` trong kết quả trả về:
   ```json
   "warning": "Văn bản gốc dài 3500 ký tự, đã được cắt ngắn an toàn về 2000 ký tự."
   ```
3. **Ký tự đặc biệt, Emojis, URLs, HTML tags:**
   - Sử dụng regex loại bỏ các chuỗi rác lặp lại vô nghĩa (`@@@###$$$`), bóc tách mã HTML, loại bỏ URL, chuẩn hóa Unicode NFC chuẩn tiếng Việt trước khi đưa vào mô hình, giúp kết quả dự đoán luôn ổn định và chính xác.

### 4.4. Ví dụ gọi API bằng cURL

```bash
# Dự đoán 1 câu tích cực
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Xe đi rất êm và tiết kiệm điện"}'

# Dự đoán 1 câu tiêu cực
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Xe quá tệ, hay hỏng vặt đại lý bảo hành thiếu trách nhiệm"}'
```

### 4.5. Ví dụ gọi API bằng Python

```python
import requests

url = "http://localhost:8000/predict"
payload = {"text": "Vf3 đi thích thật sự, luồn lách phố xá cực tiện"}

response = requests.post(url, json=payload)
print(response.json())
```

### 4.6. Kiểm thử toàn diện hệ thống (`test_100_comments.py`)
Hệ thống sử dụng script `test_100_comments.py` làm công cụ kiểm thử toàn diện:
- **Chế độ kiểm thử kép (Dual-Mode)**:
  - `--mode local`: Kiểm thử trực tiếp qua PhoBERT engine cục bộ.
  - `--mode api`: Kiểm thử End-to-End thông qua REST API Microservice (đo độ trễ mạng thực tế, kiểm tra mã HTTP và schema JSON).
- **Bộ kiểm thử trường hợp biên (Edge Cases Battery)**: Tự động chạy qua các mẫu hiểm hóc: dấu chấm đơn lẻ `.`, emoij & ký tự lạ `@@@###$$$`, tin nhắn spam số điện thoại cò mồi bán xe, câu đảo ngữ phủ định phức tạp.
- **Báo cáo trực quan**: Xuất bảng kết quả chi tiết bằng thư viện `tabulate` và lưu toàn bộ kết quả kiểm thử vào file `build_AI_model/data/test_100_results.csv`.
- **Câu lệnh kiểm thử**:
  ```bash
  # Kiểm thử nhanh trên mô hình cục bộ:
  python3 build_AI_model/test_100_comments.py --mode local --limit 100

  # Kiểm thử qua Microservice REST API:
  python3 build_AI_model/test_100_comments.py --mode api --limit 100
  ```
