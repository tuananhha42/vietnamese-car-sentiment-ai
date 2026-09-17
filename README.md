# DỰ ÁN PHÂN TÍCH & PHÂN LOẠI CẢM XÚC BÌNH LUẬN XE HƠI TIẾNG VIỆT
### (Vietnamese Car Comments Sentiment Analysis System)

> **Mô tả tổng quan**: Hệ thống hoàn chỉnh giải quyết trọn vẹn bài toán: từ cào dữ liệu Facebook về ô tô, tiền xử lý và gán nhãn tự động bằng AI, phân chia tập Train (80%) / Test (20%), xây dựng REST API Microservice chuẩn hóa và giao diện Web tương tác trực quan bằng Flask.

---

## 📑 MỤC LỤC
1. [Hướng Dẫn Setup Môi Trường (Setup Environment)](#-1-hướng-dẫn-setup-môi-trường-setup-environment)
2. [Cách Khởi Chạy Các Service (REST API & Web UI)](#-2-cách-khởi-chạy-các-service-rest-api--web-ui)
3. [Quy Trình Chạy Toàn Bộ Dự Án (Pipeline Execution)](#-3-quy-trình-chạy-toàn-bộ-dự-án-pipeline-execution)
4. [Giải Thích Chi Tiết Tác Dụng Từng Thư Mục & Từng File](#-4-giải-thích-chi-tiết-tác-dụng-từng-thư-mục--từng-file)
5. [Đặc Tả REST API Microservice & Xử Lý Lỗi Biên](#-5-đặc-tả-rest-api-microservice--xử-lý-lỗi-biên)
6. [Lựa Chọn Model & Tiêu Chí Đánh Giá Nghiệp Vụ](#-6-lựa-chọn-model--tiêu-chí-đánh-giá-nghiệp-vụ)
7. [Tính Năng AI Suggestion Chuyên Sâu (Gemini 3.1 Flash Lite)](#-7-tính-năng-ai-suggestion-chuyên-sâu-gemini-31-flash-lite)
8. [Tính Năng Phân Tích Hàng Loạt (Bulk Analyst - TXT / Excel / CSV)](#-8-tính-năng-phân-tích-hàng-loạt-bulk-analyst---txt--excel--csv)
9. [Kiến Trúc Giao Diện 3D Cockpit & Favicon Chuẩn Hóa](#-9-kiến-trúc-giao-diện-3d-cockpit--favicon-chuẩn-hóa)

---

## 💻 1. HƯỚNG DẪN SETUP MÔI TRƯỜNG (SETUP ENVIRONMENT)

### Yêu cầu hệ thống:
- **Hệ điều hành**: Linux (Ubuntu 20.04/22.04/24.04), macOS hoặc Windows (WSL2).
- **Python**: Phiên bản `3.10` trở lên.
- **Phần cứng**: Chạy được trên cả **CPU** và **GPU** (NVIDIA CUDA được tự động kích hoạt nếu có để tăng tốc suy luận ~15ms/câu).

### Các bước cài đặt chi tiết:

#### Bước 1: Mở Terminal và di chuyển vào thư mục dự án
```bash
cd MatGroupDemo
```

#### Bước 2: Tạo và kích hoạt môi trường ảo Python (Khuyến nghị)
Tạo môi trường ảo độc lập để tránh xung đột thư viện:
```bash
# Tạo môi trường ảo tên .venv
python3 -m venv .venv

# Kích hoạt môi trường ảo (trên Linux/macOS)
source .venv/bin/activate

# (Đối với Windows Command Prompt / PowerShell)
# .venv\Scripts\activate
```

#### Bước 3: Nâng cấp pip và cài đặt các thư viện cần thiết
```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Bước 4: Kiểm tra môi trường đã sẵn sàng
Chạy lệnh kiểm tra nhanh PyTorch, GPU CUDA và Transformers:
```bash
python3 -c "import torch, transformers, fastapi, flask; print('CUDA khả dụng:', torch.cuda.is_available()); print('Thiết bị:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('Môi trường đã cài đặt thành công!')"
```

---

## 🚀 2. CÁCH KHỞI CHẠY CÁC SERVICE (REST API & WEB UI)

Hệ thống bao gồm 2 Service độc lập hoạt động phối hợp:
1. **Backend Microservice**: FastAPI chạy cổng `8000`.
2. **Frontend Web UI**: Flask chạy cổng `5000`.

---

### SERVICE 1: Microservice REST API (Backend - Cổng 8000)

Dùng để cung cấp API dự đoán cảm xúc cho các ứng dụng khác (Web, Mobile, Data Pipeline).

#### 1. Khởi chạy ở chế độ Console (Xem trực tiếp log):
```bash
python3 build_AI_model/api.py
```
*Hoặc khởi chạy thông qua Uvicorn CLI:*
```bash
uvicorn build_AI_model.api:app --host 0.0.0.0 --port 8000
```

#### 2. Khởi chạy ở chế độ Chạy ngầm (Background / Production Service):
Nếu bạn muốn service tiếp tục chạy khi tắt terminal:
```bash
nohup python3 build_AI_model/api.py > api_service.log 2>&1 &
```
- Kiểm tra tiến trình đang chạy: `ps aux | grep api.py`
- Dừng service khi cần: `pkill -f api.py`

#### 3. Kiểm tra trạng thái hoạt động:
- **Địa chỉ API**: `http://localhost:8000`
- **Tài liệu Swagger UI tương tác**: Mở trình duyệt truy cập `http://localhost:8000/docs`
- **Health check bằng cURL**:
  ```bash
  curl http://localhost:8000/health
  ```
- **Gọi thử API phân loại cảm xúc**:
  ```bash
  curl -X POST "http://localhost:8000/predict" \
       -H "Content-Type: application/json" \
       -d '{"text": "Xe này đi rất êm và tiết kiệm điện"}'
  ```

---

### SERVICE 2: Giao diện Web tương tác (Frontend - Cổng 5000)

Cung cấp giao diện trực quan cho người dùng nhập bình luận, xem nhãn màu và biểu đồ độ tin cậy.

#### 1. Khởi chạy ở chế độ Console:
Mở một cửa sổ Terminal mới (đã kích hoạt môi trường `.venv`):
```bash
python3 demo_flask/app.py
```

#### 2. Khởi chạy ở chế độ Chạy ngầm (Background):
```bash
nohup python3 demo_flask/app.py > flask_service.log 2>&1 &
```
- Dừng service khi cần: `pkill -f app.py`

#### 3. Truy cập sử dụng:
Mở trình duyệt bất kỳ và truy cập địa chỉ:
👉 **`http://localhost:5000`**

*Ghi chú*: Web Flask có cơ chế tự thích ứng thông minh:
- Nếu Backend cổng `8000` đang bật, Web sẽ tự động gọi qua REST API.
- Nếu Backend chưa bật, Web sẽ tự động nạp mô hình nội bộ (Local Fallback) để phục vụ ngay mà không bị gián đoạn.

---

### ⚙️ 3. QUY TRÌNH CHẠY TOÀN BỘ DỰ ÁN (PIPELINE EXECUTION)

Nếu bạn muốn chạy lại toàn bộ quy trình từ dữ liệu thô cào về đến khi ra tập Train/Test, huấn luyện và kiểm thử:

### Bước 1: Làm sạch dữ liệu Facebook (lọc chấm hóng, spam cò mồi, số điện thoại)
```bash
python3 data_processing/crawl_fb_comment_by_bright_data/clean_comments.py
```
- *Đầu vào*: `data_processing/crawl_fb_comment_by_bright_data/facebook_car_comments.csv`
- *Kết quả*: Loại bỏ 180 bình luận rác (dấu chấm '.', spam số điện thoại rao vặt), tạo `facebook_car_comments_clean.csv`.

### Bước 2: Tự động phân loại bình luận bằng PhoBERT
```bash
python3 build_AI_model/label_and_split.py
```
- *Đầu vào*: `data_processing/crawl_fb_comment_by_bright_data/facebook_car_comments_clean.csv`
- *Kết quả*: Tự động gán nhãn cho toàn bộ dữ liệu sạch và lưu vào `build_AI_model/data/facebook_car_comments_labeled.csv`.

### Bước 3: Xuất chuẩn 80% tập Train và 20% tập Test
```bash
python3 build_AI_model/split_train_test.py
```
- *Đầu vào*: `build_AI_model/data/facebook_car_comments_labeled.csv`
- *Kết quả*: 
  - Tạo `build_AI_model/data/train.csv` (80% - giữ 6 cột cốt lõi).
  - Tạo `build_AI_model/data/test.csv` (20% - giữ 6 cột cốt lõi).

### Bước 4: Huấn luyện / Fine-tune mô hình PhoBERT với Weighted Loss (Tối ưu Negative Recall)
```bash
# Chạy huấn luyện chính thức:
python3 build_AI_model/train_model.py --epochs 2 --batch-size 16

# Hoặc chạy kiểm thử nhanh quy trình (Demo mode trong 10 giây):
python3 build_AI_model/train_model.py --demo --epochs 1
```
- *Kết quả*: Huấn luyện mô hình, tối ưu độ phủ lớp Tiêu cực và lưu checkpoint tốt nhất vào `build_AI_model/checkpoints/best_phobert_car_sentiment`.

### Bước 5: Chạy kiểm thử thực tế toàn diện trên 100 câu & Bộ Edge Cases
```bash
# Kiểm thử qua PhoBERT Engine cục bộ:
python3 build_AI_model/test_100_comments.py --mode local

# Hoặc kiểm thử End-to-End qua REST API Microservice (khi port 8000 đang bật):
python3 build_AI_model/test_100_comments.py --mode api
```
- *Kết quả*: In bảng kết quả chi tiết bằng `tabulate`, đánh giá hiệu năng (latency ms, accuracy, negative recall), kiểm thử pin Edge Cases (dấu chấm '.', spam bán xe, emoji lạ, phủ định đảo ngữ) và xuất file `build_AI_model/data/test_100_results.csv`.

---

## 📂 4. GIẢI THÍCH CHI TIẾT TÁC DỤNG TỪNG THƯ MỤC & TỪNG FILE

```text
MatGroupDemo/
├── .env.example                                     # Template cấu hình biến môi trường và API keys
├── .gitignore                                       # Cấu hình bảo vệ secret keys, .venv, logs, cache
├── requirements.txt                                 # Danh sách toàn bộ thư viện Python phụ thuộc
├── README.md                                        # Tài liệu tổng quan dự án
│
├── build_AI_model/                                  # [MODULE 1] Mô hình AI & Microservice REST API
│   ├── classifier.py                                # Lớp bọc mô hình PhoBERT & Bộ lọc tiền xử lý phòng thủ
│   ├── api.py                                       # Web Server REST API (FastAPI) phục vụ dự đoán (Cổng 8000)
│   ├── train_model.py                               # Script huấn luyện/fine-tune PhoBERT với Weighted Loss
│   ├── label_and_split.py                           # Script tự động phân loại bình luận sạch & chia tập
│   ├── split_train_test.py                          # Script xuất chuẩn 2 file train.csv (80%) và test.csv (20%)
│   ├── test_100_comments.py                         # Công cụ kiểm thử toàn diện (hỗ trợ cả mode Local & API)
│   ├── DOCUMENTATION.md                             # Báo cáo kỹ thuật phân tích chuyên sâu
│   ├── checkpoints/                                 # Thư mục lưu trữ model checkpoint sau khi fine-tune
│   └── data/                                        # Thư mục lưu trữ các tệp dữ liệu
│       ├── train.csv                                # [CHÍNH] 80% dữ liệu huấn luyện
│       ├── test.csv                                 # [CHÍNH] 20% dữ liệu kiểm thử độc lập
│       ├── facebook_car_comments_labeled.csv        # Toàn bộ dữ liệu đã được AI gán nhãn
│       ├── test_100_results.csv                     # Kết quả chi tiết kiểm thử 100 câu bình luận thực tế
│       ├── train_comments.csv                       # Bản lưu tập train đầy đủ tất cả các cột gốc
│       └── val_comments.csv                         # Bản lưu tập val đầy đủ tất cả các cột gốc
│
├── AI_suggest/                                      # [MODULE 2] Trí tuệ nhân tạo tạo sinh (GenAI Gemini)
│   ├── gemini_suggest.py                            # Module phân tích chuyên sâu + Fast Memory Cache + Key Rotation
│   └── gemini_api_key                               # Danh sách 23 Google Gemini API keys dự phòng
│
├── demo_flask/                                      # [MODULE 3] Giao diện người dùng Web Demo 3D (Flask)
│   ├── app.py                                       # Ứng dụng Flask Web Server (Port 5000, Dual-Engine, Favicon route)
│   ├── bulk_processor.py                            # Bộ quản lý phân tích hàng loạt với Single Process Lock
│   ├── bulk_outputs/                                # Thư mục lưu trữ các tệp kết quả CSV xuất ra (UTF-8-SIG)
│   ├── favicon.webp                                 # Icon Favicon WebP 3D nhận diện thương hiệu
│   ├── sample_car_comments.xlsx                     # Tệp Excel mẫu chứa 15 bình luận xe hơi thực tế
│   ├── sample_car_comments.txt                      # Tệp TXT mẫu (mỗi câu 1 dòng) để kiểm thử nhanh
│   ├── static/                                      # Thư mục tài nguyên tĩnh
│   ├── templates/
│   │   └── index.html                               # Giao diện 3D Cockpit hiện đại, Dual-Tab (Single / Bulk)
│   └── README.md                                    # Hướng dẫn chi tiết riêng cho Web Flask
│
└── data_processing/                                 # [MODULE 4] Pipeline thu thập và làm sạch dữ liệu thô
    ├── crawl_fb_comment_by_bright_data/             # Thu thập bình luận qua Bright Data API
    │   ├── crawl_fb_cmt.py                          # Script cào bình luận từ các bài viết Facebook
    │   ├── clean_comments.py                        # Bộ lọc rác thông minh (lọc dấu chấm, spam sđt, ads)
    │   ├── facebook_car_comments.csv                # Dữ liệu bình luận thô vừa cào về
    │   ├── facebook_car_comments_clean.csv          # Dữ liệu bình luận sạch đầu vào cho Module AI
    │   └── bright_data_api_key.txt                  # Khóa API cào dữ liệu Bright Data
    │
    └── crawl_fb_post_by_serp/                       # Thu thập link bài viết Facebook qua SerpApi
        ├── crawl_fb_post.py                         # Script cào danh sách link bài viết Facebook
        ├── facebook_car_posts.csv                   # Danh sách bài viết Facebook về ô tô
        └── serp_api_key.txt                         # Khóa API SerpApi
```


### Chi tiết nhiệm vụ từng file:
1. **`build_AI_model/classifier.py`**:
   - Tải và quản lý mô hình PhoBERT `wonrax/phobert-base-vietnamese-sentiment` trên GPU/CPU.
   - Hàm `preprocess_text()`: Làm sạch URL, mã HTML, ký tự đặc biệt lặp lại (`@@@###$$$`), cắt ngắn an toàn nếu văn bản quá dài.
   - Hàm `predict(text)` và `predict_batch(texts)`: Trả về nhãn tiếng Việt (`TÍCH CỰC`, `TIÊU CỰC`, `TRUNG TÍNH`), mã nhãn (`POS`, `NEG`, `NEU`), độ tin cậy `confidence_score` (từ `0.0` đến `1.0`) và điểm chi tiết từng lớp.
2. **`build_AI_model/api.py`**:
   - Khởi tạo FastAPI REST Microservice với CORS middleware.
   - Định nghĩa Schema Pydantic cho request và response.
   - Xử lý mã lỗi HTTP 400 rõ ràng khi nhận chuỗi rỗng.
3. **`build_AI_model/train_model.py`**:
   - Huấn luyện / Fine-tune mô hình PhoBERT với Loss Function có trọng số (Weighted Cross-Entropy).
   - Tối ưu chỉ số sống còn Negative Recall để hạn chế tối đa rủi ro bỏ sót khủng hoảng truyền thông.
   - Tự động lưu checkpoint tốt nhất vào `build_AI_model/checkpoints/best_phobert_car_sentiment`.
4. **`build_AI_model/test_100_comments.py`**:
   - Bộ công cụ kiểm thử toàn diện hỗ trợ 2 chế độ: `--mode local` (chạy trực tiếp trên PhoBERT) và `--mode api` (kiểm thử End-to-End qua Microservice REST API).
   - Tích hợp bộ kiểm thử trường hợp biên đặc biệt (dấu '.', spam số điện thoại, emoij lạ, phủ định đảo ngữ).
   - Xuất bảng kết quả trực quan bằng `tabulate` và lưu file `test_100_results.csv`.
5. **`build_AI_model/split_train_test.py`**:
   - Trích xuất 6 cột cần thiết: `comment_text`, `sentiment_label`, `sentiment`, `confidence_score`, `url`, `post_url`.
   - Chia 80% Train và 20% Test theo kỹ thuật Stratified Sampling.
6. **`demo_flask/app.py`**:
   - Cung cấp giao diện web thân thiện, kết nối backend API hoặc tự nạp mô hình cục bộ (có TTL Cache cho Health Check).
   - Tích hợp endpoint `/bulk/upload`, `/bulk/status`, `/bulk/download/<task_id>` và route trả icon `/favicon.ico`, `/favicon.webp`.
7. **`demo_flask/bulk_processor.py`**:
   - Bộ quản lý tiến trình Bulk Analysis độc lập với cơ chế khóa đơn tiến trình (Single Concurrent Process Lock).
   - Tự động phát hiện cột văn bản và đọc file `.txt`, `.xlsx`, `.xls`, `.csv`.
   - Kết hợp suy luận theo batch với PhoBERT, tùy chọn phân tích sâu với Gemini và xuất CSV chuẩn UTF-8-SIG cho Excel.
8. **`demo_flask/templates/index.html`**:
   - Giao diện 3D Cockpit hiện đại với hệ màu Dark Mode, hiệu ứng nổi (Skeuomorphic & 3D Shadow), favicon logo chuẩn hóa.
   - Dual-Tab linh hoạt: Tab 1 (Phân tích câu đơn kèm Gemini AI) & Tab 2 (Phân tích hàng loạt từ file kèm bảng xem trước trực tiếp).

---

## 📡 5. ĐẶC TẢ REST API MICROSERVICE & XỬ LÝ LỖI BIÊN

### Endpoint: `POST /predict`
- **URL**: `http://localhost:8000/predict`
- **Content-Type**: `application/json`

#### Request Body mẫu:
```json
{
  "text": "Xe này chạy rất êm, nội thất đẹp và rất tiết kiệm điện"
}
```

#### Response thành công (HTTP 200 OK):
```json
{
  "text": "Xe này chạy rất êm, nội thất đẹp và rất tiết kiệm điện",
  "cleaned_text": "Xe này chạy rất êm, nội thất đẹp và rất tiết kiệm điện",
  "label": "TÍCH CỰC",
  "sentiment": "POS",
  "confidence": 0.9928,
  "detail_scores": {
    "POS": 0.9928,
    "NEG": 0.0016,
    "NEU": 0.0056
  },
  "warning": null,
  "process_time_ms": 14.5
}
```

### Khả năng phòng thủ trước các trường hợp biên (Edge Cases):
1. **Chuỗi rỗng hoặc chỉ chứa khoảng trắng (`""`, `"   "`):**
   - API trả về lỗi **`HTTP 400 Bad Request`**:
     ```json
     {"detail": "Văn bản đầu vào không được để trống hoặc chỉ chứa khoảng trắng."}
     ```
2. **Văn bản quá dài (> 2.000 ký tự):**
   - Tự động cắt ngắn về độ dài giới hạn của PhoBERT (256 tokens) an toàn, tránh lỗi Out-of-Memory (OOM) và kèm cờ `warning`.
3. **Ký tự lạ, Emojis, URLs, HTML tags (`@@@###$$$ <p>xe chán vkl</p> 😡😡😡`):**
   - Tự động làm sạch, bóc tách chữ thực tế và nhận diện chính xác nhãn `TIÊU CỰC` (độ tin cậy > 97%).

---

## 🎯 6. LỰA CHỌN MODEL & TIÊU CHÍ ĐÁNH GIÁ NGHIỆP VỤ

### 1. Lý do chọn PhoBERT (`wonrax/phobert-base-vietnamese-sentiment`):
- Được đào tạo trên 20GB văn bản tiếng Việt chuẩn bởi VinAI.
- Bắt trọn ngữ cảnh, từ phủ định phức tạp (*"không thể chê vào đâu được"*, *"chả ra cái tích sự gì"*), tiếng lóng xe cộ (*"vkl"*, *"đầm"*, *"bốc"*, *"ọp ẹp"*, *"ăn xăng"*).
- Chạy cục bộ hoàn toàn miễn phí, an toàn dữ liệu và tốc độ cao (~15ms/câu).

### 2. Lý do chọn Macro F1-Score và Negative Recall:
- **Macro F1-Score**: Dữ liệu mạng xã hội có sự mất cân bằng giữa các lớp. Macro F1 tính trung bình điều hòa không trọng số, buộc mô hình phải học tốt cả 3 lớp thay vì chỉ đoán vào lớp chiếm số đông.
- **Negative Recall (Độ phủ lớp Tiêu cực)**: Trong ngành ô tô, một khiếu nại về sự cố nguy hiểm (lỗi pin, cháy nổ, kẹt chân ga) nếu bị mô hình bỏ sót (**False Negative**) có thể dẫn đến **khủng hoảng truyền thông nghiêm trọng** cho thương hiệu. Do đó, chỉ số Negative Recall là tiêu chuẩn đánh giá quan trọng nhất.

---

## 🤖 7. TÍNH NĂNG AI SUGGESTION CHUYÊN SÂU (GEMINI 3.1 FLASH LITE)

Module [`AI_suggest/`](AI_suggest/) tích hợp mô hình ngôn ngữ lớn **Gemini 3.1 Flash Lite** kết hợp cùng PhoBERT để cung cấp thông tin chi tiết và hành động cụ thể cho đội ngũ Marketing và CSKH.

### 1. Cơ chế Multi-Key Auto-Rotation (23 API Keys)
- Đọc danh sách 23 keys từ [`AI_suggest/gemini_api_key`](AI_suggest/gemini_api_key).
- Khi một key gặp lỗi hạn mức (**Rate Limit / Quota Exceeded / HTTP 429**), hệ thống tự động xoay vòng sang key tiếp theo mà không làm gián đoạn trải nghiệm của người dùng.

### 2. Các nội dung phân tích chuyên sâu:
1. 🌟 **Điểm khách hàng hài lòng / Khen ngợi**: Bóc tách chính xác những tính năng, trải nghiệm mà khách hàng ấn tượng tốt (vận hành êm ái, tiết kiệm điện, thiết kế đẹp...).
2. ⚠️ **Điểm khách hàng chưa hài lòng / Chê / Thắc mắc**: Nhận diện vấn đề cốt lõi (thời gian sạc lâu, cách âm kém, dịch vụ bảo hành chậm...).
3. 💡 **Đề xuất chiến lược cho Marketing & CSKH**: Các giải pháp thực tế (làm video hướng dẫn, tổ chức lái thử xe, CSKH liên hệ hỗ trợ...).
4. 💬 **Gợi ý câu phản hồi mẫu cho Fanpage**: Câu trả lời chuẩn mực, lịch sự, đúng tâm lý khách hàng để Admin Fanpage có thể bấm nút **Copy** và phản hồi trực tiếp ngay lập tức.

### 3. Trải nghiệm trên Giao diện Web:
1. Mở `http://localhost:5000` và phân tích một bình luận bất kỳ.
2. Bấm vào nút tím **`✨ Deep Analyst with AI`**.
3. Kết quả phân tích chuyên sâu và câu trả lời mẫu sẽ lập tức hiển thị bên dưới!

---

## ⚡ 8. TÍNH NĂNG PHÂN TÍCH HÀNG LOẠT (BULK ANALYST - TXT / EXCEL / CSV)

Hệ thống cung cấp module phân tích hàng loạt chuyên nghiệp tại tab **`⚡ Phân Tích Hàng Loạt (Bulk Analyst)`** trên giao diện Web hoặc qua API.

### 1. Định dạng tệp hỗ trợ:
- **Tệp `.txt`**: Mỗi câu bình luận nằm trên 1 dòng riêng biệt (tự động bỏ qua dòng trống). Hỗ trợ nhiều bảng mã: `UTF-8`, `UTF-8-SIG`, `CP1258`, `UTF-16`.
- **Tệp Excel (`.xlsx`, `.xls`)**: Tự động quét và phát hiện cột chứa bình luận (`comment`, `binh_luan`, `noi_dung`, `text`, `content`...) hoặc lấy cột dạng chuỗi đầu tiên.
- **Tệp `.csv`**: Đọc nhanh bằng pandas, tự động phát hiện mã hóa và cột văn bản.

### 2. Cơ chế Khóa Đơn Tiến Trình (Single Concurrent Process Lock):
- Nhằm bảo toàn tài nguyên CPU/GPU và tránh vượt quá hạn ngạch gọi LLM, hệ thống quy định **tại một thời điểm chỉ cho phép chạy DUY NHẤT 1 tiến trình Bulk Analysis**.
- Cơ chế Threading Lock tự động khóa khi tiến trình bắt đầu và giải phóng khi hoàn tất.
- Nếu người dùng gửi thêm tệp khi hệ thống đang bận, server sẽ phản hồi `HTTP 409 Conflict` kèm thông báo tiến độ của tiến trình đang chạy (`Đang bận: X/Y câu`).
- Giao diện web có huy hiệu trạng thái:
  - 🟢 `Single Process: Sẵn sàng`
  - 🟡 `Đang khóa: X/Y câu` (nhấp nháy thời gian thực).

### 3. Tùy chọn Deep Analyst với Gemini AI:
- Khi bật công tắc **"Tích hợp Gemini Deep Analyst cho từng bình luận"**, hệ thống sẽ kết hợp song song PhoBERT và Gemini Flash Lite:
  1. PhoBERT phân loại cảm xúc (Tích cực / Tiêu cực / Trung tính) và tính xác suất chi tiết theo batch.
  2. Gemini AI tự động phân tích ngữ cảnh, bóc tách điểm hài lòng, điểm chưa hài lòng, gợi ý giải pháp Marketing và soạn câu trả lời chuẩn mực cho Fanpage.
  3. Cơ chế Multi-Key Auto-Rotation và SHA-256 Memory Cache đảm bảo tốc độ ổn định và không bị gián đoạn do giới hạn API.

### 4. Xuất Báo Cáo CSV Chuẩn UTF-8-SIG:
- Sau khi hoàn thành, hệ thống lập tức xuất ra tệp CSV lưu tại `demo_flask/bulk_outputs/`.
- Tệp được định dạng chuẩn **`UTF-8-SIG`** (có tiền tố BOM) giúp người dùng mở trực tiếp bằng Microsoft Excel trên mọi hệ điều hành mà **không bao giờ bị lỗi font chữ tiếng Việt có dấu**.
- Bảng xem trước trực quan (Top 10 câu) hiển thị ngay trên giao diện web kèm nút bấm **`📥 Tải File CSV Kết Quả`**.

### 5. Các cột dữ liệu trong file CSV xuất ra:
| Cột | Ý nghĩa |
| :--- | :--- |
| `STT` | Thứ tự câu (1, 2, 3...) |
| `Nội dung bình luận` | Văn bản nhận xét xe ban đầu |
| `Nhãn cảm xúc` | `TÍCH CỰC` / `TRUNG TÍNH` / `TIÊU CỰC` |
| `Mã cảm xúc` | `POS` / `NEU` / `NEG` |
| `Độ tin cậy (%)` | Độ tự tin dự đoán của mô hình PhoBERT |
| `Xác suất Tích cực (%)` | Tỷ lệ xác suất lớp POS |
| `Xác suất Trung tính (%)` | Tỷ lệ xác suất lớp NEU |
| `Xác suất Tiêu cực (%)` | Tỷ lệ xác suất lớp NEG |
| `Cảnh báo` | Ghi chú nhận diện spam số điện thoại, cò mồi bán xe, văn bản cắt ngắn... |
| `Gemini - Điểm hài lòng` | *(Khi bật Deep Analyst)* Điểm khách hàng khen ngợi |
| `Gemini - Điểm chưa hài lòng / Thắc mắc` | *(Khi bật Deep Analyst)* Điểm chê bai, lo ngại, thắc mắc |
| `Gemini - Đề xuất Marketing & CSKH` | *(Khi bật Deep Analyst)* Giải pháp hành động cho chiến dịch |
| `Gemini - Gợi ý câu phản hồi Fanpage` | *(Khi bật Deep Analyst)* Câu trả lời mẫu lịch sự để phản hồi ngay |
| `Gemini Key sử dụng` | *(Khi bật Deep Analyst)* Key thực thi trong danh sách 23 keys |

---

## 🎨 9. KIẾN TRÚC GIAO DIỆN 3D COCKPIT & FAVICON CHUẨN HÓA

### 1. Phong cách thiết kế 3D Cockpit Dark Theme:
- Thiết kế lấy cảm hứng từ bảng điều khiển khoang lái xe điện hiện đại (Cockpit Intelligence).
- Nền chuyển sắc không gian sâu (`#090d16` kết hợp radial gradients ánh sáng tím & ngọc lục bảo).
- Các thẻ kính mờ 3D Glassmorphism (`backdrop-filter: blur(20px)`), viền phản xạ ánh sáng Specular Highlight.
- Nút bấm tạo khối nổi vật lý (Skeuomorphic 3D Push Buttons) với hiệu ứng ấn lún xúc giác khi nhấp chuột (`active:translate-y-1`).
- Thanh đo xác suất cảm xúc hình trụ 3D kèm ánh sáng phát quang Neon (`#10b981`, `#f43f5e`, `#f59e0b`).

### 2. Tích hợp Favicon Logo nhận diện thương hiệu:
- Icon định dạng WebP 3D đặc thù được tích hợp tại:
  - Header logo trung tâm có viền phát quang chuyển động (Halo Ring).
  - Tự động phục vụ qua route `@app.route('/favicon.ico')` và `@app.route('/favicon.webp')`.
  - Nhúng trực tiếp thẻ `<link rel="icon">` trên tab trình duyệt.


