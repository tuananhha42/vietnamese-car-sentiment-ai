# HƯỚNG DẪN SỬ DỤNG GIAO DIỆN WEB DEMO 3D (FLASK STUDIO)

## 1. Giới thiệu tổng quan
Ứng dụng Web Demo được thiết kế theo phong cách **3D Dark Cockpit** hiện đại, kết nối trực tiếp với **Mô hình PhoBERT Transformer Phân loại cảm xúc tiếng Việt** và **Trí tuệ Nhân tạo Gemini 3.1 Flash Lite**.

### Các tính năng cốt lõi:
1. **Thiết kế 3D Hiện Đại & Nhận Diện Thương Hiệu:**
   - Giao diện 3D Glassmorphism với nền chuyển sắc không gian sâu.
   - Nút bấm khối nổi xúc giác (3D Skeuomorphic Push Buttons).
   - Biểu đồ xác suất hình trụ phát quang Neon.
   - Favicon WebP 3D đặc thù được tích hợp trực tiếp trên tab trình duyệt và logo trung tâm.

2. **Hệ thống Dual-Tab linh hoạt:**
   - **Tab 1: Phân Tích Đơn Lẻ (Single Text)**: Nhập từng câu bình luận, nhận diện tức thì kèm phân tích chuyên sâu từ Gemini AI (bóc tách điểm khen/chê, chiến lược Marketing và câu trả lời mẫu cho Fanpage).
   - **Tab 2: Phân Tích Hàng Loạt (Bulk Analyst)**: Tải lên file chứa danh sách bình luận (mỗi câu 1 dòng), tự động chấm điểm và xuất file CSV tải về.

3. **Cơ chế Khóa Đơn Tiến Trình (Single Concurrent Process Lock):**
   - Đảm bảo tại một thời điểm **chỉ có duy nhất 1 tiến trình phân tích hàng loạt được phép chạy**, bảo vệ an toàn tài nguyên bộ nhớ và hạn ngạch API.
   - Giao diện hiển thị đèn báo trạng thái theo thời gian thực: `🟢 Single Process: Sẵn sàng` hoặc `🟡 Đang khóa: X/Y câu`.

4. **Xuất file báo cáo CSV chuẩn UTF-8-SIG cho Excel:**
   - Kết quả xuất ra file CSV có tiền tố BOM UTF-8-SIG, giúp mở trực tiếp trên Microsoft Excel (Windows/macOS) mà **không bị lỗi font tiếng Việt có dấu**.
   - Bảng xem trước trực quan (Top 10 dòng) ngay trên giao diện web kèm nút tải nhanh.

---

## 2. Cách khởi chạy ứng dụng

### Cách 1: Chạy độc lập Flask Demo (Local Fallback Engine)
Chạy trực tiếp từ thư mục gốc của dự án:
```bash
python3 demo_flask/app.py
```
Sau đó mở trình duyệt tại: **`http://localhost:5000`**

---

### Cách 2: Chạy đầy đủ Kiến trúc Microservice (Khuyến nghị)
1. **Mở Terminal 1 - Khởi động FastAPI Microservice Backend:**
   ```bash
   python3 build_AI_model/api.py
   ```
   *(Backend chạy tại cổng 8000, cung cấp API dự đoán tốc độ cao và batch inference)*

2. **Mở Terminal 2 - Khởi động Flask Web Frontend:**
   ```bash
   python3 demo_flask/app.py
   ```
   *(Frontend chạy tại cổng 5000, tự động kết nối qua REST API và fallback an toàn)*

---

## 3. Hướng dẫn sử dụng tính năng Phân Tích Hàng Loạt (Bulk Analyst)

1. Mở trình duyệt tại `http://localhost:5000` và chuyển sang tab **`⚡ Phân Tích Hàng Loạt (Bulk Analyst)`**.
2. **Chuẩn bị file dữ liệu:**
   - Hỗ trợ định dạng `.txt` (mỗi câu 1 dòng riêng biệt).
   - Hỗ trợ bảng tính Excel `.xlsx`, `.xls` (chứa cột `comment`, `binh_luan`, `noi_dung`...).
   - Hỗ trợ file `.csv`.
   - *Mẹo:* Bạn có thể bấm nút **"📁 Nạp mẫu 15 câu xe hơi (.TXT)"** trên giao diện để thử nghiệm nhanh không cần chuẩn bị file.
   - Hoặc sử dụng file mẫu có sẵn tại: `demo_flask/sample_car_comments.xlsx` / `demo_flask/sample_car_comments.txt`.
3. **Tùy chọn Gemini Deep Analyst:**
   - Bật công tắc **"Tích hợp Gemini Deep Analyst cho từng bình luận"** nếu muốn bổ sung các cột nhận xét chuyên sâu, đề xuất Marketing và câu phản hồi Fanpage vào file kết quả.
4. **Bắt đầu phân tích:**
   - Bấm nút **"🚀 Bắt Đầu Phân Tích Hàng Loạt"**.
   - Theo dõi tiến độ chạy ngầm trên thanh tiến trình 3D thời gian thực.
5. **Tải file kết quả:**
   - Khi hoàn tất, bấm nút **"📥 Tải File CSV Kết Quả"** để tải về máy tính.

---

## 4. Đặc tả các trường trong file CSV tải về

| Tên Cột | Ý Nghĩa Dữ Liệu |
| :--- | :--- |
| `STT` | Thứ tự câu bình luận (1, 2, 3...) |
| `Nội dung bình luận` | Văn bản nhận xét nguyên bản |
| `Nhãn cảm xúc` | `TÍCH CỰC` / `TRUNG TÍNH` / `TIÊU CỰC` |
| `Mã cảm xúc` | `POS` / `NEU` / `NEG` |
| `Độ tin cậy (%)` | Tỷ lệ tự tin của mô hình PhoBERT (ví dụ: `98.50%`) |
| `Xác suất Tích cực (%)` | Tỷ lệ xác suất lớp POS |
| `Xác suất Trung tính (%)` | Tỷ lệ xác suất lớp NEU |
| `Xác suất Tiêu cực (%)` | Tỷ lệ xác suất lớp NEG |
| `Cảnh báo` | Ghi chú nhận diện spam số điện thoại, văn bản cắt ngắn... |
| `Gemini - Điểm hài lòng` | *(Khi bật Deep Analyst)* Điểm khách hàng khen ngợi |
| `Gemini - Điểm chưa hài lòng / Thắc mắc` | *(Khi bật Deep Analyst)* Điểm chê bai, lo ngại, thắc mắc |
| `Gemini - Đề xuất Marketing & CSKH` | *(Khi bật Deep Analyst)* Giải pháp hành động cụ thể cho chiến dịch |
| `Gemini - Gợi ý câu phản hồi Fanpage` | *(Khi bật Deep Analyst)* Câu trả lời mẫu lịch sự để Admin copy dùng ngay |
| `Gemini Key sử dụng` | *(Khi bật Deep Analyst)* Key trong danh sách 23 keys được phân bổ |
