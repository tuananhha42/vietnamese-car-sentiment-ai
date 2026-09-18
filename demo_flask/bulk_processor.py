import os
import io
import time
import uuid
import threading
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BULK_OUTPUT_DIR = os.path.join(CURRENT_DIR, "bulk_outputs")
os.makedirs(BULK_OUTPUT_DIR, exist_ok=True)

# Các tên cột thông dụng có thể chứa văn bản bình luận trong file CSV/Excel
TEXT_COLUMN_CANDIDATES = [
    "comment", "comments", "binh_luan", "binhluan", "noi_dung", "noidung",
    "text", "content", "cau", "cau_noi", "review", "reviews", "feedback", "message",
    "bình luận", "nội dung", "câu", "đánh giá", "ý kiến", "nhận xét", "câu bình luận"
]

class BulkProcessManager:
    """
    Quản lý tiến trình phân tích hàng loạt đa luồng độc lập theo Session (Multi-Session Concurrent Processing).
    Mỗi phiên truy cập (Session) của người dùng được phân bổ một worker thread riêng biệt,
    đảm bảo phân tách hoàn toàn kết quả, tiến trình và dữ liệu giữa các máy truy cập khác nhau.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._session_active_task: Dict[str, str] = {}
        self._worker_threads: Dict[str, threading.Thread] = {}

    def is_session_running(self, session_id: Optional[str]) -> bool:
        """Kiểm tra xem session cụ thể có đang chạy tác vụ nào không"""
        if not session_id:
            return False
        with self._lock:
            task_id = self._session_active_task.get(session_id)
            if task_id and task_id in self._tasks:
                return self._tasks[task_id].get("status") == "running"
            return False

    @property
    def is_running(self) -> bool:
        """Trả về True nếu có bất kỳ worker thread nào đang chạy trên server"""
        with self._lock:
            return any(t.is_alive() for t in self._worker_threads.values())

    def parse_uploaded_file(self, file_storage) -> Tuple[List[str], str]:
        """
        Phân tích cú pháp file người dùng tải lên:
        - .txt: Mỗi dòng là 1 câu
        - .csv: Đọc bằng pandas (hỗ trợ cả có header lẫn không có header)
        - .xlsx / .xls: Đọc bảng tính Excel (hỗ trợ cả có header lẫn không có header)
        Trả về: (danh sách các câu văn bản hợp lệ, tên file gốc)
        """
        filename = file_storage.filename or "unknown_file.txt"
        ext = os.path.splitext(filename)[1].lower()
        file_bytes = file_storage.read()

        if not file_bytes:
            raise ValueError("File tải lên rỗng hoặc không có dữ liệu.")

        comments: List[str] = []

        if ext == ".txt":
            # Thử nhiều encoding phổ biến ở Việt Nam
            decoded_text = None
            for enc in ["utf-8-sig", "utf-8", "utf-16", "cp1258", "latin-1"]:
                try:
                    decoded_text = file_bytes.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue

            if decoded_text is None:
                raise ValueError("Không thể giải mã file text. Vui lòng lưu file ở định dạng UTF-8.")

            for line in decoded_text.splitlines():
                stripped = line.strip()
                if stripped:
                    comments.append(stripped)

        elif ext in [".xlsx", ".xls"]:
            try:
                df = pd.read_excel(io.BytesIO(file_bytes), header=None)
            except Exception as e:
                raise ValueError(f"Không thể đọc file Excel: {str(e)}")
            comments = self._extract_text_from_df(df)

        elif ext == ".csv":
            df = None
            for enc in ["utf-8-sig", "utf-8", "cp1258", "latin-1"]:
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc, header=None)
                    break
                except Exception:
                    continue

            if df is not None:
                comments = self._extract_text_from_df(df)
            else:
                # Fallback nếu file CSV có cấu trúc dòng không đều (ví dụ mỗi dòng 1 câu nhưng chứa dấu phẩy chưa đóng ngoặc kép)
                decoded_text = None
                for enc in ["utf-8-sig", "utf-8", "cp1258", "latin-1"]:
                    try:
                        decoded_text = file_bytes.decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
                if decoded_text is not None:
                    raw_lines = [line.strip().strip('"\'') for line in decoded_text.splitlines() if line.strip()]
                    if raw_lines and raw_lines[0].lower() in TEXT_COLUMN_CANDIDATES:
                        raw_lines = raw_lines[1:]
                    comments = raw_lines
                else:
                    raise ValueError("Không thể đọc file CSV. Vui lòng lưu file ở định dạng UTF-8.")

        else:
            raise ValueError(f"Định dạng file '{ext}' không được hỗ trợ. Vui lòng chỉ tải lên file .txt, .xlsx, .xls hoặc .csv.")

        if not comments:
            raise ValueError("Không tìm thấy dòng văn bản hợp lệ nào trong file.")

        return comments, filename

    def _extract_text_from_df(self, df: pd.DataFrame) -> List[str]:
        """
        Tự động tìm cột văn bản và trích xuất dữ liệu:
        - Hỗ trợ file không có header (mỗi dòng 1 câu bình luận ngay từ dòng đầu tiên).
        - Hỗ trợ file có header thông dụng (comment, binh_luan, text, content...).
        - Tự động chọn cột văn bản nếu file có nhiều cột.
        """
        if df.empty:
            return []

        # 1. Kiểm tra dòng đầu tiên xem có phải là dòng tiêu đề (Header) hay không
        first_row_vals = [str(x).strip().lower() for x in df.iloc[0].values]
        header_col_idx = None
        for idx, val in enumerate(first_row_vals):
            if val in TEXT_COLUMN_CANDIDATES:
                header_col_idx = idx
                break

        if header_col_idx is not None:
            # File có dòng tiêu đề -> bỏ qua dòng 0, lấy từ dòng 1
            series = df.iloc[1:, header_col_idx]
        else:
            # File KHÔNG có dòng tiêu đề (đúng chuẩn: mỗi dòng 1 câu bình luận)
            # Tự động chọn cột chứa nội dung văn bản dài nhất
            best_col = 0
            max_avg_len = -1.0
            for col_idx in range(df.shape[1]):
                col_series = df.iloc[:, col_idx].dropna().astype(str)
                non_empty = [s for s in col_series if s.strip() and s.strip().lower() != "nan"]
                if non_empty:
                    avg_len = sum(len(s) for s in non_empty) / len(non_empty)
                    if avg_len > max_avg_len:
                        max_avg_len = avg_len
                        best_col = col_idx
            series = df.iloc[:, best_col]

        extracted = [str(s).strip() for s in series.dropna() if str(s).strip() and str(s).strip().lower() != "nan"]
        return extracted

    def start_task(
        self,
        session_id: str,
        comments: List[str],
        filename: str,
        deep_analyst: bool,
        classifier_getter,
        suggester_getter,
        api_caller=None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Bắt đầu một tiến trình Bulk Analysis độc lập cho session_id cụ thể.
        Mỗi session chạy trên một worker thread riêng biệt.
        """
        with self._lock:
            prev_tid = self._session_active_task.get(session_id)
            if prev_tid and prev_tid in self._tasks and self._tasks[prev_tid].get("status") == "running":
                cur = self._tasks[prev_tid]
                cur_prog = f"{cur.get('current', 0)}/{cur.get('total', 0)}"
                return False, f"Phiên làm việc của bạn đang có 1 tiến trình phân tích đang chạy ({cur_prog} câu). Vui lòng đợi tiến trình hiện tại hoàn tất!", None

            task_id = str(uuid.uuid4())[:8]
            task_info = {
                "task_id": task_id,
                "session_id": session_id,
                "filename": filename,
                "total": len(comments),
                "current": 0,
                "progress_pct": 0,
                "status": "running",  # running | completed | error
                "deep_analyst": deep_analyst,
                "started_at": time.time(),
                "elapsed_seconds": 0,
                "current_text": "Đang khởi tạo mô hình...",
                "summary": {"POS": 0, "NEU": 0, "NEG": 0},
                "preview": [],
                "csv_filename": None,
                "csv_download_url": None,
                "error": None
            }
            self._tasks[task_id] = task_info
            self._session_active_task[session_id] = task_id

        worker_thread = threading.Thread(
            target=self._run_worker,
            args=(task_id, session_id, comments, filename, deep_analyst, classifier_getter, suggester_getter, api_caller),
            name=f"BulkWorker-{task_id}",
            daemon=True
        )
        with self._lock:
            self._worker_threads[task_id] = worker_thread
        worker_thread.start()

        return True, "Tiến trình phân tích hàng loạt đã được khởi động thành công.", task_id

    def _run_worker(
        self,
        task_id: str,
        session_id: str,
        comments: List[str],
        filename: str,
        deep_analyst: bool,
        classifier_getter,
        suggester_getter,
        api_caller
    ):
        """Worker chạy ngầm cho 1 tiến trình phân tích của session cụ thể"""
        total = len(comments)
        started_at = time.time()
        results: List[Dict[str, Any]] = []

        try:
            # 1. PHÂN LOẠI CẢM XÚC BẰNG PHOBERT (BATCH INFERENCE)
            batch_size = 32
            sentiment_results = []
            
            # Kiểm tra xem có thể gọi REST API không
            used_api = False
            if api_caller is not None:
                try:
                    for b_idx in range(0, total, batch_size):
                        batch = comments[b_idx : b_idx + batch_size]
                        res_batch = api_caller(batch)
                        sentiment_results.extend(res_batch)
                    used_api = True
                except Exception as e:
                    print(f"[BulkWorker-{task_id}] API batch thất bại, chuyển sang local classifier: {e}")
                    sentiment_results = []

            if not used_api or not sentiment_results:
                clf = classifier_getter()
                if clf is None:
                    raise RuntimeError("Không thể tải mô hình PhoBERT để phân tích.")
                sentiment_results = clf.predict_batch(comments, batch_size=batch_size)

            # Cập nhật số liệu ban đầu
            summary = {"POS": 0, "NEU": 0, "NEG": 0}
            for sr in sentiment_results:
                s_code = sr.get("sentiment", "NEU")
                summary[s_code] = summary.get(s_code, 0) + 1

            # 2. PHÂN TÍCH CHUYÊN SÂU GEMINI (NẾU ĐƯỢC CHỌN)
            suggester = None
            if deep_analyst:
                suggester = suggester_getter()

            for idx, item in enumerate(sentiment_results):
                text = item.get("text", "")
                sentiment = item.get("sentiment", "NEU")
                label = item.get("label", "TRUNG TÍNH")
                confidence = item.get("confidence", 0.0)
                details = item.get("detail_scores", {})
                warning = item.get("warning", "")

                gemini_data = None
                if deep_analyst and suggester is not None:
                    # Cập nhật trạng thái câu đang phân tích
                    with self._lock:
                        if task_id in self._tasks:
                            self._tasks[task_id]["current"] = idx + 1
                            self._tasks[task_id]["progress_pct"] = int(((idx + 1) / total) * 100)
                            self._tasks[task_id]["elapsed_seconds"] = round(time.time() - started_at, 1)
                            self._tasks[task_id]["current_text"] = (text[:70] + "...") if len(text) > 70 else text

                    try:
                        clean_input = item.get("cleaned_text") or text
                        gemini_res = suggester.analyze_comment(clean_input, sentiment_label=label)
                        gemini_data = gemini_res
                    except Exception as ge:
                        gemini_data = {
                            "hai_long": [],
                            "chua_hai_long": [f"Lỗi phân tích Gemini: {str(ge)}"],
                            "de_xuat_marketing": [],
                            "cau_phan_hoi_mau": "Cảm ơn quý khách đã gửi phản hồi.",
                            "key_used": "N/A"
                        }
                else:
                    # Nếu không dùng Gemini, cập nhật tiến độ theo từng nhóm
                    if (idx + 1) % 10 == 0 or (idx + 1) == total:
                        with self._lock:
                            if task_id in self._tasks:
                                self._tasks[task_id]["current"] = idx + 1
                                self._tasks[task_id]["progress_pct"] = int(((idx + 1) / total) * 100)
                                self._tasks[task_id]["elapsed_seconds"] = round(time.time() - started_at, 1)
                                self._tasks[task_id]["current_text"] = (text[:70] + "...") if len(text) > 70 else text

                # Đóng gói dữ liệu từng dòng
                row_record = {
                    "stt": idx + 1,
                    "text": text,
                    "label": label,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "pos_pct": details.get("POS", 0.0),
                    "neu_pct": details.get("NEU", 0.0),
                    "neg_pct": details.get("NEG", 0.0),
                    "warning": warning,
                    "gemini": gemini_data
                }
                results.append(row_record)

                # Giữ 10 kết quả gần nhất cho bảng xem trước trực tiếp trên giao diện
                with self._lock:
                    if task_id in self._tasks:
                        self._tasks[task_id]["summary"] = summary
                        if len(self._tasks[task_id]["preview"]) < 10:
                            self._tasks[task_id]["preview"].append({
                                "stt": idx + 1,
                                "text": text,
                                "label": label,
                                "sentiment": sentiment,
                                "confidence": f"{confidence * 100:.1f}%",
                                "gemini_reply": gemini_data.get("cau_phan_hoi_mau", "") if gemini_data else ""
                            })

            # 3. XUẤT RA FILE CSV CHUẨN UTF-8-SIG
            csv_rows = []
            for r in results:
                csv_item = {
                    "STT": r["stt"],
                    "Nội dung bình luận": r["text"],
                    "Nhãn cảm xúc": r["label"],
                    "Mã cảm xúc": r["sentiment"],
                    "Độ tin cậy (%)": f"{r['confidence'] * 100:.2f}%",
                    "Xác suất Tích cực (%)": f"{r['pos_pct'] * 100:.2f}%",
                    "Xác suất Trung tính (%)": f"{r['neu_pct'] * 100:.2f}%",
                    "Xác suất Tiêu cực (%)": f"{r['neg_pct'] * 100:.2f}%",
                    "Cảnh báo": r["warning"] or ""
                }
                if deep_analyst:
                    g = r.get("gemini") or {}
                    csv_item["Gemini - Điểm hài lòng"] = "\n- ".join(g.get("hai_long", [])) if g.get("hai_long") else ""
                    csv_item["Gemini - Điểm chưa hài lòng / Thắc mắc"] = "\n- ".join(g.get("chua_hai_long", [])) if g.get("chua_hai_long") else ""
                    csv_item["Gemini - Đề xuất Marketing & CSKH"] = "\n- ".join(g.get("de_xuat_marketing", [])) if g.get("de_xuat_marketing") else ""
                    csv_item["Gemini - Gợi ý câu phản hồi Fanpage"] = g.get("cau_phan_hoi_mau", "")

                csv_rows.append(csv_item)

            df_out = pd.DataFrame(csv_rows)
            time_str = time.strftime("%Y%m%d_%H%M%S")
            clean_fname = os.path.splitext(filename)[0][:20]
            mode_suffix = "with_gemini" if deep_analyst else "phobert"
            csv_filename = f"sentiment_report_{clean_fname}_{mode_suffix}_{time_str}_{task_id}.csv"
            csv_path = os.path.join(BULK_OUTPUT_DIR, csv_filename)
            
            # Ghi file với encoding utf-8-sig để Excel hiển thị đúng tiếng Việt có dấu
            df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")

            total_elapsed = round(time.time() - started_at, 2)
            with self._lock:
                if task_id in self._tasks:
                    self._tasks[task_id]["status"] = "completed"
                    self._tasks[task_id]["progress_pct"] = 100
                    self._tasks[task_id]["current"] = total
                    self._tasks[task_id]["elapsed_seconds"] = total_elapsed
                    self._tasks[task_id]["csv_filename"] = csv_filename
                    self._tasks[task_id]["csv_download_url"] = f"/bulk/download/{task_id}"
                    self._tasks[task_id]["current_text"] = "Phân tích hoàn tất! Bạn có thể tải file kết quả."

        except Exception as err:
            import traceback
            traceback.print_exc()
            with self._lock:
                if task_id in self._tasks:
                    self._tasks[task_id]["status"] = "error"
                    self._tasks[task_id]["error"] = str(err)
                    self._tasks[task_id]["current_text"] = f"Lỗi: {str(err)}"

        finally:
            with self._lock:
                self._worker_threads.pop(task_id, None)

    def _get_task_from_disk(self, target_task_id: str) -> Optional[Dict[str, Any]]:
        """Tự động khôi phục metadata của tác vụ hoàn tất từ file CSV trên đĩa NẾU có target_task_id cụ thể"""
        if not target_task_id or not target_task_id.strip():
            return None
        try:
            if not os.path.exists(BULK_OUTPUT_DIR):
                return None
            csv_files = [
                f for f in os.listdir(BULK_OUTPUT_DIR)
                if f.startswith("sentiment_report_") and f.endswith(".csv")
            ]
            matched_file = None
            clean_tid = target_task_id.strip()
            for f in csv_files:
                if clean_tid in f:
                    matched_file = f
                    break

            if not matched_file:
                return None

            csv_path = os.path.join(BULK_OUTPUT_DIR, matched_file)
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            total = len(df)
            pos_count = int((df["Mã cảm xúc"] == "POS").sum()) if "Mã cảm xúc" in df else 0
            neu_count = int((df["Mã cảm xúc"] == "NEU").sum()) if "Mã cảm xúc" in df else 0
            neg_count = int((df["Mã cảm xúc"] == "NEG").sum()) if "Mã cảm xúc" in df else 0
            
            preview = []
            for idx, row in df.head(10).iterrows():
                preview.append({
                    "stt": idx + 1,
                    "text": str(row.get("Nội dung bình luận", "")),
                    "label": str(row.get("Nhãn cảm xúc", "")),
                    "sentiment": str(row.get("Mã cảm xúc", "")),
                    "confidence": str(row.get("Độ tin cậy (%)", "")),
                    "gemini_reply": str(row.get("Gemini - Gợi ý câu phản hồi Fanpage", "")) if pd.notna(row.get("Gemini - Gợi ý câu phản hồi Fanpage")) else ""
                })

            task_info = {
                "task_id": clean_tid,
                "filename": matched_file,
                "total": total,
                "current": total,
                "progress_pct": 100,
                "status": "completed",
                "deep_analyst": "with_gemini" in matched_file,
                "started_at": os.path.getmtime(csv_path),
                "elapsed_seconds": 0.0,
                "current_text": "Phân tích hoàn tất! Bạn có thể tải file kết quả.",
                "summary": {"POS": pos_count, "NEU": neu_count, "NEG": neg_count},
                "preview": preview,
                "csv_filename": matched_file,
                "csv_download_url": f"/bulk/download/{clean_tid}",
                "error": None
            }
            self._tasks[clean_tid] = task_info
            return task_info
        except Exception as e:
            print(f"[BulkManager] Lỗi đọc tác vụ từ đĩa: {e}")
            return None

    def clear_task(self, session_id: Optional[str] = None):
        """Xóa trạng thái tác vụ của session này để người dùng bắt đầu tác vụ mới"""
        with self._lock:
            if not session_id:
                return
            if session_id in self._session_active_task:
                tid = self._session_active_task[session_id]
                if tid in self._tasks and self._tasks[tid].get("status") == "running":
                    return  # Không xóa khi đang chạy
                del self._session_active_task[session_id]

    def get_task_status(self, session_id: Optional[str] = None, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Lấy trạng thái tác vụ cho session và task_id cụ thể.
        Tuyệt đối không trả về tác vụ của session khác nếu session hiện tại chưa có tác vụ.
        """
        with self._lock:
            # 1. Nếu client truyền task_id cụ thể (ví dụ từ localStorage khi F5)
            if task_id and task_id.strip():
                clean_tid = task_id.strip()
                if clean_tid in self._tasks:
                    task = self._tasks[clean_tid]
                    # Nếu task có session_id và session_id truyền vào không khớp -> không cho thấy task của người khác
                    if session_id and task.get("session_id") and task.get("session_id") != session_id:
                        return {
                            "status": "idle",
                            "is_running": False,
                            "progress_pct": 0,
                            "total": 0,
                            "current": 0,
                            "current_text": "Hệ thống sẵn sàng tiếp nhận file."
                        }
                    info = dict(task)
                    info["is_running"] = (info.get("status") == "running")
                    return info
                
                # Thử tìm trên đĩa theo clean_tid
                disk_task = self._get_task_from_disk(clean_tid)
                if disk_task:
                    disk_task["is_running"] = False
                    return disk_task

            # 2. Nếu không có task_id, chỉ tìm tác vụ thuộc về session_id này
            if session_id and session_id in self._session_active_task:
                sess_tid = self._session_active_task[session_id]
                if sess_tid in self._tasks:
                    info = dict(self._tasks[sess_tid])
                    info["is_running"] = (info.get("status") == "running")
                    return info

            # 3. Session này chưa có tác vụ nào -> Trả về IDLE (sẵn sàng nhận file)
            return {
                "status": "idle",
                "is_running": False,
                "progress_pct": 0,
                "total": 0,
                "current": 0,
                "current_text": "Hệ thống sẵn sàng tiếp nhận file."
            }

    def get_csv_path(self, task_id: str, session_id: Optional[str] = None) -> Optional[str]:
        """Tìm đường dẫn file CSV theo task_id, bảo vệ quyền riêng tư theo session"""
        if not task_id or not task_id.strip():
            return None
        target_id = task_id.strip()
        with self._lock:
            task = self._tasks.get(target_id)
            if task:
                if session_id and task.get("session_id") and task.get("session_id") != session_id:
                    return None
                if task.get("csv_filename"):
                    path = os.path.join(BULK_OUTPUT_DIR, task["csv_filename"])
                    if os.path.exists(path):
                        return path

            # Quét file trên đĩa có chứa target_id cụ thể
            if os.path.exists(BULK_OUTPUT_DIR):
                for f in os.listdir(BULK_OUTPUT_DIR):
                    if target_id in f and f.endswith(".csv"):
                        return os.path.join(BULK_OUTPUT_DIR, f)

        return None

# Singleton Instance
bulk_manager = BulkProcessManager()
