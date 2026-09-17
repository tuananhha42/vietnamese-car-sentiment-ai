import os
import sys
import time
import requests
from flask import Flask, render_template, request, jsonify, send_file, Response
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

try:
    from AI_suggest.gemini_suggest import GeminiSuggester
except ImportError:
    from gemini_suggest import GeminiSuggester

from demo_flask.bulk_processor import bulk_manager

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Cho phép file lên tới 50MB

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
API_BATCH_URL = os.getenv("API_BATCH_URL", "http://127.0.0.1:8000/predict_batch")
API_HEALTH_URL = os.getenv("API_HEALTH_URL", "http://127.0.0.1:8000/health")
FAVICON_PATH = os.path.join(CURRENT_DIR, "favicon.webp")

_local_classifier = None
_gemini_suggester = None
_last_health_check_time = 0
_last_health_status = False

def get_local_classifier():
    global _local_classifier
    if _local_classifier is None:
        try:
            try:
                from build_AI_model.classifier import VietnameseSentimentClassifier
            except ImportError:
                from classifier import VietnameseSentimentClassifier
            print("Đang nạp mô hình PhoBERT Local Engine cho Flask...")
            _local_classifier = VietnameseSentimentClassifier()
        except Exception as e:
            print("Lỗi khởi tạo Local Classifier:", e)
    return _local_classifier

def get_gemini_suggester():
    global _gemini_suggester
    if _gemini_suggester is None:
        _gemini_suggester = GeminiSuggester()
    return _gemini_suggester

def check_api_alive(ttl_seconds: float = 3.0) -> bool:
    """
    Kiểm tra trạng thái Microservice với cơ chế TTL Cache 3 giây
    nhằm tránh gây độ trễ lặp lại cho mỗi lần người dùng tải trang.
    """
    global _last_health_check_time, _last_health_status
    now = time.time()
    if now - _last_health_check_time < ttl_seconds:
        return _last_health_status
    try:
        res = requests.get(API_HEALTH_URL, timeout=0.5)
        _last_health_status = (res.status_code == 200)
    except Exception:
        _last_health_status = False
    _last_health_check_time = now
    return _last_health_status

def call_microservice_batch(texts):
    """Hàm gọi REST API batch inference sang Microservice"""
    res = requests.post(API_BATCH_URL, json={"texts": texts}, timeout=60)
    if res.status_code == 200:
        return res.json()
    raise RuntimeError(f"Microservice Batch trả về mã lỗi {res.status_code}: {res.text}")

# ==================== FAVICON ROUTE ====================
@app.route("/favicon.ico")
@app.route("/favicon.webp")
def serve_favicon():
    if os.path.exists(FAVICON_PATH):
        with open(FAVICON_PATH, "rb") as f:
            data = f.read()
        return Response(data, mimetype="image/webp", headers={
            "Cache-Control": "public, max-age=86400"
        })
    return ("", 404)

# ==================== WEB PAGES ====================
@app.route("/", methods=["GET"])
def index():
    api_online = check_api_alive()
    mode = "api" if api_online else "local"
    bulk_status = bulk_manager.get_task_status()
    return render_template(
        "index.html",
        mode=mode,
        result=None,
        input_text="",
        bulk_running=bulk_status.get("is_running", False)
    )

@app.route("/predict", methods=["POST"])
def predict():
    global _last_health_status
    raw_text = request.form.get("text", "")
    api_online = check_api_alive()
    mode = "api" if api_online else "local"

    if not raw_text or not raw_text.strip():
        return render_template(
            "index.html",
            mode=mode,
            input_text=raw_text,
            result=None,
            error_msg="Vui lòng nhập văn bản! Bình luận không được để trống hoặc chỉ chứa khoảng trắng."
        )

    # 1. Thử gọi qua Microservice REST API (Port 8000)
    if api_online:
        try:
            resp = requests.post(API_URL, json={"text": raw_text}, timeout=10)
            if resp.status_code == 200:
                result_data = resp.json()
                return render_template(
                    "index.html",
                    mode="api",
                    input_text=raw_text,
                    result=result_data,
                    error_msg=None
                )
            else:
                err_detail = resp.json().get("detail", "Lỗi từ REST API.")
                return render_template(
                    "index.html",
                    mode="api",
                    input_text=raw_text,
                    result=None,
                    error_msg=f"Lỗi API (HTTP {resp.status_code}): {err_detail}"
                )
        except Exception as e:
            print(f"Không thể kết nối REST API, chuyển ngay sang Local Fallback: {e}")
            _last_health_status = False

    # 2. Chế độ Fallback: Chạy trực tiếp bằng mô hình nội bộ
    clf = get_local_classifier()
    if clf is None:
        return render_template(
            "index.html",
            mode=mode,
            input_text=raw_text,
            result=None,
            error_msg="Không thể kết nối đến REST API và không khởi tạo được mô hình cục bộ."
        )

    start_t = time.time()
    try:
        res = clf.predict(raw_text)
        res["process_time_ms"] = round((time.time() - start_t) * 1000, 2)
        return render_template(
            "index.html",
            mode="local",
            input_text=raw_text,
            result=res,
            error_msg=None
        )
    except ValueError as ve:
        return render_template(
            "index.html",
            mode="local",
            input_text=raw_text,
            result=None,
            error_msg=str(ve)
        )
    except Exception as e:
        return render_template(
            "index.html",
            mode="local",
            input_text=raw_text,
            result=None,
            error_msg=f"Lỗi phân loại: {str(e)}"
        )

@app.route("/api/deep_analyze", methods=["POST"])
def deep_analyze():
    data = request.get_json() or {}
    comment_text = data.get("text", "").strip()
    sentiment_label = data.get("sentiment", "").strip()

    if not comment_text:
        return jsonify({"success": False, "error": "Văn bản không được để trống."}), 400

    try:
        suggester = get_gemini_suggester()
        analysis = suggester.analyze_comment(comment_text, sentiment_label=sentiment_label)
        return jsonify({
            "success": True,
            "data": analysis
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Lỗi phân tích Gemini: {str(e)}"
        }), 500

# ==================== BULK ANALYSIS ENDPOINTS ====================
@app.route("/bulk/upload", methods=["POST"])
def bulk_upload():
    """
    Tiếp nhận file (.txt, .xlsx, .xls, .csv) để phân tích hàng loạt.
    Kiểm tra cơ chế khóa đơn tiến trình (Single Concurrent Process).
    """
    if bulk_manager.is_running:
        status_info = bulk_manager.get_task_status()
        cur_prog = f"{status_info.get('current', 0)}/{status_info.get('total', 0)}"
        return jsonify({
            "success": False,
            "busy": True,
            "error": f"Hệ thống đang bận thực hiện 1 tiến trình phân tích khác ({cur_prog} câu). Vui lòng đợi tiến trình hiện tại chạy xong!"
        }), 409

    if "file" not in request.files:
        return jsonify({"success": False, "error": "Vui lòng chọn file tải lên (.txt, .xlsx, .xls, .csv)."}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"success": False, "error": "Chưa chọn file hoặc tên file không hợp lệ."}), 400

    deep_analyst_flag = request.form.get("deep_analyst", "false").lower() in ["true", "1", "on"]

    try:
        comments, filename = bulk_manager.parse_uploaded_file(uploaded_file)
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Lỗi đọc file: {str(e)}"}), 500

    api_online = check_api_alive()
    api_caller = call_microservice_batch if api_online else None

    success, msg, task_id = bulk_manager.start_task(
        comments=comments,
        filename=filename,
        deep_analyst=deep_analyst_flag,
        classifier_getter=get_local_classifier,
        suggester_getter=get_gemini_suggester,
        api_caller=api_caller
    )

    if not success:
        return jsonify({"success": False, "busy": True, "error": msg}), 409

    return jsonify({
        "success": True,
        "message": msg,
        "task_id": task_id,
        "total": len(comments),
        "filename": filename,
        "deep_analyst": deep_analyst_flag
    })

@app.route("/bulk/status", methods=["GET"])
def bulk_status():
    """Endpoint polling trạng thái tiến trình phân tích hàng loạt"""
    task_id = request.args.get("task_id")
    status_data = bulk_manager.get_task_status(task_id)
    return jsonify(status_data)

@app.route("/bulk/download/<task_id>", methods=["GET"])
def bulk_download(task_id):
    """Tải file CSV kết quả"""
    csv_path = bulk_manager.get_csv_path(task_id)
    if not csv_path or not os.path.exists(csv_path):
        return jsonify({"error": "Không tìm thấy file kết quả hoặc file đã bị xóa."}), 404
    return send_file(
        csv_path,
        as_attachment=True,
        download_name=os.path.basename(csv_path),
        mimetype="text/csv"
    )

@app.route("/bulk/sample/<file_type>", methods=["GET"])
def download_sample_file(file_type):
    """
    Tải file dữ liệu mẫu cho khách hàng tham khảo format chuẩn (.txt, .csv, .xlsx).
    Quy chuẩn: mỗi dòng 1 câu bình luận, không cần header.
    """
    ft = file_type.lower().strip()
    mapping = {
        "txt": ("sample_car_comments.txt", "text/plain; charset=utf-8"),
        "csv": ("sample_car_comments.csv", "text/csv; charset=utf-8"),
        "xlsx": ("sample_car_comments.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        "xls": ("sample_car_comments.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    }
    if ft not in mapping:
        return jsonify({"error": f"Định dạng mẫu '{file_type}' không được hỗ trợ. Vui lòng chọn txt, csv hoặc xlsx."}), 404

    filename, mimetype = mapping[ft]
    file_path = os.path.join(CURRENT_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": f"Không tìm thấy file mẫu {filename}."}), 404

    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename,
        mimetype=mimetype
    )

@app.route("/bulk/clear", methods=["POST"])
def bulk_clear():
    """Xóa trạng thái tác vụ phân tích hàng loạt đã hoàn tất để chuẩn bị tải file mới"""
    if bulk_manager.is_running:
        return jsonify({"success": False, "error": "Không thể xóa trạng thái khi tiến trình phân tích đang chạy."}), 400
    bulk_manager.clear_task()
    return jsonify({"success": True, "message": "Đã làm mới trạng thái sẵn sàng."})

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", 5000))
    print("=" * 60)
    print("KHỞI ĐỘNG GIAO DIỆN WEB DEMO FLASK (TÍCH HỢP BULK ANALYST & GEMINI)")
    print(f"Truy cập trình duyệt: http://127.0.0.1:{port}")
    print("=" * 60)
    app.run(host=host, port=port, debug=False)
