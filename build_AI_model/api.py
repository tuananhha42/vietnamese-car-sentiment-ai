import os
import sys
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Đảm bảo import được classifier
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from classifier import VietnameseSentimentClassifier

# Khởi tạo FastAPI App
app = FastAPI(
    title="Vietnamese Sentiment Analysis Microservice",
    description="REST API phân loại cảm xúc bình luận tiếng Việt (Tích cực / Tiêu cực / Trung tính) sử dụng PhoBERT.",
    version="1.0.0"
)

# Thêm CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model khi khởi động service (Singleton)
print("Đang khởi tạo Service và nạp mô hình PhoBERT...")
classifier = VietnameseSentimentClassifier()
print("Microservice đã sẵn sàng phục vụ!")

# === Request / Response Models ===
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        description="Đoạn văn bản tiếng Việt cần phân loại cảm xúc",
        example="Xe này chạy rất êm, nội thất đẹp và tiết kiệm điện"
    )

class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        description="Danh sách các đoạn văn bản cần phân loại cảm xúc theo batch",
        example=[
            "Xe này chạy rất êm và tiết kiệm điện",
            "Xe hay hỏng vặt, bảo hành rất vô trách nhiệm",
            "Giá lăn bánh bao nhiêu vậy mọi người?"
        ]
    )

class PredictResponse(BaseModel):
    text: str
    cleaned_text: str
    label: str = Field(..., description="Nhãn cảm xúc tiếng Việt: TÍCH CỰC | TIÊU CỰC | TRUNG TÍNH")
    sentiment: str = Field(..., description="Mã nhãn: POS | NEG | NEU")
    confidence: float = Field(..., description="Độ tự tin (Confidence score từ 0.0 đến 1.0)")
    detail_scores: Dict[str, float] = Field(..., description="Điểm xác suất chi tiết cho từng lớp")
    warning: Optional[str] = Field(None, description="Cảnh báo nếu đầu vào bị cắt ngắn hoặc xử lý đặc biệt")
    process_time_ms: float = Field(..., description="Thời gian xử lý tính bằng mili-giây")

class HealthResponse(BaseModel):
    status: str
    model_name: str
    device: str
    version: str

# === API Endpoints ===

@app.get("/", tags=["General"])
def root():
    return {
        "message": "Vietnamese Sentiment Analysis API is running!",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "predict_batch": "POST /predict_batch"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    return HealthResponse(
        status="healthy",
        model_name=classifier.model_name,
        device=str(classifier.device),
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictResponse, tags=["Sentiment"])
def predict_sentiment(request: PredictRequest):
    """
    Endpoint phân loại cảm xúc cho một đoạn văn bản.
    Xử lý các trường hợp đầu vào bất thường:
    - Chuỗi rỗng hoặc chỉ có khoảng trắng / ký tự rác -> Trả về lỗi 400 Bad Request rõ ràng.
    - Văn bản quá dài (> 2000 ký tự) -> Tự động cắt ngắn an toàn và trả về kèm warning.
    - Ký tự đặc biệt (HTML tags, URLs, emoij, ký tự rác @@@###) -> Làm sạch và chuẩn hóa trước khi dự đoán.
    """
    raw_text = request.text
    
    # 1. Xử lý trường hợp chuỗi rỗng hoặc chỉ chứa khoảng trắng
    if not raw_text or not raw_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Văn bản đầu vào không được để trống hoặc chỉ chứa khoảng trắng."
        )
        
    start_time = time.time()
    try:
        result = classifier.predict(raw_text)
        process_time_ms = round((time.time() - start_time) * 1000, 2)
        result["process_time_ms"] = process_time_ms
        return PredictResponse(**result)
    except ValueError as e:
        # Trường hợp sau khi làm sạch không còn ký tự chữ nào hợp lệ
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý nội bộ: {str(e)}"
        )

@app.post("/predict_batch", response_model=List[PredictResponse], tags=["Sentiment"])
def predict_sentiment_batch(request: PredictBatchRequest):
    """
    Endpoint phân loại cảm xúc cho nhiều đoạn văn bản cùng lúc (Batch inference).
    """
    if not request.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Danh sách văn bản không được để trống."
        )
        
    start_time = time.time()
    try:
        batch_results = classifier.predict_batch(request.texts)
        total_time_ms = round((time.time() - start_time) * 1000, 2)
        avg_time_ms = round(total_time_ms / len(request.texts), 2)
        
        response_list = []
        for r in batch_results:
            r["process_time_ms"] = avg_time_ms
            response_list.append(PredictResponse(**r))
        return response_list
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý batch nội bộ: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
