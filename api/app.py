from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from src.prediction import load_model, predict_image_bytes
from src.model import MODEL_WEIGHTS_PATH

app = FastAPI(
    title="Traffic Detection API",
    version="1.0.0",
    description="API for running YOLOv8 traffic detection on uploaded images.",
)


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box: Dict[str, float]


class PredictionResponse(BaseModel):
    image_size: Optional[Dict[str, int]]
    detections: List[Detection]


@app.get("/")
def root():
    return {
        "message": "Traffic Detection API",
        "use": [
            "GET /health",
            "GET /docs",
            "POST /predict-image (multipart/form-data, field name 'file')",
        ],
    }


@app.on_event("startup")
def startup_event():
    """
    Load YOLOv8 model at startup (if weights exist).
    """
    try:
        load_model()
        print("model loaded successfully!.")
    except FileNotFoundError as e:
        # Don't crash API; health will show the issue.
        print(f"Could not load model: {e}")


@app.get("/health")
def health_check():
    weights_exists = Path(MODEL_WEIGHTS_PATH).exists()
    return {
        "status": "ok" if weights_exists else "model_missing",
        "model_path": str(MODEL_WEIGHTS_PATH),
        "model_exists": weights_exists,
    }


@app.post(
    "/predict-image",
    response_model=PredictionResponse,
    summary="Run traffic detection on an uploaded image.",
)
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image to get detections.
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file must be an image. Got content_type={file.content_type}",
        )

    try:
        image_bytes = await file.read()
        result_dict = predict_image_bytes(image_bytes)
        return PredictionResponse(**result_dict)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # This will also show details in the terminal
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}",
        )