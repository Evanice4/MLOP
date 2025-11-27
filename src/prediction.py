from pathlib import Path
from io import BytesIO
from typing import Any, Dict, List, Union

from PIL import Image
from ultralytics import YOLO

from .model import get_model, MODEL_WEIGHTS_PATH

_model_cache: YOLO | None = None


def load_model() -> YOLO:
    global _model_cache
    if _model_cache is None:
        _model_cache = get_model(MODEL_WEIGHTS_PATH)
    return _model_cache


def _results_to_dict(results) -> Dict[str, Any]:
    if not results:
        return {"detections": [], "image_size": None}

    res = results[0]
    boxes = res.boxes

    # Original image size (height, width)
    h, w = res.orig_shape
    detections: List[Dict[str, Any]] = []

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().tolist()   
        confs = boxes.conf.cpu().tolist()  
        classes = boxes.cls.cpu().tolist() 

        names = res.names  

        for bbox, conf, cls_idx in zip(xyxy, confs, classes):
            cls_idx_int = int(cls_idx)
            detections.append(
                {
                    "class_id": cls_idx_int,
                    "class_name": names.get(cls_idx_int, str(cls_idx_int)),
                    "confidence": float(conf),
                    "box": {
                        "xmin": float(bbox[0]),
                        "ymin": float(bbox[1]),
                        "xmax": float(bbox[2]),
                        "ymax": float(bbox[3]),
                    },
                }
            )

    return {
        "image_size": {"width": int(w), "height": int(h)},
        "detections": detections,
    }


def predict_image_file(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Run model detection on an image file path.
    """
    model = load_model()
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    results = model.predict(source=str(image_path), verbose=False)
    return _results_to_dict(results)


def predict_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """
    Run model detection on raw image bytes.
    """
    model = load_model()

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    results = model.predict(source=image, verbose=False)

    return _results_to_dict(results)