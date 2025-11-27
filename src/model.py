from pathlib import Path
from ultralytics import YOLO

# Path to your trained model file (from Colab)
MODEL_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "models" / "traffic_detection_model.pt"


def get_model(weights_path: str | Path = MODEL_WEIGHTS_PATH) -> YOLO:
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}."
        )

    print(f"Loading model from: {weights_path}")
    model = YOLO(str(weights_path))
    return model


if __name__ == "__main__":
    if MODEL_WEIGHTS_PATH.exists():
        print("Found trained model weights.")
        print(f"   Path: {MODEL_WEIGHTS_PATH}")
        print("   No training is performed. You are ready to use the model.")
    else:
        print("Trained model weights not found.")
        print(f" Expected here: {MODEL_WEIGHTS_PATH}")
        print(" Please copy your 'traffic_detection_model.pt' into this path.")