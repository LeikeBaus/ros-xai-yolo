# yolo_module/model_loader.py
from ultralytics import YOLO
import os

def load_model(model_path=None):
    """
    Load a YOLO model (default: YOLOv8n).
    Args:
        model_path (str): Path to a .pt file or name of a pretrained model.
    Returns:
        YOLO: Loaded YOLO model instance.
    """
    if model_path is None:
        print("[INFO] No model path provided — loading YOLOv8n (default).")
        model = YOLO("yolov8n.pt")
    elif os.path.exists(model_path):
        print(f"[INFO] Loading model from {model_path}")
        model = YOLO(model_path)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return model
