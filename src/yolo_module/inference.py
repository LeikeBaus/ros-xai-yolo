# yolo_module/inference.py
from ultralytics import YOLO

def run_inference(model: YOLO, image_path: str, conf_thres: float = 0.5):
    """
    Run object detection inference using a YOLO model.

    Args:
        model (YOLO): The loaded YOLO model.
        image_path (str): Path to the image to process.
        conf_thres (float): Confidence threshold for filtering detections.

    Returns:
        results (ultralytics.engine.results.Results): Inference results.
    """
    print(f"[INFO] Running inference on: {image_path}")
    results = model.predict(source=image_path, conf=conf_thres, verbose=False)
    return results[0]  # YOLO returns a list; we take the first item
