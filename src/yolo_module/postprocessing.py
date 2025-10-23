# yolo_module/postprocessing.py
import numpy as np

def extract_detections(result):
    """
    Extract bounding boxes, class labels, and confidence scores
    from a YOLO result object.

    Args:
        result (ultralytics.engine.results.Results): YOLO inference result.

    Returns:
        list[dict]: A list of detection dictionaries with bounding boxes, class names, and confidence.
    """
    detections = []
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        print("[INFO] No detections found.")
        return detections

    for box in boxes:
        # YOLO provides xyxy coordinates (top-left, bottom-right)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = result.names[cls] if result.names and cls in result.names else str(cls)

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class_id": cls,
            "label": label
        })

    print(f"[INFO] Extracted {len(detections)} detections.")
    return detections
