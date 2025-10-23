# main.py
from yolo_module.yolo_pipe import YOLOPipeline

def main():
    """
    Entry point for the project.
    This file only coordinates which module to run (YOLO, XAI, or ROS).
    """
    # Example: Run YOLO pipeline (later switchable to ROS or XAI)
    seg_model = "yolov8n-seg.pt"
    pipeline = YOLOPipeline(model_path="yolov8n.pt", conf_thres=0.4)
    pipeline.setup()

    detections = pipeline.process_image("sample.jpg", "data/results/output.jpg")

    print(f"[INFO] Detected {len(detections)} objects.")
    for det in detections:
        print(det)

if __name__ == "__main__":
    main()
