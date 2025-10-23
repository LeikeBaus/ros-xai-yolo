# yolo_module/yolo_pipe.py
from yolo_module.model_loader import load_model
from yolo_module.inference import run_inference
from yolo_module.postprocessing import extract_detections
from yolo_module.visualization import draw_detections

class YOLOPipeline:
    """
    Orchestrates the YOLO inference workflow:
    model loading → inference → postprocessing → visualization
    """

    def __init__(self, model_path=None, conf_thres=0.5):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.model = None

    def setup(self):
        """Load the YOLO model."""
        self.model = load_model(self.model_path)

    def process_image(self, image_path, output_path="output.jpg"):
        """
        Run the complete inference pipeline on one image.
        Returns the extracted detections.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call setup() first.")

        result = run_inference(self.model, image_path, self.conf_thres)
        detections = extract_detections(result)
        draw_detections(image_path, detections, output_path)
        return detections
