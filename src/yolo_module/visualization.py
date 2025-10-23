# yolo_module/visualization.py
import cv2
import os

def draw_detections(image_path, detections, output_path="output.jpg"):
    """
    Draw bounding boxes and class labels on the image.

    Args:
        image_path (str): Path to the input image.
        detections (list[dict]): List of detections from postprocessing.
        output_path (str): Path to save the annotated image.

    Returns:
        None
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f"{det['label']} ({det['confidence']:.2f})"

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline),
            (x1 + text_width, y1),
            (0, 255, 0),
            thickness=-1,
        )

        # Put text label
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"[INFO] Saved visualization to: {output_path}")
