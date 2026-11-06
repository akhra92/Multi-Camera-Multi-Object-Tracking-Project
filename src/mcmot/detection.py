"""Person detection with a COCO-pretrained YOLO model (Ultralytics). No training: 'person' is a COCO class."""
import numpy as np


class PersonDetector:
    def __init__(self, weights: str = "yolo11m.pt", imgsz: int = 1280, conf: float = 0.3, device=0):
        from ultralytics import YOLO  # weights are downloaded on first use

        self.model = YOLO(weights)
        self.imgsz, self.conf, self.device = imgsz, conf, device

    def __call__(self, images: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """BGR images -> per image (boxes (N, 4) x1 y1 x2 y2 in pixels, scores (N,))."""
        results = self.model.predict(images, imgsz=self.imgsz, conf=self.conf, classes=[0], device=self.device,
                                     verbose=False)
        return [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()) for r in results]
