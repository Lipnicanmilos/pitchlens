"""
Detekcia hráčov, rozhodcu a lopty pomocou YOLOv8.

Použitie:
    from detect import Detector
    detector = Detector(model_path="models/yolov8n.pt", confidence=0.35)
    detections = detector.detect(frame)
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    xyxy: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


class Detector:
    def __init__(self, model_path: str, confidence: float = 0.35):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Spustí detekciu na jednom snímku a vráti zoznam detekcií."""
        results = self.model.predict(frame, conf=self.confidence, verbose=False)[0]

        detections = []
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = results.names.get(cls_id, str(cls_id))
            detections.append(Detection(xyxy=xyxy, confidence=conf, class_id=cls_id, class_name=cls_name))

        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Spustí detekciu na viacerých snímkoch naraz (efektívnejšie na GPU)."""
        results = self.model.predict(frames, conf=self.confidence, verbose=False)

        all_detections = []
        for result in results:
            frame_detections = []
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names.get(cls_id, str(cls_id))
                frame_detections.append(
                    Detection(xyxy=xyxy, confidence=conf, class_id=cls_id, class_name=cls_name)
                )
            all_detections.append(frame_detections)

        return all_detections


# NOTE: yolov8n.pt (predtrénovaný na COCO) nepozná triedy "player"/"ball"/"referee".
# Pre reálne použitie potrebuješ buď:
#   1) fine-tuning na futbalových dátach (odporúčané: SoccerNet, https://www.soccer-net.org/)
#   2) alebo dočasne mapovať COCO triedu "person" -> player a "sports ball" -> ball
#      (menej presné, ale funguje ako rýchly prototyp)
