"""
Detekcia hráčov, rozhodcu a lopty pomocou YOLOv8.

Použitie:
    from detect import Detector
    detector = Detector.from_config(config["detection"])
    detections = detector.detect(frame)

POZNÁMKA K TRIEDAM:
Predtrénovaný `yolov8n.pt` je natrénovaný na COCO a triedy "player"/"ball"/
"referee" vôbec nepozná — COCO trieda 0 je "person" a 32 je "sports ball".
Preto sa detekcie prekladajú cez `coco_fallback.map` z config.yaml na projektové
class_id. Až keď si model fine-tuneš na futbalových dátach (SoccerNet,
https://www.soccer-net.org/), nastav `coco_fallback.enabled: false` a model
bude vracať naše triedy priamo.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    xyxy: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


class Detector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.35,
        class_map: Optional[Dict[int, int]] = None,
        class_names: Optional[Dict[int, str]] = None,
        min_box_height: float = 0.0,
    ):
        """
        class_map:  {class_id modelu -> projektové class_id}. Ak je None, triedy
                    modelu sa berú tak, ako sú (fine-tunovaný model).
                    Detekcie tried, ktoré v mape nie sú, sa zahadzujú.
        class_names: {projektové class_id -> názov} pre čitateľné labely.
        min_box_height: zahodí boxy nižšie ako N pixelov (odreže divákov v hľadisku).
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.class_map = class_map
        self.class_names = class_names or {}
        self.min_box_height = min_box_height

    @classmethod
    def from_config(cls, detection_config: dict) -> "Detector":
        """Poskladá Detector z `detection` sekcie config.yaml."""
        # {"player": 0, "ball": 1} -> {0: "player", 1: "ball"}
        class_names = {v: k for k, v in detection_config.get("classes", {}).items()}

        fallback = detection_config.get("coco_fallback", {})
        class_map = None
        min_box_height = 0.0
        if fallback.get("enabled"):
            # YAML kľúče môžu prísť ako string, preto pretypovanie na int
            class_map = {int(k): int(v) for k, v in fallback.get("map", {}).items()}
            min_box_height = float(fallback.get("min_box_height", 0.0))

        return cls(
            model_path=detection_config["model_path"],
            confidence=detection_config["confidence"],
            class_map=class_map,
            class_names=class_names,
            min_box_height=min_box_height,
        )

    def _to_detection(self, box, model_names: dict) -> Optional[Detection]:
        """Prevedie jeden YOLO box na Detection, alebo None ak ho treba zahodiť."""
        source_cls_id = int(box.cls[0])

        if self.class_map is not None:
            if source_cls_id not in self.class_map:
                return None  # trieda, ktorá nás nezaujíma (auto, lavička, ...)
            cls_id = self.class_map[source_cls_id]
        else:
            cls_id = source_cls_id

        xyxy = box.xyxy[0].cpu().numpy()

        if self.min_box_height > 0 and (xyxy[3] - xyxy[1]) < self.min_box_height:
            return None  # príliš malý box – takmer isto divák v hľadisku

        cls_name = self.class_names.get(cls_id) or model_names.get(cls_id, str(cls_id))

        return Detection(
            xyxy=xyxy,
            confidence=float(box.conf[0]),
            class_id=cls_id,
            class_name=cls_name,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Spustí detekciu na jednom snímku a vráti zoznam detekcií."""
        results = self.model.predict(frame, conf=self.confidence, verbose=False)[0]

        detections = []
        for box in results.boxes:
            detection = self._to_detection(box, results.names)
            if detection is not None:
                detections.append(detection)

        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Spustí detekciu na viacerých snímkoch naraz (efektívnejšie na GPU)."""
        results = self.model.predict(frames, conf=self.confidence, verbose=False)

        all_detections = []
        for result in results:
            frame_detections = []
            for box in result.boxes:
                detection = self._to_detection(box, result.names)
                if detection is not None:
                    frame_detections.append(detection)
            all_detections.append(frame_detections)

        return all_detections
