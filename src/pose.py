"""
Pose estimation pomocou MediaPipe – odhad orientácie tela/hlavy hráča.

DÔLEŽITÉ OBMEDZENIE:
Toto NEDÁVA presný smer pohľadu očí (eye gaze). Z bežného TV záberu, kde je
hráč vzdialený desiatky metrov od kamery, nie je možné spoľahlivo určiť smer
pohľadu očí. Realisticky dosiahnuteľné je len:
  - orientácia trupu/ramien (kam je hráč natočený telom)
  - orientácia hlavy (ak je rozlíšenie dostatočné)
Toto sa dá použiť ako približný proxy pre "kam sa hráč pravdepodobne pozerá",
nie ako presný gaze tracking.
"""

from dataclasses import dataclass
from typing import Optional

import mediapipe as mp
import numpy as np


@dataclass
class PoseResult:
    landmarks: Optional[np.ndarray]  # Nx3 (x, y, z) normalizované súradnice
    body_orientation_deg: Optional[float]  # odhadovaný smer natočenia trupu v stupňoch


class PoseEstimator:
    def __init__(self, model_complexity: int = 1):
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                f"mediapipe {mp.__version__} odstránilo legacy `mp.solutions` API, "
                "na ktorom stojí tento modul (v 1.0 zostalo len Image/ImageFormat/tasks).\n"
                "Možnosti:\n"
                "  a) pip install 'mediapipe<1.0' — rýchle, ale legacy API už Google neudržiava\n"
                "  b) prepísať tento modul na Tasks API "
                "(mediapipe.tasks.python.vision.PoseLandmarker), čo vyžaduje stiahnuť\n"
                "     model pose_landmarker_*.task\n"
                "Dovtedy nechaj v config.yaml `pose.enabled: false` — MVP (Fáza 1) "
                "pose nepotrebuje."
            )

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def estimate(self, player_crop: np.ndarray) -> PoseResult:
        """
        player_crop: výrez obrazu okolo jedného hráča (BGR, np.ndarray)
        Vráti landmarky a hrubý odhad orientácie trupu.
        """
        # BGR -> RGB. ascontiguousarray je nutné: `[:, :, ::-1]` vráti pohľad so
        # záporným krokom a MediaPipe vyžaduje súvislé pole (inak spadne na
        # "Image data must be contiguous").
        rgb = np.ascontiguousarray(player_crop[:, :, ::-1])
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return PoseResult(landmarks=None, body_orientation_deg=None)

        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
        )

        orientation = self._estimate_body_orientation(landmarks)
        return PoseResult(landmarks=landmarks, body_orientation_deg=orientation)

    def _estimate_body_orientation(self, landmarks: np.ndarray) -> Optional[float]:
        """
        Hrubý odhad natočenia trupu na základe pozície ramien.
        Index 11 = ľavé rameno, 12 = pravé rameno (MediaPipe Pose formát).
        """
        try:
            left_shoulder = landmarks[11][:2]
            right_shoulder = landmarks[12][:2]
        except IndexError:
            return None

        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]
        angle_rad = np.arctan2(dy, dx)

        return float(np.degrees(angle_rad))

    def close(self):
        self.pose.close()
