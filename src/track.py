"""
Multi-object tracking pomocou ByteTrack (implementácia z knižnice `supervision`).

Udržuje stabilné track_id pre každého hráča naprieč snímkami, aj cez
krátkodobé okluzie (napr. súboj o loptu).
"""

from typing import List

import numpy as np
import supervision as sv

from detect import Detection


class Tracker:
    def __init__(self, track_thresh: float = 0.5, match_thresh: float = 0.8, track_buffer: int = 30):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            minimum_matching_threshold=match_thresh,
            lost_track_buffer=track_buffer,
        )

    def update(self, detections: List[Detection]) -> sv.Detections:
        """
        Vezme detekcie z jedného snímku a vráti detekcie obohatené o track_id.
        """
        if not detections:
            empty = sv.Detections.empty()
            return self.tracker.update_with_detections(empty)

        xyxy = np.array([d.xyxy for d in detections])
        confidence = np.array([d.confidence for d in detections])
        class_id = np.array([d.class_id for d in detections])

        sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked = self.tracker.update_with_detections(sv_detections)

        return tracked


# Tip: track_id z tohto modulu je len "dočasná" identita v rámci videa.
# Pre skutočné priradenie k menu/číslu hráča potrebuješ modul na rozpoznanie
# čísla dresu (Fáza 3) alebo manuálne mapovanie track_id -> hráč pri prvom výskyte.
