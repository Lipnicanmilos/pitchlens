"""
Homography – mapovanie súradníc z obrazu kamery na 2D pôdorys ihriska.

MVP prístup: manuálne označíš 4+ referenčné body na obraze (napr. rohy
šestnástky, stredová čiara) a zodpovedajúce body na reálnom pôdoryse ihriska
(v metroch). Pre statickú kameru stačí spočítať homography raz.

Pre TV vysielanie s meniacimi sa zábermi by bolo potrebné počítať homography
priebežne (napr. detekciou čiar ihriska cez segmentáciu) – to je pokročilejšia
úloha nad rámec MVP, pozri poznámku na konci súboru.
"""

from typing import Tuple

import cv2
import numpy as np


class PitchHomography:
    def __init__(self, pitch_length_m: float = 105.0, pitch_width_m: float = 68.0):
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.H = None

    def calibrate(self, image_points: np.ndarray, pitch_points: np.ndarray):
        """
        image_points: Nx2 pole pixelových súradníc referenčných bodov na obraze
        pitch_points: Nx2 pole zodpovedajúcich súradníc na pôdoryse ihriska (v metroch)

        Potrebuješ minimálne 4 korešpondujúce body (napr. rohy ihriska,
        priesečníky pokutového územia so stredovou čiarou a pod.)
        """
        image_points = np.array(image_points, dtype=np.float32)
        pitch_points = np.array(pitch_points, dtype=np.float32)

        self.H, _ = cv2.findHomography(image_points, pitch_points, method=cv2.RANSAC)
        return self.H

    def image_to_pitch(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Prepočíta bod z obrazových súradníc (pixely) na súradnice ihriska (metre)."""
        if self.H is None:
            raise RuntimeError("Homography nie je nakalibrovaná. Zavolaj najprv calibrate().")

        px, py = point
        point_h = np.array([px, py, 1.0])
        mapped = self.H @ point_h
        mapped /= mapped[2]

        return float(mapped[0]), float(mapped[1])


# Referenčné body typického futbalového ihriska (v metroch, stred = [0, 0]):
# Rohy ihriska: (-52.5, -34), (52.5, -34), (52.5, 34), (-52.5, 34)
# Tieto hodnoty použi ako pitch_points pri kalibrácii.

# POZNÁMKA k pokročilejšiemu prístupu:
# Pre automatickú kalibráciu (bez manuálneho klikania bodov) sa používa
# detekcia čiar ihriska pomocou segmentačného modelu (napr. natrénovaného
# na SoccerNet-Calibration datasete) a následné fitovanie homography na
# tieto čiary. Toto je vhodné doplniť až po funkčnom MVP s manuálnou kalibráciou.
