"""Pomocné funkcie zdieľané naprieč modulmi."""

import cv2
import numpy as np


def bbox_center(xyxy: np.ndarray) -> tuple:
    """Vráti stred bounding boxu (x, y)."""
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2


def bbox_foot_point(xyxy: np.ndarray) -> tuple:
    """
    Vráti bod 'nôh' hráča (stred spodnej hrany bboxu) - vhodnejšie ako stred
    bboxu pre mapovanie pozície na pôdorys ihriska (hráč stojí na zemi nohami,
    nie v strede vlastného tela).
    """
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2, y2


def draw_text(frame: np.ndarray, text: str, position: tuple, color=(255, 255, 255)):
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
    )
    return frame
