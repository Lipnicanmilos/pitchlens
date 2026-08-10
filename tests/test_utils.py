import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import bbox_center, bbox_foot_point


def test_bbox_center():
    xyxy = np.array([0, 0, 10, 20])
    cx, cy = bbox_center(xyxy)
    assert cx == 5
    assert cy == 10


def test_bbox_foot_point():
    xyxy = np.array([0, 0, 10, 20])
    fx, fy = bbox_foot_point(xyxy)
    assert fx == 5
    assert fy == 20
