"""
Výpočet pohybových metrík z trackovaných pozícií hráčov.

Vyžaduje súradnice už prepočítané cez homography (v metroch), inak budú
rýchlosti a vzdialenosti v pixeloch/snímok, čo nemá reálnu fyzikálnu hodnotu.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class PlayerMetrics:
    def __init__(self, fps: float = 25.0):
        self.fps = fps
        # track_id -> list of (frame_idx, x_m, y_m)
        self.history: Dict[int, List[Tuple[int, float, float]]] = {}

    def add_position(self, track_id: int, frame_idx: int, x_m: float, y_m: float):
        self.history.setdefault(track_id, []).append((frame_idx, x_m, y_m))

    def compute_speed_series(self, track_id: int) -> pd.DataFrame:
        """Vráti DataFrame s rýchlosťou (m/s) pre daný track_id v čase."""
        positions = self.history.get(track_id, [])
        if len(positions) < 2:
            return pd.DataFrame(columns=["frame_idx", "x_m", "y_m", "speed_mps"])

        df = pd.DataFrame(positions, columns=["frame_idx", "x_m", "y_m"])
        df = df.sort_values("frame_idx").reset_index(drop=True)

        dt = df["frame_idx"].diff() / self.fps
        dx = df["x_m"].diff()
        dy = df["y_m"].diff()
        dist = np.sqrt(dx**2 + dy**2)

        df["speed_mps"] = dist / dt
        df.loc[0, "speed_mps"] = 0.0

        return df

    def total_distance_m(self, track_id: int) -> float:
        """Celková prejdená vzdialenosť hráča v metroch."""
        df = self.compute_speed_series(track_id)
        if df.empty:
            return 0.0

        dx = df["x_m"].diff().fillna(0)
        dy = df["y_m"].diff().fillna(0)
        return float(np.sqrt(dx**2 + dy**2).sum())

    def max_speed_mps(self, track_id: int) -> float:
        df = self.compute_speed_series(track_id)
        if df.empty:
            return 0.0
        return float(df["speed_mps"].max())

    def heatmap_positions(self, track_id: int) -> np.ndarray:
        """Vráti Nx2 pole pozícií (x_m, y_m) vhodné na vykreslenie heatmapy."""
        positions = self.history.get(track_id, [])
        if not positions:
            return np.empty((0, 2))
        return np.array([[p[1], p[2]] for p in positions])
