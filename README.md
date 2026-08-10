# PitchLens

Aplikácia na analýzu videa futbalového zápasu pomocou computer vision:
detekcia a tracking hráčov, detekcia lopty, mapovanie na pôdorys ihriska,
odhad orientácie tela hráča a základné pohybové metriky (rýchlosť, dráha, heatmapy).

## Rozsah projektu (fázovaný prístup)

### Fáza 1 – Detekcia a tracking (MVP)
- [ ] Detekcia hráčov, rozhodcu a lopty (YOLOv8/YOLOv11)
- [ ] Multi-object tracking naprieč snímkami (ByteTrack)
- [ ] Vizualizácia bounding boxov + trajektórií na výstupnom videu

### Fáza 2 – Priestorové mapovanie
- [ ] Detekcia čiar ihriska
- [ ] Homography (mapovanie z obrazu kamery na 2D pôdorys ihriska)
- [ ] Prepočet pozícií hráčov/lopty na súradnice ihriska

### Fáza 3 – Identifikácia hráčov
- [ ] Detekcia a rozpoznanie čísla dresu
- [ ] Re-identifikácia hráča naprieč snímkami/kamerami

### Fáza 4 – Pohyb a orientácia tela
- [ ] Pose estimation (MediaPipe/MMPose) – orientácia tela/hlavy
- [ ] Výpočet rýchlosti, zrýchlenia, prejdenej vzdialenosti
- [ ] Heatmapy pozícií hráčov

> Poznámka: skutočný eye-gaze tracking (kam sa hráč presne pozerá očami)
> nie je z bežného TV záberu realisticky dosiahnuteľný. Reálne dosiahnuteľné
> je len odhad orientácie tela/hlavy.

## Štruktúra projektu

```
pitchlens/
├── src/
│   ├── detect.py        # YOLO detekcia hráčov/lopty
│   ├── track.py          # ByteTrack tracking
│   ├── homography.py     # Mapovanie na pôdorys ihriska
│   ├── pose.py            # Pose estimation (orientácia tela)
│   ├── metrics.py        # Rýchlosť, dráha, heatmapy
│   ├── pipeline.py       # Hlavný pipeline spájajúci všetky kroky
│   └── utils.py
├── data/
│   ├── raw/               # Vstupné videá (negitované)
│   └── processed/         # Spracované/anotované dáta
├── models/                 # Natrénované/stiahnuté váhy modelov (negitované)
├── notebooks/               # Experimentálne Jupyter notebooky
├── tests/
├── outputs/                 # Výstupné videá, JSON dáta (negitované)
├── config.yaml
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Stiahni si predtrénované YOLO váhy (napr. `yolov8n.pt`) do `models/`,
alebo fine-tuni vlastný model na dátach zo SoccerNet (https://www.soccer-net.org/).

## Spustenie MVP (detekcia + tracking)

```bash
python src/pipeline.py --video data/raw/sample_clip.mp4 --output outputs/annotated.mp4
```

## Odporúčaný dátový zdroj

[SoccerNet](https://www.soccer-net.org/) – voľne dostupný dataset s anotáciami
na detekciu, tracking, kalibráciu kamery a re-identifikáciu hráčov.

## Tech stack

- Python 3.10+
- OpenCV
- Ultralytics YOLOv8
- ByteTrack (cez `supervision` alebo `boxmot` knižnicu)
- MediaPipe / MMPose
- NumPy / Pandas pre metriky

## Roadmapa (realistický odhad)

| Fáza | Rozsah | Odhad času |
|---|---|---|
| MVP | Detekcia + tracking na 1 statickom klipe | 3–6 týždňov |
| Funkčná appka | + homography, čísla dresov | 3–6 mesiacov |
| Produkčná úroveň | Všetko, vysoká spoľahlivosť | 1+ rok |
