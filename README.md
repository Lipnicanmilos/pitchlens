# PitchLens

Aplikácia na analýzu videa futbalového zápasu pomocou computer vision:
detekcia a tracking hráčov, detekcia lopty, mapovanie na pôdorys ihriska,
odhad orientácie tela hráča a základné pohybové metriky (rýchlosť, dráha, heatmapy).

## Rozsah projektu (fázovaný prístup)

### Fáza 1 – Detekcia a tracking (MVP)
- [x] Detekcia hráčov (COCO `person` → player, viď Mapovanie tried nižšie)
- [ ] Detekcia lopty – **nefunguje**, COCO `sports ball` ju na týchto záberoch nenájde
- [ ] Detekcia rozhodcu – COCO ho neodlíši od hráča, treba fine-tuning
- [x] Multi-object tracking naprieč snímkami (ByteTrack)
- [x] Vizualizácia bounding boxov + trajektórií na výstupnom videu

### Fáza 2 – Priestorové mapovanie
- [ ] Detekcia čiar ihriska – **netreba**, ak je kamera statická (viď Zistenia)
- [ ] Homography (mapovanie z obrazu kamery na 2D pôdorys ihriska)
- [ ] Prepočet pozícií hráčov/lopty na súradnice ihriska

### Fáza 3 – Identifikácia hráčov
- [ ] Rozlíšenie tímov podľa farby dresu (k-means na výreze hráča)
- [ ] Detekcia a rozpoznanie čísla dresu
- [ ] Re-identifikácia hráča naprieč snímkami/kamerami

### Fáza 4 – Pohyb a orientácia tela
- [ ] Pose estimation (MediaPipe/MMPose) – orientácia tela/hlavy
- [ ] Výpočet rýchlosti, zrýchlenia, prejdenej vzdialenosti
- [ ] Heatmapy pozícií hráčov

> Poznámka: skutočný eye-gaze tracking (kam sa hráč presne pozerá očami)
> nie je z bežného TV záberu realisticky dosiahnuteľný. Reálne dosiahnuteľné
> je len odhad orientácie tela/hlavy.

## Zistenia z testov na reálnom videu (2026-08-10)

MVP prebehlo na dvoch reálnych záznamoch. Namerané čísla, nie odhady.

| | halový zápas 640×360 | letecký záber 960×540 |
|---|---|---|
| dĺžka testu | 60 s (1798 snímkov) | 20 s (502 snímkov) |
| výška hráča – medián | **53 px** (p90 78, max 105) | 35 px (p90 41, max 79) |
| detekcií na snímok | 7,4 priemer | 10 medián (~⅓ hráčov chýba) |
| detekcie lopty | 0 | 0 |
| unikátnych `track_id` | 145 na ~13 ľudí | **38 na ~23 ľudí** |
| kamera | **statická** | pohyblivá (dron) |
| čas spracovania | 227 ms/snímok | 336 ms/snímok |

### Čo z toho vyplýva

**Tracking je hlavná prekážka, nie detekcia.** Detekcia hráčov na ihrisku funguje
slušne, ale 145 identít na 13 ľudí za minútu znamená, že medián života jedného
`track_id` je 1,4 sekundy. Kým sa to nevyrieši, Fáza 4 (nabehané metre, heatmapy)
je nedosiahnuteľná – nedá sa sčítať dráha hráča, ktorý je 145 rôznych identít.
Príčiny: `track_buffer: 30` je pri 30 fps len sekunda pamäte, a ByteTrack
rozhoduje výhradne podľa pohybu, bez akéhokoľvek vzhľadového modelu.

**Uhol kamery ovplyvňuje tracking viac než rozlíšenie.** Letecký záber má
horšie rozlíšenie hráča, ale trikrát lepší tracking – z vtáčej perspektívy sa
hráči neprekrývajú, takže tracker nemá kde stratiť niť.

**Čísla dresov sú hranične čitateľné pri ~100 px výšky hráča**, čo je na
halovom zázname p90–max, nie bežný prípad (medián 53 px). OCR preto nemá zmysel
púšťať po snímkoch – správny prístup je prečítať číslo **raz za track** a zbierať
hlasy cez celý jeho život. To ale opäť predpokladá funkčný tracking.

**Statická kamera zjednodušuje Fázu 2 z týždňov na odpoludnie** – homography sa
spočíta raz ručným označením 4 bodov a platí pre celý zápas. Detekcia čiar
ihriska je potrebná len pri pohyblivej kamere.

**Výkon:** bez GPU (`torch+cpu`) je spracovanie 5–7× pomalšie než realtime.
Halový zápas (37,7 min) ≈ 4,3 hodiny. Riešenie je spracovávať každý N-tý snímok –
na tracking hráčov stačí 5–10 fps, nie 25.

### Priorita ďalších krokov

1. **Tracking** – zdvihnúť `track_buffer`, prípadne prejsť na tracker s ReID
   (`boxmot` – BoT-SORT / DeepOCSORT). Blokuje Fázu 3 aj 4.
2. **Rozlíšenie tímov podľa farby dresu** – lacné, a zároveň slúži ako ReID
   signál (červený track sa nesmie spojiť so zeleným).
3. **Fine-tuning na SoccerNet** – jediná cesta k detekcii lopty a rozhodcu.
4. **Homography** – až po trackingu, ale pri statickej kamere je to rýchle.

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
├── tools/
│   └── trim_clip.py      # Vyreže testovací klip z dlhého videa (bez ffmpeg)
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

Stiahni si predtrénované YOLO váhy do `models/`:

```bash
curl -L -o models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
```

### Mapovanie tried (dôležité)

`yolov8n.pt` je predtrénovaný na COCO a triedy `player`/`ball`/`referee`
nepozná — COCO trieda 0 je `person` a 32 je `sports ball`. Pipeline preto
prekladá COCO triedy na projektové cez `detection.coco_fallback.map`
v `config.yaml` a všetko ostatné zahadzuje.

Dôsledky tohto provizória:
- **rozhodca sa nerozlíši od hráča** – COCO to nevie, všetky osoby padnú do `player`
- **diváci v hľadisku sú tiež `person`** – čiastočne ich odreže `min_box_height`,
  ale spoľahlivé riešenie je až fine-tuning

Keď si model fine-tuneš na futbalových dátach (SoccerNet,
https://www.soccer-net.org/), nastav `coco_fallback.enabled: false` a model
bude vracať projektové triedy priamo.

## Spustenie MVP (detekcia + tracking)

Vstupné videá patria do `data/raw/` (negitované). Z dlhého záznamu si najprv
vyrež krátky testovací klip – celý zápas beží na CPU hodiny:

```bash
python tools/trim_clip.py data/raw/zapas.mp4 data/raw/clip_test.mp4 790 60
```

```bash
python src/pipeline.py --video data/raw/clip_test.mp4 --output outputs/annotated.mp4
```

Výstupom je anotované video a `.json` so všetkými detekciami po snímkoch
(`track_id`, `class_id`, `bbox`).

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
