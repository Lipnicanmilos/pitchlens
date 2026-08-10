"""
Hlavný pipeline: video -> detekcia -> tracking -> (voliteľne pose) -> anotované video + JSON.

Použitie:
    python src/pipeline.py --video data/raw/sample_clip.mp4 --output outputs/annotated.mp4

Toto je MVP (Fáza 1 z README): funguje na detekciu + tracking a vykreslenie
výsledkov. Homography a rozpoznanie čísla dresu treba dorobiť v ďalších fázach
(pozri komentáre nižšie, kde sa dajú zapojiť).
"""

import argparse
import json

import cv2
import supervision as sv
import yaml

from detect import Detector
from pose import PoseEstimator
from track import Tracker


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(video_path: str, output_path: str, config_path: str = "config.yaml"):
    config = load_config(config_path)

    detector = Detector(
        model_path=config["detection"]["model_path"],
        confidence=config["detection"]["confidence"],
    )
    tracker = Tracker(
        track_thresh=config["tracking"]["track_thresh"],
        match_thresh=config["tracking"]["match_thresh"],
        track_buffer=config["tracking"]["track_buffer"],
    )

    pose_estimator = None
    if config["pose"]["enabled"]:
        pose_estimator = PoseEstimator(model_complexity=config["pose"]["model_complexity"])

    box_annotator = sv.BoxAnnotator()
    trace_annotator = sv.TraceAnnotator()
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Nepodarilo sa otvoriť video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    json_log = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracked = tracker.update(detections)

        frame_records = []
        if len(tracked) > 0:
            for i in range(len(tracked)):
                x1, y1, x2, y2 = tracked.xyxy[i]
                track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                class_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1

                record = {
                    "track_id": track_id,
                    "class_id": class_id,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                }

                # Voliteľná pose estimation na výreze hráča
                if pose_estimator is not None:
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size > 0:
                        pose_result = pose_estimator.estimate(crop)
                        record["body_orientation_deg"] = pose_result.body_orientation_deg

                # TODO Fáza 2: ak je homography nakalibrovaná, tu prepočítaj
                # stred spodnej hrany bbox-u (nohy hráča) na súradnice ihriska:
                #   foot_point = ((x1 + x2) / 2, y2)
                #   pitch_xy = homography.image_to_pitch(foot_point)

                frame_records.append(record)

        json_log.append({"frame": frame_idx, "detections": frame_records})

        # Vykreslenie
        annotated = frame.copy()
        if config["output"]["draw_boxes"]:
            annotated = box_annotator.annotate(annotated, tracked)
        if config["output"]["draw_tracks"]:
            annotated = trace_annotator.annotate(annotated, tracked)
        if tracked.tracker_id is not None:
            labels = [f"#{tid}" for tid in tracked.tracker_id]
            annotated = label_annotator.annotate(annotated, tracked, labels=labels)

        out.write(annotated)
        frame_idx += 1

    cap.release()
    out.release()
    if pose_estimator is not None:
        pose_estimator.close()

    if config["output"]["save_json"]:
        json_path = output_path.rsplit(".", 1)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_log, f, indent=2)
        print(f"JSON log uložený do: {json_path}")

    print(f"Hotovo. Anotované video uložené do: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football video analysis pipeline (MVP)")
    parser.add_argument("--video", required=True, help="Cesta k vstupnému videu")
    parser.add_argument("--output", required=True, help="Cesta k výstupnému videu")
    parser.add_argument("--config", default="config.yaml", help="Cesta ku config.yaml")
    args = parser.parse_args()

    run_pipeline(args.video, args.output, args.config)
