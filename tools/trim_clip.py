"""
Vyreže úsek z videa bez ffmpeg – len cez OpenCV, ktorý už v projekte je.

Celý zápas sa na CPU spracúva hodiny (viď Zistenia v README), takže na overenie,
či detekcia vôbec chytá hráčov, si najprv vyrež minútu hry.

Použitie:
    python tools/trim_clip.py <vstup> <výstup> <štart_v_sekundách> <dĺžka_v_sekundách>

Príklad – minúta od 13. minúty zápasu:
    python tools/trim_clip.py data/raw/zapas.mp4 data/raw/clip_test.mp4 790 60
"""

import argparse
from pathlib import Path

import cv2


def trim(src: str, dst: str, start_s: float, duration_s: float) -> None:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Nepodarilo sa otvoriť video: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_s * fps)
    if start_frame >= total:
        raise SystemExit(
            f"Štart {start_s} s je za koncom videa (dĺžka {total / fps:.1f} s)"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = cv2.VideoWriter(
        dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise SystemExit(f"Nepodarilo sa otvoriť VideoWriter pre: {dst}")

    written = 0
    for _ in range(int(duration_s * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

    writer.release()
    cap.release()

    print(
        f"OK: {Path(dst).name} – {written} snímkov, {width}x{height} @ {fps:.2f} fps "
        f"({written / fps:.1f} s od {start_s:.0f}. sekundy)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vyreže úsek z videa (bez ffmpeg)")
    parser.add_argument("input", help="Cesta k vstupnému videu")
    parser.add_argument("output", help="Cesta k výstupnému klipu")
    parser.add_argument("start", type=float, help="Začiatok úseku v sekundách")
    parser.add_argument("duration", type=float, help="Dĺžka úseku v sekundách")
    args = parser.parse_args()

    trim(args.input, args.output, args.start, args.duration)
