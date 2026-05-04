"""Склеивает PNG-кадры из папки в одно видео (MP4/AVI).

Usage:
    python tools/png_to_video.py <input_folder> <output_video.mp4> [--fps 30]

Примеры:
    python tools/png_to_video.py dataset/frames output.mp4 --fps 15
    python tools/png_to_video.py dataset/frames output.avi --fps 10 --codec mp4v
"""

import argparse
import os
import cv2
import re


def natural_sort_key(s: str):
    """Сортировка 'frame_1.png', 'frame_2.png', ..., 'frame_10.png' по порядку."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)]


def png_to_video(input_dir: str, output_path: str, fps: int = 30, codec: str = "mp4v"):
    # Собираем все PNG/JPG
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    if not files:
        raise RuntimeError(f"В папке '{input_dir}' нет изображений {valid_exts}")

    files.sort(key=natural_sort_key)
    print(f"Найдено {len(files)} кадров. FPS={fps}. Длительность ~{len(files)/fps:.1f} сек")

    # Определяем размер по первому кадру
    first_frame = cv2.imread(os.path.join(input_dir, files[0]))
    if first_frame is None:
        raise RuntimeError("Не удалось прочитать первый кадр")
    h, w = first_frame.shape[:2]

    # Инициализируем Writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось создать видео '{output_path}'. Попробуй кодек 'avc1' или 'XVID'")

    for i, fname in enumerate(files):
        path = os.path.join(input_dir, fname)
        frame = cv2.imread(path)
        if frame is None:
            print(f"⚠️ Пропуск: {fname} (не прочитан)")
            continue
        # Если размер отличается — ресайзим
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)

        if (i + 1) % 100 == 0 or i == len(files) - 1:
            print(f"  Обработано: {i + 1}/{len(files)}")

    writer.release()
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Готово: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PNG → Video")
    parser.add_argument("input_dir", help="Папка с PNG/JPG кадрами")
    parser.add_argument("output", help="Путь для сохранения видео (например, out.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="FPS выходного видео (по умолчанию 30)")
    parser.add_argument("--codec", type=str, default="mp4v",
                        help="FourCC кодек: mp4v, avc1, XVID, MJPG (по умолчанию mp4v)")
    args = parser.parse_args()

    png_to_video(args.input_dir, args.output, args.fps, args.codec)
