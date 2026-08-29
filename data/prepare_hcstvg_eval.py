#!/usr/bin/env python3
"""Create HC-STVG v1/v2 JSONL annotations for the released lmms-eval task."""

import argparse
import json
import math
from pathlib import Path


def sample_frames(frame_count, fps):
    count = int(frame_count / fps * 2.0)
    count = min(max(count, 4), 64, frame_count)
    if count <= 1:
        return [0]
    return [round(index * (frame_count - 1) / (count - 1)) for index in range(count)]


def read_video(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video metadata: {path}")
    return frame_count, fps


def normalize_box(box, width, height):
    x, y, box_width, box_height = [float(value) for value in box]
    x1, y1 = max(0.0, min(x, width)), max(0.0, min(y, height))
    x2 = max(0.0, min(x + box_width, width))
    y2 = max(0.0, min(y + box_height, height))
    return [
        round(x1 / width * 1000),
        round(y1 / height * 1000),
        round(x2 / width * 1000),
        round(y2 / height * 1000),
    ]


def repaired_trajectory(trajectory):
    parsed = []
    valid = []
    for index, box in enumerate(trajectory):
        values = (
            [float(value) for value in box]
            if isinstance(box, (list, tuple)) and len(box) == 4
            else None
        )
        if values is not None and all(math.isfinite(value) for value in values) and values[2] >= 0 and values[3] >= 0:
            parsed.append(values)
            valid.append(index)
        else:
            parsed.append(None)
    if not valid:
        raise ValueError("Trajectory has no valid boxes.")

    repaired = []
    for index, box in enumerate(parsed):
        if box is not None:
            repaired.append(box)
            continue
        left = next((candidate for candidate in reversed(valid) if candidate < index), None)
        right = next((candidate for candidate in valid if candidate > index), None)
        if left is None:
            repaired.append(list(parsed[right]))
        elif right is None:
            repaired.append(list(parsed[left]))
        else:
            alpha = (index - left) / (right - left)
            repaired.append(
                [
                    first + alpha * (second - first)
                    for first, second in zip(parsed[left], parsed[right])
                ]
            )
    return repaired


def load_video_parts(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as handle:
        parts = json.load(handle)
    return {
        key: str(Path(part) / name)
        for part, names in parts.items()
        for name in names
        for key in (name, Path(name).name, Path(name).stem)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, choices=("v1", "v2"))
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--video-parts")
    parser.add_argument("--video-prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(args.annotations, encoding="utf-8") as handle:
        annotations = json.load(handle)
    parts = load_video_parts(args.video_parts)
    rows = []
    skipped = 0
    for video_name, sample in annotations.items():
        mapped = parts.get(video_name) or parts.get(Path(video_name).name) or parts.get(Path(video_name).stem)
        relative_video = Path(mapped or video_name)
        try:
            frame_count, fps = read_video(Path(args.video_root) / relative_video)
            trajectory = repaired_trajectory(sample["bbox"])
        except (FileNotFoundError, RuntimeError, TypeError, ValueError):
            skipped += 1
            continue

        annotation_frames = int(sample["img_num"])
        tube_start = int(sample["st_frame"])
        tube_end = tube_start + len(trajectory) - 1
        if not 1 <= tube_start <= tube_end <= annotation_frames:
            raise ValueError(f"Invalid annotation tube for {video_name}")
        if sample.get("ed_frame") is not None and int(sample["ed_frame"]) != tube_end:
            raise ValueError(f"ed_frame mismatch for {video_name}")
        width = float(sample.get("width") or sample["img_size"][1])
        height = float(sample.get("height") or sample["img_size"][0])

        decoded_track = {}
        for box_index, box in enumerate(trajectory):
            annotation_frame = tube_start + box_index
            position = (
                0.0
                if annotation_frames == 1 or frame_count == 1
                else (annotation_frame - 1) * (frame_count - 1) / (annotation_frames - 1)
            )
            video_frame = max(0, min(round(position), frame_count - 1))
            item = {
                "bbox": normalize_box(box, width, height),
                "annotation_frame": annotation_frame,
                "distance": abs(video_frame - position),
            }
            if video_frame not in decoded_track or item["distance"] < decoded_track[video_frame]["distance"]:
                decoded_track[video_frame] = item

        boxes = []
        for sampled_index, video_frame in enumerate(sample_frames(frame_count, fps), start=1):
            if video_frame in decoded_track:
                boxes.append(
                    {
                        "time_index": sampled_index,
                        "bbox": decoded_track[video_frame]["bbox"],
                        "source_frame": video_frame,
                        "annotation_frame": decoded_track[video_frame]["annotation_frame"],
                    }
                )
        if not boxes:
            skipped += 1
            continue

        caption = str(sample.get("caption") or sample.get("English") or "").strip()
        if not caption:
            skipped += 1
            continue
        rows.append(
            {
                "id": f"hcstvg_{args.version}_{Path(video_name).stem}",
                "version": args.version,
                "video_path": str(Path(args.video_prefix) / relative_video),
                "caption": caption,
                "fps": fps,
                "frame_count": frame_count,
                "video_start_sec": 0.0,
                "video_end_sec": (frame_count - 1) / fps,
                "gt_sampled_frame_boxes": boxes,
                "sample_fps": 2.0,
                "max_sampled_frames": 64,
                "temporal_patch_size": 1,
            }
        )
        if args.limit and len(rows) >= args.limit:
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with output.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} {args.version} rows to {output}; skipped {skipped}")


if __name__ == "__main__":
    main()
