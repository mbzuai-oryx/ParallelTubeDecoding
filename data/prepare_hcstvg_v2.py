#!/usr/bin/env python3
"""Prepare HC-STVG v2 annotations for PTD SFT."""

import argparse
import json
from pathlib import Path


PROMPT = (
    "Localize the described object throughout the video. Use object reference "
    "tokens, time tokens, and box tokens. Return the object reference, event "
    "time segment, and per-time bbox coordinates."
)


def read_video(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frame_count <= 0 or fps <= 0:
        raise ValueError(f"Invalid video metadata: {path}")
    return frame_count, fps


def sample_frames(frame_count, fps):
    count = int(frame_count / fps * 2.0)
    count = min(max(count, 4), 64, frame_count)
    if count <= 1:
        return [0]
    return [round(index * (frame_count - 1) / (count - 1)) for index in range(count)]


def annotation_to_video_frame(frame, annotation_frames, video_frames):
    if not 1 <= frame <= annotation_frames:
        raise ValueError(f"Annotation frame {frame} is outside [1, {annotation_frames}].")
    if annotation_frames == 1 or video_frames == 1:
        return 0
    return round((frame - 1) * (video_frames - 1) / (annotation_frames - 1))


def video_to_annotation_frame(frame, annotation_frames, video_frames):
    if not 0 <= frame < video_frames:
        raise ValueError(f"Video frame {frame} is outside [0, {video_frames - 1}].")
    if annotation_frames == 1 or video_frames == 1:
        return 1
    return round(frame * (annotation_frames - 1) / (video_frames - 1)) + 1


def normalize_box(box, width, height):
    x, y, box_width, box_height = [float(value) for value in box]
    x1 = max(0.0, min(x, width))
    y1 = max(0.0, min(y, height))
    x2 = max(0.0, min(x + box_width, width))
    y2 = max(0.0, min(y + box_height, height))
    normalized = [
        round(x1 / width * 1000),
        round(y1 / height * 1000),
        round(x2 / width * 1000),
        round(y2 / height * 1000),
    ]
    if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
        return None
    return normalized


def build_record(video, caption, boxes):
    if not boxes:
        return None
    indices = [item[0] for item in boxes]
    if indices != list(range(indices[0], indices[-1] + 1)):
        return None
    query = caption if caption.endswith((".", "?", "!")) else f"{caption}."
    answer = [f"<|object_ref_start|>{caption}<|object_ref_end|>"]
    answer.append(f"<|time_start|><t{indices[0]}><t{indices[-1]}><|time_end|>")
    for time_index, box in boxes:
        coordinates = "".join(
            f"<{max(0, min(int(value), 1000))}>" for value in box
        )
        answer.append(f"<t{time_index}><|box_start|>{coordinates}<|box_end|>")
    return {
        "video": video,
        "mtp_format": "time_anchored_boxes",
        "conversations": [
            {
                "from": "human",
                "value": f"<video>\nGiven the query: '{query}' {PROMPT}",
            },
            {"from": "gpt", "value": "\n".join(answer)},
        ],
    }


def load_video_parts(path):
    if not path:
        return {}
    with open(path) as handle:
        parts = json.load(handle)
    lookup = {}
    for part, names in parts.items():
        for name in names:
            relative = str(Path(part) / name)
            lookup[name] = relative
            lookup[Path(name).name] = relative
            lookup.setdefault(Path(name).stem, relative)
    return lookup


def relative_video_path(video_name, parts):
    path = Path(video_name)
    mapped = parts.get(video_name) or parts.get(path.name) or parts.get(path.stem)
    return Path(mapped) if mapped else path


def prepare(args):
    with open(args.annotations) as handle:
        annotations = json.load(handle)
    parts = load_video_parts(args.video_parts)
    records = []
    skipped = 0
    for video_name, sample in annotations.items():
        if args.limit and len(records) >= args.limit:
            break
        relative_video = relative_video_path(video_name, parts)
        try:
            frame_count, fps = read_video(Path(args.video_root) / relative_video)
        except (FileNotFoundError, RuntimeError, ValueError):
            skipped += 1
            continue

        trajectory = sample.get("bbox", [])
        caption = str(sample.get("caption") or sample.get("English") or "").strip()
        annotation_frames = int(sample["img_num"])
        tube_start = int(sample["st_frame"])
        tube_end = int(sample["ed_frame"])
        width = float(sample.get("width") or sample["img_size"][1])
        height = float(sample.get("height") or sample["img_size"][0])
        if (
            not trajectory
            or not caption
            or annotation_frames <= 0
        ):
            skipped += 1
            continue

        tube_start_video = annotation_to_video_frame(
            tube_start, annotation_frames, frame_count
        )
        tube_end_video = annotation_to_video_frame(
            tube_end, annotation_frames, frame_count
        )
        boxes = []
        for time_index, video_frame in enumerate(sample_frames(frame_count, fps), start=1):
            if not tube_start_video <= video_frame <= tube_end_video:
                continue
            annotation_frame = video_to_annotation_frame(
                video_frame, annotation_frames, frame_count
            )
            if not tube_start <= annotation_frame <= tube_end:
                continue
            box_index = annotation_frame - tube_start
            if not 0 <= box_index < len(trajectory):
                continue
            box = normalize_box(trajectory[box_index], width, height)
            if box is None:
                boxes = []
                break
            boxes.append((time_index, box))

        record = build_record(
            str(Path(args.video_prefix) / relative_video), caption, boxes
        )
        if record is None:
            skipped += 1
        else:
            records.append(record)
    return records, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--video-parts")
    parser.add_argument(
        "--video-prefix", default="hc-stvg_v2/v2_videos_train/video_parts"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    records, skipped = prepare(args)
    output_path = Path(args.output)
    if args.append and output_path.exists():
        with output_path.open() as handle:
            records = json.load(handle) + records
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} samples to {output_path}; skipped {skipped}")


if __name__ == "__main__":
    main()
