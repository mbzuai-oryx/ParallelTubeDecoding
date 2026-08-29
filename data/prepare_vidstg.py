#!/usr/bin/env python3
"""Prepare VidSTG annotations for PTD SFT."""

import argparse
import json
import math
from pathlib import Path

import torch


PROMPT = (
    "Localize the described object throughout the video. Use object reference "
    "tokens, time tokens, and box tokens. Return the object reference, event "
    "time segment, and per-time bbox coordinates."
)


def read_video(path):
    import decord

    reader = decord.VideoReader(str(path), num_threads=1)
    frame_count = len(reader)
    fps = float(reader.get_avg_fps())
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video metadata: {path}")
    return frame_count, fps


def sample_frames(start, end, fps):
    frame_count = end - start + 1
    count = math.floor(frame_count / fps * 2.0)
    count = min(max(count, 4), 64, frame_count)
    return torch.linspace(start, end, count).round().long().tolist()


def normalize_box(box, width, height):
    x1, y1, x2, y2 = [float(value) for value in box]
    x1, x2 = max(0.0, x1), min(width, x2)
    y1, y2 = max(0.0, y1), min(height, y2)
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        return None
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
        coordinates = "".join(f"<{value}>" for value in box)
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


def prepare(args):
    with open(args.annotations) as handle:
        data = json.load(handle)
    if "videos" not in data or "trajectories" not in data:
        raise ValueError("VidSTG annotations must contain videos and trajectories.")

    records = []
    skipped = 0
    for sample in data["videos"]:
        if args.limit and len(records) >= args.limit:
            break
        video_id = sample["original_video_id"]
        track = data["trajectories"].get(video_id, {}).get(str(sample["target_id"]))
        if not track:
            skipped += 1
            continue

        relative_video = Path(sample["video_path"])
        try:
            decoded_frames, decoded_fps = read_video(Path(args.video_root) / relative_video)
        except (FileNotFoundError, RuntimeError, ValueError):
            skipped += 1
            continue

        annotation_fps = float(sample["fps"])
        annotation_frames = int(sample["frame_count"])
        used_start = int(sample.get("start_frame", 0))
        used_end = int(sample.get("end_frame", annotation_frames - 1))
        crop_start = max(0, math.ceil(used_start / annotation_fps * decoded_fps))
        crop_end = min(
            decoded_frames - 1,
            math.floor(used_end / annotation_fps * decoded_fps),
        )
        if crop_start > crop_end:
            skipped += 1
            continue

        boxes = []
        for time_index, decoded_frame in enumerate(
            sample_frames(crop_start, crop_end, decoded_fps), start=1
        ):
            annotation_frame = round(decoded_frame / decoded_fps * annotation_fps)
            item = track.get(str(annotation_frame))
            if not int(sample["tube_start_frame"]) <= annotation_frame < int(
                sample["tube_end_frame"]
            ) or item is None:
                continue
            box = normalize_box(item["bbox"], float(sample["width"]), float(sample["height"]))
            if box is not None:
                boxes.append((time_index, box))

        output_video = str(Path(args.video_prefix) / relative_video)
        if used_start != 0 or used_end != annotation_frames - 1:
            output_video = {
                "video": output_video,
                "video_start": used_start / annotation_fps,
                "video_end": used_end / annotation_fps,
            }
        record = build_record(output_video, sample["caption"].strip(), boxes)
        if record is None:
            skipped += 1
        else:
            records.append(record)
    return records, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--video-prefix", default="vidstg/video")
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
