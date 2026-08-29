#!/usr/bin/env python3
"""Create the VidSTG JSONL consumed by the released lmms-eval task."""

import argparse
import json
import math
from pathlib import Path


def sampled_frames(start, end, fps):
    count = int((end - start + 1) / fps * 2.0)
    count = min(max(count, 4), 64, end - start + 1)
    if count <= 1:
        return [start]
    return [round(start + index * (end - start) / (count - 1)) for index in range(count)]


def normalize_box(box, width, height):
    x1, y1, x2, y2 = [float(value) for value in box]
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    return [
        round(x1 / width * 1000),
        round(y1 / height * 1000),
        round(x2 / width * 1000),
        round(y2 / height * 1000),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-prefix", default="vidstg/video")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(args.annotations, encoding="utf-8") as handle:
        data = json.load(handle)
    rows = []
    skipped = 0
    for sample in data["videos"]:
        video_id = sample["original_video_id"]
        track = data["trajectories"].get(video_id, {}).get(str(sample["target_id"]))
        if not track:
            skipped += 1
            continue

        source_frames = int(sample["frame_count"])
        fps = float(sample["fps"])
        used_start = int(sample.get("start_frame", 0))
        used_end = int(sample.get("end_frame", source_frames - 1))
        if used_start < 0 or used_end < used_start or used_end >= source_frames:
            skipped += 1
            continue

        tube_start = int(sample["tube_start_frame"])
        tube_end = int(sample["tube_end_frame"])
        boxes = []
        for sampled_index, source_frame in enumerate(
            sampled_frames(used_start, used_end, fps), start=1
        ):
            item = track.get(str(source_frame))
            if tube_start <= source_frame < tube_end and item is not None:
                boxes.append(
                    {
                        "time_index": sampled_index,
                        "bbox": normalize_box(
                            item["bbox"], float(sample["width"]), float(sample["height"])
                        ),
                        "source_frame": source_frame,
                    }
                )

        video_start = video_end = None
        if used_start != 0 or used_end != source_frames - 1:
            video_start = (
                0.0
                if used_start == 0
                else math.nextafter(used_start / fps, -math.inf)
            )
            video_end = math.nextafter(used_end / fps, math.inf)
        rows.append(
            {
                "id": f"{video_id}_{sample['video_id']}",
                "video_path": str(Path(args.video_prefix) / sample["video_path"]),
                "caption": sample["caption"].strip(),
                "qtype": sample.get("qtype"),
                "fps": fps,
                "frame_count": used_end - used_start + 1,
                "video_start_sec": video_start,
                "video_end_sec": video_end,
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
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} rows to {output}; skipped {skipped}")


if __name__ == "__main__":
    main()
