#!/usr/bin/env python3
"""Create Charades-STA or ActivityNet Captions JSONL for lmms-eval."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("charades-sta", "activitynet"))
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-prefix", default="")
    parser.add_argument("--video-extension", default=".mp4")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    prefix = Path(args.video_prefix) if args.video_prefix else Path()
    rows = []
    if args.benchmark == "charades-sta":
        with open(args.annotations, encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if not line.strip():
                    continue
                video_id, interval_and_query = line.rstrip().split(" ", 1)
                interval, query = interval_and_query.split("##", 1)
                start, end = map(float, interval.split())
                rows.append(
                    {
                        "id": line_index,
                        "video_path": str(prefix / f"{video_id}{args.video_extension}"),
                        "caption": query.strip(),
                        "timestamp": [start, end],
                    }
                )
                if args.limit and len(rows) >= args.limit:
                    break
    else:
        with open(args.annotations, encoding="utf-8") as handle:
            annotations = json.load(handle)
        for video_id, annotation in annotations.items():
            sentences = annotation.get("sentences", [])
            timestamps = annotation.get("timestamps", [])
            if len(sentences) != len(timestamps):
                raise ValueError(f"Sentence/timestamp count mismatch for {video_id}")
            for sentence_index, (query, timestamp) in enumerate(zip(sentences, timestamps)):
                rows.append(
                    {
                        "id": f"{video_id}:{sentence_index}",
                        "video_path": str(prefix / f"{video_id}{args.video_extension}"),
                        "caption": str(query).strip(),
                        "timestamp": [float(timestamp[0]), float(timestamp[1])],
                    }
                )
                if args.limit and len(rows) >= args.limit:
                    break
            if args.limit and len(rows) >= args.limit:
                break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
