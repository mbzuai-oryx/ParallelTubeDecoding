"""lmms-eval helpers for PTD temporal localization on ActivityNet."""

from functools import lru_cache
import math
import os
from pathlib import Path
import re


PROMPT = (
    "Given the query: '{query}' Localize when the described event occurs in the "
    "video. Use object reference tokens and time tokens. Return the event "
    "reference followed by the event time segment."
)
_SEGMENT = re.compile(
    r"<\|time_start\|>\s*<t(?P<start>\d+)>\s*"
    r"<t(?P<end>\d+)>\s*<\|time_end\|>"
)


def _query(doc):
    for field in ("caption", "query", "sentence"):
        if doc.get(field):
            query = str(doc[field]).strip()
            return query if query.endswith((".", "?", "!")) else f"{query}."
    raise KeyError("The annotation must contain caption, query, or sentence.")


def _video_path(doc):
    path = Path(str(doc.get("video_path") or doc["video"])).expanduser()
    root = os.environ.get("ACTIVITYNET_VIDEO_ROOT")
    if not path.is_absolute() and root:
        path = Path(root).expanduser() / path
    return path


def activitynet_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    return [str(_video_path(doc))]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return PROMPT.format(query=_query(doc))


def doc_to_answer(doc):
    return doc["timestamp"]


def _segment(text):
    matches = list(_SEGMENT.finditer(str(text or "")))
    if len(matches) != 1:
        return None
    match = matches[0]
    start, end = int(match.group("start")), int(match.group("end"))
    if start <= 0 or end < start:
        return None
    return [start, end + 1]


def _interval_iou(first, second):
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union > 0 else 0.0


@lru_cache(maxsize=4096)
def _read_video_metadata(path):
    try:
        import decord
    except ImportError as error:
        raise RuntimeError("decord is required for temporal evaluation.") from error
    reader = decord.VideoReader(path, num_threads=1)
    return len(reader), float(reader.get_avg_fps())


def _video_metadata(doc):
    if doc.get("frame_count") is not None and doc.get("fps") is not None:
        return int(doc["frame_count"]), float(doc["fps"])
    return _read_video_metadata(str(_video_path(doc)))


def _sampled_frames(frame_count, source_fps):
    count = math.floor(frame_count / source_fps * 2.0)
    count = min(max(count, 4), 64, frame_count)
    if count <= 1:
        return [0]
    return [round(index * (frame_count - 1) / (count - 1)) for index in range(count)]


def _token_segment_to_seconds(prediction, frame_count, source_fps):
    segment = _segment(prediction)
    if segment is None:
        return None
    sampled = _sampled_frames(frame_count, source_fps)
    boundaries = [0.0]
    boundaries.extend(
        (left + right + 1.0) / 2.0 for left, right in zip(sampled, sampled[1:])
    )
    boundaries.append(float(frame_count))
    start_index, end_index = segment[0] - 1, segment[1] - 1
    if not 0 <= start_index < end_index <= len(sampled):
        return None
    return [boundaries[start_index] / source_fps, boundaries[end_index] / source_fps]


def activitynet_process_results(doc, result):
    frame_count, source_fps = _video_metadata(doc)
    predicted = _token_segment_to_seconds(
        result[0] if result else "", frame_count, source_fps
    )
    target = [float(value) for value in doc["timestamp"]]
    iou = _interval_iou(target, predicted) if predicted is not None else 0.0
    return {
        "ptd_activitynet_R@0.3": 100.0 * float(iou >= 0.3),
        "ptd_activitynet_R@0.5": 100.0 * float(iou >= 0.5),
        "ptd_activitynet_R@0.7": 100.0 * float(iou >= 0.7),
        "ptd_activitynet_mIoU": 100.0 * iou,
    }
