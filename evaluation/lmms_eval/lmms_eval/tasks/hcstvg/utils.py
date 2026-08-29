"""lmms-eval helpers for PTD evaluation on HC-STVG."""

from functools import partial
import os
from pathlib import Path
import re


PROMPT = (
    "Given the query: '{query}' Localize the described object throughout the "
    "video. Use object reference tokens, time tokens, and box tokens. Return "
    "the object reference, event time segment, and per-time bbox coordinates."
)
_SEGMENT = re.compile(
    r"<\|time_start\|>\s*<t(?P<start>\d+)>\s*"
    r"<t(?P<end>\d+)>\s*<\|time_end\|>"
)
_BOX = re.compile(
    r"<t(?P<time>\d+)>\s*<\|box_start\|>\s*"
    r"<(?P<x1>\d+)>\s*<(?P<y1>\d+)>\s*"
    r"<(?P<x2>\d+)>\s*<(?P<y2>\d+)>\s*<\|box_end\|>"
)


def _query(doc):
    query = str(doc["caption"]).strip()
    return query if query.endswith((".", "?", "!")) else f"{query}."


def hcstvg_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    path = Path(str(doc.get("video_path") or doc["video"])).expanduser()
    root = os.environ.get("HCSTVG_VIDEO_ROOT")
    if not path.is_absolute() and root:
        path = Path(root).expanduser() / path
    video = {"video": str(path)}
    if doc.get("video_start_sec") is not None:
        video["video_start"] = float(doc["video_start_sec"])
    if doc.get("video_end_sec") is not None:
        video["video_end"] = float(doc["video_end_sec"])
    return [video]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return PROMPT.format(query=_query(doc))


def doc_to_answer(doc):
    return doc["gt_sampled_frame_boxes"]


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


def _predicted_boxes(text):
    boxes = {}
    for match in _BOX.finditer(str(text or "")):
        time_index = int(match.group("time"))
        box = [int(match.group(name)) for name in ("x1", "y1", "x2", "y2")]
        if time_index <= 0 or time_index in boxes:
            return None
        boxes[time_index] = box
    return boxes


def _box_iou(first, second):
    ax1, ay1, ax2, ay2 = [max(0.0, min(float(value), 1000.0)) for value in first]
    bx1, by1, bx2, by2 = [max(0.0, min(float(value), 1000.0)) for value in second]
    ax1, ax2 = sorted((ax1, ax2))
    ay1, ay2 = sorted((ay1, ay2))
    bx1, bx2 = sorted((bx1, bx2))
    by1, by2 = sorted((by1, by2))
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = (ax2 - ax1) * (ay2 - ay1)
    second_area = (bx2 - bx1) * (by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _ious(doc, prediction):
    if int(doc.get("temporal_patch_size") or 1) != 1:
        raise ValueError("PTD evaluation expects temporal_patch_size=1.")
    ground_truth = {
        int(item["time_index"]): item["bbox"]
        for item in doc["gt_sampled_frame_boxes"]
    }
    predicted = _predicted_boxes(prediction)
    ground_truth_segment = (
        [min(ground_truth), max(ground_truth) + 1] if ground_truth else None
    )
    predicted_segment = _segment(prediction)
    temporal_iou = (
        _interval_iou(ground_truth_segment, predicted_segment)
        if ground_truth_segment is not None and predicted_segment is not None
        else 0.0
    )
    if predicted is None:
        return temporal_iou, 0.0
    time_indices = set(ground_truth) | set(predicted)
    video_iou = 0.0
    if time_indices:
        video_iou = sum(
            _box_iou(ground_truth[index], predicted[index])
            for index in time_indices
            if index in ground_truth and index in predicted
        ) / len(time_indices)
    return temporal_iou, video_iou


def hcstvg_process_results(doc, result):
    version = str(doc.get("version", "")).strip().lower()
    if version not in {"v1", "v2"}:
        raise ValueError(f"Unexpected version={version!r}.")
    temporal_iou, video_iou = _ious(doc, result[0] if result else "")
    record = {"group": version, "tIoU": temporal_iou, "vIoU": video_iou}
    return {
        f"ptd_hcstvg_{name}_{metric}": record
        for name in ("v1", "v2")
        for metric in ("m_tIoU", "m_vIoU", "vIoU@0.3", "vIoU@0.5")
    }


def _aggregate(results, args=None, *, group, metric):
    selected = [record for record in results if record["group"] == group]
    if not selected:
        return 0.0
    values = [record["tIoU" if metric == "m_tIoU" else "vIoU"] for record in selected]
    if metric in {"m_tIoU", "m_vIoU"}:
        return 100.0 * sum(values) / len(values)
    threshold = float(metric.rsplit("@", 1)[1])
    return 100.0 * sum(value >= threshold for value in values) / len(values)


hcstvg_v1_m_tiou = partial(_aggregate, group="v1", metric="m_tIoU")
hcstvg_v1_m_viou = partial(_aggregate, group="v1", metric="m_vIoU")
hcstvg_v1_viou03 = partial(_aggregate, group="v1", metric="vIoU@0.3")
hcstvg_v1_viou05 = partial(_aggregate, group="v1", metric="vIoU@0.5")
hcstvg_v2_m_tiou = partial(_aggregate, group="v2", metric="m_tIoU")
hcstvg_v2_m_viou = partial(_aggregate, group="v2", metric="m_vIoU")
hcstvg_v2_viou03 = partial(_aggregate, group="v2", metric="vIoU@0.3")
hcstvg_v2_viou05 = partial(_aggregate, group="v2", metric="vIoU@0.5")
