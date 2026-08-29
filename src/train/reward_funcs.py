import re

try:
    from math_verify import LatexExtractionConfig, parse, verify
    from latex2sympy2_extended import NormalizationConfig
except ImportError:
    LatexExtractionConfig = None
    NormalizationConfig = None
    parse = None
    verify = None

def accuracy_reward(completions, assistant, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    rewards = []

    for completion, sol in zip(completions, assistant):
        if parse is None or verify is None or LatexExtractionConfig is None or NormalizationConfig is None:
            rewards.append(float(completion.strip().lower() == sol.strip().lower()))
            continue

        try:
            gold_parsed = parse(sol, extraction_mode="first_match")
        except Exception as e:
            gold_parsed = []

        if len(gold_parsed) != 0:
            # Try parsing predicted answer too
            try:
                answer_parsed = parse(
                    completion,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {completion}, gold: {sol}")
                reward = None
        else:
            # fallback to text match
            reward = float(completion.strip().lower() == sol.strip().lower())

        rewards.append(reward)

    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards


_TIME_SEGMENT = re.compile(
    r"<\|time_start\|>\s*<t(\d+)>\s*<t(\d+)>\s*<\|time_end\|>"
)
_TIME_BOX = re.compile(
    r"<t(\d+)>\s*<\|box_start\|>\s*"
    r"<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<\|box_end\|>"
)


def _parse_ptd_output(text):
    text = str(text)
    segment_matches = list(_TIME_SEGMENT.finditer(text))
    if len(segment_matches) != 1:
        return None, None

    segment = tuple(map(int, segment_matches[0].groups()))
    if segment[0] <= 0 or segment[0] > segment[1]:
        return None, None

    boxes = {}
    for match in _TIME_BOX.finditer(text):
        time_index, x1, y1, x2, y2 = map(int, match.groups())
        if (
            time_index <= 0
            or time_index in boxes
            or max(x1, y1, x2, y2) > 1000
            or x1 >= x2
            or y1 >= y2
        ):
            return segment, None
        boxes[time_index] = tuple(value / 1000 for value in (x1, y1, x2, y2))
    return segment, boxes


def _spatial_score(prediction, target):
    px1, py1, px2, py2 = prediction
    tx1, ty1, tx2, ty2 = target
    intersection = max(0, min(px2, tx2) - max(px1, tx1)) * max(
        0, min(py2, ty2) - max(py1, ty1)
    )
    predicted_area = (px2 - px1) * (py2 - py1)
    target_area = (tx2 - tx1) * (ty2 - ty1)
    union = predicted_area + target_area - intersection
    iou = intersection / union
    enclosing_area = (max(px2, tx2) - min(px1, tx1)) * (
        max(py2, ty2) - min(py1, ty1)
    )
    generalized_iou = iou - (enclosing_area - union) / enclosing_area
    l1 = sum(abs(predicted - gold) for predicted, gold in zip(prediction, target))
    return generalized_iou - l1


def temporal_iou_reward(completions, assistant, **kwargs):
    """Temporal IoU between the predicted and target frame intervals."""
    predictions = kwargs.get("ptd_completion", completions)
    rewards = []
    for prediction, target in zip(predictions, assistant):
        predicted_segment, _ = _parse_ptd_output(prediction)
        target_segment, _ = _parse_ptd_output(target)
        if predicted_segment is None or target_segment is None:
            rewards.append(0.0)
            continue

        intersection = max(
            0,
            min(predicted_segment[1], target_segment[1])
            - max(predicted_segment[0], target_segment[0])
            + 1,
        )
        union = max(predicted_segment[1], target_segment[1]) - min(
            predicted_segment[0], target_segment[0]
        ) + 1
        rewards.append(intersection / union)
    return rewards


def spatial_reward(completions, assistant, **kwargs):
    """Mean GIoU-minus-L1 score over the temporal intersection."""
    predictions = kwargs.get("ptd_completion", completions)
    rewards = []
    for prediction, target in zip(predictions, assistant):
        predicted_segment, predicted_boxes = _parse_ptd_output(prediction)
        target_segment, target_boxes = _parse_ptd_output(target)
        if (
            predicted_segment is None
            or target_segment is None
            or predicted_boxes is None
            or target_boxes is None
        ):
            rewards.append(0.0)
            continue

        start = max(predicted_segment[0], target_segment[0])
        end = min(predicted_segment[1], target_segment[1])
        time_indices = range(start, end + 1)
        if start > end or any(
            index not in predicted_boxes or index not in target_boxes
            for index in time_indices
        ):
            rewards.append(0.0)
            continue

        scores = [
            _spatial_score(predicted_boxes[index], target_boxes[index])
            for index in time_indices
        ]
        rewards.append(sum(scores) / len(scores))
    return rewards
