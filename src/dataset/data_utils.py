import os
import re
from functools import lru_cache
from types import MethodType

import torch

from transformers import AutoConfig

from qwen_vl_utils import process_vision_info

from constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    LLAVA_IMAGE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
)


def replace_image_tokens(input_string, is_video=False, preserve_whitespace=False):
    if is_video:
        token = LLAVA_VIDEO_TOKEN
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        token = LLAVA_IMAGE_TOKEN
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    # Default: also swallow the newlines around the token, which matches how the official
    # Qwen chat template renders a content list and keeps the classic LLaVA
    # "<image>\nQuestion" datasets rendering exactly as they always have.
    #
    # With preserve_whitespace the author's layout is kept verbatim. This matters for
    # structured multi-image prompts -- "Image 1: <image>\nImage 2: <image>\n\nTask: ..."
    # otherwise collapses to "Image 1: <...>Image 2: <...>Task: ..." with no warning, and
    # the training-time prompt stops matching whatever the same string renders to at
    # inference time.
    if preserve_whitespace:
        return input_string.replace(token, replacement)

    return re.sub(r'\n*' + re.escape(token) + r'\n*', replacement, input_string)

def llava_to_openai(conversations, is_video=False, preserve_whitespace=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(
            conversation["value"], is_video=is_video, preserve_whitespace=preserve_whitespace
        )
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        if "reasoning" in conversation:
            transformed_entry["reasoning"] = conversation["reasoning"]
        transformed_data.append(transformed_entry)

    return transformed_data


def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def get_mm_token_type_ids(inputs, input_ids):
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if mm_token_type_ids is None:
        return torch.zeros_like(input_ids, dtype=torch.long)
    return mm_token_type_ids.to(dtype=torch.long)


@lru_cache(maxsize=32)
def get_qwen_multimodal_settings(model_id_or_path):
    model_type = AutoConfig.from_pretrained(model_id_or_path).model_type
    if model_type in {"qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"}:
        return model_type, 16, True
    return model_type, 14, False


def use_default_system_message(model_type):
    return model_type in {"qwen2_vl", "qwen2_5_vl"}


def chat_template_uses_reasoning_prefill(processor, model_type=None):
    template = getattr(processor, "chat_template", None)
    if not template and hasattr(processor, "tokenizer"):
        template = getattr(processor.tokenizer, "chat_template", None)
    template = template or ""
    supported_model_types = {"qwen3_vl", "qwen3_5", "qwen3_5_moe"}
    if model_type not in supported_model_types:
        return False
    return (
        "reasoning_content" in template
        and "<think>" in template
        and "add_generation_prompt" in template
        and "<|im_start|>assistant" in template
    )


def model_supports_optional_reasoning(model_type):
    return model_type in {"qwen3_5", "qwen3_5_moe"}


def format_assistant_response(
    content,
    reasoning=None,
    *,
    enable_reasoning=False,
    use_reasoning_prefill=False,
    use_closed_think_prefill=False,
):
    if use_closed_think_prefill:
        return "<think>\n\n</think>\n\n", content.lstrip("\n")

    if not enable_reasoning or not isinstance(reasoning, str) or not reasoning.strip():
        return "", content

    reasoning = reasoning.strip("\n")
    content = content.lstrip("\n")

    if use_reasoning_prefill:
        return "<think>\n", f"{reasoning}\n</think>\n\n{content}"

    return "", f"<think>\n{reasoning}\n</think>\n\n{content}"

def get_image_info(image_path, min_pixel, max_pixel, width, height, image_patch_size):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    content = {
        "type": "image", 
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {
            "role": "user", 
            "content": [content]
        }
    ]

    image_input, _ = process_vision_info(messages, image_patch_size=image_patch_size)

    return image_input[0]

def resolve_video_spec(video, video_folder=None):
    """Resolve a video path while preserving optional start/end times."""
    is_dict = isinstance(video, dict)
    if is_dict:
        video = dict(video)
        path = video.get("video")
        if path is None:
            raise ValueError("A video dictionary must contain a 'video' path.")
    elif isinstance(video, str):
        path = video
    else:
        raise TypeError("A video entry must be a path string or dictionary.")

    path = str(path)
    if not os.path.exists(path) and not path.startswith(("http://", "https://")):
        if video_folder is None:
            raise ValueError(f"Relative video path {path!r} requires --image_folder.")
        path = os.path.join(video_folder, path)

    if is_dict:
        video["video"] = path
        return video
    return path


def get_video_info(
    video,
    min_pixels,
    max_pixels,
    width,
    height,
    fps,
    image_patch_size,
    max_frames=64,
    return_video_metadata=False,
    nframes=None,
    temporal_patch_size=1,
):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    if fps is not None and nframes is not None:
        raise ValueError("Only one of fps and nframes may be set.")

    if temporal_patch_size == 1:
        # qwen-vl-utils 0.0.14 otherwise rounds sampling to pairs of distinct
        # frames. PTD keeps Qwen's Conv3d width but samples one distinct frame
        # per time token; patch_qwen3_video_processor duplicates it only when
        # constructing the Conv3d input.
        import qwen_vl_utils.vision_process as qwen_vision_process

        qwen_vision_process.FRAME_FACTOR = 1

    video_options = dict(video) if isinstance(video, dict) else {}
    content = {
        "type": "video",
        "video": video_options.pop("video", video),
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }
    content.update(video_options)

    sample_fps = content.pop("fps", fps)
    sample_frames = content.pop("nframes", nframes)
    if sample_fps is not None and sample_frames is not None:
        raise ValueError("A video cannot specify both fps and nframes.")
    if sample_frames is not None:
        content["nframes"] = sample_frames
    elif sample_fps is not None:
        content["fps"] = sample_fps
        content["max_frames"] = content.get("max_frames", max_frames)
    content["temporal_patch_size"] = temporal_patch_size

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {
            "role": "user", 
            "content": [content]
        }
    ]

    _, video_input, video_kwargs = process_vision_info(
        messages, 
        return_video_kwargs=True, 
        image_patch_size=image_patch_size, 
        return_video_metadata=return_video_metadata
    )

    return video_input[0], video_kwargs


def patch_qwen3_video_processor(processor):
    """Keep Qwen's temporal kernel while creating one visual token per frame."""
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None or video_processor.__class__.__name__ != "Qwen3VLVideoProcessor":
        raise TypeError("PTD requires a Qwen3-VL processor.")
    if getattr(video_processor, "_ptd_patched", False):
        return processor

    # Current lmms-eval already installs a Qwen3-VL video processor that keeps
    # the pretrained temporal kernel and exposes ``frame_factor`` separately.
    # Setting it to one is sufficient
    if hasattr(video_processor, "frame_factor"):
        video_processor.frame_factor = 1
        video_processor.max_frames = 64
        video_processor._ptd_patched = True
        return processor

    temporal_width = int(video_processor.temporal_patch_size)
    original_preprocess = video_processor._preprocess

    def preprocess(self, videos, *args, **kwargs):
        videos = [video.repeat_interleave(temporal_width, dim=0) for video in videos]
        kwargs["temporal_patch_size"] = temporal_width
        return original_preprocess(videos, *args, **kwargs)

    video_processor._preprocess = MethodType(preprocess, video_processor)
    video_processor.frame_factor = 1
    video_processor.max_frames = 64
    video_processor._ptd_patched = True
    return processor


def patch_processor_with_time_tokens(processor, max_time_tokens=100):
    """Place one learned time token before each sampled video frame."""
    if max_time_tokens <= 0:
        raise ValueError("max_time_tokens must be positive.")
    if not hasattr(processor, "replace_video_token"):
        raise TypeError("PTD requires a Qwen3-VL processor.")

    tokenizer = processor.tokenizer
    for token in ("<t1>", f"<t{max_time_tokens}>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(f"Missing PTD time token {token}.")
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f"PTD time token {token} must encode as one token.")

    def replace_video_token(self, *args, **kwargs):
        video_inputs = kwargs.get("video_inputs", args[0] if args else None)
        video_index = kwargs.get("video_idx", args[1] if len(args) > 1 else None)
        if video_inputs is None or video_index is None:
            raise TypeError("Video inputs and video index are required.")

        grid = video_inputs["video_grid_thw"][int(video_index)]
        frame_count = int(grid[0])
        if frame_count > max_time_tokens:
            raise ValueError(
                f"PTD received {frame_count} frames but has only "
                f"{max_time_tokens} time tokens."
            )
        tokens_per_frame = int(grid[1] * grid[2]) // int(
            self.video_processor.merge_size ** 2
        )
        return "".join(
            f"<t{frame + 1}>"
            + self.vision_start_token
            + self.video_token * tokens_per_frame
            + self.vision_end_token
            for frame in range(frame_count)
        )

    processor.replace_video_token = MethodType(replace_video_token, processor)
    return processor

def samples_per_class_from_ids(label_ids, num_classes):
    
    counts = torch.bincount(
        torch.as_tensor(label_ids, dtype=torch.long),
        minlength=num_classes
    )
    
    return counts.tolist()
