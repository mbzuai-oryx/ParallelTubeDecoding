"""Qwen3-VL lmms-eval adapter for Quantized and PTD generation."""

import json
from pathlib import Path
import re
import sys
import time

import torch
from tqdm import tqdm
from transformers import LogitsProcessor

from lmms_eval import utils
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL


_SEGMENT = re.compile(
    r"<\|time_start\|>\s*<t\d+>\s*<t\d+>\s*<\|time_end\|>"
)
_BOX_ATTEMPT = re.compile(r"<t\d+>\s*<\|box_start\|>")
_COMPLETE_BOX = re.compile(
    r"<t\d+>\s*<\|box_start\|>\s*"
    r"<\d+>\s*<\d+>\s*<\d+>\s*<\d+>\s*<\|box_end\|>"
)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def _optional_attention(value):
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return str(value).strip().lower()


class _DecodeOnlyTimer(LogitsProcessor):
    """Start TCL after the autoregressive prefill produces first-token logits."""

    def __init__(self):
        self.start_time = None
        self.device = None

    def __call__(self, input_ids, scores):
        if self.start_time is None:
            self.device = scores.device
            torch.cuda.synchronize(self.device)
            self.start_time = time.perf_counter()
        return scores

    def elapsed(self):
        if self.start_time is None:
            return 0.0
        torch.cuda.synchronize(self.device)
        return max(time.perf_counter() - self.start_time, 0.0)


@register_model("ptd_qwen3_vl")
class PTDQwen3VL(Qwen3_VL):
    def __init__(
        self,
        pretrained,
        ptd_root,
        decoding="ptd",
        generation_format="spatio_temporal_grounding",
        batch_size=1,
        system_prompt=None,
        fps=2,
        max_num_frames=64,
        temporal_patch_size=1,
        min_pixels=131072,
        max_pixels=786432,
        attn_implementation="sdpa",
        vision_attn_implementation=None,
        ptd_attn_implementation=None,
        measure_efficiency=False,
        efficiency_log=None,
        **kwargs,
    ):
        if int(batch_size) != 1:
            raise ValueError("PTD evaluation uses batch_size=1.")
        self.decoding = str(decoding).strip().lower()
        if self.decoding not in {"quantized", "ptd"}:
            raise ValueError("decoding must be 'quantized' or 'ptd'.")
        if generation_format not in {
            "spatio_temporal_grounding",
            "temporal_localization",
        }:
            raise ValueError(f"Unsupported PTD generation format: {generation_format}")
        if float(fps) != 2 or int(max_num_frames) != 64:
            raise ValueError("PTD evaluation requires fps=2 and max_num_frames=64.")
        if int(temporal_patch_size) != 1:
            raise ValueError("PTD evaluation requires temporal_patch_size=1.")
        if int(min_pixels) != 131072 or int(max_pixels) != 786432:
            raise ValueError(
                "PTD evaluation requires min_pixels=131072 and max_pixels=786432."
            )

        attn_implementation = str(attn_implementation).strip().lower()
        vision_attn_implementation = _optional_attention(
            vision_attn_implementation
        )
        ptd_attn_implementation = _optional_attention(ptd_attn_implementation)
        if attn_implementation != "sdpa":
            raise ValueError("Quantized and PTD text inference use SDPA.")
        if self.decoding == "quantized":
            if vision_attn_implementation is not None:
                raise ValueError(
                    "Quantized inference uses attn_implementation=sdpa without "
                    "a separate vision attention override."
                )
            if ptd_attn_implementation is not None:
                raise ValueError("Quantized inference does not use PTD block attention.")
            self.ptd_attn_implementation = "sdpa"
        else:
            vision_attn_implementation = (
                vision_attn_implementation or "flash_attention_2"
            )
            ptd_attn_implementation = (
                ptd_attn_implementation or "flash_attention_2"
            )
            if vision_attn_implementation != "flash_attention_2":
                raise ValueError("PTD inference uses vision FlashAttention-2.")
            if ptd_attn_implementation != "flash_attention_2":
                raise ValueError("PTD inference uses PTD block FlashAttention-2.")
            self.ptd_attn_implementation = ptd_attn_implementation

        self.measure_efficiency = _as_bool(measure_efficiency)
        self.efficiency_log = (
            Path(efficiency_log).expanduser() if efficiency_log else None
        )
        if self.measure_efficiency:
            if generation_format != "spatio_temporal_grounding":
                raise ValueError("TCL/BPS measurement is only defined for tube generation.")
            if self.efficiency_log is None:
                raise ValueError("measure_efficiency=true requires efficiency_log.")

        source = Path(ptd_root).expanduser().resolve() / "src"
        if not source.is_dir():
            raise FileNotFoundError(f"PTD source directory not found: {source}")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))

        from train.monkey_patch_forward import replace_qwen3_with_ptd_forward

        replace_qwen3_with_ptd_forward()
        if isinstance(system_prompt, str) and system_prompt.lower() == "none":
            system_prompt = None
        super().__init__(
            pretrained=pretrained,
            batch_size=1,
            fps=float(fps),
            max_num_frames=int(max_num_frames),
            temporal_patch_size=int(temporal_patch_size),
            attn_implementation=attn_implementation,
            vision_attn_implementation=vision_attn_implementation,
            system_prompt=system_prompt,
            min_pixels=int(min_pixels),
            max_pixels=int(max_pixels),
            skip_special_tokens=False,
            **kwargs,
        )

        if self.measure_efficiency:
            if self.world_size != 1:
                raise ValueError(
                    "The paper's TCL/BPS protocol uses one GPU and one process."
                )
            self.efficiency_log.parent.mkdir(parents=True, exist_ok=True)
            self.efficiency_log.write_text("", encoding="utf-8")

        from dataset.data_utils import (
            patch_processor_with_time_tokens,
            patch_qwen3_video_processor,
        )

        patch_qwen3_video_processor(self.processor)
        patch_processor_with_time_tokens(self.processor)
        self.ptd_generation_format = generation_format
        self.fps = float(fps)
        self.max_num_frames = int(max_num_frames)
        self.temporal_patch_size = int(temporal_patch_size)

    def _write_efficiency_record(
        self,
        *,
        answer,
        tcl,
        ptd_stopped,
        task,
        split,
        doc_id,
    ):
        attempts = len(_BOX_ATTEMPT.findall(answer))
        boxes = len(_COMPLETE_BOX.findall(answer))
        complete = boxes > 0 and attempts == boxes and bool(_SEGMENT.search(answer))
        if ptd_stopped is not None:
            complete = complete and ptd_stopped
        record = {
            "schema_version": 1,
            "sample_id": f"{task}:{split}:{doc_id}",
            "decoding": self.decoding,
            "tcl_s": float(tcl),
            "num_boxes": int(boxes),
            "bps": float(boxes / tcl) if tcl > 0 else 0.0,
            "trajectory_complete": bool(complete),
            "attn_implementation": "sdpa",
            "vision_attn_implementation": (
                "flash_attention_2" if self.decoding == "ptd" else None
            ),
            "ptd_attn_implementation": (
                self.ptd_attn_implementation if self.decoding == "ptd" else None
            ),
        }
        with self.efficiency_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def generate_until(self, requests):
        from model.ptd_generation import generate_ptd
        from qwen_vl_utils import process_vision_info
        import qwen_vl_utils.vision_process as qwen_vision_process

        def collate(request):
            return -len(self.tokenizer.encode(request[0])), request[0]

        responses = []
        progress = tqdm(
            total=len(requests),
            disable=self.rank != 0,
            desc="Model Responding",
        )
        ordered = utils.Collator(
            [request.args for request in requests], collate, grouping=True
        )
        chunks = ordered.get_batched(n=1, batch_fn=None)

        for chunk in chunks:
            contexts, generation_kwargs, doc_to_visual, doc_id, task, split = zip(
                *chunk
            )
            context = contexts[0].replace("<image>", "")
            generation = dict(generation_kwargs[0])
            until = generation.pop(
                "until", [self.tokenizer.decode(self.eot_token_id)]
            )
            if isinstance(until, str):
                until = [until]
            until = [term for term in until if term != "\n\n"]

            visuals = doc_to_visual[0](
                self.task_dict[task[0]][split[0]][doc_id[0]]
            )
            content = []
            for visual in visuals or []:
                video = dict(visual) if isinstance(visual, dict) else {"video": visual}
                video.update(
                    {
                        "type": "video",
                        "fps": self.fps,
                        "min_frames": 4,
                        "max_frames": self.max_num_frames,
                        "temporal_patch_size": self.temporal_patch_size,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    }
                )
                content.append(video)
            content.append({"type": "text", "text": context})
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": content})
            batched_messages = [messages]
            texts = self.processor.apply_chat_template(
                batched_messages, tokenize=False, add_generation_prompt=True
            )

            # Match the released temporal-patch-size-1 preprocessing. The
            # processor patch below duplicates each distinct frame only across
            # Qwen's fixed Conv3d temporal width.
            qwen_vision_process.FRAME_FACTOR = 1
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                batched_messages,
                return_video_kwargs=True,
                image_patch_size=16,
                return_video_metadata=True,
            )
            video_metadata = None
            if video_inputs is not None:
                video_inputs, video_metadata = zip(*video_inputs)
                video_inputs, video_metadata = list(video_inputs), list(video_metadata)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata,
                **video_kwargs,
                do_resize=False,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda" if self.device_map == "auto" else self.device)
            generation = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "top_k": None,
                "num_beams": 1,
                **generation,
            }
            if generation.get("num_beams", 1) != 1:
                raise ValueError("PTD does not use beam search.")

            ptd_result = None
            if self.decoding == "ptd":
                completion, ptd_result = generate_ptd(
                    self.model,
                    self.tokenizer,
                    dict(inputs),
                    max_new_tokens=generation["max_new_tokens"],
                    max_time_tokens=int(inputs["video_grid_thw"][0, 0].item()),
                    temperature=float(generation.get("temperature") or 0.0),
                    top_p=generation.get("top_p"),
                    top_k=generation.get("top_k"),
                    generation_format=self.ptd_generation_format,
                    ptd_attn_implementation=self.ptd_attn_implementation,
                    measure_decode=self.measure_efficiency,
                )
                tcl = ptd_result.decode_latency_s
            else:
                timer = _DecodeOnlyTimer()
                autoregressive_kwargs = {
                    "max_new_tokens": generation["max_new_tokens"],
                    "do_sample": float(generation.get("temperature") or 0.0) > 0,
                    "num_beams": 1,
                    "use_cache": True,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id
                    or self.tokenizer.eos_token_id,
                }
                if autoregressive_kwargs["do_sample"]:
                    autoregressive_kwargs.update(
                        temperature=float(generation["temperature"]),
                        top_p=generation.get("top_p"),
                        top_k=generation.get("top_k"),
                    )
                if self.measure_efficiency:
                    autoregressive_kwargs["logits_processor"] = [timer]
                generated = self.model.generate(
                    **inputs,
                    **autoregressive_kwargs,
                )
                completion = generated[:, inputs["input_ids"].shape[1] :]
                tcl = timer.elapsed() if self.measure_efficiency else None
            answers = self.processor.batch_decode(
                completion,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for answer in answers:
                if self.measure_efficiency:
                    self._write_efficiency_record(
                        answer=answer,
                        tcl=tcl,
                        ptd_stopped=(
                            ptd_result.stopped if ptd_result is not None else None
                        ),
                        task=task[0],
                        split=split[0],
                        doc_id=doc_id[0],
                    )
                for stop in until:
                    if stop:
                        answer = answer.split(stop)[0]
                responses.append(answer)
                self.cache_hook.add_partial(
                    "generate_until", (context, generation), answer
                )
                progress.update(1)

        progress.close()
        return ordered.get_original(responses)
