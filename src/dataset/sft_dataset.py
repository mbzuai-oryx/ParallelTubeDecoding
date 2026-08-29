import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from params import DataArguments
from constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
)

from .data_utils import (
    chat_template_uses_reasoning_prefill,
    format_assistant_response,
    get_image_info,
    get_mm_token_type_ids,
    get_qwen_multimodal_settings,
    get_video_info,
    llava_to_openai,
    model_supports_optional_reasoning,
    pad_sequence,
    resolve_video_spec,
    use_default_system_message,
)

PTD_BLOCK_SIZE = 6


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.nframes = data_args.nframes

        self.model_type, self.image_patch_size, self.return_video_metadata = get_qwen_multimodal_settings(
            self.model_id
        )
        self.reasoning_supported = chat_template_uses_reasoning_prefill(self.processor, self.model_type)
        self.use_reasoning_prefill = self.data_args.enable_reasoning and self.reasoning_supported
        self.optional_reasoning_supported = model_supports_optional_reasoning(self.model_type)
        if self.data_args.enable_reasoning and not self.reasoning_supported:
            raise ValueError(
                f"`enable_reasoning` is only supported for Qwen3-VL Thinking or Qwen3.5 models with official reasoning chat templates. "
                f"Current model_type={self.model_type!r} does not qualify."
            )

        tokenizer = self.processor.tokenizer

        def token_id(token):
            value = tokenizer.convert_tokens_to_ids(token)
            if value is None or value == tokenizer.unk_token_id:
                raise ValueError(f"Missing PTD token {token}.")
            if tokenizer.encode(token, add_special_tokens=False) != [value]:
                raise ValueError(f"PTD token {token} must encode as one token.")
            return int(value)

        self.ptd_token_ids = {
            token: token_id(token)
            for token in (
                "<text_mask>",
                "<null>",
                "<|object_ref_start|>",
                "<|object_ref_end|>",
                "<|time_start|>",
                "<|time_end|>",
                "<|box_start|>",
                "<|box_end|>",
                DEFAULT_IM_END_TOKEN,
            )
        }
        newline = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline) != 1:
            raise ValueError("PTD requires a newline to encode as one token.")
        self.newline_id = newline[0]
        self.time_id_to_index = {
            token_id(f"<t{index}>"): index for index in range(1, 101)
        }
        self.time_index_to_id = {
            index: value for value, index in self.time_id_to_index.items()
        }
        self.coordinate_id_to_value = {
            token_id(f"<{value}>"): value for value in range(1001)
        }

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        conversations = sources.get("conversations")
        if "video" not in sources or "image" in sources:
            raise ValueError("A PTD SFT sample must contain one video.")
        if not isinstance(conversations, list) or len(conversations) != 2:
            raise ValueError("A PTD SFT sample must contain one user/assistant pair.")
        if conversations[0].get("from") != "human" or conversations[1].get("from") != "gpt":
            raise ValueError("PTD conversation roles must be human followed by gpt.")
        if conversations[0].get("value", "").count("<video>") != 1:
            raise ValueError("A PTD prompt must contain one <video> token.")

        is_video = False

        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []

            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                image_input = get_image_info(
                        image_file, 
                        self.image_min_pixel, 
                        self.image_max_pixel, 
                        self.image_resized_w, 
                        self.image_resized_h, 
                        self.image_patch_size
                    )
                images.append(image_input)

        elif "video" in sources:
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, (str, dict)):
                video_files = [video_files]
            if len(video_files) != 1:
                raise ValueError("A PTD SFT sample must contain one video.")

            videos = []
            for video_file in video_files:
                video_file = resolve_video_spec(video_file, video_folder)
                video_input, video_kwargs = get_video_info(
                    video_file, 
                    self.video_min_pixel, 
                    self.video_max_pixel, 
                    self.video_resized_w, 
                    self.video_resized_h, 
                    self.data_args.fps,
                    self.image_patch_size,
                    max_frames=self.data_args.max_frames,
                    return_video_metadata=self.return_video_metadata,
                    nframes=self.data_args.nframes,
                    temporal_patch_size=self.data_args.temporal_patch_size,
                )
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None

        sources = copy.deepcopy(llava_to_openai(
            sources['conversations'], is_video=is_video,
            preserve_whitespace=getattr(self.data_args, 'preserve_media_token_whitespace', False),
        ))

        all_input_ids = []
        all_labels = []
        all_mm_token_type_ids = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        image_curr_count = 0
        video_curr_count = 0
        
        # Qwen2-VL uses a default system message so I've added this.
        # Qwen3-Vl does not use a system message by default.
        if len(SYSTEM_MESSAGE) > 0 and use_default_system_message(self.model_type):
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)

            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))
            all_mm_token_type_ids.append(torch.zeros_like(system_message_input_ids, dtype=torch.long).squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]
            has_reasoning = isinstance(gpt_response.get("reasoning"), str) and gpt_response["reasoning"].strip()
            if self.data_args.enable_reasoning and self.reasoning_supported and not self.optional_reasoning_supported and not has_reasoning:
                raise ValueError(
                    "When `--enable_reasoning True` is used with Qwen3-VL Thinking, every assistant turn must include a non-empty `reasoning` field."
                )
            use_reasoning_prefill = self.use_reasoning_prefill and has_reasoning
            use_closed_think_prefill = self.optional_reasoning_supported and not use_reasoning_prefill
            assistant_prefill, assistant_content = format_assistant_response(
                gpt_response["content"],
                gpt_response.get("reasoning"),
                enable_reasoning=self.data_args.enable_reasoning,
                use_reasoning_prefill=use_reasoning_prefill,
                use_closed_think_prefill=use_closed_think_prefill,
            )

            user_input = (
                f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n"
                f"{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n{assistant_prefill}"
            )
            gpt_response = f"{assistant_content}{DEFAULT_IM_END_TOKEN}\n"

            if DEFAULT_IMAGE_TOKEN in user_input:
                num_images = user_input.count(DEFAULT_IMAGE_TOKEN)
                # Slice the images list to get the images for the current turn.
                images_for_this_turn = images[image_curr_count : image_curr_count + num_images]
                inputs = processor(text=[user_input], images=images_for_this_turn, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, prompt_input_ids)
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                image_curr_count += num_images

            elif DEFAULT_VIDEO_TOKEN in user_input:
                num_videos = user_input.count(DEFAULT_VIDEO_TOKEN)
                # Slice the videos list to get the videos for the current turn.
                videos_for_this_turn = videos[video_curr_count : video_curr_count + num_videos]
                if self.model_type == "qwen2_5_vl":
                    inputs = processor(
                        text=[user_input], 
                        images=images, 
                        videos=videos_for_this_turn, 
                        padding=False, 
                        do_resize=False, 
                        return_tensors='pt', 
                        **video_kwargs
                    )
                    prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, inputs["input_ids"])
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                elif self.model_type in {"qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"}:
                    videos_for_this_turn = videos[video_curr_count : video_curr_count + num_videos]
                    video_datas_for_turn, video_metadatas_for_turn = zip(*videos_for_this_turn)
                    video_datas_for_turn = list(video_datas_for_turn)
                    video_metadatas_for_turn = list(video_metadatas_for_turn)

                    inputs = processor(
                        text=[user_input],
                        images=images,
                        videos=video_datas_for_turn,
                        padding=False,
                        do_resize=False,
                        return_tensors='pt',
                        **video_kwargs,
                        video_metadata=video_metadatas_for_turn,
                    )
                    prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, inputs["input_ids"])
                else:
                    inputs = processor(
                        text=[user_input], 
                        images=images, 
                        videos=videos_for_this_turn, 
                        padding=False, 
                        do_resize=False, 
                        return_tensors='pt'
                    )
                    prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, inputs["input_ids"])
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                video_curr_count += num_videos

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
                prompt_mm_token_type_ids = torch.zeros_like(prompt_input_ids, dtype=torch.long)

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
            response_mm_token_type_ids = torch.zeros_like(response_input_ids, dtype=torch.long)

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            mm_token_type_ids = torch.cat([prompt_mm_token_type_ids, response_mm_token_type_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_mm_token_type_ids.append(mm_token_type_ids)

        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        mm_token_type_ids = torch.cat(all_mm_token_type_ids, dim=0).to(torch.long)

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            mm_token_type_ids=mm_token_type_ids,
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird

        response_tokens = torch.nonzero(labels.ne(IGNORE_INDEX), as_tuple=False).flatten()
        if response_tokens.numel() == 0:
            raise ValueError("A PTD SFT sample has no supervised response.")
        return self._append_ptd_targets(data_dict, int(response_tokens[0]))

    def _append_ptd_targets(self, data, response_start):
        input_ids = data["input_ids"]
        labels = data["labels"]
        ids = self.ptd_token_ids
        blocks = []
        targets = []
        positions = []
        context_limits = []

        def add_block(query_id, position, context_limit, target):
            block = input_ids.new_full((PTD_BLOCK_SIZE,), ids["<text_mask>"])
            block[0] = query_id
            padded_target = labels.new_full((PTD_BLOCK_SIZE,), ids["<null>"])
            padded_target[: target.numel()] = target
            blocks.append(block)
            targets.append(padded_target)
            positions.append(torch.arange(position, position + PTD_BLOCK_SIZE))
            context_limits.append(
                torch.full((PTD_BLOCK_SIZE,), context_limit, dtype=torch.long)
            )

        ref_start_id = ids["<|object_ref_start|>"]
        ref_end_id = ids["<|object_ref_end|>"]
        if int(input_ids[response_start]) != ref_start_id:
            raise ValueError("A PTD response must begin with <|object_ref_start|>.")
        ref_end_matches = torch.nonzero(
            input_ids[response_start:].eq(ref_end_id), as_tuple=False
        ).flatten()
        if ref_end_matches.numel() != 1:
            raise ValueError("A PTD response must contain one <|object_ref_end|>.")
        ref_end = response_start + int(ref_end_matches[0])

        cursor = response_start
        while cursor <= ref_end:
            stop = min(cursor + PTD_BLOCK_SIZE, ref_end + 1)
            query_position = cursor - 1
            while query_position >= 0 and int(input_ids[query_position]) == self.newline_id:
                query_position -= 1
            if query_position < 0:
                raise ValueError("The PTD object reference has no preceding query token.")
            context_limit = cursor if query_position != cursor - 1 else cursor - 1
            add_block(
                int(input_ids[query_position]),
                cursor - 1,
                context_limit,
                input_ids[cursor:stop],
            )
            cursor = stop

        time_start_id = ids["<|time_start|>"]
        time_end_id = ids["<|time_end|>"]
        segment_starts = []
        for index in range(ref_end + 1, input_ids.numel() - 3):
            candidate = [int(value) for value in input_ids[index : index + 4]]
            if (
                candidate[0] == time_start_id
                and candidate[1] in self.time_id_to_index
                and candidate[2] in self.time_id_to_index
                and candidate[3] == time_end_id
            ):
                segment_starts.append(index)
        if len(segment_starts) != 1:
            raise ValueError("A PTD response must contain one temporal segment.")
        segment_start = segment_starts[0]
        if segment_start != ref_end + 2 or int(input_ids[ref_end + 1]) != self.newline_id:
            raise ValueError("The object reference and temporal segment require one newline.")

        segment = input_ids[segment_start : segment_start + 4]
        start_time = self.time_id_to_index[int(segment[1])]
        end_time = self.time_id_to_index[int(segment[2])]
        if start_time > end_time:
            raise ValueError("PTD temporal start must not exceed temporal end.")
        sampled_frames = int(data["video_grid_thw"][0, 0])
        if end_time > sampled_frames:
            raise ValueError(
                f"PTD uses <t{end_time}> but the video has {sampled_frames} frames."
            )
        add_block(ref_end_id, segment_start - 1, segment_start, segment)

        cursor = segment_start + 4
        if cursor >= input_ids.numel() or int(input_ids[cursor]) != self.newline_id:
            raise ValueError("The PTD temporal segment must be followed by one newline.")
        cursor += 1
        first_box = cursor
        for time_index in range(start_time, end_time + 1):
            unit = input_ids[cursor : cursor + 7]
            expected_time_id = self.time_index_to_id[time_index]
            if unit.numel() != 7 or [int(unit[0]), int(unit[1]), int(unit[6])] != [
                expected_time_id,
                ids["<|box_start|>"],
                ids["<|box_end|>"],
            ]:
                raise ValueError(f"Invalid PTD box record for <t{time_index}>.")
            coordinates = [
                self.coordinate_id_to_value.get(int(value)) for value in unit[2:6]
            ]
            if (
                None in coordinates
                or coordinates[0] >= coordinates[2]
                or coordinates[1] >= coordinates[3]
            ):
                raise ValueError(
                    f"Invalid PTD coordinates for <t{time_index}>: {coordinates}."
                )
            add_block(expected_time_id, cursor, first_box, unit[1:])
            cursor += 7
            if time_index < end_time:
                if cursor >= input_ids.numel() or int(input_ids[cursor]) != self.newline_id:
                    raise ValueError("PTD box records must be separated by one newline.")
                cursor += 1

        if cursor >= input_ids.numel() or int(input_ids[cursor]) != ids[DEFAULT_IM_END_TOKEN]:
            raise ValueError(
                f"The final PTD box must be followed by {DEFAULT_IM_END_TOKEN}."
            )
        trailing = input_ids[cursor + 1 :]
        if trailing.numel() and not bool(trailing.eq(self.newline_id).all()):
            raise ValueError("Only newlines may follow the assistant end token.")

        prefix_length = input_ids.numel()
        block_inputs = torch.cat(blocks)
        block_targets = torch.cat(targets)
        block_positions = torch.cat(positions)
        block_context_limits = torch.cat(context_limits)
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("PTD requires a tokenizer pad token.")

        data["input_ids"] = torch.cat(
            [input_ids, block_inputs, input_ids.new_tensor([pad_id])]
        )
        data["labels"] = torch.cat(
            [labels, labels.new_tensor([IGNORE_INDEX]), block_targets]
        )
        data["mm_token_type_ids"] = torch.cat(
            [
                data["mm_token_type_ids"],
                torch.zeros(block_inputs.numel() + 1, dtype=torch.long),
            ]
        )
        data["attention_mask"] = data["input_ids"].ne(pad_id).long()
        data["ptd_position_ids"] = torch.cat(
            [
                torch.arange(prefix_length),
                block_positions,
                block_positions[-1:].add(1),
            ]
        )
        data["ptd_context_limits"] = torch.cat(
            [
                torch.zeros(prefix_length, dtype=torch.long),
                block_context_limits,
                torch.zeros(1, dtype=torch.long),
            ]
        )
        data["ptd_prefix_length"] = torch.tensor(prefix_length, dtype=torch.long)
        return data

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        batch_mm_token_type_ids = []
        batch_ptd_position_ids = []
        batch_ptd_context_limits = []
        batch_ptd_prefix_lengths = []

        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            batch_mm_token_type_ids.append(example["mm_token_type_ids"])
            batch_ptd_position_ids.append(example["ptd_position_ids"])
            batch_ptd_context_limits.append(example["ptd_context_limits"])
            batch_ptd_prefix_lengths.append(example["ptd_prefix_length"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)
        mm_token_type_ids = pad_sequence(batch_mm_token_type_ids, padding_side='right', padding_value=0)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'mm_token_type_ids': mm_token_type_ids,
            'ptd_position_ids': pad_sequence(
                batch_ptd_position_ids, padding_side='right', padding_value=0
            ),
            'ptd_context_limits': pad_sequence(
                batch_ptd_context_limits, padding_side='right', padding_value=0
            ),
            'ptd_prefix_lengths': torch.stack(batch_ptd_prefix_lengths),
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict

def make_supervised_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    eval_dataset = None
    if data_args.eval_path is not None:
        eval_dataset = SupervisedDataset(
              data_path=data_args.eval_path,
              processor=processor,
              data_args=data_args,
              model_id=model_id
          )
        
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
