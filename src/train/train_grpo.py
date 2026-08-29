import os
import torch
from peft import LoraConfig
import ast
import pathlib
from transformers import (
    AutoProcessor, 
    BitsAndBytesConfig, 
    HfArgumentParser, 
)
from model.load_model import get_qwen_vl_generation_backbone, load_qwen_vl_generation_model
from model.ptd_generation import build_ptd_token_ids, configure_ptd_model
from model.ptd_tokens import (
    NUM_TIME_TOKENS,
    ptd_token_ids as validate_ptd_token_ids,
    train_only_ptd_token_rows,
)

from trainer import QwenGRPOTrainer
from dataset import make_grpo_data_module
from dataset.data_utils import patch_processor_with_time_tokens, patch_qwen3_video_processor
from params import DataArguments, ModelArguments, GRPOArguments
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, patch_deepspeed_zero3_peft_hooks, safe_save_model_for_hf_trainer
from utils import load_reward_funcs

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    backbone = get_qwen_vl_generation_backbone(model)
    vision_tower = backbone.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = backbone.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = backbone.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

    if hasattr(backbone.visual, "deepstack_merger_list"):
        deepstack_merger_list_params = backbone.visual.deepstack_merger_list.parameters()
        set_requires_grad(deepstack_merger_list_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    backbone = get_qwen_vl_generation_backbone(model)
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = backbone.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def unfreeze_topk_layers(model, k_llm: int = 0, k_vis: int = 0):
    backbone = get_qwen_vl_generation_backbone(model)

    if k_llm and hasattr(backbone, "language_model") and hasattr(backbone.language_model, "layers"):
        for layer in backbone.language_model.layers[-k_llm:]:
            for p in layer.parameters():
                p.requires_grad = True

    if k_vis and hasattr(backbone, "visual") and hasattr(backbone.visual, "blocks"):
        for blk in backbone.visual.blocks[-k_vis:]:
            for p in blk.parameters():
                p.requires_grad = True



def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, GRPOArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError("You cannot set both `nframes` and `fps` at the same time. Please set only one of them.")
    if data_args.fps != 2:
        raise ValueError("PTD GRPO uses 2 FPS.")
    if data_args.max_frames != 64:
        raise ValueError("PTD GRPO uses at most 64 sampled frames.")
    if data_args.temporal_patch_size != 1:
        raise ValueError("PTD GRPO uses one logical frame per temporal patch.")
    if not training_args.disable_flash_attn2:
        raise ValueError("PTD GRPO requires SDPA attention.")
    if training_args.use_vllm:
        raise ValueError("PTD GRPO requires local model rollouts.")
    if not training_args.lora_enable:
        raise ValueError("PTD GRPO requires LoRA training.")
    if training_args.bits != 16:
        raise ValueError("PTD GRPO requires 16-bit model weights.")
    if training_args.weight_decay != 0:
        raise ValueError("PTD token-row training requires zero weight decay.")
    if not training_args.freeze_vision_tower or not training_args.freeze_merger:
        raise ValueError("PTD GRPO keeps the vision encoder and merger frozen.")
    if training_args.vision_lora:
        raise ValueError("PTD GRPO applies LoRA only to the language model.")
    if training_args.unfreeze_topk_llm or training_args.unfreeze_topk_vision:
        raise ValueError("PTD GRPO does not unfreeze base language or vision layers.")

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]
        training_args.lora_namespan_exclude += ["embed_tokens", "lm_head"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    model = load_qwen_vl_generation_model(
        model_args.model_id,
        dtype=compute_dtype,
        attn_implementation="sdpa" if training_args.disable_flash_attn2 else "flash_attention_2",
        **bnb_model_from_pretrained_args,
    )
    if model.config.model_type != "qwen3_vl":
        raise ValueError("PTD GRPO requires a Qwen3-VL model.")
    if getattr(model.config, "ptd_block_size", None) != 6:
        raise ValueError("PTD GRPO must start from a merged PTD SFT checkpoint.")
    model.config.use_cache = False
    configure_ptd_model(model)
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    unfreeze_topk_layers(
        model_to_configure,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )

    if training_args.gradient_checkpointing:
        if training_args.vision_lora:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        else:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        
        model.enable_input_require_grads()

    if training_args.bits in [4,8]:
        model.config.dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)

    peft_config = None

    if training_args.lora_enable:
        patch_deepspeed_zero3_peft_hooks()
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            modules_to_save=["embed_tokens", "lm_head"],
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)

    processor = AutoProcessor.from_pretrained(model_args.model_id)
    validate_ptd_token_ids(
        processor.tokenizer, num_time_tokens=NUM_TIME_TOKENS
    )
    if getattr(model.config, "ptd_num_time_tokens", None) != NUM_TIME_TOKENS:
        raise ValueError("PTD GRPO must start from a merged PTD SFT checkpoint.")
    ptd_tokens = build_ptd_token_ids(
        processor.tokenizer,
        max_time_tokens=NUM_TIME_TOKENS,
    )
    output_embeddings = model.get_output_embeddings()
    if (
        model.get_input_embeddings().num_embeddings != len(processor.tokenizer)
        or output_embeddings is None
        or output_embeddings.weight.shape[0] != len(processor.tokenizer)
    ):
        raise ValueError("PTD GRPO must start from a merged PTD SFT checkpoint.")
    patch_qwen3_video_processor(processor)
    patch_processor_with_time_tokens(
        processor,
        max_time_tokens=NUM_TIME_TOKENS,
    )
    processor.image_processor.do_resize = False

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    dataset_module = make_grpo_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args)

    reward_names = tuple(
        name.strip() for name in training_args.reward_names.split(",") if name.strip()
    )
    if reward_names != ("temporal_iou_reward", "spatial_reward"):
        raise ValueError(
            "PTD GRPO uses --reward_names "
            "temporal_iou_reward,spatial_reward."
        )
    available_rewards = {
        reward.__name__: reward for reward in load_reward_funcs("train.reward_funcs")
    }
    reward_funcs = [
        available_rewards[name]
        for name in reward_names
        if name in available_rewards
    ]
    if len(reward_funcs) != 2:
        raise ValueError("PTD GRPO requires temporal_iou_reward and spatial_reward.")
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    trainer = QwenGRPOTrainer(
        model=model,
        train_dataset=dataset_module["train_dataset"],
        eval_dataset=dataset_module["eval_dataset"],
        processing_class=processor,
        reward_funcs=reward_funcs,
        args=training_args,
        peft_config=peft_config,
        ptd_generation_config={
            "block_size": 6,
            "max_time_tokens": 64,
        },
    )
    trainable_ptd_tokens = [
        ptd_tokens["null"],
        ptd_tokens["text_mask"],
        ptd_tokens["time_start"],
        ptd_tokens["time_end"],
        *ptd_tokens["coord_id_to_value"],
        *ptd_tokens["ordered_time_tokens"],
    ]
    train_only_ptd_token_rows(trainer.model, trainable_ptd_tokens)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    trained_model = trainer.model
    trained_model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            trained_model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            trained_model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            trained_model.config.save_pretrained(training_args.output_dir)
            trained_model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
