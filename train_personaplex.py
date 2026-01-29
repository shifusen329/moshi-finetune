"""
PersonaPlex Training Script

Extends moshi-finetune for PersonaPlex with:
- dep_q=16 (doubled depth codebooks)
- Voice prompt conditioning
- Text prompt conditioning with <system> tags

Usage:
    torchrun --nproc_per_node=1 train_personaplex.py config.yaml
"""

import dataclasses
import logging
import os
import pprint
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Optional

import fire
import torch
import torch.cuda
import torch.distributed as dist
from safetensors.torch import load_file
from torch.optim import AdamW, lr_scheduler

from finetune.args import TrainArgs, ModelPaths
from finetune.checkpointing import Checkpointer
from finetune.data.data_loader import build_data_loader
from finetune.data.interleaver import InterleavedTokenizer, Interleaver, PrecomputedTokenizer
from finetune.distributed import (
    BACKEND,
    avg_aggregate,
    get_rank,
    get_world_size,
    is_torchrun,
    set_device,
)
from finetune.eval import evaluate
from finetune.loss import compute_loss_with_mask
from finetune.mixed_precision import (
    downcast_mixed_precision,
    prepare_mixed_precision,
    upcast_mixed_precision,
)
from finetune.monitoring.metrics_logger import (
    MetricsLogger,
    eval_log_msg,
    get_eval_logs,
    get_train_logs,
    train_log_msg,
)
from finetune.monitoring.utils import set_logger
from finetune.utils import TrainState, logged_closing, set_random_seed
from moshi.models import loaders
from moshi.models.lm import LMModel

logger = logging.getLogger("train_personaplex")


def is_precomputed_manifest(manifest_path: str) -> bool:
    """Check if a manifest uses pre-computed codes."""
    import json
    with open(manifest_path) as f:
        first_line = f.readline()
        if first_line.strip():
            entry = json.loads(first_line)
            return "codes_path" in entry
    return False


# PersonaPlex-specific LM kwargs (dep_q=16 instead of 8)
_personaplex_lm_kwargs = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 16,  # PersonaPlex uses 16 depth codebooks
    "card": 2048,
    "num_heads": 32,
    "num_layers": 32,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 3000,
    "max_period": 10000,
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_dim": 1024,
    "depformer_dim_feedforward": int(4.125 * 1024),
    "depformer_num_heads": 16,
    "depformer_num_layers": 6,
    "depformer_causal": True,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
    "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
}


def expand_moshiko_to_personaplex(
    state_dict: dict,
    model_state_dict: dict,
    copy_missing_weights: bool = True,
) -> dict:
    """
    Expand Moshiko weights (dep_q=8) to PersonaPlex (dep_q=16).

    This replicates the logic from NVIDIA's PersonaPlex loaders.py:
    1. Expand depformer self_attn weights by concatenation
    2. Copy weights from indices 0-7 to indices 8-15 for gating, linears, etc.
    """
    # Patch 1: expand depformer self_attn weights if needed
    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_state_dict:
            if tensor.shape != model_state_dict[name].shape:
                logger.info(f"Expanding {name}: {tensor.shape} -> {model_state_dict[name].shape}")
                missing = (
                    tensor
                    if copy_missing_weights
                    else model_state_dict[name][tensor.shape[0]:]
                )
                state_dict[name] = torch.concat([tensor, missing], dim=0)

    # Patch 2: fill missing keys by copying 0..7 -> 8..15 for certain groups
    # Note: depformer_emb has 7 items (0-6), others have 8 (0-7)
    # Map: 0→8, 1→9, 2→10, 3→11, 4→12, 5→13, 6→14, 7→15 (if exists)
    if copy_missing_weights:
        to_replace = ["gating", "linears", "depformer_in", "depformer_emb"]
        for name in model_state_dict.keys():
            if name in state_dict:
                continue
            replaced = False
            for old, new in zip(range(8), range(8, 16)):
                for rep in to_replace:
                    needle = f"{rep}.{new}."
                    if needle in name:
                        src = name.replace(needle, f"{rep}.{old}.")
                        if src in state_dict:
                            logger.info(f"Copying {src} -> {name}")
                            state_dict[name] = state_dict[src].clone()
                            replaced = True
                        elif rep == "depformer_emb" and old == 7:
                            # depformer_emb only has indices 0-6, skip 7→15 mapping
                            logger.info(f"Skipping {name} (depformer_emb has no index 7)")
                            replaced = True  # Mark as handled to suppress warning
                        break
                if replaced:
                    break
            if not replaced and name not in state_dict:
                logger.warning(f"Missing weight: {name}")

    return state_dict


def validate_weight_loading(
    model_state_dict: dict,
    loaded_state_dict: dict,
    critical_prefixes: list[str] | None = None,
) -> tuple[int, int, list[str]]:
    """
    Validate that weights loaded correctly.

    Returns:
        (loaded_count, missing_count, missing_critical_keys)
    """
    if critical_prefixes is None:
        critical_prefixes = [
            "transformer.layers",
            "depformer.layers",
            "text_emb",
            "emb.",
        ]

    model_keys = set(model_state_dict.keys())
    loaded_keys = set(loaded_state_dict.keys())

    # Keys that will load successfully (present in both)
    loaded = model_keys & loaded_keys
    missing = model_keys - loaded_keys

    # Check critical prefixes
    missing_critical = []
    for key in missing:
        for prefix in critical_prefixes:
            if key.startswith(prefix):
                missing_critical.append(key)
                break

    return len(loaded), len(missing), missing_critical


def get_personaplex_model(
    moshi_weights: str | Path,
    lm_kwargs: Optional[dict] = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    expand_from_moshiko: bool = True,
    **kwargs,
) -> LMModel:
    """
    Load a PersonaPlex model with dep_q=16.

    Args:
        moshi_weights: Path to model weights (Moshiko or PersonaPlex)
        lm_kwargs: Override LM configuration
        device: Target device
        dtype: Model dtype
        expand_from_moshiko: If True, expand Moshiko (dep_q=8) weights to PersonaPlex (dep_q=16)
    """
    if lm_kwargs is None:
        lm_kwargs = dict(_personaplex_lm_kwargs)
    else:
        lm_kwargs = dict(lm_kwargs)

    # Handle any extra kwargs
    lm_kwargs.pop("depformer_causal", None)  # deprecated
    lora = lm_kwargs.pop("lora", False)
    lora_rank = lm_kwargs.pop("lora_rank", 128)
    lora_scaling = lm_kwargs.pop("lora_scaling", 2.0)

    # Merge any overrides
    for k, v in kwargs.items():
        if k in lm_kwargs:
            lm_kwargs[k] = v

    init_device = device
    if moshi_weights is not None:
        init_device = torch.device('meta')

    model = LMModel(device=init_device, dtype=dtype, **lm_kwargs)

    if moshi_weights is not None:
        moshi_weights = str(moshi_weights)

        if moshi_weights.endswith(".safetensors"):
            dev = torch.device(device) if isinstance(device, str) else device
            if dev.type == "mps":
                state_dict = load_file(moshi_weights, device="cpu")
            else:
                state_dict = load_file(moshi_weights, device=dev.type)
        else:
            with open(moshi_weights, "rb") as f:
                state_dict = torch.load(f, map_location="cpu")

        # Convert dtypes
        for key, value in state_dict.items():
            if value.dtype.is_floating_point:
                state_dict[key] = value.to(dtype=dtype)

        # Expand weights if loading from Moshiko
        if expand_from_moshiko:
            model_state_dict = {k: v for k, v in model.state_dict().items()}
            state_dict = expand_moshiko_to_personaplex(state_dict, model_state_dict)

        # Validate weight loading before applying
        model_sd = {k: v for k, v in model.state_dict().items()}
        loaded_count, missing_count, missing_critical = validate_weight_loading(
            model_sd, state_dict
        )

        logger.info(f"Weight loading: {loaded_count} matched, {missing_count} missing")

        if missing_critical:
            logger.error(f"CRITICAL weights missing ({len(missing_critical)}):")
            for k in missing_critical[:10]:
                logger.error(f"  {k}")
            if len(missing_critical) > 10:
                logger.error(f"  ... and {len(missing_critical) - 10} more")
            raise RuntimeError(
                f"Critical weights missing! {len(missing_critical)} keys from "
                f"transformer/depformer/embeddings not found in checkpoint. "
                f"This likely means a checkpoint format mismatch."
            )

        model.load_state_dict(state_dict, strict=False, assign=True)

    if lora:
        from moshi.modules.lora import replace_all_linear_with_lora
        replace_all_linear_with_lora(model, lora_rank, lora_scaling, device=device)

    model.eval()
    return model


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as PersonaPlex expects."""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def train(config: str):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    set_logger(logging.INFO)

    with ExitStack() as exit_stack:
        _train(args, exit_stack)
    logger.info("Closed everything!")


def _train(args: TrainArgs, exit_stack: ExitStack):
    # 1. Initial setup
    set_random_seed(args.seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Init NCCL
    if "LOCAL_RANK" in os.environ:
        set_device()
        logger.info("Going to init comms...")
        dist.init_process_group(backend=BACKEND)
    else:
        logger.warning(
            "PyTorch environment is not correctly initialized. "
            "Running in single-GPU mode."
        )

    # 2. Init run dir
    main_logger_info(f"Run dir: {args.run_dir}")
    run_dir = Path(args.run_dir)

    if is_torchrun():
        if run_dir.exists() and not args.overwrite_run_dir:
            raise RuntimeError(
                f"Run dir {run_dir} already exists. "
                "Set overwrite_run_dir=True or remove the directory."
            )
        elif run_dir.exists():
            main_logger_info(f"Removing run dir {run_dir}...")
            shutil.rmtree(run_dir)

    dist.barrier() if dist.is_initialized() else None
    run_dir.mkdir(exist_ok=True, parents=True)

    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        args.save(args_path)

    main_logger_info(f"TrainArgs: {pprint.pformat(dataclasses.asdict(args))}")

    # 3. Get loggers
    metrics_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="train",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(metrics_logger, "metrics_logger"))

    eval_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="eval",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(eval_logger, "eval_logger"))

    # 4. Load Mimi (audio tokenizer) - same as base Moshi
    main_logger_info("Loading checkpoint info...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=args.moshi_paths.hf_repo_id,
        moshi_weights=args.moshi_paths.moshi_path,
        mimi_weights=args.moshi_paths.mimi_path,
        tokenizer=args.moshi_paths.tokenizer_path,
        config_path=args.moshi_paths.config_path,
    )

    # Check if using pre-computed codes
    use_precomputed = is_precomputed_manifest(args.data.train_data)
    if use_precomputed:
        main_logger_info("Detected pre-computed codes manifest - skipping Mimi encoder loading")
        mimi = None
        # Mimi frame rate is 12.5 Hz (24000 sample_rate / 1920 hop_length)
        frame_rate = 12.5
    else:
        main_logger_info("Loading Mimi...")
        mimi = checkpoint_info.get_mimi(device="cuda")
        mimi.eval()
        for p in mimi.parameters():
            p.requires_grad = False
        frame_rate = mimi.frame_rate

    # 5. Load PersonaPlex model with dep_q=16
    main_logger_info("Loading PersonaPlex model (dep_q=16)...")
    param_dtype = getattr(torch, args.param_dtype)

    # Determine if we need to expand from Moshiko
    # Allow explicit override via config, otherwise auto-detect from repo name
    expand_from_moshiko = getattr(args, "expand_from_moshiko", None)
    if expand_from_moshiko is None:
        expand_from_moshiko = "moshiko" in str(args.moshi_paths.hf_repo_id or "").lower()
        main_logger_info(f"Auto-detected expand_from_moshiko={expand_from_moshiko}")
    else:
        main_logger_info(f"Using configured expand_from_moshiko={expand_from_moshiko}")

    lm_kwargs = dict(_personaplex_lm_kwargs)
    lm_kwargs["lora"] = args.lora.enable
    lm_kwargs["lora_rank"] = args.lora.rank
    lm_kwargs["lora_scaling"] = args.lora.scaling
    # Note: gradient_checkpointing handled separately after model creation

    model = get_personaplex_model(
        moshi_weights=checkpoint_info.moshi_weights,
        lm_kwargs=lm_kwargs,
        device="cuda",
        dtype=param_dtype,
        expand_from_moshiko=expand_from_moshiko,
    )

    # Handle gradient checkpointing after model creation
    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            main_logger_info("Gradient checkpointing enabled")
        else:
            logger.warning(
                "gradient_checkpointing requested but model doesn't support it"
            )

    # Freeze non-LoRA params if using LoRA
    if args.lora.enable and not args.full_finetuning:
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            elif args.lora.ft_embed and "emb" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    num_params = sum(p.numel() for p in model.parameters())
    num_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    main_logger_info(f"Trainable: {num_train:,} / {num_params:,} ({100*num_train/num_params:.2f}%)")

    # 6. Setup text tokenizer and interleaver
    spm = checkpoint_info.get_text_tokenizer()

    interleaver = Interleaver(
        spm,
        frame_rate,
        model.text_padding_token_id,
        model.end_of_text_padding_id,
        model.zero_token_id,
        keep_main_only=True,
    )

    if use_precomputed:
        main_logger_info("Using PrecomputedTokenizer for fast data loading")
        instruct_tokenizer = PrecomputedTokenizer(
            interleaver, duration_sec=args.duration_sec, frame_rate=frame_rate
        )
    else:
        instruct_tokenizer = InterleavedTokenizer(
            mimi, interleaver, duration_sec=args.duration_sec
        )

    # 7. Load data
    data_loader = build_data_loader(
        instruct_tokenizer=instruct_tokenizer,
        args=args.data,
        batch_size=args.batch_size,
        seed=args.seed,
        rank=get_rank(),
        world_size=get_world_size(),
        is_eval=False,
    )

    # 8. Optimizer
    optim_dtype = torch.float32
    optimizer = AdamW(
        model.parameters(),
        lr=args.optim.lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.optim.lr,
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
    )

    state = TrainState(args.max_steps)

    # 9. Checkpointer
    if args.do_ckpt:
        checkpointer = Checkpointer(
            model=model,
            state=state,
            config=lm_kwargs,
            run_dir=run_dir,
            optimizer=optimizer,
            num_ckpt_keep=args.num_ckpt_keep,
            full_finetuning=args.full_finetuning,
        )

    # 10. Mixed precision setup
    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    # 11. Training loop
    model.train()
    torch.cuda.empty_cache()

    main_logger_info("Starting training...")

    while state.step < args.max_steps:
        state.start_step()
        is_last_step = state.step == args.max_steps

        optimizer.zero_grad()

        loss = torch.tensor([0.0], device="cuda")
        n_batch_tokens: int = 0
        n_real_tokens: int = 0

        for i in range(args.num_microbatches):
            batch = next(data_loader)
            codes = batch.codes

            condition_tensors = None
            if batch.condition_attributes is not None:
                condition_tensors = model.condition_provider.prepare(
                    batch.condition_attributes
                )

            # Forward pass
            output = model(codes=codes, condition_tensors=condition_tensors)

            # Text loss
            text_loss = compute_loss_with_mask(
                output.text_logits,
                codes[:, : model.audio_offset],
                output.text_mask,
                mode="text",
                text_padding_weight=args.text_padding_weight,
                text_padding_ids={
                    model.text_padding_token_id,
                    model.end_of_text_padding_id,
                },
            )

            # Audio loss - use dep_q=16 for PersonaPlex
            audio_loss = compute_loss_with_mask(
                output.logits,
                codes[:, model.audio_offset : model.audio_offset + model.dep_q],
                output.mask,
                mode="audio",
                first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
            )

            mb_loss = text_loss + audio_loss
            mb_loss.backward()

            loss += mb_loss.detach()
            n_batch_tokens += output.text_mask.numel() + output.mask.numel()
            n_real_tokens += (
                torch.sum(output.text_mask).item() + torch.sum(output.mask).item()
            )

            if i < args.num_microbatches - 1:
                torch.cuda.synchronize()

        if args.num_microbatches > 1:
            loss /= args.num_microbatches
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.div_(args.num_microbatches)

        # Upcast for optimizer
        upcast_mixed_precision(model.parameters(), optim_dtype=optim_dtype)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        # Optimizer step
        optimizer.step()

        # Downcast back
        downcast_mixed_precision(model.parameters(), param_dtype=param_dtype)

        last_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Aggregate loss across ranks
        loss_item = loss.item()
        avg_loss = avg_aggregate(loss_item) if dist.is_initialized() else loss_item

        # Evaluation
        if args.do_eval and (
            (args.eval_freq > 0 and state.step % args.eval_freq == 0) or is_last_step
        ):
            # Recreate eval iterator each time to avoid exhaustion
            eval_data_loader = build_data_loader(
                instruct_tokenizer=instruct_tokenizer,
                args=args.data,
                batch_size=args.batch_size,
                seed=None,
                rank=get_rank(),
                world_size=get_world_size(),
                is_eval=True,
            )
            evaluate(model, eval_data_loader, state, args)

            eval_logs = get_eval_logs(
                state.step,
                avg_loss,
                state.this_eval_perplexity,
                state.this_eval_loss,
            )

            main_logger_info(eval_log_msg(eval_logs))
            eval_logger.log(eval_logs, step=state.step)

        # Timing
        state.end_step(n_batch_tokens)

        if state.step % args.log_freq == 0:
            train_logs = get_train_logs(
                state,
                avg_loss,
                n_real_tokens,
                last_lr,
                torch.cuda.max_memory_allocated(),
                torch.cuda.memory_allocated(),
                args,
            )
            main_logger_info(train_log_msg(state, logs=train_logs, loss=avg_loss))
            metrics_logger.log(train_logs, step=state.step)

        # Checkpointing
        if args.do_ckpt and (
            (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
        ):
            checkpointer.save_checkpoint(
                save_only_lora=not args.full_finetuning and args.save_adapters,
                dtype=param_dtype,
            )

    main_logger_info("Training complete!")


if __name__ == "__main__":
    fire.Fire(train)
