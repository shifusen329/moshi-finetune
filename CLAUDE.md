# Moshi Finetune - Development Notes

## PersonaPlex Architecture Comparison

PersonaPlex (nvidia/personaplex-7b-v1) is a finetune of Moshiko (kyutai/moshiko-pytorch-bf16) with persona conditioning.

### Core Architecture (Shared)

| Component | Value |
|-----------|-------|
| **Temporal Transformer** | |
| dim | 4096 |
| num_layers | 32 |
| num_heads | 32 |
| hidden_scale | 4.125 (FFN = 16,896) |
| gating | SiLU |
| norm | RMS Norm (f32) |
| positional_embedding | RoPE |
| context | 3000 |
| **Depth Transformer (Depformer)** | |
| depformer_dim | 1024 |
| depformer_num_layers | 6 |
| depformer_num_heads | 16 |
| depformer_dim_feedforward | 4224 |
| depformer_multi_linear | True (parallel FFNs per codebook) |
| depformer_context | 8 |
| **Audio/Text** | |
| text_card (vocab) | 32000 (+1 padding = 32001) |
| card (codebook size) | 2048 (+1 = 2049) |
| Sample rate | 24kHz |
| Frame rate | 12.5 Hz |

### Key Difference: Depth Codebooks

| Parameter | Moshiko | PersonaPlex |
|-----------|---------|-------------|
| `n_q` | 16 | 16 |
| `dep_q` | **8** | **16** |

PersonaPlex doubles the depth codebooks from 8 to 16, which affects:
- `depformer.self_attn.in_proj_weight`: [24576, 1024] → [49152, 1024]
- `depformer.self_attn.out_proj.weight`: [8192, 1024] → [16384, 1024]
- `depformer.gating.N`: N = 0..7 → N = 0..15
- `linears.N`: N = 0..7 → N = 0..15
- `depformer_in.N`: N = 0..7 → N = 0..15
- `depformer_emb.N`: N = 0..7 → N = 0..15

### PersonaPlex Weight Expansion

When loading Moshiko weights into PersonaPlex architecture:
1. Depformer self_attn weights are concatenated (doubled)
2. Codebook weights 0-7 are copied to indices 8-15

See `train_personaplex.py:expand_moshiko_to_personaplex()` for implementation.

### PersonaPlex Conditioning

1. **Voice Prompt**: Pre-computed `.pt` files with embeddings + KV cache, or raw audio encoded through Mimi
2. **Text Prompt**: Wrapped with `<system>` tags: `<system> {prompt} <system>`
3. **Injection Order**: Voice prompt → Audio silence → Text prompt → Audio silence

### Voice Prompt Files

PersonaPlex includes 18 pre-packaged voice embeddings in `voices/`:
```
Natural (female): NATF0, NATF1, NATF2, NATF3
Natural (male):   NATM0, NATM1, NATM2, NATM3
Variety (female): VARF0, VARF1, VARF2, VARF3, VARF4
Variety (male):   VARM0, VARM1, VARM2, VARM3, VARM4
```

---

## Code Changes

### annotate.py

Added multi-worker support and PersonaPlex mode:

```bash
# Multi-worker processing
python annotate.py egs.jsonl.gz --local --num-workers 4

# PersonaPlex mode - adds text_prompt and voice_prompt to output JSON
python annotate.py egs.jsonl.gz --local --personaplex \
    --text-prompt "You are a helpful assistant." \
    --voice-prompt "NATF2.pt"
```

**Output format with `--personaplex`:**
```json
{
  "alignments": [["word", [start, end], "SPEAKER_MAIN"], ...],
  "text_prompt": "<system> You are a helpful assistant. <system>",
  "voice_prompt": "NATF2.pt"
}
```

### finetune/distributed.py

Fixed `CUDA_VISIBLE_DEVICES` handling - now gracefully handles when the env var is not set (uses all available GPUs).

### train.py

Added `torch._dynamo.config.disable = True` to work around torch.compile/inductor stride mismatch bugs when running with FSDP on Blackwell (B200) GPUs. The error manifests as:
```
AssertionError: expected size 128==128, stride 1250==1280 at dim=0
```
This globally disables dynamo compilation, forcing eager mode execution.

Added support for pre-computed Mimi codes:
- Auto-detects pre-computed manifests by checking for `codes_path` field
- Creates `PrecomputedTokenizer` instead of `InterleavedTokenizer` when using pre-computed data
- Skips loading Mimi encoder weights when using pre-computed codes (saves ~2GB GPU memory)

### precompute_codes.py

New script to pre-encode audio with Mimi for faster training. Eliminates the encoding bottleneck during training by processing all audio files offline.

```bash
# Pre-compute codes for training and eval data
python precompute_codes.py manifest_train.jsonl manifest_eval.jsonl --output-dir data_encoded/

# Options
--hf-repo          HuggingFace repo for Mimi (default: kyutai/moshiko-pytorch-bf16)
--num-load-threads Number of threads for loading audio (default: 4)
--device           Device for encoding (default: cuda)
```

**Output:** Creates `*_precomputed.jsonl` manifests with `codes_path` field pointing to `.pt` files.

### finetune/data/interleaver.py

Added `PrecomputedTokenizer` class:
- Loads pre-computed Mimi codes from `.pt` files instead of encoding on-the-fly
- Caches loaded codes in memory to avoid repeated disk reads
- Slices codes based on `start_sec` for chunking
- Same interface as `InterleavedTokenizer` for seamless integration

### finetune/data/dataset.py

Updated `get_dataset_iterator()` to support both modes:
- Auto-detects pre-computed vs live encoding based on manifest format
- `get_live_encoding_iterator()`: Original behavior using sphn + Mimi encoding
- `get_precomputed_iterator()`: Fast iterator loading pre-computed `.pt` files

### Performance Impact

The original data loading bottleneck was caused by Mimi encoding running synchronously in the training loop. With pre-computed codes:

| Metric | Before | After |
|--------|--------|-------|
| Data loading | ~0.37s/sample | ~0.001s/sample |
| GPU utilization | 100% (waiting) | 100% (training) |
| Effective throughput | ~30k words/sec | Expected 10-20x improvement |

### train_personaplex.py

New training script for PersonaPlex with:
- `dep_q=16` model configuration
- Weight expansion from Moshiko (8 → 16 codebooks)
- Compatible with existing data pipeline

```bash
torchrun --nproc_per_node=1 train_personaplex.py example/personaplex_config.yaml
```

### example/personaplex_config.yaml

Example config for PersonaPlex finetuning with LoRA.

### pyproject.toml

Updated dependencies for Blackwell (sm_120) compatibility:
- `triton>=3.2` - required for Blackwell kernel support
- `llvmlite>=0.44` - LLVM bindings for Blackwell codegen
- `numba>=0.61` - JIT compiler with Blackwell support

---

## Local Model Cache

Models cached at `/mnt/cache/huggingface/hub/`:
- `models--kyutai--moshiko-pytorch-bf16/`
- `models--nvidia--personaplex-7b-v1/`

---

## Dependencies Notes

For Blackwell GPUs (sm_120), requires PyTorch with CUDA 12.8+:
```bash
uv pip install 'torch>=2.7.0,<2.10.0' torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
