#!/usr/bin/env python3
"""Pre-compute Mimi audio codes for faster training.

This script processes all audio files in a manifest and saves the encoded
Mimi codes to disk. During training, these pre-computed codes are loaded
directly, eliminating the encoding bottleneck.

Usage:
    python precompute_codes.py manifest_train.jsonl --output-dir data_encoded/
    python precompute_codes.py manifest_train.jsonl manifest_eval.jsonl --output-dir data_encoded/

The script will:
1. Load each audio file referenced in the manifest(s)
2. Encode the full audio with Mimi
3. Save the codes as .pt files alongside metadata

A new manifest is generated pointing to the pre-computed files.
"""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from moshi.models import loaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    """Load audio file and resample if necessary."""
    import sphn
    wav = sphn.read(path, sample_rate=sample_rate)
    return wav


def process_manifest(manifest_path: Path) -> list[dict]:
    """Load manifest and return list of entries."""
    entries = []
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def encode_worker(
    mimi,
    input_queue: Queue,
    output_queue: Queue,
    sample_rate: int,
    device: str = "cuda",
):
    """Worker that encodes audio on GPU."""
    while True:
        item = input_queue.get()
        if item is None:
            break

        idx, entry, audio = item
        try:
            with torch.no_grad():
                audio_tensor = torch.from_numpy(audio).float().to(device)
                # Mimi expects [batch, channels, samples] or [channels, samples]
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)  # [1, samples]
                codes = mimi.encode(audio_tensor.unsqueeze(0))  # [1, n_q, T]
                codes = codes.squeeze(0).cpu()  # [n_q, T]
            output_queue.put((idx, entry, codes, None))
        except Exception as e:
            output_queue.put((idx, entry, None, str(e)))

        input_queue.task_done()


def load_worker(
    entries: list[tuple[int, dict]],
    output_queue: Queue,
    sample_rate: int,
    num_threads: int = 4,
):
    """Worker that loads audio files from disk."""
    def load_single(item):
        idx, entry = item
        try:
            audio = load_audio(entry["path"], sample_rate)
            return (idx, entry, audio)
        except Exception as e:
            logger.error(f"Failed to load {entry['path']}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for result in executor.map(load_single, entries):
            if result is not None:
                output_queue.put(result)

    # Signal end
    output_queue.put(None)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute Mimi codes for training")
    parser.add_argument("manifests", nargs="+", type=Path, help="Input manifest file(s)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for encoded files")
    parser.add_argument("--hf-repo", type=str, default="kyutai/moshiko-pytorch-bf16", help="HuggingFace repo for Mimi")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for encoding (currently unused)")
    parser.add_argument("--num-load-threads", type=int, default=4, help="Number of threads for loading audio")
    parser.add_argument("--device", type=str, default="cuda", help="Device for encoding")
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load Mimi
    logger.info(f"Loading Mimi from {args.hf_repo}...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(hf_repo=args.hf_repo)
    mimi = checkpoint_info.get_mimi(device=args.device)
    mimi.eval()
    sample_rate = mimi.sample_rate
    frame_rate = mimi.frame_rate
    logger.info(f"Mimi loaded: sample_rate={sample_rate}, frame_rate={frame_rate}")

    # Process each manifest
    for manifest_path in args.manifests:
        logger.info(f"Processing {manifest_path}...")
        entries = process_manifest(manifest_path)
        logger.info(f"Found {len(entries)} entries")

        # Set up queues
        load_queue: Queue = Queue(maxsize=32)
        encode_done: Queue = Queue()

        # Start load thread
        indexed_entries = list(enumerate(entries))
        load_thread = Thread(
            target=load_worker,
            args=(indexed_entries, load_queue, sample_rate, args.num_load_threads),
        )
        load_thread.start()

        # Process entries
        new_manifest_entries = [None] * len(entries)
        num_processed = 0
        num_failed = 0

        pbar = tqdm(total=len(entries), desc="Encoding")

        while True:
            item = load_queue.get()
            if item is None:
                break

            idx, entry, audio = item
            original_path = Path(entry["path"])

            try:
                with torch.no_grad():
                    audio_tensor = torch.from_numpy(audio).float().to(args.device)
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    codes = mimi.encode(audio_tensor.unsqueeze(0))  # [1, n_q, T]
                    codes = codes.squeeze(0).cpu()  # [n_q, T]

                # Save codes
                output_filename = original_path.stem + "_codes.pt"
                output_path = args.output_dir / output_filename

                # Save codes and metadata
                torch.save({
                    "codes": codes,
                    "sample_rate": sample_rate,
                    "frame_rate": frame_rate,
                    "duration": entry["duration"],
                    "original_path": str(original_path),
                }, output_path)

                # Create new manifest entry
                new_entry = {
                    "path": str(original_path),  # Keep original for alignment JSON
                    "codes_path": str(output_path),
                    "duration": entry["duration"],
                }
                new_manifest_entries[idx] = new_entry
                num_processed += 1

            except Exception as e:
                logger.error(f"Failed to encode {entry['path']}: {e}")
                num_failed += 1

            pbar.update(1)

        pbar.close()
        load_thread.join()

        # Write new manifest
        output_manifest = args.output_dir / f"{manifest_path.stem}_precomputed.jsonl"
        with open(output_manifest, "w") as f:
            for entry in new_manifest_entries:
                if entry is not None:
                    f.write(json.dumps(entry) + "\n")

        logger.info(f"Wrote {output_manifest} with {num_processed} entries ({num_failed} failed)")

    logger.info("Done!")


if __name__ == "__main__":
    main()
