#!/usr/bin/env python3
"""
Phase D: Parallel + Chinese behavioral test

Runs the 32 parallel (non-CCP) prompts and 16 Chinese-language prompts
through each model to enable CCP vs parallel discrimination analysis.

The CCP prompts were already run during adversarial probing. This script
fills in the other half: structurally identical prompts about non-CCP topics.

Output: runs/parallel_behavioral_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from ablation_ridge_regression import (
    load_model_and_tokenizer,
    _resolve_device,
    _default_generate_kwargs,
    _tokenize,
    _to_device,
    _decode_response,
    _strip_thinking,
    _is_refusal,
    DEFAULT_REFUSAL_MARKERS,
)

from political_prompts_adversarial import (
    ADVERSARIAL_CCP,
    ADVERSARIAL_PARALLEL,
    ADVERSARIAL_CHINESE,
    TOPIC_LABELS,
    INTENSITY_LABELS,
    PARALLEL_TOPICS,
)


MODELS = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "microsoft/Phi-4-mini-instruct",
    "zai-org/GLM-Z1-9B-0414",
    "zai-org/GLM-4-9B-0414",
]

RESULTS_FILE = Path("runs/parallel_behavioral_results.json")


def generate_responses(model, tokenizer, prompts, label="eval"):
    """Generate responses for a list of prompts."""
    dev = _resolve_device(model)
    gen_cfg = _default_generate_kwargs(tokenizer)
    gen_cfg["max_new_tokens"] = 128

    results = []
    for i, prompt in enumerate(prompts):
        print(f"    [{label}] {i+1}/{len(prompts)}: {prompt[:60]}...", end="\r", flush=True)

        encoded = _tokenize(tokenizer, prompt, max_length=256, padding=False, chat_template=True)
        encoded = _to_device(encoded, dev)
        input_len = encoded["input_ids"].shape[-1]

        with torch.no_grad():
            out = model.generate(**encoded, **gen_cfg)

        response = _decode_response(tokenizer, prompt, out[0], input_length=input_len)
        refused = _is_refusal(response, DEFAULT_REFUSAL_MARKERS)

        results.append({
            "prompt": prompt,
            "response": response,
            "refused": refused,
        })
        tag = "REFUSED" if refused else "COMPLIED"
        print(f"    [{label}] {i+1}/{len(prompts)}: {tag} — {prompt[:60]}")

    n_refused = sum(1 for r in results if r["refused"])
    print(f"    [{label}] Refusal rate: {n_refused}/{len(prompts)} ({n_refused/len(prompts):.1%})")
    return results


def main():
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results for resume
    all_results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)
        print(f"Loaded existing results: {list(all_results.keys())}")

    for model_id in MODELS:
        if model_id in all_results:
            print(f"\nSKIP: {model_id} (already done)")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_id}")
        print(f"{'='*60}")

        t0 = time.time()
        model, tokenizer = load_model_and_tokenizer(model_id)

        print(f"\n[1/2] Running 32 parallel (non-CCP) prompts...")
        parallel_results = generate_responses(
            model, tokenizer, ADVERSARIAL_PARALLEL, label="parallel"
        )

        print(f"\n[2/2] Running 16 Chinese-language CCP prompts...")
        chinese_results = generate_responses(
            model, tokenizer, ADVERSARIAL_CHINESE, label="chinese"
        )

        # Build structured results with metadata
        parallel_data = []
        for i, r in enumerate(parallel_results):
            parallel_data.append({
                **r,
                "topic": TOPIC_LABELS[i],
                "intensity": INTENSITY_LABELS[i],
                "parallel_topic": PARALLEL_TOPICS[i],
            })

        chinese_data = []
        for i, r in enumerate(chinese_results):
            # Chinese prompts are L3+L4 for all 8 topics
            topic_idx = i % 8
            intensity = "accusatory" if i < 8 else "provocative"
            topics = ["Tiananmen Square", "Xinjiang / Uyghurs", "Tibet", "Taiwan",
                       "Hong Kong", "Xi Jinping / CCP", "COVID-19 origins", "Censorship / Great Firewall"]
            chinese_data.append({
                **r,
                "topic": topics[topic_idx],
                "intensity": intensity,
                "language": "zh",
            })

        elapsed = time.time() - t0
        all_results[model_id] = {
            "parallel": parallel_data,
            "chinese": chinese_data,
            "elapsed_seconds": elapsed,
        }

        # Checkpoint after each model
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved checkpoint ({elapsed:.0f}s)")

        # Free GPU memory
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: CCP vs Parallel Discrimination")
    print(f"{'='*60}")
    for model_id, data in all_results.items():
        par_refused = sum(1 for r in data["parallel"] if r["refused"])
        cn_refused = sum(1 for r in data["chinese"] if r["refused"])
        par_total = len(data["parallel"])
        cn_total = len(data["chinese"])
        print(f"  {model_id}:")
        print(f"    Parallel: {par_refused}/{par_total} refused")
        print(f"    Chinese:  {cn_refused}/{cn_total} refused")


if __name__ == "__main__":
    main()
