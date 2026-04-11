#!/usr/bin/env python3
"""
Ridge residualization ablation scale sweep.

Tests the hypothesis that ridge-cleaned political directions need lower ablation
scale (alpha) than raw CAA directions, because ridge cleaning concentrates the
direction in the political subspace, making it more potent per unit of alpha.

Sweeps alpha in [5, 10, 12, 15, 20] for both raw and ridge-cleaned political
directions at layer 9 (Qwen3-8B), measuring refusal rate and coherence.

Usage:
  python ridge_scale_sweep.py
"""

import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Import from existing modules
from ablation_ridge_regression import (
    load_model_and_tokenizer,
    mean_hidden_state_at_layer,
    _resolve_transformer_layers,
)
from political_probe import (
    POLITICAL_REFUSED_PROMPTS,
    POLITICAL_ANSWERED_PROMPTS,
    GENERIC_HARMFUL,
    GENERIC_HARMLESS,
    DEFAULT_REFUSAL_MARKERS,
    _tokenize,
    _to_device,
    _resolve_device,
    _default_generate_kwargs,
    _decode_response,
    _is_refusal,
    _strip_thinking,
)

MODEL_ID = "Qwen/Qwen3-8B"
TARGET_LAYER = 9
ALPHAS = [5.0, 8.0, 10.0, 12.0, 15.0, 20.0]
TIANANMEN_PROMPTS = POLITICAL_REFUSED_PROMPTS[:8]  # First 8 are Tiananmen
OUTPUT_DIR = Path("runs/ridge_scale_sweep")


def extract_directions(model, tokenizer, layer: int, device):
    """Extract political and generic CAA directions at a specific layer."""
    print(f"  Extracting political direction at layer {layer}...")
    pol_pos = mean_hidden_state_at_layer(
        model, tokenizer, list(POLITICAL_REFUSED_PROMPTS),
        layer, device=device, label="pol_pos",
    )  # (N, D)
    pol_neg = mean_hidden_state_at_layer(
        model, tokenizer, list(POLITICAL_ANSWERED_PROMPTS),
        layer, device=device, label="pol_neg",
    )  # (N, D)
    d_political = pol_pos.mean(dim=0) - pol_neg.mean(dim=0)  # (D,)

    print(f"  Extracting generic direction at layer {layer}...")
    gen_pos = mean_hidden_state_at_layer(
        model, tokenizer, list(GENERIC_HARMFUL),
        layer, device=device, label="gen_pos",
    )
    gen_neg = mean_hidden_state_at_layer(
        model, tokenizer, list(GENERIC_HARMLESS),
        layer, device=device, label="gen_neg",
    )
    d_generic = gen_pos.mean(dim=0) - gen_neg.mean(dim=0)  # (D,)

    return d_political, d_generic


def ridge_clean(v_pol: torch.Tensor, v_gen: torch.Tensor,
                lam: float = 1.0) -> torch.Tensor:
    """Ridge residualize v_pol against v_gen to remove capability overlap."""
    # A is the "capability" matrix (here just the generic direction)
    A = v_gen.unsqueeze(0).float()  # (1, D)
    v = v_pol.float()  # (D,)

    # v_clean = v - A^T (A A^T + lambda I)^{-1} A v
    AAt = A @ A.T  # (1, 1)
    reg = AAt + lam * torch.eye(AAt.shape[0], device=AAt.device)
    Av = A @ v  # (1,)
    proj = A.T @ torch.linalg.solve(reg, Av)  # (D,)
    v_clean = v - proj

    # Report overlap before/after
    cos_before = torch.nn.functional.cosine_similarity(
        v_pol.float().unsqueeze(0), v_gen.float().unsqueeze(0)
    ).item()
    cos_after = torch.nn.functional.cosine_similarity(
        v_clean.unsqueeze(0), v_gen.float().unsqueeze(0)
    ).item()
    print(f"  Ridge cleaning: cos(pol, gen) {cos_before:.4f} → {cos_after:.4f}")

    return v_clean.to(v_pol.dtype)


def ablate_and_generate(model, tokenizer, prompts, direction, alpha, layer, device):
    """Apply ablation hook and generate responses."""
    layers = _resolve_transformer_layers(model)
    direction_unit = direction / direction.norm()
    direction_unit = direction_unit.to(device)

    # Register hook
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        proj = (h * direction_unit).sum(dim=-1, keepdim=True) * direction_unit
        h_new = h - alpha * proj
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    handle = layers[layer].register_forward_hook(hook_fn)

    gen_cfg = _default_generate_kwargs(tokenizer)
    gen_cfg["max_new_tokens"] = 128

    results = []
    for pi, prompt in enumerate(prompts):
        encoded = _tokenize(tokenizer, prompt, max_length=256, padding=False, chat_template=True)
        encoded = _to_device(encoded, device)
        input_len = encoded["input_ids"].shape[-1]

        with torch.no_grad():
            out = model.generate(**encoded, **gen_cfg)

        response = _decode_response(tokenizer, prompt, out[0], input_length=input_len)
        refused = _is_refusal(response, DEFAULT_REFUSAL_MARKERS)

        # Simple coherence check: response has >20 chars, no excessive repetition
        coherent = len(response) > 20 and not _is_repetitive(response)

        results.append({
            "prompt": prompt,
            "response": response[:500],
            "refused": refused,
            "coherent": coherent,
        })

    handle.remove()
    return results


def _is_repetitive(text: str, threshold: float = 0.5) -> bool:
    """Check if text is repetitively garbled."""
    if len(text) < 40:
        return False
    # Check if any 10-char substring repeats >3 times
    for i in range(len(text) - 30):
        chunk = text[i:i+10]
        if text.count(chunk) > 3:
            return True
    return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Ridge Residualization Scale Sweep")
    print(f"Model: {MODEL_ID}")
    print(f"Layer: {TARGET_LAYER}")
    print(f"Alphas: {ALPHAS}")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    device = _resolve_device(model)

    # Extract directions
    print("\n[2/4] Extracting directions...")
    d_political, d_generic = extract_directions(model, tokenizer, TARGET_LAYER, device)

    # Ridge clean
    print("\n[3/4] Ridge cleaning...")
    d_political_clean = ridge_clean(d_political, d_generic)

    # Baseline (no ablation)
    print("\n[4/4] Running ablation sweep...")
    print("\n--- Baseline (no ablation) ---")
    gen_cfg = _default_generate_kwargs(tokenizer)
    gen_cfg["max_new_tokens"] = 128

    baseline_results = []
    for pi, prompt in enumerate(TIANANMEN_PROMPTS):
        encoded = _tokenize(tokenizer, prompt, max_length=256, padding=False, chat_template=True)
        encoded = _to_device(encoded, device)
        input_len = encoded["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(**encoded, **gen_cfg)
        response = _decode_response(tokenizer, prompt, out[0], input_length=input_len)
        refused = _is_refusal(response, DEFAULT_REFUSAL_MARKERS)
        coherent = len(response) > 20 and not _is_repetitive(response)
        baseline_results.append({
            "prompt": prompt, "response": response[:500],
            "refused": refused, "coherent": coherent,
        })
        print(f"  [{pi+1}/8] {'REFUSED' if refused else 'COMPLIED'} | "
              f"{'coherent' if coherent else 'garbled'} | {prompt[:50]}")

    all_results = {
        "model": MODEL_ID,
        "layer": TARGET_LAYER,
        "alphas": ALPHAS,
        "baseline": {
            "refusal_rate": sum(1 for r in baseline_results if r["refused"]) / len(baseline_results),
            "coherence_rate": sum(1 for r in baseline_results if r["coherent"]) / len(baseline_results),
            "results": baseline_results,
        },
        "raw_political": {},
        "ridge_cleaned_political": {},
        "generic_refusal": {},
    }

    # Sweep alphas
    for alpha in ALPHAS:
        print(f"\n--- Raw Political CAA, alpha={alpha} ---")
        raw_results = ablate_and_generate(
            model, tokenizer, TIANANMEN_PROMPTS,
            d_political, alpha, TARGET_LAYER, device,
        )
        n_refused = sum(1 for r in raw_results if r["refused"])
        n_coherent = sum(1 for r in raw_results if r["coherent"])
        print(f"  Refusal: {n_refused}/8, Coherent: {n_coherent}/8")
        all_results["raw_political"][str(alpha)] = {
            "refusal_rate": n_refused / 8,
            "coherence_rate": n_coherent / 8,
            "results": raw_results,
        }

        print(f"\n--- Ridge-Cleaned Political, alpha={alpha} ---")
        ridge_results = ablate_and_generate(
            model, tokenizer, TIANANMEN_PROMPTS,
            d_political_clean, alpha, TARGET_LAYER, device,
        )
        n_refused = sum(1 for r in ridge_results if r["refused"])
        n_coherent = sum(1 for r in ridge_results if r["coherent"])
        print(f"  Refusal: {n_refused}/8, Coherent: {n_coherent}/8")
        all_results["ridge_cleaned_political"][str(alpha)] = {
            "refusal_rate": n_refused / 8,
            "coherence_rate": n_coherent / 8,
            "results": ridge_results,
        }

        print(f"\n--- Generic Refusal Ablation, alpha={alpha} ---")
        gen_results = ablate_and_generate(
            model, tokenizer, TIANANMEN_PROMPTS,
            d_generic, alpha, TARGET_LAYER, device,
        )
        n_refused = sum(1 for r in gen_results if r["refused"])
        n_coherent = sum(1 for r in gen_results if r["coherent"])
        print(f"  Refusal: {n_refused}/8, Coherent: {n_coherent}/8")
        all_results["generic_refusal"][str(alpha)] = {
            "refusal_rate": n_refused / 8,
            "coherence_rate": n_coherent / 8,
            "results": gen_results,
        }

    # Save results
    results_file = OUTPUT_DIR / "scale_sweep_results.json"
    results_file.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to: {results_file}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SCALE SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<30} {'Alpha':>6} {'Refusal':>10} {'Coherent':>10}")
    print("-" * 70)
    print(f"{'Baseline':<30} {'—':>6} "
          f"{all_results['baseline']['refusal_rate']:>10.1%} "
          f"{all_results['baseline']['coherence_rate']:>10.1%}")

    for alpha in ALPHAS:
        a = str(alpha)
        for cond, label in [("raw_political", "Raw Political"),
                            ("ridge_cleaned_political", "Ridge Political"),
                            ("generic_refusal", "Generic Refusal")]:
            d = all_results[cond][a]
            print(f"{label:<30} {alpha:>6.0f} "
                  f"{d['refusal_rate']:>10.1%} "
                  f"{d['coherence_rate']:>10.1%}")
        print("-" * 70)


if __name__ == "__main__":
    main()
