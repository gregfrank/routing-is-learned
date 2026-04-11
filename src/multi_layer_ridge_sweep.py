#!/usr/bin/env python3
"""
Multi-Layer Ridge Residualization Sweep with KL Divergence.

Extends the single-layer ridge scale sweep to test across multiple layers
of Qwen3-8B, measuring:
  - Cosine overlap between political and generic directions (before/after ridge)
  - Direction norms
  - KL divergence from baseline under ablation (raw, ridge-cleaned, generic)
  - Refusal rate and coherence
  - Full response content for confabulation analysis

Tests layers [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36] with
alpha sweep [2, 5, 8, 12, 20] at each layer.

Usage:
  python multi_layer_ridge_sweep.py
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import from existing modules
from ablation_ridge_regression import (
    load_model_and_tokenizer,
    mean_hidden_state_at_layer,
    _resolve_transformer_layers,
    _cache_baseline_logits,
    eval_kl_and_perplexity,
    get_benign_kl_corpus,
    temporary_ablation,
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


# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = "Qwen/Qwen3-8B"
# Every 3rd layer from 3 to 33, plus final layer 35 (Qwen3-8B has 36 layers, indices 0-35)
LAYERS = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]
ALPHAS = [2.0, 5.0, 8.0, 12.0, 20.0]
TIANANMEN_PROMPTS = list(POLITICAL_REFUSED_PROMPTS[:8])
N_KL_TEXTS = 10  # Number of benign texts for KL divergence measurement
OUTPUT_DIR = Path("runs/multi_layer_ridge_sweep")


# ============================================================================
# Logging
# ============================================================================

class TeeWriter:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stdout = sys.stdout
        self._file = open(log_path, "w", buffering=1)

    def write(self, text):
        self._stdout.write(text)
        self._file.write(text)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    @property
    def encoding(self):
        return getattr(self._stdout, 'encoding', 'utf-8')


# ============================================================================
# Direction extraction and ridge cleaning
# ============================================================================

def extract_directions(model, tokenizer, layer: int, device):
    """Extract political and generic CAA directions at a specific layer."""
    print(f"    Political direction at layer {layer}...")
    pol_pos = mean_hidden_state_at_layer(
        model, tokenizer, list(POLITICAL_REFUSED_PROMPTS),
        layer, device=device, label="pol_pos",
    )
    pol_neg = mean_hidden_state_at_layer(
        model, tokenizer, list(POLITICAL_ANSWERED_PROMPTS),
        layer, device=device, label="pol_neg",
    )
    d_political = pol_pos.mean(dim=0) - pol_neg.mean(dim=0)

    print(f"    Generic direction at layer {layer}...")
    gen_pos = mean_hidden_state_at_layer(
        model, tokenizer, list(GENERIC_HARMFUL),
        layer, device=device, label="gen_pos",
    )
    gen_neg = mean_hidden_state_at_layer(
        model, tokenizer, list(GENERIC_HARMLESS),
        layer, device=device, label="gen_neg",
    )
    d_generic = gen_pos.mean(dim=0) - gen_neg.mean(dim=0)

    return d_political, d_generic


def ridge_clean(v_pol: torch.Tensor, v_gen: torch.Tensor,
                lam: float = 1.0) -> torch.Tensor:
    """Ridge residualize v_pol against v_gen to remove capability overlap."""
    A = v_gen.unsqueeze(0).float()
    v = v_pol.float()

    AAt = A @ A.T
    reg = AAt + lam * torch.eye(AAt.shape[0], device=AAt.device)
    Av = A @ v
    proj = A.T @ torch.linalg.solve(reg, Av)
    v_clean = v - proj

    return v_clean.to(v_pol.dtype)


def compute_direction_stats(d_pol, d_gen, d_pol_clean):
    """Compute cosine similarities and norms."""
    cos_raw = F.cosine_similarity(
        d_pol.float().unsqueeze(0), d_gen.float().unsqueeze(0)
    ).item()
    cos_clean = F.cosine_similarity(
        d_pol_clean.float().unsqueeze(0), d_gen.float().unsqueeze(0)
    ).item()

    return {
        "cos_pol_gen_raw": cos_raw,
        "cos_pol_gen_ridge": cos_clean,
        "norm_political": d_pol.float().norm().item(),
        "norm_generic": d_gen.float().norm().item(),
        "norm_political_ridge": d_pol_clean.float().norm().item(),
    }


# ============================================================================
# KL divergence measurement under ablation
# ============================================================================

def measure_kl_under_ablation(model, tokenizer, baseline_cache, direction,
                               alpha, layer_idx, device):
    """Measure KL divergence from baseline when ablation hook is applied."""
    with temporary_ablation(model, layer_idx, direction, alpha=alpha,
                            mode="hook", operation="subtract"):
        kl_results = eval_kl_and_perplexity(
            model, tokenizer, baseline_cache, device=device,
        )
    return kl_results


# ============================================================================
# Generation under ablation
# ============================================================================

def _is_repetitive(text: str) -> bool:
    """Check if text is repetitively garbled."""
    if len(text) < 40:
        return False
    for i in range(len(text) - 30):
        chunk = text[i:i+10]
        if text.count(chunk) > 3:
            return True
    return False


def generate_under_ablation(model, tokenizer, prompts, direction, alpha,
                             layer_idx, device):
    """Generate responses with ablation hook and measure refusal/coherence."""
    gen_cfg = _default_generate_kwargs(tokenizer)
    gen_cfg["max_new_tokens"] = 128

    results = []
    with temporary_ablation(model, layer_idx, direction, alpha=alpha,
                            mode="hook", operation="subtract"):
        for prompt in prompts:
            encoded = _tokenize(tokenizer, prompt, max_length=256,
                                padding=False, chat_template=True)
            encoded = _to_device(encoded, device)
            input_len = encoded["input_ids"].shape[-1]

            with torch.no_grad():
                out = model.generate(**encoded, **gen_cfg)

            response = _decode_response(tokenizer, prompt, out[0],
                                         input_length=input_len)
            refused = _is_refusal(response, DEFAULT_REFUSAL_MARKERS)
            coherent = len(response) > 20 and not _is_repetitive(response)

            results.append({
                "prompt": prompt,
                "response": response[:500],
                "refused": refused,
                "coherent": coherent,
            })

    n_refused = sum(1 for r in results if r["refused"])
    n_coherent = sum(1 for r in results if r["coherent"])
    return results, n_refused, n_coherent


def generate_baseline(model, tokenizer, prompts, device):
    """Generate baseline responses (no ablation)."""
    gen_cfg = _default_generate_kwargs(tokenizer)
    gen_cfg["max_new_tokens"] = 128

    results = []
    for prompt in prompts:
        encoded = _tokenize(tokenizer, prompt, max_length=256,
                            padding=False, chat_template=True)
        encoded = _to_device(encoded, device)
        input_len = encoded["input_ids"].shape[-1]

        with torch.no_grad():
            out = model.generate(**encoded, **gen_cfg)

        response = _decode_response(tokenizer, prompt, out[0],
                                     input_length=input_len)
        refused = _is_refusal(response, DEFAULT_REFUSAL_MARKERS)
        coherent = len(response) > 20 and not _is_repetitive(response)

        results.append({
            "prompt": prompt,
            "response": response[:500],
            "refused": refused,
            "coherent": coherent,
        })

    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_results(all_results, output_dir):
    """Generate summary plots from sweep results."""
    layers = all_results["layers"]
    alphas = all_results["alphas"]

    # --- Plot 1: Cosine overlap across layers (raw vs ridge) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cos_raw = [all_results["per_layer"][str(l)]["direction_stats"]["cos_pol_gen_raw"]
               for l in layers]
    cos_ridge = [all_results["per_layer"][str(l)]["direction_stats"]["cos_pol_gen_ridge"]
                 for l in layers]

    axes[0].plot(layers, cos_raw, "b-o", label="Raw political–generic", markersize=6)
    axes[0].plot(layers, cos_ridge, "r-s", label="Ridge-cleaned–generic", markersize=6)
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Layer", fontsize=11)
    axes[0].set_ylabel("Cosine Similarity", fontsize=11)
    axes[0].set_title("Political–Generic Direction Overlap by Layer", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Direction norms
    norm_pol = [all_results["per_layer"][str(l)]["direction_stats"]["norm_political"]
                for l in layers]
    norm_gen = [all_results["per_layer"][str(l)]["direction_stats"]["norm_generic"]
                for l in layers]
    norm_ridge = [all_results["per_layer"][str(l)]["direction_stats"]["norm_political_ridge"]
                  for l in layers]

    axes[1].plot(layers, norm_pol, "b-o", label="Political (raw)", markersize=6)
    axes[1].plot(layers, norm_gen, "g-^", label="Generic refusal", markersize=6)
    axes[1].plot(layers, norm_ridge, "r-s", label="Political (ridge)", markersize=6)
    axes[1].set_xlabel("Layer", fontsize=11)
    axes[1].set_ylabel("Direction L2 Norm", fontsize=11)
    axes[1].set_title("Direction Norm by Layer", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")

    plt.tight_layout()
    fig.savefig(output_dir / "cosine_and_norms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved cosine_and_norms.png")

    # --- Plot 2: KL divergence heatmaps (3 conditions × layers × alphas) ---
    conditions = ["raw_political", "ridge_political", "generic_refusal"]
    cond_labels = ["Raw Political", "Ridge Political", "Generic Refusal"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ci, (cond, label) in enumerate(zip(conditions, cond_labels)):
        kl_matrix = np.full((len(alphas), len(layers)), np.nan)
        for ai, alpha in enumerate(alphas):
            for li, layer in enumerate(layers):
                layer_data = all_results["per_layer"].get(str(layer), {})
                alpha_data = layer_data.get("ablation_results", {}).get(cond, {}).get(str(alpha), {})
                kl_val = alpha_data.get("full_token_kl", np.nan)
                kl_matrix[ai, li] = kl_val

        im = axes[ci].imshow(kl_matrix, aspect="auto", cmap="YlOrRd",
                             interpolation="nearest")
        axes[ci].set_xticks(range(len(layers)))
        axes[ci].set_xticklabels([str(l) for l in layers], rotation=45, fontsize=8)
        axes[ci].set_yticks(range(len(alphas)))
        axes[ci].set_yticklabels([str(a) for a in alphas])
        axes[ci].set_xlabel("Layer", fontsize=10)
        axes[ci].set_ylabel("Alpha", fontsize=10)
        axes[ci].set_title(f"{label}\nFull-Token KL Divergence", fontsize=11, fontweight="bold")

        # Annotate cells
        for ai in range(len(alphas)):
            for li in range(len(layers)):
                val = kl_matrix[ai, li]
                if not np.isnan(val):
                    color = "white" if val > np.nanmax(kl_matrix) * 0.6 else "black"
                    axes[ci].text(li, ai, f"{val:.3f}", ha="center", va="center",
                                  fontsize=7, color=color)

        plt.colorbar(im, ax=axes[ci], shrink=0.8)

    plt.tight_layout()
    fig.savefig(output_dir / "kl_divergence_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved kl_divergence_heatmaps.png")

    # --- Plot 3: Refusal rate heatmaps ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ci, (cond, label) in enumerate(zip(conditions, cond_labels)):
        ref_matrix = np.full((len(alphas), len(layers)), np.nan)
        for ai, alpha in enumerate(alphas):
            for li, layer in enumerate(layers):
                layer_data = all_results["per_layer"].get(str(layer), {})
                alpha_data = layer_data.get("ablation_results", {}).get(cond, {}).get(str(alpha), {})
                ref_val = alpha_data.get("refusal_rate", np.nan)
                ref_matrix[ai, li] = ref_val

        im = axes[ci].imshow(ref_matrix, aspect="auto", cmap="RdYlGn_r",
                             vmin=0, vmax=1, interpolation="nearest")
        axes[ci].set_xticks(range(len(layers)))
        axes[ci].set_xticklabels([str(l) for l in layers], rotation=45, fontsize=8)
        axes[ci].set_yticks(range(len(alphas)))
        axes[ci].set_yticklabels([str(a) for a in alphas])
        axes[ci].set_xlabel("Layer", fontsize=10)
        axes[ci].set_ylabel("Alpha", fontsize=10)
        axes[ci].set_title(f"{label}\nRefusal Rate", fontsize=11, fontweight="bold")

        for ai in range(len(alphas)):
            for li in range(len(layers)):
                val = ref_matrix[ai, li]
                if not np.isnan(val):
                    color = "white" if val > 0.5 else "black"
                    axes[ci].text(li, ai, f"{val:.0%}", ha="center", va="center",
                                  fontsize=7, color=color)

        plt.colorbar(im, ax=axes[ci], shrink=0.8)

    plt.tight_layout()
    fig.savefig(output_dir / "refusal_rate_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved refusal_rate_heatmaps.png")

    # --- Plot 4: KL divergence comparison — ridge vs raw at optimal alpha ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # For each layer, find the alpha that gives 0% refusal with lowest KL
    for cond, label, color, marker in [
        ("raw_political", "Raw Political", "blue", "o"),
        ("ridge_political", "Ridge Political", "red", "s"),
        ("generic_refusal", "Generic Refusal", "green", "^"),
    ]:
        kl_at_best = []
        best_alphas = []
        for layer in layers:
            layer_data = all_results["per_layer"].get(str(layer), {})
            cond_data = layer_data.get("ablation_results", {}).get(cond, {})

            # Find lowest-KL alpha that achieves 0% refusal
            best_kl = float("inf")
            best_alpha = None
            for alpha in alphas:
                ad = cond_data.get(str(alpha), {})
                if ad.get("refusal_rate", 1.0) == 0.0:
                    kl = ad.get("full_token_kl", float("inf"))
                    if kl < best_kl:
                        best_kl = kl
                        best_alpha = alpha

            if best_alpha is not None:
                kl_at_best.append(best_kl)
                best_alphas.append(best_alpha)
            else:
                kl_at_best.append(np.nan)
                best_alphas.append(np.nan)

        ax.plot(layers, kl_at_best, f"-{marker}", color=color, label=label,
                markersize=7, linewidth=1.5)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Full-Token KL Divergence (at 0% refusal)", fontsize=11)
    ax.set_title("KL Divergence at Lowest Alpha Achieving 0% Refusal",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(output_dir / "kl_at_zero_refusal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved kl_at_zero_refusal.png")

    # --- Plot 5: Ridge improvement ratio ---
    fig, ax = plt.subplots(figsize=(10, 5))

    ridge_improvement = []
    for layer in layers:
        layer_data = all_results["per_layer"].get(str(layer), {})
        raw_data = layer_data.get("ablation_results", {}).get("raw_political", {})
        ridge_data = layer_data.get("ablation_results", {}).get("ridge_political", {})

        # Compare KL at alpha=8 (or nearest)
        for alpha in [8.0, 5.0, 12.0]:
            raw_kl = raw_data.get(str(alpha), {}).get("full_token_kl", None)
            ridge_kl = ridge_data.get(str(alpha), {}).get("full_token_kl", None)
            if raw_kl is not None and ridge_kl is not None and raw_kl > 0:
                ridge_improvement.append(raw_kl / ridge_kl)
                break
        else:
            ridge_improvement.append(np.nan)

    ax.bar(range(len(layers)), ridge_improvement, color="steelblue", alpha=0.8)
    ax.axhline(y=1.0, color="red", linestyle="--", label="No improvement (ratio=1)")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("KL Ratio (raw / ridge)", fontsize=11)
    ax.set_title("Ridge Cleaning KL Improvement by Layer (>1 = ridge is better)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_dir / "ridge_kl_improvement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ridge_kl_improvement.png")


# ============================================================================
# Main
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_path = OUTPUT_DIR / "sweep.log"
    tee = TeeWriter(log_path)
    sys.stdout = tee

    pipeline_start = time.time()

    print("=" * 70)
    print("Multi-Layer Ridge Residualization Sweep with KL Divergence")
    print(f"Model: {MODEL_ID}")
    print(f"Layers: {LAYERS}")
    print(f"Alphas: {ALPHAS}")
    print(f"Prompts: {len(TIANANMEN_PROMPTS)} Tiananmen")
    print(f"KL texts: {N_KL_TEXTS}")
    print(f"Total ablation conditions: {len(LAYERS)} layers × {len(ALPHAS)} alphas × 3 conditions = {len(LAYERS) * len(ALPHAS) * 3}")
    print("=" * 70)

    # Load model
    print("\n[1/5] Loading model...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    device = _resolve_device(model)
    print(f"  Model loaded in {time.time() - t0:.0f}s on {device}")

    # Cache baseline logits for KL measurement
    print("\n[2/5] Caching baseline logits...")
    t0 = time.time()
    benign_texts = get_benign_kl_corpus()[:N_KL_TEXTS]
    baseline_cache = _cache_baseline_logits(model, tokenizer, benign_texts, device=device)
    print(f"  Cached {len(baseline_cache)} texts in {time.time() - t0:.0f}s")

    # Generate baseline responses
    print("\n[3/5] Generating baseline responses...")
    t0 = time.time()
    baseline_results = generate_baseline(model, tokenizer, TIANANMEN_PROMPTS, device)
    n_ref = sum(1 for r in baseline_results if r["refused"])
    n_coh = sum(1 for r in baseline_results if r["coherent"])
    print(f"  Baseline: {n_ref}/8 refused, {n_coh}/8 coherent ({time.time() - t0:.0f}s)")

    all_results = {
        "model": MODEL_ID,
        "layers": LAYERS,
        "alphas": ALPHAS,
        "n_kl_texts": N_KL_TEXTS,
        "baseline": {
            "refusal_rate": n_ref / len(TIANANMEN_PROMPTS),
            "coherence_rate": n_coh / len(TIANANMEN_PROMPTS),
            "results": baseline_results,
        },
        "per_layer": {},
    }

    # Resume from checkpoint if available
    checkpoint_file = OUTPUT_DIR / "sweep_results_checkpoint.json"
    if checkpoint_file.exists():
        print(f"  Resuming from checkpoint: {checkpoint_file}")
        with open(checkpoint_file) as f:
            saved = json.load(f)
        all_results["per_layer"] = saved.get("per_layer", {})
        print(f"  Layers already done: {sorted(all_results['per_layer'].keys(), key=int)}")

    # Main sweep: iterate over layers
    total_layers = len(LAYERS)
    total_conditions = total_layers * len(ALPHAS) * 3
    condition_count = 0

    print(f"\n[4/5] Running multi-layer sweep ({total_conditions} conditions)...")

    for li, layer in enumerate(LAYERS):
        # Skip already-completed layers (check ablation_results has all 15 conditions)
        existing = all_results["per_layer"].get(str(layer), {})
        existing_abl = existing.get("ablation_results", {})
        n_existing = sum(len(v) for v in existing_abl.values() if isinstance(v, dict))
        if n_existing >= len(ALPHAS) * 3:
            condition_count += n_existing
            print(f"\n  Layer {layer} ({li+1}/{total_layers}): SKIP (already complete, {n_existing} conditions)")
            continue

        layer_start = time.time()
        print(f"\n{'='*60}")
        print(f"  Layer {layer} ({li+1}/{total_layers})")
        print(f"{'='*60}")

        # Extract directions at this layer
        print(f"  Extracting directions...")
        d_pol, d_gen = extract_directions(model, tokenizer, layer, device)

        # Ridge clean
        d_pol_clean = ridge_clean(d_pol, d_gen)

        # Direction statistics
        stats = compute_direction_stats(d_pol, d_gen, d_pol_clean)
        print(f"  cos(pol, gen): {stats['cos_pol_gen_raw']:.4f} → {stats['cos_pol_gen_ridge']:.4f} (ridge)")
        print(f"  Norms: pol={stats['norm_political']:.2f}, gen={stats['norm_generic']:.2f}, ridge={stats['norm_political_ridge']:.2f}")

        layer_results = {
            "direction_stats": stats,
            "ablation_results": {
                "raw_political": {},
                "ridge_political": {},
                "generic_refusal": {},
            },
        }

        # Sweep alphas for each condition
        directions = {
            "raw_political": ("Raw Political", d_pol),
            "ridge_political": ("Ridge Political", d_pol_clean),
            "generic_refusal": ("Generic Refusal", d_gen),
        }

        for cond_key, (cond_label, direction) in directions.items():
            for alpha in ALPHAS:
                condition_count += 1
                print(f"\n  [{condition_count}/{total_conditions}] {cond_label}, L{layer}, α={alpha}")

                # Measure KL divergence
                kl = measure_kl_under_ablation(
                    model, tokenizer, baseline_cache, direction,
                    alpha, layer, device,
                )
                print(f"    KL: first={kl['first_token_kl']:.4f}, full={kl['full_token_kl']:.4f}, "
                      f"PPL: {kl['ppl_candidate']:.2f}")

                # Generate responses
                gen_results, n_refused, n_coherent = generate_under_ablation(
                    model, tokenizer, TIANANMEN_PROMPTS, direction,
                    alpha, layer, device,
                )
                print(f"    Refusal: {n_refused}/8, Coherent: {n_coherent}/8")

                layer_results["ablation_results"][cond_key][str(alpha)] = {
                    "refusal_rate": n_refused / len(TIANANMEN_PROMPTS),
                    "coherence_rate": n_coherent / len(TIANANMEN_PROMPTS),
                    "first_token_kl": kl["first_token_kl"],
                    "full_token_kl": kl["full_token_kl"],
                    "ppl_candidate": kl["ppl_candidate"],
                    "ppl_base": kl["ppl_base"],
                    "results": gen_results,
                }

        all_results["per_layer"][str(layer)] = layer_results

        layer_elapsed = time.time() - layer_start
        print(f"\n  Layer {layer} complete in {layer_elapsed:.0f}s")

        # Save intermediate checkpoint after each layer
        checkpoint_file = OUTPUT_DIR / "sweep_results_checkpoint.json"
        checkpoint_file.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        print(f"  Checkpoint saved to {checkpoint_file}")

    # Save final results
    print(f"\n[5/5] Saving results and generating plots...")
    results_file = OUTPUT_DIR / "sweep_results.json"
    results_file.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"  Results saved to {results_file}")

    # Generate plots
    plot_results(all_results, OUTPUT_DIR)

    # Print summary table
    total_elapsed = time.time() - pipeline_start
    print(f"\n{'='*70}")
    print(f"MULTI-LAYER RIDGE SWEEP SUMMARY")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*70}")

    # Direction stats table
    print(f"\n{'Layer':>5} {'cos(raw)':>10} {'cos(ridge)':>12} {'‖pol‖':>8} {'‖gen‖':>8} {'‖ridge‖':>8}")
    print("-" * 60)
    for layer in LAYERS:
        s = all_results["per_layer"][str(layer)]["direction_stats"]
        print(f"{layer:>5} {s['cos_pol_gen_raw']:>10.4f} {s['cos_pol_gen_ridge']:>12.6f} "
              f"{s['norm_political']:>8.1f} {s['norm_generic']:>8.1f} {s['norm_political_ridge']:>8.1f}")

    # KL and refusal summary for key alpha values
    print(f"\n{'Condition':<25} {'Layer':>5} {'Alpha':>6} {'Refusal':>8} {'KL(full)':>10} {'PPL':>8}")
    print("-" * 70)

    for layer in LAYERS:
        ld = all_results["per_layer"][str(layer)]["ablation_results"]
        for cond, label in [("raw_political", "Raw Political"),
                            ("ridge_political", "Ridge Political"),
                            ("generic_refusal", "Generic Refusal")]:
            for alpha in ALPHAS:
                ad = ld[cond].get(str(alpha), {})
                ref = ad.get("refusal_rate", -1)
                kl = ad.get("full_token_kl", -1)
                ppl = ad.get("ppl_candidate", -1)
                print(f"{label:<25} {layer:>5} {alpha:>6.0f} {ref:>8.0%} {kl:>10.4f} {ppl:>8.1f}")
        print("-" * 70)

    # Restore stdout
    sys.stdout = tee._stdout
    tee.close()
    print(f"\nDone. Results: {results_file}")
    print(f"Log: {log_path}")
    print(f"Plots: {OUTPUT_DIR}/*.png")


if __name__ == "__main__":
    main()
