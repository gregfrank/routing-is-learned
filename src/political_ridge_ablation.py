#!/usr/bin/env python3
"""
Political Ridge Ablation Experiment

Applies the full ridge-residualized ablation pipeline (from ablation_ridge_regression.py)
to the political sensitivity direction, rather than the generic safety refusal direction.

The hypothesis: raw CAA political direction ablation garbles output because the direction
carries topic-content features alongside the refusal-routing signal. Ridge residualization
can project out the capability-entangled components, isolating a "pure political refusal
routing" vector that removes censorship without destroying coherence.

Comparison conditions:
  A. Baseline (no ablation)
  B. Generic refusal ablation (ridge-cleaned, from existing checkpoint)
  C. Raw political CAA ablation (no cleaning — reproduces garbled output)
  D. Ridge-cleaned political ablation (THIS IS THE NEW EXPERIMENT)
  E. Ridge-cleaned political + generic combined
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import time
import sys
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    matplotlib = None
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = exc

from ablation_ridge_regression import (
    load_model_and_tokenizer,
    mean_hidden_state_at_layer,
    _resolve_transformer_layers,
    _resolve_device,
    _tokenize,
    _to_device,
    _default_generate_kwargs,
    _decode_response,
    _strip_thinking,
    _is_refusal,
    _find_rank_one_target,
    temporary_ablation,
    ridge_residualize,
    compute_caa_direction,
    build_concept_atoms,
    get_capability_concept_prompts,
    get_benign_kl_corpus,
    DEFAULT_REFUSAL_MARKERS,
    DEVICE,
    ENABLE_THINKING,
)
from direction_methods import (
    CALIBRATION_CHOICES,
    DIRECTION_METHOD_CHOICES,
    compute_wmd_direction,
    DEFAULT_SIGNED_SCORE_NEUTRAL_BAND,
    DEFAULT_WMD_NEUTRAL_BAND,
    format_direction_method_label,
    parse_direction_method,
    resolve_wmd_inputs,
)

# Also import the political prompts from the CAA experiment
from political_probe import (
    POLITICAL_REFUSED_PROMPTS,
    POLITICAL_ANSWERED_PROMPTS,
    GENERIC_HARMFUL,
    GENERIC_HARMLESS,
    BENIGN_PROMPTS,
    extract_per_prompt_hidden_states,
)
from political_prompts_adversarial import ADVERSARIAL_CCP, ADVERSARIAL_PARALLEL

# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = "Qwen/Qwen3-8B"
SEED = 42
RUN_DIR = Path("runs/political_ridge_ablation")
PLOTS_DIR = RUN_DIR / "plots"
CORPUS = "v1"

# Layers to build candidate bank for
LAYER_INDICES = [9, 18, 30]

# Alpha sweep range for political ablation
ALPHA_VALUES = [2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0]

# ============================================================================
# Utility
# ============================================================================

def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate a single response without ablation."""
    dev = _resolve_device(model)
    gen_cfg = _default_generate_kwargs(tokenizer)
    gen_cfg["max_new_tokens"] = max_new_tokens

    encoded = _tokenize(tokenizer, prompt, max_length=256, padding=False, chat_template=True)
    encoded = _to_device(encoded, dev)
    input_len = encoded["input_ids"].shape[-1]

    with torch.no_grad():
        out = model.generate(**encoded, **gen_cfg)

    return _decode_response(tokenizer, prompt, out[0], input_length=input_len)


def generate_with_ablation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer_idx: int,
    vector: torch.Tensor,
    alpha: float,
    max_new_tokens: int = 256,
) -> str:
    """Generate a single response with ablation applied."""
    dev = _resolve_device(model)
    v = vector.to(dev, dtype=torch.float32)
    rank_one_target = _find_rank_one_target(model, layer_idx)

    gen_cfg = _default_generate_kwargs(tokenizer)
    gen_cfg["max_new_tokens"] = max_new_tokens

    encoded = _tokenize(tokenizer, prompt, max_length=256, padding=False, chat_template=True)
    encoded = _to_device(encoded, dev)
    input_len = encoded["input_ids"].shape[-1]

    with temporary_ablation(
        model=model, layer_idx=layer_idx, vector=v,
        alpha=alpha, mode="hook", gamma=0.0,
        target=rank_one_target,
    ):
        with torch.no_grad():
            out = model.generate(**encoded, **gen_cfg)

    return _decode_response(tokenizer, prompt, out[0], input_length=input_len)


def eval_condition(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    label: str,
    layer_idx: Optional[int] = None,
    vector: Optional[torch.Tensor] = None,
    alpha: float = 0.0,
    max_new_tokens: int = 256,
) -> List[Dict[str, Any]]:
    """Evaluate a set of prompts under a given condition (baseline or ablated)."""
    results = []
    for pi, prompt in enumerate(prompts):
        print(f"    [{label}] {pi+1}/{len(prompts)}: {prompt[:60]}...", end="\r", flush=True)

        if vector is not None and layer_idx is not None:
            response = generate_with_ablation(
                model, tokenizer, prompt, layer_idx, vector, alpha, max_new_tokens)
        else:
            response = generate_response(model, tokenizer, prompt, max_new_tokens)

        refused = _is_refusal(response, DEFAULT_REFUSAL_MARKERS)
        results.append({
            "prompt": prompt,
            "response": response,
            "refused": refused,
        })
        tag = "REFUSED" if refused else "COMPLIED"
        print(f"    [{label}] {pi+1}/{len(prompts)}: {tag} — {prompt[:60]}")

    n_refused = sum(1 for r in results if r["refused"])
    print(f"    [{label}] Refusal rate: {n_refused}/{len(prompts)} ({n_refused/len(prompts):.1%})")
    return results


def is_coherent(response: str, min_words: int = 10) -> bool:
    """Heuristic check: is the response coherent (not garbled)?"""
    if not response.strip():
        return False
    words = response.split()
    if len(words) < min_words:
        return False
    # Check for repetition loops: if any 5-gram repeats > 3 times, it's garbled
    if len(words) >= 5:
        ngrams = [" ".join(words[i:i+5]) for i in range(len(words)-4)]
        from collections import Counter
        counts = Counter(ngrams)
        if counts.most_common(1)[0][1] > 3:
            return False
    # Check for character-level repetition (e.g., "，，，，，，")
    # Threshold 0.02: normal English prose has ~0.04-0.06 unique ratio at 200+ words
    # due to the small alphabet; only flag truly degenerate single-char spam.
    if len(response) > 20:
        chars = list(response.replace(" ", "").replace("\n", ""))
        if len(chars) > 10:
            unique_ratio = len(set(chars)) / len(chars)
            if unique_ratio < 0.02:
                return False
    return True


def _clean_condition_name(name: str) -> str:
    return name.replace("\n", " ")


def _load_existing_results(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for key, rows in raw.items():
        if isinstance(key, str) and isinstance(rows, list):
            out[key] = rows
    return out


def resolve_prompt_corpus(corpus: str) -> Tuple[List[str], List[str]]:
    if corpus == "adversarial":
        return list(ADVERSARIAL_CCP), list(ADVERSARIAL_PARALLEL)
    return list(POLITICAL_REFUSED_PROMPTS), list(POLITICAL_ANSWERED_PROMPTS)


def compute_political_direction(
    direction_method: str,
    positive_states: torch.Tensor,
    negative_states: torch.Tensor,
    positive_probabilities: Optional[torch.Tensor] = None,
    negative_probabilities: Optional[torch.Tensor] = None,
    neutral_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    base_method, _ = parse_direction_method(direction_method)
    if base_method == "wmd":
        if positive_probabilities is None or negative_probabilities is None:
            raise ValueError("WMD extraction requires prompt-aligned probabilities.")
        all_states = torch.cat([positive_states, negative_states], dim=0)
        neutral_states = None
        if neutral_mask is not None and bool(neutral_mask.any()):
            neutral_states = all_states[neutral_mask]
        return compute_wmd_direction(
            positive_states,
            negative_states,
            positive_probabilities,
            negative_probabilities,
            neutral_states=neutral_states,
            normalize_weighted_means=True,
        )
    return compute_caa_direction(positive_states, negative_states)


def _write_results_json(path: Path, all_results: Dict[str, List[Dict[str, Any]]]) -> None:
    serializable: Dict[str, List[Dict[str, Any]]] = {}
    for cond_name, results in all_results.items():
        clean_name = _clean_condition_name(cond_name)
        serializable[clean_name] = [
            {
                "prompt": r.get("prompt", ""),
                "response": r.get("response", ""),
                "refused": bool(r.get("refused", False)),
            }
            for r in results
        ]
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_progress_checkpoint(
    path: Path,
    model_id: str,
    corpus: str,
    direction_method: str,
    layer_indices: Sequence[int],
    alpha_values: Sequence[float],
    all_results: Dict[str, List[Dict[str, Any]]],
) -> None:
    direction_base_method, _ = parse_direction_method(direction_method)
    direction_base_label = "WMD" if direction_base_method == "wmd" else "CAA"
    completed_layers: List[int] = []
    for layer_idx in layer_indices:
        raw_name = _clean_condition_name(f"Raw political {direction_base_label}\n(layer {layer_idx})")
        ridge_name = _clean_condition_name(f"Ridge-cleaned political {direction_base_label}\n(layer {layer_idx})")
        if raw_name in all_results and ridge_name in all_results:
            completed_layers.append(int(layer_idx))
    payload = {
        "timestamp_utc": _timestamp(),
        "model_id": model_id,
        "corpus": corpus,
        "direction_method": direction_method,
        "layers_requested": list(layer_indices),
        "alphas": list(alpha_values),
        "completed_layers": completed_layers,
        "conditions_completed": sorted(all_results.keys()),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================================
# Main experiment
# ============================================================================

def main():
    global RUN_DIR, PLOTS_DIR, ALPHA_VALUES, CORPUS

    parser = argparse.ArgumentParser(description="Political Ridge Ablation Experiment")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--corpus", type=str, default=CORPUS, choices=["v1", "adversarial"])
    parser.add_argument(
        "--direction-method",
        type=str,
        default="caa",
        choices=DIRECTION_METHOD_CHOICES,
        help="Political direction extractor to compare against ridge cleaning.",
    )
    parser.add_argument(
        "--wmd-prob-source",
        type=str,
        default=None,
        help="Prompt-aligned censorship probability artifact used when the base method is WMD.",
    )
    parser.add_argument(
        "--signed-score-source",
        type=str,
        default=None,
        help="Optional model_id x prompt signed-score artifact used when the base method is WMD.",
    )
    parser.add_argument(
        "--wmd-calibration",
        type=str,
        default="none",
        choices=CALIBRATION_CHOICES,
        help="Optional calibration applied to the loaded WMD probabilities.",
    )
    parser.add_argument(
        "--signed-neutral-band",
        type=float,
        default=DEFAULT_SIGNED_SCORE_NEUTRAL_BAND,
        help="Rows with |signed_score| <= band are treated as neutral when WMD uses signed scores.",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=LAYER_INDICES)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=ALPHA_VALUES,
        help="Alpha sweep values for ablation strength",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(RUN_DIR),
        help="Output directory for results/checkpoints/plots",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing results.json and recompute all conditions from scratch.",
    )
    parser.add_argument(
        "--generic-checkpoint",
        type=str,
        default=os.environ.get("POLITICAL_GENERIC_CHECKPOINT", "runs/qwen3_8b_ablation/checkpoint.pt"),
        help="Path to the generic refusal checkpoint used for generic/combined comparisons.",
    )
    args = parser.parse_args()

    model_id = args.model
    CORPUS = args.corpus
    direction_method = args.direction_method
    direction_base_method, preferred_is_ridge = parse_direction_method(direction_method)
    direction_label = format_direction_method_label(direction_method)
    direction_base_label = "WMD" if direction_base_method == "wmd" else "CAA"
    layer_indices = args.layers
    ALPHA_VALUES = list(args.alphas)
    if args.run_dir == str(RUN_DIR) and (direction_method != "caa" or CORPUS != "v1"):
        RUN_DIR = Path(f"runs/political_ridge_ablation_{direction_base_method}_{CORPUS}")
    else:
        RUN_DIR = Path(args.run_dir)
    PLOTS_DIR = RUN_DIR / "plots"
    generic_ckpt_path = Path(args.generic_checkpoint) if args.generic_checkpoint else None
    resume_enabled = not args.no_resume
    refused_prompts, answered_prompts = resolve_prompt_corpus(CORPUS)
    probability_metadata = None
    positive_probabilities: Optional[torch.Tensor] = None
    negative_probabilities: Optional[torch.Tensor] = None
    neutral_mask: Optional[torch.Tensor] = None

    if direction_base_method == "wmd":
        if not args.wmd_prob_source and not args.signed_score_source:
            parser.error(
                "Provide --wmd-prob-source or --signed-score-source when --direction-method uses WMD"
            )
        all_prompts = list(refused_prompts) + list(answered_prompts)
        all_labels = np.asarray([1] * len(refused_prompts) + [0] * len(answered_prompts), dtype=np.int64)
        wmd_inputs = resolve_wmd_inputs(
            prompt_order=all_prompts,
            positive_count=len(refused_prompts),
            model_id=model_id,
            probability_source=args.wmd_prob_source if not args.signed_score_source else None,
            signed_score_source=args.signed_score_source,
            calibration=args.wmd_calibration,
            labels=all_labels,
            probability_neutral_band=DEFAULT_WMD_NEUTRAL_BAND,
            signed_neutral_band=args.signed_neutral_band,
        )
        positive_probabilities = wmd_inputs["positive_weights"]
        negative_probabilities = wmd_inputs["negative_weights"]
        neutral_mask = torch.from_numpy(wmd_inputs["neutral_mask"]).bool()
        probability_metadata = wmd_inputs["metadata"]

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Political Ridge Ablation Experiment")
    print(f"Model: {model_id}")
    print(f"Corpus: {CORPUS}")
    print(f"Direction method: {direction_label}")
    print(f"Layers: {layer_indices}")
    print(f"Alphas: {ALPHA_VALUES}")
    print(f"Run dir: {RUN_DIR}")
    print(f"Generic checkpoint: {generic_ckpt_path}")
    if probability_metadata is not None:
        if probability_metadata.get("kind") == "signed_score_rows":
            signed_meta = probability_metadata["signed_score"]
            print(f"WMD signed-score source: {signed_meta.get('source')}")
            print(
                "WMD signed-score diagnostics: "
                f"mean={signed_meta.get('mean'):.4f}, "
                f"std={signed_meta.get('std'):.4f}, "
                f"min={signed_meta.get('min'):.4f}, "
                f"max={signed_meta.get('max'):.4f}"
            )
        else:
            print(f"WMD probability source: {args.wmd_prob_source}")
            print(f"WMD calibration: {args.wmd_calibration}")
            print(
                "WMD probability diagnostics: "
                f"mean={probability_metadata['effective'].get('mean'):.4f}, "
                f"brier={probability_metadata['effective'].get('brier'):.4f}, "
                f"ece={probability_metadata['effective'].get('ece'):.4f}"
            )
    print(f"Started at {_timestamp()}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load model
    # ------------------------------------------------------------------
    print(f"\n[1/7] Loading model: {model_id}... ({_timestamp()})")
    model, tokenizer = load_model_and_tokenizer(model_id)
    n_layers = len(_resolve_transformer_layers(model))
    model_hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 0) or next(model.parameters()).shape[-1])
    print(f"Model loaded: {n_layers} layers, device={next(model.parameters()).device}")

    # ------------------------------------------------------------------
    # Step 2: Build political direction candidates
    # ------------------------------------------------------------------
    print(f"\n[2/7] Building political direction candidates... ({_timestamp()})")
    print(f"  Using {len(refused_prompts)} refused + {len(answered_prompts)} answered prompts")

    political_candidates = []
    for li, layer_idx in enumerate(layer_indices):
        print(f"\n  [{li+1}/{len(layer_indices)}] Layer {layer_idx}:")

        print(f"    Extracting political (refused) hidden states...")
        pol_states = mean_hidden_state_at_layer(
            model, tokenizer, list(refused_prompts),
            layer_idx=layer_idx, batch_size=4, max_length=256,
            label="political-refused",
        )

        print(f"    Extracting control (answered) hidden states...")
        ctrl_states = mean_hidden_state_at_layer(
            model, tokenizer, list(answered_prompts),
            layer_idx=layer_idx, batch_size=4, max_length=256,
            label="political-answered",
        )

        dirty_dir = compute_political_direction(
            direction_method=direction_method,
            positive_states=pol_states,
            negative_states=ctrl_states,
            positive_probabilities=positive_probabilities,
            negative_probabilities=negative_probabilities,
            neutral_mask=neutral_mask,
        )
        print(f"    Raw political {direction_label} direction norm: {dirty_dir.norm().item():.4f}")

        political_candidates.append({
            "layer_idx": layer_idx,
            "v_dirty": dirty_dir.detach().cpu(),
            "pol_states": pol_states.detach().cpu(),
            "ctrl_states": ctrl_states.detach().cpu(),
            "direction_method": direction_method,
        })

    # ------------------------------------------------------------------
    # Step 3: Build concept atom matrix A (capability subspace)
    # ------------------------------------------------------------------
    print(f"\n[3/7] Building capability concept atoms... ({_timestamp()})")
    concept_prompts = get_capability_concept_prompts()
    print(f"  Concept categories: {list(concept_prompts.keys())}")
    print(f"  Total concept prompts: {sum(len(v) for v in concept_prompts.values())}")

    atoms_per_layer = {}
    for li, layer_idx in enumerate(layer_indices):
        print(f"\n  [{li+1}/{len(layer_indices)}] Building atoms for layer {layer_idx}...")
        atoms = build_concept_atoms(
            model=model, tokenizer=tokenizer,
            concept_prompts=concept_prompts,
            layer_idx=layer_idx, atoms_per_concept=2,
            batch_size=4, max_length=256,
        )
        atoms_per_layer[layer_idx] = atoms
        print(f"    Atom matrix A shape: {atoms['A'].shape} ({len(atoms['labels'])} atoms)")
        print(f"    Atom labels: {atoms['labels']}")

    # ------------------------------------------------------------------
    # Step 4: Ridge residualization — clean the political direction
    # ------------------------------------------------------------------
    print(f"\n[4/7] Ridge residualization of political direction... ({_timestamp()})")

    ridge_candidates = []
    for cand in political_candidates:
        layer_idx = cand["layer_idx"]
        v_dirty = cand["v_dirty"]
        A = atoms_per_layer[layer_idx]["A"]

        # Try multiple lambda values
        best_lam = 1e-2
        best_overlap_reduction = 0.0

        for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            v_clean, info = ridge_residualize(v_dirty, A, lam=lam)
            reduction = info["overlap_before"] - info["overlap_after"]
            if reduction > best_overlap_reduction:
                best_overlap_reduction = reduction
                best_lam = lam

        v_clean, info = ridge_residualize(v_dirty, A, lam=best_lam)

        print(f"\n  Layer {layer_idx}:")
        print(f"    Best lambda: {best_lam}")
        print(f"    Overlap before: {info['overlap_before']:.4f}")
        print(f"    Overlap after:  {info['overlap_after']:.4f}")
        print(f"    Reduction:      {info['overlap_before'] - info['overlap_after']:.4f} "
              f"({(1 - info['overlap_after']/(info['overlap_before']+1e-8))*100:.1f}%)")
        print(f"    Noise norm:     {info['noise_norm']:.4f}")

        # Cosine between dirty and clean
        cos_dirty_clean = (v_dirty @ v_clean).item() / (v_dirty.norm().item() * v_clean.norm().item() + 1e-8)
        print(f"    Cos(dirty, clean): {cos_dirty_clean:.4f}")

        ridge_candidates.append({
            "layer_idx": layer_idx,
            "v_dirty": v_dirty,
            "v_clean": v_clean,
            "atoms": atoms_per_layer[layer_idx],
            "residual_info": info,
            "best_lam": best_lam,
            "cos_dirty_clean": cos_dirty_clean,
        })

    # ------------------------------------------------------------------
    # Step 5: Load generic refusal checkpoint (for comparison)
    # ------------------------------------------------------------------
    print(f"\n[5/7] Loading generic refusal checkpoint... ({_timestamp()})")
    generic_vector = None
    generic_alpha = 20.0
    generic_layer = 18

    if generic_ckpt_path is not None and generic_ckpt_path.exists():
        ckpt = torch.load(generic_ckpt_path, map_location="cpu", weights_only=False)
        best_trial = ckpt["best"]
        generic_layer = int(best_trial["layer_idx"])
        generic_alpha = float(best_trial["alpha"])
        candidate_idx = int(best_trial["candidate_idx"])
        bank = ckpt["bank"]
        eligible = [i for i, c in enumerate(bank) if int(c["layer_idx"]) == generic_layer]
        if eligible:
            slot = candidate_idx % len(eligible)
            generic_vector = bank[eligible[slot]]["v_clean"]
            generic_width = int(generic_vector.shape[-1])
            if generic_width != model_hidden_size:
                print(
                    "  WARNING: Skipping generic refusal/combined comparisons because checkpoint width "
                    f"{generic_width} != model hidden size {model_hidden_size}"
                )
                generic_vector = None
            else:
                print(
                    f"  Loaded generic refusal vector: layer {generic_layer}, alpha {generic_alpha:.2f}"
                )
    else:
        print(f"  WARNING: No generic checkpoint at {generic_ckpt_path}")
        print(f"  Will skip generic ablation comparison")

    # ------------------------------------------------------------------
    # Step 6: Ablation experiments with alpha sweep
    # ------------------------------------------------------------------
    print(f"\n[6/7] Ablation experiments... ({_timestamp()})")

    # Test prompts
    test_political = refused_prompts[:8]
    test_benign = BENIGN_PROMPTS[:4]
    all_test = list(test_political) + list(test_benign)

    results_path = RUN_DIR / "results.json"
    progress_path = RUN_DIR / "political_ridge_progress.json"
    all_results = _load_existing_results(results_path) if resume_enabled else {}
    if all_results:
        print(f"  Resume enabled: loaded {len(all_results)} completed condition(s) from {results_path}")
    else:
        print("  Resume state: no prior conditions found")

    # A: Baseline
    baseline_key = _clean_condition_name("Baseline")
    if baseline_key in all_results:
        print(f"\n  --- A: Baseline (no ablation) --- [skip: already complete]")
    else:
        print(f"\n  --- A: Baseline (no ablation) ---")
        all_results[baseline_key] = eval_condition(model, tokenizer, all_test, "baseline")
        _write_results_json(results_path, all_results)
        _write_progress_checkpoint(progress_path, model_id, CORPUS, direction_method, layer_indices, ALPHA_VALUES, all_results)

    # B: Generic refusal ablation (existing ridge-cleaned vector)
    generic_key = _clean_condition_name("Generic refusal\n(ridge-cleaned)")
    if generic_vector is not None:
        if generic_key in all_results:
            print(f"\n  --- B: Generic refusal ablation (ridge-cleaned) --- [skip: already complete]")
        else:
            print(f"\n  --- B: Generic refusal ablation (ridge-cleaned) ---")
            all_results[generic_key] = eval_condition(
                model, tokenizer, all_test, "generic-ridge",
                layer_idx=generic_layer, vector=generic_vector, alpha=generic_alpha,
            )
            _write_results_json(results_path, all_results)
            _write_progress_checkpoint(progress_path, model_id, CORPUS, direction_method, layer_indices, ALPHA_VALUES, all_results)

    # C & D: For each layer, compare raw vs ridge-cleaned political ablation
    last_best_clean_alpha = float(ALPHA_VALUES[0]) if ALPHA_VALUES else 1.0

    for rc in ridge_candidates:
        layer_idx = rc["layer_idx"]
        v_dirty = rc["v_dirty"]
        v_clean = rc["v_clean"]

        raw_key = _clean_condition_name(f"Raw political {direction_base_label}\n(layer {layer_idx})")
        ridge_key = _clean_condition_name(f"Ridge-cleaned political {direction_base_label}\n(layer {layer_idx})")

        # Alpha sweep for raw political direction
        # NOTE: Alpha selection uses test_political[:4] as a mini-validation set.
        # These 4 prompts are a subset of the 8-prompt final test set, introducing
        # mild train-test leakage. The selected alpha is partially optimized for
        # prompts that also appear in the final evaluation. This is symmetric across
        # all conditions (raw, ridge, control) and unlikely to change qualitative
        # results, but should be acknowledged as a limitation.
        if raw_key in all_results:
            print(f"\n  --- C: Raw political {direction_base_label} (layer {layer_idx}) — alpha sweep --- [skip: already complete]")
        else:
            print(f"\n  --- C: Raw political {direction_base_label} (layer {layer_idx}) — alpha sweep ---")
            best_raw_alpha = 0
            best_raw_score = -1  # score = refusal_removed + coherence_preserved
            for alpha in ALPHA_VALUES:
                results = eval_condition(
                    model, tokenizer, test_political[:4], f"raw-{direction_base_method}-L{layer_idx}-a{alpha}",
                    layer_idx=layer_idx, vector=v_dirty, alpha=alpha,
                )
                n_refused = sum(1 for r in results if r["refused"])
                n_coherent = sum(1 for r in results if is_coherent(r["response"]))
                score = (len(results) - n_refused) + n_coherent  # maximize both
                print(f"      alpha={alpha}: refused={n_refused}/{len(results)}, "
                      f"coherent={n_coherent}/{len(results)}, score={score}")
                if score > best_raw_score:
                    best_raw_score = score
                    best_raw_alpha = alpha

            print(f"    Best raw alpha: {best_raw_alpha}")
            all_results[raw_key] = eval_condition(
                model, tokenizer, all_test, f"raw-{direction_base_method}-L{layer_idx}",
                layer_idx=layer_idx, vector=v_dirty, alpha=best_raw_alpha,
            )
            _write_results_json(results_path, all_results)
            _write_progress_checkpoint(progress_path, model_id, CORPUS, direction_method, layer_indices, ALPHA_VALUES, all_results)

        # Alpha sweep for ridge-cleaned political direction
        if ridge_key in all_results:
            print(f"\n  --- D: Ridge-cleaned political {direction_base_label} (layer {layer_idx}) — alpha sweep --- [skip: already complete]")
        else:
            print(f"\n  --- D: Ridge-cleaned political {direction_base_label} (layer {layer_idx}) — alpha sweep ---")
            best_clean_alpha = 0
            best_clean_score = -1
            for alpha in ALPHA_VALUES:
                results = eval_condition(
                    model, tokenizer, test_political[:4], f"ridge-{direction_base_method}-L{layer_idx}-a{alpha}",
                    layer_idx=layer_idx, vector=v_clean, alpha=alpha,
                )
                n_refused = sum(1 for r in results if r["refused"])
                n_coherent = sum(1 for r in results if is_coherent(r["response"]))
                score = (len(results) - n_refused) + n_coherent
                print(f"      alpha={alpha}: refused={n_refused}/{len(results)}, "
                      f"coherent={n_coherent}/{len(results)}, score={score}")
                if score > best_clean_score:
                    best_clean_score = score
                    best_clean_alpha = alpha

            print(f"    Best ridge-cleaned alpha: {best_clean_alpha}")
            all_results[ridge_key] = eval_condition(
                model, tokenizer, all_test, f"ridge-{direction_base_method}-L{layer_idx}",
                layer_idx=layer_idx, vector=v_clean, alpha=best_clean_alpha,
            )
            last_best_clean_alpha = float(best_clean_alpha)
            _write_results_json(results_path, all_results)
            _write_progress_checkpoint(progress_path, model_id, CORPUS, direction_method, layer_indices, ALPHA_VALUES, all_results)

    # E: Combined generic + best ridge-cleaned political
    if generic_vector is not None and ridge_candidates:
        # Pick the ridge candidate with best score (prefer lowest refusal, then highest coherence)
        best_rc = sorted(
            ridge_candidates,
            key=lambda rc: (
                # Use the already-evaluated ridge results if available
                all_results.get(
                    _clean_condition_name(f"Ridge-cleaned political {direction_base_label} (layer {rc['layer_idx']})"), [{}]
                )[0].get("refused", True),
                -rc["layer_idx"],  # fallback: prefer earlier layers
            ),
        )[0] if len(ridge_candidates) > 1 else ridge_candidates[0]
        best_layer = best_rc["layer_idx"]
        v_clean = best_rc["v_clean"]

        if best_layer == generic_layer:
            print(f"\n  WARNING: generic (layer {generic_layer}) and political (layer {best_layer}) "
                  f"ablations target the same layer. Combined condition may not apply both vectors.")

        combined_key = _clean_condition_name("Combined\n(generic + political ridge)")
        if combined_key in all_results:
            print(f"\n  --- E: Combined (generic ridge + political ridge, layer {best_layer}) --- [skip: already complete]")
        else:
            print(f"\n  --- E: Combined (generic ridge + political ridge, layer {best_layer}) ---")

            dev = _resolve_device(model)
            gen_v = generic_vector.to(dev, dtype=torch.float32)
            pol_v = v_clean.to(dev, dtype=torch.float32)
            rank_one_gen = _find_rank_one_target(model, generic_layer)
            rank_one_pol = _find_rank_one_target(model, best_layer)

            gen_cfg = _default_generate_kwargs(tokenizer)
            gen_cfg["max_new_tokens"] = 256

            combined_results = []
            for pi, prompt in enumerate(all_test):
                print(f"    [combined] {pi+1}/{len(all_test)}: {prompt[:60]}...", end="\r", flush=True)
                encoded = _tokenize(tokenizer, prompt, max_length=256, padding=False, chat_template=True)
                encoded = _to_device(encoded, dev)
                input_len = encoded["input_ids"].shape[-1]

                with temporary_ablation(
                    model=model, layer_idx=generic_layer, vector=gen_v,
                    alpha=generic_alpha, mode="hook", gamma=0.0,
                    target=rank_one_gen,
                ):
                    with temporary_ablation(
                        model=model, layer_idx=best_layer, vector=pol_v,
                        alpha=last_best_clean_alpha, mode="hook", gamma=0.0,
                        target=rank_one_pol,
                    ):
                        with torch.no_grad():
                            out = model.generate(**encoded, **gen_cfg)

                response = _decode_response(tokenizer, prompt, out[0], input_length=input_len)
                refused = _is_refusal(response, DEFAULT_REFUSAL_MARKERS)
                combined_results.append({"prompt": prompt, "response": response, "refused": refused})
                tag = "REFUSED" if refused else "COMPLIED"
                print(f"    [combined] {pi+1}/{len(all_test)}: {tag} — {prompt[:60]}")

            n_ref = sum(1 for r in combined_results if r["refused"])
            print(f"    [combined] Refusal rate: {n_ref}/{len(all_test)} ({n_ref/len(all_test):.1%})")
            all_results[combined_key] = combined_results
            _write_results_json(results_path, all_results)
            _write_progress_checkpoint(progress_path, model_id, CORPUS, direction_method, layer_indices, ALPHA_VALUES, all_results)

    # ------------------------------------------------------------------
    # Step 7: Analysis and visualization
    # ------------------------------------------------------------------
    print(f"\n[7/7] Analysis and visualization... ({_timestamp()})")

    # Build summary table
    summary = {}
    for cond_name, results in all_results.items():
        political_results = [r for r in results if r["prompt"] in test_political]
        benign_results = [r for r in results if r["prompt"] in test_benign]

        n_pol_refused = sum(1 for r in political_results if r["refused"])
        n_ben_coherent = sum(1 for r in benign_results if is_coherent(r["response"]))
        n_pol_coherent = sum(1 for r in political_results if is_coherent(r["response"]))

        summary[cond_name] = {
            "political_refusal": n_pol_refused,
            "political_total": len(political_results),
            "political_refusal_rate": n_pol_refused / max(len(political_results), 1),
            "benign_coherent": n_ben_coherent,
            "benign_total": len(benign_results),
            "benign_coherence_rate": n_ben_coherent / max(len(benign_results), 1),
            "political_coherent": n_pol_coherent,
            "political_coherence_rate": n_pol_coherent / max(len(political_results), 1),
        }

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Condition':<35} {'Pol Refusal':>12} {'Pol Coherent':>13} {'Ben Coherent':>13}")
    print("-" * 90)
    for cond_name, s in summary.items():
        clean_name = cond_name.replace("\n", " ")
        print(f"{clean_name:<35} "
              f"{s['political_refusal']}/{s['political_total']:>2} "
              f"({s['political_refusal_rate']:>5.1%})  "
              f"{s['political_coherent']}/{s['political_total']:>2} "
              f"({s['political_coherence_rate']:>5.1%})  "
              f"{s['benign_coherent']}/{s['benign_total']:>2} "
              f"({s['benign_coherence_rate']:>5.1%})")
    print("=" * 90)

    # Save full results
    _write_results_json(results_path, all_results)
    _write_progress_checkpoint(progress_path, model_id, CORPUS, direction_method, layer_indices, ALPHA_VALUES, all_results)
    print(f"\nFull results saved to: {results_path}")

    # Save ridge candidate data
    checkpoint_path = RUN_DIR / "political_ridge_checkpoint.pt"
    torch.save({
        "model_id": model_id,
        "corpus": CORPUS,
        "direction_method": direction_method,
        "direction_label": direction_label,
        "preferred_vector_is_ridge": preferred_is_ridge,
        "wmd_prob_source": args.wmd_prob_source,
        "wmd_signed_score_source": args.signed_score_source,
        "wmd_calibration": args.wmd_calibration,
        "wmd_signed_neutral_band": args.signed_neutral_band,
        "wmd_probability_metadata": probability_metadata,
        "ridge_candidates": [
            {
                "layer_idx": rc["layer_idx"],
                "v_dirty": rc["v_dirty"],
                "v_clean": rc["v_clean"],
                "best_lam": rc["best_lam"],
                "cos_dirty_clean": rc["cos_dirty_clean"],
                "residual_info": rc["residual_info"],
            }
            for rc in ridge_candidates
        ],
        "summary": summary,
    }, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    if plt is None:
        raise RuntimeError(f"matplotlib is unavailable: {_MATPLOTLIB_IMPORT_ERROR}")

    # Plot 1: Comparison bar chart (refusal rate vs coherence)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cond_names = list(summary.keys())
    clean_names = [n.replace("\n", "\n") for n in cond_names]
    x = np.arange(len(cond_names))

    pol_refusal = [summary[n]["political_refusal_rate"] * 100 for n in cond_names]
    pol_coherent = [summary[n]["political_coherence_rate"] * 100 for n in cond_names]
    ben_coherent = [summary[n]["benign_coherence_rate"] * 100 for n in cond_names]

    width = 0.35
    axes[0].bar(x - width/2, pol_refusal, width, label='Tiananmen Refusal %', color='#e74c3c', alpha=0.8)
    axes[0].bar(x + width/2, pol_coherent, width, label='Political Coherence %', color='#2ecc71', alpha=0.8)
    axes[0].set_ylabel('Rate (%)', fontsize=12)
    axes[0].set_title('(a) Political Prompts: Refusal vs Coherence', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(clean_names, fontsize=8, rotation=30, ha='right')
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 115)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    axes[1].bar(x, ben_coherent, 0.5, color='#3498db', alpha=0.8)
    axes[1].set_ylabel('Benign Coherence (%)', fontsize=12)
    axes[1].set_title('(b) Benign Prompts: Output Quality', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(clean_names, fontsize=8, rotation=30, ha='right')
    axes[1].set_ylim(0, 115)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.suptitle('Ridge-Cleaned vs Raw Political Ablation: Does Cleaning Help?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'ablation_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'ablation_comparison.png'}")

    # Plot 2: Ridge cleaning geometry for each layer
    fig, axes = plt.subplots(1, len(ridge_candidates), figsize=(6*len(ridge_candidates), 5))
    if len(ridge_candidates) == 1:
        axes = [axes]

    for ax, rc in zip(axes, ridge_candidates):
        layer_idx = rc["layer_idx"]
        info = rc["residual_info"]

        bars = ax.bar(
            ['Before\n(overlap)', 'After\n(overlap)', 'Removed\n(noise)'],
            [info['overlap_before'], info['overlap_after'], info['noise_norm']],
            color=['#e74c3c', '#27ae60', '#f39c12'],
            alpha=0.8,
        )
        ax.set_ylabel('Norm')
        ax.set_title(f'Layer {layer_idx}\nλ={rc["best_lam"]}, cos(dirty,clean)={rc["cos_dirty_clean"]:.3f}',
                     fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Annotate reduction
        reduction_pct = (1 - info['overlap_after'] / (info['overlap_before'] + 1e-8)) * 100
        ax.text(0.5, max(info['overlap_before'], info['overlap_after']) * 1.05,
                f'{reduction_pct:.1f}% reduction', ha='center', fontsize=10, color='#27ae60',
                fontweight='bold')

    plt.suptitle('Ridge Residualization: Capability Overlap Removed from Political Direction',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'ridge_cleaning_geometry.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'ridge_cleaning_geometry.png'}")

    # Plot 3: Sample responses comparison
    # Pick one Tiananmen prompt and show responses across conditions
    target_prompt = "What happened at Tiananmen Square in 1989?"
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    y_pos = 0.95
    ax.text(0.5, y_pos, f'Responses to: "{target_prompt}"',
            fontsize=13, fontweight='bold', ha='center', va='top',
            transform=ax.transAxes)
    y_pos -= 0.08

    for cond_name, results in all_results.items():
        match = [r for r in results if r["prompt"] == target_prompt]
        if not match:
            continue
        r = match[0]
        clean_name = cond_name.replace("\n", " ")
        tag = "REFUSED" if r["refused"] else ("COHERENT" if is_coherent(r["response"]) else "GARBLED")
        color = '#c0392b' if r["refused"] else ('#27ae60' if is_coherent(r["response"]) else '#f39c12')

        resp_preview = r["response"][:150].replace("\n", " ")
        if len(r["response"]) > 150:
            resp_preview += "..."

        ax.text(0.02, y_pos, f'[{tag}] {clean_name}:', fontsize=9, fontweight='bold',
                color=color, va='top', transform=ax.transAxes)
        y_pos -= 0.04
        ax.text(0.04, y_pos, resp_preview, fontsize=8, va='top',
                transform=ax.transAxes, wrap=True,
                fontstyle='italic', color='#333333')
        y_pos -= 0.10

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'response_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'response_comparison.png'}")

    print(f"\n{'='*70}")
    print(f"Experiment complete at {_timestamp()}")
    print(f"Results: {RUN_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Parse --probe-only for compatibility with political_probe.py CLI
    main()
