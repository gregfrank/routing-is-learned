#!/usr/bin/env python3
"""
Category-held-out cross-validation on existing political probe data.

For each probe run (runs/political_probe*/probe_data.pt), this script:
  1. Loads or extracts per-prompt hidden states at each probed layer
  2. Assigns category labels to prompts (v1: 6 categories, v2: 15 categories)
  3. Performs leave-one-category-out cross-validation with RidgeClassifier
  4. Reports per-fold accuracy and mean across folds at each layer

The script caches extracted hidden states as hidden_states_cache.pt in each
run directory, so subsequent runs skip the expensive model forward passes.

Usage:
    python3 probe_cross_validation.py                  # all runs, extract if needed
    python3 probe_cross_validation.py --cached-only    # only runs with cached states
    python3 probe_cross_validation.py --model Qwen/Qwen3-8B --run runs/political_probe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

# V1 category boundaries (index into POLITICAL_REFUSED_PROMPTS / POLITICAL_ANSWERED_PROMPTS)
V1_CATEGORIES = [
    ("Tiananmen", 0, 8),
    ("Tibet", 8, 12),
    ("Xinjiang", 12, 16),
    ("Xi_CCP", 16, 20),
    ("Hong_Kong", 20, 22),
    ("COVID", 22, 24),
]

# Sanity check
V1_N_POLITICAL = sum(end - start for _, start, end in V1_CATEGORIES)
V1_N_CONTROL = V1_N_POLITICAL
assert V1_N_POLITICAL == 24


def _load_v1_prompt_pairs() -> Tuple[List[str], List[str]]:
    """Load v1 political/control prompt lists lazily (torch dependency friendly)."""
    from political_prompts_v1 import (
        POLITICAL_REFUSED_PROMPTS,
        POLITICAL_ANSWERED_PROMPTS,
    )
    return list(POLITICAL_REFUSED_PROMPTS), list(POLITICAL_ANSWERED_PROMPTS)


def build_category_labels_v1() -> List[str]:
    """Return a category string for each of the 48 prompts (24 political + 24 control)."""
    cats = []
    for cat_name, start, end in V1_CATEGORIES:
        for _ in range(start, end):
            cats.append(cat_name)
    # Controls get the same categories (index-matched)
    return cats + cats


def build_binary_labels_v1() -> np.ndarray:
    """Return 0/1 labels: 1 = political (first 24), 0 = control (last 24)."""
    return np.array([1] * V1_N_POLITICAL + [0] * V1_N_CONTROL, dtype=np.int64)


# ---------------------------------------------------------------------------
# Disk management
# ---------------------------------------------------------------------------

def _clean_hf_cache(model_id: str) -> None:
    """Remove HuggingFace cache for a model to free disk space."""
    import shutil
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    # HF stores models as models--org--name
    safe = "models--" + model_id.replace("/", "--")
    model_cache = cache_dir / safe
    if model_cache.exists():
        size_mb = sum(f.stat().st_size for f in model_cache.rglob("*") if f.is_file()) / 1e6
        shutil.rmtree(model_cache)
        print(f"  Cleaned HF cache: {safe} ({size_mb:.0f} MB freed)")


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write JSON to avoid corrupt partial files on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _load_hidden_states_from_npz(run_dir: Path) -> Optional[Dict[int, np.ndarray]]:
    """Load portable hidden-state cache (numpy) if present."""
    npz_path = run_dir / "portable" / "hidden_states_cache.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    result: Dict[int, np.ndarray] = {}
    for key in data.files:
        if key.startswith("layer_"):
            layer = int(key.split("_", 1)[1])
            result[layer] = np.asarray(data[key])
    return result if result else None


def _save_hidden_states_npz(
    run_dir: Path,
    hidden_states: Dict[int, Any],
    model_id: str,
    prompts: Sequence[str],
) -> None:
    """Save a portable hidden-state cache for local CPU analysis."""
    portable_dir = run_dir / "portable"
    portable_dir.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, np.ndarray] = {}
    for layer, value in hidden_states.items():
        arr = value.detach().cpu().numpy() if torch is not None and torch.is_tensor(value) else np.asarray(value)
        arrays[f"layer_{int(layer)}"] = arr
    np.savez_compressed(portable_dir / "hidden_states_cache.npz", **arrays)

    meta = {
        "model_id": model_id,
        "n_political": V1_N_POLITICAL,
        "n_control": V1_N_CONTROL,
        "layers": sorted(int(l) for l in hidden_states.keys()),
        "prompts": list(prompts),
    }
    _write_json_atomic(portable_dir / "hidden_states_meta.json", meta)


def _load_probe_metadata(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load probe metadata from portable JSON first, then .pt."""
    portable_json = run_dir / "portable" / "probe_data_portable.json"
    if portable_json.exists():
        try:
            return json.loads(portable_json.read_text())
        except Exception as e:
            print(f"  WARNING: Failed reading portable metadata {portable_json}: {e}")

    probe_pt = run_dir / "probe_data.pt"
    if not probe_pt.exists():
        return None
    if torch is None:
        print("  WARNING: torch not installed and no portable metadata available.")
        return None

    raw = torch.load(probe_pt, map_location="cpu", weights_only=False)
    political = raw.get("political_probe", {})
    layers = sorted(int(l) for l in political.keys())
    return {
        "model_id": raw.get("model_id"),
        "layers": layers,
        "political_probe": {
            str(int(layer)): {"accuracy": float(political[layer].get("accuracy", float("nan")))}
            for layer in political
        },
    }


# ---------------------------------------------------------------------------
# Hidden state extraction (requires model)
# ---------------------------------------------------------------------------

def extract_hidden_states(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    layers: Sequence[int],
) -> Dict[int, torch.Tensor]:
    """Extract last-token hidden states at each layer for each prompt.

    Returns: {layer_idx: tensor of shape (n_prompts, hidden_dim)}
    """
    from ablation_ridge_regression import (
        _resolve_device,
        _resolve_transformer_layers,
        _tokenize,
        _to_device,
    )

    dev = _resolve_device(model)
    model.eval()
    transformer_layers = _resolve_transformer_layers(model)

    result = {}
    for li, layer_idx in enumerate(layers):
        target_module = transformer_layers[layer_idx]
        states = []

        for pi, prompt in enumerate(prompts):
            print(f"  Layer {layer_idx} [{li+1}/{len(layers)}], "
                  f"prompt {pi+1}/{len(prompts)}", end="\r", flush=True)

            encoded = _tokenize(tokenizer, prompt, max_length=256,
                                padding=False, chat_template=True)
            encoded = _to_device(encoded, dev)

            captured = []
            def _hook_fn(_module, _input, output, _captured=captured):
                if isinstance(output, tuple):
                    _captured.append(output[0].detach())
                else:
                    _captured.append(output.detach())

            handle = target_module.register_forward_hook(_hook_fn)
            with torch.no_grad():
                model(**encoded)
            handle.remove()

            if captured:
                h = captured[0][0, -1, :].cpu().float()
                states.append(h)

        result[layer_idx] = torch.stack(states)
        print(f"  Layer {layer_idx}: extracted {len(states)} states, "
              f"shape {result[layer_idx].shape}          ")

    return result


def load_or_extract_hidden_states(
    run_dir: Path,
    probe_data: Dict,
    model_id: Optional[str] = None,
    force_extract: bool = False,
) -> Optional[Dict[int, Any]]:
    """Load cached hidden states, or extract from model if not cached.

    Returns None if extraction is not possible (no model available).
    """
    cache_pt_path = run_dir / "hidden_states_cache.pt"
    cache_npz = _load_hidden_states_from_npz(run_dir)
    if cache_npz is not None and not force_extract:
        print(f"  Loading portable hidden states from {run_dir / 'portable' / 'hidden_states_cache.npz'}")
        return cache_npz

    if cache_pt_path.exists() and not force_extract:
        if torch is None:
            print("  WARNING: hidden_states_cache.pt exists but torch is unavailable.")
            return None
        print(f"  Loading cached hidden states from {cache_pt_path}")
        cached = torch.load(cache_pt_path, map_location="cpu", weights_only=False)
        hidden_states = cached["hidden_states"]
        _save_hidden_states_npz(
            run_dir,
            hidden_states,
            model_id=str(cached.get("model_id") or model_id or probe_data.get("model_id") or run_dir.name),
            prompts=list(cached.get("prompts") or []),
        )
        return hidden_states

    # Need to extract -- requires model
    if torch is None:
        print("  WARNING: torch unavailable; cannot extract hidden states.")
        return None

    mid = model_id or probe_data.get("model_id")
    if mid is None:
        print(f"  WARNING: No model_id found and no cache; skipping {run_dir.name}")
        return None

    print(f"  Extracting hidden states from model: {mid}")
    try:
        from ablation_ridge_regression import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(mid)
    except Exception as e:
        print(f"  ERROR loading model {mid}: {e}")
        return None

    layers = sorted(probe_data["political_probe"].keys())

    political_prompts, control_prompts = _load_v1_prompt_pairs()
    all_prompts = political_prompts + control_prompts
    hidden_states = extract_hidden_states(model, tokenizer, all_prompts, layers)

    # Cache for future runs
    torch.save({
        "hidden_states": hidden_states,
        "model_id": mid,
        "prompts": all_prompts,
        "n_political": V1_N_POLITICAL,
        "n_control": V1_N_CONTROL,
    }, cache_pt_path)
    print(f"  Cached hidden states to {cache_pt_path}")
    _save_hidden_states_npz(run_dir, hidden_states, model_id=str(mid), prompts=all_prompts)

    # Free model memory and disk (HF cache can fill small disks)
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc; gc.collect()

    # Clean HF cache for this model to free disk for the next one
    _clean_hf_cache(mid)

    return hidden_states


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def leave_one_category_out_cv(
    hidden_states: Dict[int, torch.Tensor],
    labels: np.ndarray,
    categories: List[str],
    alpha: float = 1.0,
) -> Dict[int, Dict[str, Any]]:
    """Leave-one-category-out cross-validation at each layer.

    Args:
        hidden_states: {layer_idx: (n_samples, hidden_dim)}
        labels: binary labels (n_samples,)
        categories: category string per sample (n_samples,)
        alpha: regularization strength for RidgeClassifier

    Returns: {layer_idx: {"per_fold": {cat: acc}, "mean_acc": float, ...}}
    """
    from sklearn.linear_model import RidgeClassifier

    unique_cats = sorted(set(categories))
    cat_array = np.array(categories)
    layers = sorted(hidden_states.keys())

    results = {}
    for layer_idx in layers:
        layer_data = hidden_states[layer_idx]
        if torch is not None and torch.is_tensor(layer_data):
            X = layer_data.detach().cpu().numpy()
        else:
            X = np.asarray(layer_data)
        y = labels

        fold_accuracies = {}
        all_preds = np.zeros_like(y, dtype=float)
        all_correct = np.zeros_like(y, dtype=bool)

        for cat in unique_cats:
            # Test set: prompts from this category (both political and control)
            # The categories list has the same category for political and its
            # matched control, so masking by category grabs both.
            test_mask = cat_array == cat
            train_mask = ~test_mask

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                # Skip degenerate fold
                fold_accuracies[cat] = float("nan")
                continue

            clf = RidgeClassifier(alpha=alpha)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)

            acc = (preds == y_test).mean()
            fold_accuracies[cat] = float(acc)
            all_correct[test_mask] = preds == y_test

        valid_accs = [a for a in fold_accuracies.values() if not np.isnan(a)]
        mean_acc = float(np.mean(valid_accs)) if valid_accs else float("nan")
        overall_acc = float(all_correct.mean())

        results[layer_idx] = {
            "per_fold": fold_accuracies,
            "mean_fold_acc": mean_acc,
            "overall_acc": overall_acc,
            "n_folds": len(valid_accs),
        }

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary_table(
    run_name: str,
    cv_results: Dict[int, Dict[str, Any]],
    full_train_results: Dict[int, float],
):
    """Print a formatted summary table."""
    layers = sorted(cv_results.keys())
    cats = sorted(next(iter(cv_results.values()))["per_fold"].keys())

    # Header
    print(f"\n{'='*80}")
    print(f"  {run_name}")
    print(f"{'='*80}")

    # Column headers
    header = f"{'Layer':>6}  {'Full':>6}  {'CV-Mean':>7}  {'Overall':>7}"
    for cat in cats:
        short = cat[:8]
        header += f"  {short:>8}"
    print(header)
    print("-" * len(header))

    for layer_idx in layers:
        cv = cv_results[layer_idx]
        full = full_train_results.get(layer_idx, float("nan"))
        line = f"{layer_idx:>6}  {full:>6.3f}  {cv['mean_fold_acc']:>7.3f}  {cv['overall_acc']:>7.3f}"
        for cat in cats:
            acc = cv["per_fold"].get(cat, float("nan"))
            if np.isnan(acc):
                line += f"  {'N/A':>8}"
            else:
                line += f"  {acc:>8.3f}"
        print(line)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_one_run(
    run_dir: Path,
    model_id: Optional[str] = None,
    cached_only: bool = False,
    force_extract: bool = False,
    alpha: float = 1.0,
) -> Optional[Dict]:
    """Process a single probe run. Returns results dict or None."""
    probe_data = _load_probe_metadata(run_dir)
    if probe_data is None:
        return None

    print(f"\n--- Processing: {run_dir.name} ---")

    mid = model_id or probe_data.get("model_id", run_dir.name)
    print(f"  Model: {mid}")

    # Get layers from probe data
    if probe_data.get("layers"):
        layers = sorted(int(l) for l in probe_data["layers"])
    else:
        layers = sorted(int(l) for l in probe_data.get("political_probe", {}).keys())
    print(f"  Layers: {layers}")

    # Check cache availability
    cache_pt_path = run_dir / "hidden_states_cache.pt"
    cache_npz_path = run_dir / "portable" / "hidden_states_cache.npz"
    if cached_only and not cache_pt_path.exists() and not cache_npz_path.exists():
        print(f"  SKIPPED (no cache and --cached-only)")
        return None

    if cached_only and force_extract:
        print("  WARNING: --cached-only with --force-extract is contradictory; honoring --cached-only.")
        force_extract = False

    hidden_states = load_or_extract_hidden_states(
        run_dir, probe_data, model_id=model_id, force_extract=force_extract,
    )

    if hidden_states is None:
        return None

    # Verify shapes
    n_prompts = V1_N_POLITICAL + V1_N_CONTROL
    sample_layer = list(hidden_states.keys())[0]
    actual_n = hidden_states[sample_layer].shape[0]
    if actual_n != n_prompts:
        print(f"  WARNING: Expected {n_prompts} prompts but got {actual_n}. "
              f"Skipping (prompt corpus mismatch).")
        return None

    # Build labels and categories
    labels = build_binary_labels_v1()
    categories = build_category_labels_v1()

    # Filter hidden_states to only layers present in probe data
    available_layers = sorted(set(hidden_states.keys()) & set(layers))
    if not available_layers:
        print(f"  WARNING: No overlapping layers between cache and probe data")
        return None

    hs_filtered = {l: hidden_states[l] for l in available_layers}

    # Run cross-validation
    print(f"  Running leave-one-category-out CV (alpha={alpha})...")
    cv_results = leave_one_category_out_cv(hs_filtered, labels, categories, alpha=alpha)

    # Get full-training accuracies from probe data for comparison
    full_train: Dict[int, float] = {}
    for layer_key, value in probe_data.get("political_probe", {}).items():
        li = int(layer_key)
        if li in available_layers:
            full_train[li] = float(value.get("accuracy", float("nan")))

    # Print summary
    print_summary_table(f"{run_dir.name} (model: {mid})", cv_results, full_train)

    # Build serializable result
    result = {
        "run_dir": str(run_dir),
        "model_id": str(mid),
        "alpha": alpha,
        "layers": {},
    }
    for layer_idx in available_layers:
        cv = cv_results[layer_idx]
        result["layers"][int(layer_idx)] = {
            "full_train_acc": full_train.get(layer_idx, None),
            "cv_mean_fold_acc": cv["mean_fold_acc"],
            "cv_overall_acc": cv["overall_acc"],
            "per_fold": cv["per_fold"],
            "n_folds": cv["n_folds"],
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Category-held-out cross-validation on political probe data"
    )
    parser.add_argument(
        "--run", type=str, default=None,
        help="Process a specific run directory (default: all runs/political_probe*/)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model ID for hidden state extraction"
    )
    parser.add_argument(
        "--cached-only", action="store_true",
        help="Only process runs that already have cached hidden states"
    )
    parser.add_argument(
        "--force-extract", action="store_true",
        help="Re-extract hidden states even if cache exists"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Ridge regularization strength (default: 1.0)"
    )
    parser.add_argument(
        "--output", type=str, default="runs/cross_validation_results.json",
        help="Output JSON path (default: runs/cross_validation_results.json)"
    )
    args = parser.parse_args()

    # Find runs to process
    if args.run:
        run_dirs = [Path(args.run)]
    else:
        runs_root = Path("runs")
        run_dirs = sorted(runs_root.glob("political_probe*"))
        run_dirs = [
            d for d in run_dirs
            if d.is_dir() and (
                (d / "probe_data.pt").exists()
                or (d / "portable" / "probe_data_portable.json").exists()
            )
        ]

    if not run_dirs:
        print("No probe runs found.")
        sys.exit(1)

    print(f"Found {len(run_dirs)} probe run(s)")
    for d in run_dirs:
        print(f"  {d}")

    # Process each run
    out_path = Path(args.output)
    checkpoint_path = out_path.with_name(f"{out_path.stem}_checkpoint{out_path.suffix}")
    all_results = {}
    for run_dir in run_dirs:
        result = process_one_run(
            run_dir,
            model_id=args.model,
            cached_only=args.cached_only,
            force_extract=args.force_extract,
            alpha=args.alpha,
        )
        if result is not None:
            all_results[run_dir.name] = result
            # Incremental checkpoint so preemption does not lose all progress.
            _write_json_atomic(checkpoint_path, all_results)
            print(f"  Checkpoint saved: {checkpoint_path}")

    # Save combined results
    if all_results:
        _write_json_atomic(out_path, all_results)
        print(f"\nResults saved to: {out_path}")

        # Print cross-model summary
        print(f"\n{'='*80}")
        print("  CROSS-MODEL SUMMARY")
        print(f"{'='*80}")
        print(f"{'Run':<55} {'Best-Layer':>10} {'Full-Acc':>9} {'CV-Mean':>8} {'CV-Overall':>10}")
        print("-" * 95)
        for name, res in sorted(all_results.items()):
            # Find best layer by CV mean accuracy
            best_layer = max(
                res["layers"],
                key=lambda l: res["layers"][l]["cv_mean_fold_acc"],
            )
            bl = res["layers"][best_layer]
            full_acc = bl["full_train_acc"] if bl["full_train_acc"] is not None else float("nan")
            print(f"{name:<55} {best_layer:>10} {full_acc:>9.3f} "
                  f"{bl['cv_mean_fold_acc']:>8.3f} {bl['cv_overall_acc']:>10.3f}")
        print()
    else:
        print("\nNo runs could be processed. Use --force-extract with --model "
              "to extract hidden states, or ensure hidden_states_cache.pt files exist.")


if __name__ == "__main__":
    main()
