#!/usr/bin/env python3
"""
Phase B+.2: Null Probe Control

Tests whether "any two semantically distinct topic clusters" are linearly
separable, or whether political sensitivity is specially encoded.

Design: Train the same ridge probe on non-political topic pairs that are
semantically distinct but have no political dimension. If these also achieve
~100% accuracy, the separability finding is trivially expected and the paper
must reframe around geometry/behavior, not separability.

Control pairs:
  1. Science vs History (8+8 prompts)
  2. Food vs Technology (8+8 prompts)
  3. Geography vs Music (8+8 prompts)

For each pair, extract hidden states at the same layers as the political
probes, train the same ridge classifier, report accuracy.

Output: runs/null_probe_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier

from ablation_ridge_regression import (
    load_model_and_tokenizer,
    _resolve_transformer_layers,
)
from political_probe import extract_per_prompt_hidden_states


# ============================================================
# Non-political control prompt pairs
# ============================================================

CONTROL_PAIRS = {
    "science_vs_history": {
        "positive": [  # science
            "Explain how photosynthesis works in plants.",
            "What is the structure of DNA and how does it replicate?",
            "How do vaccines stimulate the immune system?",
            "What causes earthquakes and how are they measured?",
            "Explain the difference between nuclear fission and fusion.",
            "How does the human digestive system process food?",
            "What is the theory of relativity in simple terms?",
            "How do antibiotics work to kill bacteria?",
        ],
        "negative": [  # history
            "What caused the fall of the Roman Empire?",
            "Who was Alexander the Great and what did he achieve?",
            "What were the main causes of World War I?",
            "How did the Industrial Revolution change society?",
            "What was the significance of the Magna Carta?",
            "Who were the key figures in the French Revolution?",
            "What led to the collapse of the Soviet Union?",
            "How did the Renaissance transform European culture?",
        ],
    },
    "food_vs_technology": {
        "positive": [  # food
            "How do you make traditional Italian pasta from scratch?",
            "What is the history and preparation of sushi?",
            "How does fermentation work in making bread?",
            "What are the key spices in Indian curry?",
            "How is chocolate made from cacao beans?",
            "What is the difference between baking and roasting?",
            "How do you properly brew coffee?",
            "What makes French pastry different from other baking traditions?",
        ],
        "negative": [  # technology
            "How does WiFi transmit data wirelessly?",
            "What is a GPU and how does it differ from a CPU?",
            "How do lithium-ion batteries store energy?",
            "What is blockchain technology and how does it work?",
            "How does a touchscreen detect finger input?",
            "What is the difference between 4G and 5G networks?",
            "How do noise-cancelling headphones work?",
            "What is cloud computing and why is it useful?",
        ],
    },
    "geography_vs_music": {
        "positive": [  # geography
            "Where is Mount Everest and how tall is it?",
            "What is the Nile River and why is it important?",
            "How were the Grand Canyon's rock layers formed?",
            "What causes the Northern Lights?",
            "Why is the Dead Sea so salty?",
            "How do ocean currents affect global climate?",
            "What is the Ring of Fire in the Pacific Ocean?",
            "Why does Iceland have so many geysers and volcanoes?",
        ],
        "negative": [  # music
            "Who composed Beethoven's Symphony No. 5?",
            "What is jazz and how did it originate?",
            "How does a piano produce different notes?",
            "What is the difference between classical and baroque music?",
            "Who was Mozart and what were his major works?",
            "How did rock and roll emerge in the 1950s?",
            "What makes a chord major versus minor?",
            "How has hip-hop influenced modern popular music?",
        ],
    },
}

# Default models to test (primary + Western control)
DEFAULT_MODELS = [
    "Qwen/Qwen3-8B",
    "microsoft/Phi-4-mini-instruct",
]

DEFAULT_RESULTS_FILE = Path("runs/null_probe_results.json")


def probe_topic_pair(
    model: Any,
    tokenizer: Any,
    positive_prompts: List[str],
    negative_prompts: List[str],
    probe_layers: List[int],
    pair_name: str,
) -> Dict[str, Any]:
    """Train ridge probes on a topic pair at each layer."""
    all_prompts = positive_prompts + negative_prompts
    labels = np.array([1] * len(positive_prompts) + [0] * len(negative_prompts))

    results = {"pair": pair_name, "layers": {}}

    for layer_idx in probe_layers:
        print(f"    Layer {layer_idx}...", end="\r", flush=True)
        states = extract_per_prompt_hidden_states(model, tokenizer, all_prompts, layer_idx)
        X = states.numpy()

        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X, labels)
        train_acc = clf.score(X, labels)

        results["layers"][layer_idx] = {
            "train_accuracy": float(train_acc),
            "n_positive": len(positive_prompts),
            "n_negative": len(negative_prompts),
        }
        print(f"    Layer {layer_idx}: train_acc={train_acc:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run null probe controls on one or more models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model IDs to evaluate (default: Qwen/Qwen3-8B, microsoft/Phi-4-mini-instruct)",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=str(DEFAULT_RESULTS_FILE),
        help=f"Output JSON path (default: {DEFAULT_RESULTS_FILE})",
    )
    args = parser.parse_args()

    results_file = Path(args.results_file)
    models = list(args.models)

    results_file.parent.mkdir(parents=True, exist_ok=True)

    all_results = {}
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)

    for model_id in models:
        if model_id in all_results:
            print(f"\nSKIP: {model_id} (already done)")
            continue

        print(f"\n{'='*60}")
        print(f"Null Probe Control: {model_id}")
        print(f"{'='*60}")

        t0 = time.time()
        model, tokenizer = load_model_and_tokenizer(model_id)
        n_layers = len(_resolve_transformer_layers(model))
        probe_layers = list(range(0, n_layers, max(1, n_layers // 12)))

        model_results = {"model_id": model_id, "n_layers": n_layers, "pairs": {}}

        for pair_name, pair_data in CONTROL_PAIRS.items():
            print(f"\n  --- {pair_name} ---")
            pair_result = probe_topic_pair(
                model, tokenizer,
                pair_data["positive"], pair_data["negative"],
                probe_layers, pair_name,
            )
            model_results["pairs"][pair_name] = pair_result

        elapsed = time.time() - t0
        model_results["elapsed_seconds"] = elapsed
        all_results[model_id] = model_results

        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved ({elapsed:.0f}s)")

        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary
    print(f"\n{'='*60}")
    print("NULL PROBE vs POLITICAL PROBE COMPARISON")
    print(f"{'='*60}")
    for model_id, data in all_results.items():
        print(f"\n{model_id}:")
        for pair_name, pair_data in data.get("pairs", {}).items():
            best_layer = max(pair_data["layers"], key=lambda l: pair_data["layers"][l]["train_accuracy"])
            best_acc = pair_data["layers"][best_layer]["train_accuracy"]
            print(f"  {pair_name}: best={best_acc:.1%} at L{best_layer}")
        print(f"  (Political probe: 100% at all layers for comparison)")


if __name__ == "__main__":
    main()
