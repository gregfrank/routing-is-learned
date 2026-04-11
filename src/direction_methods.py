#!/usr/bin/env python3
"""
Shared direction-extraction utilities for CAA and WMD experiments.

This module is intentionally torch/lightweight so GPU-side scripts can reuse it
for vector construction while CPU-side scripts can reuse the probability-loading
and calibration path against cached probe activations.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


DIRECTION_METHOD_CHOICES: Tuple[str, ...] = (
    "caa",
    "caa_ridge",
    "wmd",
    "wmd_ridge",
)
WMD_VARIANT_CHOICES: Tuple[str, ...] = (
    "legacy",
    "cyberey",
)
CALIBRATION_CHOICES: Tuple[str, ...] = (
    "none",
    "platt",
    "isotonic",
)
DEFAULT_WMD_NEUTRAL_BAND: float = 0.05
DEFAULT_SIGNED_SCORE_NEUTRAL_BAND: float = 0.25

_PROMPT_FIELD_CANDIDATES: Tuple[str, ...] = (
    "prompt",
    "prompt_text",
    "text",
    "question",
    "input",
)
_PROBABILITY_FIELD_CANDIDATES: Tuple[str, ...] = (
    "p_censor",
    "probability",
    "prob",
    "score",
    "p_refusal",
    "refusal_probability",
    "censorship_probability",
)


def parse_direction_method(method: str) -> Tuple[str, bool]:
    if method not in DIRECTION_METHOD_CHOICES:
        raise ValueError(
            f"Unsupported direction method {method!r}. "
            f"Expected one of {DIRECTION_METHOD_CHOICES}."
        )
    return ("wmd" if method.startswith("wmd") else "caa"), method.endswith("_ridge")


def compute_caa_direction(
    harmful_states: torch.Tensor,
    harmless_states: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    if harmful_states.ndim != 2 or harmless_states.ndim != 2:
        raise ValueError("Expected 2D tensors [n, d] for hidden states.")
    if harmful_states.shape[1] != harmless_states.shape[1]:
        raise ValueError("Hidden state widths must match.")
    v = harmful_states.mean(dim=0) - harmless_states.mean(dim=0)
    v_norm = v.norm()
    if v_norm < eps:
        import warnings
        warnings.warn(
            f"CAA direction norm is {v_norm.item():.2e}, near zero. "
            "Positive and negative distributions may be indistinguishable."
        )
    return F.normalize(v, dim=0, eps=eps)


def compute_wmd_direction(
    positive_states: torch.Tensor,
    negative_states: torch.Tensor,
    positive_probabilities: torch.Tensor,
    negative_probabilities: torch.Tensor,
    eps: float = 1e-12,
    neutral_states: Optional[torch.Tensor] = None,
    normalize_weighted_means: bool = False,
) -> torch.Tensor:
    if positive_states.ndim != 2 or negative_states.ndim != 2:
        raise ValueError("Expected 2D tensors [n, d] for hidden states.")
    if positive_states.shape[1] != negative_states.shape[1]:
        raise ValueError("Hidden state widths must match.")

    pos_states = positive_states.to(dtype=torch.float32)
    neg_states = negative_states.to(dtype=torch.float32)
    pos_weights = positive_probabilities.to(dtype=torch.float32)
    neg_weights = negative_probabilities.to(dtype=torch.float32)

    if pos_weights.ndim != 1 or pos_weights.shape[0] != pos_states.shape[0]:
        raise ValueError("Positive weight vector must be 1D and aligned with positive state rows.")
    if neg_weights.ndim != 1 or neg_weights.shape[0] != neg_states.shape[0]:
        raise ValueError("Negative weight vector must be 1D and aligned with negative state rows.")

    if neutral_states is not None:
        if neutral_states.ndim != 2:
            raise ValueError("neutral_states must be 2D [n, d].")
        if neutral_states.shape[1] != pos_states.shape[1]:
            raise ValueError("neutral_states width must match positive/negative states.")
        offset = neutral_states.to(dtype=torch.float32).mean(dim=0)
        pos_states = pos_states - offset
        neg_states = neg_states - offset

    positive_mass = float(pos_weights.sum().abs().item())
    negative_mass = float(neg_weights.sum().abs().item())
    if positive_mass < eps or negative_mass < eps:
        raise ValueError("WMD requires non-zero positive and negative weight mass.")

    pos_mean = (pos_states * pos_weights.unsqueeze(1)).sum(dim=0) / pos_weights.sum()
    neg_mean = (neg_states * neg_weights.unsqueeze(1)).sum(dim=0) / neg_weights.sum()
    if normalize_weighted_means:
        pos_mean = F.normalize(pos_mean, dim=0, eps=eps)
        neg_mean = F.normalize(neg_mean, dim=0, eps=eps)
    v = pos_mean - neg_mean
    return F.normalize(v, dim=0, eps=eps)


def build_cyberey_neutral_mask(
    probabilities: np.ndarray | Sequence[float],
    neutral_band: float = 0.05,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if neutral_band < 0:
        raise ValueError("neutral_band must be non-negative")
    return np.abs(probs - 0.5) <= neutral_band


def build_label_signed_scores(
    probabilities: np.ndarray | Sequence[float],
    labels: np.ndarray | Sequence[int],
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    labs = np.asarray(labels, dtype=np.int64)
    if probs.shape != labs.shape:
        raise ValueError("probabilities and labels must have the same shape")
    return np.where(labs >= 1, probs, -(1.0 - probs))


def load_model_prompt_signed_scores(
    source: str | Path,
    model_id: str,
    prompt_order: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Signed-score artifact not found: {path}")

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            records = list(csv.DictReader(fh))
    elif path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "records" in payload:
            records = list(payload["records"])
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError("Unsupported signed-score JSON payload structure.")
    else:
        raise ValueError("Signed-score artifact must be .csv, .jsonl, or .json.")

    mapping: Dict[str, float] = {}
    category_counts: Dict[str, int] = {}
    for row in records:
        if str(row.get("model_id")) != model_id:
            continue
        prompt = str(row.get("prompt"))
        score = float(row.get("signed_score"))
        if prompt in mapping and abs(mapping[prompt] - score) > 1e-9:
            raise ValueError(f"Conflicting signed scores for {model_id} prompt {prompt!r}")
        mapping[prompt] = score
        category = str(row.get("category", "UNKNOWN"))
        category_counts[category] = category_counts.get(category, 0) + 1

    missing = [prompt for prompt in prompt_order if prompt not in mapping]
    if missing:
        raise KeyError(
            f"Signed-score artifact is missing {len(missing)} prompts for {model_id}. "
            f"Examples: {missing[:3]!r}"
        )

    scores = np.asarray([mapping[prompt] for prompt in prompt_order], dtype=np.float32)
    metadata = {
        "source": str(path),
        "model_id": model_id,
        "prompt_count": int(len(prompt_order)),
        "min": float(scores.min()) if scores.size else None,
        "max": float(scores.max()) if scores.size else None,
        "mean": float(scores.mean()) if scores.size else None,
        "std": float(scores.std()) if scores.size else None,
        "category_counts": dict(category_counts),
    }
    return scores, metadata


def resolve_wmd_inputs(
    prompt_order: Sequence[str],
    positive_count: int,
    *,
    model_id: Optional[str] = None,
    probability_source: Optional[str | Path] = None,
    signed_score_source: Optional[str | Path] = None,
    calibration: str = "none",
    labels: Optional[np.ndarray | Sequence[int]] = None,
    probability_neutral_band: float = DEFAULT_WMD_NEUTRAL_BAND,
    signed_neutral_band: float = DEFAULT_SIGNED_SCORE_NEUTRAL_BAND,
) -> Dict[str, Any]:
    total_count = len(prompt_order)
    negative_count = total_count - positive_count
    if positive_count < 0 or negative_count < 0:
        raise ValueError("positive_count must be between 0 and len(prompt_order)")
    if signed_score_source and probability_source:
        raise ValueError("Provide either signed_score_source or probability_source, not both.")

    if labels is None:
        labels_array = np.asarray([1] * positive_count + [0] * negative_count, dtype=np.int64)
    else:
        labels_array = np.asarray(labels, dtype=np.int64)
        if labels_array.shape != (total_count,):
            raise ValueError("labels must align with prompt_order")

    if signed_score_source:
        if not model_id:
            raise ValueError("model_id is required when loading signed_score_source")
        signed_scores, score_metadata = load_model_prompt_signed_scores(
            source=signed_score_source,
            model_id=str(model_id),
            prompt_order=prompt_order,
        )
        positive_weights_np = np.clip(signed_scores[:positive_count], 0.0, 1.0)
        negative_weights_np = np.clip(-signed_scores[positive_count:], 0.0, 1.0)
        neutral_mask = np.abs(signed_scores) <= float(signed_neutral_band)
        metadata: Dict[str, Any] = {
            "kind": "signed_score_rows",
            "signed_score": score_metadata,
            "signed_neutral_band": float(signed_neutral_band),
        }
    else:
        if probability_source is None:
            raise ValueError("Provide probability_source or signed_score_source for WMD extraction.")
        all_probabilities, metadata = resolve_prompt_probabilities(
            prompt_order=prompt_order,
            source=probability_source,
            calibration=calibration,
            labels=labels_array,
        )
        signed_scores = build_label_signed_scores(all_probabilities, labels_array)
        positive_weights_np = np.asarray(all_probabilities[:positive_count], dtype=np.float32)
        negative_weights_np = np.asarray(1.0 - all_probabilities[positive_count:], dtype=np.float32)
        neutral_mask = build_cyberey_neutral_mask(all_probabilities, neutral_band=probability_neutral_band)
        metadata = {
            **metadata,
            "kind": "prompt_probability",
            "probability_neutral_band": float(probability_neutral_band),
        }

    return {
        "positive_weights": torch.from_numpy(np.asarray(positive_weights_np, dtype=np.float32)).float(),
        "negative_weights": torch.from_numpy(np.asarray(negative_weights_np, dtype=np.float32)).float(),
        "signed_scores": np.asarray(signed_scores, dtype=np.float32),
        "neutral_mask": np.asarray(neutral_mask, dtype=bool),
        "metadata": metadata,
    }


def scalar_projection(
    acts: torch.Tensor,
    direction: torch.Tensor,
    offset: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    if acts.ndim != 2:
        raise ValueError("acts must be [n, d]")
    centered = acts.float()
    if offset is not None:
        centered = centered - offset.float()
    unit_direction = F.normalize(direction.float(), dim=0, eps=eps)
    cosine = F.cosine_similarity(centered, unit_direction.unsqueeze(0), dim=-1)
    return centered.norm(dim=-1) * cosine


def cyberey_rmse(
    projections: np.ndarray | Sequence[float],
    target_scores: np.ndarray | Sequence[float],
) -> float:
    projs = np.asarray(projections, dtype=np.float64)
    scores = np.asarray(target_scores, dtype=np.float64)
    if projs.shape != scores.shape:
        raise ValueError("projections and target_scores must have the same shape")
    mask = np.where(np.sign(scores) != np.sign(projs), 1.0, 0.0)
    return float(np.sqrt(np.mean((scores * mask) ** 2)))


def ridge_residualize(
    v_dirty: torch.Tensor,
    reference_atoms: torch.Tensor,
    lam: float = 1e-2,
    ortho_strength: float = 1.0,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if v_dirty.ndim != 1:
        raise ValueError("v_dirty must be 1D")
    if reference_atoms.ndim == 1:
        reference_atoms = reference_atoms.unsqueeze(1)
    if reference_atoms.ndim != 2:
        raise ValueError("reference_atoms must be 1D or 2D")
    if reference_atoms.shape[0] != v_dirty.shape[0]:
        raise ValueError("reference_atoms and v_dirty dimensions are incompatible")

    v = v_dirty.to(dtype=torch.float32)
    A = reference_atoms.to(device=v.device, dtype=torch.float32)

    k = int(A.shape[1])
    if k == 0:
        clean = F.normalize(v, dim=0, eps=eps)
        return clean, {
            "weights": torch.empty(0, dtype=clean.dtype),
            "overlap_before": 0.0,
            "overlap_after": 0.0,
            "noise_norm": 0.0,
            "lam": float(lam),
            "ortho_strength": float(ortho_strength),
        }

    at_a = A.T @ A
    rhs = A.T @ v
    lhs = at_a + lam * torch.eye(k, device=v.device, dtype=v.dtype)
    w = torch.linalg.solve(lhs, rhs)

    noise = A @ w
    clean = v - noise
    if ortho_strength > 0:
        clean = clean - ortho_strength * (A @ (A.T @ clean))
    clean = F.normalize(clean, dim=0, eps=eps)

    return clean, {
        "weights": w.detach().cpu(),
        "noise_norm": float(noise.norm().item()),
        "overlap_before": float(torch.norm(A.T @ v).item()),
        "overlap_after": float(torch.norm(A.T @ clean).item()),
        "lam": float(lam),
        "ortho_strength": float(ortho_strength),
    }


def format_direction_method_label(method: str) -> str:
    base_method, uses_ridge = parse_direction_method(method)
    parts = ["WMD" if base_method == "wmd" else "CAA"]
    if uses_ridge:
        parts.append("ridge")
    return "+".join(parts)


def load_prompt_probability_mapping(source: str | Path) -> Dict[str, float]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Probability artifact not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return _records_to_mapping(records)

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as fh:
            return _records_to_mapping(csv.DictReader(fh, delimiter=delimiter))

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _json_payload_to_mapping(payload)

    raise ValueError(
        f"Unsupported probability artifact format {path.suffix!r}. "
        "Expected .json, .jsonl, .csv, or .tsv."
    )


def resolve_prompt_probabilities(
    prompt_order: Sequence[str],
    source: str | Path,
    calibration: str = "none",
    labels: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if calibration not in CALIBRATION_CHOICES:
        raise ValueError(
            f"Unsupported calibration mode {calibration!r}. "
            f"Expected one of {CALIBRATION_CHOICES}."
        )

    mapping = load_prompt_probability_mapping(source)
    missing = [prompt for prompt in prompt_order if prompt not in mapping]
    if missing:
        preview = missing[:3]
        raise KeyError(
            f"Probability artifact is missing {len(missing)} prompts. "
            f"Examples: {preview!r}"
        )

    raw_probabilities = np.asarray([mapping[prompt] for prompt in prompt_order], dtype=np.float64)
    raw_probabilities = np.clip(raw_probabilities, 0.0, 1.0)

    label_array = None
    if labels is not None:
        label_array = np.asarray(labels, dtype=np.int64)
        if label_array.shape != raw_probabilities.shape:
            raise ValueError("labels must be aligned with prompt_order")

    calibrated = calibrate_probabilities(raw_probabilities, label_array, calibration)

    metadata: Dict[str, Any] = {
        "source": str(Path(source)),
        "prompt_count": int(len(prompt_order)),
        "calibration": calibration,
        "raw": compute_probability_diagnostics(raw_probabilities, label_array),
        "effective": compute_probability_diagnostics(calibrated, label_array),
    }
    return calibrated.astype(np.float32), metadata


def calibrate_probabilities(
    probabilities: np.ndarray,
    labels: Optional[np.ndarray],
    calibration: str,
) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if calibration == "none":
        return clipped
    if labels is None:
        raise ValueError(f"Calibration mode {calibration!r} requires binary labels.")

    if calibration == "platt":
        from sklearn.linear_model import LogisticRegression

        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        clf = LogisticRegression(solver="lbfgs", max_iter=1000)
        clf.fit(logits, labels)
        return clf.predict_proba(logits)[:, 1]

    if calibration == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        ir = IsotonicRegression(out_of_bounds="clip")
        return np.asarray(ir.fit_transform(clipped, labels), dtype=np.float64)

    raise AssertionError(f"Unexpected calibration mode {calibration!r}")


def compute_probability_diagnostics(
    probabilities: np.ndarray,
    labels: Optional[np.ndarray],
    n_bins: int = 10,
) -> Dict[str, Any]:
    probs = np.asarray(probabilities, dtype=np.float64)
    payload: Dict[str, Any] = {
        "min": float(probs.min()) if probs.size else None,
        "max": float(probs.max()) if probs.size else None,
        "mean": float(probs.mean()) if probs.size else None,
        "std": float(probs.std()) if probs.size else None,
        "positive_mass": float(probs.sum()),
        "negative_mass": float((1.0 - probs).sum()),
    }

    if labels is None or probs.size == 0:
        return payload

    labels = np.asarray(labels, dtype=np.float64)
    payload["brier"] = float(np.mean((probs - labels) ** 2))
    payload["ece"] = float(expected_calibration_error(probs, labels, n_bins=n_bins))
    if labels.size:
        pos_mask = labels >= 0.5
        neg_mask = ~pos_mask
        payload["mean_positive_probability"] = (
            float(probs[pos_mask].mean()) if pos_mask.any() else None
        )
        payload["mean_negative_probability"] = (
            float(probs[neg_mask].mean()) if neg_mask.any() else None
        )
    return payload


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    labs = np.asarray(labels, dtype=np.float64)
    if probs.shape != labs.shape:
        raise ValueError("probabilities and labels must have the same shape")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = float(len(probs))
    for i in range(n_bins):
        left = bins[i]
        right = bins[i + 1]
        if i == n_bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not mask.any():
            continue
        bin_probs = probs[mask]
        bin_labels = labs[mask]
        ece += abs(bin_probs.mean() - bin_labels.mean()) * (len(bin_probs) / total)
    return float(ece)


def _json_payload_to_mapping(payload: Any) -> Dict[str, float]:
    if isinstance(payload, Mapping):
        if "records" in payload and isinstance(payload["records"], list):
            return _records_to_mapping(payload["records"])
        if "items" in payload and isinstance(payload["items"], list):
            return _records_to_mapping(payload["items"])
        if "prompts" in payload and "probabilities" in payload:
            prompts = payload["prompts"]
            probabilities = payload["probabilities"]
            if not isinstance(prompts, list) or not isinstance(probabilities, list):
                raise ValueError("JSON prompts/probabilities payload must contain lists.")
            if len(prompts) != len(probabilities):
                raise ValueError("JSON prompts and probabilities lists must be the same length.")
            return {
                str(prompt): _coerce_probability_value(probability)
                for prompt, probability in zip(prompts, probabilities)
            }
        if payload and all(not isinstance(v, (dict, list)) for v in payload.values()):
            return {str(prompt): _coerce_probability_value(value) for prompt, value in payload.items()}
    if isinstance(payload, list):
        return _records_to_mapping(payload)
    raise ValueError("Unsupported JSON probability artifact structure.")


def _records_to_mapping(records: Iterable[Any]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("Probability records must be objects with prompt and probability fields.")
        prompt_field = next((field for field in _PROMPT_FIELD_CANDIDATES if field in raw), None)
        prob_field = next((field for field in _PROBABILITY_FIELD_CANDIDATES if field in raw), None)
        if prompt_field is None or prob_field is None:
            raise ValueError(
                "Probability records must contain one prompt field from "
                f"{_PROMPT_FIELD_CANDIDATES} and one probability field from "
                f"{_PROBABILITY_FIELD_CANDIDATES}."
            )
        prompt = str(raw[prompt_field])
        probability = _coerce_probability_value(raw[prob_field])
        previous = mapping.get(prompt)
        if previous is not None and abs(previous - probability) > 1e-9:
            raise ValueError(
                f"Duplicate prompt {prompt!r} has conflicting probabilities "
                f"{previous} vs {probability}."
            )
        mapping[prompt] = probability
    return mapping


def _coerce_probability_value(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith("%"):
            return float(stripped[:-1]) / 100.0
        return float(stripped)
    raise ValueError(f"Unsupported probability value: {value!r}")
