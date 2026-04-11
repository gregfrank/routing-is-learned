# Reproducibility Guide

Step-by-step instructions for reproducing all experiments in "Detection Is Cheap, Routing Is Learned: Why Refusal-Based Alignment Evaluation Fails" (arXiv:2603.18280).

## Hardware Requirements

| Experiment | GPU VRAM | Approx. Time |
|-----------|----------|------------|
| Political probe (single model) | 16 GB | 15–30 min |
| Multi-layer ridge sweep | 16 GB | 1–2 hours |
| Ablation ridge regression | 16 GB | 30–60 min |
| Full 9-model probe sweep | 16 GB | 4–8 hours total |

**Total compute for full replication:** ~10–15 GPU-hours on a single A5000/RTX 4090.

## Software Requirements

```bash
Python 3.10+
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Tested versions:** torch 2.4+, transformers 4.45+.

## Authentication

Some models are gated and require a Hugging Face token:

```bash
export HF_TOKEN=<your_huggingface_token>
# Or: huggingface-cli login
```

## Environment Variables

The ablation script is configurable via environment variables:

```bash
export ABLATION_MODEL_ID="Qwen/Qwen3-8B"   # Target model
export ABLATION_SEED=42                      # Random seed
export ABLATION_LAYER_INDICES="9 18 30"      # Layers to probe
export ABLATION_NUM_PAIRS=120                # Number of prompt pairs
export ABLATION_ENABLE_THINKING=false        # Disable thinking mode (Qwen3)
```

## Experiment Reproduction

All commands should be run from the `src/` directory:

```bash
cd src
```

### Experiment 1: Political Sensitivity Probe

The main probe script runs all four phases: direction extraction, layer-by-layer probing, PCA decomposition, and targeted ablation.

```bash
# Qwen3-8B (primary model)
python political_probe.py --model Qwen/Qwen3-8B

# With expanded v2 corpus (120 pairs)
python political_probe.py --model Qwen/Qwen3-8B --corpus v2

# With adversarial corpus
python political_probe.py --model Qwen/Qwen3-8B --corpus adversarial
```

**Expected output:** 
- `runs/political_probe/probe_data.pt` — cached activations and probe results
- `runs/political_probe/plots/` — layer accuracy plots, PCA visualizations

**Key metrics to verify:**
- Ridge probe accuracy > 95% at best layer
- Political direction geometrically distinct from generic refusal direction

### Experiment 2: Cross-Model Sweep

Run the probe across all models:

```bash
for model in "Qwen/Qwen3-8B" "Qwen/Qwen2.5-7B-Instruct" \
             "Qwen/Qwen3.5-4B" "Qwen/Qwen3.5-9B" \
             "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
             "zai-org/GLM-Z1-9B-0414" "zai-org/GLM-4-9B-0414" \
             "microsoft/Phi-4-mini-instruct" \
             "meta-llama/Llama-3.2-3B"; do
    python political_probe.py --model "$model"
done
```

**Note:** GLM models use the `zai-org/` hub prefix (not `THUDM/`). Llama-3.2-3B is the base model, not the Instruct variant.

**Expected:** Near-perfect probe accuracy across all Chinese-origin models; weaker but detectable signal in Western models.

### Experiment 3: Multi-Layer Ridge Sweep

Tests direction quality across all layers with KL divergence measurement:

```bash
python multi_layer_ridge_sweep.py
```

**Expected output:** `runs/multi_layer_ridge_sweep/` with per-layer KL, cosine similarity, and refusal rate data.

### Experiment 4: Ablation Ridge Regression

Extracts the refusal direction, cleans it via ridge regression, and tests behavioral ablation:

```bash
python ablation_ridge_regression.py
```

**Expected output:** `runs/qwen3_8b_ablation/` with:
- `checkpoint.pt` — direction checkpoint
- `plots/` — ablation effect visualizations
- `ablation.log` — full log

### Experiment 5: Direction Methods Comparison

The `direction_methods.py` module supports CAA and ridge-cleaned CAA directions:

```bash
python political_probe.py --model Qwen/Qwen3-8B --direction-method caa
python political_probe.py --model Qwen/Qwen3-8B --direction-method caa_ridge
```

### Note on Ablation Pair Count

The shipped `results/ablation/config.json` records 120 pairs sourced from `curated_local+harmbench`. To reproduce this exactly:

```bash
pip install datasets
ABLATION_HARMBENCH_ENABLED=true python ablation_ridge_regression.py
```

Without `datasets` installed, the script uses only the 80 built-in curated pairs.

## Verifying Results

### Probe Accuracy
Each probe run produces a layer-accuracy plot and `probe_data.pt`. The best layer should show >95% accuracy for Chinese-origin models. To capture a log, redirect output:

```bash
python political_probe.py --model Qwen/Qwen3-8B 2>&1 | tee runs/political_probe/run.log
```

### Cross-Validation
Leave-one-category-out CV results should match `results/cross_validation_results.json` within ±2% per fold.

### Null Probe Control
The null probe experiment (`results/null_probe_results.json`) shows that non-political topic pairs (Science/History, Food/Technology, Geography/Music) also achieve 100% train-set accuracy at all layers. This confirms that raw linear separability in high-dimensional hidden spaces is trivially expected and not evidence of special political encoding. The meaningful evidence comes from category-held-out generalization, ablation behavior, and behavioral asymmetry.

### Behavioral Results
CCP-vs-parallel discrimination (`results/parallel_behavioral_results.json`) should show:
- Chinese-origin models: high CCP refusal rate, ~0% parallel refusal
- Western models: low/zero refusal on both

## Known Issues

1. **Qwen3 thinking mode**: Set `ABLATION_ENABLE_THINKING=false` to disable Qwen3's extended thinking, which dramatically slows inference.
2. **MiniCPM**: Custom attention code is incompatible with hidden-state extraction. Probe fails — use behavioral testing via external API instead.
3. **Large models (>14B)**: May require multi-GPU setup or quantization. The probe works with fp16 on a single GPU up to ~14B.
4. **Yi-1.5-9B-Chat**: Inconsistent results across runs. Dropped from the final paper.
