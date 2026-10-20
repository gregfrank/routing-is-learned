# Detection Is Cheap, Routing Is Learned: Why Refusal-Based Alignment Evaluation Fails

**Paper:** [arXiv:2603.18280](https://arxiv.org/abs/2603.18280)

The abstract below matches the current arXiv version.

## Abstract

Current alignment evaluation mostly measures whether models encode dangerous concepts and whether they refuse harmful requests. Both miss the layer where alignment often operates: routing from concept detection to behavioral policy. We study political censorship in Chinese-origin language models as a natural experiment, using probes, surgical ablations, and behavioral tests across nine open-weight models from five labs. Three findings follow. First, probe accuracy alone is non-diagnostic: political probes, null controls, and permutation baselines can all reach 100%, so held-out category generalization is the informative test. Second, surgical ablation reveals lab-specific routing. Removing the political-sensitivity direction eliminates censorship and restores accurate factual output in most models tested, while one model confabulates because its architecture entangles factual knowledge with the censorship mechanism. Cross-model transfer fails, indicating that routing geometry is model- and lab-specific. Third, refusal is no longer the dominant censorship mechanism. Within one model family, hard refusal falls to zero while narrative steering rises to the maximum, making censorship invisible to refusal-only benchmarks. These results support a three-stage descriptive framework: detect, route, generate. Models often retain the relevant knowledge; alignment changes how that knowledge is expressed. Evaluations that audit only detection or refusal therefore miss the routing mechanism that most directly determines behavior.

## Quick Start

```bash
# Clone and set up environment
git clone https://github.com/gregfrank/routing-is-learned.git
cd routing-is-learned
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the political probe on Qwen3-8B (requires GPU with ≥16GB VRAM)
cd src
python political_probe.py --model Qwen/Qwen3-8B

# Run the multi-layer ridge sweep
python multi_layer_ridge_sweep.py

# Run the ablation experiment (extracts and tests refusal direction)
python ablation_ridge_regression.py
```

## Repository Structure

```
routing-is-learned/
├── README.md
├── REPRODUCIBILITY.md          # Step-by-step reproduction guide
├── LICENSE                     # MIT
├── requirements.txt
│
├── src/                        # Core code
│   ├── political_probe.py      # Main probe script (phases 1-4: extraction, probing, PCA, ablation)
│   ├── ablation_ridge_regression.py  # Ridge-cleaned refusal direction extraction + ablation
│   ├── direction_methods.py    # Shared direction computation (CAA, ridge cleaning)
│   ├── multi_layer_ridge_sweep.py    # Multi-layer sweep with KL divergence measurement
│   ├── probe_cross_validation.py     # Category-held-out cross-validation
│   ├── null_probe_control.py         # Null probe control (non-political topic pairs)
│   ├── parallel_behavioral_test.py   # CCP vs parallel + Chinese behavioral test
│   ├── ridge_scale_sweep.py          # Ridge ablation alpha sweep
│   ├── political_ridge_ablation.py   # Political-direction ridge ablation
│   ├── control_direction_prompts.py  # Control direction corpora (sentiment, formality)
│   ├── political_prompts_v1.py       # Political corpus v1 (24 pairs)
│   ├── political_prompts_v2.py       # Political corpus v2 (120 pairs, 15 categories)
│   ├── political_prompts_adversarial.py  # Adversarial corpus (32 CCP + 32 parallel pairs)
│   └── safety_prompts_v3.py          # Safety corpus (120 harmful + 120 benign pairs)
│
└── results/                    # Key experimental results
    ├── cross_model_political_probe_summary.md  # Summary across 9 models (+ Yi exploratory)
    ├── cross_validation_results.json           # Leave-one-category-out CV
    ├── null_probe_results.json                 # Null probe control experiment
    ├── parallel_behavioral_results.json        # CCP vs parallel refusal rates
    ├── propaganda_scores.json         # Gemini judge propaganda scores (external, not locally reproducible)
    ├── minicpm_sglang_behavioral.json # MiniCPM behavioral data via SGLang (external)
    ├── probes/                 # Selected per-model probe run logs
    ├── ridge_sweep/            # Multi-layer ridge sweep data
    ├── ridge_scale_sweep/      # Ridge ablation alpha sweep data
    └── ablation/               # Ablation configuration and vectors (Qwen3-8B)
```

## Models Probed

Nine models are included in the final paper. Yi-1.5-9B was probed during development but dropped due to inconsistent results.

| Model | Lab | Origin | Probe Accuracy | Key Finding |
|-------|-----|--------|---------------|-------------|
| Qwen3-8B | Alibaba | Chinese | >99% | Strongest political routing signal |
| Qwen2.5-7B | Alibaba | Chinese | >99% | Consistent with Qwen3 |
| Qwen3.5-4B | Alibaba | Chinese | >98% | Political routing in smaller Qwen |
| Qwen3.5-9B | Alibaba | Chinese | >99% | Political routing persists at 9B |
| DeepSeek-R1-Distill-7B | DeepSeek | Chinese | >97% | Political sensitivity in reasoning model |
| GLM-Z1-9B | Zhipu | Chinese | >95% | No political refusal, but direction still detectable |
| GLM-4-9B | Zhipu | Chinese | >95% | Consistent with GLM-Z1 |
| Phi-4-mini | Microsoft | Western | >95% | Safety-only; no political sensitivity |
| Llama-3.2-3B | Meta | Western | >90% | Weakest signal (smallest model) |

## Related Work

The companion paper ["How Alignment Routes: Localizing, Scaling, and Controlling Policy Circuits in Language Models"](https://arxiv.org/abs/2604.04385) (code: [how-alignment-routes](https://github.com/gregfrank/how-alignment-routes)) extends this work with four-method circuit analyses in three models and operational routing-head screens in nine more (2B–72B).

## Citation

```bibtex
@article{frank2026detectionischeap,
  title={Detection Is Cheap, Routing Is Learned: Why Refusal-Based Alignment Evaluation Fails},
  author={Frank, Gregory N.},
  journal={arXiv preprint arXiv:2603.18280},
  year={2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
