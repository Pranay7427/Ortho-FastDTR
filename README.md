# DTR: Dynamic Token Reweighting for Robust Vision-Language Models

<p align="center">
  <b>CVPR 2026</b>
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2505.17132">[Paper]</a> •
  <a href="https://arxiv.org/abs/2505.17132">[arXiv]</a> •
  <a href="demo+Visualization.ipynb">[Demo Notebook]</a>
</p>

> **Tanqiu Jiang<sup>1</sup>, Jiacheng Liang<sup>1</sup>, Rongyi Zhu<sup>1</sup>, Jiawei Zhou<sup>1</sup>, Fenglong Ma<sup>2</sup>, Ting Wang<sup>1</sup>**
>
> <sup>1</sup>Stony Brook University &nbsp; <sup>2</sup>Pennsylvania State University

---

## Overview

Large vision-language models (VLMs) are highly vulnerable to **multimodal jailbreak attacks** that exploit visual-textual interactions to bypass safety guardrails. **DTR** is a novel **inference-time defense** that mitigates these attacks by optimizing the model's key-value (KV) caches — without requiring safety-specific fine-tuning data or costly image-to-text conversion.

DTR introduces a new formulation of the **safety-relevant distributional shift** induced by the visual modality. It dynamically adjusts visual token weights to redirect harmful queries along safety-enhancing trajectories, effectively counteracting adversarial visual inputs while preserving the model's general capabilities and inference efficiency.

<p align="center">
  <img src="assets/DTR.png" width="85%" alt="DTR Overview"/>
</p>

### Key Advantages

- **Effective** — Eliminates reliance on error-prone safety data curation or image-to-text conversion.
- **Efficient** — Maintains (or even improves) inference efficiency through strategic eviction of less important visual tokens.
- **Interpretable** — Optimized visual-token weights directly reveal their impact on safety-relevant shifts, enabling intuitive investigation of suspicious inputs.

---

## Installation

This repository includes the necessary [LLaVA](https://github.com/haotian-liu/LLaVA) components directly as local source code, so no separate LLaVA installation is required.

```bash
# 1. Clone this repository
git clone https://github.com/TanqiuJiang/DTR.git
cd DTR

# 2. Create and activate a conda environment
conda create -n dtr python=3.10 -y
conda activate dtr

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** If you run into dependency conflicts or compatibility issues, you can refer to the official [LLaVA installation guide](https://github.com/haotian-liu/LLaVA#install) to set up the base LLaVA environment first, then install any remaining dependencies from our `requirements.txt`.

### Hardware Requirements

All experiments in the paper were conducted on an **NVIDIA H100 GPU**. A GPU with at least 24 GB VRAM is recommended for running the LLaVA-Llama2-7B model with DTR.

---

## Quick Start

### Interactive Demo & Visualization

We provide a Jupyter notebook for a quick hands-on demo of DTR, including visualization of the optimized scaling vector α:

```bash
jupyter notebook demo+Visualization.ipynb
```

The notebook walks through:
1. Loading a pretrained VLM (LLaVA-Llama2-7B).
2. Computing the refusal direction from reference prompts.
3. Running DTR's dynamic token reweighting on sample inputs.
4. Visualizing the α heatmaps that highlight adversarial vs. feature tokens.

### Defending LLaVA-Llama2 against HADES

To evaluate DTR on the [HADES](https://huggingface.co/datasets/Monosail/HADES) benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 python HADES_LLaVA_Llama2_with_DTR.py
```

#### Command-Line Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset_path` | `str` | `Monosail/HADES` | Path or HuggingFace ID of the HADES dataset |
| `--model_path` | `str` | `liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview` | Path to the LLaVA model |
| `--output_path` | `str` | `result/defended_llava_llama.json` | Output path for the results JSON |
| `--ensure_accurate_OCR` | `bool` | `True` | Retry generation if the model fails to identify text in the image |
| `--max_attempts` | `int` | `1` | Maximum retry attempts for OCR verification |
| `--step` | `str` | `last` | Evaluate on last-step images (`last`) or all steps (`all`) |
| `--category`, `-c` | `str` | `S+A` | HADES attack category: `S` (Stable Diffusion), `S+A` (SD + Adversarial Perturbation), or `S+T+A` (SD + Typography + AP) |

#### Example: Evaluate on the S+T+A Attack

```bash
CUDA_VISIBLE_DEVICES=0 python HADES_LLaVA_Llama2_with_DTR.py \
    --category S+T+A \
    --step last \
    --output_path result/defended_llava_llama_S+T+A.json
```

---

## Method

DTR operates in three stages:

1. **Pre-compute the refusal direction** — Estimate the safety-relevant direction `d_ref` from a small set of contrastive prompts (32 harmful from [AdvBench](https://github.com/llm-attacks/llm-attacks) + 32 harmless from [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)). This is computed once and cached.

2. **Optimize the scaling vector α** — For each incoming query, DTR optimizes a per-token scaling vector α ∈ [0,1]ⁿ over the visual tokens by minimizing:

   ```
   L(α) = f(x(α)) · d_ref / ‖d_ref‖  +  λ ‖f(x) − f(x(α))‖²
   ```

   The first term reverses the safety-relevant shift; the second preserves the model's latent representations for benign queries.

3. **Reweight and (optionally) evict tokens** — Apply the optimized α to visual tokens in the KV cache. Tokens with α below a threshold β can be evicted entirely to improve inference speed.

### Default Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `n_ref` | 32 | Number of reference samples per set |
| `λ` | 0.1 | Balance between safety shift and utility preservation |
| `m` | 4 | Optimization steps for α |
| `η` | 0.01 | Learning rate (AdamW) |
| `β` | — | Token eviction threshold (optional; 20% eviction rate recommended) |


## Project Structure

```
DTR/
├── HADES_LLaVA_Llama2_with_DTR.py   # Main evaluation script
├── DTR_util/
│   └── util.py                       # Core DTR utilities (refusal direction, optimization, hooks)
├── demo+Visualization.ipynb          # Interactive demo and α visualization
├── result/                           # Output directory for evaluation results
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{Jiang_2026_CVPR,
    author    = {Jiang, Tanqiu and Liang, Jiacheng and Zhu, Rongyi and Zhou, Jiawei and Ma, Fenglong and Wang, Ting},
    title     = {Dynamic Token Reweighting for Robust Vision-Language Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {24481-24491}
}
```

---

## Acknowledgements

This work was supported by the National Science Foundation under Grant No. 2405136 and 2406572.

---

## License

This project is released under the [MIT License](LICENSE).