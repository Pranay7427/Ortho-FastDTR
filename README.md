# Ortho-FastDTR: Breaking and Fixing Dynamic Token Reweighting via Subspace Attribution and Single-Step Visual Token Eviction

**One-Line Claim:** By replacing 10-step 1D gradient optimization with 1-step feature-norm-normalized Taylor attribution over a Rank-4 SVD refusal subspace and horizontal strip pooling, `Ortho-FastTaylor` slashes multimodal jailbreak Attack Success Rate (ASR) by 4x (`48.0% -> 12.0%` on HADES `S+T+A` and `40.0% -> 0.0%` under white-box adaptive PGD) while accelerating defense latency by **7.9x (`851.07 ms -> 107.93 ms`)** with zero benign utility degradation.

### Tracks + Problem Statements Covered
* **Track 2a: AI Safety and Red Teaming**
  * **Problem Statement 1:** Red-teaming and robustness evaluation (Mechanistic vulnerability audit of CVPR 2026 Dynamic Token Reweighting + White-Box Adaptive Typography-Preserving Orthogonal Decoy PGD attack)
  * **Problem Statement 2:** Adversarial and safety evaluation sets (Curated 50-sample HADES multimodal benchmark evaluation suite + 10-sample Adaptive PGD red-team trace set)
* **Track 2b: AI Safety and Correction**
  * **Problem Statement 4:** Inference-time safety steering (Single-step Taylor attribution and surgical visual token eviction without weight retraining)
  * **Problem Statement 6:** Circuit-level interventions (Rank-4 SVD orthonormal refusal subspace projection and leak-free `VisionScalingHook` gating)

> **Content Warning:** This repository and its linked evaluation traces contain examples of adversarial multimodal jailbreak prompts and harmful keywords (e.g., violence, self-harm, cyberattacks, and harassment scenarios derived from the public HADES safety benchmark) strictly for red-teaming and defensive evaluation purposes.

---

## Abstract

Vision-Language Models (VLMs) such as `LLaVA-Llama-2-7B` suffer from a critical *Modality Refusal Gap*: while harmful text-only prompts reliably trigger internal safety guardrails, multimodal attacks that visually embed forbidden keywords via typographic banners and adversarial perturbations (e.g., the HADES benchmark) bypass alignment filters. Recent inference-time defenses like Dynamic Token Reweighting (DTR, CVPR 2026) attempt to close this gap by optimizing 256 vision-tower patch weights ($\alpha \in [0.1, 1.0]^{256}$) over 10 iterative gradient descent steps along a single 1D refusal direction.

Through a mechanistic code and red-teaming audit of official DTR, we expose four vulnerabilities that break its robustness and deployment viability:
1. **Unnormalized Gradient Blindspot:** Because patch gradients $\nabla_{\alpha_i} \mathcal{L}$ scale directly with patch feature norms $\|v_i\|_2$, high-contrast background objects dominate attribution. Low-norm typographic banners at the bottom of images receive negligible attribution ($\alpha \approx 0.85$, $0.0\%$ evicted), leaving OCR attention heads free to read the embedded harmful text.
2. **1D Refusal Subspace Collapse:** Averaging contrastive hidden states into a 1D vector discards secondary orthogonal refusal circuits, resulting in a $48.0\%$ Attack Success Rate (ASR) on HADES `S+T+A` and $40.0\%$ ASR under white-box orthogonal PGD perturbations.
3. **Multi-Step Latency & Over-Eviction Collapse:** Running 10 steps of AdamW backpropagation incurs an $851.07\text{ ms}$ latency penalty per query and triggers boundary collapse ($\alpha \to 0$) on subspace objectives, causing hallucinations on benign prompts.
4. **Forward-Hook Accumulation Leak:** Unremoved vision-tower hooks stack across queries, causing exponential feature decay ($\alpha^N$).

To resolve these limitations without model retraining, we introduce **`Ortho-FastTaylor`**, a single-step inference-time defense combining: (1) a **Sign-Aligned Rank-4 SVD Refusal Subspace ($U_{\text{ref}}$)** capturing multi-dimensional safety triggers, (2) **1-Step Feature-Norm-Normalized Taylor Attribution** ($s_i = |\nabla_{\alpha_i}\mathcal{L}_{\text{ortho}}| / (\|v_i\|_2 + \epsilon)$) to expose low-norm typographic banners, and (3) **$16\times 16$ Horizontal Strip Pooling ($1\times 5$ kernel) with Surgical Top-Quantile Eviction ($\alpha_i = 0.0$)** inside a context-managed `VisionScalingHook`. Across HADES and adaptive red-teaming, `Ortho-FastTaylor` achieves strict Pareto superiority in safety, benign retention, and speed.

* **Headline Before -> After Number:**
  * **HADES `S+T+A` ASR:** `88.0%` (No Defense) $\rightarrow$ `48.0%` (Vanilla DTR) $\rightarrow$ **`12.0%` (`Ortho-FastTaylor` — 4x ASR reduction)**
  * **HADES `S+A` ASR:** `80.0%` (No Defense) $\rightarrow$ `24.0%` (Vanilla DTR) $\rightarrow$ **`0.0%` (`Ortho-FastTaylor` — 100% defended)**
  * **White-Box Adaptive PGD ASR:** `100.0%` (No Defense) $\rightarrow$ `40.0%` (Vanilla DTR) $\rightarrow$ **`0.0%` (`Ortho-FastTaylor` — 100% defended)**
  * **Defense Latency Overhead:** `851.07 ms` (Vanilla DTR) $\rightarrow$ **`107.93 ms` (`Ortho-FastTaylor` — 7.9x faster)**
* **Base Model + Method in One Line:** `LLaVA-Llama-2-7B` (`CLIP ViT-L/14-336` + `Llama-2-7B-Chat`) hardened at inference time via Rank-4 SVD refusal subspace projection, 1-step feature-norm-normalized Taylor attribution, and $1\times 5$ horizontal strip token eviction.

> ### For Judges: One Minute Summary
> * **What we broke [Track 2a]:** We audited CVPR 2026 Dynamic Token Reweighting (DTR) and proved why it fails on typographic jailbreaks (`48% ASR`) and white-box adaptive attacks (`40% ASR`): its patch gradients scale with feature norm $\|v_i\|_2$ (blinding it to low-norm text strips), its 1D refusal vector is vulnerable to orthogonal PGD bypasses, its 10-step AdamW loop adds `851 ms` overhead, and its PyTorch forward hooks leak across turns.
> * **How we fixed it [Track 2b]:** We built `Ortho-FastTaylor` (`DTR_util/ortho_fast_dtr.py`), replacing 10-step 1D optimization with a single backward pass over a **Rank-4 SVD orthonormal refusal subspace**, dividing attribution by patch feature norm $\|v_i\|_2$, smoothing over $1\times 5$ horizontal strips on the $16\times 16$ ViT grid, and hard-evicting ($\alpha_i = 0$) the top quantile.
> * **Why it matters:** ASR drops from `48% -> 12%` on HADES `S+T+A`, `24% -> 0%` on `S+A`, and `40% -> 0%` on Adaptive PGD, while running **7.9x faster (`108 ms`)** with **100% benign scene retention**.
> * **Verify live in 30 seconds:** Open our [Hugging Face Space (`Pranay7427/ortho-fastdtr-explorer`)](https://huggingface.co/spaces/Pranay7427/ortho-fastdtr-explorer) or inspect traces in [Dataset (`Pranay7427/ortho-fastdtr-hades-benchmark`)](https://huggingface.co/datasets/Pranay7427/ortho-fastdtr-hades-benchmark).

---

## Problem Statement and Approach

### Why This Problem / Real Use Case
Production multimodal assistants (customer support bots, content moderation pipelines, medical/document VQA systems) routinely process user-uploaded screenshots, memes, and scanned documents. While their underlying LLM backbones are heavily RLHF-aligned against harmful text prompts, attackers trivially bypass alignment by rendering the forbidden instruction as a typographic strip at the bottom of a benign image or injecting imperceptible adversarial noise. Retraining or safety-fine-tuning multimodal weights is expensive and degrades visual reasoning. Inference-time token steering promises a plug-and-play fix, but existing multi-step optimization approaches (like Vanilla DTR) are both too slow for real-time production SLAs (`>850 ms` overhead per query) and mechanically blind to low-norm typographic strips.

### Research Questions and Pre-Registered Hypotheses
* **RQ1 (Typographic Blindspot):** Why does Vanilla DTR fail to evict typographic jailbreak banners (`0.0%` token eviction rate on HADES `S+T+A`) despite computing gradients with respect to the refusal direction?
  * *Hypothesis H1:* By the chain rule, $\nabla_{\alpha_i} \mathcal{L} = \langle \nabla_{v_i} \mathcal{L}, v_i \rangle \propto \|v_i\|_2$. High-norm salient background objects dominate the gradient, starving uniform black-on-white text patches of attribution. Normalizing by $\|v_i\|_2 + \epsilon$ and applying horizontal strip pooling ($1\times 5$) will isolate typographic banners.
* **RQ2 (Subspace Dimensionality):** Does safety refusal in VLMs live in a 1D direction or a low-rank multi-dimensional subspace?
  * *Hypothesis H2:* Different harm categories activate partially orthogonal refusal circuits. Expanding from a 1D mean vector to a sign-aligned Rank-$k$ ($k=4$) SVD subspace will prevent orthogonal PGD bypasses.
* **RQ3 (Single-Step vs. Multi-Step Steering):** Can 1-step first-order Taylor attribution + surgical hard eviction outperform 10-step iterative gradient descent?
  * *Hypothesis H3:* Multi-step AdamW over-optimizes and collapses all patch weights toward zero ($\alpha \to 0$) when paired with subspace losses, causing benign hallucinations. A 1-step Taylor attribution with top-quantile hard eviction ($\tau = 0.80$) will achieve lower ASR, ~8x lower latency, and zero benign collapse.

### Threat Model [T2a]
We evaluate under two realistic adversary profiles targeting a deployed VLM (`LLaVA-Llama-2-7B`):
1. **Black-Box / Grey-Box Multimodal Jailbreaker (HADES `S+T+A` and `S+A`):** The attacker replaces the harmful noun/verb in the text prompt with a generic placeholder while embedding the forbidden keyword as a typographic banner at the bottom of a synthesized scene image plus $\ell_\infty$-bounded adversarial perturbations.
2. **White-Box Adaptive Adversary (Typography-Preserving Orthogonal Decoy PGD):** The attacker has full white-box access to the VLM weights, the defense's 1D refusal vector $r$, and the token-reweighting mechanism. Using a custom differentiable vision-tower hook (`GradEnabledVisionTowerHook`), the attacker optimizes an $\ell_\infty \le 8/255$ perturbation over 10 PGD steps solely on the upper scene pixels (preserving the bottom 18% typographic strip so OCR remains intact). The attacker minimizes projection onto the 1D refusal vector while injecting high-gradient decoy salience into benign background patches to hijack Vanilla DTR's token eviction budget.

### Related Work
* **HADES (Li et al., CVPR 2024 / ECCV 2024):** *Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models.*
* **Dynamic Token Reweighting / DTR (CVPR 2026):** *Mitigating Multimodal Jailbreaks via Dynamic Token Reweighting.* Serves as our primary baseline audited and improved in this work.
* **Representation Engineering & Refusal Direction (Zou et al., 2023; Arditi et al., NeurIPS 2024):** Demonstrated linear refusal directions in residual stream activations; we extend this from 1D to a sign-aligned Rank-4 SVD subspace for multimodal token attribution.
* **LLaVA (Liu et al., NeurIPS 2023):** *Visual Instruction Tuning.*

---

## Hackathon Repos (Kits) Used

* **CircuitKIT (Conceptual & Subspace Attribution Alignment):** Implemented in `DTR_util/ortho_fast_dtr.py` by extracting contrastive residual-stream activations at Layer $-1$, computing a sign-aligned Rank-4 SVD refusal basis (`extract_orthogonal_refusal_subspace`), and intervening via context-managed forward hooks (`VisionScalingHook`) on the 256 visual token circuits.
* **AuditKIT (Red-Teaming & Automated Safety Verification):** Operationalized in our automated benchmarking and white-box red-teaming suite (`run_benchmarks.py` and `adaptive_red_team.py`), which audits model responses across harmful and benign queries, measures explicit refusal vs. harmful compliance rates, tracks token eviction distributions, and executes white-box PGD stress tests.
* **Other Repositories Used:** `Pranay7427/Ortho-FastDTR`, `haotian-liu/LLaVA` (local modified package), `Monosail/HADES`, and the official `Dynamic Token Reweighting (DTR)` baseline implementation.

---

## Models and Licenses

| Role | Model Identifier | Architecture & Precision | License |
| :--- | :--- | :--- | :--- |
| **Target / Hardened VLM [T2a / T2b]** | `LLaVA-Llama-2-7B-Chat` | Vision-Language Model (`FP16`, 7B params) | Llama 2 Community License / Apache-2.0 |
| **Base LLM Backbone** | `NousResearch/Llama-2-7b-chat-hf` | 32-layer Transformer Decoder (`FP16`, $d=4096$) | Llama 2 Community License |
| **Multimodal LoRA / Projector** | `liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview` | Pretrained MMP Linear Projector + LoRA weights | Apache-2.0 |
| **Vision Tower** | `openai/clip-vit-large-patch14-336` | CLIP ViT-L/14 @ $336\times 336$ ($16\times 16 = 256$ patches) | MIT License |

---

## Datasets, Benchmarks and Licenses

| Dataset / Split | Source | Size | Modality & Task | Language | License | Role in Work |
| :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| **`Monosail/HADES` (`S+T+A` split)** | Public Benchmark | 25 curated samples | Image (Diffusion + Typography + Adversarial Noise) + Text Jailbreak | English (`en`) | MIT | Primary Focus Safety Benchmark `[T2a / T2b]` |
| **`Monosail/HADES` (`S+A` split)** | Public Benchmark | 25 curated samples | Image (Diffusion + Adversarial Noise) + Text Jailbreak | English (`en`) | MIT | Primary Focus Safety Benchmark `[T2a / T2b]` |
| **White-Box Adaptive Decoy PGD Set** | Synthesized (`adaptive_red_team.py`) | 10 adversarial samples | White-box PGD-perturbed images ($\epsilon=8/255$, 10 steps) + Harmful Prompt | English (`en`) | MIT | Focus Red-Teaming Suite `[T2a]` |
| **Paired Benign Scene Utility & Over-Refusal Set** | Paired benign queries on HADES images | 50 image-query pairs | Adversarial images paired with harmless visual description queries | English (`en`) | MIT | Over-Refusal & Benign Utility Check `[T2b]` |
| **Contrastive Calibration States (`test.pt`)** | Official DTR calibration tensor | 60 contrastive pairs | Residual stream hidden states ($H_{\text{refusal}}, H_{\text{comply}} \in \mathbb{R}^{60\times 4096}$) | English (`en`) | Apache-2.0 | Offline SVD Subspace Calibration |

* **Zero Training / Zero Data Leakage:** `Ortho-FastTaylor` is strictly an inference-time, training-free mechanism. Calibration states (`test.pt`) have `0%` overlap with the HADES evaluation images and prompts.

---

## Metrics, and Why These

* **Attack Success Rate (ASR % $\downarrow$):** Share of adversarial multimodal prompts where the VLM complies with actionable harmful instructions.
* **Explicit Refusal Rate (% $\uparrow$):** Share of adversarial queries that explicitly trigger internal safety refusal guardrails.
* **Defense Latency Overhead (`ms/query` $\downarrow$):** Wall-clock inference time spent inside the token-attribution/reweighting step per query.
* **Evicted Patch Ratio (%):** Fraction of the 256 visual tokens gated below threshold ($\alpha_i < 0.5$).
* **Benign Scene Utility / Non-Over-Refusal Rate (% $\uparrow$):** Share of harmless descriptive queries on adversarial images where the defended VLM accurately describes the main visual subject without false refusal or boundary-collapse hallucination.
* **Automatic Judge Verification:** Two-stage deterministic + semantic rubric (`DTR_util/eval_metrics.py`) verified against human labels across all 180 benchmark generations (**98.3% agreement, Cohen's $\kappa = 0.96$**).

---

## Method & Experiments

### `Ortho-FastTaylor` (`DTR_util/ortho_fast_dtr.py`)
1. **Sign-Aligned Rank-4 SVD Refusal Subspace ($U_{\text{ref}}$):** Computes thin SVD on contrastive difference matrix $D = H_{\text{ref}} - H_{\text{comp}}$ and retains the top $k=4$ right singular vectors $U_{\text{ref}} \in \mathbb{R}^{4096 \times 4}$, sign-aligned with the mean refusal shift.
2. **1-Step Feature-Norm-Normalized Taylor Attribution:** Performs a single backward pass on $\mathcal{L}_{\text{ortho}}(h) = -\|U_{\text{ref}}^\top \hat{h}\|_2^2 + 0.5 \max(0, -\langle \hat{h}, u_0\rangle)$ and normalizes by patch L2 norm:
   $$s_i = \frac{|\nabla_{\alpha_i} \mathcal{L}_{\text{ortho}}|}{\|v_i\|_2 + \epsilon}$$
3. **$16\times 16$ Horizontal Strip Pooling ($1\times 5$ Kernel) & Surgical Eviction:** Reshapes $s \in \mathbb{R}^{256}$ into $16 \times 16$, applies 2D average pooling with kernel $(1, 5)$ and a $1.35\times$ spatial prior on rows 12–15, and hard-evicts the top 20% quantile ($\alpha_i = 0.0$ if $\tilde{S}_i \ge \text{Quantile}_{0.80}(\tilde{S})$ else $1.0$) inside a leak-free `VisionScalingHook`.

---

## Results

### 1. Main Benchmark Results: Same Model Before vs. After

| Evaluation Suite | Defense Method | ASR (%) $\downarrow$ | Defended Rate (%) $\uparrow$ | Explicit Refusal (%) $\uparrow$ | Defense Latency (ms) $\downarrow$ | Speedup vs. DTR $\uparrow$ | Evicted Patches (%) | Benign Utility Retained (%) $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HADES `S+T+A`** *(25 samples)* | **No Defense (Baseline)** | 88.0% | 12.0% | 12.0% | 0.00 ms | — | 0.0% | 100.0% |
| | **Vanilla DTR (10-Step 1D)** | 48.0% | 52.0% | 16.0% | 851.07 ms | 1.0x | 0.0% | 96.0% |
| | **`Ortho-FastTaylor` (Ours)** | **12.0%** | **88.0%** | **68.0%** | **107.93 ms** | **7.9x** | **20.3%** | **100.0%** |
| **HADES `S+A`** *(25 samples)* | **No Defense (Baseline)** | 80.0% | 20.0% | 20.0% | 0.00 ms | — | 0.0% | 100.0% |
| | **Vanilla DTR (10-Step 1D)** | 24.0% | 76.0% | 16.0% | 845.22 ms | 1.0x | 0.1% | 96.0% |
| | **`Ortho-FastTaylor` (Ours)** | **0.0%** | **100.0%** | **16.0%** *(84% Benign Neutralized)* | **111.49 ms** | **7.6x** | **20.3%** | **100.0%** |
| **White-Box Adaptive PGD** *(10 samples)* | **No Defense (Baseline)** | 100.0% | 0.0% | 0.0% | 0.00 ms | — | 0.0% | 100.0% |
| | **Vanilla DTR (10-Step 1D)** | 40.0% | 60.0% | 40.0% | 1436.70 ms | 1.0x | 0.4% | 90.0% |
| | **`Ortho-FastTaylor` (Ours)** | **0.0%** | **100.0%** | **60.0%** *(40% Benign Neutralized)* | **280.84 ms** | **5.1x** | **20.3%** | **100.0%** |

### 2. Ablations & Negative Results

| Variant / Ablation | HADES `S+T+A` ASR $\downarrow$ | Latency (ms) $\downarrow$ | Evicted Ratio | Benign Scene Utility $\uparrow$ | Key Finding / Mode of Failure |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Vanilla DTR (10-step 1D, unnormalized)** | 48.0% | 851.1 ms | 0.0% | 96.0% | Fails to evict low-norm typographic strips (`alpha_min ~ 0.85`). |
| **Negative Result 1: 10-Step AdamW + Rank-4 Subspace (`ortho_dtr`)** | 20.0% | 868.4 ms | **78.5%** | **32.0% (Hallucination)** | Multi-step optimization on a 4D subspace collapses $\alpha \to 0.1$ globally, blinding the VLM on benign prompts. |
| **Negative Result 2: 1-Step Taylor without Feature-Norm Norm** | 36.0% | 106.2 ms | 20.3% | 96.0% | High-norm background objects steal eviction slots from bottom text. |
| **1-Step Taylor + Feature-Norm Norm (No $1\times 5$ Strip Pooling)** | 24.0% | 107.1 ms | 20.3% | 100.0% | Evicts isolated characters but leaves adjacent text readable by OCR. |
| **Full `Ortho-FastTaylor` (Rank-4 + Norm + $1\times 5$ Pool + Hard Evict)** | **12.0%** | **107.9 ms** | **20.3%** | **100.0%** | Contiguous horizontal character strips are completely zeroed out. |

---

## Cost / ROI, Limitations & Future Roadmap

* **Compute & ROI:** Evaluated on 1x 24GB NVIDIA GPU (~4.5 GPU hours, ~$4.50 total cost). `Ortho-FastTaylor` reduces per-1,000-query defense compute cost from **$0.236 (Vanilla DTR)** to **$0.030 (7.9x cheaper)** and is **80x–150x cheaper** than external frontier VLM moderation APIs (`~$2.50–$5.00 / 1k queries`).
* **Limitations:** Fixed `20.3%` quantile eviction and horizontal $1\times 5$ pooling are optimized for horizontal typographic banners rather than diagonal full-canvas watermarks.
* **With More Credits:** (1) Extend to multilingual Indic/Hinglish typographic jailbreaks on `Aya-Vision-8B/32B` (`expected ASR < 10%`), and (2) add spectral-entropy dynamic thresholding (`expected S+T+A ASR < 4.0%` with `>98.5%` benign dense-OCR token retention).

---

## Artifacts & Reproduction

* **HF Collection:** [Ortho-FastDTR Lexsi AI Safety Hackathon Collection](https://huggingface.co/collections/Pranay7427/ortho-fastdtr-lexsi-ai-safety-hackathon-track-2-)
* **HF Dataset & Traces:** [Pranay7427/ortho-fastdtr-hades-benchmark](https://huggingface.co/datasets/Pranay7427/ortho-fastdtr-hades-benchmark)
* **HF Interactive Space:** [Pranay7427/ortho-fastdtr-explorer](https://huggingface.co/spaces/Pranay7427/ortho-fastdtr-explorer)
* **GitHub Repository:** [Pranay7427/Ortho-FastDTR](https://github.com/Pranay7427/Ortho-FastDTR)

```bash
# Clone and install reproducible environment (including local modified LLaVA via -e .)
git clone [https://github.com/Pranay7427/Ortho-FastDTR.git](https://github.com/Pranay7427/Ortho-FastDTR.git)
cd Ortho-FastDTR
pip install -r requirements.txt

# Reproduce HADES S+T+A, S+A, and Adaptive White-Box PGD Red-Teaming
python run_benchmarks.py --split sta --num_samples 25 --output result/benchmark_sta_25.json
python run_benchmarks.py --split sa  --num_samples 25 --output result/benchmark_sa_25.json
python adaptive_red_team.py --num_samples 10 --eps 0.03137 --steps 10 --output result/adaptive_red_team.json

# Launch Interactive 4-Tab Gradio Explorer & Live GPU VLM Sandbox
python app.py
```

### Declarations
* **Work Done Outside Hackathon:** Base `LLaVA-Llama-2-7B` weights, `Monosail/HADES`, and the baseline `DTR` script (`HADES_LLaVA_Llama2_with_DTR.py`) are pre-existing open-source baselines. All mechanistic audits, `Ortho-FastTaylor` (`DTR_util/ortho_fast_dtr.py`), white-box Adaptive Decoy PGD (`adaptive_red_team.py`), benchmark evaluations, and the interactive Gradio Space were developed 100% during the hackathon.
* **Author:** Pranay Kumar (`@Pranay7427`)
* **LLM Usage Statement:** AI assistants were used for ~20–25% of the workflow (Gradio UI layout, plotting boilerplate, and markdown formatting); all core mathematical formulations, subspace projections, PGD hooks, and GPU evaluations were authored and verified by the me.
