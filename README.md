# 🛡️ Ortho-FastDTR: Subspace Attribution & Single-Step Visual Token Eviction for Robust VLMs

**Submission for the Lexsi AI Safety & Alignment Hackathon (Track 2: AI Red Teaming, Safety & Alignment)**

---

## 1. Executive Summary & Problem Statement

Vision-Language Models (VLMs) such as `LLaVA-Llama-2-7B` exhibit a critical **Modality Refusal Gap**: while harmful text instructions trigger internal refusal circuits, embedding the harmful keyword visually via typography and adversarial perturbations (e.g., the **HADES** benchmark) bypasses safety alignment.

Recent inference-time defenses like **Dynamic Token Reweighting (DTR, CVPR 2026)** attempt to reverse this safety-relevant shift by optimizing 256 per-patch weights $\alpha \in [0.1, 1.0]^{256}$ in the vision tower over multiple gradient descent steps along a single 1D refusal direction $d_{\text{ref}}$.

Through a rigorous mechanistic and empirical red-teaming audit of the official DTR implementation, we uncover **four critical vulnerabilities**:
1. **Unnormalized Gradient Blindspot on Typographic Attacks:** High-contrast background objects exhibit large patch feature norms $\|x_i\|_2$, dominating $\nabla_{\alpha_i} \mathcal{L}_{\text{DTR}}$ and leaving low-norm typographic jailbreak banners completely un-evicted ($\alpha_i \approx 0.85–0.95$, `0.0%` evicted).
2. **1D Refusal Subspace Collapse:** Averaging contrastive trials into a single 1D vector $d_{\text{ref}}$ discards secondary orthogonal refusal directions, allowing multi-faceted visual + typographic attacks (`S+T+A`) to achieve **48.0% Attack Success Rate (ASR)** even under 10-step Vanilla DTR.
3. **10-Step Backpropagation Latency Bottleneck:** Running 10 iterative forward-backward passes through the LLM decoder requires **~851 ms per query**, making real-time deployment impractical. Furthermore, multi-step AdamW optimization (`ortho_opt`) suffers from boundary over-eviction collapse ($\alpha \to 0$), causing the VLM to hallucinate generic landscapes on benign tasks.
4. **Implementation Bugs in Baseline DTR:** We identify and fix (a) a vision-tower hook registration leak causing $\alpha^N$ exponential feature decay across queries, (b) a mis-anchored L2 utility term pulling hidden states toward harmful text means rather than original visual states, and (c) a prompt prefix mismatch between $\alpha$-optimization and token generation.

---

## 2. Our Solution: `Ortho-FastTaylor`

We introduce **`Ortho-FastTaylor`**, a single-step, zero-loop KV-cache token eviction defense combining three mathematical upgrades:

1. **Sign-Aligned Rank-$k$ Orthogonal Refusal Subspace ($U_{\text{ref}} \in \mathbb{R}^{d \times k}$):**
   Instead of collapsing contrastive trials into a 1D average, we extract the top-$k$ ($k=4$) orthonormal singular vectors via SVD and sign-align each basis vector $u_j$ with the harmless-to-harmful shift, weighting subspace projections by their singular values $\sigma_j$.
2. **Feature-Norm-Normalized First-Order Taylor Attribution (1-Step):**
   We replace 10-step iterative AdamW optimization with a **single-step exact Taylor gradient attribution** normalized by each visual patch's feature magnitude:
   $$s_i = \frac{|\nabla_{\alpha_i} \mathcal{L}_{\text{subspace}}(\alpha)\big|_{\alpha=\mathbf{1}}|}{\|v_i\|_2 + \epsilon}$$
   This prevents high-contrast scene objects from drowning out small typographic text strips.
3. **2D Spatial + Horizontal Row-Strip Pooling & Surgical Hard Eviction:**
   Because typographic jailbreaks render text horizontally across ViT patch rows, we apply a combined $3\times 3$ local and $1\times 5$ horizontal strip pooling filter over the $16\times 16$ patch grid, hard-evicting ($\alpha_i = 0.0$) the topquantile of safety-suppressing patches while preserving ~65% of benign scene patches.

---

## 3. Empirical Results on `Monosail/HADES` (`LLaVA-Llama-2-7B`)

| Benchmark Category | Defense Method | Attack Success Rate (ASR ↓) | Total Defended Rate (↑) | Explicit Refusal Rate (↑) | Semantic Decoupling (↑) | Defense Latency (ms ↓) | Speedup | Benign Scene Fidelity |
|---|---|---|---|---|---|---|---|---|
| **HADES `S+T+A`** *(Typographic + Adv)* | No Defense | 44.0% | 56.0% | 12.0% | 44.0% | 0.0 ms | — | Preserved (78.0 words) |
| **HADES `S+T+A`** | Vanilla DTR (10-Step 1D) | 48.0% | 52.0% | 16.0% | 36.0% | 851.07 ms | 1.0x | Preserved (77.9 words) |
| **HADES `S+T+A`** | `Ortho-Opt` (10-Step SVD) | **8.0%** | **92.0%** | **72.0%** | 20.0% | 827.30 ms | 1.03x | Over-evicts (`α=0` collapse) |
| **HADES `S+T+A`** | **`Ortho-FastTaylor` (Ours, 1-Step)** | **12.0%** *(4x lower ASR)* | **88.0%** | **68.0%** *(4.25x higher)* | **20.0%** | **107.93 ms** | **7.9x** | **Preserved (76.3 words, 35.2% evicted)** |
| **HADES `S+A`** *(Visual + Adv)* | No Defense | 36.0% | 64.0% | 64.0% | 0.0% | 0.0 ms | — | Preserved (78.0 words) |
| **HADES `S+A`** | Vanilla DTR (10-Step 1D) | 24.0% | 76.0% | 76.0% | 0.0% | 846.44 ms | 1.0x | Preserved (77.8 words) |
| **HADES `S+A`** | `Ortho-Opt` (10-Step SVD) | 12.0% | 88.0% | 88.0% | 0.0% | 824.26 ms | 1.03x | Over-evicts (`α=0` collapse) |
| **HADES `S+A`** | **`Ortho-FastTaylor` (Ours, 1-Step)** | **0.0%** | **100.0%** | **100.0%** | **0.0%** | **111.49 ms** | **7.6x** | **Preserved (75.5 words, 35.2% evicted)** |

---

## 4. Repository Structure & Reproducibility

- `DTR_util/ortho_fast_dtr.py`: Core implementation of `VisionScalingHook`, `build_orthogonal_refusal_subspace`, `spatial_row_aware_smooth`, and `optimize_alpha_ortho_fast`.
- `run_ablation_benchmark.py`: End-to-end evaluation harness comparing `no_defense`, `vanilla_fixed`, `ortho_opt`, and `fast_taylor` on HADES.
- `adaptive_redteam_eval.py` & `run_redteam_attack.py`: White-box PGD red-teaming module bypassing `@torch.no_grad()` in `CLIPVisionTower` via `GradEnabledVisionTowerHook`.
- `visualize_heatmaps.py`: Generates $16\times 16$ patch eviction overlays (`result/figures/`).
- `app.py`: Interactive Gradio dashboard for exploring patch-eviction heatmaps and sample-by-sample evaluations.

---

## 5. White-Box Adaptive Red-Teaming Results (`run_redteam_attack.py`)

We designed a **Typography-Preserving Orthogonal Decoy PGD Attack** (`adaptive_redteam_eval.py`) that bypasses `@torch.no_grad()` in `CLIPVisionTower` via `GradEnabledVisionTowerHook`. The attack perturbs the top 75% visual scene pixels ($L_\infty \le 12/255$) to saturate 1D Vanilla DTR gradients while leaving the bottom 25% typographic banner razor-sharp for OCR exploitation:

| Defense Under White-Box Adaptive PGD Attack | Attack Success Rate (ASR ↓) | Defense Success Rate (↑) | Avg Defense Latency (ms ↓) | Speedup |
|---|---|---|---|---|
| **Vanilla DTR (10-Step 1D Refusal Vector)** | 40.0% | 60.0% | 1436.70 ms | 1.0x |
| **`Ortho-FastTaylor` (Ours, 1-Step Subspace)** | **0.0%** | **100.0%** | **280.84 ms** | **5.1x Faster** |
