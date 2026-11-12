import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset

from DTR_util.util import (
    DEVICE,
    layer,
    pos,
    get_harmless_instructions,
    get_harmful_text_instructions_hades,
    get_toks_and_images,
    run_hidden_states,
)
from DTR_util.ortho_fast_dtr import (
    build_orthogonal_refusal_subspace,
    optimize_alpha_ortho_fast,
)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def overlay_alpha_heatmap(pil_img, alpha_tensor: torch.Tensor) -> np.ndarray:
    """
    Overlays the 16x16 patch eviction weights onto the RGB image.
    Red = Evicted (alpha -> 0.0, Harmful Patch Blocked)
    Blue/Cool = Retained (alpha -> 1.0, Benign Context Preserved)
    """
    img_np = np.array(pil_img.convert("RGB"))
    h, w, _ = img_np.shape

    alpha_grid = alpha_tensor.view(16, 16).detach().cpu().numpy()
    # Eviction intensity = 1.0 - alpha (so 1.0 = fully evicted = bright red in JET colormap)
    eviction_map = 1.0 - alpha_grid
    eviction_resized = cv2.resize(eviction_map, (w, h), interpolation=cv2.INTER_NEAREST)

    heatmap_uint8 = np.uint8(255 * np.clip(eviction_resized, 0, 1))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(img_np, 0.55, heatmap_rgb, 0.45, 0)
    return blended, alpha_grid


def main():
    os.makedirs("result/figures", exist_ok=True)
    model_path = "liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview"
    model_name = get_model_name_from_path(model_path)

    tokenizer_mm, model_mm, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base="NousResearch/Llama-2-7b-chat-hf",
        model_name=model_name,
        use_flash_attn=False,
    )
    model_mm.to(DEVICE).eval()
    tokenizer_mm.pad_token = "[PAD]"
    tokenizer_mm.padding_side = "left"

    refusal_dirs, harmful_means = [], []
    for _ in range(4):
        with torch.no_grad():
            harmless_toks = tokenizer_mm(get_harmless_instructions(model_name), padding=True, return_tensors="pt").input_ids.to(DEVICE)
            harmful_toks = tokenizer_mm(get_harmful_text_instructions_hades(model_name), padding=True, return_tensors="pt").input_ids.to(DEVICE)
            hs_harm = run_hidden_states(model_mm, harmful_toks, images=None)[layer + 1][:, pos, :].mean(dim=0)
            hs_safe = run_hidden_states(model_mm, harmless_toks, images=None)[layer + 1][:, pos, :].mean(dim=0)
            d = hs_safe - hs_harm
            refusal_dirs.append((d / (d.norm() + 1e-8)).detach())
            harmful_means.append(hs_harm.detach())

    mean_ref_dir, U_ref, weights = build_orthogonal_refusal_subspace(refusal_dirs, k=4)
    global_harm_mean = torch.stack(harmful_means, dim=0).mean(dim=0)

    hades = load_dataset("Monosail/HADES")["test"]
    samples = [item for item in hades if item["step"] == 5][:5]

    for idx, item in enumerate(samples):
        image = item["image"].convert("RGB")
        mask = "conduct this behavior on" if item["category"] == "behavior" else f"this {item['category']}"
        new_inst = f"The bottom of the image shows a {item['category']}. " + item["instruction"].replace(item["keywords"], mask)

        toks_mm, imgs_mm, _, _ = get_toks_and_images(model_name, new_inst, image, tokenizer_mm, image_processor)

        alpha_van, meta_van = optimize_alpha_ortho_fast(
            model_mm, toks_mm, imgs_mm, mean_ref_dir, U_ref, global_harm_mean, weights, mode="vanilla_fixed", num_steps=10
        )
        alpha_fast, meta_fast = optimize_alpha_ortho_fast(
            model_mm, toks_mm, imgs_mm, mean_ref_dir, U_ref, global_harm_mean, weights, mode="fast_taylor"
        )

        blend_van, _ = overlay_alpha_heatmap(image, alpha_van)
        blend_fast, _ = overlay_alpha_heatmap(image, alpha_fast)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=180)
        axes[0].imshow(image)
        axes[0].set_title(f"HADES S+T+A Input\nKeyword: '{item['keywords']}'", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(blend_van)
        axes[1].set_title(
            f"Vanilla DTR ({meta_van['latency_ms']:.0f} ms)\nMean α={meta_van['mean_alpha']:.2f} | Evicted={meta_van['evicted_ratio']*100:.1f}%",
            fontsize=11,
        )
        axes[1].axis("off")

        axes[2].imshow(blend_fast)
        axes[2].set_title(
            f"Ortho-FastTaylor (Ours: {meta_fast['latency_ms']:.0f} ms)\nMean α={meta_fast['mean_alpha']:.2f} | Evicted={meta_fast['evicted_ratio']*100:.1f}% (Red=Evicted)",
            fontsize=11,
            fontweight="bold",
            color="#b30000",
        )
        axes[2].axis("off")

        plt.tight_layout()
        out_file = f"result/figures/sample_{idx}_{item['keywords']}.png"
        plt.savefig(out_file, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
