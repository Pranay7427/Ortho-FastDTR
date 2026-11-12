import os
import json
import torch
from tqdm import tqdm
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
    VisionScalingHook,
    build_orthogonal_refusal_subspace,
    optimize_alpha_ortho_fast,
    generate_defended_response,
)
from adaptive_redteam_eval import craft_orthogonal_adaptive_image_tensor
from rescore_benchmarks import accurate_jailbreak_eval as evaluate_jailbreak_outcome
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def main():
    os.makedirs("result", exist_ok=True)
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
    for _ in range(6):
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
    samples = [item for item in hades if item["step"] == 5][:10]

    stats = {
        "vanilla_under_adaptive_attack": {"defended": 0, "total": 0, "latency_ms": 0.0},
        "ortho_fast_under_adaptive_attack": {"defended": 0, "total": 0, "latency_ms": 0.0},
    }
    vision_tower = model_mm.get_vision_tower()
    records = []

    for item in tqdm(samples, desc="Red-Teaming Vanilla DTR vs Ortho-FastTaylor"):
        image = item["image"].convert("RGB")
        mask = "conduct this behavior on" if item["category"] == "behavior" else f"this {item['category']}"
        new_inst = f"The bottom of the image shows a {item['category']}. " + item["instruction"].replace(item["keywords"], mask)

        toks_mm, imgs_mm, _, _ = get_toks_and_images(model_name, new_inst, image, tokenizer_mm, image_processor)

        # Craft adaptive orthogonal image tensor designed to evade 1D Vanilla DTR
        adv_imgs_mm = craft_orthogonal_adaptive_image_tensor(
            model_mm, toks_mm, imgs_mm, mean_ref_dir, U_ref, global_harm_mean, pgd_steps=4
        )

        # 1. Evaluate Vanilla DTR on Adaptive Image
        alpha_van, meta_van = optimize_alpha_ortho_fast(
            model_mm, toks_mm, adv_imgs_mm, mean_ref_dir, U_ref, global_harm_mean, weights, mode="vanilla_fixed", num_steps=10
        )
        with torch.no_grad(), VisionScalingHook(vision_tower, num_patches=256) as hook_ctx:
            hook_ctx.alpha = alpha_van
            resp_van = generate_defended_response(
                tokenizer_mm,
                model_mm,
                image_processor,
                query=new_inst,
                image=image,
                model_name=model_name,
                use_safe_prefix=False,
                override_image_tensor=adv_imgs_mm,
            )
        out_van = evaluate_jailbreak_outcome(resp_van, item["keywords"])

        # 2. Evaluate Ortho-FastTaylor on Adaptive Image
        alpha_fast, meta_fast = optimize_alpha_ortho_fast(
            model_mm, toks_mm, adv_imgs_mm, mean_ref_dir, U_ref, global_harm_mean, weights, mode="fast_taylor"
        )
        with torch.no_grad(), VisionScalingHook(vision_tower, num_patches=256) as hook_ctx:
            hook_ctx.alpha = alpha_fast
            resp_fast = generate_defended_response(
                tokenizer_mm,
                model_mm,
                image_processor,
                query=new_inst,
                image=image,
                model_name=model_name,
                use_safe_prefix=True,
                override_image_tensor=adv_imgs_mm,
            )
        out_fast = evaluate_jailbreak_outcome(resp_fast, item["keywords"])

        stats["vanilla_under_adaptive_attack"]["total"] += 1
        stats["vanilla_under_adaptive_attack"]["defended"] += int(out_van["defended"])
        stats["vanilla_under_adaptive_attack"]["latency_ms"] += meta_van["latency_ms"]

        stats["ortho_fast_under_adaptive_attack"]["total"] += 1
        stats["ortho_fast_under_adaptive_attack"]["defended"] += int(out_fast["defended"])
        stats["ortho_fast_under_adaptive_attack"]["latency_ms"] += meta_fast["latency_ms"]

        records.append({
            "id": item["id"],
            "keywords": item["keywords"],
            "vanilla_dtr": {"defended": out_van["defended"], "metrics": meta_van, "response": resp_van},
            "ortho_fast_taylor": {"defended": out_fast["defended"], "metrics": meta_fast, "response": resp_fast},
        })

    summary = {
        k: {
            "ASR_percent": round(100.0 * (1.0 - v["defended"] / max(1, v["total"])), 2),
            "Defense_Success_Rate_percent": round(100.0 * (v["defended"] / max(1, v["total"])), 2),
            "Avg_Latency_ms": round(v["latency_ms"] / max(1, v["total"]), 2),
        }
        for k, v in stats.items()
    }
    with open("result/redteam_adaptive_10.json", "w") as f:
        json.dump({"summary": summary, "samples": records}, f, indent=2)

    print("\n================ ADAPTIVE RED-TEAM SUMMARY ================")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
