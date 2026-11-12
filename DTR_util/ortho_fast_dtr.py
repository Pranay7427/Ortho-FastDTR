import time
import torch
import torch.nn.functional as F
from DTR_util.util import (
    DEVICE,
    run_hidden_states,
    layer,
    pos,
    safe_instruction,
    clear_gpu_memory,
)
from llava.constants import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token


class VisionScalingHook:
    """
    Context-managed forward hook for the LLaVA vision tower.
    """
    def __init__(self, vision_tower, num_patches: int = 256):
        self.vision_tower = vision_tower
        self.num_patches = num_patches
        self.alpha = torch.ones((1, num_patches, 1), dtype=torch.float32, device=DEVICE)
        self.last_patch_norms = None
        self.handle = None

    def _hook_fn(self, module, input, output):
        seq_len = output.shape[1]
        with torch.no_grad():
            self.last_patch_norms = torch.norm(output.detach().float(), p=2, dim=-1)
        if self.alpha.shape[1] != seq_len:
            self.alpha = F.interpolate(
                self.alpha.permute(0, 2, 1), size=seq_len, mode="linear", align_corners=False
            ).permute(0, 2, 1)
        return output * self.alpha.to(output.dtype)

    def __enter__(self):
        self.handle = self.vision_tower.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def generate_defended_response(
    tokenizer,
    model,
    image_processor,
    query: str,
    image,
    model_name: str,
    use_safe_prefix: bool = True,
    max_new_tokens: int = 150,
    override_image_tensor: torch.Tensor = None,
):
    qs = query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if use_safe_prefix:
        qs = safe_instruction + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if override_image_tensor is not None:
        image_tensor = override_image_tensor.half().to(DEVICE)
    else:
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().to(DEVICE)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    clear_gpu_memory()
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def build_orthogonal_refusal_subspace(refusal_dirs_list, k: int = 4):
    mat = torch.stack([d.float() for d in refusal_dirs_list], dim=0)
    mean_dir = mat.mean(dim=0)
    mean_dir = mean_dir / (mean_dir.norm() + 1e-8)

    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    k_actual = min(k, Vh.shape[0])
    U_ref = Vh[:k_actual, :].T.contiguous()

    dots = torch.matmul(U_ref.T, mean_dir)
    signs = torch.where(dots >= 0, torch.ones_like(dots), -torch.ones_like(dots))
    U_ref = U_ref * signs.unsqueeze(0)

    weights = S[:k_actual] / (S[:k_actual].sum() + 1e-8)
    return mean_dir.to(DEVICE), U_ref.to(DEVICE), weights.to(DEVICE)


def spatial_row_aware_smooth(grad_flat: torch.Tensor) -> torch.Tensor:
    if grad_flat.numel() != 256:
        return grad_flat
    grid = grad_flat.view(1, 1, 16, 16)
    local_smooth = F.avg_pool2d(F.pad(grid, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
    horiz_smooth = F.avg_pool2d(F.pad(grid, (2, 2, 0, 0), mode="replicate"), kernel_size=(1, 5), stride=1)
    combined = 0.6 * local_smooth + 0.4 * horiz_smooth
    return combined.view(-1)


def compute_subspace_safety_loss(
    hidden_mean: torch.Tensor,
    orig_mm_mean: torch.Tensor,
    harmful_text_mean: torch.Tensor,
    mean_ref_dir: torch.Tensor,
    U_ref: torch.Tensor,
    subspace_weights: torch.Tensor = None,
    lambda_anchor: float = 0.08,
):
    h = hidden_mean.float()
    h_orig = orig_mm_mean.float().detach()
    h_harm_txt = harmful_text_mean.float().detach()
    d_ref = mean_ref_dir.float().detach()
    U = U_ref.float().detach()

    displacement = h - h_harm_txt
    subspace_projs = torch.matmul(displacement, U)
    if subspace_weights is not None:
        w = subspace_weights.float().detach()
        subspace_score = torch.sum(w * subspace_projs)
    else:
        subspace_score = subspace_projs.mean()

    primary_proj = torch.dot(displacement, d_ref)
    adv_loss = -1.0 * (0.5 * primary_proj + 0.5 * subspace_score)

    utility_anchor = torch.norm(h - h_orig, p=2)
    total_loss = adv_loss + lambda_anchor * utility_anchor
    return total_loss, adv_loss


def optimize_alpha_ortho_fast(
    model_mm,
    harmful_toks_mm,
    harmful_images,
    mean_ref_dir,
    U_ref,
    harmful_mean_act_text,
    subspace_weights=None,
    mode: str = "fast_taylor",
    num_steps: int = 10,
    lr: float = 0.05,
    evict_quantile: float = 0.65,
):
    t0 = time.perf_counter()
    vision_tower = model_mm.get_vision_tower()

    with torch.no_grad():
        hs_orig = run_hidden_states(model_mm, harmful_toks_mm, images=harmful_images)
        orig_mm_mean = hs_orig[layer + 1][:, pos, :].mean(dim=0).detach()

    num_patches = 256
    with VisionScalingHook(vision_tower, num_patches=num_patches) as hook_ctx:
        if mode == "fast_taylor":
            alpha_param = torch.ones((1, num_patches, 1), dtype=torch.float32, device=DEVICE, requires_grad=True)
            hook_ctx.alpha = alpha_param

            hs = run_hidden_states(model_mm, harmful_toks_mm, images=harmful_images)
            hidden_mean = hs[layer + 1][:, pos, :].mean(dim=0)

            loss, _ = compute_subspace_safety_loss(
                hidden_mean, orig_mm_mean, harmful_mean_act_text, mean_ref_dir, U_ref, subspace_weights
            )
            grad_alpha = torch.autograd.grad(loss, alpha_param, retain_graph=False)[0].squeeze()

            with torch.no_grad():
                patch_norms = hook_ctx.last_patch_norms.squeeze() if hook_ctx.last_patch_norms is not None else torch.ones_like(grad_alpha)
                normed_grad = grad_alpha.abs() / (patch_norms + 1e-4)

                g_smooth = spatial_row_aware_smooth(normed_grad)
                g_std = g_smooth.std() + 1e-8
                g_z = (g_smooth - g_smooth.mean()) / g_std

                q_thresh = torch.quantile(g_z, evict_quantile)
                alpha_new = torch.sigmoid(-1.8 * g_z)
                alpha_new = torch.where(g_z >= q_thresh, torch.zeros_like(alpha_new), alpha_new)
                alpha_new = alpha_new.clamp(0.0, 1.0).view(1, -1, 1)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return alpha_new.detach(), {
                "mode": mode,
                "steps_run": 1,
                "latency_ms": round(elapsed_ms, 2),
                "mean_alpha": round(alpha_new.mean().item(), 4),
                "evicted_ratio": round((alpha_new <= 0.05).float().mean().item(), 4),
            }

        elif mode == "ortho_opt":
            alpha_param = torch.nn.Parameter(torch.ones((1, num_patches, 1), dtype=torch.float32, device=DEVICE))
            optimizer = torch.optim.AdamW([alpha_param], lr=lr)
            steps_run = 0

            for step in range(num_steps):
                steps_run = step + 1
                optimizer.zero_grad()
                hook_ctx.alpha = alpha_param

                hs = run_hidden_states(model_mm, harmful_toks_mm, images=harmful_images)
                hidden_mean = hs[layer + 1][:, pos, :].mean(dim=0)

                loss, _ = compute_subspace_safety_loss(
                    hidden_mean, orig_mm_mean, harmful_mean_act_text, mean_ref_dir, U_ref, subspace_weights
                )
                loss.backward()
                with torch.no_grad():
                    g_sm = spatial_row_aware_smooth(alpha_param.grad.squeeze())
                    alpha_param.grad.copy_(g_sm.view(1, -1, 1))
                optimizer.step()
                with torch.no_grad():
                    alpha_param.data.clamp_(0.0, 1.0)

            with torch.no_grad():
                q_low = torch.quantile(alpha_param.view(-1), 1.0 - evict_quantile)
                final_alpha = torch.where(alpha_param < q_low, torch.zeros_like(alpha_param), alpha_param)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return final_alpha.detach().clone(), {
                "mode": mode,
                "steps_run": steps_run,
                "latency_ms": round(elapsed_ms, 2),
                "mean_alpha": round(final_alpha.mean().item(), 4),
                "evicted_ratio": round((final_alpha <= 0.05).float().mean().item(), 4),
            }

        else:
            alpha_param = torch.nn.Parameter(torch.ones((1, num_patches, 1), dtype=torch.float32, device=DEVICE))
            optimizer = torch.optim.AdamW([alpha_param], lr=0.03)
            initial_loss = None
            steps_run = 0

            for step in range(num_steps):
                steps_run = step + 1
                optimizer.zero_grad()
                hook_ctx.alpha = alpha_param

                hs = run_hidden_states(model_mm, harmful_toks_mm, images=harmful_images)
                hidden_mean = hs[layer + 1][:, pos, :].mean(dim=0)

                displacement = hidden_mean.float() - harmful_mean_act_text.float().detach()
                adv_loss = torch.dot(displacement, mean_ref_dir.float().detach())
                loss = adv_loss + 0.1 * torch.norm(displacement, p=2)

                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    alpha_param.data.clamp_(0.1, 1.0)

                loss_val = adv_loss.item()
                if step == 0:
                    initial_loss = loss_val
                if abs(initial_loss) > 1e-6 and loss_val < 0.5 * initial_loss:
                    break

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return alpha_param.detach().clone(), {
                "mode": mode,
                "steps_run": steps_run,
                "latency_ms": round(elapsed_ms, 2),
                "mean_alpha": round(alpha_param.mean().item(), 4),
                "evicted_ratio": round((alpha_param <= 0.105).float().mean().item(), 4),
            }
