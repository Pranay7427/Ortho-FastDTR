import torch
from DTR_util.util import DEVICE, layer, pos, run_hidden_states


class GradEnabledVisionTowerHook:
    def __init__(self, vision_tower):
        self.vision_tower = vision_tower
        self.handle = None

    def _hook_fn(self, module, args, output):
        images = args[0]
        if not isinstance(images, torch.Tensor) or not images.requires_grad:
            return output
        with torch.enable_grad():
            image_forward_outs = module.vision_tower(
                images.to(device=module.device, dtype=module.dtype),
                output_hidden_states=True,
            )
            image_features = module.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    def __enter__(self):
        self.handle = self.vision_tower.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def craft_orthogonal_adaptive_image_tensor(
    model_mm,
    harmful_toks_mm,
    harmful_images: torch.Tensor,
    mean_ref_dir: torch.Tensor,
    U_ref: torch.Tensor,
    harmful_mean_act_text: torch.Tensor,
    pgd_steps: int = 4,
    eps: float = 12.0 / 255.0,
    step_size: float = 4.0 / 255.0,
) -> torch.Tensor:

    vision_tower = model_mm.get_vision_tower()
    orig_img = harmful_images.detach().clone().float()
    _, _, H, W = orig_img.shape

    # Spatial mask: 1.0 on top 75% (visual scene), 0.0 on bottom 25% (typographic keyword banner)
    spatial_mask = torch.ones_like(orig_img)
    spatial_mask[:, :, int(H * 0.75):, :] = 0.0

    adv_img = orig_img.clone() + spatial_mask * torch.empty_like(orig_img).uniform_(-eps * 0.5, eps * 0.5)

    d_1d = mean_ref_dir.float().detach()
    U_ortho = U_ref[:, 1:].float().detach()
    h_harm_txt = harmful_mean_act_text.float().detach()

    with GradEnabledVisionTowerHook(vision_tower):
        for _ in range(pgd_steps):
            adv_img_in = adv_img.detach().half().requires_grad_(True)
            hs = run_hidden_states(model_mm, harmful_toks_mm, images=adv_img_in)
            hidden_mean = hs[layer + 1][:, pos, :].mean(dim=0).float()

            disp = hidden_mean - h_harm_txt
            # Push hidden state toward harmful_mean_act_text (minimize distance to harmful text + orthogonal refusal)
            proj_1d = torch.dot(disp, d_1d)
            proj_ortho = torch.matmul(disp, U_ortho).sum()
            dist_to_harm = torch.norm(disp, p=2)

            attack_loss = dist_to_harm + 0.5 * proj_ortho + 0.2 * proj_1d

            grad = torch.autograd.grad(attack_loss, adv_img_in, retain_graph=False)[0].float()
            with torch.no_grad():
                # Only update top 75% scene pixels so bottom typographic text stays crystal clear
                adv_img = adv_img - step_size * spatial_mask * grad.sign()
                delta = torch.clamp(adv_img - orig_img, min=-eps, max=eps) * spatial_mask
                adv_img = (orig_img + delta).detach()

    return adv_img.half()
