"""Frozen Wan2.2 flow teacher. Its forward remains differentiable w.r.t. context.

The training distribution below is an explicit pilot choice, not a claim about
Wan's unpublished pretraining recipe. No sampling loop or generated RGB is used.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn


def make_flow_batch(clean, patch_size, *, shift=5.0, num_train_timesteps=1000,
                    noise=None, u=None):
    """I2V-style clean first latent, noisy later latents, token-wise times."""
    if clean.ndim != 5 or clean.shape[2] < 2:
        raise ValueError("Expected [B,C,T,H,W] with at least two temporal latents")
    if shift <= 0:
        raise ValueError("Flow shift must be positive")
    _, _, temporal, height, width = clean.shape
    if tuple(patch_size) != (1, 2, 2) or height % 2 or width % 2:
        raise ValueError("TI2V flow pilot requires patch_size=(1,2,2) and even latent H,W")
    noise = torch.randn_like(clean) if noise is None else noise
    u = torch.rand(clean.shape[0], device=clean.device) if u is None else u.to(clean.device)
    sigma = shift * u / (1 + (shift - 1) * u)
    sigma = sigma.clamp(1e-5, 1-1e-5)
    weight = torch.ones_like(clean)
    weight[:, :, 0] = 0
    noisy = (1-sigma[:, None, None, None, None])*clean + sigma[:, None, None, None, None]*noise
    noisy = noisy*weight + clean*(1-weight)
    token_mask = weight[:, 0, :, ::2, ::2].flatten(1)
    times = token_mask * (sigma[:, None]*num_train_timesteps)
    return noisy, noise-clean, weight, times


def conditional_flow_loss(model, clean, contexts, *, shift=5.0, noise=None, u=None):
    noisy, velocity, mask, times = make_flow_batch(
        clean, model.patch_size, shift=shift, noise=noise, u=u)
    seq_len = math.prod(clean.shape[2:]) // math.prod(model.patch_size)
    prediction = torch.stack(model(list(noisy.unbind()), t=times,
                                  context=list(contexts.unbind()), seq_len=seq_len))
    error = (prediction.float()-velocity.float()).square()*mask
    return (error.flatten(1).sum(1)/mask.flatten(1).sum(1)).mean()


class Wan22ForesightTeacher:
    """Runtime-only object: never registered in LM optimizer or checkpoint.

    Each rank holds one frozen teacher. Wan weights are bf16 except its explicit
    fp32 time MLPs; VAE parameters remain fp32 with native bf16 autocast.
    """
    def __init__(self, config, device):
        from .wan22.modules import model as wan_module
        from .wan22.modules.vae2_2 import Wan2_2_VAE
        from .wan22_ti2v_encoder import masked_sdpa

        path = Path(config.foresight_wan_checkpoint)
        stored = json.loads((path/"config.json").read_text())
        expected = {"model_type": "ti2v", "dim": 3072, "in_dim": 48,
                    "out_dim": 48, "num_layers": 30}
        for key, value in expected.items():
            if stored.get(key) != value:
                raise ValueError(f"Incompatible Wan2.2 TI2V-5B checkpoint: {key}")
        if stored.get("text_dim", 4096) != 4096:
            raise ValueError("Wan teacher must use its native 4096-dimensional text interface")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("The pretrained Wan teacher requires an explicitly selected CUDA device")
        wan_module.flash_attention = masked_sdpa
        # Lazy loading occurs outside the LM construction/ZeRO initialization.
        self.model = wan_module.WanModel.from_pretrained(str(path)).eval().requires_grad_(False)
        if self.model.text_dim != config.foresight_context_dim:
            raise ValueError("Foresight projection must match Wan's native text dimension")
        self.model.to(device=self.device, dtype=torch.bfloat16)
        self.model.time_embedding.float()
        self.model.time_projection.float()
        self.model.gradient_checkpointing = bool(config.foresight_wan_gradient_checkpointing)
        self.vae = Wan2_2_VAE(vae_pth=str(path/"Wan2.2_VAE.pth"),
                             dtype=torch.bfloat16, device=self.device)
        self.shift = float(config.foresight_flow_shift)
        self.height = int(config.foresight_video_height)
        self.width = int(config.foresight_video_width)
        self.frames = int(config.foresight_video_frames)
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Wan teacher must remain frozen")

    def __call__(self, contexts, pixels):
        if pixels.shape[1:] != (3, self.frames, self.height, self.width):
            raise ValueError(f"Unexpected teacher RGB shape {tuple(pixels.shape)}")
        pixels = pixels.to(self.device, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            latents = self.vae.encode(list(pixels.unbind()))
            if latents is None or any(z.shape[0] != 48 for z in latents):
                raise ValueError("Wan2.2 temporal VAE must produce 48-channel latents")
            clean = torch.stack(latents).detach()
        # Deliberately no no_grad / inference_mode / detach on the context path.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return conditional_flow_loss(self.model, clean, contexts.to(self.device), shift=self.shift)
