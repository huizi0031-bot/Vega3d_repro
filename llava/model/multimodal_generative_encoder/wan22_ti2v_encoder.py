"""Frozen Wan2.2 TI2V-5B features, retaining VEGA's per-frame extraction protocol."""
import json
import math
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from .common import resolve_inference_dtype, resize_center_crop, to_neg_one_to_one, split_frames
from .wan22.configs.wan_ti2v_5B import ti2v_5B
from .wan22.modules import model as wan_model_module
from .wan22.modules.model import WanModel
from .wan22.modules.vae2_2 import Wan2_2_VAE
from .wan22.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def masked_sdpa(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                softmax_scale=None, q_scale=None, causal=False,
                window_size=(-1,-1), deterministic=False, dtype=torch.bfloat16,
                version=None):
    """Exact scaled-dot-product attention with valid-key masking; no FA ABI dependency."""
    if window_size != (-1,-1):
        raise ValueError('Window attention is not used by this Wan2.2 configuration')
    if q_scale is not None:
        q = q*q_scale
    mask = None
    if k_lens is not None:
        valid = torch.arange(k.shape[1], device=k.device)[None,:] < k_lens.to(k.device)[:,None]
        mask = valid[:,None,None,:]
    if causal and mask is not None:
        triangle = torch.ones(q.shape[1], k.shape[1],dtype=torch.bool,device=q.device).tril()
        mask = mask & triangle[None,None,:,:]
    result = F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
        attn_mask=mask,dropout_p=dropout_p,is_causal=causal and mask is None,scale=softmax_scale)
    result = result.transpose(1,2).contiguous()
    if q_lens is not None:
        valid = torch.arange(q.shape[1],device=q.device)[None,:] < q_lens.to(q.device)[:,None]
        result = result*valid[:,:,None,None]
    return result


class Wan22TI2VOnlineEncoder(nn.Module):
    """[N,3,H,W] -> [N,3072,14,14]; T2V mode, not image-conditioned generation.

    Reuse VEGA sampling/normalization and hook the requested DiT block. Each
    scanned view remains an independent one-frame latent, as in the Qwen3 base.
    """
    def __init__(self, config):
        nn.Module.__init__(self)
        self.task = getattr(config, 'generative_vision_tower_task', 'ti2v-5B')
        if self.task != 'ti2v-5B':
            raise ValueError('This adapter requires Wan2.2-TI2V-5B; A14B experts are not interchangeable')
        self.checkpoint_dir = str(getattr(config, 'generative_vision_tower_checkpoint', ''))
        path = Path(self.checkpoint_dir)
        if not self.checkpoint_dir or not (path/'config.json').is_file():
            raise FileNotFoundError('Prepare the official Wan2.2-TI2V-5B checkpoint first')
        stored = json.loads((path/'config.json').read_text())
        expected = {'model_type': 'ti2v', 'dim': 3072, 'in_dim': 48, 'out_dim': 48, 'num_layers': 30}
        for key, value in expected.items():
            if stored.get(key) != value:
                raise ValueError(f'Incompatible Wan2.2 checkpoint: {key}={stored.get(key)}, expected {value}')
        self.size = getattr(config, 'generative_vision_tower_size', '1280*704')
        if self.size not in ('1280*704', '704*1280'):
            raise ValueError('TI2V-5B uses 1280*704 or 704*1280, divisible by VAE/patch stride 32')
        self.frame_width, self.frame_height = map(int, self.size.split('*'))
        self.timestep = int(getattr(config, 'generative_vision_tower_timestep', 300))
        self.shift = float(getattr(config, 'generative_vision_tower_shift', 5.0))
        self.feat_block_idx = int(getattr(config, 'generative_vision_tower_feat_block_idx', 20))
        if not 0 <= self.feat_block_idx < 30 or not 0 <= self.timestep <= 1000:
            raise ValueError('Invalid Wan2.2 feature block or timestep')
        self.cfg = ti2v_5B
        self.param_dtype = resolve_inference_dtype(config)
        self.num_train_timesteps = self.cfg.num_train_timesteps
        self.vae_stride, self.patch_size = self.cfg.vae_stride, self.cfg.patch_size
        self._model_device = torch.device('cpu')
        self._scheduler_device = None
        self.prompt_emb_path = str(getattr(config, 'generative_vision_tower_prompt_emb_path',
                                          'llava/model/multimodal_generative_encoder/wan22_prompt_embedding.pt'))
        if not Path(self.prompt_emb_path).is_file():
            raise FileNotFoundError('Export the Wan2.2 empty-text embedding before extraction: '+self.prompt_emb_path)
        prompt = torch.load(self.prompt_emb_path, map_location='cpu', weights_only=True)
        if not isinstance(prompt, dict) or prompt.get('model_family') != 'wan2.2-ti2v-5B':
            raise ValueError('Use a versioned Wan2.2 text embedding, not the Wan2.1 cache')
        self.prompt_context = prompt['context'].detach().cpu().contiguous()
        if self.prompt_context.ndim != 2 or self.prompt_context.shape[-1] != 4096:
            raise ValueError('Expected UMT5 context [tokens,4096]')
        self.vae = Wan2_2_VAE(vae_pth=str(path/self.cfg.vae_checkpoint), dtype=self.param_dtype,
                             device=torch.device('cpu'))
        # Match official precision: parameters stay fp32; autocast handles the
        # DiT while its time embedding explicitly remains float32.
        # Use Torch SDPA for this version; do not import a flash-attn binary
        # built against the old project's different Torch version.
        wan_model_module.flash_attention = masked_sdpa
        self.model = WanModel.from_pretrained(str(path)).eval().requires_grad_(False)
        self.scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=self.num_train_timesteps,
                                                     shift=1, use_dynamic_shifting=False)

    def _move_vae_to_device(self, device):
        if torch.device(self.vae.device) != device:
            self.vae.model.to(device)
            self.vae.scale = [value.to(device) for value in self.vae.scale]
            self.vae.device = device

    def _forward_single_video(self, frames, device):
        if frames.ndim != 4 or frames.shape[0] == 0:
            raise ValueError('Expected nonempty [N,3,H,W] frames')
        x = self._prepare_frames(frames)
        frame_list = [frame.unsqueeze(1) for frame in x]
        self._ensure_scheduler_ready(device)
        tau = self._select_timestep(self.scheduler.timesteps, self.timestep)
        context = self._get_text_context(device, len(frame_list))
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=self.param_dtype,
                                                   enabled=self.param_dtype != torch.float32):
            latents = self.vae.encode(frame_list)
            if not latents or any(z.shape[0] != 48 for z in latents):
                raise RuntimeError('Wan2.2 VAE must produce 48-channel latents')
            shape = latents[0].shape
            seq_len = math.ceil(shape[1]*shape[2]*shape[3]/math.prod(self.patch_size))
            batch = torch.stack(latents)
            noise = torch.randn_like(batch)
            noisy = self.scheduler.add_noise(batch, noise, tau.expand(len(latents)))
            captured = {}
            handle = self.model.blocks[self.feat_block_idx].register_forward_hook(
                lambda module, inputs, output: captured.update(feature=output.detach()))
            try:
                # TI2V-5B uses token-wise time embeddings, unlike the 2.1 call.
                timesteps = tau.reshape(1, 1).expand(len(latents), seq_len).long()
                self.model(list(noisy.unbind(0)), t=timesteps, context=context, seq_len=seq_len)
            finally:
                handle.remove()
            feature = captured.get('feature')
            grid_h = self.frame_height // (self.vae_stride[1]*self.patch_size[1])
            grid_w = self.frame_width // (self.vae_stride[2]*self.patch_size[2])
            expected = (len(latents), grid_h*grid_w, self.cfg.dim)
            if feature is None or tuple(feature.shape) != expected:
                raise RuntimeError(f'Unexpected Wan2.2 intermediate feature, expected {expected}')
            feature = feature.reshape(len(latents), grid_h, grid_w, self.cfg.dim).permute(0,3,1,2)
            return F.adaptive_avg_pool2d(feature, (14,14)).contiguous()


    def _prepare_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Convert input frames to [-1, 1], resize/crop to WAN resolution.
        """
        x = to_neg_one_to_one(frames)
        x = resize_center_crop(x, self.frame_height, self.frame_width)
        return x.to(dtype=self.param_dtype)


    def _get_text_context(self, device: torch.device, batch_size: int):
        context = self.prompt_context.to(device=device, non_blocking=True)
        if torch.is_floating_point(context):
            context = context.to(self.param_dtype)
        return [context] * batch_size


    def _ensure_scheduler_ready(self, device: torch.device):
        if self._scheduler_device != device:
            # Use full train-time schedule to preserve exact timestep semantics.
            self.scheduler.set_timesteps(self.num_train_timesteps, device=device, shift=self.shift)
            self._scheduler_device = device


    def _select_timestep(self, timesteps: torch.Tensor, target_timestep: int) -> torch.Tensor:
        if timesteps.numel() == 0:
            raise ValueError("Scheduler timesteps is empty.")
        tau_tensor = torch.tensor(int(target_timestep), device=timesteps.device, dtype=timesteps.dtype)
        idx = torch.argmin(torch.abs(timesteps - tau_tensor))
        return timesteps[idx]


    def _move_models_to_device(self, device: torch.device):
        if self._model_device != device:
            self.model.to(device)
            self._move_vae_to_device(device)
            self._model_device = device


    def forward(
        self,
        frames: torch.Tensor,
        split_sizes: List[int] | None = None,
        video_contexts=None,
    ) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"Expected [N, 3, H, W], got {tuple(frames.shape)}")
        if frames.shape[0] == 0:
            return frames.new_zeros((0, getattr(self.cfg, "dim", 1280), 14, 14))

        device = frames.device
        self._move_models_to_device(device)
        chunks = split_frames(frames, split_sizes)
        outs = []
        for chunk in chunks:
            outs.append(self._forward_single_video(chunk, device=device))
        return torch.cat(outs, dim=0)
