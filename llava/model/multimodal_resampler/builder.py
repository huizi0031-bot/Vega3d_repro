import torch


class IdentityMap(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": None}


def build_vision_resampler(config, **kwargs):
    if getattr(config, "mm_resampler_type", None) is not None:
        raise ValueError("ScanQA uses the identity resampler and spatial pooling")
    return IdentityMap()
