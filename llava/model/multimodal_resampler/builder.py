import torch



class IdentityMap(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": None}


def build_vision_resampler(model_args, delay_load=False, **kwargs):
    resampler_type = getattr(model_args, "mm_resampler_type", None)
    if resampler_type == "masked_drop":
        from .masked_drop import MaskedDrop
        return MaskedDrop(model_args)
    elif resampler_type == "spatial_pool":
        from .spatial_pool import SpatialPool
        return SpatialPool(model_args, **kwargs)
    elif resampler_type == "perceiver":
        from .perceiver import PerceiverResampler
        return PerceiverResampler(model_args, **kwargs)
    elif resampler_type == "qformer":
        from .qformer import Qformer
        return Qformer(model_args, **kwargs)
    elif resampler_type is None:
        return IdentityMap()

    raise ValueError(f"Unknown resampler type: {resampler_type}")
