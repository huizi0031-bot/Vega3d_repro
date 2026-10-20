from .siglip_encoder import SigLipVisionTower


def build_vision_tower(config, **kwargs):
    name = getattr(config, "mm_vision_tower", None) or getattr(config, "vision_tower", None)
    if not isinstance(name, str) or "siglip" not in name.lower():
        raise ValueError("Select the SigLIP SO400M semantic encoder")
    if name == "google/siglip-so400m-patch14-384":
        name = "data/models/siglip-so400m-patch14-384"
    return SigLipVisionTower(name, vision_tower_cfg=config, **kwargs)
