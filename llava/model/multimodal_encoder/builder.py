import os
from .siglip_encoder import SigLipVisionTower

# from .eva_clip.eva_clip_encoder import EvaClipVisionTower
# from .dev_eva_clip.eva_vit import EvaViTWrapper

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = (
        getattr(vision_tower_cfg, "mm_vision_tower", None)
        or getattr(vision_tower_cfg, "vision_tower", None)
    )

    if not isinstance(vision_tower, str) or not vision_tower:
        raise ValueError(
            f"Invalid vision tower: {vision_tower!r}"
        )

    use_s2 = getattr(vision_tower_cfg, "s2", False)

    # 先识别 SigLIP，避免本地路径被下面的 CLIP 分支截获。
    if "siglip" in vision_tower.lower():
        model_name = vision_tower

        if vision_tower == "google/siglip-so400m-patch14-384":
            model_name = "data/models/siglip-so400m-patch14-384"

        return SigLipVisionTower(
            model_name,
            vision_tower_cfg=vision_tower_cfg,
            **kwargs,
        )

    raise ValueError("This variant preserves the SigLIP semantic encoder; select a SigLIP checkpoint")

    is_absolute_path_exists = os.path.exists(vision_tower)

    if (
        is_absolute_path_exists
        or vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
        or "ShareGPT4V" in vision_tower
    ):
        if use_s2:
            return CLIPVisionTowerS2(
                vision_tower,
                args=vision_tower_cfg,
                **kwargs,
            )

        return CLIPVisionTower(
            vision_tower,
            args=vision_tower_cfg,
            **kwargs,
        )

    elif vision_tower.startswith("hf:"):
        return HFVisionTower(
            vision_tower,
            args=vision_tower_cfg,
            **kwargs,
        )

    elif vision_tower == "imagebind_huge":
        return ImageBindWrapper(
            vision_tower,
            args=vision_tower_cfg,
            **kwargs,
        )

    elif vision_tower.startswith("open_clip_hub"):
        return OpenCLIPVisionTower(
            vision_tower,
            args=vision_tower_cfg,
            **kwargs,
        )

    raise ValueError(f"Unknown vision tower: {vision_tower}")