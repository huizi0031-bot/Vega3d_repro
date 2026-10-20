def build_generative_vision_tower(config):
    name = getattr(config, "generative_vision_tower_type", "wan22_ti2v_online")
    if name != "wan22_ti2v_online":
        raise ValueError("Only Wan2.2 TI2V-5B feature extraction is supported")
    from .wan22_ti2v_encoder import Wan22TI2VOnlineEncoder
    return Wan22TI2VOnlineEncoder(config)
