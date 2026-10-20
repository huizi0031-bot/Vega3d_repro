"""Move ScanQA visual inputs to the model device without changing integer tensors."""
from typing import Dict
import torch


def move_video_dict_to_device(video_dict: Dict, device, floating_dtype: torch.dtype = torch.bfloat16):
    if "images" not in video_dict:
        raise KeyError("`images` is required in video_dict")

    image_tensors = video_dict.pop("images")
    if torch.is_floating_point(image_tensors):
        image_tensors = image_tensors.to(device=device, dtype=floating_dtype)
    else:
        image_tensors = image_tensors.to(device=device)

    for key, value in list(video_dict.items()):
        if not torch.is_tensor(value):
            continue
        if torch.is_floating_point(value):
            video_dict[key] = value.to(device=device, dtype=floating_dtype)
        else:
            video_dict[key] = value.to(device=device)

    return image_tensors, video_dict
