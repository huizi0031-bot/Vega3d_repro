#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import json
import os
import torch
from transformers import AutoConfig, AutoTokenizer
from llava.model.language_model.llava_qwen import LlavaQwen35Config, LlavaQwen35ForCausalLM
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.utils import rank0_print


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False,
                          load_4bit=False, device_map="auto", torch_dtype="float16",
                          attn_implementation="sdpa", customized_config=None,
                          overwrite_config=None, **kwargs):
    if model_base is not None or load_8bit or load_4bit:
        raise ValueError("Use a full BF16/FP16 Qwen3.5 multimodal checkpoint")
    kwargs.pop("multimodal", None)
    with open(os.path.join(model_path, "config.json")) as stream:
        saved = json.load(stream)
    if saved.get("model_type") != "llava_qwen35":
        raise ValueError("Expected a saved llava_qwen35 multimodal checkpoint")
    kwargs["device_map"] = device_map
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if torch_dtype not in dtypes:
        raise ValueError("Unsupported model dtype: " + str(torch_dtype))
    kwargs["torch_dtype"] = dtypes[torch_dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    if customized_config is not None:
        base_config = customized_config
    else:
        base_config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
        )

    if isinstance(base_config, LlavaQwen35Config):
        llava_cfg = base_config
    else:
        config_dict = base_config.to_dict()
        config_dict.pop("model_type", None)
        llava_cfg = LlavaQwen35Config.from_dict(config_dict)

    if overwrite_config is not None:
        rank0_print(f"Overwriting config with {overwrite_config}")
        for key, value in overwrite_config.items():
            setattr(llava_cfg, key, value)

    if getattr(llava_cfg, "mm_vision_tower", None) is None:
        raise ValueError(
            "VEGA-Qwen3.5 inference requires a multimodal "
            "configuration containing mm_vision_tower. "
            "The original Qwen3.5-2B-Language checkpoint alone "
            "is not a trained VEGA-3D checkpoint."
        )

    # Avoid passing config twice.
    kwargs.pop("config", None)

    model = LlavaQwen35ForCausalLM.from_pretrained(
        model_path,
        config=llava_cfg,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        local_files_only=True,
        **kwargs,
    )
    rank0_print(f"Loaded model: {type(model).__name__}")
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    if hasattr(model.config, "text_config") and isinstance(model.config.text_config, dict):
        class _DummyTextConfig:
            pass
        dummy = _DummyTextConfig()
        if hasattr(model.config, "vocab_size"):
            dummy.vocab_size = model.config.vocab_size
        else:
            dummy.vocab_size = model.get_input_embeddings().weight.shape[0]
        model.config.text_config = dummy
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != "auto":
        vision_tower.to(device=model.device, dtype=model.dtype)
    image_processor = vision_tower.image_processor
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048
    return tokenizer, model, image_processor, context_len
