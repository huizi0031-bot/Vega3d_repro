# Copyright 2024 Hao Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VEGA-3D on the Qwen3.5 text backbone: multimodal causal LM and 3D grounding heads."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3_5TextConfig, Qwen3_5TextModel, Qwen3_5ForCausalLM
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.model.foresight import ForesightSlots, ForesightCausalLMOutput, answer_loss


class LlavaQwen35Config(Qwen3_5TextConfig):
    model_type = "llava_qwen35"


class LlavaQwen35Model(LlavaMetaModel, Qwen3_5TextModel):
    config_class = LlavaQwen35Config

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)


class LlavaQwen35ForCausalLM(Qwen3_5ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwen35Config
    # Preserve the Qwen3.5 variant's per-microbatch loss normalization, including grounding.
    accepts_loss_kwargs = False

    def __init__(self, config):
        # Initialize the HF pretrained-model base without instantiating a
        # redundant Qwen3.5 backbone. The backbone must also be LlavaMetaModel.
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = LlavaQwen35Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.foresight = None
        if getattr(config, "foresight_enabled", False):
            self.foresight = ForesightSlots(config.hidden_size, config.foresight_num_queries,
                                            getattr(config, "foresight_context_dim", 4096))
        self.__dict__["_foresight_teacher_runtime"] = None

        # Only one grounding-head initializer. All dimensions follow Qwen3.5's
        # config.hidden_size (2048 for the current Qwen3.5-2B checkpoint).
        self.ground_head_type = getattr(config, "ground_head_type", None)
        hidden_size = config.hidden_size

        if self.ground_head_type == "mlp":
            self.ground_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
            )
        elif self.ground_head_type == "score":
            self.ground_head_temperature = float(
                getattr(config, "ground_head_temperature", 0.07)
            )
            self.ground_head_obj = nn.Sequential(
                nn.Linear(hidden_size, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
            )
            self.ground_head_query = nn.Sequential(
                nn.Linear(hidden_size, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
            )
            self.ground_head_score = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                nn.Linear(1024, 1),
            )
        elif self.ground_head_type == "infonce":
            self.ground_head_temperature = float(
                getattr(config, "ground_head_temperature", 0.07)
            )
            if self.ground_head_temperature <= 0:
                raise ValueError("ground_head_temperature must be positive")
            self.ground_head_zero_target = nn.Parameter(torch.randn(hidden_size))
            self.ground_head_obj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
            )
            self.ground_head_query = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
            )
        elif self.ground_head_type is not None:
            raise NotImplementedError(
                f"Unsupported grounding head: {self.ground_head_type!r}"
            )

        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, ForesightSlots):
            nn.init.normal_(module.queries, mean=0., std=self.config.initializer_range)
        if isinstance(module, LlavaQwen35Model) and hasattr(module, 'image_newline'):
            # A new bridge from a language-only checkpoint has no pretrained
            # image-newline vector. Initialize it explicitly, not with empty memory.
            nn.init.normal_(module.image_newline, mean=0., std=self.config.hidden_size**-0.5)

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = None,
        dpo_forward: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        video_dict=None,
        use_object_proposals: bool = False,
        box_labels=None,
        foresight_pixels=None,
        foresight_prepared: bool = False,
        return_foresight_embeddings: bool = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Route ordinary LM, DPO and proposal-based 3D grounding batches."""
        object_features = None
        object_boxes = None
        if modalities is None:
            modalities = ["image"]

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                object_features,
                object_boxes,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                modalities,
                image_sizes,
                video_dict,
                use_object_proposals=use_object_proposals,
            )

        if self.foresight is not None and (use_object_proposals or dpo_forward):
            raise ValueError("Foresight supports single-turn text generation; object-proposal grounding and DPO are not connected")

        if self.foresight is not None and not foresight_prepared and past_key_values is None:
            if position_ids is not None and position_ids.ndim > 2:
                raise ValueError("Foresight uses 1D sequence positions; 3D mRoPE is unsupported")
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds, attention_mask, position_ids, labels, slots = self.foresight(
                inputs_embeds, attention_mask, labels,
                max_length=getattr(self.config, "tokenizer_model_max_length", None),
                padding_side=getattr(self.config, "tokenizer_padding_side", "right"))
            return self._forward_foresight(inputs_embeds, attention_mask, position_ids, labels,
                slots, foresight_pixels, use_cache, output_attentions, output_hidden_states,
                return_dict, return_foresight_embeddings, logits_to_keep)

        if foresight_pixels is not None:
            raise ValueError("Teacher pixels are accepted only on an unprepared foresight text forward")

        if use_object_proposals:
            if object_features is None:
                raise ValueError(
                    "Grounding needs object_features from multimodal "
                    "preprocessing; provide video and object proposals."
                )
            return self.predict_box(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                object_features=object_features,
                object_boxes=object_boxes,
                box_labels=box_labels,
                cache_position=cache_position,
            )

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
            )
            logits = self.lm_head(outputs.last_hidden_state)
            return logits, labels

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    def _forward_foresight(self, embeddings, mask, positions, labels, slots, pixels,
                          use_cache, output_attentions, output_hidden_states,
                          return_dict, return_embeddings, logits_to_keep):
        if labels is not None and use_cache:
            raise ValueError("Foresight training must use use_cache=False")
        outputs = self.model(inputs_embeds=embeddings, attention_mask=mask,
            position_ids=positions, use_cache=bool(use_cache) if labels is None else False,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=True)
        hidden = outputs.last_hidden_state
        z = hidden.gather(1, slots[..., None].expand(-1, -1, hidden.shape[-1]))
        qa_loss = video_loss = loss = None
        if labels is not None:
            qa_loss, logits = answer_loss(self.lm_head, hidden, labels)
            loss = qa_loss
        else:
            selected = hidden[:, -logits_to_keep:] if isinstance(logits_to_keep, int) and logits_to_keep else hidden
            logits = self.lm_head(selected)
        weight = float(getattr(self.config, "foresight_video_loss_weight", 0.))
        if self.training and labels is not None and weight > 0:
            if pixels is None:
                raise ValueError("Video supervision is enabled but foresight_pixels are missing")
            teacher = self.__dict__["_foresight_teacher_runtime"]
            if teacher is None:
                from llava.model.multimodal_generative_encoder.wan22_foresight import Wan22ForesightTeacher
                teacher = Wan22ForesightTeacher(self.config, z.device)
                self.__dict__["_foresight_teacher_runtime"] = teacher
            video_loss = teacher(self.foresight.to_wan(z), pixels)
            loss = loss + weight*video_loss
        elif pixels is not None:
            raise ValueError("Teacher pixels supplied while training video loss is disabled")
        # Keep E01 projector in the distributed graph, without affecting QA.
        if loss is not None and video_loss is None:
            loss = loss + 0.*self.foresight.to_wan(z).sum()
        result = ForesightCausalLMOutput(loss=loss, logits=logits,
            past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states,
            attentions=outputs.attentions, qa_loss=qa_loss, video_loss=video_loss,
            foresight_embeddings=z if return_embeddings else None)
        return result if return_dict is not False else result.to_tuple()

    def predict_box(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        object_features=None,
        object_boxes=None,
        box_labels=None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Ground <ground> query against N object proposals.

        In InfoNCE mode, scores has N+1 entries; the last is the
        learnable 'no object' target. This method must keep autograd on.
        """
        if self.ground_head_type not in ("mlp", "score", "infonce"):
            raise ValueError(f"Grounding head is not initialized: {self.ground_head_type}")
        if object_features is None or object_features.ndim != 2:
            raise ValueError("object_features must have shape [N, hidden_size]")
        if object_features.size(-1) != self.config.hidden_size:
            raise ValueError(
                f"object_features last dim must be {self.config.hidden_size}, "
                f"got {object_features.size(-1)}"
            )
        if labels is None or labels.ndim != 2:
            raise ValueError("Grounding needs labels with shape [B, L]")
        ground_ids = getattr(self.config, "ground_token_ids", None)
        if ground_ids is None or len(ground_ids) != 1:
            raise ValueError("Config must contain exactly one ground_token_id")
        ground_id = int(ground_ids[0])
        loc = (labels == ground_id).nonzero(as_tuple=False)
        if loc.shape[0] != 1:
            raise ValueError(
                f"Expected exactly one supervised <ground> token; got {loc.shape[0]}"
            )
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds is required")

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )
        hidden_states = outputs.last_hidden_state
        batch_idx, token_idx = map(int, loc[0].tolist())
        if token_idx >= hidden_states.shape[1]:
            raise ValueError("Grounding label position exceeds hidden-state length")
        ground_hidden = hidden_states[batch_idx, token_idx].unsqueeze(0)

        # Match the trainable head's parameters rather than relying on autocast.
        if self.ground_head_type == "mlp":
            head_parameter = next(self.ground_head.parameters())
        else:
            head_parameter = next(self.ground_head_obj.parameters())
        obj = object_features.to(
            device=head_parameter.device, dtype=head_parameter.dtype
        )
        query = ground_hidden.to(
            device=head_parameter.device, dtype=head_parameter.dtype
        )
        num_objects = obj.shape[0]

        if self.ground_head_type == "mlp":
            if num_objects == 0:
                raise ValueError("mlp grounding requires object proposals")
            query_feat = self.ground_head(query)
            scores = (obj * query_feat).sum(dim=-1)
        elif self.ground_head_type == "score":
            if num_objects == 0:
                raise ValueError("score grounding requires object proposals")
            object_proj = self.ground_head_obj(obj)
            query_proj = self.ground_head_query(query)
            scores = self.ground_head_score(object_proj * query_proj).squeeze(-1)
        else:
            zero_target = self.ground_head_zero_target.to(
                device=obj.device, dtype=obj.dtype
            ).unsqueeze(0)
            object_proj = self.ground_head_obj(torch.cat([obj, zero_target], dim=0))
            query_proj = self.ground_head_query(query)
            object_proj = F.normalize(object_proj.float(), dim=-1)
            query_proj = F.normalize(query_proj.float(), dim=-1)
            scores = (object_proj @ query_proj.T).squeeze(-1)

        loss = None
        if box_labels is not None:
            if len(box_labels) != 1:
                raise ValueError("Grounding currently supports batch size 1")
            indices = [int(x) for x in box_labels[0]]
            if self.ground_head_type == "infonce":
                if not indices:
                    indices = [num_objects]  # no-object target
                indices = [num_objects if i == -1 else i for i in indices]
                indices = sorted(set(indices))
                if any(i < 0 or i > num_objects for i in indices):
                    raise ValueError("Invalid grounding box index")
                scaled = scores.float() / self.ground_head_temperature
                positives = torch.tensor(indices, dtype=torch.long, device=scores.device)
                loss = torch.logsumexp(scaled, dim=0) - torch.logsumexp(
                    scaled[positives], dim=0
                )
            else:
                if any(i < 0 or i >= num_objects for i in indices):
                    raise ValueError("Invalid grounding box index")
                target = torch.zeros_like(scores)
                if indices:
                    target[indices] = 1
                weights = torch.ones_like(scores)
                if indices:
                    weights[indices] *= (num_objects - len(set(indices))) / len(set(indices))
                loss = (F.binary_cross_entropy_with_logits(
                    scores, target, reduction="none"
                ) * weights).mean()

        return loss, scores

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """Generate from text or from VEGA multimodal image/video tokens."""
        video_dict = kwargs.pop("video_dict", None)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if modalities is None:
            modalities = ["image"]

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                modalities,
                image_sizes=image_sizes,
                video_dict=video_dict,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        if self.foresight is not None:
            if kwargs.get("foresight_pixels") is not None:
                raise ValueError("Generation never accepts teacher target frames")
            inputs_embeds, attention_mask, position_ids, _, _ = self.foresight(
                inputs_embeds, attention_mask, max_length=getattr(self.config, "tokenizer_model_max_length", None),
                padding_side=getattr(self.config, "tokenizer_padding_side", "right"))
            kwargs["foresight_prepared"] = True
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        video_dict = kwargs.pop("video_dict", None)
        foresight_prepared = kwargs.pop("foresight_prepared", False)
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            model_inputs["images"] = images
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes
        if video_dict is not None:
            model_inputs["video_dict"] = video_dict
        model_inputs["foresight_prepared"] = foresight_prepared
        return model_inputs


AutoConfig.register("llava_qwen35", LlavaQwen35Config)
AutoModelForCausalLM.register(LlavaQwen35Config, LlavaQwen35ForCausalLM)
