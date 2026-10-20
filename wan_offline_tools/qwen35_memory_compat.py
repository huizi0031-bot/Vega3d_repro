"""Opt-in, training-only selection of non-ignored causal-LM output positions.

The decoder still sees every token. The native LM head and native causal loss
compute the same supervised targets, without materializing ignored-token logits.
No tensors or optimizer state are offloaded. Grounding bypasses the LM forward.
"""
import functools
import torch
import torch.nn.functional as F


def install_supervised_logits_only():
    from transformers import Qwen3_5ForCausalLM

    native = Qwen3_5ForCausalLM.forward
    if getattr(native, '_vega_supervised_logits_only', False):
        return native

    @functools.wraps(native)
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, cache_position=None, logits_to_keep=0, **kwargs):
        sequence_length = (inputs_embeds.shape[1] if inputs_embeds is not None
                           else input_ids.shape[1] if input_ids is not None else None)
        if (self.training and labels is not None and labels.ndim == 2
                and labels.shape[1] == sequence_length
                and isinstance(logits_to_keep, int) and logits_to_keep == 0
                and kwargs.get('shift_labels') is None):
            # Native ForCausalLMLoss uses precisely this one-token shift.
            shifted = F.pad(labels, (0, 1), value=-100)[..., 1:]
            selected = (shifted != -100).any(dim=0).nonzero(as_tuple=True)[0]
            if 0 < selected.numel() < labels.shape[1]:
                logits_to_keep = selected
                kwargs['shift_labels'] = shifted.index_select(1, selected)
                self._vega_supervised_logits_stats = {
                    'sequence_positions':labels.shape[1],
                    'selected_positions':selected.numel(),
                    'batch_size':labels.shape[0],
                }
                if not getattr(self, '_vega_supervised_logits_reported', False):
                    print('[Qwen3.5] Native supervised logits:', selected.numel(),
                          '/', labels.shape[1], 'positions; decoder context and causal loss preserved.',
                          flush=True)
                    self._vega_supervised_logits_reported = True
        return native(self, input_ids=input_ids, attention_mask=attention_mask,
                      position_ids=position_ids, past_key_values=past_key_values,
                      inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                      cache_position=cache_position, logits_to_keep=logits_to_keep, **kwargs)

    forward._vega_supervised_logits_only = True
    Qwen3_5ForCausalLM.forward = forward
    print('[Qwen3.5] Supervised-logits memory adapter enabled; GPU state placement unchanged.', flush=True)
    return native
