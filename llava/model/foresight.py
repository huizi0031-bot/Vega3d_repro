"""Continuous foresight slots, inserted after the prompt and before answer tokens."""
from dataclasses import dataclass

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.constants import IGNORE_INDEX


@dataclass
class ForesightCausalLMOutput(CausalLMOutputWithPast):
    qa_loss: torch.Tensor | None = None
    video_loss: torch.Tensor | None = None
    foresight_embeddings: torch.Tensor | None = None


class ForesightSlots(nn.Module):
    def __init__(self, hidden_size, count, context_dim=4096):
        super().__init__()
        if count < 1 or count > 512:
            raise ValueError("foresight_num_queries must be between 1 and 512")
        self.queries = nn.Parameter(torch.empty(count, hidden_size))
        nn.init.normal_(self.queries, std=0.02)
        self.to_wan = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, context_dim))

    def forward(self, embeddings, attention_mask=None, labels=None, **kwargs):
        # Use Module.__call__ in the LM so ZeRO-3 can gather the query parameter
        # around this forward. Calling only a custom .insert method skips hooks.
        return self.insert(embeddings, attention_mask, labels, **kwargs)

    def insert(self, embeddings, attention_mask=None, labels=None, *, max_length=None,
               padding_side="right"):
        """Return packed inputs and slot indices; labels never enter the slot prefix.

        Each supervised row must be a single answer at the end of its prompt.
        Inference appends the same slots to the complete generation prompt.
        No truncation is performed here: exceeding the context is an error.
        """
        if attention_mask is None:
            attention_mask = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
        if padding_side not in ("left", "right"):
            raise ValueError("Unsupported padding side")
        rows, targets, starts = [], [], []
        query = self.queries.to(embeddings)
        for i, mask in enumerate(attention_mask.bool()):
            row = embeddings[i, mask]
            target = labels[i, mask] if labels is not None else None
            cut = len(row)
            if target is not None:
                supervised = (target != IGNORE_INDEX).nonzero().flatten()
                if not len(supervised):
                    raise ValueError("No answer tokens remain; refusing to train foresight on an empty answer")
                cut = int(supervised[0])
                if (target[cut:int(supervised[-1])+1] == IGNORE_INDEX).any():
                    raise ValueError("Foresight requires a single contiguous answer suffix")
            if max_length is not None and len(row) + len(query) > max_length:
                raise ValueError("Prompt + answer + foresight queries exceeds model_max_length")
            rows.append(torch.cat((row[:cut], query, row[cut:])))
            starts.append(cut)
            if target is not None:
                targets.append(torch.cat((target[:cut], target.new_full((len(query),), IGNORE_INDEX), target[cut:])))
        size = max(map(len, rows))
        packed, masks, positions, slots, packed_targets = [], [], [], [], []
        for i, row in enumerate(rows):
            n = len(row)
            offset = size - n if padding_side == "left" else 0
            pad = row.new_zeros((size - n, row.shape[-1]))
            packed.append(torch.cat((pad, row)) if offset else torch.cat((row, pad)))
            mask = torch.zeros(size, device=row.device, dtype=torch.bool)
            mask[offset:offset+n] = True
            masks.append(mask)
            positions.append((mask.long().cumsum(0) - 1).clamp_min(0))
            slots.append(torch.arange(starts[i]+offset, starts[i]+offset+len(query), device=row.device))
            if labels is not None:
                target = labels.new_full((size,), IGNORE_INDEX)
                target[offset:offset+n] = targets[i]
                packed_targets.append(target)
        return (torch.stack(packed), torch.stack(masks), torch.stack(positions),
                torch.stack(packed_targets) if labels is not None else None, torch.stack(slots))


def answer_loss(lm_head, hidden, labels):
    """Causal CE at supervised positions only, without projecting all visual tokens."""
    shifted = labels[:, 1:]
    selected = shifted != IGNORE_INDEX
    if not selected.any():
        raise ValueError("No shifted answer supervision")
    logits = lm_head(hidden[:, :-1][selected]).float()
    return nn.functional.cross_entropy(logits, shifted[selected]), logits
