#!/usr/bin/env python3
"""Export an explicitly versioned empty-text condition using Wan2.2's UMT5."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from llava.model.multimodal_generative_encoder.wan22.configs.wan_ti2v_5B import ti2v_5B
from llava.model.multimodal_generative_encoder.wan22.modules.t5 import T5EncoderModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', default='data/models/Wan2.2-TI2V-5B')
    parser.add_argument('--output', default='llava/model/multimodal_generative_encoder/wan22_prompt_embedding.pt')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    root = Path(args.checkpoint_dir)
    encoder = T5EncoderModel(text_len=ti2v_5B.text_len, dtype=ti2v_5B.t5_dtype,
                             device=torch.device(args.device),
                             checkpoint_path=str(root/ti2v_5B.t5_checkpoint),
                             tokenizer_path=str(root/ti2v_5B.t5_tokenizer), shard_fn=None)
    with torch.inference_mode():
        context = encoder([''], torch.device(args.device))[0].detach().cpu().contiguous()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError('Refusing to overwrite prompt cache: '+str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_family':'wan2.2-ti2v-5B','prompt':'','context':context}, target)
    print('Saved Wan2.2 empty-text embedding:', target, tuple(context.shape))


if __name__ == '__main__':
    main()
