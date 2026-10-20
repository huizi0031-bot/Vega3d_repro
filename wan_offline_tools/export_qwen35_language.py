#!/usr/bin/env python3
"""Extract official Qwen3.5 text weights, explicitly remapping VL checkpoint keys.

No random replacement backbone, Qwen vision encoder, merger or MTP is exported.
Safetensors shards are read one at a time on CPU; no GPU is needed.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file


def _export_language_into(source, destination):
    source, destination = Path(source), Path(destination)
    config = json.loads((source/'config.json').read_text())
    if config.get('model_type') != 'qwen3_5' or 'text_config' not in config:
        raise ValueError('Source must be an official Qwen3.5 multimodal checkpoint')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Refusing to overwrite a populated language export: '+str(destination))
    text = dict(config['text_config'])
    text.update(model_type='qwen3_5_text', architectures=['Qwen3_5ForCausalLM'])
    if int(text.get('hidden_size', 0)) <= 0:
        raise ValueError('Invalid source text configuration')
    shards = sorted(source.glob('*.safetensors'))
    if not shards:
        raise FileNotFoundError('No official safetensors weights in '+str(source))
    destination.mkdir(parents=True, exist_ok=True)
    if text.get('tie_word_embeddings', False):
        locations = {}
        for shard in shards:
            with safe_open(str(shard), framework='pt', device='cpu') as reader:
                for name in ['model.language_model.embed_tokens.weight', 'lm_head.weight']:
                    if name in reader.keys(): locations[name] = shard
        if 'lm_head.weight' in locations:
            import torch
            with safe_open(str(locations['lm_head.weight']), framework='pt', device='cpu') as a, \
                 safe_open(str(locations['model.language_model.embed_tokens.weight']), framework='pt', device='cpu') as b:
                if not torch.equal(a.get_tensor('lm_head.weight'), b.get_tensor('model.language_model.embed_tokens.weight')):
                    raise ValueError('Source declares tied embeddings but has unequal LM-head weights')
    weight_map, total_size, exported = {}, 0, []
    for index, shard in enumerate(shards):
        tensors = {}
        with safe_open(str(shard), framework='pt', device='cpu') as reader:
            for name in reader.keys():
                if name.startswith('model.language_model.'):
                    target = 'model.'+name[len('model.language_model.'):]
                elif name == 'lm_head.weight':
                    if text.get('tie_word_embeddings', False):
                        continue
                    target = name
                else:
                    continue
                if target in weight_map:
                    raise ValueError('Duplicate text weight '+target)
                tensor = reader.get_tensor(name).contiguous()
                tensors[target] = tensor
                total_size += tensor.numel()*tensor.element_size()
        if not tensors:
            continue
        filename = f'language-{index+1:05d}.safetensors'
        save_file(tensors, str(destination/filename), metadata={'format':'pt'})
        weight_map.update({name: filename for name in tensors})
        exported.append({'source_shard': shard.name, 'output_shard': filename, 'tensors':len(tensors)})
        del tensors
    needed = ['model.embed_tokens.weight', 'model.norm.weight']
    needed += [f'model.layers.{i}.mlp.down_proj.weight' for i in range(text['num_hidden_layers'])]
    if not text.get('tie_word_embeddings', False):
        needed.append('lm_head.weight')
    if any(name not in weight_map for name in needed):
        raise ValueError('Incomplete language weights: '+str([n for n in needed if n not in weight_map]))
    (destination/'config.json').write_text(json.dumps(text,indent=2)+'\n')
    (destination/'model.safetensors.index.json').write_text(
        json.dumps({'metadata':{'total_size':total_size}, 'weight_map':weight_map},indent=2)+'\n')
    names = ['tokenizer.json','tokenizer_config.json','special_tokens_map.json',
             'added_tokens.json','vocab.json','merges.txt','tokenizer.model',
             'chat_template.jinja','generation_config.json']
    for name in names:
        if (source/name).is_file():
            shutil.copy2(source/name, destination/name)
    report = {'source':str(source.resolve()), 'source_model_type':config['model_type'],
              'exported_model_type':'qwen3_5_text', 'hidden_size':text['hidden_size'],
              'language_tensor_count':len(weight_map), 'language_weight_bytes':total_size,
              'excluded':['model.visual.*','mtp.*'], 'shards':exported}
    (destination/'language_export.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def export_language(source, destination):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Refusing to overwrite a populated language export: '+str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.qwen35-language-', dir=destination.parent) as temporary:
        staged = Path(temporary)/'export'
        report = _export_language_into(source, staged)
        if destination.exists(): destination.rmdir()
        staged.rename(destination)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(export_language(args.source, args.output),indent=2))
