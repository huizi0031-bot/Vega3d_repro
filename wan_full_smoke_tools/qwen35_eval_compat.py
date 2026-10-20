"""Standalone Qwen3.5 prompt adapter; repository evaluation algorithms stay unchanged."""
import ast
import copy
from pathlib import Path
from typing import Dict

import torch
import transformers
from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX


def _training_preprocessor():
    path = Path(__file__).resolve().parents[1] / 'llava/train/train_3d.py'
    if not path.is_file():
        # The copied evaluation entry points live under tools/runs/<name>/eval/.
        for parent in Path(__file__).resolve().parents:
            if (parent / 'llava/train/train_3d.py').is_file():
                path = parent / 'llava/train/train_3d.py'
                break
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'preprocess_qwen35']
    if len(nodes) != 1:
        raise RuntimeError('Expected the existing Qwen3.5 training preprocessor')
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = dict(globals())
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['preprocess_qwen35']


def preprocess_grounding(sources, tokenizer, has_image=False, **kwargs):
    result = _training_preprocessor()([sources], tokenizer, has_image=has_image)
    return result['input_ids'], result['labels']


def preprocess_generation(sources, tokenizer, has_image=False, max_len=2048,
                          system_message='You are a helpful assistant.'):
    roles = {'human': 'user', 'gpt': 'assistant', 'user': 'user', 'assistant': 'assistant', 'system': 'system'}
    messages = [{'role': 'system', 'content': system_message}]
    for sentence in sources:
        if sentence.get('value') is None:
            if roles[sentence['from']] != 'assistant':
                raise ValueError('Only the assistant generation placeholder may be empty')
            continue
        messages.append({'role': roles[sentence['from']], 'content': sentence['value']})
    local = copy.deepcopy(tokenizer)
    local.add_tokens(['<image>'], special_tokens=True)
    image_id = local.convert_tokens_to_ids('<image>')
    prompt = local.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    ids = local(prompt, add_special_tokens=False)['input_ids']
    ids = [IMAGE_TOKEN_INDEX if i == image_id else i for i in ids]
    expected = sum(m['content'].count('<image>') for m in messages)
    if ids.count(IMAGE_TOKEN_INDEX) != expected or (has_image and expected == 0):
        raise ValueError('Qwen3.5 image placeholder alignment failed')
    return torch.tensor([ids], dtype=torch.long)
