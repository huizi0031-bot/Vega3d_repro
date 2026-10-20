"""CPU integration tests for Qwen3.5, SigLIP, Wan and checkpoint resume."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import save_file
from transformers import Qwen3_5ForCausalLM

from llava.model.language_model.llava_qwen import LlavaQwen35Config, LlavaQwen35ForCausalLM
from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionConfig, SigLipVisionModel
from llava.model.multimodal_generative_encoder.wan22_ti2v_encoder import Wan22TI2VOnlineEncoder
from wan_offline_tools.export_qwen35_language import export_language

torch.set_num_threads(1)


def tiny_config(**kwargs):
    values = dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                  head_dim=16, linear_key_head_dim=8, linear_value_head_dim=8,
                  linear_num_key_heads=2, linear_num_value_heads=2,
                  layer_types=['linear_attention']*3+['full_attention'],
                  max_position_embeddings=128, pad_token_id=0, eos_token_id=2,
                  tie_word_embeddings=True, use_cache=False,
                  rope_parameters={'rope_type':'default','rope_theta':10000,
                                   'partial_rotary_factor':0.5,'mrope_section':[1,1,2]})
    values.update(kwargs)
    config = LlavaQwen35Config(**values)
    config._attn_implementation = 'eager'
    return config


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SigLipVisionConfig(hidden_size=32, intermediate_size=64,
                                        num_hidden_layers=2, num_attention_heads=4,
                                        image_size=56, patch_size=14)
        self.encoder = SigLipVisionModel(self.config)

    def forward(self, images):
        return self.encoder(images, return_dict=True).last_hidden_state


class VariantTests(unittest.TestCase):
    def test_t5_import_does_not_initialize_cuda(self):
        import importlib
        with patch('torch.cuda._lazy_init', side_effect=AssertionError('CUDA initialized on import')):
            module = importlib.import_module('llava.model.multimodal_generative_encoder.wan22.modules.t5')
            importlib.reload(module)
        self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(module.T5EncoderModel.__init__.__defaults__[1], torch.device('cuda'))

    def test_actual_training_model_factory_loads_language_and_siglip(self):
        from llava.train.train_3d import get_model, ModelArguments, TrainingArguments
        from transformers import Qwen3_5TextConfig
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            language, vision = root/'Qwen3.5-2B-Language', root/'siglip-tiny'
            config = tiny_config().to_dict()
            config.pop('model_type',None)
            original = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config))
            original.save_pretrained(language, save_original_format=False)
            semantic = SigLipVisionModel(SigLipVisionConfig(hidden_size=32,intermediate_size=64,
                num_hidden_layers=2,num_attention_heads=4,image_size=56,patch_size=14))
            semantic.save_pretrained(vision)
            args = ModelArguments(model_name_or_path=str(language), vision_tower=str(vision),
                mm_projector_type='mlp2x_gelu',mm_patch_merge_type='spatial_unpad',
                use_generative_feature=True,generative_feature_source='offline',
                generative_feat_dim=3072,use_feat_fusion=True,feature_fusion_method='token_gated',
                ground_head_type='infonce')
            training = TrainingArguments(output_dir=str(root/'output'), use_cpu=True,
                attn_implementation='eager',report_to='none',gradient_checkpointing=False)
            loaded = get_model(args,training,{})
            self.assertEqual(loaded.config.model_type,'llava_qwen35')
            self.assertEqual(loaded.model.generative_projector[0].in_features,3072)
            torch.testing.assert_close(loaded.model.layers[0].linear_attn.in_proj_qkv.weight,
                                       original.model.layers[0].linear_attn.in_proj_qkv.weight)
            torch.testing.assert_close(loaded.model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight,
                                       semantic.vision_model.embeddings.patch_embedding.weight)
            self.assertTrue(all(torch.isfinite(p).all() for p in loaded.parameters()))
            trained = root/'saved-full'
            loaded.save_pretrained(trained)
            restored = LlavaQwen35ForCausalLM.from_pretrained(trained)
            self.assertTrue(restored.model.vision_tower.is_loaded)
            torch.testing.assert_close(restored.model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight,
                                       loaded.model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight)
            from llava.model.builder import load_pretrained_model
            class Tokenizer:
                def __len__(self):
                    return loaded.get_input_embeddings().weight.shape[0]
                def add_tokens(self, *args, **kwargs):
                    return 0
            with patch('llava.model.builder.AutoTokenizer.from_pretrained', return_value=Tokenizer()):
                _, inference, processor, _ = load_pretrained_model(
                    str(trained), None, 'qwen35-test', device_map={'':'cpu'},
                    torch_dtype='float32', attn_implementation='eager')
            self.assertIsNotNone(processor)
            self.assertIsNone(inference.__dict__['_foresight_teacher_runtime'])
            torch.testing.assert_close(inference.model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight,
                                       loaded.model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight)

    def test_native_chat_template_supervision_if_tokenizer_available(self):
        directory = os.environ.get('QWEN35_TEST_TOKENIZER')
        if not directory:
            self.skipTest('Set QWEN35_TEST_TOKENIZER to an official local tokenizer snapshot')
        from transformers import AutoTokenizer
        from llava.train.train_3d import preprocess_qwen35
        from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        tokenizer.add_tokens(['<ground>'])
        for answer in ['a wooden chair', '<ground>']:
            result = preprocess_qwen35([[{'from':'human','value':'<image>\nFind the chair.'},
                                         {'from':'gpt','value':answer}]], tokenizer, has_image=True)
            ids, labels = result['input_ids'][0], result['labels'][0]
            self.assertEqual((ids == IMAGE_TOKEN_INDEX).sum().item(), 1)
            supervised = labels[labels != IGNORE_INDEX]
            decoded = tokenizer.decode(supervised, skip_special_tokens=False)
            self.assertEqual(decoded, answer+'<|im_end|>')

    def test_real_trainer_step_and_complete_checkpoint_resume(self):
        from llava.train.train_3d import TrainingArguments
        from llava.train.llava_trainer import LLaVATrainer
        from transformers import Trainer
        from wan_offline_tools.train_entry import install_final_checkpoint
        class Data(torch.utils.data.Dataset):
            task_lengths = [(i%3,6) for i in range(20)]
            def __len__(self): return 20
            def __getitem__(self, index):
                ids = torch.tensor([4,5,6,7,8,9])
                return {'input_ids':ids,'labels':ids.clone()}
        def collate(rows):
            return {key:torch.stack([row[key] for row in rows]) for key in rows[0]}
        previous = Trainer.train
        previous_save = Trainer._save_checkpoint
        install_final_checkpoint(Trainer)
        try:
            with tempfile.TemporaryDirectory() as directory:
                def arguments(steps):
                    return TrainingArguments(output_dir=directory, use_cpu=True,
                        max_steps=steps, per_device_train_batch_size=1,
                        gradient_accumulation_steps=1, group_by_task_length=True,
                        logging_steps=1, save_steps=1, save_total_limit=1,
                        report_to='none', gradient_checkpointing=False,
                        dataloader_num_workers=0, disable_tqdm=True,
                        learning_rate=1e-5, weight_decay=0.0)
                trainer = LLaVATrainer(model=LlavaQwen35ForCausalLM(tiny_config()),
                    args=arguments(1), train_dataset=Data(), data_collator=collate)
                self.assertFalse(trainer.model_accepts_loss_kwargs)
                with patch.object(trainer, '_save_checkpoint', wraps=trainer._save_checkpoint) as saved:
                    trainer.train()
                    self.assertEqual(saved.call_count, 1)
                checkpoint = Path(directory)/'checkpoint-1'
                self.assertTrue((checkpoint/'optimizer.pt').is_file())
                self.assertTrue((checkpoint/'rng_state.pth').is_file())
                trainer = LLaVATrainer(model=LlavaQwen35ForCausalLM(tiny_config()),
                    args=arguments(2), train_dataset=Data(), data_collator=collate)
                with patch.object(trainer, '_save_checkpoint', wraps=trainer._save_checkpoint) as saved:
                    trainer.train(resume_from_checkpoint=str(checkpoint))
                    self.assertEqual(saved.call_count, 1)
                self.assertEqual(trainer.state.global_step, 2)
                self.assertTrue((Path(directory)/'checkpoint-2/optimizer.pt').is_file())
                self.assertTrue(all(torch.isfinite(torch.tensor(row['loss']))
                                    for row in trainer.state.log_history if 'loss' in row))
        finally:
            Trainer.train = previous
            Trainer._save_checkpoint = previous_save

    def test_language_loss_backward_generation_and_reload(self):
        torch.manual_seed(7)
        model = LlavaQwen35ForCausalLM(tiny_config())
        ids = torch.tensor([[4,5,6,7,8,9]])
        result = model(input_ids=ids, labels=ids, use_cache=False)
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        self.assertGreater(model.model.layers[0].linear_attn.in_proj_qkv.weight.grad.abs().sum().item(), 0)
        model.eval()
        reference = model(input_ids=ids).logits.detach()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = LlavaQwen35ForCausalLM.from_pretrained(directory).eval()
            torch.testing.assert_close(restored(input_ids=ids).logits, reference)
            self.assertEqual(restored.lm_head.weight.data_ptr(), restored.model.embed_tokens.weight.data_ptr())
        generated = model.generate(inputs=ids, max_new_tokens=2, do_sample=False, use_cache=True)
        self.assertGreaterEqual(generated.shape[-1], 1)

    def test_siglip_projectors_fusion_position_and_grounding_receive_gradients(self):
        config = tiny_config(mm_vision_tower='test-siglip', mm_hidden_size=32,
                             mm_projector_type='mlp2x_gelu', use_generative_feature=True,
                             generative_feature_source='offline', generative_feat_dim=3072,
                             generative_projector_type='mlp2x_gelu', use_feat_fusion=True,
                             feature_fusion_method='token_gated', ground_head_type='infonce',
                             ground_head_temperature=0.07, ground_token_ids=[10],
                             world_position_embedding_type='avg-discrete-sin3d')
        with patch('llava.model.llava_arch.build_vision_tower', return_value=TinyVision()):
            model = LlavaQwen35ForCausalLM(config)
        base = model.get_model()
        visual = base.mm_projector(base.vision_tower(torch.randn(1,3,56,56)))
        generative = base.generative_projector(torch.randn(1,16,3072))
        fused = base.feature_fusion(visual, generative)
        fused = fused+base.world_position_embedding(torch.randn(1,16,3))
        labels = torch.tensor([[4]*15+[10]])
        loss, scores = model.predict_box(inputs_embeds=fused, labels=labels,
                                         object_features=fused[0,:3], box_labels=[[1]])
        self.assertEqual(tuple(scores.shape), (4,))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for module in [base.vision_tower, base.mm_projector, base.generative_projector,
                       base.feature_fusion, model.ground_head_obj, model.ground_head_query,
                       base.layers[0]]:
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                                and p.grad.abs().sum()>0 for p in module.parameters()))
        self.assertIsNone(base.generative_vision_tower)

    def test_language_export_maps_exact_pretrained_weights_and_excludes_vision(self):
        model = LlavaQwen35ForCausalLM(tiny_config()).eval()
        state = {('model.language_model.'+k[len('model.'):]) if k.startswith('model.') else k:
                 value.detach().contiguous().clone() for k,value in model.state_dict().items()}
        state['model.visual.test.weight'] = torch.ones(2,2)
        state['mtp.test.weight'] = torch.ones(2,2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root/'source', root/'language'
            source.mkdir()
            (source/'config.json').write_text(json.dumps({'model_type':'qwen3_5',
                                                         'text_config':tiny_config().to_dict()}))
            save_file(state, str(source/'model.safetensors'))
            report = export_language(source, destination)
            self.assertGreater(report['language_tensor_count'], 0)
            index = json.loads((destination/'model.safetensors.index.json').read_text())['weight_map']
            self.assertFalse(any('visual' in k or 'mtp' in k or 'language_model' in k for k in index))
            loaded, info = Qwen3_5ForCausalLM.from_pretrained(destination, output_loading_info=True)
            self.assertFalse(info['missing_keys'])
            self.assertFalse(info['unexpected_keys'])
            torch.testing.assert_close(loaded.model.layers[0].linear_attn.in_proj_qkv.weight,
                                       model.model.layers[0].linear_attn.in_proj_qkv.weight)

    def test_wan22_real_dit_hook_and_spatial_packing_contract(self):
        # Real tiny official DiT; use an exact torch SDPA backend only for CPU.
        from types import SimpleNamespace
        from llava.model.multimodal_generative_encoder.wan22.modules import model as module
        from llava.model.multimodal_generative_encoder.wan22.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        def cpu_attention(q,k,v,k_lens=None,**kwargs):
            mask = None
            if k_lens is not None:
                mask = torch.arange(k.shape[1])[None,:] < k_lens[:,None]
                mask = mask[:,None,None,:]
            return nn.functional.scaled_dot_product_attention(
                q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=mask).transpose(1,2)
        tower = Wan22TI2VOnlineEncoder.__new__(Wan22TI2VOnlineEncoder)
        nn.Module.__init__(tower)
        tower.model = module.WanModel(model_type='ti2v', patch_size=(1,2,2), text_len=4,
                                     in_dim=48, out_dim=48, dim=32, ffn_dim=64,
                                     freq_dim=16, text_dim=16, num_heads=4, num_layers=2)
        tower.model.requires_grad_(False)
        tower.cfg = SimpleNamespace(dim=32)
        tower.param_dtype=torch.float32
        tower.vae_stride=(4,16,16);tower.patch_size=(1,2,2)
        tower.frame_width=tower.frame_height=64
        tower.feat_block_idx=1;tower.timestep=300;tower.shift=5.0
        tower.num_train_timesteps=1000;tower._scheduler_device=None
        tower.scheduler=FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1,use_dynamic_shifting=False)
        tower.prompt_context=torch.randn(4,16)
        # VAE dimensions are stubbed here; pretrained VAE/GPU numerical testing
        # needs actual checkpoint files and is separately reported as pending.
        tower.vae=SimpleNamespace(encode=lambda frames:[torch.randn(48,1,4,4) for _ in frames])
        with patch.object(module,'flash_attention',cpu_attention):
            feature=tower._forward_single_video(torch.rand(2,3,64,64),torch.device('cpu'))
        self.assertEqual(tuple(feature.shape),(2,32,14,14))
        self.assertFalse(feature.requires_grad)
        self.assertTrue(torch.isfinite(feature).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
