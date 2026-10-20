"""CPU-only architecture contracts. No pretrained teacher or optimizer steps."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from llava.model.foresight import ForesightSlots, answer_loss
from llava.model.language_model.llava_qwen import LlavaQwen35ForCausalLM
from llava.model.multimodal_generative_encoder.wan22_foresight import make_flow_batch, conditional_flow_loss
from llava.model.multimodal_generative_encoder.wan22.modules.model import WanModel
from llava.model.multimodal_generative_encoder.wan22_ti2v_encoder import masked_sdpa
from llava.train.foresight_data import load_teacher_pixels, validate_pair, TARGET_MODE

# Reuse the existing tiny architecture definitions, without invoking its training test.
spec = importlib.util.spec_from_file_location("variant_test_helpers", Path(__file__).with_name("test_qwen35_wan22.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
torch.set_num_threads(1)


def config(**kwargs):
    return helpers.tiny_config(foresight_enabled=True, foresight_num_queries=4,
        foresight_context_dim=64, foresight_video_loss_weight=kwargs.pop("weight", 0.),
        tokenizer_model_max_length=128, **kwargs)


def tiny_teacher():
    model = WanModel(model_type="ti2v", patch_size=(1,2,2), text_len=8,
        in_dim=48, out_dim=48, dim=32, ffn_dim=64, freq_dim=16,
        text_dim=64, num_heads=2, num_layers=2)
    # A pretrained Wan head is nonzero; native scratch construction zeros it.
    nn.init.normal_(model.head.head.weight, std=0.02)
    return model.eval().requires_grad_(False)


class ForesightTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)

    def test_slot_boundaries_left_right_padding_and_context_overflow(self):
        slots = ForesightSlots(8, 3, 16)
        embeddings = torch.randn(2, 6, 8)
        mask = torch.tensor([[1,1,1,1,1,1], [0,0,1,1,1,1]])
        labels = torch.tensor([[-100,-100,-100,4,5,-100],[-100,-100,-100,-100,6,7]])
        for side in ["left", "right"]:
            packed, valid, positions, targets, indices = slots.insert(embeddings, mask, labels, padding_side=side)
            self.assertTrue((targets.gather(1, indices) == -100).all())
            self.assertEqual(targets[targets != -100].tolist(), [4,5,6,7])
            self.assertEqual(valid.sum(1).tolist(), [9,7])
            self.assertTrue((positions[valid] >= 0).all())
        with self.assertRaisesRegex(ValueError, "exceeds"):
            slots.insert(embeddings, mask, labels, max_length=8)
        bad = labels.clone(); bad[0, 4] = -100; bad[0, 5] = 8
        with self.assertRaisesRegex(ValueError, "contiguous"):
            slots.insert(embeddings, mask, bad)

    def test_video_mask_velocity_and_clean_anchor(self):
        clean = torch.randn(2, 48, 2, 4, 6)
        noise = torch.randn_like(clean)
        u = torch.tensor([0.2,0.7])
        x, velocity, mask, times = make_flow_batch(clean, (1,2,2), noise=noise, u=u)
        sigma = 5*u/(1+4*u)
        torch.testing.assert_close(x[:,:,0], clean[:,:,0])
        torch.testing.assert_close(x[:,:,1], (1-sigma[:,None,None,None])*clean[:,:,1]+sigma[:,None,None,None]*noise[:,:,1])
        torch.testing.assert_close(velocity, noise-clean)
        self.assertTrue((mask[:,:,0] == 0).all())
        self.assertTrue((times[:,:6] == 0).all())
        torch.testing.assert_close(times[:,6:], (sigma[:,None]*1000).expand(-1,6))

    def test_real_qwen_answer_cannot_leak_into_foresight_and_qa_reaches_queries(self):
        model = LlavaQwen35ForCausalLM(config()).eval()
        ids = torch.tensor([[4,5,6,7,8,9]])
        labels = torch.tensor([[-100,-100,-100,-100,8,9]])
        result = model(input_ids=ids, labels=labels, use_cache=False, return_foresight_embeddings=True)
        changed = ids.clone(); changed[:,4:] = torch.tensor([11,12])
        other = model(input_ids=changed, labels=labels, use_cache=False, return_foresight_embeddings=True)
        torch.testing.assert_close(result.foresight_embeddings, other.foresight_embeddings)
        changed[:,1] = 13
        contextual = model(input_ids=changed, labels=labels, use_cache=False, return_foresight_embeddings=True)
        self.assertFalse(torch.allclose(result.foresight_embeddings, contextual.foresight_embeddings))
        result.qa_loss.backward()
        self.assertGreater(model.foresight.queries.grad.abs().sum(), 0)
        self.assertGreater(model.model.layers[0].linear_attn.in_proj_qkv.weight.grad.abs().sum(), 0)

    def test_real_wan_video_loss_reaches_qwen_while_teacher_is_frozen(self):
        model = LlavaQwen35ForCausalLM(config(weight=0.1)).train()
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        teacher = tiny_teacher()
        teacher.gradient_checkpointing = True
        clean = torch.randn(1,48,2,4,4)
        noise = torch.randn_like(clean)
        def run(contexts, pixels):
            return conditional_flow_loss(teacher, clean, contexts, noise=noise, u=torch.tensor([0.5]))
        model.__dict__["_foresight_teacher_runtime"] = run
        with patch("llava.model.multimodal_generative_encoder.wan22.modules.model.flash_attention", masked_sdpa):
            output = model(input_ids=torch.tensor([[4,5,6,7,8,9]]),
                labels=torch.tensor([[-100,-100,-100,-100,8,9]]),
                foresight_pixels=torch.zeros(1,3,5,32,32), use_cache=False)
            torch.testing.assert_close(output.loss, output.qa_loss+0.1*output.video_loss)
            output.video_loss.backward()
        for parameter in [model.foresight.queries, model.foresight.to_wan[1].weight,
                          model.model.layers[0].linear_attn.in_proj_qkv.weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))
        self.assertFalse(any("teacher" in key for key in model.state_dict()))

    def test_wan_checkpointing_matches_context_gradient(self):
        model = tiny_teacher()
        clean = torch.randn(1,48,2,4,4)
        noise = torch.randn_like(clean)
        context = torch.randn(1,4,64, requires_grad=True)
        with patch("llava.model.multimodal_generative_encoder.wan22.modules.model.flash_attention", masked_sdpa):
            a = conditional_flow_loss(model, clean, context, noise=noise, u=torch.tensor([0.3]))
            grad_a = torch.autograd.grad(a, context)[0]
            model.gradient_checkpointing = True
            b = conditional_flow_loss(model, clean, context, noise=noise, u=torch.tensor([0.3]))
            grad_b = torch.autograd.grad(b, context)[0]
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(grad_a,grad_b)

    def test_reload_generation_inserts_slots_once_without_loading_teacher(self):
        model = LlavaQwen35ForCausalLM(config(weight=0.1)).eval()
        ids = torch.tensor([[4,5,6,7]])
        reference = model(input_ids=ids, use_cache=False).logits.detach()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = LlavaQwen35ForCausalLM.from_pretrained(directory).eval()
            torch.testing.assert_close(restored(input_ids=ids, use_cache=False).logits, reference)
            self.assertEqual(restored.__dict__["_foresight_teacher_runtime"], None)
            lengths = []
            handle = restored.model.register_forward_pre_hook(lambda module, args, kwargs:
                lengths.append(kwargs["inputs_embeds"].shape[1] if kwargs.get("inputs_embeds") is not None else kwargs["input_ids"].shape[1]), with_kwargs=True)
            try:
                generated = restored.generate(inputs=ids, max_new_tokens=2, do_sample=False,
                    use_cache=True, eos_token_id=None)
            finally:
                handle.remove()
            self.assertEqual(lengths, [8,1])
            self.assertEqual(generated.shape[1], 2)
            self.assertIsNone(restored.__dict__["_foresight_teacher_runtime"])

    def test_real_clip_loading_and_collator_keep_teacher_pixels_separate(self):
        from llava.train.train_3d import DataCollatorForSupervisedDataset
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for number in range(0, 50, 10):
                name = Path(directory)/f"{number:06d}.jpg"
                Image.fromarray(np.full((40,60,3), number, dtype=np.uint8)).save(name)
                paths.append(str(name))
            row = {"video":"scannet/unit", "foresight":{"video":"scannet/unit", "target_mode":TARGET_MODE,
                "observed_frames":[paths[0]], "teacher_clip_frames":paths, "anchor_frame":paths[0],
                "heldout_target_frames":paths[1:]}}
            pixels = load_teacher_pixels(row, 32, 64, observed_frames=[paths[0]])
            self.assertEqual(tuple(pixels.shape), (3,5,32,64))
            self.assertTrue((pixels >= -1).all() and (pixels <= 1).all())
            tokenizer = SimpleNamespace(pad_token_id=0, model_max_length=128, padding_side="right")
            collator = DataCollatorForSupervisedDataset(tokenizer)
            item = {"input_ids":torch.tensor([4,5]),"labels":torch.tensor([-100,5]),"foresight_pixels":pixels}
            batch = collator([item])
            self.assertEqual(tuple(batch["foresight_pixels"].shape), (1,3,5,32,64))
            self.assertNotIn("images", batch)
            bad = copy.deepcopy(row); bad["foresight"]["observed_frames"].append(paths[1])
            with self.assertRaisesRegex(ValueError,"overlap"):
                validate_pair(bad)
            with self.assertRaisesRegex(ValueError,"selection"):
                validate_pair(row, [paths[1]])

    def test_selected_answer_ce_matches_full_causal_ce(self):
        hidden = torch.randn(2,7,8,requires_grad=True)
        head = nn.Linear(8,16)
        labels = torch.tensor([[-100,-100,-100,4,5,6,-100],[-100,-100,-100,-100,7,8,9]])
        selected, _ = answer_loss(head, hidden, labels)
        full = nn.functional.cross_entropy(head(hidden[:,:-1]).reshape(-1,16),labels[:,1:].reshape(-1),ignore_index=-100)
        torch.testing.assert_close(selected,full)

    def test_optimizer_groups_and_loss_logging_without_optimizer_steps(self):
        from llava.train.train_3d import TrainingArguments
        from llava.train.llava_trainer import LLaVATrainer
        with tempfile.TemporaryDirectory() as directory:
            model = LlavaQwen35ForCausalLM(config()).train()
            args = TrainingArguments(output_dir=directory,use_cpu=True,report_to="none",
                learning_rate=1e-5,foresight_lr=1e-4,gradient_checkpointing=False)
            trainer = LLaVATrainer(model=model,args=args)
            optimizer = trainer.create_optimizer()
            for name, parameter in model.named_parameters():
                groups = [g for g in optimizer.param_groups if any(parameter is p for p in g["params"])]
                self.assertEqual(len(groups), 1)
                self.assertEqual(groups[0]["lr"], 1e-4 if name.startswith("foresight.") else 1e-5)
            loss = trainer.compute_loss(model,{"input_ids":torch.tensor([[4,5,6,7]]),
                "labels":torch.tensor([[-100,-100,6,7]]),"use_cache":False})
            trainer.log({"loss":loss.item()})
            self.assertIn("qa_loss",trainer.state.log_history[-1])
            self.assertEqual(trainer.state.log_history[-1]["video_loss"],0.)
            self.assertEqual(trainer.state.global_step,0)

    def test_factory_initializes_new_queries_from_language_only_checkpoint(self):
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        from llava.train.train_3d import get_model, ModelArguments, TrainingArguments
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionConfig, SigLipVisionModel
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); language=root/"Qwen3.5-2B-Language"; vision=root/"siglip-tiny"
            settings=helpers.tiny_config().to_dict();settings.pop("model_type",None)
            Qwen3_5ForCausalLM(Qwen3_5TextConfig(**settings)).save_pretrained(language,save_original_format=False)
            SigLipVisionModel(SigLipVisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                num_attention_heads=4,image_size=56,patch_size=14)).save_pretrained(vision)
            args=ModelArguments(model_name_or_path=str(language),vision_tower=str(vision),
                mm_projector_type="mlp2x_gelu",foresight_enabled=True,foresight_num_queries=4,
                foresight_context_dim=64)
            training=TrainingArguments(output_dir=str(root/"out"),use_cpu=True,report_to="none",attn_implementation="eager")
            loaded=get_model(args,training,{})
            self.assertEqual(tuple(loaded.foresight.queries.shape),(4,32))
            self.assertTrue(torch.isfinite(loaded.foresight.queries).all())
            self.assertGreater(loaded.foresight.queries.std(),0)
            self.assertEqual(loaded.config.foresight_num_queries,4)

    def test_native_generation_prompt_matches_training_answer_boundary(self):
        path=os.environ.get("QWEN35_TEST_TOKENIZER")
        if not path:
            self.skipTest("Set QWEN35_TEST_TOKENIZER to the prepared native tokenizer")
        from transformers import AutoTokenizer
        from llava.train.train_3d import preprocess_qwen35
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token
        spec=importlib.util.spec_from_file_location("foresight_eval_entry",Path(__file__).parents[1]/"scripts/3d/eval/eval_scanqa_foresight.py")
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True)
        tokenizer.model_max_length=32768
        row={"conversations":[{"from":"human","value":"<image>\nWhat is next to the chair?"},
                              {"from":"gpt","value":"a table"}]}
        prompt=entry.generation_prompt(row,tokenizer)
        inference=tokenizer_image_token(prompt,tokenizer,IMAGE_TOKEN_INDEX,return_tensors="pt")
        source=copy.deepcopy(row["conversations"])
        source[0]["value"]="<image>\nThe video captures 3D spatial information of a scene. Please focus on the spatial relationships in the video and answer the following questions.\n\nWhat is next to the chair?"
        training=preprocess_qwen35([source],tokenizer,has_image=True)
        cut=int((training["labels"][0]!=-100).nonzero()[0])
        torch.testing.assert_close(inference,training["input_ids"][0,:cut])

    def test_all_experiment_arguments_match_actual_training_parser(self):
        import yaml
        from transformers import HfArgumentParser
        from llava.train.train_3d import ModelArguments,DataArguments,TrainingArguments
        root=Path(__file__).parents[1]
        spec=importlib.util.spec_from_file_location("foresight_train_entry",root/"scripts/3d/train/train_scanqa_foresight.py")
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        settings=yaml.safe_load((root/"configs/foresight/scanqa_pilot.yaml").read_text())
        parser=HfArgumentParser((ModelArguments,DataArguments,TrainingArguments))
        for name,experiment in settings["experiments"].items():
            args=entry.training_arguments(settings,experiment,Path("unused.yaml"),Path("unused-output"),"qwen35-test",2)
            # Namespace parsing avoids TrainingArguments CUDA/ZeRO initialization.
            parsed=parser.parse_args(args)
            self.assertEqual(parsed.foresight_enabled,name!="E00")
            self.assertEqual(parsed.foresight_video_loss_weight,0.1 if name=="E03" else 0.)
            self.assertEqual(parsed.gradient_accumulation_steps,4)
            self.assertFalse(parsed.group_by_task_length)


if __name__ == "__main__":
    unittest.main(verbosity=2)
