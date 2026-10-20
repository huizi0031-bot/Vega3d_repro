"""Dense-versus-selected loss/gradient and API contracts; tiny models on CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import unittest
import torch
from transformers import Qwen3_5ForCausalLM
from test_qwen35_wan22 import tiny_config
from llava.model.language_model.llava_qwen import LlavaQwen35ForCausalLM
from wan_offline_tools.qwen35_memory_compat import install_supervised_logits_only


class SupervisedLogitTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.original = Qwen3_5ForCausalLM.forward

    def tearDown(self):
        Qwen3_5ForCausalLM.forward = self.original

    def test_loss_all_parameter_and_input_gradients_match(self):
        for batch in [1,2]:
            for denominator in [None, torch.tensor(11)]:
                with self.subTest(batch=batch, denominator=denominator):
                    Qwen3_5ForCausalLM.forward = self.original
                    model = LlavaQwen35ForCausalLM(tiny_config()).train()
                    embeds = torch.randn(batch,19,32,requires_grad=True)
                    labels = torch.full((batch,19),-100,dtype=torch.long)
                    labels[:,0] = 4  # A causal decoder never supervises this first target.
                    labels[0,13:15] = torch.tensor([7,8])
                    labels[0,18] = 2
                    if batch == 2:
                        labels[1,10:12] = torch.tensor([9,10])
                        labels[1,17] = 2
                    kwargs = {} if denominator is None else {'num_items_in_batch':denominator}
                    dense = model(inputs_embeds=embeds, labels=labels, use_cache=False, **kwargs)
                    dense.loss.backward()
                    expected_loss = dense.loss.detach().clone()
                    expected_input = embeds.grad.detach().clone()
                    expected_gradients = {n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
                    selected = (torch.nn.functional.pad(labels,(0,1),value=-100)[...,1:] != -100).any(0).nonzero(as_tuple=True)[0]
                    expected_logits = dense.logits[:,selected].detach().clone()
                    model.zero_grad(set_to_none=True)
                    embeds.grad = None
                    install_supervised_logits_only()
                    sparse = model(inputs_embeds=embeds, labels=labels, use_cache=False, **kwargs)
                    sparse.loss.backward()
                    torch.testing.assert_close(sparse.loss,expected_loss,atol=1e-6,rtol=1e-5)
                    torch.testing.assert_close(sparse.logits,expected_logits,atol=1e-6,rtol=1e-5)
                    torch.testing.assert_close(embeds.grad,expected_input,atol=1e-6,rtol=1e-5)
                    actual = {n:p.grad for n,p in model.named_parameters() if p.grad is not None}
                    self.assertEqual(actual.keys(),expected_gradients.keys())
                    for name,value in actual.items():
                        torch.testing.assert_close(value,expected_gradients[name],atol=2e-6,rtol=1e-4,msg=name)
                    self.assertEqual(sparse.logits.shape[1],selected.numel())

    def test_eval_generation_and_all_ignored_targets_keep_native_behavior(self):
        model = LlavaQwen35ForCausalLM(tiny_config()).eval()
        ids = torch.tensor([[3,4,5,2]])
        labels = torch.tensor([[-100,-100,5,2]])
        before = model(input_ids=ids,labels=labels,use_cache=False)
        install_supervised_logits_only()
        after = model(input_ids=ids,labels=labels,use_cache=False)
        torch.testing.assert_close(before.logits,after.logits,atol=0,rtol=0)
        torch.testing.assert_close(before.loss,after.loss,atol=0,rtol=0)
        model.train()
        no_labels = model(input_ids=ids,use_cache=False)
        self.assertEqual(no_labels.logits.shape[1],4)
        last_only = model(input_ids=ids,use_cache=False,logits_to_keep=1)
        self.assertEqual(last_only.logits.shape[1],1)
        ignored = model(input_ids=ids,labels=torch.full_like(ids,-100),use_cache=False)
        self.assertEqual(ignored.logits.shape[1],4)
        self.assertTrue(torch.isnan(ignored.loss))

    def test_real_vocabulary_projects_only_supervised_positions(self):
        model = LlavaQwen35ForCausalLM(tiny_config(vocab_size=248320)).train()
        ids = torch.tensor([[3,4,5,6,7,8,9,10,11,12,13,2]])
        labels = torch.full_like(ids,-100)
        labels[:,-2:] = ids[:,-2:]
        observed = []
        handle = model.lm_head.register_forward_pre_hook(lambda module,args:observed.append(tuple(args[0].shape)))
        install_supervised_logits_only()
        output = model(input_ids=ids,labels=labels,use_cache=False)
        output.loss.backward()
        handle.remove()
        self.assertEqual(observed,[(1,2,32)])
        self.assertEqual(tuple(output.logits.shape),(1,2,248320))
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == '__main__':
    unittest.main(verbosity=2)
