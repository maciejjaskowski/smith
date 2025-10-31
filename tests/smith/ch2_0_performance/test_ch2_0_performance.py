import einops
import pytest
import torch as t
import torch
import os, random, numpy as np
from smith.ch2_0_performance.baseline import BaselineTransformer
from smith.ch2_0_performance.ch2_0_performance import get_dataloader
from smith.ch2_0_performance.ch2_0_performance import reproducible_step
from smith.ch2_0_performance.ch2_0_performance import TransformerConfig

# Parametrized so different Baseline implementations can be injected by tests/fixtures.
@pytest.mark.parametrize("model", [BaselineTransformer])
def test_smoke_test(model):
    max_seq_len = 7
    _, vocab_to_token, token_to_vocab = get_dataloader(max_seq_len=max_seq_len, batch_size=2)
    baseline_model = model(TransformerConfig(
        vocab_size=len(vocab_to_token),
        d_model=128,
        d_head=64,
        n_head=8,
        max_seq_len=max_seq_len,
        n_layers=4
    ))
    device = "cuda"
    baseline_model.to(device)
    baseline_model = baseline_model.to("cuda")
    bos_tensor = t.Tensor([vocab_to_token["_bos_"]] * 7).to(t.long)
    inp = einops.repeat(
        bos_tensor, "s -> batch s",
        batch=2
    )
    baseline_model(inp.to("cuda"))
    
    
@pytest.mark.parametrize("model", [BaselineTransformer])
def test_smoke_test_with_dataloader(model):
    max_seq_len = 7
    dataloader, vocab_to_token, token_to_vocab = get_dataloader(max_seq_len=max_seq_len, batch_size=2)
    baseline_model = model(TransformerConfig(
        vocab_size=len(vocab_to_token),
        d_model=128,
        d_head=64,
        n_head=8,
        max_seq_len=max_seq_len,
        n_layers=4
    ))
    device = "cuda"
    baseline_model.to(device)
    baseline_model = baseline_model.to("cuda")
    
    inp = next(iter(dataloader))
    baseline_model(inp['inputs'].to("cuda"))
    
    
def test_dataloader_reproducability():
    set_repro(1337)
    dataloader,_,_ = get_dataloader(max_seq_len=7, batch_size=2, reproducible=True)
    inputs1 = next(iter(dataloader))['inputs']
    dataloader, _, _ = get_dataloader(max_seq_len=7, batch_size=2, reproducible=True)
    inputs2 = next(iter(dataloader))['inputs']
    assert t.allclose(inputs1, inputs2)
    
@pytest.mark.parametrize("model", [BaselineTransformer])
def test_reproducibility(model):

    config = TransformerConfig(
        vocab_size=None,
        d_model=128,
        d_head=64,
        n_head=8,
        max_seq_len=128,
        n_layers=2
    )
    set_repro(1337)
    loss1, grad1 = reproducible_step(model, config)
    set_repro(1337)
    loss2, grad2 = reproducible_step(model, config)
    assert t.allclose(loss1, loss2, atol=1e-4), f"Actual values {loss1=} {loss2=}"
    for k in grad1.keys():
        assert t.allclose(grad1[k], grad2[k], atol=1e-4)
    
    
