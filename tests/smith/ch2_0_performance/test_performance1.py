import einops
import pytest
import torch as t
import torch
import os, random, numpy as np
from smith.ch2_0_performance.baseline import BaselineTransformer
from smith.ch2_0_performance.ch2_0_performance import get_dataloader
from smith.ch2_0_performance.ch2_0_performance import reproducible_step
from smith.ch2_0_performance.baseline import TransformerConfig

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