import einops
import pytest
import torch as t
from smith.ch2_0_performance.baseline import BaselineTransformer
from smith.ch2_0_performance.ch2_0_performance import get_dataloader
from smith.ch2_0_performance.baseline import TransformerConfig

# Parametrized so different Baseline implementations can be injected by tests/fixtures.
@pytest.mark.parametrize("model", [BaselineTransformer])
def test_smoke_test(model):
    max_seq_len = 7
    _, vocab_to_token, token_to_vocab = get_dataloader(max_seq_len=max_seq_len, batch_size=2)
    baseline_model = model(TransformerConfig(
        vocab_size=len(vocab_to_token),
        d_model=512,
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
        d_model=512,
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