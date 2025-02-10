import os
import random
import numpy as np

import torch
import torch.nn as nn

import triton
import triton.language as tl
import benchmark

# from liger_kernel.ops.rms_norm import LigerRMSNormFunction
# from liger_kernel.transformers.functional import liger_rms_norm
from liger_kernel.transformers.rms_norm import LigerRMSNorm

device = "cpu"

def set_seed(seed=42):
    """
    Fix all random seeds we use for reproducibility.
    """
    # Python random seed
    random.seed(seed)
    # Numpy random seed
    np.random.seed(0)
    # PyTorch random seed
    torch.manual_seed(seed)

    if device == "cuda":
        # If you are using CUDA
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

        # PyTorch backend settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device == "xpu":
        # If you are using XPU
        torch.xpu.manual_seed(seed)
        torch.xpu.manual_seed_all(seed)

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L112
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    

def test_liger_rms_norm():
    set_seed(42)
    bs, sl, hd = 2, 128, 512
    dtype = torch.float32   
    atol, rtol = 1e-4, 1e-6
    offset = 0.0
    casting_mode = "llama"
    in_place = False    

    _tensor = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    h1 = _tensor.clone().requires_grad_(True)
    h2 = _tensor.clone().requires_grad_(True)

    # do
    do = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    # reference (llama or gemma)
    ref_rms = LlamaRMSNorm(hidden_size=hd).to(device).to(dtype)
    ref_o = ref_rms(h1)
    ref_o.backward(do, retain_graph=True)

    # triton
    triton_rms = (
        LigerRMSNorm(hidden_size=hd, offset=offset, casting_mode=casting_mode, in_place=in_place).to(device).to(dtype)
    )
    triton_o = triton_rms(h2)
    triton_o.backward(do, retain_graph=True)

    # compare
    assert torch.allclose(ref_o, triton_o, atol=atol, rtol=rtol), (ref_o, triton_o)
    assert torch.allclose(h1.grad, h2.grad, atol=atol, rtol=rtol), (h1.grad, h2.grad)

if __name__ == "__main__":
    benchmark.select_cpu_backend()