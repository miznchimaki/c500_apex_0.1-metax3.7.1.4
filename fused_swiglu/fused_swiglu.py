import importlib
import torch
import torch.nn.functional as F
global fused_swiglu_cuda
fused_swiglu_cuda = None

class BiasSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        global fused_swiglu_cuda
        if fused_swiglu_cuda is None:
            fused_swiglu_cuda = importlib.import_module("fused_swiglu_cuda")
        output, input_plus_bias = fused_swiglu_cuda.forward_with_bias(input, bias)
        ctx.save_for_backward(input_plus_bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_plus_bias = ctx.saved_tensors[0]
        tmp = fused_swiglu_cuda.backward(grad_output, input_plus_bias)
        return tmp, tmp


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        ctx.save_for_backward(input)
        ctx.ori_input_dtype = input.dtype
        global fused_swiglu_cuda
        if fused_swiglu_cuda is None:
            fused_swiglu_cuda = importlib.import_module("fused_swiglu_cuda")
        return fused_swiglu_cuda.forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        tmp = fused_swiglu_cuda.backward(grad_output, input)
        return tmp


class FusedSwiGLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, bias=None):
        ori_shape = input.shape
        assert len(ori_shape) in [2, 3]
        assert input.dtype in [torch.bfloat16, torch.float16]
        assert(ori_shape[-1] % 2 == 0)
        if bias is not None:
            assert ori_shape[-1] == bias.shape[-1] and len(bias.shape) == 1
            assert(input.dtype == bias.dtype)
            output = BiasSwiGLUFunction.apply(input, bias)
        else:
            output = SwiGLUFunction.apply(input)
        return output