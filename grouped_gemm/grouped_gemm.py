import torch
import grouped_gemm_cuda

def _allocate_output(a, b, batch_sizes, trans_a, trans_b):
    assert not (trans_a and trans_b)
    assert batch_sizes.ndim == 1, "Expected 1d tensor for batch_sizes"
    assert a.ndim == 2, "Expected 2d tensor for 'a'"
    assert b.ndim == (2 if trans_a else 3)

    shape = (
        (batch_sizes.shape[0], a.shape[1], b.shape[1])
        if trans_a else
        (a.shape[0], (b.shape[1] if trans_b else b.shape[2]))
    )
    return torch.empty(*shape, device=a.device, dtype=a.dtype)

def _gmm(a, b, batch_sizes, trans_a=False, trans_b=False, c=None):
    if c is None:
        c = _allocate_output(a, b, batch_sizes, trans_a, trans_b)
    grouped_gemm_cuda.gmm(a, b, c, batch_sizes, trans_a, trans_b)
    return c

def gmm_op(a, b, batch_sizes, trans_a=False, trans_b=False, c=None):
    assert ((a.ndim == 2 and b.ndim == 3 and (c is None or c.ndim == 2) and (not trans_a)) or
            (a.ndim == 2 and b.ndim == 2 and (c is None or c.ndim == 3) and trans_a and (not trans_b))), "Input shape and stride is not supported!"
    if not trans_a:
        # a = [m, k]
        # b = [e, n, k] if trans_b else [e, k, n]
        assert (a.shape[1] == b.shape[2] if trans_b else b.shape[1])
        if c is not None:
            assert c.shape == (a.shape[0], (b.shape[1] if trans_b else b.shape[2]))
        return _gmm(a, b, batch_sizes, trans_a=False, trans_b=trans_b, c=c)
    else:
        # grad [m, n]
        # a [m ,k]
        assert (a.shape[0] == b.shape[0])
        if c is not None:
            assert c.shape == (batch_sizes.shape[0], a.shape[1], b.shape[1])
        return _gmm(a, b, batch_sizes, trans_a=True, trans_b=False, c=c)

class GroupedGemm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, batch_sizes, trans_b):
        assert torch.count_nonzero(batch_sizes) != 0, "Input batch_sizes should not be all zeros!"
        ctx.save_for_backward(a, b, batch_sizes)
        ctx.trans_b = trans_b
        return _gmm(a, b, batch_sizes, trans_a=False, trans_b=trans_b)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        a, b, batch_sizes = ctx.saved_tensors
        trans_b = ctx.trans_b

        agrad = None
        if ctx.needs_input_grad[0]:
            agrad = _gmm(
                grad, b, batch_sizes, trans_a=False, trans_b=not trans_b)

        bgrad = None
        if ctx.needs_input_grad[1]:
            lhs, rhs = (grad, a) if trans_b else (a, grad)
            bgrad = _gmm(
                lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
        return agrad, bgrad, None, None


def gmm(a, b, batch_sizes, trans_b=False):
    return GroupedGemm.apply(a, b, batch_sizes, trans_b)
