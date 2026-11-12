import numpy as np
import torch
import torch.nn.functional as F


def _is_bf16(dtype):
    name = str(dtype).lower()
    return "bfloat16" in name or "bf16" in name


def impl(x):
    """Mish 参考实现（直接调用 torch 对标接口，高精度）。

    输入/输出均为 numpy ndarray；内部统一升到 float32 后调用
    torch.nn.functional.mish，再转回原 dtype。torch 的 mish 内部为
    x * tanh(softplus(x))，softplus 在 x>20 时直接返回 x，避免 exp 溢出。
    """
    orig_dtype = x.dtype
    is_bf16 = _is_bf16(orig_dtype)

    # fp16/bf16/fp32 统一升到 fp32 计算，最后转回原 dtype
    xf = np.ascontiguousarray(np.asarray(x, dtype=np.float32))

    y = F.mish(torch.from_numpy(xf)).detach().numpy()

    # np.asarray 保证 0-D 输入也返回 ndarray，而不是 numpy 标量
    if is_bf16:
        from ml_dtypes import bfloat16
        return np.asarray(y, dtype=bfloat16)
    return np.asarray(y, dtype=orig_dtype)
