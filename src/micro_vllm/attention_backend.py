"""Attention policy: reference, native single-token decode, and Flash prefill."""
import torch
import torch.nn.functional as F

from .attention import _validate_inputs, causal_attention

ATTENTION_BACKENDS = ('auto', 'torch', 'cuda')


def validate_attention_backend(name):
    if not isinstance(name, str) or name not in ATTENTION_BACKENDS:
        raise ValueError("attention_backend must be 'auto', 'torch', or 'cuda'")
    return name


def validate_attention_implementation(name):
    if not isinstance(name, str) or name not in ('torch', 'cuda', 'flash'):
        raise ValueError("attention implementation must be 'torch', 'cuda', or 'flash'")
    return name


def _flash_eligible(device, dtype, batch_size, head_dim):
    return (device.type == 'cuda' and batch_size == 1 and dtype == torch.float16
            and 0 < head_dim <= 256 and head_dim % 8 == 0
            and torch.backends.cuda.is_flash_attention_available()
            and torch.cuda.get_device_capability(device)[0] >= 8)


def validate_layer_attention(implementation, *, device, dtype, batch_size,
                             query_length, head_dim, attention_mask=None):
    """Reject predictable direct-layer failures before that layer appends K/V."""
    validate_attention_implementation(implementation)
    if implementation == 'torch' or attention_mask is not None:
        return
    device = torch.device(device)
    if implementation == 'cuda':
        if (device.type != 'cuda' or batch_size != 1 or query_length != 1
                or dtype not in (torch.float16, torch.float32)):
            raise ValueError('native decode requires CUDA batch size 1, one query, and float16 or float32')
    elif query_length <= 0 or not _flash_eligible(device, dtype, batch_size, head_dim):
        raise ValueError('Flash prefill requires supported CUDA float16 geometry and FlashAttention availability')


def select_attention_backend(name, *, device, dtype, batch_size, query_length,
                             head_dim, past_length, has_cache):
    """Resolve policy before cache writes; fallback is based on capabilities only."""
    validate_attention_backend(name)
    device = torch.device(device)
    native_eligible = (device.type == 'cuda' and batch_size == 1
                       and dtype in (torch.float16, torch.float32))
    if name == 'cuda' and not native_eligible:
        raise ValueError("attention_backend='cuda' requires CUDA batch size 1 and float16 or float32")
    if name == 'torch' or not native_eligible:
        return 'torch'
    if has_cache and past_length > 0 and query_length == 1:
        return 'cuda'
    if name == 'auto' and _flash_eligible(device, dtype, batch_size, head_dim):
        return 'flash'
    return 'torch'


def flash_attention(query, key, value, *, past_length=0):
    """Use Flash exclusively, including bottom-right causality for cached chunks."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.bias import causal_lower_right

    mask = causal_lower_right(query.shape[2], key.shape[2]) if past_length else None
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, is_causal=not past_length,
            dropout_p=0.0, enable_gqa=True)


def attention_forward(query, key, value, *, past_length=0, attention_mask=None,
                      implementation='torch'):
    """Execute one resolved implementation; an explicit blocked-position mask is honored."""
    validate_attention_implementation(implementation)
    _validate_inputs(query, key, value, past_length)
    if implementation == 'torch' or attention_mask is not None:
        return causal_attention(query, key, value, past_length=past_length,
                                attention_mask=attention_mask)
    if implementation == 'cuda':
        from .cuda_attention import decode_attention_gqa

        return decode_attention_gqa(query, key, value)
    return flash_attention(query, key, value, past_length=past_length)
