"""Causal attention over cached prefixes without repeating grouped K/V heads."""
import torch
import torch.nn.functional as F


def _validate_lengths(query_length: int, key_length: int, past_length: int) -> None:
    if (any(type(length) is not int for length in (query_length, key_length, past_length))
            or query_length <= 0 or key_length <= 0 or past_length < 0):
        raise ValueError('query/key lengths must be positive integers and past_length nonnegative')
    if key_length != past_length + query_length:
        raise ValueError('key length must equal past_length plus query length')


def build_causal_mask(
    query_length: int,
    key_length: int,
    past_length: int,
    device: torch.device | str,
) -> torch.Tensor | None:
    """Return [query, key] blocked positions; one-token decode needs no mask.

    Query row i has absolute position past_length + i. True blocks a future
    key; the diagonal stays visible. All key/value rows contain valid tokens.
    """
    _validate_lengths(query_length, key_length, past_length)
    if query_length == 1:
        return None
    query_positions = torch.arange(past_length, past_length + query_length, device=device)
    key_positions = torch.arange(key_length, device=device)
    return key_positions.unsqueeze(0) > query_positions.unsqueeze(1)


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    past_length: int,
) -> None:
    if (query.layout != torch.strided or key.layout != torch.strided
            or value.layout != torch.strided):
        raise ValueError('Q, K, V must have dense strided layouts')
    if (query.ndim != 4 or key.ndim != 4 or value.ndim != 4
            or key.shape != value.shape):
        raise ValueError('Q, K, V must be rank four and K/V must have matching shapes')
    if min(query.shape) <= 0 or min(key.shape) <= 0:
        raise ValueError('attention dimensions must be positive')
    if (query.shape[0], query.shape[3]) != (key.shape[0], key.shape[3]):
        raise ValueError('Q, K, V must share batch size and head dimension')
    _validate_lengths(query.shape[2], key.shape[2], past_length)
    if query.shape[1] % key.shape[1]:
        raise ValueError('query head count must be divisible by KV head count')
    if (not query.is_floating_point() or not query.dtype == key.dtype == value.dtype
            or not query.device == key.device == value.device):
        raise ValueError('Q, K, V must share a floating-point dtype and device')


def causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    past_length: int = 0,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attend Q [B,Hq,Q,D] to cached K/V [B,Hkv,past_length+Q,D].

    Consecutive groups of Hq/Hkv query heads share each original KV head.
    Folding those groups into query rows avoids allocating repeated K/V.
    Inputs may be noncontiguous and are unchanged. Softmax reduces in FP32,
    then probabilities return to the input dtype before the value product.
    A supplied boolean [Q,K] mask marks blocked positions and can be reused
    across layers; otherwise the causal mask is built once for this call.
    """
    _validate_inputs(query, key, value, past_length)
    batch_size, query_heads, query_length, head_dim = query.shape
    kv_heads, key_length = key.shape[1:3]
    if attention_mask is not None:
        if (attention_mask.layout != torch.strided or attention_mask.dtype != torch.bool
                or attention_mask.device != query.device
                or attention_mask.shape != (query_length, key_length)):
            raise ValueError('attention_mask must be a dense boolean [query, key] tensor on the Q device')
    else:
        attention_mask = build_causal_mask(query_length, key_length, past_length, query.device)

    groups = query_heads // kv_heads
    grouped_query = query.reshape(batch_size, kv_heads, groups * query_length, head_dim)
    scores = torch.matmul(grouped_query, key.transpose(-2, -1)).mul_(head_dim ** -0.5)
    scores = scores.reshape(batch_size, kv_heads, groups, query_length, key_length)
    if attention_mask is not None:
        scores.masked_fill_(attention_mask, float('-inf'))
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    probabilities = probabilities.reshape(batch_size, kv_heads, groups * query_length, key_length)
    output = torch.matmul(probabilities, value)
    return output.reshape(batch_size, query_heads, query_length, head_dim)
