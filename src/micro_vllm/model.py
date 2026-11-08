"""Model computations using ordinary PyTorch operations and supplied weights."""
from collections.abc import Mapping
import math

import torch
import torch.nn.functional as F

from .attention import causal_attention
from .attention_backend import attention_forward, validate_layer_attention
from .kv_cache import ContiguousKVCache


def embed_tokens(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Look up pretrained vectors without changing or reinitializing the table.

    input_ids: integer token IDs with shape [batch, sequence_length].
    weight: embedding table with shape [vocabulary_size, hidden_size].
    Returns a tensor with shape [batch, sequence_length, hidden_size].
    PyTorch rejects non-integer IDs and IDs outside the table.
    """
    return F.embedding(input_ids, weight)

def rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Normalize each token across its hidden dimension, then apply its scale.

    hidden_states: [batch, sequence_length, hidden_size].
    weight: pretrained scale vector [hidden_size].
    eps: positive stability constant from the model configuration.
    Statistics use FP32; normalized values return to the input dtype before
    multiplication, matching Qwen's precision behavior. Inputs are unchanged.
    """
    input_dtype = hidden_states.dtype
    hidden_fp32 = hidden_states.to(torch.float32)
    mean_square = hidden_fp32.pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden_fp32 * torch.rsqrt(mean_square + eps)
    return weight * normalized.to(input_dtype)

def project_qkv(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    v_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Qwen's three learned projections before head reshaping and RoPE.

    hidden_states: normalized token vectors [batch, sequence_length, hidden_size].
    Each weight is [output_width, hidden_size]; each bias is [output_width].
    Returns Q, K, V with shape [batch, sequence_length, output_width].
    The supplied pretrained tensors are unchanged.
    """
    query = F.linear(hidden_states, q_weight, q_bias)
    key = F.linear(hidden_states, k_weight, k_bias)
    value = F.linear(hidden_states, v_weight, v_bias)
    return query, key, value

def split_heads(
    projected: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Arrange [batch, tokens, heads * head_dim] as [batch, heads, tokens, head_dim].

    num_heads and head_dim come from the model configuration. Values, dtype,
    and device are preserved. The result may share storage with projected.
    """
    batch_size, sequence_length, _ = projected.shape
    headed = projected.reshape(batch_size, sequence_length, num_heads, head_dim)
    return headed.transpose(1, 2)

def rope_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    theta: float,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build [batch, tokens, head_dim] tables for default, full-head Qwen RoPE.

    position_ids contains absolute token positions on the activation device.
    theta comes from config.rope_theta. Frequencies and trigonometry use FP32,
    then tables are cast to the activation dtype. Scaled RoPE is outside scope.
    """
    if head_dim <= 0 or head_dim % 2:
        raise ValueError('head_dim must be a positive even integer')
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError('theta must be finite and positive')
    dimensions = torch.arange(0, head_dim, 2, dtype=torch.float32,
                              device=position_ids.device)
    inverse_freq = 1.0 / (theta ** (dimensions / head_dim))
    angles = position_ids.to(torch.float32).unsqueeze(-1) * inverse_freq
    duplicated_angles = torch.cat((angles, angles), dim=-1)
    return duplicated_angles.cos().to(dtype), duplicated_angles.sin().to(dtype)


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    """Map [first_half, second_half] to [-second_half, first_half] for RoPE."""
    width = values.shape[-1]
    if width <= 0 or width % 2:
        raise ValueError('the final dimension must be positive and even')
    first_half, second_half = values[..., :width // 2], values[..., width // 2:]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q/K [batch, heads, tokens, head_dim] without changing inputs.

    Tables [batch, tokens, head_dim] broadcast over independent Q and KV heads.
    V is not part of this operation.
    """
    head_cos, head_sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rotated_query = query * head_cos + rotate_half(query) * head_sin
    rotated_key = key * head_cos + rotate_half(key) * head_sin
    return rotated_query, rotated_key


def merge_heads(heads: torch.Tensor) -> torch.Tensor:
    """Concatenate head results per token: [batch, heads, tokens, dim] to [batch, tokens, heads * dim].

    The transpose puts head vectors beside each other for each token. Reshape
    also supports noncontiguous inputs. Values and input tensors are unchanged.
    """
    batch_size, num_heads, sequence_length, head_dim = heads.shape
    token_heads = heads.transpose(1, 2)
    return token_heads.reshape(batch_size, sequence_length, num_heads * head_dim)


def attention_output(heads: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Merge attention heads and apply Qwen's pretrained, bias-free output projection.

    heads: [batch, query_heads, tokens, head_dim].
    weight: [hidden_size, query_heads * head_dim].
    Returns [batch, tokens, hidden_size], before residual addition.
    """
    return F.linear(merge_heads(heads), weight)

def mlp(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply Qwen's bias-free SiLU-gated feed-forward network per token.

    hidden_states: [batch, tokens, hidden_size].
    gate_weight and up_weight: [intermediate_size, hidden_size].
    down_weight: [hidden_size, intermediate_size].
    Returns [batch, tokens, hidden_size], before residual addition.
    All supplied weights and activations are unchanged.
    """
    gate = F.linear(hidden_states, gate_weight)
    up = F.linear(hidden_states, up_weight)
    gated = F.silu(gate) * up
    return F.linear(gated, down_weight)

def _layer_attention(
    normalized, weights, cos, sin, *, num_attention_heads,
    num_key_value_heads, cache, layer_idx, attention_mask, attention_backend='torch',
):
    """Compute this layer's attention, appending only newly computed rotated K/V."""
    past = 0
    if cache is not None:
        if not isinstance(cache, ContiguousKVCache) or layer_idx is None:
            raise ValueError('cached layers require a cache and layer index')
        past = cache.length(layer_idx)
    tokens = normalized.shape[1]
    if attention_mask is not None and (
            not isinstance(attention_mask, torch.Tensor)
            or attention_mask.layout != torch.strided or attention_mask.dtype != torch.bool
            or attention_mask.device != normalized.device
            or attention_mask.shape != (tokens, past + tokens)):
        raise ValueError('attention_mask must be a boolean [new_tokens, total_tokens] tensor')
    head_dim = normalized.shape[-1] // num_attention_heads
    validate_layer_attention(
        attention_backend, device=normalized.device, dtype=normalized.dtype,
        batch_size=normalized.shape[0], query_length=tokens, head_dim=head_dim,
        attention_mask=attention_mask)
    query, key, value = project_qkv(
        normalized, weights['self_attn.q_proj.weight'], weights['self_attn.k_proj.weight'],
        weights['self_attn.v_proj.weight'], weights['self_attn.q_proj.bias'],
        weights['self_attn.k_proj.bias'], weights['self_attn.v_proj.bias'])
    query = split_heads(query, num_attention_heads, head_dim)
    key = split_heads(key, num_key_value_heads, head_dim)
    value = split_heads(value, num_key_value_heads, head_dim)
    query, key = apply_rope(query, key, cos, sin)
    if cache is not None:
        cache.append(layer_idx, key, value)
        key, value = cache.read(layer_idx)
    heads = attention_forward(query, key, value, past_length=past,
                              attention_mask=attention_mask, implementation=attention_backend)
    return attention_output(heads, weights['self_attn.o_proj.weight'])


def decoder_layer(
    hidden_states: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    eps: float,
    cache: ContiguousKVCache | None = None,
    layer_idx: int | None = None,
    attention_mask: torch.Tensor | None = None,
    attention_backend: str = 'torch',
) -> torch.Tensor:
    """Run one unpadded Qwen pre-norm layer, optionally extending its KV cache."""
    if num_attention_heads <= 0 or num_key_value_heads <= 0:
        raise ValueError('head counts must be positive')
    if hidden_states.ndim != 3 or hidden_states.shape[-1] % num_attention_heads:
        raise ValueError('hidden states must be rank three with width divisible by query heads')
    if num_attention_heads % num_key_value_heads:
        raise ValueError('query head count must be divisible by KV head count')
    head_dim = hidden_states.shape[-1] // num_attention_heads
    if (cos.ndim != 3 or sin.shape != cos.shape
            or cos.shape[0] not in (1, hidden_states.shape[0])
            or cos.shape[1:] != (hidden_states.shape[1], head_dim)):
        raise ValueError('RoPE tables must match the batch, token count, and head dimension')
    normalized = rms_norm(hidden_states, weights['input_layernorm.weight'], eps)
    attention_update = _layer_attention(
        normalized, weights, cos, sin, num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads, cache=cache,
        layer_idx=layer_idx, attention_mask=attention_mask, attention_backend=attention_backend)
    after_attention = hidden_states + attention_update
    normalized_mlp = rms_norm(after_attention, weights['post_attention_layernorm.weight'], eps)
    update = mlp(normalized_mlp, weights['mlp.gate_proj.weight'],
                 weights['mlp.up_proj.weight'], weights['mlp.down_proj.weight'])
    return after_attention + update



def _validate_forward_config(config) -> int:
    """Reject architectures or options that this Qwen2 implementation cannot execute."""
    if (config.model_type != 'qwen2' or config.hidden_act != 'silu'
            or getattr(config, 'rope_scaling', None) is not None):
        raise ValueError('forward supports Qwen2 with SiLU and unscaled RoPE')
    if (getattr(config, 'use_sliding_window', False)
            or any(kind != 'full_attention' for kind in config.layer_types)):
        raise ValueError('forward currently supports full attention layers only')
    if (config.num_hidden_layers <= 0 or config.hidden_size <= 0
            or config.num_attention_heads <= 0 or config.num_key_value_heads <= 0
            or config.hidden_size % config.num_attention_heads
            or config.num_attention_heads % config.num_key_value_heads):
        raise ValueError('layer count, hidden width, and head counts must be positive and compatible')
    head_dim = config.hidden_size // config.num_attention_heads
    if (getattr(config, 'head_dim', None) not in (None, head_dim)
            or getattr(config, 'partial_rotary_factor', 1.0) != 1.0):
        raise ValueError('forward requires full-head RoPE with the standard head dimension')
    return head_dim


def _forward_positions(input_ids: torch.Tensor, position_ids, max_positions: int) -> torch.Tensor:
    """Create shared absolute positions or validate the caller's position tensor."""
    batch_size, tokens = input_ids.shape
    if tokens > max_positions:
        raise ValueError('sequence length exceeds the configured context length')
    if position_ids is None:
        return torch.arange(tokens, device=input_ids.device).unsqueeze(0)
    if (position_ids.ndim != 2 or position_ids.shape[0] not in (1, batch_size)
            or position_ids.shape[1] != tokens):
        raise ValueError('position_ids must have shape [1 or batch_size, tokens]')
    if (position_ids.dtype not in (torch.int32, torch.int64)
            or position_ids.device != input_ids.device):
        raise ValueError('position_ids must be integer positions on the input device')
    if (position_ids < 0).any() or (position_ids >= max_positions).any():
        raise ValueError('position_ids must be inside the configured context length')
    return position_ids



def prepare_model(weights, config, *, attention_backend='auto'):
    """Bind checkpoint tensors and settings once for repeated forward calls."""
    from .runtime import prepare_model as prepare

    return prepare(weights, config, attention_backend=attention_backend)


def forward(input_ids, weights, config=None, *, position_ids=None, cache=None,
            logits_to_keep=0, attention_backend=None):
    """Run inference with raw checkpoint weights or a previously prepared model.

    Raw mappings require config. A prepared model reuses its bound settings.
    Cache mutation is in place; 0 keeps all new-position logits, N the last N.
    """
    from .runtime import PreparedModel

    prepared = weights if isinstance(weights, PreparedModel) else prepare_model(
        weights, config, attention_backend='auto' if attention_backend is None else attention_backend)
    return prepared.forward(input_ids, position_ids=position_ids, cache=cache,
                            logits_to_keep=logits_to_keep, attention_backend=attention_backend)
