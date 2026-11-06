"""Model computations using ordinary PyTorch operations and supplied weights."""
from collections.abc import Mapping
import math

import torch
import torch.nn.functional as F


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

def causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Compute uncached, unpadded causal attention with grouped KV heads.

    Q is [batch, query_heads, tokens, head_dim]; K/V are
    [batch, kv_heads, tokens, head_dim]. All inputs share dtype and device.
    Returns [batch, query_heads, tokens, head_dim], before output projection.
    This version requires equally long full sequences and applies no dropout.
    """
    if query.ndim != 4 or key.ndim != 4 or key.shape != value.shape:
        raise ValueError('Q, K, V must be rank four and K/V must have matching shapes')
    if min(query.shape) <= 0 or min(key.shape) <= 0:
        raise ValueError('attention dimensions must be positive')
    batch_size, query_heads, tokens, head_dim = query.shape
    if (key.shape[0], key.shape[2], key.shape[3]) != (batch_size, tokens, head_dim):
        raise ValueError('Q, K, V must share batch size, token count, and head dimension')
    if query_heads % key.shape[1]:
        raise ValueError('query head count must be divisible by KV head count')
    if (not query.is_floating_point() or not query.dtype == key.dtype == value.dtype
            or not query.device == key.device == value.device):
        raise ValueError('Q, K, V must share a floating-point dtype and device')

    groups = query_heads // key.shape[1]
    repeated_key = key.repeat_interleave(groups, dim=1)
    repeated_value = value.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query, repeated_key.transpose(-2, -1)) * head_dim ** -0.5
    future_mask = torch.ones(tokens, tokens, dtype=torch.bool, device=query.device).triu(1)
    masked_scores = scores.masked_fill(future_mask, float('-inf'))
    weights = F.softmax(masked_scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, repeated_value)

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

def decoder_layer(
    hidden_states: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    eps: float,
) -> torch.Tensor:
    """Run one uncached, unpadded Qwen pre-norm layer using supplied tensors.

    weights uses local layer keys, e.g. self_attn.q_proj.weight, as in a layer
    state_dict. Cos/sin are precomputed for the sequence and shared by layers.
    Input/output shape is [batch, tokens, hidden_size]. Inputs are unchanged.
    """
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
    query, key, value = project_qkv(
        normalized,
        weights['self_attn.q_proj.weight'], weights['self_attn.k_proj.weight'],
        weights['self_attn.v_proj.weight'], weights['self_attn.q_proj.bias'],
        weights['self_attn.k_proj.bias'], weights['self_attn.v_proj.bias'])
    query = split_heads(query, num_attention_heads, head_dim)
    key = split_heads(key, num_key_value_heads, head_dim)
    value = split_heads(value, num_key_value_heads, head_dim)
    query, key = apply_rope(query, key, cos, sin)
    heads = causal_attention(query, key, value)
    after_attention = hidden_states + attention_output(heads, weights['self_attn.o_proj.weight'])
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


def forward(
    input_ids: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    config,
    *,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return all vocabulary logits for an uncached, unpadded Qwen2 sequence.

    weights uses complete checkpoint keys such as model.embed_tokens.weight.
    config supplies Qwen2 dimensions and options; no Hugging Face forward is
    called. Returns [batch, tokens, vocabulary_size], without mutating inputs.
    Call under torch.inference_mode() during generation.
    """
    head_dim = _validate_forward_config(config)
    if input_ids.ndim != 2 or input_ids.numel() == 0:
        raise ValueError('input_ids must be a nonempty [batch, tokens] tensor')
    embedding_weight = weights['model.embed_tokens.weight']
    if (input_ids.dtype not in (torch.int32, torch.int64)
            or input_ids.device != embedding_weight.device):
        raise ValueError('input_ids must be integer token IDs on the embedding device')
    expected_shape = (config.vocab_size, config.hidden_size)
    if embedding_weight.shape != expected_shape:
        raise ValueError('embedding table shape must match the model configuration')
    final_norm = weights['model.norm.weight']
    output_weight = embedding_weight if config.tie_word_embeddings else weights['lm_head.weight']
    if final_norm.shape != (config.hidden_size,) or output_weight.shape != expected_shape:
        raise ValueError('final norm and vocabulary projection must match the model configuration')
    positions = _forward_positions(input_ids, position_ids, config.max_position_embeddings)
    hidden_states = embed_tokens(input_ids, embedding_weight)
    cos, sin = rope_cos_sin(positions, head_dim, config.rope_theta, dtype=hidden_states.dtype)
    for index in range(config.num_hidden_layers):
        prefix = f'model.layers.{index}.'
        layer_weights = {name[len(prefix):]: value for name, value in weights.items()
                         if name.startswith(prefix)}
        hidden_states = decoder_layer(
            hidden_states, layer_weights, cos, sin,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads, eps=config.rms_norm_eps)
    normalized = rms_norm(hidden_states, final_norm, config.rms_norm_eps)
    return F.linear(normalized, output_weight)

