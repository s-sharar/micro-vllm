"""Prepared Qwen weights and inference execution over contiguous KV caches.

Preparation binds tensor references and snapshots scalar settings once. Do not
modify those weights during execution. A cache belongs to one request; discard
or reset it after an unexpected execution failure, since layer writes are not
a whole-forward transaction. Explicit positions remain an uncached-only API.
"""
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import model
from .attention import build_causal_mask
from .kv_cache import ContiguousKVCache


def _checked_tensor(name, tensor, shape, reference=None):
    if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
            or tensor.shape != shape):
        raise ValueError(f'{name} must be a dense tensor with shape {shape}')
    if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f'{name} must use float32, float16, or bfloat16')
    if reference is not None and (tensor.dtype != reference.dtype or tensor.device != reference.device):
        raise ValueError(f'{name} must match the embedding dtype and device')
    return tensor


def _layer_shapes(config, head_dim):
    hidden, intermediate = config.hidden_size, config.intermediate_size
    kv_width = config.num_key_value_heads * head_dim
    return {
        'input_layernorm.weight': (hidden,),
        'post_attention_layernorm.weight': (hidden,),
        'self_attn.q_proj.weight': (hidden, hidden),
        'self_attn.k_proj.weight': (kv_width, hidden),
        'self_attn.v_proj.weight': (kv_width, hidden),
        'self_attn.q_proj.bias': (hidden,),
        'self_attn.k_proj.bias': (kv_width,),
        'self_attn.v_proj.bias': (kv_width,),
        'self_attn.o_proj.weight': (hidden, hidden),
        'mlp.gate_proj.weight': (intermediate, hidden),
        'mlp.up_proj.weight': (intermediate, hidden),
        'mlp.down_proj.weight': (hidden, intermediate),
    }


def _prepared_layers(weights, config, head_dim, embedding):
    shapes = _layer_shapes(config, head_dim)
    layers = []
    for index in range(config.num_hidden_layers):
        prefix = f'model.layers.{index}.'
        layers.append({name: _checked_tensor(prefix + name, weights[prefix + name], shape, embedding)
                       for name, shape in shapes.items()})
    return tuple(layers)


@dataclass
class PreparedModel:
    """Bound weights plus scalar configuration; forward is inference-only."""

    embedding_weight: torch.Tensor
    layers: tuple[Mapping[str, torch.Tensor], ...]
    final_norm: torch.Tensor
    output_weight: torch.Tensor
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    eps: float
    theta: float
    max_position_embeddings: int

    def _past_length(self, input_ids, cache):
        if cache is None:
            return 0
        if not isinstance(cache, ContiguousKVCache):
            raise ValueError('cache must be a ContiguousKVCache or None')
        expected = (len(self.layers), input_ids.shape[0], self.num_key_value_heads, self.head_dim)
        actual = (cache.num_layers, cache.batch_size, cache.num_kv_heads, cache.head_dim)
        if actual != expected:
            raise ValueError('cache geometry must match the prepared model and input batch')
        if cache.dtype != self.embedding_weight.dtype or cache.device != self.embedding_weight.device:
            raise ValueError('cache dtype and device must match the prepared model')
        past = cache.sequence_length
        if past + input_ids.shape[1] > cache.capacity:
            raise ValueError('KV cache capacity exceeded')
        return past

    def _positions(self, input_ids, position_ids, cache, past):
        if past + input_ids.shape[1] > self.max_position_embeddings:
            raise ValueError('sequence length exceeds the configured context length')
        if position_ids is not None:
            if cache is not None:
                raise ValueError('cached execution derives positions from the cache length')
            return model._forward_positions(input_ids, position_ids, self.max_position_embeddings)
        return torch.arange(past, past + input_ids.shape[1], device=input_ids.device).unsqueeze(0)

    @torch.inference_mode()
    def forward(self, input_ids, *, position_ids=None, cache=None, logits_to_keep=0):
        """Process new tokens, mutate cache if supplied, and return the requested logits.

        logits_to_keep=0 returns every new position; positive N returns the last
        N (or all if N exceeds the chunk). Cached calls use contiguous absolute
        positions and contain no padding. Cache preflight precedes layer writes.
        """
        if input_ids.ndim != 2 or input_ids.numel() == 0:
            raise ValueError('input_ids must be a nonempty [batch, tokens] tensor')
        if (input_ids.dtype not in (torch.int32, torch.int64)
                or input_ids.device != self.embedding_weight.device):
            raise ValueError('input_ids must be integer token IDs on the embedding device')
        if type(logits_to_keep) is not int or logits_to_keep < 0:
            raise ValueError('logits_to_keep must be a nonnegative integer')
        past = self._past_length(input_ids, cache)
        positions = self._positions(input_ids, position_ids, cache, past)
        hidden = model.embed_tokens(input_ids, self.embedding_weight)
        cos, sin = model.rope_cos_sin(positions, self.head_dim, self.theta, dtype=hidden.dtype)
        mask = build_causal_mask(input_ids.shape[1], past + input_ids.shape[1], past, input_ids.device)
        for index, layer_weights in enumerate(self.layers):
            hidden = model.decoder_layer(
                hidden, layer_weights, cos, sin,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads, eps=self.eps,
                cache=cache, layer_idx=index, attention_mask=mask)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:, :]
        normalized = model.rms_norm(hidden, self.final_norm, self.eps)
        return F.linear(normalized, self.output_weight)


def prepare_model(weights, config):
    """Bind a Qwen2 checkpoint once without copying tensor storage or scanning per token."""
    if not isinstance(weights, Mapping) or config is None:
        raise ValueError('preparation requires a weight mapping and model configuration')
    sizes = (config.num_hidden_layers, config.hidden_size, config.intermediate_size,
             config.num_attention_heads, config.num_key_value_heads,
             config.vocab_size, config.max_position_embeddings)
    if any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError('model dimensions must be positive integers')
    head_dim = model._validate_forward_config(config)
    if head_dim % 2:
        raise ValueError('RoPE requires an even head dimension')
    shape = (config.vocab_size, config.hidden_size)
    embedding = _checked_tensor('model.embed_tokens.weight', weights['model.embed_tokens.weight'], shape)
    final_norm = _checked_tensor('model.norm.weight', weights['model.norm.weight'],
                                 (config.hidden_size,), embedding)
    output = embedding if config.tie_word_embeddings else _checked_tensor(
        'lm_head.weight', weights['lm_head.weight'], shape, embedding)
    layers = _prepared_layers(weights, config, head_dim, embedding)
    return PreparedModel(embedding, layers, final_norm, output,
                         config.num_attention_heads, config.num_key_value_heads, head_dim,
                         config.rms_norm_eps, config.rope_theta, config.max_position_embeddings)
