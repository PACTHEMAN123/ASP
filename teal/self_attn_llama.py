import torch
import types

from torch import nn

from teal_utils import (
    Distribution,
    SparsifyFn,
    get_module_device,
    TealActivation
)

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv
)

from transformers.integrations.sdpa_attention import sdpa_attention_forward

def _monkeypatch_self_attn(self_attn, file_path, grabbing_mode=False):
    self_attn.forward_old = self_attn.forward

    self_attn.forward = types.MethodType(_naive_forward, self_attn)

    self_attn.file_path = file_path
    self_attn.grabbing_mode = grabbing_mode

    if not grabbing_mode:
        self_attn.distrs = {}
        self_attn.distrs['h1'] = Distribution(file_path, hidden_type='h1')
        self_attn.distrs['h2'] = Distribution(file_path, hidden_type='h2')

        self_attn.sparse_fns = nn.ModuleDict({
            'q': SparsifyFn(self_attn.distrs['h1']).to(get_module_device(self_attn)),
            'k': SparsifyFn(self_attn.distrs['h1']).to(get_module_device(self_attn)),
            'v': SparsifyFn(self_attn.distrs['h1']).to(get_module_device(self_attn)),
            'o': SparsifyFn(self_attn.distrs['h2']).to(get_module_device(self_attn))
        })

    self_attn.activation_module = TealActivation(file_path)

    return self_attn


def _naive_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask = None, #: Optional[torch.LongTensor] = None,
    position_ids = None, #: Optional[torch.LongTensor] = None,
    past_key_values = None, #: Optional[Cache] = None,
    output_attentions = False, #: bool = False,
    use_cache = False, #: bool = False,
    cache_position = None, #: Optional[torch.LongTensor] = None,
    activation_module = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
    **kwargs, 
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # MONKEYPATCH HERE
    if self.grabbing_mode:
        self.activation_module.grab_activations(hidden_states, 'h1')
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
    else: 
        x_q = self.sparse_fns['q'](hidden_states)
        x_k = self.sparse_fns['k'](hidden_states)
        x_v = self.sparse_fns['v'](hidden_states)
        query_states = self.q_proj(x_q)
        key_states = self.k_proj(x_k)
        value_states = self.v_proj(x_v)

    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)
        
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)


    attn_output, attn_weights = sdpa_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()

    # MONKEYPATCH HERE
    if self.grabbing_mode:
        self.activation_module.grab_activations(attn_output, 'h2')
        attn_output = self.o_proj(attn_output)
    else:
        attn_output = self.sparse_fns['o'](attn_output)
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights