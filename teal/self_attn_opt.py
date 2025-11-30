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
    past_key_values = None, #: Optional[Cache] = None,
    attention_mask = None, #: Optional[torch.LongTensor] = None,
    layer_head_mask = None,
    output_attentions = False, #: bool = False,
    cache_position = None, #: Optional[torch.LongTensor] = None,
    **kwargs, 
):
    """Input shape: Batch x Time x Channel"""
    bsz, tgt_len, _ = hidden_states.size()

    # Scaling is susceptible to floating point arithmetics' inprecisions
    # which can lead to different results (this is dependent from model
    # to model, e.g. whisper is one such case). We therefore keep the
    # original order of scaling to follow the original implementation
    # and enforce no scaling (1.0) in the attention call below.
    if self.grabbing_mode:
        self.activation_module.grab_activations(hidden_states, 'h1')
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
    else:
        x_q = self.sparse_fns['q'](hidden_states)
        x_k = self.sparse_fns['k'](hidden_states)
        x_k = self.sparse_fns['v'](hidden_states)
        query_states = self.q_proj(x_q) * self.scaling
        key_states = self.k_proj(x_k)
        value_states = self.v_proj(x_k)

    query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

    if past_key_values is not None:
        # save all key/value_states to cache to be re-used for fast auto-regressive generation
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, {"cache_position": cache_position}
        )

    # attention interface here
    attn_weights = torch.matmul(
        query_states,
        key_states.transpose(-1, -2)
    )

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2)

    attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()

    if self.grabbing_mode:
        self.activation_module.grab_activations(attn_output, 'h2')
        attn_output = self.out_proj(attn_output)
    else:
        attn_output = self.sparse_fns['o'](attn_output)
        attn_output = self.out_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights