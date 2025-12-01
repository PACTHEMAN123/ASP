from teal_utils import (
    Distribution,
    SparsifyFn,
    TealActivation,
    get_module_device
)

from torch import nn

import types

def _monkeypatch_fc(fc, file_path, grabbing_mode=False):
    orig_forward = fc.forward 
    fc.forward_old = orig_forward
    fc.forward = types.MethodType(_fc_forward, fc)

    fc.file_path = file_path
    fc.grabbing_mode = grabbing_mode

    if not grabbing_mode:
        fc.distrs = {}
        fc.distrs['h'] = Distribution(file_path, hidden_type='h')

        fc.sparse_fns = nn.ModuleDict({
            'fc': SparsifyFn(fc.distrs['h']).to(get_module_device(fc))
        })

    fc.activation_module = TealActivation(file_path)

    return fc



def _fc_forward(self, x, activation_module=None):
    if hasattr(self, 'config') and self.config.pretraining_tp > 1:
        # TODO: UNTESTED

        assert 1 == 0, "Pretraining TP > 1 not implemented yet"
    else:
        if self.grabbing_mode:
            self.activation_module.grab_activations(x, 'h')
            out = self.forward_old(x)
        else:
            x_fc = self.sparse_fns['fc'](x)
            out = self.forward_old(x_fc)

    return out
