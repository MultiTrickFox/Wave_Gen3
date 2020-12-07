import config
from ext import pickle_save, pickle_load, now

from torch import tensor, Tensor, cat, stack
from torch import zeros, ones, eye, randn
from torch import sigmoid, tanh, relu, softmax
from torch import pow, log, exp, sqrt, norm, mean, abs
from torch import float32, no_grad
from torch.nn.init import xavier_normal_

from torch.distributions import Normal, Beta
from torch import lgamma ; gamma = lambda x: exp(lgamma(x))

from collections import namedtuple
from copy import deepcopy
from math import ceil


##


FF = namedtuple('FF', 'w')
FFS = namedtuple('FFS', 'w')
FFT = namedtuple('FFT', 'w')


def make_Flayer(in_size, layer_size, act=None):

    layer_type = FF if not act else (FFS if act=='s' else FFT)

    layer = layer_type(
        randn(in_size, layer_size, requires_grad=True, dtype=float32),
    )

    if config.init_xavier:
        if act == 's':
            xavier_normal_(layer.w)
        elif act == 't':
            xavier_normal_(layer.w, gain=5/3)

    return layer


make_layer = {
    'f': make_Flayer,
    'fs': lambda i,l: make_Flayer(i,l,act='s'),
    'ft': lambda i,l: make_Flayer(i,l,act='t'),
}


def prop_Flayer(layer, inp):

    return inp@layer.w


prop_layer = {
    FF: prop_Flayer,
    FFS: lambda l,i: sigmoid(prop_Flayer(l,i)),
    FFT: lambda l,i: tanh(prop_Flayer(l,i)),
}


def make_model(creation_info=None):

    if not creation_info: creation_info = config.creation_info

    layer_sizes = [e for e in creation_info if type(e)==int]
    layer_types = [e for e in creation_info if type(e)==str]

    return [make_layer[layer_type](layer_sizes[i], layer_sizes[i+1]) for i,layer_type in enumerate(layer_types)]

def prop_model(model, states, inp):

    out = cat([inp,states],dim=-1)

    for layer in model:
        
        out = prop_Flayer(layer,out)
        # dropout(out, inplace=True)

    states = states + out[...,config.timestep_size:config.timestep_size+config.state_size]

    out = out[...,:config.timestep_size]

    # states = states * sigmoid(out[...,-states.size(-1)*3:-states.size(-1)*2]) + \
    #          tanh(out[...,-states.size(-1):]) * sigmoid(out[...,-states.size(-1)*2:-states.size(-1)])
    #
    # out = states * sigmoid(out[...,:-states.size(-1)*3])

    # states = tanh(out[...,config.timestep_size:config.timestep_size+config.state_size])
    #
    # out = tanh(out[...,:config.timestep_size])

    return out, states


def respond_to(model, sequences, state=None, training_run=True, extra_steps=0):
    responses = []

    loss = 0
    sequences = deepcopy(sequences)
    if not state:
        state = empty_state(model, len(sequences))

    max_seq_len = max(len(sequence) for sequence in sequences)
    hm_windows = ceil(max_seq_len / config.seq_stride_len)
    has_remaining = list(range(len(sequences)))

    for i in range(hm_windows):

        window_start = i * config.seq_stride_len
        is_last_window = window_start + config.seq_window_len >= max_seq_len
        window_end = window_start + config.seq_window_len if not is_last_window else max_seq_len

        for window_t in range(window_end - window_start - 1):

            seq_force_ratio = config.seq_force_ratio ** window_t

            t = window_start + window_t

            has_remaining = [i for i in has_remaining if len(sequences[i][t + 1:t + 2])]

            if window_t:
                inp = stack([sequences[i][t] for i in has_remaining], dim=0) * seq_force_ratio
                if seq_force_ratio != 1:
                    inp = inp + stack([responses[t-1][i] for i in has_remaining],dim=0) * (1-seq_force_ratio)
            else:
                inp = stack([sequences[i][t] for i in has_remaining], dim=0)

            lbl = stack([sequences[i][t+1] for i in has_remaining], dim=0)

            partial_state = cat([state[i:i+1,:] for i in has_remaining], dim=0)

            out, partial_state = prop_model(model, partial_state, inp)

            loss += sequence_loss(lbl, out)

            if t >= len(responses):
                responses.append(
                    [out[has_remaining.index(i),:] if i in has_remaining else None for i in range(len(sequences))])
            else:
                responses[t] = [out[has_remaining.index(i),:] if i in has_remaining else None for i in
                                range(len(sequences))]

            for s, ps in zip(state, partial_state):
                for ii, i in enumerate(has_remaining):
                    s[i:i+1] = ps[ii:ii+1]

            if window_t+1==config.seq_stride_len:
                state_to_transfer = state.detach()

        if not is_last_window:
            state = state_to_transfer
            responses = [[r.detach() if r is not None else None for r in resp] if t >= window_start else resp for
                         t, resp in enumerate(responses)]
        else:
            break

    if training_run:
        loss.backward()
        return float(loss)

    else:

        if len(sequences) == 1:

            for t_extra in range(extra_steps):
                t = max_seq_len + t_extra - 1

                prev_responses = [response[0] for response in reversed(responses[-1:])]
                # for i in range(1, config.hm_steps_back+1):
                #     if len(sequences[0][t-1:t]):
                #         prev_responses[i-1] = sequences[0][t-1]

                inp = cat([response.view(1, -1) for response in prev_responses], dim=1)  # todo: stack ?

                out, state = prop_model(model, state, inp)

                responses.append([out.view(-1)])

            responses = stack([ee for e in responses for ee in e], dim=0)

        return float(loss), responses


def sequence_loss(label, out, do_stack=False):

    if do_stack:
        label = stack(label,dim=0)
        out = stack(out,dim=0)

    loss = pow(label-out,2) if config.loss_squared else (label-out).abs()

    return loss.sum()


def sgd(model, lr=None, batch_size=None):

    if not lr: lr = config.learning_rate
    if not batch_size: batch_size = config.batch_size

    with no_grad():

        for param in model:

            param = param.w

            param.grad /=batch_size

            if config.gradient_clip:
                param.grad.clamp(min=-config.gradient_clip,max=config.gradient_clip)

            param -= lr * param.grad
            param.grad = None


moments, variances, ep_nr = [], [], 0


def adaptive_sgd(model, lr=None, batch_size=None,
                 alpha_moment=0.9, alpha_variance=0.999, epsilon=1e-8,
                 do_moments=True, do_variances=True, do_scaling=False):

    if not lr: lr = config.learning_rate
    if not batch_size: batch_size = config.batch_size

    global moments, variances, ep_nr
    if not (moments or variances):
        if do_moments: moments = [zeros(layer.w.size()) if not config.use_gpu else zeros(layer.w.size()).cuda() for layer in model]
        if do_variances: variances = [zeros(layer.w.size()) if not config.use_gpu else zeros(layer.w.size()).cuda() for layer in model]

    ep_nr +=1

    with no_grad():
            for _,param in enumerate(model):

                param = param.w

                lr_ = lr
                param.grad /= batch_size

                if do_moments:
                    moments[_] = alpha_moment * moments[_] + (1-alpha_moment) * param.grad
                    moment_hat = moments[_] / (1-alpha_moment**(ep_nr+1))
                if do_variances:
                    variances[_] = alpha_variance * variances[_] + (1-alpha_variance) * param.grad**2
                    variance_hat = variances[_] / (1-alpha_variance**(ep_nr+1))
                if do_scaling:
                    lr_ *= norm(param)/norm(param.grad)

                param -= lr_ * (moment_hat if do_moments else param.grad) / ((sqrt(variance_hat)+epsilon) if do_variances else 1)
                param.grad = None


def empty_state(model, batch_size=1):
    return randn(batch_size,config.state_size)


def load_model(path=None, fresh_meta=None):
    if not path: path = config.model_path
    if not fresh_meta: fresh_meta = config.fresh_meta
    path = path+'.pk'
    obj = pickle_load(path)
    if obj:
        model, meta, configs = obj
        if config.use_gpu:
            model.cuda()
        global moments, variances, ep_nr
        if fresh_meta:
            moments, variances, ep_nr = [], [], 0
        else:
            moments, variances, ep_nr = meta
            if config.use_gpu:
                moments = [e.cuda() for e in moments]
                variances = [e.cuda() for e in variances]
        for k_saved, v_saved in configs:
            v = getattr(config, k_saved)
            if v != v_saved:
                print(f'config conflict resolution: {k_saved} {v} -> {v_saved}')
                setattr(config, k_saved, v_saved)
        return model

def save_model(model, path=None):
    from warnings import filterwarnings
    filterwarnings("ignore")
    if not path: path = config.model_path
    path = path+'.pk'
    if config.use_gpu:
        moments_ = [e.detach().cuda() for e in moments]
        variances_ = [e.detach().cuda() for e in variances]
        meta = [moments_, variances_]
        model = pull_copy_from_gpu(model)
    else:
        meta = [moments, variances]
    meta.append(ep_nr)
    configs = [[field,getattr(config,field)] for field in dir(config) if field in config.config_to_save]
    pickle_save([model,meta,configs],path)


##


from torch.nn import Module, Parameter

class TorchModel(Module):

    def __init__(self, model):
        super(TorchModel, self).__init__()
        for layer_name, layer in enumerate(model):
            for field_name, field in layer._asdict().items():
                if type(field) != Parameter:
                    field = Parameter(field)
                setattr(self,f'layer{layer_name}_field{field_name}',field)
            setattr(self,f'layertype{layer_name}',type(layer))

            model[layer_name] = (getattr(self, f'layertype{layer_name}')) \
                (*[getattr(self, f'layer{layer_name}_field{field_name}') for field_name in getattr(self, f'layertype{layer_name}')._fields])
        self.model = model

    def forward(self, states, inp):
        prop_model(self.model, states, inp)


def pull_copy_from_gpu(model):
    model_copy = [type(layer)(*[weight.detach().cpu() for weight in layer._asdict().values()]) for layer in model]
    for layer in model_copy:
        for w in layer._asdict().values():
            w.requires_grad = True
    return model_copy

