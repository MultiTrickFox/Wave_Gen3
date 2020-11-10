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

from dnc import DNC

##


def make_model():
    rnn = DNC(
        input_size=config.in_size,
        hidden_size=config.hidden_size,
        rnn_type='lstm',
        num_layers=config.hm_hiddens,
        nr_cells=config.hm_memory_locs,
        cell_size=config.memory_loc_size,
        read_heads=config.hm_heads,
        batch_first=True,
        gpu_id=0 if config.use_gpu else -1
    )
    return rnn


def respond_to(model, sequences, state=None, training_run=True, extra_steps=0):

    responses = []

    loss = 0
    sequences = deepcopy(sequences)
    if not state: state = model._init_hidden(None,len(sequences),reset_experience=True)
    all_states = [[[],
                   [],
                   []]]
    # for each t (aka this..)
    # (controller_hidden, memory, read_vectors)
    # (each dim(0) if present else None) for 3 of groups

    max_seq_len = max(len(sequence) for sequence in sequences)
    hm_windows = max((max_seq_len-config.seq_window_len)//config.seq_stride_len +1, 1)
    has_remaining = list(range(len(sequences)))

    for i in range(hm_windows):

        window_start = i*config.seq_stride_len
        is_last_window = window_start+config.seq_window_len>=max_seq_len
        window_end = window_start+config.seq_window_len if not is_last_window else max_seq_len

        for window_t in range(window_end-window_start -1):

            seq_force_ratio = config.seq_force_ratio**window_t

            t = window_start+window_t

            has_remaining = [i for i in has_remaining if len(sequences[i][t+1:t+2])]

            if window_t:
                inp = stack([sequences[i][t] for i in has_remaining],dim=0) *seq_force_ratio
                if seq_force_ratio != 1:
                    inp = inp + stack([responses[t-1][i] for i in has_remaining],dim=0) *(1-seq_force_ratio)
            else:
                inp = stack([sequences[i][t] for i in has_remaining], dim=0)

            lbl = stack([sequences[i][t+1] for i in has_remaining], dim=0)


            #state = [stack([layer_state[i] for i in has_remaining], dim=0) for layer_state in state] if state!=[None,None,None] else state

            inp = inp.unsqueeze(1)

            out, state = model(inp, state, reset_experience=window_t)

            partial_hidden, partial_memory, partial_read = state

            out = out.squeeze(1)

            print('input size:',inp.size())
            print('out size:',out.size())

            print(type(partial_hidden), type(partial_memory), type(partial_read))

            print(list(partial_memory.keys())) # ['memory', 'link_matrix', 'precedence', 'read_weights', 'write_weights', 'usage_vector']
            print(list(partial_memory.values())[0].size())

            print('partial sizes:', len(partial_hidden), len(partial_memory.items()), partial_read.size())


            input("Halt here..")


            loss += sequence_loss(lbl, out)

            if t >= len(responses):
                responses.append([out[has_remaining.index(i),:] if i in has_remaining else None for i in range(len(sequences))])
            else:
                responses[t] = [out[has_remaining.index(i),:] if i in has_remaining else None for i in range(len(sequences))]

            for s, ps in zip(states, state):
                for ii,i in enumerate(has_remaining):
                    s[i] = ps[ii]

            if window_t+1 == config.seq_stride_len:
                state_to_transfer = [e.detach() for e in state]

        if not is_last_window:
            states = state_to_transfer
            responses = [[r.detach() if r is not None else None for r in resp] if t>=window_start else resp for t,resp in enumerate(responses)]
        else: break

    if training_run:
        loss.backward()
        return float(loss)

    else:

        if len(sequences) == 1:

            for t_extra in range(extra_steps):
                t = max_seq_len+t_extra-1

                prev_responses = [response[0] for response in reversed(responses[-(config.hm_steps_back+1):])]
                # for i in range(1, config.hm_steps_back+1):
                #     if len(sequences[0][t-1:t]):
                #         prev_responses[i-1] = sequences[0][t-1]

                inp = cat([response.view(1,-1) for response in prev_responses],dim=1) # todo: stack ?
                
                out, state = prop_model(model, state, inp)

                responses.append([out.view(-1)])

            responses = stack([ee for e in responses for ee in e], dim=0)

        return float(loss), responses


def sequence_loss(label, out, do_stack=False):

    if do_stack:
        label = stack(label,dim=0)
        out = stack(out, dim=0)

    loss = pow(label-out, 2) if config.loss_squared else (label-out).abs()

    return loss.sum()


def sgd(model, lr=None, batch_size=None):

    if not lr: lr = config.learning_rate
    if not batch_size: batch_size = config.batch_size

    with no_grad():

        for param in model.parameters():

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
        if do_moments: moments = [[zeros(weight.size()) if not config.use_gpu else zeros(weight.size()).cuda() for weight in layer._asdict().values()] for layer in model]
        if do_variances: variances = [[zeros(weight.size()) if not config.use_gpu else zeros(weight.size()).cuda() for weight in layer._asdict().values()] for layer in model]

    ep_nr +=1

    with no_grad():
            for _,param in enumerate(model.parameters()):

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


def pull_copy_from_gpu(model):
    return model.cpu()
