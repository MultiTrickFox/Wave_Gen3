import config
from ext import pickle_save, pickle_load

from torch import              \
    (tensor, Tensor,
    zeros, ones, eye, randn,
    cat, stack,
    sigmoid, tanh, relu, softmax,
    pow, sqrt,
    abs, sum, norm, mean,
    float32, no_grad)
from torch.nn.init import xavier_normal_

##


def make_layer(in_size, layer_size, act=None):

    layer = (act,
        randn(in_size, layer_size, requires_grad=True, dtype=float32),
        # zeros(1, layer_size,       requires_grad=True, dtype=float32),
    )

    if config.init_xavier and act:
        xavier_normal_(layer[1], gain=5/3 if act=='t' else 1)

    return layer

def prop_layer(layer, inp):

    if not layer[0]: return inp@layer[1] # + layer[2]
    elif layer[0] == 't': return tanh(inp@layer[1]) # + layer[2]
    else: return sigmoid(inp@layer[1]) # + layer[2]


##


def make_model():

    in_size = config.in_size+config.state_size
    # out_size = config.out_size+config.state_size*3

    if not config.hidden_sizes:
        # model = [make_layer(in_size,out_size)]
        model = [(None, cat([make_layer(in_size,config.out_size,'t')[1], make_layer(in_size,config.state_size,'t')[1], make_layer(in_size,config.state_size,'s')[1], make_layer(in_size,config.state_size,'s')[1]], -1))]
    else:
        model = [make_layer(in_size,config.hidden_sizes[0],config.hidden_acts)]
        for i,hs in enumerate(config.hidden_sizes[1:]):
            model.append(make_layer(config.hidden_sizes[i-1],config.hidden_sizes[i],config.hidden_acts))
        # model.append(make_layer(config.hidden_sizes[-1],out_size))
        model.append((None, cat([make_layer(config.hidden_sizes[-1],config.out_size,'t')[1], make_layer(config.hidden_sizes[-1],config.state_size,'t')[1], make_layer(config.hidden_sizes[-1],config.state_size,'s')[1], make_layer(config.hidden_sizes[-1],config.state_size,'s')[1]], -1)))

    return model

def prop_model(model, state, inp):

    io = cat([inp,state],-1)
    for layer in model:
        io = prop_layer(layer, io)

    out = tanh(io[...,:config.out_size])

    state = state * sigmoid(io[...,config.out_size:config.out_size+config.state_size]) + \
            tanh(io[...,config.out_size+config.state_size:config.out_size+config.state_size*2]) * sigmoid(io[...,-config.state_size:])

    return out, state


def empty_state(batch_size=1):

    return zeros(batch_size, config.state_size) if not config.use_gpu else zeros(batch_size, config.state_size).cuda()


##


def respond_to(model, sequences, state=None, training_run=True, extra_steps=0):

    responses = []
    loss = 0
    state = empty_state(len(sequences)) if not state else state

    max_seq_len = max(len(sequence) for sequence in sequences)
    has_remaining = list(range(len(sequences)))

    for t in range(max_seq_len-1):

        has_remaining = [i for i in has_remaining if len(sequences[i][t+1:t+2])]

        inp = stack([sequences[i][t] for i in has_remaining],0)
        lbl = stack([sequences[i][t+1] for i in has_remaining],0)
        partial_state = stack([state[i] for i in has_remaining],0)

        # print(f't: {t}')
        # print(f'inp size: {inp.size()}')
        # print(f'lbl size: {lbl.size()}')
        # print(f'state size: {partial_state.size()}')

        out, partial_state = prop_model(model, partial_state, inp)

        #  print(f'out_size: {out.size()}')
        # print(f'state_out_size: {partial_state.size()}')

        loss_t = sequence_loss(lbl, out)
        for i,l in zip(has_remaining,loss_t):
            l /= len(sequences[i])
        loss += sum(loss_t)

        responses.append([out[has_remaining.index(i),:] if i in has_remaining else None for i in range(len(sequences))])

        for ii,i in enumerate(has_remaining):
            state[i] = partial_state[ii]

    if training_run:

        loss.backward()
        return float(loss)

    else:
        if len(sequences) == 1:

            for t_extra in range(extra_steps):

                out, state = prop_model(model, state, out)

                responses.append(out)

            responses = stack([ee for e in responses for ee in e], dim=0)

        return float(loss), responses


##


def sequence_loss(label, out, do_sum=False):

    loss = pow(label-out, 2) if config.loss_squared else abs(label-out)

    return sum(loss) if do_sum else loss


def sgd(model, lr=None, batch_size=None):

    if not lr: lr = config.learning_rate
    if not batch_size: batch_size = config.batch_size

    with no_grad():

        for layer in model:
            for param in layer[1:]:
                if param.requires_grad:

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
        if do_moments: moments = [[zeros(weight.size()) if not config.use_gpu else zeros(weight.size()).cuda() for weight in layer[1:]] for layer in model]
        if do_variances: variances = [[zeros(weight.size()) if not config.use_gpu else zeros(weight.size()).cuda() for weight in layer[1:]] for layer in model]

    ep_nr +=1

    with no_grad():
            for _, layer in enumerate(model):
                for __, param in enumerate(layer[1:]):
                    if param.requires_grad:

                        lr_ = lr
                        param.grad /= batch_size

                        if do_moments:
                            moments[_][__] = alpha_moment * moments[_][__] + (1-alpha_moment) * param.grad
                            moment_hat = moments[_][__] / (1-alpha_moment**(ep_nr+1))
                        if do_variances:
                            variances[_][__] = alpha_variance * variances[_][__] + (1-alpha_variance) * param.grad**2
                            variance_hat = variances[_][__] / (1-alpha_variance**(ep_nr+1))
                        if do_scaling:
                            lr_ *= norm(param)/norm(param.grad)

                        param -= lr_ * (moment_hat if do_moments else param.grad) / ((sqrt(variance_hat)+epsilon) if do_variances else 1)
                        param.grad = None


##


def save_model(model, path=None):
    from warnings import filterwarnings
    filterwarnings("ignore")
    if not path: path = config.model_path
    if path[-3:]!='.pk': path+='.pk'
    if config.use_gpu:
        moments_ = [[e2.detach().cuda() for e2 in e1] for e1 in moments]
        variances_ = [[e2.detach().cuda() for e2 in e1] for e1 in variances]
        meta = [moments_, variances_]
        model = pull_copy_from_gpu(model)
    else:
        meta = [moments, variances]
    meta.append(ep_nr)
    configs = [[field,getattr(config,field)] for field in dir(config) if field in config.config_to_save]
    pickle_save([model,meta,configs],path)


def load_model(path=None, fresh_meta=None):
    if not path: path = config.model_path
    if not fresh_meta: fresh_meta = config.fresh_meta
    if path[-3:]!='.pk': path+='.pk'
    obj = pickle_load(path)
    if obj:
        model, meta, configs = obj
        if config.use_gpu:
            TorchModel(model).cuda()
        global moments, variances, ep_nr
        if fresh_meta:
            moments, variances, ep_nr = [], [], 0
        else:
            moments, variances, ep_nr = meta
            if config.use_gpu:
                moments = [[e2.cuda() for e2 in e1] for e1 in moments]
                variances = [[e2.cuda() for e2 in e1] for e1 in variances]
        for k_saved, v_saved in configs:
            v = getattr(config, k_saved)
            if v != v_saved:
                if v=='all_losses' and fresh_meta: continue
                print(f'config conflict resolution: {k_saved} {v} -> {v_saved}')
                setattr(config, k_saved, v_saved)
        return model


##


from torch.nn import Module, Parameter

class TorchModel(Module):

    def __init__(self, model):
        super(TorchModel, self).__init__()
        for layer_i, layer in enumerate(model):
            for param_i,param in enumerate(layer[1:]):
                if type(param) != Parameter:
                    param = Parameter(param)
                setattr(self,f'layer{layer_i}_param{param_i}',param)
            setattr(self,f'act{layer_i}',layer[0])

            model[layer_i] = [getattr(self, f'act{layer_i}')] + list(getattr(self, f'layer{layer_i}_param{param_i}') for param_i in range(len(model[layer_i])-1))
        self.model = model

    def forward(self, states, inp):
        prop_model(self.model, states, inp)


def pull_copy_from_gpu(model):
    return [[layer[0]]+list(weight.detach().cpu() for weight in layer[1:]) for layer in model]
