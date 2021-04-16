import config
from ext import now
from model import make_model, respond_to
from model import load_model, save_model
from model import sgd, adaptive_sgd
from data import load_data, split_data, batchify_data

from torch import no_grad

from matplotlib.pyplot import plot, show

##


def main():

    if config.fresh_model:
        config.all_losses = []
        save_model(make_model())
        model = load_model()
        print('created model.',end=' ')
    else:
        model = load_model()
        if not model:
            save_model(make_model())
            model = load_model()
            print('created model.',end=' ')
        else:
            print('loaded model.',end=' ')

    data = load_data()
    data, data_dev = split_data(data)
    # from random import choice
    # from torch import randn
    # data = [[randn(config.in_size) for _ in range(choice(range(2_000,3_000)))] for _ in range(40)]
    # data_dev = []
    if config.max_seq_len: data = [d[:config.max_seq_len] for d in data]

    if not config.batch_size or config.batch_size >= len(data):
        config.batch_size = len(data)
        one_batch = True
    elif config.batch_size < 1:
        config.batch_size = int(len(data)*config.batch_size)
        one_batch = False
    else: one_batch = False

    print(f'hm data: {len(data)}, hm dev: {len(data_dev)}, bs: {config.batch_size}, lr: {config.learning_rate}, \ntraining started @ {now()}')

    data_losss, dev_losss = [], []
    if not one_batch:
        if not config.all_losses: config.all_losses.append(dev_loss(model, data))
        data_losss.append(config.all_losses[-1])
    if config.dev_ratio:
        dev_losss.append(dev_loss(model, data_dev))

    if data_losss or dev_losss:
        print(f'initial loss(es): {data_losss[-1] if data_losss else ""} {dev_losss[-1] if dev_losss else ""}')

    for ep in range(config.hm_epochs):

        loss = 0

        for i, batch in enumerate(batchify_data(data)):

            if not one_batch: print(f'\tbatch {i}, started @ {now()}', flush=True)
            batch_size = sum(len(sequence) for sequence in batch)

            loss += respond_to(model, batch)

            sgd(model, batch_size=batch_size) if config.optimizer == 'sgd' else \
                adaptive_sgd(model, batch_size=batch_size)

        loss /= sum(len(sequence) for sequence in data)

        if not one_batch: loss = dev_loss(model, data)
        data_losss.append(loss)
        config.all_losses.append(loss)
        if config.dev_ratio: dev_losss.append(dev_loss(model, data_dev))

        print(f'epoch {ep}, loss {loss}, dev loss {dev_losss[-1] if config.dev_ratio else ""}, completed @ {now()}', flush=True)
        if config.ckp_per_ep and ((ep+1)%config.ckp_per_ep==0):
                save_model(model,config.model_path+f'_ckp{ep}')

    if one_batch: data_losss.append(dev_loss(model, data))

    print(f'training ended @ {now()} \nfinal losses: {data_losss[-1]}, {dev_losss[-1] if config.dev_ratio else ""}', flush=True)
    show(plot(data_losss))
    if config.dev_ratio:
        show(plot(dev_losss))
    show(plot(config.all_losses))

    return model, [data_losss, dev_losss]


def dev_loss(model, batch):
    with no_grad():
        loss,_ = respond_to(model, batch, training_run=False)
    return loss /sum(len(sequence) for sequence in batch)


##


if __name__ == '__main__':
    main()