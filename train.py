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

    data, data_dev = split_data(load_data())

    if not config.batch_size or config.batch_size >= len(data):
        config.batch_size = len(data)
        one_batch = True
    elif config.batch_size < 1:
        config.batch_size = int(len(data)*config.batch_size)
        one_batch = False
    else: one_batch = False

    if not config.batch_size:
        config.batch_size = len(data_dev) if config.dev_ratio else len(data)
    elif config.batch_size > len(data):
        config.batch_size = len(data)
    
    print(f'hm data: {len(data)}, hm dev: {len(data_dev)}, bs: {config.batch_size}, lr: {config.learning_rate}, \ntraining started @ {now()}')

    data = [e[:100] for e in data] # todo: rm me

    data_losss, dev_losss = [], []
    if 1: #config.batch_size != len(data):
        data_losss.append(dev_loss(model, data))
    if config.dev_ratio:
        dev_losss.append(dev_loss(model, data_dev))

    if 1: # data_losss or dev_losss:
        print(f'initial loss(es): {data_losss[-1] if data_losss else ""} {dev_losss[-1] if dev_losss else ""}')

    for ep in range(config.hm_epochs):

        loss = 0

        for i, batch in enumerate(batchify_data(data)):

            batch_size = sum(len(sequence) for sequence in batch)

            loss += respond_to(model, batch)
            sgd(model, batch_size=batch_size) if config.optimizer == 'sgd' else \
                adaptive_sgd(model, batch_size=batch_size)

        loss /= sum(len(sequence) for sequence in data)
        if not one_batch: loss = dev_loss(model, data)
        data_losss.append(loss)
        if config.dev_ratio:
            dev_losss.append(dev_loss(model, data_dev))

        print(f'epoch {ep}, loss {data_losss[-1]}, dev loss {dev_losss[-1] if config.dev_ratio else ""}, completed @ {now()}', flush=True)
        if config.ckp_per_ep and ((ep+1)%config.ckp_per_ep==0):
                save_model(model,config.model_path+f'_ckp{ep}')

    print(f'training ended @ {now()} \nfinal losses: {data_losss[-1]}, {dev_losss[-1] if config.dev_ratio else ""}', flush=True)
    show(plot(data_losss))
    if config.dev_ratio:
        show(plot(dev_losss))

    return model, [data_losss, dev_losss]


def dev_loss(model, batch):
    with no_grad():
        loss,_ = respond_to(model, batch, training_run=False)
    return loss /sum(len(sequence) for sequence in batch)





if __name__ == '__main__':
    main()