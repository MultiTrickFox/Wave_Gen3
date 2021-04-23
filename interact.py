def main():

    import config

    from model import load_model
    model = load_model()
    while not model:
        config.model_path = input('valid model: ')
        model = load_model()

    from data import load_data, split_data
    d = load_data(with_meta=True)
    d, _ = split_data(d)

    # from random import shuffle
    # shuffle(d)
    d = d[:config.hm_output_file]

    for i,(seq,meta) in enumerate(d):

        from model import respond_to
        _, seq = respond_to(model, [seq[:config.hm_extra_steps]], training_run=False, extra_steps=config.hm_extra_steps)
        seq = seq.detach()
        if config.use_gpu:
            seq = seq.cpu()
        seq = seq.numpy()

        from data import data_to_audio, write
        seq = data_to_audio(seq, meta)
        write(f'{config.output_file}{i}.wav', config.sample_rate, seq)

if __name__ == '__main__':
    main()