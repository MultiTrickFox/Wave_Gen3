def main():

    import config

    from model import load_model
    model = load_model()
    while not model:
        config.model_path = input('valid model: ')
        model = load_model()

    if config.do_fourier:
        import data as data
    else: import data_direct as data

    d = data.load_data(with_meta=True)
    d, _ = data.split_data(d)

    from random import shuffle
    #shuffle(d)
    d = d[:config.hm_wav_gen]

    for i,(seq,meta) in enumerate(d):

        from model import respond_to
        _, seq = respond_to(model, [seq], training_run=False, extra_steps=config.hm_extra_steps)
        seq = seq.detach()
        if config.use_gpu:
            seq = seq.cpu()
        seq = seq.numpy()

        seq = data.data_to_audio(seq, meta)

        data.write(f'{config.output_file}{i}.wav', config.sample_rate, seq)

if __name__ == '__main__':
    main()