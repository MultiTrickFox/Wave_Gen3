from librosa import fft_frequencies
from librosa.core import frames_to_time

## data params

sample_rate = 8_000 # 22_050 # 44_100

fft_bins = 2048
fft_window_len = fft_bins
fft_hop_len = fft_window_len//4

mfcc_bins = 20
mel_bins = 128

silence_thr_db = 0

frequencies_of_bins = list(fft_frequencies(sample_rate, fft_bins))
frequencies_to_pick = frequencies_of_bins
frequency_strength_thr = 1e1
times_of_bins = lambda hm_steps: frames_to_time(range(0,hm_steps), sample_rate, fft_hop_len, fft_bins)

zscore_scale = True
minmax_scale = False
log_scale = False

data_path = 'data'
dev_ratio = 0

## model params

timestep_size = len(frequencies_to_pick)
state_size = timestep_size # 4
in_size = timestep_size + state_size
out_size = timestep_size + state_size*3
creation_info = [in_size, 'ft', out_size]
init_xavier = True

## train params

seq_window_len = 999_999
seq_stride_len = seq_window_len-1
seq_force_ratio = 1

loss_squared = False

learning_rate = 1e-5

batch_size = 0
gradient_clip = 0
hm_epochs = 100
optimizer = 'custom'

model_path = 'models/model'
fresh_model = True
fresh_meta = True
ckp_per_ep = hm_epochs//10

use_gpu = False

## interact params

hm_extra_steps = 1000 #seq_window_len

hm_wav_gen = 5

output_file = 'resp'

##

config_to_save = ['sample_rate', 'fft_bins', 'fft_window_len', 'fft_hop_len', 'mfcc_bins', 'mel_bins',
                  'frequency_strength_thr', 'frequencies_of_bins', 'frequencies_to_pick',
                  'zscore_scale', 'minmax_scale', 'log_scale',
                  'seq_window_len', 'seq_stride_len', 'seq_force_len',
                  'timestep_size', 'in_size', 'out_size', 'state_size', 'creation_info'
                  ]
