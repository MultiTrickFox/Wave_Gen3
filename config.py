from librosa import fft_frequencies
from librosa.core import frames_to_time


## data params

sample_rate = 8_000 # 22_050 # 44_100

do_fourier = True

fft_bins = 2048
fft_window_len = fft_bins
fft_hop_len = fft_window_len//4

mfcc_bins = 20
mel_bins = 128

silence_thr_db = 10

frequencies_of_bins = list(fft_frequencies(sample_rate, fft_bins))
frequencies_to_pick = [] # [109.375, 113.28125, 121.09375, 125.0, 128.90625, 136.71875, 140.625, 144.53125, 148.4375, 152.34375, 160.15625, 164.0625, 167.96875, 171.875, 175.78125, 179.6875, 183.59375, 187.5, 191.40625, 195.3125, 199.21875, 203.125, 207.03125, 210.9375, 214.84375, 218.75, 222.65625, 226.5625, 230.46875, 234.375, 238.28125, 242.1875, 246.09375, 250.0, 253.90625, 257.8125, 261.71875, 265.625, 269.53125, 273.4375, 277.34375, 281.25, 285.15625, 289.0625, 292.96875, 296.875, 300.78125, 304.6875, 308.59375, 312.5, 316.40625, 320.3125, 324.21875, 328.125, 332.03125, 335.9375, 339.84375, 343.75, 347.65625, 351.5625, 355.46875, 359.375, 363.28125, 367.1875, 371.09375, 375.0, 378.90625, 382.8125, 386.71875, 390.625, 394.53125, 398.4375, 402.34375, 406.25, 410.15625, 414.0625, 417.96875, 421.875, 425.78125, 429.6875, 433.59375, 437.5, 441.40625, 445.3125, 449.21875, 453.125, 457.03125, 460.9375, 464.84375, 468.75, 472.65625, 476.5625, 480.46875, 484.375, 488.28125, 492.1875, 496.09375, 500.0, 503.90625, 507.8125, 511.71875, 515.625, 519.53125, 523.4375, 527.34375, 546.875, 550.78125, 554.6875, 558.59375, 562.5, 566.40625, 570.3125, 574.21875, 582.03125, 585.9375, 589.84375, 593.75, 597.65625, 609.375, 613.28125, 617.1875, 621.09375, 625.0, 628.90625, 632.8125, 636.71875, 640.625, 644.53125, 656.25, 660.15625, 664.0625, 667.96875, 671.875, 675.78125, 679.6875, 683.59375, 687.5, 691.40625, 695.3125, 699.21875, 703.125, 707.03125, 734.375, 738.28125, 742.1875, 746.09375, 750.0, 753.90625, 781.25, 785.15625, 789.0625, 792.96875, 828.125, 832.03125, 835.9375, 839.84375, 843.75, 847.65625, 851.5625, 855.46875, 882.8125, 886.71875, 929.6875, 933.59375, 941.40625, 984.375, 988.28125, 992.1875, 996.09375, 1000.0, 1003.90625, 1476.5625, 1480.46875, 1566.40625, 1570.3125, 1574.21875, 1964.84375, 2359.375]
frequency_strength_thr = 1e2
times_of_bins = lambda hm_steps: frames_to_time(range(0,hm_steps), sample_rate, fft_hop_len, fft_bins)

zscore_scale = do_fourier
minmax_scale = False
log_scale = False

data_path = 'data'
dev_ratio = 0


## model params

timestep_size = len(frequencies_to_pick)

state_size = timestep_size//2
state_out_delta = False

in_size = timestep_size
out_size = timestep_size

hm_heads = 10

init_xavier = False


## train params

seq_window_len = 999_999
seq_stride_len = seq_window_len-1
seq_force_ratio = 1 #0

loss_squared = True

learning_rate = 1e-4 #1e-3 #2e-5

batch_size = 0
gradient_clip = 0
hm_epochs = 100
optimizer = 'custom'

model_path = 'models/model'
fresh_model = True
fresh_meta = True
ckp_per_ep = hm_epochs//10

use_gpu = False

all_losses = []


## interact params

hm_extra_steps = 1000 #seq_window_len

hm_wav_gen = 5

output_file = 'resp'


##

config_to_save = ['do_fourier', 'sample_rate', 'fft_bins', 'fft_window_len', 'fft_hop_len',
                  'frequency_strength_thr', 'frequencies_of_bins', 'frequencies_to_pick',
                  'mfcc_bins', 'mel_bins', 'zscore_scale', 'minmax_scale', 'log_scale',
                  'state_size', 'state_out_delta', 'timestep_size', 'in_size', 'out_size', 'hm_heads',
                  'seq_window_len', 'seq_stride_len', 'all_losses'
                  ]

