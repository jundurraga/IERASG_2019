import sys
sys.path.append('/home/jundurraga/Documents/source_code/pysignal_generator')
from matplotlib import pyplot as plt
import pysignal_generator.noise_functions as nf
import matplotlib
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.ticker as ticker
import pyfftw
import multiprocessing

__author__ = 'jundurraga'


def fftw_hilbert(x, axis=0):
    _fft = pyfftw.builders.fft(x, overwrite_input=False, planner_effort='FFTW_ESTIMATE', axis=axis,
                               threads=multiprocessing.cpu_count())
    fx = _fft()
    N = fx.shape[axis]
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if len(x.shape) > 1:
        ind = [np.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[ind]
    _ifft = pyfftw.builders.ifft(fx * h, overwrite_input=False, planner_effort='FFTW_ESTIMATE', axis=axis,
                                 threads=multiprocessing.cpu_count())
    return _ifft()

matplotlib.rcParams.update({'font.size': 8})

uul = {'itd': -0.0005,
       'alt_itd': 0.0005,
       'modulation_index': 0.0,
       'modulation_frequency': 41.0,
       'carrier_frequency': 520.0,
       'cycle_rate': 16384.0,
       'fs': 48000.0,
       'duration': 4.109375,
       'itd_mod_rate': 6.0
       }

pul = {'itd': -0.0005,
       'alt_itd': 0.0005,
       'modulation_index': 1.0,
       'modulation_frequency': 41.0,
       'carrier_frequency': 520.0,
       'cycle_rate': 16384.0,
       'fs': 48000.0,
       'duration': 4.109375,
       'itd_mod_rate': 6.0
       }

ustimuli, time = nf.generate_noise_alt_delay_signal(uul)
stimuli, time = nf.generate_noise_alt_delay_signal(pul)
ure = np.expand_dims(ustimuli[:, 1], axis=1)
ule = np.expand_dims(ustimuli[:, 0], axis=1)

re = np.expand_dims(stimuli[:, 1], axis=1)
le = np.expand_dims(stimuli[:, 0], axis=1)

time *= 1000.0
env_ure = np.abs(fftw_hilbert(ure, axis=0))
env_ule = np.abs(fftw_hilbert(ule, axis=0))
env_re = np.abs(fftw_hilbert(re, axis=0))
env_le = np.abs(fftw_hilbert(le, axis=0))
inch = 2.54
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=False)
fig.set_size_inches(14/inch, 14 / inch)
ax1.plot(time, ule, 'b')
ax1.plot(time, ure, 'r')
ax2.plot(time, le, 'b')
ax2.plot(time, re, 'r')
alt_period = 1 / pul['itd_mod_rate'] * 1000
n_cycles = time[-1] / (alt_period)
b = alt_period
n_cycles_plot = 4.0
not_zoomed_time = n_cycles_plot * alt_period
ipd_factor_l = np.abs(pul['itd']) / 0.0005
ipd_factor_r = np.abs(pul['alt_itd']) / 0.0005
r = alt_period / 2.0 / 1.2

for _n in np.arange(0, n_cycles, 2):
    _ini = np.floor(_n * alt_period * pul['fs'] / 1000.0).astype(np.int)
    _end = np.floor((_n + 1) * alt_period * pul['fs'] / 1000.0).astype(np.int)
    ax1.plot(np.squeeze(time[_ini:_end, 0]), np.squeeze(env_ule[_ini:_end, 0]), color='b')
    ax1.plot(np.squeeze(time[_ini:_end, 0]), np.squeeze(env_ure[_ini:_end, 0]), color='r')
    ax2.plot(np.squeeze(time[_ini:_end, 0]), np.squeeze(env_le[_ini:_end, 0]), color='b')
    ax2.plot(np.squeeze(time[_ini:_end, 0]), np.squeeze(env_re[_ini:_end, 0]), color='r')
    ax3.fill_between(np.squeeze(time[_ini:_end, 0]), np.squeeze(-env_le[_ini:_end, 0]),
                     np.squeeze(env_le[_ini:_end, 0]), facecolor='b')
    # Plot rectangle
    ax1.add_patch(Rectangle((time[_ini, 0], -1.1), alt_period, .1, color='b'))
    ax2.add_patch(Rectangle((time[_ini, 0], -1.1), alt_period, .1, color='b'))
    ax3.add_patch(Rectangle((time[_ini, 0], -1.1), alt_period, .1, color='b'))
    ax4.add_patch(Rectangle((time[_ini, 0], 0), alt_period, .01, color='b'))
    # plot nose
    ax4.plot(time[_ini, 0] + b / 2.0, b / 2.0 + r * ipd_factor_l, '^', color='gray', markersize=10, zorder=0)
    # plot head
    head = plt.Circle((time[_ini, 0] + b / 2.0, b / 2.0), r * ipd_factor_l, color='gray')
    ax4.add_patch(head)
    # Plot sound source
    ax4.plot(time[_ini, 0] + b / 2.0 - r * ipd_factor_l / 1.5, b / 2.0, 'o', color='b', markersize=8)


for _n in np.arange(1, n_cycles, 2):
    _ini = np.floor(_n * alt_period * pul['fs'] / 1000.0).astype(np.int)
    _end = np.floor((_n + 1) * alt_period * pul['fs'] / 1000.0).astype(np.int)
    ax1.plot(np.squeeze(time[_ini:_end, 0]), np.squeeze(env_ule[_ini:_end, 0]), color='b')
    ax1.plot(np.squeeze(time[_ini:_end, 0]), np.squeeze(env_ure[_ini:_end, 0]), color='r')
    ax2.plot(np.squeeze(time[_ini:_end, 0]), np.squeeze(env_re[_ini:_end, 0]), color='r')
    ax2.plot(np.squeeze(time[_ini:_end, 0]), np.squeeze(env_le[_ini:_end, 0]), color='b')
    ax3.fill_between(np.squeeze(time[_ini:_end, 0]), np.squeeze(-env_re[_ini:_end, 0]),
                     np.squeeze(env_re[_ini:_end, 0]), facecolor='r')
    # plot nose
    ax4.plot(time[_ini, 0] + b / 2.0, b / 2.0 + r * ipd_factor_l, '^', color='gray', markersize=10, zorder=0)
    # plot head
    head = plt.Circle((time[_ini, 0] + b / 2.0, b / 2.0), r * ipd_factor_l, color='gray')
    ax4.add_patch(head)
    # Plot sound source
    ax4.plot(time[_ini, 0] + b / 2.0 + r * ipd_factor_l / 1.5, b / 2.0, 'o', color='r', markersize=8)
    # Plot rectangle
    ax1.add_patch(Rectangle((time[_ini, 0], -1.1), alt_period, .1, color='r'))
    ax2.add_patch(Rectangle((time[_ini, 0], -1.1), alt_period, .1, color='r'))
    ax3.add_patch(Rectangle((time[_ini, 0], -1.1), alt_period, .1, color='r'))
    ax4.add_patch(Rectangle((time[_ini, 0], 0), alt_period, .01, color='r'))

# set xticks for all panels
ax1.set_xticks(np.arange(0, time[-1], 0.25 * 1.0 / pul['modulation_frequency'] / 2.0 * 1000.0))
ax2.set_xticks(np.arange(0, time[-1], 1.0 / pul['modulation_frequency'] / 2.0 * 1000.0))
ax3.set_xticks(np.arange(0, time[-1], 1.0 / pul['modulation_frequency'] * 2.0 * 1000.0))
ax4.set_xticks(np.arange(0, time[-1], 1.0 / pul['modulation_frequency'] * 2.0 * 1000.0))

# set yticks for all panels
ax1.set_yticks([])
ax2.set_yticks([])
ax3.set_yticks([])
ax4.set_yticks([])

ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax4.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))


# set zooms for panels
ax4.set_ylim(ax4.get_xlim() / n_cycles)
ax1.set_xlim([n_cycles_plot / 2.0 * alt_period - 4 * 0.25 * 1.0 / pul['modulation_frequency'] * 1000.0,
              n_cycles_plot / 2.0 * alt_period + 4 * 0.25 * 1.0 / pul['modulation_frequency'] * 1000.0])
ax2.set_xlim([n_cycles_plot / 2.0 * alt_period - 4 * 1.0 / pul['modulation_frequency'] * 1000.0,
              n_cycles_plot / 2.0 * alt_period + 4 * 1.0 / pul['modulation_frequency'] * 1000.0])
ax2.axvline(x=n_cycles_plot / 2.0 * alt_period - 4 * 0.25 * 1.0 / pul['modulation_frequency'] * 1000.0, color='k', zorder=0)
ax2.axvline(x=n_cycles_plot / 2.0 * alt_period + 4 * 0.25 * 1.0 / pul['modulation_frequency'] * 1000.0, color='k', zorder=0)

ax3.set_xlim([0, not_zoomed_time])
ax4.set_xlim([0, not_zoomed_time])

ax3.axvline(x=n_cycles_plot / 2.0 * alt_period - 1.0 / pul['modulation_frequency'] * 1000.0, color='k', zorder=0)
ax3.axvline(x=n_cycles_plot / 2.0 * alt_period + 1.0 / pul['modulation_frequency'] * 1000.0, color='k', zorder=0)

ax1.set_ylim([-1.1, 1.1])
ax2.set_ylim([-1.1, 1.1])
ax3.set_ylim([-1.1, 1.1])
ax1.set_title('[A] Unmodulated Noise (' + '{:.1f}'.format(np.diff(ax1.get_xlim())[0]) + ' ms)')
ax2.set_title('[B] AM Noise (' + '{:.1f}'.format(np.diff(ax2.get_xlim())[0]) + ' ms)')
ax3.set_title('[C] AM Noise (' + '{:.1f}'.format(np.diff(ax3.get_xlim())[0]) + ' ms)')
ax4.set_title('[D] Intercranial Image (' + '{:.1f}'.format(np.diff(ax4.get_xlim())[0]) + ' ms)')

ax1.set_ylabel('Amplitude')
ax2.set_ylabel('Amplitude')
ax3.set_ylabel('Amplitude')
ax4.set_ylabel('')
ax4.set_xlabel('Time [ms]')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)

ax1.tick_params(axis='x', which='both', top='off')
ax2.tick_params(axis='x', which='both', top='off')
ax3.tick_params(axis='x', which='both', top='off')
ax4.tick_params(axis='x', which='both', top='off')

plt.tight_layout()
fig.savefig('/home/jundurraga/pCloudDrive/Documents/Presentations/international_binaural_workshop_2018/figures/stimulus_example.png', bbox_inches='tight')


