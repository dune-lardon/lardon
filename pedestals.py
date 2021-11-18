import config as cf
import data_containers as dc

import numpy as np


def compute_pedestal_mean(wf_noise):
    return np.mean(wf_noise, axis=-1)

def compute_pedestal_rms(wf_noise):
    return np.std(wf_noise, axis=-1)

def compute_pedestal_raw():    
    wf_noise = set_mask_wf_rms_all()
    mean, std = compute_pedestal_mean(wf_noise), compute_pedestal_rms(wf_noise)
    ped = dc.noise( mean, std )
    dc.evt_list[-1].set_noise_raw(ped)

    dc.data_daq -= mean[:,None]

def compute_pedestal():
    wf_noise = set_mask_wf_rms_all()
    #wf_noise = dc.data_daq
    ped = dc.noise(compute_pedestal_mean(wf_noise), compute_pedestal_rms(wf_noise) )
    dc.evt_list[-1].set_noise_filt(ped)

def set_mask_wf_rms_all():
    # return cleaned waveform from signal

    # subtract mean to waveform
    wf_base  = dc.data_daq - np.mean(dc.data_daq)
    # remove samples exceeding +/- 4 std of the input waveform
    wf_noise = np.where(np.abs(wf_base) < 4*np.std(wf_base), dc.data_daq, np.mean(dc.data_daq))

    return wf_noise

def set_mask_wf_rms(channel=1, to_be_shown=False):
    # for debug purpose only

    wf_base  = dc.data_daq[channel] - np.mean(dc.data_daq[channel])
    wf_noise = np.where(np.abs(wf_base) < 4*np.std(wf_base), dc.data_daq[channel], np.mean(dc.data_daq[channel]))

    if(to_be_shown==True):
      import matplotlib.pyplot as plt

      print("<raw> = ",np.mean(dc.data_daq[channel]),"<noise> = ",np.mean(wf_noise))
      print("S_raw = ",np.std(dc.data_daq[channel]),"S_noise = ",np.std(wf_noise))

      fig, axs = plt.subplots(3, 1)
      axs[0].set_title('raw waveform')
      axs[0].plot(dc.data_daq[channel, :])

      axs[1].set_title('baseline subtracted waveform')
      axs[1].plot(wf_base)
      axs[1].axhline(y = 4*np.std(wf_base), xmin = 0, xmax = len(wf_base), linestyle = '--', color='r')
      axs[1].axhline(y = -4*np.std(wf_base), xmin = 0, xmax = len(wf_base), linestyle = '--', color='r')

      axs[2].set_title('cleanup waveform')
      axs[2].plot(wf_noise)

      for ax in axs:
        ax.grid()

      plt.tight_layout()
      plt.show()

    return wf_noise

