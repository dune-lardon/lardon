import config as cf
import data_containers as dc

import numpy as np
import numba as nb
import numexpr as ne


@nb.jit(nopython = True)
def compute_pedestal_nb(data, mask):
    """ do not remove the @jit above, the code would then take ~40 s to run """
    shape = data.shape

    """ to avoid cases where the rms goes to 0"""
    min_val = 1e-5

    mean  = np.zeros(shape[:-1])
    res   = np.zeros(shape[:-1])
    for idx,v in np.ndenumerate(res):
        ch = data[idx]
        ma  = mask[idx]
        """ use the assumed mean method """
        K = ch[0]
        n, Ex, Ex2, tmp = 0., 0., 0., 0.
        for x,v in zip(ch,ma):
            if(v == True):
                n += 1
                tmp += x
                Ex += x-K
                Ex2 += (x-K)*(x-K)

        """cut at min. 10 pts to compute a proper RMS"""
        if( n < 10 ):
            mean[idx] = -1.
            res[idx] = -1.
        else:
            val = np.sqrt((Ex2 - (Ex*Ex)/n)/(n-1))
            res[idx] = min_val if val < min_val else val
            mean[idx] = tmp/n

    return mean, res

def compute_pedestal(noise_type='None'):
    mean, std = compute_pedestal_nb(dc.data_daq, dc.mask_daq)
    ped = dc.noise( mean, std )

    dc.data_daq -= mean[:,None]

    if(noise_type=='raw'):
      dc.evt_list[-1].set_noise_raw(ped)

    elif(noise_type=='filt'): 
        dc.evt_list[-1].set_noise_filt(ped)
    else:
      print('ERROR: You must set a noise type for pedestal setting')
      sys.exit()

def update_mask(thresh):
    dc.mask_daq = ne.evaluate( "where((abs(data) > thresh*rms) | ~alive_chan, 0, 1)", global_dict={'data':dc.data_daq, 'alive_chan':dc.alive_chan, 'rms':dc.evt_list[-1].noise_filt.ped_rms[:,None]}).astype(bool)


def set_mask_wf_rms_all():
    # set signal mask bins from threshold - simplest method

    # subtract mean to waveform
    wf_base  = dc.data_daq - np.mean(dc.data_daq)
    # remove samples exceeding +/- 4 std of the input waveform
    dc.mask_daq = np.where(np.abs(wf_base) < 4*np.std(wf_base), 1, 0)


def set_mask_wf_rms(channel=1, to_be_shown=False):
    # plot cleaned waveform from signal 
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

