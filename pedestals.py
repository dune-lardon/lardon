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


def refine_mask(pars,debug=False):
    if(debug == True): import matplotlib.pyplot as plt

    mask = dc.mask_daq
    for ch in range(len(mask)):
       data = dc.data_daq[ch]
       rms  = dc.evt_list[-1].noise_filt.ped_rms[ch]
       view = dc.chmap[ch].view

       if(view == 2): mask_collection_signal(ch,mask,data,pars.ped_rise_thr[view],pars.ped_ramp_thr[view],rms*pars.ped_amp_thr[view],pars.ped_dt_thr)  
       else:          mask_induction_signal(ch,mask,data,pars.ped_rise_thr[view],pars.ped_ramp_thr[view],rms*pars.ped_amp_thr[view],pars.ped_dt_thr,pars.ped_zero_cross_thr)  
       if(debug==True and ch > 1 and (ch % 50) == 0):
           nrows, ncols = 10, 5
           fig = plt.figure()
           for it,channel in enumerate(range(ch-50,ch)):    
                  title = 'v'+str(view)+'-ch'+str(channel)
                  ax = fig.add_subplot(nrows, ncols, it+1)
                  ax.set_title(title)
                  ax.plot(dc.data_daq[channel],'o',markersize=0.2)
                  ax.axhline(y=dc.evt_list[-1].noise_filt.ped_rms[channel]*pars.ped_amp_thr[dc.chmap[channel].view], color='gray')
                  ax.axhline(y=0, color='lightgray')
                  ax.axhline(y=-dc.evt_list[-1].noise_filt.ped_rms[channel]*pars.ped_amp_thr[dc.chmap[channel].view], color='gray')
                  ax.plot(np.max(dc.data_daq[channel, :])*dc.mask_daq[channel,:],linestyle='-',linewidth=1,color='r')
                  ax.set(xlabel=None,ylabel=None)
                  ax.axis('off')
           plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.95, wspace=0.05, hspace=1)
           plt.show()


    dc.mask_daq = mask

@nb.jit('(int64,boolean[:,:],float64[:],int64,int64,float64,int64)',nopython = True)
def mask_collection_signal(ch,mask,data,rise_thr,ramp_thr,amp_thr,dt_thr):

      tmp_start   = 0 # store candidate signal start to be registered when stop is found
      trig_pos    = 0 # store position at which sample exceed  amp_thrs*ped_rms - 0 otherwise
      find_pos    = 0 # counter for consecutive positive sample 
      ramp        = 0 # counter 3 consecutive rising samples
      ongoing     = 0 # flag triggered when signal start found
      oldval1     = 0 # store previous sample 
      oldval2     = 0 # store pre-previous sample
      zero_cross_check = 0

      for x,val in np.ndenumerate(data):

        # set positive signal trigger
        if(val >= amp_thr):   trig_pos = x[0]
        # set back positive trigger to zero if a certain period is over
        elif(x[0] - trig_pos > dt_thr): trig_pos = 0

        # if rising edge found - register begining of signal
        if(trig_pos > 0 and find_pos >= rise_thr and ramp > ramp_thr and ongoing == 0):
          tmp_start = x[0] - find_pos
          ongoing = 1

        # signal found - look when amp. reaches noise floor (neg. value)
        if(find_pos >= rise_thr and ongoing == 1 and val < 0):
          start = tmp_start
          stop  = x[0]
          for it in range(start,stop): mask[ch][it] = 0

          # reset counters
          zero_cross_check = 0
          find_pos         = 0
          ongoing          = 0
          trig_pos         = 0

        if(val > 0):
           # start counter for positive samples
           find_pos += 1
           # identify ramping 
           if(val > oldval1 and val > oldval2): ramp += 1
           else: ramp -= 1
        else:
           zero_cross_check += 1

        # reset counters when negative sample is found (deep in the noise) for col.
        if(zero_cross_check >= 1):
           zero_cross_check = 0
           find_pos         = 0
           ramp             = 0
           ongoing          = 0

        oldval2 = oldval1
        oldval1 = val

@nb.jit('(int64,boolean[:,:],float64[:],int64,int64,float64,int64,int64)',nopython = True)
def mask_induction_signal(ch,mask,data,rise_thr,ramp_thr,amp_thr,dt_thr,zero_cross_thr):

      tmp_start   = 0 # store candidate signal start to be registered when stop is found
      trig_pos    = 0 # store position at which sample exceed  amp_thrs*ped_rms - 0 otherwise
      trig_neg    = 0 # store position at which sample exceed -amp_thrs*ped_rms - 0 otherwise
      find_pos    = 0 # counter for consecutive positive sample 
      find_neg    = 0 # counter for consecutive negative sample 
      ramp        = 0 # counter 3 consecutive rising samples
      ongoing     = 0 # flag triggered when signal start found
      oldval1     = 0 # store previous sample 
      oldval2     = 0 # store pre-previous sample
      inv         = 0 # invert flag for induction signal
      zero_cross_check = 0

      for x,val in np.ndenumerate(data):

        # set positive signal trigger
        if(val >= amp_thr):   trig_pos = x[0]
        # set back positive trigger to zero if a certain period is over
        elif(x[0] - trig_pos > dt_thr): trig_pos = 0

        # check for sample above threshold for induction signal start
        if(trig_pos > 0 and find_pos > rise_thr and val > 0 and ongoing == 0):
          tmp_start = x[0] - find_pos
          ongoing = 1

        if(trig_pos > 0 and find_pos >= rise_thr and ongoing == 1):
          # set negative signal trigger
          if(val <= -amp_thr): trig_neg = x[0] 
          elif(x[0] - trig_neg > dt_thr): trig_neg = 0        

          # set invert flag when induction crosses zero
          if(val < 0 and inv == 0):
            inv = 1
            x_zero_cross = x[0]

          # set induction signal stop once a positive sample is found - 5 samples after zero crossing required
          if(val > 0 and inv == 1 and x[0] > x_zero_cross+zero_cross_thr):

            # require a minimal signal length/negative amplitude/duration to remove spikes
            if(x[0] - tmp_start > dt_thr/4 and trig_pos - tmp_start > 0 and trig_neg >0 and find_neg >= rise_thr):
              start= tmp_start
              stop = x[0]
              for it in range(start,stop): mask[ch][it] = 0

              # reset counters
              zero_cross_check= 0
              find_pos = 0
              find_neg = 0
              ongoing  = 0
              trig_pos = 0
              trig_neg = 0
              inv      = 0

        if(val > 0 and inv == 0):
           # start counter for positive samples
           find_pos += 1
           # identify ramping 
           if(val > oldval1 and val > oldval2): ramp += 1
           else: ramp -= 1
        elif(val < 0 and inv == 1):
           # start counter for negative samples
           find_neg += 1
        else:
           zero_cross_check += 1

        # reset counters when negative/positive sample is found (deep in the noise) for col. and ind. resp.
        if(zero_cross_check >= 5):
           zero_cross_check = 0
           find_pos         = 0
           find_neg         = 0
           ramp             = 0
           ongoing          = 0
           inv              = 0

        oldval2 = oldval1
        oldval1 = val

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

