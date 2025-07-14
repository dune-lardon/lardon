import config as cf
import data_containers as dc


import numpy as np
import numexpr as ne 
from scipy.fft import rfft, irfft, rfftfreq

import bottleneck as bn



def gaussian(x, mu, sig):
    return np.exp(-((x - mu) ** 2) / (2 * sig ** 2))

def FFT_low_pass(save_ps=False):
    freq_cut    = dc.reco['noise']['fft']['freq']
    lowpass_cut = dc.reco['noise']['fft']['low_cut']    
    gaus_sigma  = dc.reco['noise']['fft']['gaus_sigma']

    
    nsamples      = cf.n_sample[cf.imod]
    sampling_rate = cf.sampling[cf.imod] #in MHz


    freq = rfftfreq(nsamples, d=1.0/sampling_rate)

    
    """ Build Gaussian low-pass filter """
    gauss_cut = np.ones_like(freq)
    mask = freq >= lowpass_cut
    gauss_cut[mask] = gaussian(freq[mask], lowpass_cut, gaus_sigma)
    

    if(freq_cut > 0):
        print('frequency at ', freq_cut, ' removed')
        gauss_cut *= 1.0 - gaussian(freq, freq_cut, 0.001)


    """ go to frequency domain"""
    fdata = rfft(dc.data_daq, axis=1)  # axis=1 for multi-channel

    
    """get power spectrum (before cut)"""
    if(save_ps==True):
        ps = np.abs(fdata) / cf.n_sample[cf.imod] 
        #ps = 10.*np.log10(np.abs(fdata)+1e-1) 

    """Apply filter"""
    fdata *= gauss_cut[None, :]
    
    """go back to time"""
    dc.data_daq = irfft(fdata, n=cf.n_sample[cf.imod], axis=1)

    if(save_ps==True):
        """get power spectrum after cut"""
        #ps = 1/cf.n_sample * np.abs(fdata)
        #ps = 10.*np.log10(np.abs(fdata)+1e-1) 

        return ps



def coherent_noise():
    if(dc.reco['noise']['coherent']['per_view'] ==1):
        return coherent_noise_per_view()
    else :
        return regular_coherent_noise()


def regular_coherent_noise():

    """
    1. Computes the mean along group of channels for non ROI points
    2. Subtract mean to all points
    """
    groupings = dc.reco['noise']['coherent']['groupings']

    n_chan = cf.module_nchan[cf.imod]
    n_sample = cf.n_sample[cf.imod]

    
    for group in groupings:
        if( (n_chan % group_size) > 0):
            print(f"[CNR] groups of {group} channels is not possible for {n_channels} channels! ")
            return

        n_slices = n_chan // group_size

        """ reshape data into (n_slices, group, nsamples) """
        data_sliced = dc.data_daq.reshape((n_slices, group, n_samples))
        mask_sliced = dc.mask_daq.reshape((n_slices, group, n_samples))


        """ Compute the masked mean per group """
        mask_sum = np.sum(mask_sliced, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            group_mean = np.einsum('klm,klm->km', data_sliced, mask_sliced) / mask_sum

        """ set  mean to 0 where too few valid points """
        group_mean[mask_sum < 3] = 0.

        """ Subtract group mean from each channel """
        data_sliced -= group_mean[:, None, :]

        """ Restore shape """
        dc.data_daq = data_sliced.reshape((n_channels, n_samples))
        dc.mask_daq = mask_sliced.reshape((n_channels, n_samples))

def coherent_noise_per_view():

    groupings = dc.reco['noise']['coherent']['groupings']
    capa_weight = bool(dc.reco['noise']['coherent']['capa_weight'])
    calibrated  = bool(dc.reco['noise']['coherent']['calibrated'])


    n_chan = cf.module_nchan[cf.imod]
    n_sample = cf.n_sample[cf.imod]
    daqch_start = cf.module_daqch_start[cf.imod]

    
    """ Initialize arrays """
    v_daq = np.empty((n_chan, n_sample), dtype=np.int32)
    capa = np.ones(n_chan)
    calib = np.ones(n_chan)

    """ Vectorized channel mapping """
    views = np.array([dc.chmap[i+daqch_start].view for i in range(n_chan)])
    views = np.clip(views, -1, cf.n_view-1)
    v_daq[:] = views[:, None]  # Broadcast to all samples
    
    if capa_weight:
        capa[:] = [dc.chmap[i+daqch_start].capa for i in range(n_chan)]
    if calibrated:
        calib[:] = [dc.chmap[i+daqch_start].gain for i in range(n_chan)]

    """ Apply calibration and capacitance weighting """
    if calibrated or capa_weight:
        dc.data_daq = dc.data_daq * (calib / capa)[:, None]

    for group in groupings:
        # Validate groupings
        if (n_chan % group) != 0:
            print(f"[CNR] groups of {group} is not possible for {n_chan} channels!")
            return

        """ Reshape along slices """
        n_slices = n_chan // group
        data_sliced  = dc.data_daq.reshape(n_slices, group, n_sample)
        mask_sliced  = dc.mask_daq.reshape(n_slices, group, n_sample)
        v_daq_sliced = v_daq.reshape(n_slices, group, n_sample)


        mean = np.zeros((n_slices, n_sample))
    
        for i in range(cf.n_view):
            """ Create view mask """
            view_mask = (v_daq_sliced == i)
            combined_mask = view_mask & mask_sliced
        
            """ Compute sum and count """
            sum_data = np.sum(data_sliced * combined_mask, axis=1)
            count = np.sum(combined_mask, axis=1)
        
            """ Compute mean if count >= 3 """
            np.divide(sum_data, count, out=mean, where=count >= 3)
            mean[count < 3] = 0
        
            """ Subtract mean """
            data_sliced -= mean[:, None, :] * view_mask
            
        """ Restore original shape """
        dc.data_daq = data_sliced.reshape(n_chan, n_sample)
        dc.mask_daq = mask_sliced.reshape(n_chan, n_sample)
    
    """ Reverse calibration """ 
    if calibrated or capa_weight:
        dc.data_daq = dc.data_daq * (capa / calib)[:, None]

        
def centered_median_filter(array, size):
    """ pads the array such that the output is the centered sliding median"""
    rsize = size - size // 2 - 1
    array = np.pad(array, pad_width=((0, 0) , (0, rsize)),
                   mode='constant', constant_values=np.nan)
    return bn.move_median(array, size, min_count=1, axis=-1)[:, rsize:]




def median_filter():
    """
    Removes microphonic noise by subtracting a median-filtered baseline
    over a specified window, ignoring regions of interest (ROI).
    """

    window = dc.reco['noise']['microphonic']['window']

    if not (0 < window <= cf.n_sample[cf.imod]):
        return

    """ Mask ROI with NaN for median computation """
    data_masked = np.where(dc.mask_daq, dc.data_daq, np.nan)    
    
    """ Apply centered median filter """
    baseline = centered_median_filter(data_masked, window)
    
    """ Replace any full-NaN windows with 0 """
    np.nan_to_num(baseline, copy=False, nan=0.0)

    """ Subtract median-filtered noise """
    dc.data_daq -= baseline



def median_filter_pds():
    ''' same as above : code for flattening the pds waveforms '''
    
    window = dc.reco['pds']['noise']['flat_baseline_window']

    if not (0 < window <= cf.n_pds_sample[cf.imod]):
        return

    """ Mask ROI with NaN for median computation """
    data_masked = np.where(dc.mask_pds, dc.data_pds, np.nan)

    """ Apply centered median filter """
    baseline = centered_median_filter(data_masked, window)

    """ Replace any full-NaN windows with 0 """
    np.nan_to_num(baseline, copy=False, nan=0.0)

    """ Subtract median-filtered noise """
    dc.data_pds -= baseline
    
