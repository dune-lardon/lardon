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


    if(fdata.shape[1] != gauss_cut.shape[0]):
        if(fdata.shape[1] < gauss_cut.shape[0]):
            gauss_cut = gauss_cut[:fdata.shape[1]]

    """get power spectrum (before cut)"""
    
    if(save_ps==True):
        ps = np.abs(fdata) / cf.n_sample[cf.imod] 
        print("PS shape ", ps.shape)
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
    if(dc.reco['noise']['coherent']['per_view_per_card'] ==1):
        return coherent_noise_per_view_per_card()
    elif(dc.reco['noise']['coherent']['per_view'] ==1):
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
    n_samples = cf.n_sample[cf.imod]

    
    for group in groupings:
        if( (n_chan % group) > 0):
            print(f"[CNR] groups of {group} channels is not possible for {n_channels} channels! ")
            return

        n_slices = n_chan // group

        """ reshape data into (n_slices, group, nsamples) """
        data_sliced = dc.data_daq.reshape((n_slices, group, n_samples))
        mask_sliced = dc.mask_daq.reshape((n_slices, group, n_samples))

        print(data_sliced.shape, " and ", mask_sliced.shape)
        """ Compute the masked mean per group """
        mask_sum = np.sum(mask_sliced, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            group_mean = np.einsum('klm,klm->km', data_sliced, mask_sliced) / mask_sum

        """ set  mean to 0 where too few valid points """
        group_mean[mask_sum < 3] = 0.

        """ Subtract group mean from each channel """
        data_sliced -= group_mean[:, None, :]

        """ Restore shape """
        dc.data_daq = data_sliced.reshape((n_chan, n_samples))
        dc.mask_daq = mask_sliced.reshape((n_chan, n_samples))


def coherent_noise_per_view_per_card():
    capa_weight = bool(dc.reco['noise']['coherent']['capa_weight'])
    calibrated  = bool(dc.reco['noise']['coherent']['calibrated'])
    
    n_chan = cf.module_nchan[cf.imod]
    n_tot_chan = cf.n_tot_channels
    n_sample = cf.n_sample[cf.imod]


    """ channel mapping """
    views = np.array([dc.chmap[i].view for i in range(n_tot_chan) if dc.chmap[i].module==cf.imod])
    cards = np.array([dc.chmap[i].card for i in range(n_tot_chan) if dc.chmap[i].module==cf.imod])

    
    # Optional per-channel weights
    capa = np.ones(n_chan)
    calib = np.ones(n_chan)
    if capa_weight:
        capa = np.array([dc.chmap[i].capacitance for i in range(n_tot_chan) if dc.chmap[i].module==cf.imod])
    if calibrated:
        calib = np.array([dc.calib[i] for i in range(n_tot_chan) if dc.chmap[i].module==cf.imod])

    """ Apply calibration and capacitance weighting """
    if calibrated or capa_weight:
        dc.data_daq = dc.data_daq * (calib / capa)[:, None]


    # get the noise only data 
    data = dc.data_daq * dc.mask_daq 

                
    """ Vectorized grouping """
    # Assign integer group IDs from (view, card)
    # We use structured array trick to make them unique pairs
    keys = np.core.defchararray.add(views.astype(str), "_" + cards.astype(str))
    _, group_ids = np.unique(keys, return_inverse=True)
    n_groups = group_ids.max() + 1

    
    # Weighted sums per group for all samples
    sums = np.zeros((n_groups, n_sample))
    norm = np.zeros((n_groups, n_sample))

    # Use np.add.at to accumulate directly
    np.add.at(sums, group_ids, data)
    np.add.at(norm, group_ids, dc.mask_daq)

    # Compute means (shape = (n_groups, n_sample))
    #means = sums / norm[:, None]
    # safe division
    denom = norm.copy()
    denom[denom == 0] = 1.0
    means = sums / denom
    means[norm == 0] = 0.0

    """ Subtract group mean from each channel """

    dc.data_daq -= means[group_ids]

    """ Reverse calibration """ 
    if calibrated or capa_weight:
        dc.data_daq = dc.data_daq * (capa / calib)[:, None]

      
    
def coherent_noise_per_view():


    groupings = dc.reco['noise']['coherent']['groupings']
    capa_weight = bool(dc.reco['noise']['coherent']['capa_weight'])
    calibrated  = bool(dc.reco['noise']['coherent']['calibrated'])


    n_chan = cf.module_nchan[cf.imod]
    n_tot_chan = cf.n_tot_channels
    n_sample = cf.n_sample[cf.imod]
    daqch_start = cf.module_daqch_start[cf.imod]

    
    """ Initialize arrays """
    v_daq = np.empty((n_chan, n_sample), dtype=np.int32)
    capa = np.ones(n_chan)
    calib = np.ones(n_chan)

    """ Vectorized channel mapping """
    views = np.array([dc.chmap[i].view for i in range(n_tot_chan) if dc.chmap[i].module==cf.imod])
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

        #tmp_views = views.reshape(n_slices, group)
        #print('temp :: ', tmp_views.shape)
        
        #print('data: ', data_sliced.shape)
        #data_tmp  = np.zeros((n_slices, group, n_sample))
        
        mean = np.zeros((n_slices, n_sample))
    
        for i in range(cf.n_view):
            #tmp_views_sel = (tmp_views == i)
            #print([(k,np.sum(tmp_views_sel[k])) for k in range(5)])
            """ Create view mask """
            view_mask = (v_daq_sliced == i)
            #print(" view ", i, view_mask.shape, "=::", np.sum(view_mask, axis=0), ' or ', np.sum(view_mask, axis=1))
            
            combined_mask = view_mask & mask_sliced
        
            """ Compute sum and count """
            sum_data = np.sum(data_sliced * combined_mask, axis=1)
            count = np.sum(combined_mask, axis=1)
        
            """ Compute mean if count >= 3 """
            np.divide(sum_data, count, out=mean, where=count >= 3)
            mean[count < 3] = 0
        
            """ Subtract mean """
            data_sliced -= mean[:, None, :] * view_mask
            #data_tmp += mean[:, None, :] * view_mask

            
        """ Restore original shape """
        dc.data_daq = data_sliced.reshape(n_chan, n_sample)
        dc.mask_daq = mask_sliced.reshape(n_chan, n_sample)

        #data_tmp = data_tmp.reshape(n_chan, n_sample)
        #dc.data_daq = data_tmp        
        
    """ Reverse calibration """ 
    if calibrated or capa_weight:
        dc.data_daq = dc.data_daq * (capa / calib)[:, None]


def shield_coupling():
    if(dc.evt_list[-1].det != 'pdvd'):
        return
    if(cf.imod < 2):
        return
    
    capa_weight = True
    calibrated  = False
    
    group = 476
    n_tot_chan = cf.n_tot_channels#
    n_chan = cf.module_nchan[cf.imod]##
    n_sample = cf.n_sample[cf.imod]
    daqch_start = cf.module_daqch_start[cf.imod]

    #print(n_chan, 'and ', daqch_start)
    
    """ Initialize arrays """
    v_daq = np.empty((n_chan, n_sample), dtype=np.int32)
    capa = np.ones(n_chan)
    calib = np.ones(n_chan)

    if capa_weight:
        capa[:] = [dc.chmap[i].capa for i in range(n_tot_chan) if dc.chmap[i].module==cf.imod]
    if calibrated:
        calib[:] = [dc.chmap[i].gain for i in range(n_tot_chan) if dc.chmap[i].module==cf.imod]


    
    """ Apply calibration and capacitance weighting """
    if calibrated or capa_weight:
        dc.data_daq = dc.data_daq * (calib / capa)[:, None]


    views = np.array([int(dc.chmap[i].vchan // group)  if dc.chmap[i].view==0  else -1 for i in range(n_tot_chan) if dc.chmap[i].module==cf.imod])



    for icru in range(2):
        idx = (views == icru)
        if not np.any(idx):
            continue
        med = np.median(dc.data_daq[idx, :], axis=0)
        dc.data_daq[idx, :] -= med

    
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

    if not (0 < window <= cf.n_pds_sample):
        return

    """ Mask ROI with NaN for median computation """
    data_masked = np.where(dc.mask_pds, dc.data_pds, np.nan)

    """ Apply centered median filter """
    baseline = centered_median_filter(data_masked, window)

    """ Replace any full-NaN windows with 0 """
    np.nan_to_num(baseline, copy=False, nan=0.0)

    """ Subtract median-filtered noise """
    dc.data_pds -= baseline
    
