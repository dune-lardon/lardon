import config as cf
import data_containers as dc
import pedestals as ped

import numpy as np
import numexpr as ne 

import bottleneck as bn



def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def FFT_low_pass(save_ps=False):
    lowpass_cut = dc.reco['noise']['fft']['low_cut']
    freq_cut    = dc.reco['noise']['fft']['freq']

    n    = int(cf.n_sample/2) + 1
    rate = cf.sampling #in MHz

    freq = np.linspace(0, rate/2., n)

    """define gaussian low pass filter"""
    gauss_cut = np.where(freq < lowpass_cut, 1., gaussian(freq, lowpass_cut, 0.02))

    if(freq_cut > 0):
        print('frequency at ', freq_cut, ' removed')
        f_cut = 1.-gaussian(freq,freq_cut,0.001)
        gauss_cut = gauss_cut*f_cut


    """go to frequency domain"""
    fdata = np.fft.rfft(dc.data_daq)

    """get power spectrum (before cut)"""
    if(save_ps==True):
        ps = 1/cf.n_sample * np.abs(fdata)
        #ps = 10.*np.log10(np.abs(fdata)+1e-1) 

    
    """Apply filter"""
    fdata *= gauss_cut[None, :]

    """go back to time"""
    dc.data_daq = np.fft.irfft(fdata)


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


def regular_coherent_noise():#groupings):
    """
    1. Computes the mean along group of channels for non ROI points
    2. Subtract mean to all points
    """
    groupings = dc.reco['noise']['coherent']['groupings']
    
    for group in groupings:
        if( (cf.n_tot_channels % group) > 0):
            print(" Coherent Noise Filter in groups of ", group, " is not a possible ! ")
            return

        nslices = int(cf.n_tot_channels / group)
        
        dc.data_daq = np.reshape(dc.data_daq, (nslices, group, cf.n_sample))
        dc.mask_daq = np.reshape(dc.mask_daq, (nslices, group, cf.n_sample))

        """sum data if mask is true"""
        with np.errstate(divide='ignore', invalid='ignore'):
            """sum the data along the N channels (subscript l) if mask is true,
            divide by nb of trues"""
            mean = np.einsum('klm,klm->km', dc.data_daq, dc.mask_daq)/dc.mask_daq.sum(axis=1)

            """require at least 3 points to take into account the mean"""
            mean[dc.mask_daq.sum(axis=1) < 3] = 0.
        

        """Apply the correction to all data points"""
        dc.data_daq -= mean[:,None,:]

        """ restore original data shape """
        dc.data_daq = np.reshape(dc.data_daq, (cf.n_tot_channels, cf.n_sample))
        dc.mask_daq = np.reshape(dc.mask_daq, (cf.n_tot_channels, cf.n_sample))


def coherent_noise_per_view():#groupings, capa_weight, calibrated):
    """
    1. Get which daq channels is which view 
    2. Computes the mean along this group of channels and this view for non ROI points
    3. Subtract mean to all points
    """

    groupings = dc.reco['noise']['coherent']['groupings']
    capa_weight = bool(dc.reco['noise']['coherent']['capa_weight'])
    calibrated  = bool(dc.reco['noise']['coherent']['calibrated'])



    v_daq = np.empty((cf.n_tot_channels,cf.n_sample))
    capa = np.ones((cf.n_tot_channels))
    calib = np.ones((cf.n_tot_channels))


    for i in range(cf.n_tot_channels):
        view = dc.chmap[i].view
        if(view >= cf.n_view or view < 0):
            view = -1

        v_daq[i,:] = view
        if(capa_weight):
            capa[i] = dc.chmap[i].capa
        if(calibrated):
            calib[i] = dc.chmap[i].gain

    
    dc.data_daq *= calib[:,None]
    dc.data_daq /= capa[:,None]
         
    for group in groupings:
        if( (cf.n_tot_channels % group) > 0):
            print(" Coherent Noise Filter in groups of ", group, " is not a possible ! ")
            return

    nslices = int(cf.n_tot_channels / group)
        
    dc.data_daq = np.reshape(dc.data_daq, (nslices, group, cf.n_sample))
    dc.mask_daq = np.reshape(dc.mask_daq, (nslices, group, cf.n_sample))


    v_daq = np.reshape(v_daq, (nslices, group, cf.n_sample))

    
    for i in range(cf.n_view):
        v_mask = np.where(v_daq==i, dc.mask_daq, 0)

        
        """sum data if mask is true"""
        with np.errstate(divide='ignore', invalid='ignore'):
            """sum the data along the N channels (subscript l) if mask is true,
                divide by nb of trues"""
            mean = np.einsum('klm,klm->km', dc.data_daq, v_mask)/v_mask.sum(axis=1)

        """require at least 3 points to take into account the mean"""
        mean[v_mask.sum(axis=1) < 3] = 0.

        dc.data_daq -= mean[:,None,:]*np.where(v_daq==i,1,0)


    """ restore original data shape """
    dc.data_daq = np.reshape(dc.data_daq, (cf.n_tot_channels, cf.n_sample))
    dc.mask_daq = np.reshape(dc.mask_daq, (cf.n_tot_channels, cf.n_sample))
    dc.data_daq /= calib[:,None]
    dc.data_daq *= capa[:,None]


def centered_median_filter(array, size):
    """ pads the array such that the output is the centered sliding median"""
    rsize = size - size // 2 - 1
    array = np.pad(array, pad_width=((0, 0) , (0, rsize)),
                   mode='constant', constant_values=np.nan)
    return bn.move_median(array, size, min_count=1, axis=-1)[:, rsize:]


def median_filter():
    window = dc.reco['noise']['microphonic']['window']
    
    if(window < 0):
        return

    """ apply median filter on data to remove microphonic noise """    

    """ mask the data with nan where potential signal is (ROI)"""
    med = centered_median_filter(np.where(dc.mask_daq==True, dc.data_daq, np.nan), window)

    """ in median computation, if everything is masked (all nan) nan is returnedso changed these cases to 0 """
    med = np.nan_to_num(med, nan=0.)

    """ apply correction to data points """
    dc.data_daq -= med
