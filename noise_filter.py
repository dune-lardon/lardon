import config as cf
import data_containers as dc
import pedestals as ped

import numpy as np
import numexpr as ne 


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def FFT_low_pass(lowpass_cut):
    n    = int(cf.n_sample/2) + 1
    rate = cf.sampling #in MHz

    freq = np.linspace(0, rate/2., n)

    """define gaussian low pass filter"""
    gauss_cut = np.where(freq < lowpass_cut, 1., gaussian(freq, lowpass_cut, 0.02))

    """go to frequency domain"""
    fdata = np.fft.rfft(dc.data_daq)

    """get power spectrum (before cut)"""
    ps = 1/cf.n_sample * np.abs(fdata)
    #ps = 10.*np.log10(np.abs(fdata)+1e-1) 

    
    """Apply filter"""
    fdata *= gauss_cut[None, :]

    """go back to time"""
    dc.data_daq = np.fft.irfft(fdata)


    """get power spectrum after cut"""
    #ps = 1/cf.n_sample * np.abs(fdata)
    #ps = 10.*np.log10(np.abs(fdata)+1e-1) 
    return ps



def coherent_noise(groupings):
    """
    1. Computes the mean along group of channels for non ROI points
    2. Subtract mean to all points
    """

    for group in groupings:
        if( (cf.n_tot_channels % group) > 0):
            print(" Coherent Noise Filter in groups of ", group, " is not a possible ! ")
            return

        nslices = int(cf.n_tot_channels / group)
        #print(' slice in ', nslices, ' shape is ', dc.data_daq.shape)
        #print(dc.data_daq[5,400:440])
        #print(dc.data_daq[10,400:440])
        #print(dc.data_daq[120,400:440])
        
        dc.data_daq = np.reshape(dc.data_daq, (nslices, group, cf.n_sample))
        dc.mask_daq = np.reshape(dc.mask_daq, (nslices, group, cf.n_sample))

        """sum data if mask is true"""
        with np.errstate(divide='ignore', invalid='ignore'):
            """sum the data along the N channels (subscript l) if mask is true,
            divide by nb of trues"""
            mean = np.einsum('klm,klm->km', dc.data_daq, dc.mask_daq)/dc.mask_daq.sum(axis=1)

            """require at least 3 points to take into account the mean"""
            mean[dc.mask_daq.sum(axis=1) < 3] = 0.
        
        #for ig in range(nslices):
            #print(ig, ' has ', dc.mask_daq.sum(axis=1)[ig,10])
        #print(dc.data_daq[0,:,400])

        """Apply the correction to all data points"""
        dc.data_daq -= mean[:,None,:]
        #print("sub")
        #print(mean[0,400:440])

        """ restore original data shape """
        dc.data_daq = np.reshape(dc.data_daq, (cf.n_tot_channels, cf.n_sample))
        dc.mask_daq = np.reshape(dc.mask_daq, (cf.n_tot_channels, cf.n_sample))

        #print("now")
        #print(dc.data_daq[5,400:440])
        #print(dc.data_daq[10,400:440])
        #print(dc.data_daq[120,400:440])



