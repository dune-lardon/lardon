import lardon.config as cf
import lardon.data_containers as dc

import numpy as np
import numba as nb
import numexpr as ne

import bottleneck as bn

import time


def set_dummy_pedestals():
    """ when the data has no sample """
    shape = dc.data_daq.shape

    mean  = np.zeros(shape[:-1])
    std   = np.zeros(shape[:-1])
    mean -= 1
    std  -= 1

    
    ped = dc.noise( mean, std )
    
    dc.evt_list[-1].set_noise_raw(ped)
    dc.evt_list[-1].set_noise_filt(ped)

    if(dc.reco['noise']['study']['to_be_done'] > 0):
        if(dc.evt_list[-1].noise_study == None):
            dc.evt_list[-1].noise_study = [[] for x in range(cf.n_module_used)]

        dc.evt_list[-1].set_noise_study(ped)
    
@nb.jit(nopython = True)
def compute_pedestal_nb(data, mask, is_raw):
    """ do not remove the @jit above, the code would then take ~40 s to run """
    shape = data.shape


    """ to avoid cases where the rms goes to 0"""
    min_val = 1e-5

    mean  = np.zeros(shape[:-1])
    res   = np.zeros(shape[:-1])

    #for idx,r in np.ndenumerate(res):
    for idx in range(shape[0]):
        ch = data[idx]
        ma  = mask[idx]
        """ use the assumed mean method """
        #make it float
        K = 1.*ch[0] if np.isnan(ch[0])==False else 0.

        minv, maxv = 99999999,0
        
        n, Ex, Ex2, tmp = 0., 0., 0., 0.
        for x,v in zip(ch,ma):
            if(is_raw == True):
                v = True
                
            if(np.isnan(x) == False and v == True):
                n += 1
                tmp += x
                Ex += x-K
                Ex2 += (x-K)*(x-K)
                if(x<minv):minv=x
                if(x>maxv):maxv=x
                
                #print(n, Ex)
                
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
        
    if(noise_type=='raw'):
        ''' As nothing is masked yet, the computed raw pedestal is biased when there is signal '''
        ''' a rough mask is computed from the RMS '''
        ''' and pedestal and rms are computed again '''
        
        mean, std = compute_pedestal_nb(dc.data_daq, dc.mask_daq, True)

        thresh = dc.reco['pedestal']['raw_rms_thr']
        n_iter = dc.reco['pedestal']['n_iter']
        
        for i in range(n_iter):
            update_mask_inputs(thresh, mean, std)
            mean, std = compute_pedestal_nb(dc.data_daq, dc.mask_daq, False)


            
        ped = dc.noise( mean, std )
        dc.evt_list[-1].set_noise_raw(ped)

        inv = np.array([-1 if cf.signal_is_inverted[cf.imod]==True else 1 for x in range(cf.module_nchan[cf.imod])]) 

        """ remove the nans introduced to align the frames """
        dc.data_daq = np.where(np.isnan(dc.data_daq), mean[:,None]*inv[:,None], dc.data_daq)

        """ subtract the mean pedestal """
        dc.data_daq = dc.data_daq*inv[:,None] - mean[:,None]*inv[:,None]



    elif(noise_type=='filt'):
        mean, std = compute_pedestal_nb(dc.data_daq, dc.mask_daq, False)

        ped = dc.noise( mean, std )
        dc.evt_list[-1].set_noise_filt(ped)
        dc.data_daq -= mean[:,None]
    else:
      print('ERROR: You must set a noise type for pedestal setting')
      sys.exit()


def update_mask(thresh):       
    dc.mask_daq = ne.evaluate("abs(data) <= thresh * rms", global_dict={'data': dc.data_daq, 'rms': dc.evt_list[-1].noise_filt.ped_rms[:, None],'thresh': thresh})
    dc.mask_daq = np.logical_and(dc.mask_daq, dc.alive_chan[:,None])


def update_mask_inputs(thresh, mean, rms):
    dc.mask_daq = ne.evaluate( "abs(data-baseline) <=  thresh*std", global_dict={'data':dc.data_daq, 'baseline':mean[:,None],'std':rms[:,None]})
    dc.mask_daq = np.logical_and(dc.mask_daq, dc.alive_chan[:,None])


def refine_mask(n_pass = 1):
    
    for ch in range(cf.module_nchan[cf.imod]):

        daq_ch = ch+cf.module_daqch_start[cf.imod]

        if(dc.alive_chan[ch]==False):
            dc.mask_daq[ch,:] = False
            continue

        view = dc.chmap[daq_ch].view
        if(view >= cf.n_view or view < 0):
            continue

        rms  = dc.evt_list[-1].noise_filt[cf.imod].ped_rms[ch]



        debug = False
        if(cf.view_type[cf.imod][view] == "Collection"): 

            mask_collection_signal(dc.mask_daq[ch], dc.data_daq[ch],
                                   dc.reco['mask']['coll']['min_dt'],
                                   dc.reco['mask']['coll']['low_thr'][n_pass-1]*rms,
                                   dc.reco['mask']['coll']['high_thr'][n_pass-1]*rms,
                                   dc.reco['mask']['coll']['min_rise'],
                                   dc.reco['mask']['coll']['min_fall'],
                                   dc.reco['mask']['coll']['pad_bef'],
                                   dc.reco['mask']['coll']['pad_aft'], debug)
            
        else:
            mask_induction_signal(dc.mask_daq[ch], dc.data_daq[ch],
                                  dc.reco['mask']['ind']['max_dt_pos_neg'],
                                  dc.reco['mask']['ind']['pos']['min_dt'],
                                  dc.reco['mask']['ind']['pos']['low_thr'][n_pass-1]*rms,
                                  dc.reco['mask']['ind']['pos']['high_thr'][n_pass-1]*rms,
                                  dc.reco['mask']['ind']['pos']['min_rise'],
                                  dc.reco['mask']['ind']['pos']['min_fall'],
                                  dc.reco['mask']['ind']['neg']['min_dt'],
                                  dc.reco['mask']['ind']['neg']['low_thr'][n_pass-1]*rms,
                                  dc.reco['mask']['ind']['neg']['high_thr'][n_pass-1]*rms,
                                  dc.reco['mask']['ind']['neg']['min_rise'],
                                  dc.reco['mask']['ind']['neg']['min_fall'],
                                  dc.reco['mask']['ind']['pad_bef'],
                                  dc.reco['mask']['ind']['pad_aft'])
            



            
@nb.jit('(boolean[:],float64[:],int64,float64,float64,int64,int64,int64,int64,boolean)',nopython = True)
def mask_collection_signal(mask, data, dt_thr, low_thr, high_thr, rise_thr, fall_thr, pad_bef, pad_aft,debug):
    mask[:]=1

    start, stop, t_max, val_max, dt = -1, -1, -1, -1, 0
    tot = len(data)

    for x,val in np.ndenumerate(data):    
        t = x[0]
        if(debug and t < 100):
            print(t, val, start, ' test', val > low_thr)
            
        if(val > low_thr and start >= 0):
            dt += 1
            stop = t
            
            if(val > val_max):
                val_max = val
                t_max = t

        if(val > low_thr and start < 0):
            start = t
            
        if(val <= low_thr and start >= 0):                
            if(dt >= dt_thr and (t_max-start)>=rise_thr and (stop-t_max)>=fall_thr and val_max >= high_thr):
                start -= pad_bef
                if(start < 0): start = 0

                stop += pad_aft
                if(stop >= tot): stop = tot

                mask[start:stop+1] = 0
            
            start, stop, t_max, val_max, dt = -1, -1, -1, -1, 0


            
@nb.jit('(boolean[:],float64[:],int64,int64,float64,float64,int64,int64,int64,float64,float64,int64,int64,int64,int64)',nopython = True)
def mask_induction_signal(mask, data, dt_posneg_thr, dt_pos_thr, low_pos_thr, high_pos_thr, rise_pos_thr, fall_pos_thr, dt_neg_thr, low_neg_thr, high_neg_thr, rise_neg_thr, fall_neg_thr, pad_bef, pad_aft):


    mask[:]=1

    start_pos = -1
    stop_pos = -1

    start_neg = -1
    stop_neg = -1

    t_max = -1
    val_max = -1

    t_min = -1
    val_min = 1

    dt_pos = 0
    dt_neg = 0

    tot = len(data)

    last_t_max = -1
    last_stop_pos = -1

    for x,val in np.ndenumerate(data):    
        t = x[0]
        
        if(val > low_pos_thr ):
            if(start_pos < 0):
                start_pos = t

            if(start_pos>=0):                    
                dt_pos += 1
                stop_pos = t
            
            if(val > val_max):
                val_max = val
                t_max = t

            
        if(val <= low_pos_thr and start_pos >= 0):
            if(dt_pos >= dt_pos_thr and (t_max-start_pos)>=rise_pos_thr and (stop_pos-t_max)>=fall_pos_thr and val_max >= high_pos_thr):

                start_pos -= pad_bef
                if(start_pos < 0): start_pos = 0
                stop_pos += pad_aft
                if(stop_pos >= tot): stop_pos = tot

                mask[start_pos:stop_pos+1] = 0
                last_t_max = t_max
                last_stop_pos = stop_pos
                
            start_pos, stop_pos, t_max, val_max, dt_pos = -1, -1, -1, -1, 0

        if(val < low_neg_thr):
            if(start_neg < 0):
                start_neg = t
                
            if(start_neg >= 0):
                dt_neg += 1
                stop_neg = t
            
            if(val < val_min):
                val_min = val
                t_min = t
            
        if(val >= low_neg_thr and start_neg >= 0):
            if(dt_neg >= dt_neg_thr and (t_min-start_neg)>=rise_neg_thr and (stop_neg-t_min)>=fall_neg_thr and val_min <= high_neg_thr):


                start_neg -= pad_bef
                if(start_neg < 0): start_neg = 0
                stop_neg += pad_aft
                if(stop_neg >= tot): stop_neg = tot
                mask[start_neg:stop_neg+1] = 0

                if((t_min-last_t_max) < dt_posneg_thr):
                    if(last_stop_pos < start_neg+1):
                        mask[last_stop_pos:start_neg+1] = 0                    
            start_neg, stop_neg, t_min, val_min, dt_neg = -1, -1, -1, 1, 0





def study_noise():
    """ some attempt at caracterizing the microphonic noise """

    if(dc.reco['noise']['study']['to_be_done'] == 0):
        return
    
    if(dc.evt_list[-1].noise_study == None):
        dc.evt_list[-1].noise_study = [[] for x in range(cf.n_module_used)]

    nchunks = dc.reco['noise']['study']['nchunk']
    if(cf.n_sample[cf.imod]%nchunks != 0):
        shape = dc.data_daq.shape

        mean  = np.zeros(shape[:-1])
        std   = np.zeros(shape[:-1])
        mean -= 1
        std  -= 1
        ped = dc.noise( mean, std )
        dc.evt_list[-1].set_noise_study(ped)
        return
    chunk = int(cf.n_sample[cf.imod]/nchunks)

    dc.data_daq = np.reshape(dc.data_daq, (cf.module_nchan[cf.imod], nchunks, chunk))
    dc.mask_daq = np.reshape(dc.mask_daq, (cf.module_nchan[cf.imod], nchunks, chunk))
    
    mean = [[-999 for x in range(cf.module_nchan[cf.imod])] for i in range(nchunks)]
    std = [[-999 for x in range(cf.module_nchan[cf.imod])] for i in range(nchunks)]

    
    for i in range(nchunks):
        mean[i], std[i] = compute_pedestal_nb(dc.data_daq[:,i,:], dc.mask_daq[:,i,:], False)
        
    
    mean = np.asarray(mean)
    delta_mean = np.max(mean, axis=0) - np.min(mean, axis=0)
    std_mean   = np.mean(std, axis=0)

    
    ped = dc.noise( delta_mean, std_mean )
    dc.evt_list[-1].set_noise_study(ped)


    """ restore original data shape """
    dc.data_daq = np.reshape(dc.data_daq, (cf.module_nchan[cf.imod], cf.n_sample[cf.imod]))
    dc.mask_daq = np.reshape(dc.mask_daq, (cf.module_nchan[cf.imod], cf.n_sample[cf.imod]))






def compute_pedestal_pds(first=False):

    adc_thresh = dc.reco['pds']['pedestal']['raw_adc_thresh']
    rms_thresh = dc.reco['pds']['pedestal']['rms_thresh']
    n_iter = dc.reco['pds']['pedestal']['n_iter']

    if(first==True):
        """ very simple raw mean pedestal computation atm, 
        remove the median value of the waveform """
        s_med = bn.median(dc.data_stream_pds, axis=1)
        dc.data_stream_pds -= s_med[:,None]                
        dc.mask_stream_pds = ne.evaluate( "(isnan(data)) | (data <=  adc_thresh)| (data <=  -abs(baseline)+10)", global_dict={'data':dc.data_stream_pds, 'baseline':s_med[:,None]})

        t_med = bn.nanmedian(dc.data_trig_pds, axis=1)
        dc.data_trig_pds -= t_med[:,None]                
        dc.mask_trig_pds = ne.evaluate( "(isnan(data)) | (data <=  adc_thresh)| (data <=  -abs(baseline)+10)", global_dict={'data':dc.data_trig_pds, 'baseline':t_med[:,None]})


    #else:
        #med = dc.evt_list[-1].noise_pds_raw.ped_mean


        
    s_mean, s_std = compute_pedestal_nb(dc.data_stream_pds, dc.mask_stream_pds, False)
    dc.data_stream_pds -= s_mean[:,None]

    t_mean, t_std = compute_pedestal_nb(dc.data_trig_pds, dc.mask_trig_pds, False)
    dc.data_trig_pds -= t_mean[:,None]


    for i in range(n_iter):
        dc.mask_stream_pds = ne.evaluate( "(isnan(data)) | (data <=  rms_thresh*rms)|(data <=  -abs(baseline)+10)", global_dict={'data':dc.data_stream_pds,'rms':s_std[:,None], 'baseline':s_mean[:,None]})

        s_mean, s_std = compute_pedestal_nb(dc.data_stream_pds, dc.mask_stream_pds, False)
        dc.data_stream_pds -= s_mean[:,None]

        ######
        dc.mask_trig_pds = ne.evaluate( "(isnan(data)) | (data <=  rms_thresh*rms)|(data <=  -abs(baseline)+10)", global_dict={'data':dc.data_trig_pds,'rms':t_std[:,None], 'baseline':t_mean[:,None]})

        t_mean, t_std = compute_pedestal_nb(dc.data_trig_pds, dc.mask_trig_pds, False)
        dc.data_trig_pds -= t_mean[:,None]



    if(first==True):
        ped = dc.noise( s_med, s_std )
        ped.add(t_med, t_std)         
        dc.evt_list[-1].set_noise_pds_raw(ped)
    else:
        ped = dc.noise( s_mean, s_std )
        ped.add(t_mean, t_std)
        dc.evt_list[-1].set_noise_pds_filt(ped)

