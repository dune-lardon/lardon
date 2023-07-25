import config as cf
import data_containers as dc

import numpy as np
import numba as nb
import numexpr as ne

import time

@nb.jit(nopython = True)
def compute_pedestal_nb(data, mask):
    """ do not remove the @jit above, the code would then take ~40 s to run """
    shape = data.shape

    """ to avoid cases where the rms goes to 0"""
    min_val = 1e-5

    mean  = np.zeros(shape[:-1])
    res   = np.zeros(shape[:-1])
    for idx,r in np.ndenumerate(res):
        ch = data[idx]
        ma  = mask[idx]
        """ use the assumed mean method """
        K = 1.*ch[0] #make it float
        n, Ex, Ex2, tmp = 0., 0., 0., 0.
        for x,v in zip(ch,ma):
            if(v == True):
                n += 1
                tmp += x
                Ex += x-K
                Ex2 += (x-K)*(x-K)

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
    t0 = time.time()
    mean, std = compute_pedestal_nb(dc.data_daq, dc.mask_daq)

    if(noise_type=='raw'):
        ''' As nothing is masked yet, the computed raw pedestal is biased when there is signal '''
        ''' a rough mask is computed from the RMS '''
        ''' and pedestal and rms are computed again -- twice '''
        ''' this is still not ideal '''

        thresh = dc.reco['pedestal']['first_pass_thrs']
        for i in range(2):
            update_mask_inputs(thresh, mean, std)
            mean, std = compute_pedestal_nb(dc.data_daq, dc.mask_daq)
        ped = dc.noise( mean, std )
        dc.evt_list[-1].set_noise_raw(ped)

        inv = np.array([-1 if cf.signal_is_inverted[x.module]==True else 1 for x in dc.chmap])

        dc.data_daq = dc.data_daq*inv[:,None] + mean[:,None]*inv[:,None]


    elif(noise_type=='filt'): 
        ped = dc.noise( mean, std )
        dc.evt_list[-1].set_noise_filt(ped)
        dc.data_daq -= mean[:,None]
    else:
      print('ERROR: You must set a noise type for pedestal setting')
      sys.exit()


def update_mask(thresh):
    dc.mask_daq = ne.evaluate( "where((abs(data) > thresh*rms), 0, 1)", global_dict={'data':dc.data_daq, 'rms':dc.evt_list[-1].noise_filt.ped_rms[:,None]}).astype(bool)
    dc.mask_daq = np.logical_and(dc.mask_daq, dc.alive_chan)


def update_mask_inputs(thresh, mean, rms):
    dc.mask_daq = ne.evaluate( "where((abs(data-baseline) >  thresh*std) , 0, 1)", global_dict={'data':dc.data_daq, 'baseline':mean[:,None],'std':rms[:,None]}).astype(bool)
    dc.mask_daq = np.logical_and(dc.mask_daq, dc.alive_chan)


def refine_mask(n_pass = 1):

    for ch in range(cf.n_tot_channels):

        if(dc.alive_chan[ch,0]==False):
            dc.mask_daq[ch,:] = False
            continue

        view = dc.chmap[ch].view
        if(view >= cf.n_view or view < 0):
            continue

        rms  = dc.evt_list[-1].noise_filt.ped_rms[ch]

        if(cf.view_type[view] == "Collection"): 
            mask_collection_signal(dc.mask_daq[ch], dc.data_daq[ch],
                                   dc.reco['mask']['coll']['min_dt'],
                                   dc.reco['mask']['coll']['low_thr'][n_pass-1]*rms,
                                   dc.reco['mask']['coll']['high_thr'][n_pass-1]*rms,
                                   dc.reco['mask']['coll']['min_rise'],
                                   dc.reco['mask']['coll']['min_fall'],
                                   dc.reco['mask']['coll']['pad_bef'],
                                   dc.reco['mask']['coll']['pad_aft'])
            
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
            
            
@nb.jit('(boolean[:],float64[:],int64,float64,float64,int64,int64,int64,int64)',nopython = True)
def mask_collection_signal(mask, data, dt_thr, low_thr, high_thr, rise_thr, fall_thr, pad_bef, pad_aft):
    mask[:]=1

    start, stop, t_max, val_max, dt = -1, -1, -1, -1, 0
    tot = len(data)

    for x,val in np.ndenumerate(data):    
        t = x[0]
        if(val > low_thr and start > 0):
            dt += 1
            stop = t
            
            if(val > val_max):
                val_max = val
                t_max = t

        if(val > low_thr and start < 0):
            start = t
            
        if(val <= low_thr and start > 0):
            if(dt >= dt_thr and (t_max-start)>rise_thr and (stop-t_max)>fall_thr and val_max > high_thr):
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
        
        if(val > low_pos_thr and start_pos > 0):
            dt_pos += 1
            stop_pos = t
            
            if(val > val_max):
                val_max = val
                t_max = t

        if(val > low_pos_thr and start_pos < 0):
            start_pos = t
            
        if(val <= low_pos_thr and start_pos > 0):
            if(dt_pos >= dt_pos_thr and (t_max-start_pos)>rise_pos_thr and (stop_pos-t_max)>fall_pos_thr and val_max > high_pos_thr):

                start_pos -= pad_bef
                if(start_pos < 0): start_pos = 0
                stop_pos += pad_aft
                if(stop_pos >= tot): stop_pos = tot

                mask[start_pos:stop_pos+1] = 0
                last_t_max = t_max
                last_stop = stop_pos
                
            start_pos, stop_pos, t_max, val_max, dt_pos = -1, -1, -1, -1, 0

        if(val < low_neg_thr and start_neg > 0):
            dt_neg += 1
            stop_neg = t
            
            if(val < val_min):
                val_min = val
                t_min = t

        if(val < low_neg_thr and start_neg < 0):
            start_neg = t
            
        if(val >= low_neg_thr and start_neg > 0):
            if(dt_neg >= dt_neg_thr and (t_min-start_neg)>rise_neg_thr and (stop_neg-t_min)>fall_neg_thr and val_min < high_neg_thr):


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
    t_test = time.time()

    nchunks = 16
    if(cf.n_sample%nchunks != 0):
        return
    chunk = int(cf.n_sample/nchunks)
    #print("chunking in ", chunk)
    dc.data_daq = np.reshape(dc.data_daq, (cf.n_tot_channels, nchunks, chunk))
    dc.mask_daq = np.reshape(dc.mask_daq, (cf.n_tot_channels, nchunks, chunk))
    
    mean = [[-999 for x in range(cf.n_tot_channels)] for i in range(nchunks)]
    std = [[-999 for x in range(cf.n_tot_channels)] for i in range(nchunks)]
    #print("LENGTHS : ", len(mean), len(mean[0]))
    for i in range(nchunks):
        mean[i], std[i] = compute_pedestal_nb(dc.data_daq[:,i,:], dc.mask_daq[:,i,:])
        #mean[i] = m
        #std[i] = s
    #print(time.time()-t_test, " s to compute")
    
    mean = np.asarray(mean)
    #print(mean.shape)
    #min_mean = np.min(mean, axis=0)
    #max_mean = np.max(mean, axis=0)
    #print(min_mean.shape)
    delta_mean = np.max(mean, axis=0) - np.min(mean, axis=0)#max_mean - min_mean
    std_mean   = np.mean(std, axis=0)

    """
    for i in [2113, 2863, 529]:
        print("CHANNEL ", i)
        print("means : ")
        print([mean[k][i] for k in range(nchunks)])
        print("stds: ")
        print([std[k][i] for k in range(nchunks)])
        print(min_mean[i], " and ", max_mean[i])
        print("delta : ", delta_mean[i])
    """

    ped = dc.noise( delta_mean, std_mean )
    dc.evt_list[-1].set_noise_study(ped)




    """ restore original data shape """
    dc.data_daq = np.reshape(dc.data_daq, (cf.n_tot_channels, cf.n_sample))
    dc.mask_daq = np.reshape(dc.mask_daq, (cf.n_tot_channels, cf.n_sample))
