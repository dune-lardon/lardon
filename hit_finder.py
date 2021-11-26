import config as cf
import data_containers as dc
import lar_param as lar

import numpy as np
import numba as nb


def hit_search(data, view, daq_chan, start, dt_min, thr1, thr2, thr3):

    ll = []
    if(cf.view_type[view] != "Collection" and cf.view_type[view] != "Induction"): 
        print(cf.view_type[view], " is not recognized")
        sys.exit()
    elif(cf.view_type[view] == "Collection"):
        h_num, h_start, h_stop, h_charge_int, h_max_t, h_max_adc, h_min_t, h_min_adc = hit_search_collection_nb(data, view, daq_chan, start, dt_min, thr1, thr2)
    elif(cf.view_type[view] == "Induction"):
        h_num, h_start, h_stop, h_charge_int, h_max_t, h_max_adc, h_min_t, h_min_adc = hit_search_induction_nb(data, view, daq_chan, start, dt_min, thr3)

    if(h_num == 0): return ll

    for start, stop, charge_int, max_t, max_adc, min_t, min_adc in zip(h_start, h_stop, h_charge_int, h_max_t, h_max_adc, h_min_t, h_min_adc):
       h = dc.hits(view, daq_chan, start, stop, charge_int, max_t, max_adc, min_t, min_adc)
       ll.append(h)
       if(len(ll) == h_num): break

    return ll 

@nb.njit('Tuple((int64,int16[:],int16[:],float64[:],int16[:],float64[:],int16[:],float64[:]))(float64[:],int64,int64,int64,int64,float64)')
def hit_search_induction_nb(data, view, daq_chan, start, dt_min, thr):
    """ very basic induction-like hit finder """
    """ WARNING : CANNOT FIND OVERLAPPING HITS """
    
    npts = len(data)

    h_num = 0 # store number of found hits
    # list of hits parameters to be returned by this numba function (see dc.hits)
    h_start     = np.zeros(npts,dtype=np.int16)
    h_stop      = np.zeros(npts,dtype=np.int16)
    h_charge_int= np.zeros(npts)
    h_max_t     = np.zeros(npts,dtype=np.int16)
    h_max_adc   = np.zeros(npts)
    h_min_t     = np.zeros(npts,dtype=np.int16)
    h_min_adc   = np.zeros(npts)

    hitPosFlag = False
    hitNegFlag = False

    i=0

    posSamp = 0
    negSamp = 0

    h_start[h_num] = start

    while(i<npts):

        if(i < npts and hitPosFlag == False and hitNegFlag == False):
            i += 1

        """ start with the positive blob of the hit """
        val = data[i]
        while(i < npts and val >= thr and hitNegFlag == False):
            val = data[i]        
            it = i+start
            posSamp += 1

            """ first point above thr """
            if(hitPosFlag == False):
                hitPosFlag = True

                h_start[h_num] = it

                h_max_t[h_num] = it
                h_max_adc[h_num] = val
                
                
            """ update the maximum case """
            if(val > h_max_adc[h_num]):
                h_max_t[h_num] = it
                h_max_adc[h_num] = val                
                    
            if(hitPosFlag):
                h_charge_int[h_num] += val
                
            i+=1

        if(posSamp < dt_min):
            hitPosFlag = False
            posSamp = 0

        val = data[i]
        while(i < npts and hitPosFlag and hitNegFlag == False and np.fabs(val) < thr):
            h_charge_int[h_num] += np.fabs(val)
            i += 1
            val = data[i]
        """ now the negative part """

        val = data[i]
        while(i < npts and hitPosFlag and val < -1.*thr):            
            val = data[i]        
            it = i+start
            negSamp += 1

            """ first point below thr """
            if(hitNegFlag == False):
                hitNegFlag = True

                h_min_t[h_num] = it
                h_min_adc[h_num] = val
                
                
            """ update the minimum case """
            if(val < h_min_adc[h_num]):
                h_min_t[h_num] = it
                h_min_adc[h_num] = val                
                    
            if(hitNegFlag):
                h_charge_int[h_num] += -1.*val

            h_stop[h_num] = it
            i+=1

        if(negSamp < dt_min):
            hitNegFlag = False
            negSamp = 0

        if(hitPosFlag and hitNegFlag):
            h_num += 1 
            break


    return h_num, h_start, h_stop, h_charge_int, h_max_t, h_max_adc, h_min_t, h_min_adc

@nb.njit('Tuple((int64,int16[:],int16[:],float64[:],int16[:],float64[:],int16[:],float64[:]))(float64[:],int64,int64,int64,int64,float64,float64)')
def hit_search_collection_nb(data, view, daq_chan, start, dt_min, thr1, thr2):
    """search hit-shape in a list of points"""
    """algorithm from qscan"""
    npts = len(data)
    
    h_num = 0 # store number of found hits
    # list of hits parameters to be returned by this numba function (see dc.hits)
    h_start     = np.zeros(npts,dtype=np.int16)
    h_stop      = np.zeros(npts,dtype=np.int16)
    h_charge_int= np.zeros(npts)
    h_max_t     = np.zeros(npts,dtype=np.int16)
    h_max_adc   = np.zeros(npts)
    h_min_t     = np.zeros(npts,dtype=np.int16)
    h_min_adc   = np.zeros(npts)
 
    hitFlag = False

    i=0
    minimum = cf.n_sample
    minSamp = -1
    singleHit = True

    while(i<npts):
        while(i < npts and data[i] >= thr1):
            val = data[i]        
            it = i+start

            if(hitFlag == False):
                hitFlag = True
                singleHit = True
                
                h_start[h_num]     = it
                h_stop[h_num]      = 0
                h_charge_int[h_num]= 0.
                h_max_t[h_num]     = it
                h_max_adc[h_num]   = val
                h_min_t[h_num]     = 0.
                h_min_adc[h_num]   = 0.

                #h = dc.hits(view,daq_chan,it,0,0.,it,val, 0., 0.)
                minSamp = -1
                
            if(it > h_max_t[h_num] and val < h_max_adc[h_num] - thr2 and (minSamp==-1 or minimum >= val)):
                minSamp = it
                minimum = val

                
            if(minSamp >= 0 and it > minSamp and val > minimum + thr2 and (it-h_start[h_num]) >= dt_min):
                h_stop[h_num]      = minSamp-1
                h_num += 1
                hitFlag = True
                singleHit = False

                h_start[h_num]     = minSamp
                h_stop[h_num]      = 0
                h_charge_int[h_num]= 0.
                h_max_t[h_num]     = it
                h_max_adc[h_num]   = val
                h_min_t[h_num]     = 0.
                h_min_adc[h_num]   = 0.

                minSamp = -1

                
            if(h_stop[h_num] == 0 and val > h_max_adc[h_num]):
                h_max_t[h_num] = it
                h_max_adc[h_num] = val
                if(minSamp >= 0):
                    minSamp = -1
                    minimum = val
                    
            h_charge_int[h_num] += val
            i+=1
        if(hitFlag == True):
            hitFlag = False
            h_stop[h_num] = it-1

            #if((singleHit and (h_stop[h_num]-h_start[h_num] >= dt_min)) or not singleHit):
            if(h_stop[h_num]-h_start[h_num] >= dt_min):
                h_num += 1 

        i+=1
    return h_num, h_start, h_stop, h_charge_int, h_max_t, h_max_adc, h_min_t, h_min_adc


def recompute_hit_charge(hit):
    daq_ch, start, stop = hit.daq_channel, hit.start, hit.stop
    val = 0.
    mean = dc.evt_list[-1].noise_filt.ped_mean[daq_ch]

    for t in range(start, stop):
        val += np.fabs(dc.data_daq[daq_ch, t] - mean)

    hit.charge_int = val


def find_hits(pad_left, pad_right, dt_min, n_sig_coll_1, n_sig_coll_2, n_sig_ind): 

    """ get boolean roi based on mask and alive channels """
    ROI = np.array(~dc.mask_daq & dc.alive_chan, dtype=bool)

    """ adds 0 (False) and the start and end of each waveform """
    falses = np.zeros((cf.n_tot_channels,1),dtype=int)
    ROIs = np.r_['-1',falses,np.asarray(ROI,dtype=int),falses]
    d = np.diff(ROIs)

    """ a change from false to true in difference is = 1 """
    start = np.where(d==1)
    """ a change from true to false in difference is = -1 """
    end   = np.where(d==-1)
    """ look at long enough sequences of trues """
    gpe = (end[1]-start[1])>=dt_min

    assert len(start[0])==len(end[0]), " Mismatch in groups of hits"
    assert len(gpe)==len(start[0]), "Mismatch in groups of hits"    

    for g in range(len(gpe)):
        if(gpe[g]):

            """ make sure starts and ends of hit group are in the same channel """

            assert start[0][g] == end[0][g], "Hit Mismatch"

            daq_chan = start[0][g]
            
            view, channel = dc.chmap[daq_chan].view, dc.chmap[daq_chan].vchan
            if(view < 0 or view >= cf.n_view):
                continue

            tdc_start = start[1][g]
            tdc_stop = end[1][g]            
            
            """ For the induction view, merge the pos & neg ROI together if they are separated """
            if(cf.view_type[view]=="Induction" and g < len(gpe)-1):
                merge = False
                if(np.mean(dc.data_daq[daq_chan, tdc_start:tdc_stop+1]) > 0.):
                    if(start[0][g+1] == daq_chan):
                        if(np.mean(dc.data_daq[daq_chan, start[1][g+1]:end[1][g+1]]) < 0.):
                            if(start[1][g+1] - tdc_stop < 10):
                                tdc_stop = end[1][g+1]
                                merge=True
                if(merge==False):
                    if(tdc_stop-tdc_start < 20):
                        continue

            """ add l/r paddings """
            for il in range(pad_left, 0, -1):
                if(tdc_start-1>=0 and not ROI[daq_chan, tdc_start-1]):
                    tdc_start -= 1
                else:
                    break

            for ir in range(0, pad_right):
                if(tdc_stop+1 < cf.n_sample and not ROI[daq_chan,tdc_stop+1]):
                    tdc_stop += 1
                else:
                    break
                      
            
            adc = dc.data_daq[daq_chan, tdc_start:tdc_stop+1]                
            mean, rms = dc.evt_list[-1].noise_filt.ped_mean[daq_chan], dc.evt_list[-1].noise_filt.ped_rms[daq_chan]
            thr1 = mean + n_sig_coll_1 * rms
            thr2 = mean + n_sig_coll_2 * rms
            thr3 = mean + n_sig_ind * rms

            if(thr1 < 0.5): thr1 = 0.5
            if(thr2 < 0.5): thr2 = 0.5
            if(thr3 < 0.5): thr3 = 0.5



            hh = hit_search(adc, view, daq_chan, tdc_start, dt_min, thr1, thr2, thr3)
                                        
            """add padding to found hits"""
            for i in range(len(hh)): 
                """ to the left """
                if(i == 0): 
                    if(hh[i].start > pad_left):
                        hh[i].start -= pad_left
                    else:
                        hh[i].start = 0
                else:
                    if(hh[i].start - pad_left > hh[i-1].stop):
                        hh[i].start -= pad_left
                    else:
                        hh[i].start = hh[i-1].stop + 1
                

                """ to the right """
                if(i == len(hh)-1):
                    if(hh[i].stop < cf.n_sample - pad_right):
                        hh[i].stop += pad_right
                    else:
                        hh[i].stop = cf.n_sample
                else:
                    if(hh[i].stop + pad_right < hh[i+1].start):
                        hh[i].stop += pad_right
                    else:
                        hh[i].stop = hh[i+1].start - 1


            dc.evt_list[-1].n_hits[view] += len(hh)
            dc.hits_list.extend(hh)


    v = lar.drift_velocity()


    """ transforms hit channel and tdc to positions """
    [x.hit_positions(v) for x in dc.hits_list]

    """ set hit an index number """
    [dc.hits_list[i].set_index(i) for i in range(len(dc.hits_list))]
    
    """ compute hit charge in fC """
    [recompute_hit_charge(x) for x in dc.hits_list]
    [x.hit_charge() for x in dc.hits_list]

    """ debug """
    #[x.dump() for x in dc.hits_list]

