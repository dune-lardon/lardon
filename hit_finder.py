import config as cf
import data_containers as dc
import lar_param as lar

import numpy as np
import numba as nb


def hit_search(data, module, view, daq_chan, start, dt_min, thr1, thr2, thr3):

    ll = []

    if(cf.view_type[view] != "Collection" and cf.view_type[view] != "Induction"): 
        print(cf.view_type[view], " is not recognized")
        sys.exit()

    elif(cf.view_type[view] == "Collection"):
        n, h_start, h_stop, h_max_t, h_max_adc = hit_search_collection_nb(data,start, dt_min, thr1, thr2, cf.n_sample)

        for i in range(n):
            ll.append(dc.hits(module, view, daq_chan, h_start[i], h_stop[i],  h_max_t[i], h_max_adc[i], -1, 0., -1, "Collection"))
        return ll


    else: 
        """ look for collection-like signal in the cumulative sum of the waveforms """        
        csum = np.cumsum(data)
        c_n, c_h_start, c_h_stop, c_h_max_t, c_h_max_adc = hit_search_collection_nb(csum, start, dt_min, thr1, thr2, cf.n_sample)

        if(c_n > 0):
            for i in range(c_n):
                idx_b, idx_e = c_h_start[i]-start, c_h_stop[i]-start        
                ll.extend([h for h in search_induction(data[idx_b:idx_e], module, view, daq_chan, start+idx_b, dt_min, thr1, thr2, thr3)])
            return ll

        
        """ nothing found, test if the hit is collection-type """        
        if(c_n == 0):
            ll.extend([h for h in search_induction(data, module, view, daq_chan, start, dt_min, thr1, thr2, thr3)])
            return ll
    return ll


def search_induction(data, module, view, daq_chan, start, dt_min, thr1, thr2, thr3): 
    """ search for a bipolar hit """
    n, h_start, h_stop, h_max_t, h_max_adc, h_min_t, h_min_adc, h_zero_t = hit_search_induction_nb(data, start, dt_min, thr3)
    ll = []
    
    if(n > 0):
        for i in range(n):
            ll.append(dc.hits(module, view, daq_chan, h_start[i], h_stop[i],  h_max_t[i], h_max_adc[i], h_min_t[i], h_min_adc[i], h_zero_t[i], "Induction"))
        return ll
    else:
        """ if nothing found, search for positive bump """
        if(np.mean(data) > thr1):
            n, h_start, h_stop, h_max_t, h_max_adc = hit_search_collection_nb(data, start, dt_min, thr1, thr2, cf.n_sample)
            
            for i in range(n):
                ll.append(dc.hits(module, view, daq_chan, h_start[i], h_stop[i],  h_max_t[i], h_max_adc[i], -1, 0., -1, "Collection"))
            return ll

        """ last try is a negative bump """
        if(np.mean(data) < -1*thr1):
            n, h_start, h_stop, h_max_t, h_max_adc = hit_search_collection_nb(-1*data, start, dt_min, thr1, thr2, cf.n_sample)
            
            for i in range(n):
                ll.append(dc.hits(module, view, daq_chan, h_start[i], h_stop[i],  -1, 0., h_max_t[i], -1.*h_max_adc[i], -1, "Collection"))
                    
            return ll

    return ll

    

@nb.njit('Tuple((int64,int32[:],int32[:],int32[:],float64[:],int32[:],float64[:],int32[:]))(float64[:],int64,int64,float64)')
def hit_search_induction_nb(data, start, dt_min, thr):
    """ very basic induction-like hit finder """
    """ WARNING : CANNOT FIND OVERLAPPING HITS """
    
    npts = len(data)

    h_num = 0 # store number of found hits
    # list of hits parameters to be returned by this numba function (see dc.hits)
    h_start     = np.zeros(npts,dtype=np.int32)
    h_stop      = np.zeros(npts,dtype=np.int32)
    h_max_t     = np.zeros(npts,dtype=np.int32)
    h_max_adc   = np.zeros(npts)
    h_min_t     = np.zeros(npts,dtype=np.int32)
    h_min_adc   = np.zeros(npts)
    h_zero_t    = np.zeros(npts,dtype=np.int32)

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
                h_zero_t[h_num]  = it
                
            """ update the maximum case """
            if(val > h_max_adc[h_num]):
                h_max_t[h_num] = it
                h_max_adc[h_num] = val                
            
            

            i+=1

        if(posSamp < dt_min):
            hitPosFlag = False
            posSamp = 0

        val = data[i]

        h_zero_t[h_num] = i+start

        """ in between the two polarities """
        while(i < npts and hitPosFlag and hitNegFlag == False and val >= -1.*thr):            
            i += 1
            val = data[i]
            if(val >= 0):
                h_zero_t[h_num] = i+start
            
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
                    

            h_stop[h_num] = it
            i+=1

        if(negSamp < dt_min):
            hitNegFlag = False
            negSamp = 0

        if(hitPosFlag and hitNegFlag):
            h_num += 1 
            break


    return h_num, h_start, h_stop, h_max_t, h_max_adc, h_min_t, h_min_adc, h_zero_t

@nb.njit('Tuple((int64,int32[:],int32[:],int32[:],float64[:]))(float64[:],int64,int64,float64,float64, int64)')
def hit_search_collection_nb(data, start, dt_min, thr1, thr2, nsamp):
    """search hit-shape in a list of points"""
    """algorithm from qscan"""
    npts = len(data)
    
    h_num = 0 # store number of found hits
    # list of hits parameters to be returned by this numba function (see dc.hits)
    h_start     = np.zeros(npts,dtype=np.int32)
    h_stop      = np.zeros(npts,dtype=np.int32)
    h_charge_int= np.zeros(npts)
    h_max_t     = np.zeros(npts,dtype=np.int32)
    h_max_adc   = np.zeros(npts)
 
    hitFlag = False

    i=0
    minimum = nsamp
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
                h_max_t[h_num]     = it
                h_max_adc[h_num]   = val
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
                h_max_t[h_num]     = it
                h_max_adc[h_num]   = val

                minSamp = -1

                
            if(h_stop[h_num] == 0 and val > h_max_adc[h_num]):
                h_max_t[h_num] = it
                h_max_adc[h_num] = val
                if(minSamp >= 0):
                    minSamp = -1
                    minimum = val
                    
            i+=1
        if(hitFlag == True):
            hitFlag = False
            h_stop[h_num] = it-1

            #if((singleHit and (h_stop[h_num]-h_start[h_num] >= dt_min)) or not singleHit):
            if(h_stop[h_num]-h_start[h_num] >= dt_min):
                h_num += 1 

        i+=1
    return h_num, h_start, h_stop, h_max_t, h_max_adc


def recompute_hit_charge(hit):
    view, daq_ch, pad_start, pad_stop, zero, sig = hit.view, hit.daq_channel, hit.pad_start, hit.pad_stop, hit.zero_t, hit.signal

    val = 0.
    mean = dc.evt_list[-1].noise_filt.ped_mean[daq_ch]

    if(sig == "Collection"):
        for t in range(pad_start, pad_stop):
            val += dc.data_daq[daq_ch, t] - mean

        hit.charge_pos = val
        hit.charge_neg = -0.

    elif(sig == "Induction"):
        for t in range(pad_start, zero):
            val += dc.data_daq[daq_ch, t] - mean
        hit.charge_pos = val

        val = 0
        for t in range(zero, pad_stop):
            val += dc.data_daq[daq_ch, t] + mean
        hit.charge_neg = val

    else:
        print('type of view not recognized ... ')
        sys.exit()

        
def find_hits():

    pad_left = dc.reco['hit_finder']['pad']['left']
    pad_right = dc.reco['hit_finder']['pad']['left']
    dt_min = dc.reco['hit_finder']['coll']['dt_min']
    n_sig_coll_1 = dc.reco['hit_finder']['coll']['amp_sig'][0]
    n_sig_coll_2 = dc.reco['hit_finder']['coll']['amp_sig'][1]
    n_sig_ind  = dc.reco['hit_finder']['ind']['amp_sig'][0]
    

    
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
    merge = False
    
    for g in range(len(gpe)):
        if(gpe[g]):
            if(merge == True):
                merge = False
                continue
            
            """ make sure starts and ends of hit group are in the same channel """
            assert start[0][g] == end[0][g], "Hit Mismatch"

            daq_chan = start[0][g]
            
            module, view, channel = dc.chmap[daq_chan].get_ana_chan()
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
                    if(tdc_stop-tdc_start < dt_min):
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

            
                
            hh = hit_search(adc, module, view, daq_chan, tdc_start, dt_min, thr1, thr2, thr3)

            
            """add padding to found hits"""
            for i in range(len(hh)): 
                """ to the left """
                if(i == 0): 
                    if(hh[i].start > pad_left):
                        hh[i].pad_start -= pad_left
                    else:
                        hh[i].pad_start = 0
                else:
                    if(hh[i].start - pad_left > hh[i-1].pad_stop):
                        hh[i].pad_start -= pad_left
                    else:
                        hh[i].pad_start = hh[i-1].stop #+ 1
                

                """ to the right """
                if(i == len(hh)-1):
                    if(hh[i].stop < cf.n_sample - pad_right):
                        hh[i].pad_stop += pad_right
                    else:
                        hh[i].pad_stop = cf.n_sample
                else:
                    if(hh[i].stop + pad_right < hh[i+1].pad_start):
                        hh[i].pad_stop += pad_right
                    else:
                        hh[i].pad_stop = hh[i+1].start #- 1


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




def find_pds_peak():
    
    pad_left = dc.reco['pds']['hit_finder']['pad']['left']
    pad_right = dc.reco['pds']['hit_finder']['pad']['right']
    dt_min = dc.reco['pds']['hit_finder']['dt_min']
    n_sig_coll_1 = dc.reco['pds']['hit_finder']['amp_sig'][0]
    n_sig_coll_2 = dc.reco['pds']['hit_finder']['amp_sig'][1]


    
    """ get boolean roi based on mask and alive channels """
    ROI = np.array(~dc.mask_pds, dtype=bool)

    """ adds 0 (False) and the start and end of each waveform """
    falses = np.zeros((cf.n_pds_channels,1),dtype=int)
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
            glob_chan = start[0][g]
            
            tdc_start = start[1][g]
            tdc_stop = end[1][g]            

            
            """ add l/r paddings """
            for il in range(pad_left, 0, -1):
                if(tdc_start-1>=0 and not ROI[glob_chan, tdc_start-1]):
                    tdc_start -= 1
                else:
                    break

            for ir in range(0, pad_right):
                if(tdc_stop+1 < cf.n_sample and not ROI[glob_chan,tdc_stop+1]):
                    tdc_stop += 1
                else:
                    break
                      
            
            adc = dc.data_pds[glob_chan, tdc_start:tdc_stop+1].astype(np.float64)
            mean, rms = dc.evt_list[-1].noise_pds.ped_mean[glob_chan], dc.evt_list[-1].noise_pds.ped_rms[glob_chan]
            thr1 = n_sig_coll_1 * rms
            thr2 = n_sig_coll_2 * rms


            if(thr1 < 0.5): thr1 = 0.5
            if(thr2 < 0.5): thr2 = 0.5

            chan = dc.chmap_pds[glob_chan].chan

            hh = []
            
            n, h_start, h_stop, h_max_t, h_max_adc = hit_search_collection_nb(adc,tdc_start, dt_min, thr1, thr2, cf.n_pds_sample)

            for i in range(n):
                hh.append(dc.pds_peak(glob_chan, chan,  h_start[i], h_stop[i],  h_max_t[i], h_max_adc[i]))

            
            """add padding to found hits"""
            for i in range(len(hh)): 
                """ to the left """
                if(i == 0): 
                    if(hh[i].pad_start > pad_left):
                        hh[i].pad_start -= pad_left
                    else:
                        hh[i].pad_start = 0
                else:
                    if(hh[i].pad_start - pad_left > hh[i-1].pad_stop):
                        hh[i].pad_start -= pad_left
                    else:
                        hh[i].pad_start = hh[i-1].pad_stop + 1
                

                """ to the right """
                if(i == len(hh)-1):
                    if(hh[i].pad_stop < cf.n_pds_sample - pad_right):
                        hh[i].pad_stop += pad_right
                    else:
                        hh[i].pad_stop = cf.n_pds_sample
                else:
                    if(hh[i].pad_stop + pad_right < hh[i+1].pad_start):
                        hh[i].pad_stop += pad_right
                    else:
                        hh[i].pad_stop = hh[i+1].pad_start - 1


            dc.evt_list[-1].n_pds_peak[glob_chan] += len(hh)
            dc.pds_peak_list.extend(hh)

    print('found ', dc.evt_list[-1].n_pds_peak, ' PDS peaks')
    #[x.dump() for x in dc.pds_peak_list]
