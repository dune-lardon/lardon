import config as cf
import data_containers as dc
import lar_param as lar

import numpy as np


def hit_search(data, view, daq_chan, start, dt_min, thr1, thr2, thr3):
    if(cf.view_type[view] == "Collection"):
        return hit_search_collection(data, view, daq_chan, start, dt_min, thr1, thr2)

    elif(cf.view_type[view] == "Induction"):
        return hit_search_induction(data, view, daq_chan, start, dt_min, thr3)
    else : 
        print(cf.view_type[view], " is not recognized")
        sys.exit()




def hit_search_induction(data, view, daq_chan, start, dt_min, thr):
    """ very basic induction-like hit finder """
    """ WARNING : CANNOT FIND OVERLAPPING HITS """

    ll = []

    npts = len(data)
    hitPosFlag = False
    hitNegFlag = False

    i=0

    posSamp = 0
    negSamp = 0

    h = dc.hits(view, daq_chan, start, 0, 0., 0., 0., 0., 0.)


    while(i<npts):

        if(i < npts and hitPosFlag == False and hitNegFlag == False):
            i += 1

        """ start with the positive blob of the hit """
        while(i < npts and data[i] >= thr and hitNegFlag == False):
            val = data[i]        
            it = i+start
            posSamp += 1

            """ first point above thr """
            if(hitPosFlag == False):
                hitPosFlag = True

                h.start = it

                h.max_t = it
                h.max_adc = val
                
                
            """ update the maximum case """
            if(val > h.max_adc):
                h.max_t = it
                h.max_adc = val                
                    
            if(hitPosFlag):
                h.charge_int += val
                
            i+=1

        if(posSamp < dt_min):
            hitPosFlag = False
            posSamp = 0

        while(i < npts and hitPosFlag and hitNegFlag == False and np.fabs(data[i]) < thr):
            h.charge_int += np.fabs(val)
            i += 1

        """ now the negative part """

        while(i < npts and hitPosFlag and data[i] < -1.*thr):            
            val = data[i]        
            it = i+start
            negSamp += 1

            """ first point below thr """
            if(hitNegFlag == False):
                hitNegFlag = True

                h.min_t = it
                h.min_adc = val
                
                
            """ update the minimum case """
            if(val < h.min_adc):
                h.min_t = it
                h.min_adc = val                
                    
            if(hitNegFlag):
                h.charge_int += -1.*val

            h.stop = it
            i+=1

        if(negSamp < dt_min):
            hitNegFlag = False
            negSamp = 0

        if(hitPosFlag and hitNegFlag):
            ll.append(h)
            break
    

    return ll



def hit_search_collection(data, view, daq_chan, start, dt_min, thr1, thr2):
    """search hit-shape in a list of points"""
    """algorithm from qscan"""
    
    ll = []
    npts = len(data)
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
                
                h = dc.hits(view,daq_chan,it,0,0.,it,val, 0., 0.)
                minSamp = -1
                
            if(it > h.max_t and val < h.max_adc - thr2 and (minSamp==-1 or minimum >= val)):
                minSamp = it
                minimum = val

                
            if(minSamp >= 0 and it > minSamp and val > minimum + thr2 and (it-h.start) >= dt_min):
                h.stop = minSamp-1
                ll.append(h)
                hitFlag = True
                singleHit = False
                h = dc.hits(view,daq_chan,minSamp,0,0,it,val, 0., 0.)
                minSamp = -1

                
            if(h.stop == 0 and val > h.max_adc):
                h.max_t = it
                h.max_adc = val
                if(minSamp >= 0):
                    minSamp = -1
                    minimum = val
                    
            h.charge_int += val
            i+=1
        if(hitFlag == True):
            hitFlag = False
            h.stop = it-1

            #if((singleHit and (h.stop-h.start >= dt_min)) or not singleHit):
            if(h.stop-h.start >= dt_min):
                ll.append(h)

        i+=1
    return ll

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

