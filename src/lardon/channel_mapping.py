import os

import lardon.config as cf
import lardon.data_containers as dc

import numpy as np
from itertools import tee, islice, chain


def set_unused_channels():
    if(len(cf.broken_channels) > 0):
       print(" Removing ",len(cf.broken_channels)," broken channels")

    daqch_start = cf.module_daqch_start[cf.imod]



    for i in range(cf.module_nchan[cf.imod]):
        idaq = i + daqch_start
        module, view, chan, glob = dc.chmap[idaq].module, dc.chmap[idaq].view, dc.chmap[idaq].vchan, dc.chmap[idaq].globch

        
        if(view >= cf.n_view or view < 0 or glob in cf.broken_channels):
            dc.alive_chan[i] = False
            #print('module ', module, ' v ', view, ' channel ', chan, ' = ', glob)

        else:
            dc.alive_chan[i] = True


    
def arange_in_view_channels():
    daqch_start = cf.module_daqch_start[cf.imod]
    daqch_stop = daqch_start + cf.module_nchan[cf.imod]
    
    for i in range(daqch_start, daqch_stop):
        module, view, chan = dc.chmap[i].get_ana_chan()

        glob = dc.chmap[i].get_globch()        
        if(view >= cf.n_view or view < 0 or glob in cf.broken_channels):
            continue
        dc.data[view, chan] = dc.data_daq[i-daqch_start]


def arange_in_glob_channels(array):
    glob_order = [(i, dc.chmap[i].get_globch()) for i in range(cf.n_tot_channels)]
    glob_order = sorted(glob_order, key=lambda tup: tup[1])            

    array = [array[x[0]] for x in glob_order]    
    return array

        
    
def previous_and_next(some_iterable):
    #from https://stackoverflow.com/questions/1011938/loop-that-also-accesses-previous-and-next-values
    prevs, items, nexts = tee(some_iterable, 3)
    prevs = chain([None], prevs)
    nexts = chain(islice(nexts, 1, None), [None])
    return zip(prevs, items, nexts)
    
def is_true_channel(x):
    (module, view, channel, daq) = x
    if(module < 0 or view < 0 or channel < 0 or view >= cf.n_view):    
        return False
    else:
        return True

def get_neighbour(chan, other):
    if(other == None):
        return -1
    elif(is_true_channel(other)==False):
        return -1
    else:
        if(chan[0] == other[0] and chan[1] == other[1]):
            return other[3]
        else:
            return -1
    
def get_mapping(detector):
    if(os.path.exists(cf.channel_map) is False):
        print('the channel mapping file ', cf.channel_map, ' does not exists')
        exit()
    if(detector == "cb1top"):
        get_cb_top_mapping()
    elif(detector == "cbtop"):
        get_cb_top_mapping()
    elif(detector == "cb1bot"):
        get_cb_bot_mapping()
    elif(detector == "cbbot"):
        get_cb_bot_mapping()
    elif(detector == "dp"):
        get_dp_mapping()
    elif(detector == "50l"):
        get_50l_bot_mapping()
    elif(detector == "pdhd"):
        get_pdhd_bot_mapping()
    elif(detector == "pdvd"):
        get_pdvd_mapping()
        
    else:
        print("the detector ", detector, " is not recognized by the channel mapping")
        exit()

    #daq_shift = cf.module_daqch_start[cf.imod]
    """ get the previous and next physical channel """
    ch_list = [(x.module, x.view, x.vchan, x.daqch) for x in dc.chmap]    
    ch_list = sorted(ch_list, key=lambda x:(x[0],x[1],x[2]))
    
    for prev, item, nxt in previous_and_next(ch_list):
        if(is_true_channel(item) == False):
            continue
        else:
            daqch = item[3]
            
            prev_daqch = get_neighbour(item, prev)
            next_daqch = get_neighbour(item, nxt)

            view, vchan = item[1], item[2]
            if(vchan==0):
                prev_daqch = -1
            if(vchan==cf.view_nchan[view]):
                next_daqch = -1
            if(vchan%cf.view_chan_repet[view]==0):
                prev_daqch = -1
            if((vchan-1)%cf.view_chan_repet[view]==0):
                next_daqch = -1

            dc.chmap[daqch].set_prev_next(prev_daqch, next_daqch)


def get_pds_mapping(detector):
    """ not the best channel mapping, to be improved """
    if(os.path.exists(cf.pds_channel_map) is False):
        print('the channel mapping file ', cf.pds_channel_map, ' does not exists')
        exit()

    with open(cf.pds_channel_map, 'r') as f:
        for line in f.readlines()[1:]:
            li = line.split()
            daqch  =  int(li[0])
            globch = int(li[1])
            det    = li[2]
            chan   = int(li[3])
            mod    = int(li[4])
            data   = li[5]
            c = dc.channel_pds(daqch, globch, det, chan, mod, data)
            dc.chmap_daq_pds.append(c)

            if(globch >= 0):
                dc.chmap_pds.append(c)
    dc.chmap_pds.sort(key=lambda x: x.globch)

    
    
def get_cb_top_mapping():
    strip = get_strip_length()
    calib  = get_calibration()

    module = 0

    with open(cf.channel_map, 'r') as f:
        for line in f.readlines()[1:]:
            li = line.split()
            daqch =  int(li[0])
            kel   =  int(li[1])
            kelch =  int(li[2])
            AB    =  int(li[3])
            crate =  int(li[4])
            slot  =  int(li[5])
            slotch =  int(li[6])
            view =  int(li[7])
            channel =  int(li[8])
            globch = int(li[9])
            gain = calib[daqch] if len(calib)==cf.n_tot_channels else calib[module]



            if(globch >= 0 and view >= 0 and view < cf.n_view):
                
                length, capa = strip[globch]
                
                nrepet = int(np.floor(channel/cf.view_chan_repet[view]))
                pos = channel%cf.view_chan_repet[view] * cf.view_pitch[view] + cf.view_offset_repet[module][view][nrepet] +  cf.view_pitch[view]/2.

            else:
                length, capa = -1, -1
                pos=-9999.
            c = dc.channel(daqch, globch, module, view, channel, length, capa, gain, pos)
            dc.chmap.append(c)

def get_cb_bot_mapping():
    strip = get_strip_length()
    calib  = get_calibration()
    # TO BE UPDATED IN THE FUTURE
    module = 0

    with open(cf.channel_map, 'r') as f:
        for line in f.readlines()[1:]:
            li = line.split()
            daqch =  int(li[0])
            globch = int(li[1])
            AB = int(li[2])
            femb = int(li[3])
            asic = int(li[4])
            asic_ch = int(li[5])        
            view = int(li[6])
            channel = int(li[7])
            gain = calib[daqch] if len(calib)==cf.n_tot_channels else calib[module]

            if(globch >= 0 and view >= 0 and view < cf.n_view):
                length, capa = strip[globch]

                nrepet = int(np.floor(channel/cf.view_chan_repet[view]))
                pos = channel%cf.view_chan_repet[view] * cf.view_pitch[view] + cf.view_offset_repet[module][view][nrepet] +  cf.view_pitch[view]/2.

                #pos = channel%cf.view_chan_repet[view] * cf.view_pitch[view] + cf.view_offset[module][view] +  cf.view_pitch[view]/2.

            else:
                length, capa = -1, -1
                pos=-9999.            
            c = dc.channel(daqch, globch, module, view, channel, length, capa, gain, pos)
            dc.chmap.append(c)


def get_50l_bot_mapping():
    strip = get_strip_length()
    calib  = get_calibration()
    module = 0

    with open(cf.channel_map, 'r') as f:
        for line in f.readlines()[1:]:
            li = line.split()
            daqch =  int(li[0])
            globch = int(li[1])
            AB = int(li[2])
            femb = int(li[3])
            asic = int(li[4])
            asic_ch = int(li[5])        
            view = int(li[6])
            channel = int(li[7])
            gain = calib[daqch] if len(calib)==cf.n_tot_channels else calib[module]
            if(globch >= 0 and view >= 0 and view < cf.n_view):
                length, capa = strip[globch]

                nrepet = int(np.floor(channel/cf.view_chan_repet[view]))
                if(view == 0):
                    pos = int(np.fabs(cf.view_nchan[view]-(channel%cf.view_chan_repet[view])+0.5)) * cf.view_pitch[view] + cf.view_offset_repet[module][view][nrepet]
                else:
                    pos = (channel%cf.view_chan_repet[view]+0.5) * cf.view_pitch[view] + cf.view_offset_repet[module][view][nrepet]

                    
            else:
                length, capa = -1, -1
                pos=-9999.            
            c = dc.channel(daqch, globch, module, view, channel, length, capa, gain, pos)
            dc.chmap.append(c)


def get_dp_mapping():
    calib  = get_calibration()
    with open(cf.channel_map, 'r') as f:
        for line in f.readlines()[1:]:
            li = line.split()
            daqch =  int(li[0])
            crp   =  int(li[1])
            view =  int(li[2])
            channel =  int(li[3])
            #gain = calib[daqch]
            
            globch = 3840*view + 960*crp + channel

            pos = channel*cf.view_pitch[view]
            if(view == 0):
                if(crp == 1 or crp == 2):
                    pos -= 300.
            else:
                if(crp == 2 or crp == 3):
                    pos -= 300.

            length, capa = 300., 1.
            gain = calib[daqch] if len(calib)==cf.n_tot_channels else calib[crp-1]
            c = dc.channel(daqch, globch, crp, view, channel, length, capa, gain, pos)
            dc.chmap.append(c)



def get_pdhd_bot_mapping():
    strip = get_strip_length()
    calib  = get_calibration(idx=1)


    with open(cf.channel_map, 'r') as f:
        for line in f.readlines()[1:]:
            li = line.split()
            daqch =  int(li[0]) 
            globch = int(li[1]) 
            AB = int(li[2])
            femb = int(li[3])
            asic = int(li[4])
            asic_ch = int(li[5])        
            view = int(li[6])
            channel = int(li[7])
            module = int(li[8])-1


            gain = calib[globch] if len(calib)==cf.n_tot_channels else calib[module]

            
            if(globch >= 0 and view >= 0 and view < cf.n_view):
                length, capa = strip[globch]

                nrepet = int(np.floor(channel/cf.view_chan_repet[view]))

                pos = channel%cf.view_chan_repet[view] * cf.view_pitch[view] + cf.view_offset_repet[module][view][nrepet] +  cf.view_pitch[view]/2.

            else:
                length, capa = -1, -1
                pos=-9999.            
            c = dc.channel(daqch, globch, module, view, channel, length, capa, gain, pos)
            dc.chmap.append(c)


def get_pdvd_mapping():

    
    strip = get_strip_length()
    calib  = get_calibration(idx=1)
    
    n_dummy = 0
    with open(cf.channel_map, 'r') as f:
        for line in f.readlines()[1:]:
            li = line.split()
            daqch =  int(li[0])
            globch = int(li[1])
            AB = int(li[2])
            femb = int(li[3])
            asic = int(li[4])
            asic_ch = int(li[5])        
            view = int(li[6])
            channel = int(li[7])
            module  = int(li[8])
            slot    = int(li[9])
            card = int(10*slot+femb)

            gain = calib[daqch] if len(calib)==cf.n_tot_channels else calib[module]

            if(globch >= 0 and view >= 0 and view < cf.n_view):
                length, capa = strip[globch]

                nrepet = int(np.floor(channel/cf.view_chan_repet[view]))
                
                
                pos = channel%cf.view_chan_repet[view] * cf.view_pitch[view] + cf.view_offset_repet[module][view][nrepet] +  cf.view_pitch[view]/2.                

            else:
                n_dummy += 1
                length, capa = -1, -1
                pos=-9999.            
            c = dc.channel(daqch, globch, module, view, channel, length, capa, gain, pos, card)
            dc.chmap.append(c)
            
        print('nb of dummy channels : ', n_dummy)

def get_strip_length():
    strip = []
    if(len(cf.strips_length) > 0):
        print('---> strip length is ', cf.strips_length, len(cf.strips_length))
        with open(cf.strips_length,"r") as f:
            for line in f.readlines()[1:]:
                li = line.split()
                view =  int(li[0])
                vch  =  int(li[1])
                globch = int(li[2])
                length = float(li[3])
                
                capa = length*cf.view_capa[view]
                
                strip.append( (length, capa) )
    else:
        strip = [(0,0) for x in range(cf.n_tot_channels)]
    return strip




def get_calibration(idx=7):
    gain = []
    fC_per_e = 1.602e-4 #e- charge in fC
    
    if(len(cf.channel_calib) > 0):
        with open(cf.channel_calib,"r") as f:
            for line in f.readlines()[1:]:
                li = line.split()
                g = float(li[idx])
                gain.append(g*fC_per_e)
    else:
        #gain = [cf.e_per_ADCtick*fC_per_e for x in range(cf.n_tot_channels)]
        gain = [cf.e_per_ADCtick[x]*fC_per_e for x in range(cf.n_module)]
    return gain
            
