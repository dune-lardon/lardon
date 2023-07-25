import os
import config as cf
import data_containers as dc
import numpy as np
from abc import ABC, abstractmethod


def set_unused_channels():
    if(len(cf.broken_channels) > 0):
       print(" Removing ",len(cf.broken_channels)," broken channels :")

    for i in range(cf.n_tot_channels):
        
        module, view, chan = dc.chmap[i].module, dc.chmap[i].view, dc.chmap[i].vchan
        
        if(view >= cf.n_view or view < 0 or i in cf.broken_channels or cf.module_used[module]==False):
            dc.alive_chan[i,:] = False

            if(i in cf.broken_channels):
                print(i, ' [m',module,' v',view,' ch',chan,'], ',sep='',end='')
    print('\n')

def arange_in_view_channels():
    for i in range(cf.n_tot_channels):
        module, view, chan = dc.chmap[i].get_ana_chan()
        if(view >= cf.n_view or view < 0 or i in cf.broken_channels):
            continue
        dc.data[module, view, chan] = dc.data_daq[i]



def get_mapping(detector, elec):
    if(os.path.exists(cf.channel_map) is False):
        print('the channel mapping file ', fmap, ' does not exists')
        exit()

    if(detector == "cb1" or detector == 'cb2' or detector == 'cb'):
        if(elec == "top"):
            get_cb_top_mapping()
        elif(elec == "bot"):
            get_cb_bot_mapping()
        else : 
            print("the electronic ",elec, " for ", detector, " is not recognized")
            exit()
    elif(detector == "dp" and elec == "top"):
        get_dp_mapping()

    elif(detector == "50l" and elec== "bot"):
        get_50l_bot_mapping()

    else :
        print("the electronic ",elec, " for ", detector, " is not recognized")
        exit()

def get_cb_top_mapping():
    strip = get_strip_length()
    calib  = get_calibration()

    # TO BE UPDATED IN THE FUTURE
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
            gain = calib[daqch]



            if(globch >= 0 and view >= 0 and view < cf.n_view):
                
                length, capa = strip[globch]
                
                nrepet = int(np.floor(channel/cf.view_chan_repet[view]))
                pos = channel%cf.view_chan_repet[view] * cf.view_pitch[view] + cf.view_offset_repet[module][view][nrepet] +  cf.view_pitch[view]/2.
                #print('view ', view, ' channel ', channel, ' at ', pos)
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
            gain = calib[daqch]


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
            gain = calib[daqch]


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
            gain = calib[daqch]
            
            globch = 3840*view + 960*crp + channel

            pos = channel*cf.view_pitch[view]
            if(view == 0):
                if(crp == 1 or crp == 2):
                    pos -= 300.
            else:
                if(crp == 2 or crp == 3):
                    pos -= 300.

            length, capa = 300., 1.
            c = dc.channel(daqch, globch, crp, view, channel, length, capa, gain, pos)
            dc.chmap.append(c)

                

def get_strip_length():
    strip = []
    with open(cf.strips_length,"r") as f:
        for line in f.readlines()[1:]:
            li = line.split()
            view =  int(li[0])
            vch  =  int(li[1])
            globch = int(li[2])
            length = float(li[3])

            capa = length*cf.view_capa[view]

            strip.append( (length, capa) )
        return strip




def get_calibration():
    gain = []
    fC_per_e = 1.602e-4 #e- charge in fC
    if(len(cf.channel_calib) > 0):
        with open(cf.channel_calib,"r") as f:
            for line in f.readlines()[1:]:
                li = line.split()
                g = float(li[7])
                gain.append(g*fC_per_e)
    else:
        gain = [cf.e_per_ADCtick*fC_per_e for x in range(cf.n_tot_channels)]
    return gain
            
