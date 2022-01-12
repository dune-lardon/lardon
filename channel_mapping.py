import os
import config as cf
import data_containers as dc
import numpy as np
from abc import ABC, abstractmethod


def set_unused_channels():
    if(len(cf.broken_channels) > 0):
       print("remove daq broken channels: ",cf.broken_channels)

    for i in range(cf.n_tot_channels):
        view, chan = dc.chmap[i].view, dc.chmap[i].vchan
        if(view >= cf.n_view or view < 0 or i in cf.broken_channels):
            dc.alive_chan[i,:] = False
            
def arange_in_view_channels():
    for i in range(cf.n_tot_channels):
        view, chan = dc.chmap[i].view, dc.chmap[i].vchan
        if(view >= cf.n_view or view < 0 or i in cf.broken_channels):
            continue
        dc.data[view, chan] = dc.data_daq[i]



def get_mapping(elec):
    print("channel mapping file is ", cf.channel_map)
    if(os.path.exists(cf.channel_map) is False):
        print('the channel mapping file ', fmap, ' does not exists')
        sys.exit()
    if(elec == "top"):
        get_top_mapping()
    elif(elec == "bot"):
        get_bot_mapping()
    else : 
        print("the electronic setting is not recognized")

        
def get_top_mapping():
    strip = get_strip_length()
    with open(cf.channel_map, 'r') as f:
        for line in f.readlines():
            li = line.split()
            daqch =  int(li[0])
            kel   =  int(li[1])
            kelch =  int(li[2])
            crate =  int(li[3])
            slot  =  int(li[4])
            slotch =  int(li[5])
            view =  int(li[6])
            channel =  int(li[7])
            globch = int(li[8])

            if(globch >= 0 and view >= 0 and view < cf.n_view):
                length, capa, tot_length, tot_capa = strip[globch]
            else:
                length, capa, tot_length, tot_capa = -1, -1, -1, -1
            c = dc.channel(daqch, globch, view, channel, length, capa, tot_length, tot_capa)
            dc.chmap.append(c)

def get_bot_mapping():
    strip = get_strip_length()
    with open(cf.channel_map, 'r') as f:
        for line in f.readlines():
            li = line.split()
            daqch =  int(li[0])
            globch = int(li[1])
            view = int(li[2])
            channel = int(li[3])
            

            if(globch >= 0 and view >= 0 and view < cf.n_view):
                length, capa, tot_length, tot_capa = strip[globch]
            else:
                length, capa, tot_length, tot_capa = -1, -1, -1, -1
            
            c = dc.channel(daqch, globch, view, channel, length, capa, tot_length, tot_capa)
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
            tot_length = float(li[4])

            capa = length*cf.view_capa[view]
            tot_capa = tot_length*cf.view_capa[view]
            strip.append( (length, capa, tot_length, tot_capa) )

        return strip
            
