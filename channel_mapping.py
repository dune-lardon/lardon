import os
import config as cf
import data_containers as dc
import numpy as np
from abc import ABC, abstractmethod


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
    print("reading top channel mapping ")
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
            
            c = dc.channel(daqch, globch, view, channel)
            dc.chmap.append(c)
            
        

def get_bot_mapping():
    print("reading bottom channel mapping ")
    with open(cf.channel_map, 'r') as f:
        for line in f.readlines():
            li = line.split()
            daqch =  int(li[0])
            globch = int(li[1])
            view = int(li[2])
            channel = int(li[3])
            
            c = dc.channel(daqch, globch, view, channel)
            dc.chmap.append(c)
                

