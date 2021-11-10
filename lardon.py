import sys
import argparse
import numpy as np
from datetime import datetime
import det_spec as det


parser = argparse.ArgumentParser()
parser.add_argument('-elec', dest='elec', help='which electronics is used [tde, top, bde, bot]', default='', required=True)
parser.add_argument('-run', dest='run', help='run number to be processed', default="", required=True)
parser.add_argument('-sub', dest='sub', help='which subfile number [default is the first]', default="", required=True)
parser.add_argument('-n', dest='nevent', type=int, help='number of events to process in the file [default (or -1) is all]', default=-1)
parser.add_argument('-det', dest='detector', help='which detector is looked at [default is coldbox]', default='coldbox')
parser.add_argument('-period', dest='period', help='which detector period is looked at [default is 1]', default='1')
args = parser.parse_args()



if(args.elec == 'top' or args.elec == 'tde'):
    elec = 'top'
elif(args.elec == 'bot' or args.elec == 'bde'):
    elec = 'bot'
else:
    print('electronic ', args.elec, ' is not recognized !')
    print('please mention the electronics with -elec argument : tde, top, bde or bot')
    sys.exit()


run = args.run
sub = args.sub
nevent = args.nevent
detector = args.detector
period = args.period

det.configure(detector, period, elec, run)

print("Welcome to LARDON !")

import config as cf
import data_containers as dc
import read_raw_file as read
import channel_mapping as cmap
import plotting as plot
import pedestals as ped


plot.set_style()

print(" will use ", cf.channel_map)

cmap.get_mapping(elec)

print('channel map has ', len(dc.chmap), ' elements')
reader = read.top_decoder(run, sub) if elec == "top" else read.bot_decoder(run, sub)
reader.open_file()
nb_evt = reader.read_run_header()

print(" --->> Will process ", nevent, " events [ out of ", nb_evt, "] of run ", run)


for ievent in range(nevent):
    dc.reset_event()
    
    print("-*-*-*-*-*-*-*-*-*-*-")
    print(" READING EVENT ", ievent)
    print("-*-*-*-*-*-*-*-*-*-*-")

    reader.read_evt_header(ievent)
    dc.evt_list[-1].dump()
    reader.read_evt(ievent)
    
    ped.compute_pedestal_raw()

    cmap.arange_in_view_channels()

    #plot.event_display_per_view()
    #plot.event_display_per_daqch()
    #plot.plot_raw_noise_daqch()
    plot.plot_raw_noise_view()

reader.close_file()
