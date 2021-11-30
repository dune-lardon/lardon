import sys
import argparse
import numpy as np
from datetime import datetime
import det_spec as det
import tables as tab
import time as time

tstart = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-elec', help='Which electronics are used [tde, top, bde, bot]', required=True, choices=["bot", "bde", "top", "tde"])
parser.add_argument('-run', help='Run number to be processed', required=True)
parser.add_argument('-sub', help='Subfile to read', type=int, required=True)
parser.add_argument('-n', '--nevent', type=int, help='number of events to process in the file [default (or -1) is all]', default=-1)
parser.add_argument('-det', dest='detector', help='which detector is looked at [default is coldbox]', default='coldbox')
parser.add_argument('-period', help='which detector period is looked at [default is 1]', default='1')
parser.add_argument('-out', dest='outname', help='extra name on the output', default='')
parser.add_argument('-skip', dest='evt_skip', type=int, help='nb of events to skip', default=-1)
parser.add_argument('-f', '--file', help="Override derived filename")
args = parser.parse_args()

if(args.elec == 'top' or args.elec == 'tde'):
    elec = 'top'
elif(args.elec == 'bot' or args.elec == 'bde'):
    elec = 'bot'

run = args.run
sub = args.sub
nevent = args.nevent
detector = args.detector
period = args.period
outname_option = args.outname
evt_skip = args.evt_skip
det.configure(detector, period, elec, run)

print("Welcome to LARDON !")

import config as cf
import data_containers as dc
import read_raw_file as read
import channel_mapping as cmap
import plotting as plot
import pedestals as ped
import noise_filter as noise
import store as store
import hit_finder as hf
import track_2d as trk2d



plot.set_style()


""" output file """
if(outname_option):
    outname_option = "_"+outname_option
else:
    outname_option = ""
name_out = f"{cf.store_path}/{elec}_{run}_{sub}{outname_option}.h5"
output = tab.open_file(name_out, mode="w", title="Reconstruction Output")
store.create_tables(output)

""" set the channel mapping """
print(" will use ", cf.channel_map)
cmap.get_mapping(elec)
cmap.set_unused_channels()



""" mask the unused channels """
dc.mask_daq = ~dc.alive_chan


""" setup the decoder """
reader = (read.top_decoder if elec == "top" else read.bot_decoder)(run, str(sub), args.file)
reader.open_file()
nb_evt = reader.read_run_header()

if(nevent > nb_evt or nevent < 0):
    nevent = nb_evt

print(" --->> Will process ", nevent, " events [ out of ", nb_evt, "] of run ", run)

""" store basic informations """
store.store_run_infos(output, int(run), int(sub), elec, nevent, time.time())
store.store_chan_map(output)



for ievent in range(nevent):
    t0 = time.time()
    if(evt_skip > 0 and ievent < evt_skip):
        continue
    dc.reset_event()
    
    print("-*-*-*-*-*-*-*-*-*-*-")
    print(" READING EVENT ", ievent)
    print("-*-*-*-*-*-*-*-*-*-*-")

    reader.read_evt_header(ievent)
    dc.evt_list[-1].dump()
    store.store_event(output)

    reader.read_evt(ievent)
    if(elec == 'top'):
        dc.data_daq *= -1

    #ped.set_mask_wf_rms_all()
    ped.compute_pedestal(noise_type='raw')

    
    vmax = 900 if elec == 'bot' else 15

    #plot.plot_noise_daqch(noise_type='raw',vmin=0,vmax=vmax,to_be_shown=False)
    #plot.plot_noise_vch(noise_type='raw', vmin=0, vmax=vmax,to_be_shown=False)
    #plot.plot_noise_globch(noise_type='raw', vmin=0, vmax=vmax,to_be_shown=False)
    #plot.event_display_per_daqch(-1000,1000,option='raw',to_be_shown=False)
    #cmap.arange_in_view_channels()
    #plot.event_display_per_view(-1000,1000,-500,500,option='raw', to_be_shown=False)


    fft_low_cut = 0.6 if elec=='top' else 0.4
    fft_freq = -1 if elec=='top' else 0.0225
    tf = time.time()
    ps = noise.FFT_low_pass(fft_low_cut, fft_freq)

    """ DO NOT STORE ALL FFT PS !! """
    #store.store_fft(output, ps)



    #ped.set_mask_wf_rms_all()
    tp = time.time()
    for i in range(2):
        ped.compute_pedestal(noise_type='filt')
        ped.update_mask(4.)
    #plot.plot_noise_daqch(noise_type='filt',option='fft', vmin=0, vmax=vmax)
    #plot.plot_noise_vch(noise_type='filt', vmin=0, vmax=vmax,option='fft')#,to_be_shown=True)

    
    #plot.plot_FFT_daqch(ps,option='raw',to_be_shown=False)    
    #plot.plot_FFT_vch(ps,option='raw',to_be_shown=False)    

    #cmap.arange_in_view_channels()
    #plot.event_display_per_view(-100,100,-50,50,option='fft', to_be_shown=False)

    #plot.plot_correlation_daqch(option='fft',to_be_shown=True)
    #plot.plot_correlation_globch(option='fft', to_be_shown=False)


    noise_group = [32] if elec == 'top' else [32]

    tcoh = time.time()
    noise.coherent_noise(noise_group)
    print("coherent noise : ", time.time()-tcoh)


    #ped.set_mask_wf_rms_all()
    tpm = time.time()
    for i in range(2):
        ped.compute_pedestal(noise_type='filt')
        ped.update_mask(4.)


    #plot.plot_noise_daqch(noise_type='filt',option='coherent', vmin=0, vmax=vmax)
    #plot.plot_noise_vch(noise_type='filt', vmin=0, vmax=vmax,option='coherent',to_be_shown=False)


    #plot.event_display_per_daqch(-1000,1000,option='coherent',to_be_shown=False)
    #cmap.arange_in_view_channels()
    #plot.event_display_per_view(-100, 100,-50,50,option='coherent', to_be_shown=False)
    #plot.plot_noise_daqch(noise_type='filt',option='coherent', vmin=0, vmax=vmax)
    #plot.plot_noise_vch(noise_type='filt', vmin=0, vmax=vmax,option='coherent',to_be_shown=False)


    store.store_pedestals(output)
    print('  %.2f s to process '%(time.time()-t0))


    
    th = time.time()
    hf.find_hits(6, 10, 10, 3., 6., 2.)
    print("hit %.2f s"%(time.time()-th))
    print(dc.evt_list[-1].n_hits)
    plot.plot_2dview_hits(to_be_shown=False)

    """parameters : min nb hits, rcut, chi2cut, y error, slope error, pbeta"""

    trk2d.find_tracks_rtree(5, 6., 8., 0.5, 1., 3.)

    [t.mini_dump() for t in dc.tracks2D_list]
    plot.plot_2dview_2dtracks(to_be_shown=True)


reader.close_file()
output.close()
print('it took %.2f s to run'%(time.time()-tstart))
