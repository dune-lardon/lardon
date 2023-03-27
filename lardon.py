import sys
import argparse
import numpy as np
from datetime import datetime
import det_spec as det
import tables as tab
import time as time

print("\nWelcome to LARDON !\n")

tstart = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-elec', help='Which electronics are used [tde, top, bde, bot]',default="top", choices=["bot", "bde", "top", "tde"])#, required=True
parser.add_argument('-run', help='Run number to be processed', required=True)
parser.add_argument('-sub', help='Subfile to read', type=str, required=True)
parser.add_argument('-n', '--nevent', type=int, help='number of events to process in the file [default (or -1) is all]', default=-1)
parser.add_argument('-det', dest='detector', help='which detector is looked at [default is coldbox]', default='cb', choices=['cb1','dp','cb2', 'cb'])
parser.add_argument('-out', dest='outname', help='extra name on the output', default='')
parser.add_argument('-skip', dest='evt_skip', type=int, help='nb of events to skip', default=0)
parser.add_argument('-f', '--file', help="Override derived filename")
parser.add_argument('-pulse', dest='is_pulse', action='store_true', help='Used for pulsing data')

args = parser.parse_args()

if(args.elec == 'top' or args.elec == 'tde'):
    elec = 'top'
elif(args.elec == 'bot' or args.elec == 'bde'):
    elec = 'bot'

print('Looking at ',args.detector, ' data with ', args.elec, ' electronics')

if(args.detector == 'dp' and args.elec == 'bot'):
    print(' ... this is not possible!')
    sys.exit()


run = args.run
sub = args.sub
nevent = args.nevent
detector = args.detector
if(detector == 'cb2'):
    detector = 'cb'

outname_option = args.outname
evt_skip = args.evt_skip
det.configure(detector, elec, run)

is_pulse = args.is_pulse


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
import reconstruction_parameters as params
import track_3d as trk3d
import single_hits as sh
import ghost as ghost


plot.set_style()


""" output file """
if(outname_option):
    outname_option = "_"+outname_option
else:
    outname_option = ""
name_out = f"{cf.store_path}/{elec}_{run}_{sub}{outname_option}.h5"
output = tab.open_file(name_out, mode="w", title="Reconstruction Output")


if(is_pulse):
    print('This is PULSING data reconstruction. Pulses will be found and fitted')
    print('WARNING : IT TAKES A LOT OF TIME')
    import pulse_waveforms as pulse
    store.create_tables_pulsing(output)
else:
    store.create_tables(output)


""" set analysis parameters """
params.build_default_reco()
params.configure(detector, elec)
params.dump()


""" set the channel mapping """
cmap.get_mapping(detector, elec)
cmap.set_unused_channels()



""" setup the decoder """
if(detector=="dp"):
    reader = read.dp_decoder(run, str(sub), args.file)
else:
    reader = (read.top_decoder if elec == "top" else read.bot_decoder)(run, str(sub), args.file, detector)

reader.open_file()
nb_evt = reader.read_run_header()

if(nevent > nb_evt or nevent < 0):
    print(f"WARNING: Requested {nevent} events from a file containing only {nb_evt} events.")
    nevent = nb_evt

print(f" --->> Will process {nevent - evt_skip} events [out of {nb_evt}] of run {run}")

""" store basic informations """
store.store_run_infos(output, int(run), str(sub), elec, nevent, time.time())
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


    reader.read_evt(ievent)

    if(cf.n_sample == 0):
        ''' some self-trigger event have no samples? '''
        store.store_event(output)
        cf.n_sample = 999 #will be changed at the next event
        print(' EVENT HAS NO SAMPLE ... Skipping')
        continue

    """ mask the unused channels """
    dc.mask_daq = dc.alive_chan

    """ compute the raw pedestal """
    """ produces a rough mask estimate """
    ped.compute_pedestal(noise_type='raw')

    """ update the pedestal """
    ped.compute_pedestal(noise_type='filt')

    #plot.event_display_per_view_noise([-100,100],[-50, 300],option='noise_raw', to_be_shown=True)

    #plot.plot_correlation_globch(option="raw", to_be_shown=True)
    #plot.plot_noise_daqch(noise_type='raw', vrange=[0,10],to_be_shown=True)
    #plot.plot_noise_vch(noise_type='raw', vrange=[0,20],to_be_shown=True)

    if(is_pulse==True):

        """ pulse analysis does not need noise filtering """
        pulse.find_pulses()
    
        store.store_event(output)
        store.store_pedestals(output)
        store.store_pulse(output)
        #store.store_avf_wvf(output)
        continue



    """ low pass FFT cut """
    tf = time.time()
    ps = noise.FFT_low_pass()



    """ WARNING : DO NOT STORE ALL FFT PS !! """
    #store.store_fft(output, ps)
    #plot.plot_FFT_vch(ps, to_be_shown=True)


    if(dc.data_daq.shape[-1] != cf.n_sample):
        """ when the nb of sample is odd, the FFT returns an even nb of sample. Need to append an extra value (0) at the end of each waveform to make it work """
        #SHOULD MAKE SURE THIS BUG-FIX IS FINE
        dc.data_daq = np.insert(dc.data_daq, dc.data_daq.shape[-1], 0, axis=-1)



    for n_iter in range(2):
        ped.compute_pedestal(noise_type='filt')
        ped.refine_mask(n_pass=2, test=True)


    #plot.event_display_per_view(adc_ind=[-100,100],adc_coll=[-10, 300], option='raw', to_be_shown=True)
    #plot.plot_noise_vch(noise_type='filt', vrange=[0,20],to_be_shown=True)

    
    #to study the microphonic noise, can be removed
    ped.study_noise()

    """ CNR """
    noise.coherent_noise()


    #plot.event_display_per_view_noise([-100,100],[-50, 300],option='noise_cnr', to_be_shown=True)



    """ microphonic noise """
    noise.median_filter()


    ped.compute_pedestal(noise_type='filt')
    ped.refine_mask(n_pass=2, test=True)
    ped.compute_pedestal(noise_type='filt')


    #plot.event_display_per_view(adc_ind=[-30,30],adc_coll=[-10, 100], option='filt', to_be_shown=True)
    #plot.event_display_per_view(adc_ind=[-100,100],adc_coll=[-10, 300], option='filt', to_be_shown=True)
    #plot.plot_noise_vch(noise_type='filt', vrange=[0,20],to_be_shown=True)


    th = time.time()
    hf.find_hits()

    print("--- Number Of Hits found : ", dc.evt_list[-1].n_hits)



    #plot.event_display_per_view_noise([-40,40],[-50, 100],option='noise_cnr', to_be_shown=True)
    #plot.event_display_per_view_hits_found([-100,100],[-50, 300],option='hits', to_be_shown=True)
    
    #plot.plot_2dview_hits(to_be_shown=True)


    

    

    trk2d.find_tracks_rtree()
    """
    pars.trk2D_nhits,
    pars.trk2D_rcut,
    pars.trk2D_chi2cut,
    pars.trk2D_yerr,
    pars.trk2D_slope_err,
    pars.trk2D_pbeta)
    """


    #[t.mini_dump() for t in dc.tracks2D_list]

    #plot.plot_2dview_2dtracks(to_be_shown=True)


    ghost.ghost_finder(threshold=10)

    #print(len(dc.ghost_list), " POTENTIAL GHOST FOUND")

    trk3d.find_tracks_rtree()
    """
    pars.trk3D_ztol,
    pars.trk3D_qfrac,
    pars.trk3D_len_min,
    pars.trk3D_dx_tol, 
    pars.trk3D_dy_tol,
    pars.trk3D_dz_tol)
    """

    #[t.dump() for t in dc.tracks3D_list]
    ghost.ghost_trajectory()
    #plot.plot_3d(to_be_shown=True)



    sh.single_hit_finder()


    print("------- Number of 3D tracks found : ", len(dc.tracks3D_list))
    print('------ Found ', len(dc.single_hits_list), ' Single Hits!')
    print('----- Found ', len(dc.ghost_list), ' Ghosts!')
    print('  %.2f s to process '%(time.time()-t0))
    #[x.dump() for x in dc.single_hits_list]


    #plot.plot_2dview_hits_2dtracks(to_be_shown=True)
    #plot.plot_2dview_hits_3dtracks(to_be_shown=True)
    


    store.store_event(output)
    store.store_pedestals(output)
    store.store_noisestudy(output)
    store.store_hits(output)
    store.store_tracks2D(output)
    store.store_tracks3D(output)
    store.store_single_hits(output)
    store.store_ghost(output)
    
    dc.n_tot_hits += sum(dc.evt_list[-1].n_hits)

if(is_pulse==True):
    store.store_avf_wvf(output)

reader.close_file()
output.close()
print('it took %.2f s to run'%(time.time()-tstart))
