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
#parser.add_argument('-elec', help='Which electronics are used [tde, top, bde, bot]',default="top", choices=["bot", "bde", "top", "tde"])#, required=True

parser.add_argument('-run', help='Run number to be processed', required=True)
parser.add_argument('-sub', help='Subfile to read', type=str, required=True)
parser.add_argument('-n', '--nevent', type=int, help='number of events to process in the file [default (or -1) is all]', default=-1)

parser.add_argument('-det', dest='detector', help='which detector is looked at [default is coldbox]', default='cb', choices=['cb1top', 'cb1bot','dp', 'cbtop', 'cbbot', '50l'])
parser.add_argument('-out', dest='outname', help='extra name on the output', default='')

parser.add_argument('-skip', dest='evt_skip', type=int, help='nb of events to skip', default=0)
parser.add_argument('-f', '--file', help="Custom input filename")

parser.add_argument('-pulse', dest='is_pulse', action='store_true', help='Used for pulsing data')

parser.add_argument('-flow', type=str, default="-1", help="dataflow number (bde-only)", dest='dataflow')
parser.add_argument('-writer', type=str, default="-1", help="datawriter number (bde-only)", dest='datawriter')
parser.add_argument('-job', dest='is_job', action='store_true', help='Flag that lardon is running on a job')

args = parser.parse_args()

print('Looking at ',args.detector, ' data' )

run = args.run
sub = args.sub
nevent = args.nevent
detector = args.detector

outname_option = args.outname
evt_skip = args.evt_skip
det.configure(detector, run)

is_pulse = args.is_pulse

""" some bde special case """
dataflow = args.dataflow
datawriter = args.datawriter


is_job = args.is_job

import config as cf
import data_containers as dc
import decode_data as decoder
import channel_mapping as cmap

if(is_job == False):
    import plotting as plot
    plot.set_style()

import reconstruction_parameters as params
import pedestals as ped
import noise_filter as noise
import store as store
import hit_finder as hf
import track_2d as trk2d
import track_3d as trk3d
import single_hits as sh
import ghost as ghost




""" Special case when BDE-DAQ has to use multiple dataflow/datawriter app to write the data (e.g. high trigger rate, long window) """

multipass_daqname = ""
if(dataflow != "-1" or datawriter != "-1"):
    multipass_daqname = "_"

    if(dataflow !=  "-1"):
      multipass_daqname += dataflow
      
    if(datawriter != "-1"):
        multipass_daqname += datawriter
        
else:
    if(dataflow ==  "-1"):
        dataflow = "0"
    if(datawriter == "-1"):  
        datawriter = "0"


""" output file """
if(outname_option):
    outname_option = "_"+outname_option
else:
    outname_option = ""



if(is_job == False):
    name_out = f"{cf.store_path}/{detector}_{run}_{sub}{multipass_daqname}{outname_option}.h5"
else:
    name_out = f"{detector}_{run}_{sub}{multipass_daqname}{outname_option}.h5"

print('Output name is : ', name_out)
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
params.configure(detector)
#params.dump()


""" set the channel mapping """
cmap.get_mapping(detector)
cmap.set_unused_channels()



""" setup the decoder """
reader = decoder.decoder(detector, run, str(sub), dataflow+"-"+datawriter, args.file)
reader.open_file()
nb_evt = reader.read_run_header()



if(nevent > nb_evt):# or nevent < 0):
    print(f"WARNING: Requested {nevent} events from a file containing only {nb_evt} events.")
    nevent = nb_evt

if( nevent < 0):
    nevent = nb_evt
    if( evt_skip == 0):
        print(f" --->> Will process all {nb_evt} events of run {run}")
    else:
        if(evt_skip >= nevent):
            print("Too many skipped events asked ... bye!")
            reader.close_file()
            output.close()
            exit()
else:
    print(f" --->> Will process {nevent - evt_skip} events [out of {nb_evt}] of run {run}")



""" store basic informations """
store.store_run_infos(output, int(run), str(sub), nevent, time.time())
store.store_chan_map(output)
store.save_reco_param(output)

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


    if(cf.n_sample <= 0):
        ''' some self-trigger event have no samples? '''
        store.store_event(output)
        cf.n_sample = 0 #will be changed at the next event
        print(' EVENT HAS NO SAMPLE ... Skipping')
        continue

    """ mask the unused channels """
    dc.mask_daq = dc.alive_chan


    """ compute the raw pedestal """
    """ produces a rough mask estimate """
    ped.compute_pedestal(noise_type='raw')

    """ update the pedestal """
    ped.compute_pedestal(noise_type='filt')

    #plot.event_display_per_view([-100,100],[-10, 150],option='raw', to_be_shown=True) 

    #plot.event_display_per_view_noise([-100,100],[-50, 150],option='noise_raw', to_be_shown=True)

    if(is_pulse==True):

        """ pulse analysis does not need noise filtering """
        pulse.find_pulses()
    
        store.store_event(output)
        store.store_pedestals(output)
        store.store_pulse(output)
        #store.store_avf_wvf(output)
        continue

    """ low pass FFT cut """
    ps = noise.FFT_low_pass(False)
    



    """ WARNING : DO NOT STORE ALL FFT PS !! """
    #store.store_fft(output, ps)
    #plot.plot_FFT_vch(ps, to_be_shown=True)
    
    
    if(dc.data_daq.shape[-1] != cf.n_sample):
        """ when the nb of sample is odd, the FFT returns an even nb of sample. Need to append an extra value (0) at the end of each waveform to make it work """

        dc.data_daq = np.insert(dc.data_daq, dc.data_daq.shape[-1], 0, axis=-1)


    for n_iter in range(2):
        ped.compute_pedestal(noise_type='filt')
        ped.refine_mask(n_pass=2)


    """ special microphonic noise study """
    ped.study_noise()

    

    """ CNR """
    noise.coherent_noise()
    """
    aft_corr_glob += plot.plot_correlation_globch(option='afterCNR',to_be_shown=False)
    aft_corr_daq += plot.plot_correlation_daqch(option='afterCNR', to_be_shown=False)
    """
    """ microphonic noise """
    noise.median_filter()


    ped.compute_pedestal(noise_type='filt')
    ped.refine_mask(n_pass=2)
    ped.compute_pedestal(noise_type='filt')

    
    #plot.event_display_per_view_noise([-100,100],[-50, 150],option='noise_filt', to_be_shown=True)
    #plot.event_display_per_view([-100,100],[-10, 150],option='filt', to_be_shown=True) 
    hf.find_hits()

    print("----- Number Of Hits found : ", dc.evt_list[-1].n_hits)


    #plot.event_display_per_view_noise([-40,40],[-50, 100],option='noise_cnr', to_be_shown=True)
    #plot.event_display_per_view_hits_found([-100,100],[-10, 150],option='hits', to_be_shown=True)    

    #plot.plot_2dview_hits(to_be_shown=True)

    

    trk2d.find_tracks_rtree()


    print("---- Number Of 2D tracks found : ", dc.evt_list[-1].n_tracks2D)




    ghost.ghost_finder(threshold=10)



    trk3d.find_tracks_rtree()
    #[t.dump() for t in dc.tracks3D_list]

    ghost.ghost_trajectory()

    

    
    sh.single_hit_finder()
    

    """
    if(len(dc.tracks3D_list) > 0):
        [t.dump() for t in dc.tracks3D_list]
        #plot.plot_2dview_hits_3dtracks(to_be_shown=True)
        #plot.event_display_per_view_hits_found([-400,400],[-10, 600],option='hits', to_be_shown=True)            

        #plot.plot_3d(to_be_shown=True)
    """

    print("--- Number of 3D tracks found : ", len(dc.tracks3D_list))
    print('-- Found ', len(dc.single_hits_list), ' Single Hits!')
    print('- Found ', len(dc.ghost_list), ' Ghosts!')
    print('%.2f s to process '%(time.time()-t0))



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
