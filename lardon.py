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

parser.add_argument('-run', help='Run number to be processed', required=True)
parser.add_argument('-sub', help='Subfile to read', type=str, required=True)
parser.add_argument('-n', '--nevent', type=int, help='number of events to process in the file [default (or -1) is all]', default=-1)

parser.add_argument('-det', dest='detector', help='which detector is looked at [default is coldbox]', default='cb', choices=['cb1top', 'cb1bot','dp', 'cbtop', 'cbbot', '50l'])
parser.add_argument('-out', dest='outname', help='extra name on the output', default='')

parser.add_argument('-skip', dest='evt_skip', type=int, help='nb of events to skip', default=0)
parser.add_argument('-event', dest='single_event', type=int, help='Look at a specific event in the file', default=-1)

parser.add_argument('-f', '--file', help="Custom input filename")

parser.add_argument('-pulse', dest='is_pulse', action='store_true', help='Used for charge pulsing data')

parser.add_argument('-flow', type=str, default="-1", help="dataflow number (bde-only)", dest='dataflow')
parser.add_argument('-writer', type=str, default="-1", help="datawriter number (bde-only)", dest='datawriter')

parser.add_argument('-job', dest='is_job', action='store_true', help='Flag that lardon is running on a job, no need to import the plot libraries')

parser.add_argument('-pds', dest='do_pds', action='store_true', help='Flag that lardon is reconstructing the PDS data')

parser.add_argument('-trk', dest='do_charge', action='store_true', help='Flag that lardon is reconstructing the charge data')

args = parser.parse_args()

print('Looking at ',args.detector, ' data' )

run = args.run
sub = args.sub
nevent = args.nevent
detector = args.detector

outname_option = args.outname
evt_skip = args.evt_skip
do_pds = args.do_pds
do_charge = args.do_charge


if(do_pds == False and do_charge == False):
    print('Nothing asked to be reconstructed, please set -trk and/or -pds when you call LARDON')
    parser.print_help()
    exit()
else:
    print('Reconstructing', 'charge' if do_charge==True else '', 'light' if do_pds==True else '', 'data')
    


det.configure(detector, run, do_pds)

is_pulse = args.is_pulse
if(is_pulse == True):
    do_charge = True

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
import clusters as clu
import matching as mat



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

if(do_pds):
    store.create_tables_pds(output)
    cmap.get_pds_mapping(detector)



    
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

single_event = args.single_event
if(single_event >= 0):
    nevent = single_event+1
    evt_skip = single_event
    

if(nevent > nb_evt):
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
store.save_reco_param(output)

if(do_charge):
    store.store_chan_map(output)

if(do_pds):
    store.store_pds_infos(output, int(run), str(sub), nevent, time.time())
    
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


    """ Decode the charge data and equalize the pedestals"""
    if(do_charge):
        reader.read_evt(ievent)

        if(cf.n_sample <= 0):
            print(' EVENT HAS NO CHARGE SAMPLE ...')
            cf.n_sample = 0 #will be changed at the next event
            store.store_event(output)
            
        else:
            """ mask the unused channels """
            dc.mask_daq = dc.alive_chan
            
            """ compute the raw pedestal """
            """ produces a rough mask estimate """
            ped.compute_pedestal(noise_type='raw')
            
            """ update the pedestal """
            ped.compute_pedestal(noise_type='filt')
            

    ''' decode the light data and equalize the pedestals '''
    if(do_pds == True):       
        reader.read_pds_evt(ievent)

        if(cf.n_pds_sample <=0):
            print(' EVENT HAS NO PDS SAMPLE ...')
            cf.n_pds_sample = 0
            store.store_pds_event(output)
            
        else:
            """ compute the pedestal """
            ped.compute_pedestal_pds()


    
    if(do_charge and cf.n_sample > 0):
        #plot.event_display_per_view([-100,100],[-10, 150],option='raw', to_be_shown=True) 
        #plot.plot_wvf_current_vch([(0,750),(1,750),(2,750)], adc_min=-1, adc_max=-1, tmin=0, tmax=cf.n_sample, option='rawbnl', to_be_shown=True)
        
        if(is_pulse==True):
            """ pulse analysis does not need noise filtering """
            pulse.find_pulses()
    
            store.store_event(output)
            store.store_pedestals(output)
            store.store_pulse(output)
            #store.store_avf_wvf(output)
            continue

        ''' Perform the charge noise filtering '''
        
        """ low pass FFT cut """
        ps = noise.FFT_low_pass(False)
    
        """ WARNING : DO NOT STORE ALL FFT PS !! """
        #store.store_fft(output, ps)
        #plot.plot_FFT_vch(ps, to_be_shown=True)
    
        #plot.event_display_per_view([-100,100],[-10, 150],option='fftbnl', to_be_shown=True)
        #plot.plot_wvf_current_vch([(0,750),(1,750),(2,750)], adc_min=-1, adc_max=-1, tmin=0, tmax=cf.n_sample, option='fftbnl', to_be_shown=True)
        
        if(dc.data_daq.shape[-1] != cf.n_sample):
            """ 
            when the nb of sample is odd, the FFT returns 
            an even nb of sample. 
            Need to append an extra value (0) at the end 
            of each waveform to make it work """
            
            dc.data_daq = np.insert(dc.data_daq, dc.data_daq.shape[-1], 0, axis=-1)


        for n_iter in range(2):
            ped.compute_pedestal(noise_type='filt')
            ped.refine_mask(n_pass=2)

        
        """ special microphonic noise study """
        ped.study_noise()

    
        """ CNR """
        noise.coherent_noise()

        #plot.event_display_per_view([-100,100],[-10, 150],option='cnrbnl', to_be_shown=True)
        #plot.plot_wvf_current_vch([(0,750),(1,750),(2,750)], adc_min=-1, adc_max=-1, tmin=0, tmax=cf.n_sample, option='cnrbnl', to_be_shown=True)
        """ microphonic noise """
        noise.median_filter()

        """ finalize pedestal RMS and ROI """
        ped.compute_pedestal(noise_type='filt')
        ped.refine_mask(n_pass=2)
        ped.compute_pedestal(noise_type='filt')
        #plot.event_display_per_view([-100,100],[-10, 150],option='microbnl', to_be_shown=True) 
        #plot.plot_wvf_current_vch([(0,750),(1,750),(2,750)], adc_min=-1, adc_max=-1, tmin=0, tmax=cf.n_sample, option='microbnl', to_be_shown=True)
        
        #plot.event_display_per_view_noise([-100,100],[-50, 150],option='noise_filt', to_be_shown=True)
        #plot.event_display_per_view([-100,100],[-10, 150],option='filt', to_be_shown=True) 
        #plot.event_display_per_view_roi(adc_ind=[-100,100], adc_coll=[-10,150], option='fit_roibnl', to_be_shown=True)
    
        hf.find_hits()    
        print("----- Number Of Hits found : ", dc.evt_list[-1].n_hits)
        #plot.event_display_per_view([-100,100],[-10, 150],option='filt', to_be_shown=True) 
        #plot.event_display_per_view_hits_found([-100,100],[-10, 150],option='hitsbnl', to_be_shown=True)    

        #plot.plot_2dview_hits(to_be_shown=True)
        #plot.plot_wvf_current_hits_roi_vch([(2,834)], tmin=5335, tmax=5435, option='coll_double_2bnl', to_be_shown=True)
        #plot.plot_wvf_current_hits_roi_vch([(2,400)], tmin=6425, tmax=6550, option='coll_single_2bnl', to_be_shown=True)
        #plot.plot_wvf_current_hits_roi_vch([(0,757)], tmin=4050, tmax=4300, option='ind_double_2bnl', to_be_shown=True)
        #plot.plot_wvf_current_hits_roi_vch([(1,620)], tmin=1250, tmax=1500, option='ind_double_3bnl', to_be_shown=True)
        #plot.plot_wvf_current_hits_roi_vch([(0, 250)], tmin=1800, tmax=2150, option='ind_single_bnl', to_be_shown=True)
        #plot.plot_cumsum_wvf([(0,757)], tmin=4050, tmax=4300, option='ind_double_cumsum2bnl', to_be_shown=True)
        #plot.plot_cumsum_wvf([(0, 250)], tmin=1800, tmax=2150, option='ind_single_cumsumbnl', to_be_shown=True)
        #plot.plot_cumsum_wvf([(1,620)], tmin=1250, tmax=1500, option='ind_double_cumsum3bnl', to_be_shown=True)
        
        trk2d.find_tracks_rtree()
        print("---- Number Of 2D tracks found : ", dc.evt_list[-1].n_tracks2D)

        ghost.ghost_finder(threshold=10)
        trk3d.find_tracks_rtree()    
        ghost.ghost_trajectory()
        
        sh.single_hit_finder()

        '''
        for x in  dc.single_hits_list:
            x.dump()
            ids = x.IDs
            hits = [dc.hits_list[x-dc.n_tot_hits] for hh in ids for x in hh]
            daqchs = [h.daq_channel for h in hits]
            #charges = [h.charge_pos for h in hits]
            start, stop = min(x.start), max(x.stop)
            """
            for h in hits:
                print(h.ID, ':', h.view, h.channel, ' -> ', h.charge, h.charge_pos, h.max_adc, ' from ', h.start, ' to ', h.stop)
                print('integral from ', h.pad_start, ' to ', h.pad_stop)
                print('test=', np.sum(dc.data_daq[h.daq_channel,h.pad_start:h.pad_stop]))
            """
            #if(x.veto[0]==False and x.veto[1]==False and x.veto[2]==False):
                #plot.plot_wvf_current_daqch(daqchs, option='sh_'+str(x.ID_SH), to_be_shown=True, tmin=start, tmax=stop)
        plot.event_display_per_view_hits_found([-400,400],[-10, 600],option='hits', to_be_shown=True)            
        '''
        #if(len(dc.tracks3D_list) > 0):
            #[t.dump() for t in dc.tracks3D_list]
            #plot.plot_2dview_hits_3dtracks(to_be_shown=True)
            #plot.event_display_per_view_hits_found([-400,400],[-10, 600],option='hits', to_be_shown=True)            
            #plot.plot_3d(to_be_shown=True)
            #plot.charge_pds_zoom([0,1], (138, 152), (7580, 7700), option=None, to_be_shown=True)
            #plot.event_display_coll_pds(draw_trk_t0 = True, to_be_shown=True)
            

        print("--- Number of 3D tracks found : ", len(dc.tracks3D_list))
        print('-- Found ', len(dc.single_hits_list), ' Single Hits!')
        print('- Found ', len(dc.ghost_list), ' Ghosts!')
        print('%.2f s to process '%(time.time()-t0))

        
    ''' WORKFLOW FOR THE LIGHT RECONSTRUCTION '''
    if(do_pds and cf.n_pds_sample > 0):                
        #plot.event_display_coll_pds(draw_trk_t0 = True, to_be_shown=True)
        hf.find_pds_peak()
        #plot.draw_pds_ED(to_be_shown=True)#, draw_peak=False, draw_cluster=False, draw_roi=False)
        #[p.dump() for p in dc.pds_peak_list]
        
        clu.light_clustering()
        print('--> Found ', dc.evt_list[-1].n_pds_clusters, ' clusters ')
        #[c.dump() for c in dc.pds_cluster_list]
        #plot.draw_pds_ED(to_be_shown=True, draw_peak=True, draw_cluster=True, draw_roi=False)

    
        if(do_charge and  cf.n_sample > 0):
                mat.matching_charge_pds()                

                


    """ store the results """
    if(do_charge and cf.n_sample > 0):
        store.store_event(output)
        store.store_pedestals(output)
        store.store_noisestudy(output)
        store.store_hits(output)
        store.store_tracks2D(output)
        store.store_tracks3D(output)
        store.store_single_hits(output)
        store.store_ghost(output)

    if(do_pds and cf.n_pds_sample > 0):
        store.store_pds_event(output)
        store.store_pds_pedestals(output)
        store.store_pds_peak(output)

        if(do_charge and  cf.n_sample > 0):
            store.store_pds_cluster(output)

                
    dc.n_tot_hits  += sum(dc.evt_list[-1].n_hits)
    dc.n_tot_pds_peaks += sum(dc.evt_list[-1].n_pds_peaks)    
    dc.n_tot_trk2d += sum(dc.evt_list[-1].n_tracks2D)
    dc.n_tot_trk3d += dc.evt_list[-1].n_tracks3D
    dc.n_tot_ghosts += dc.evt_list[-1].n_ghosts
    dc.n_tot_sh += dc.evt_list[-1].n_single_hits
    dc.n_tot_pds_clusters += dc.evt_list[-1].n_pds_clusters

if(is_pulse==True):
    store.store_avf_wvf(output)

reader.close_file()
output.close()

print('**************')
print('* Reco total *')
print('**************')
print('* Nb of Hits', dc.n_tot_hits)
print('* Nb of 2D Tracks', dc.n_tot_trk2d)
print('* Nb of 3D Tracks', dc.n_tot_trk3d)
print('* Nb of Ghosts', dc.n_tot_ghosts)
print('* Nb of Single Hits', dc.n_tot_sh)
print('* Nb of PDS Peaks', dc.n_tot_pds_peaks)
print('* Nb of PDS Clusters', dc.n_tot_pds_clusters)
print('**************')
print('it took %.2f s to run'%(time.time()-tstart))
