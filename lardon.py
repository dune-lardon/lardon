import sys
import argparse
import numpy as np
from datetime import datetime
import det_spec as det
import tables as tab
import time as time

from psutil import Process



print("\nWelcome to LARDON !\n")

tstart = time.time()

parser = argparse.ArgumentParser()

parser.add_argument('-run', help='Run number to be processed', required=True)
parser.add_argument('-sub', help='Subfile to read', type=str, required=True)
parser.add_argument('-n', '--nevent', type=int, help='number of events to process in the file [default (or -1) is all]', default=-1)

parser.add_argument('-det', dest='detector', help='which detector is looked at [default is coldbox]', default='cb', choices=['cb1top', 'cb1bot','dp', 'cbtop', 'cbbot', '50l', 'pdhd', 'pdvd'])
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

parser.add_argument('-hash', dest='hash_path', type=str, default='xx/xx', help='data hashed directories')

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

hash_path = args.hash_path
#print('hash: ', hash_path)

if(do_pds == False and do_charge == False):
    print('Nothing asked to be reconstructed, please set -trk and/or -pds when you call LARDON')
    parser.print_help()
    exit()
else:
    print('Reconstructing', 'charge' if do_charge==True else '', 'light' if do_pds==True else '', 'data')
    


det.configure(detector, run, do_pds, hash_path)

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


import workflow as work


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



""" set analysis parameters """
params.build_default_reco()
params.configure(detector)
#params.dump()

    
print('Output file is : ', name_out)
output = tab.open_file(name_out, mode="w", title="Reconstruction Output")

import store as store



if(is_pulse):
    print('This is PULSING data reconstruction. Pulses will be found and fitted')
    print('WARNING : IT TAKES A LOT OF TIME')
    import pulse_waveforms as pulse
    dc.set_waveforms()
    store.create_tables_pulsing(output)

#elif(detector == 'pdvd'):
#    store.create_tables_commissioning(output)
#    print('THIS IS TEMPORARY SIMPLE RECO CHANGE WHEN CATHODE IS ON')
    
else:
    store.create_tables(output)

if(do_pds):
    store.create_tables_pds(output)
    cmap.get_pds_mapping(detector)



    


""" set the channel mapping """

cmap.get_mapping(detector)
#cmap.set_unused_channels()




""" setup the decoder """
reader = decoder.decoder(detector, run, str(sub), dataflow+"-"+datawriter, hash_path, args.file)
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

'''
if(do_charge):
    store.store_chan_map(output)
'''

if(do_pds):
    store.store_pds_infos(output, int(run), str(sub), nevent, time.time())


deb = dc.debug()
store.create_table_debug(output)

for ievent in range(nevent):

    t0 = time.time()
    ini_mem = Process().memory_info().rss

    
    if(evt_skip > 0 and ievent < evt_skip):
        continue

    dc.reset_evt()

    print("-*-*-*-*-*-*-*-*-*-*-")
    print(" READING EVENT ", ievent)
    print("-*-*-*-*-*-*-*-*-*-*-")
    
    reader.read_evt_header(ievent)
    dc.evt_list[-1].dump()
    

    ''' Workflow for PDS '''
    if(do_pds == True):       
        dc.reset_event_pds()
        reader.read_containers_evt(ievent)

        if(cf.n_pds_sample <=0):
            print(' EVENT HAS NO PDS SAMPLE ...')
            cf.n_pds_sample = 0
            store.store_pds_event(output)
            

        work.pds_signal_proc()
        work.pds_reco()            
                       

    #fft_ps = []
    """ Workflow for charge """
    if(do_charge == True):

        for imodule in cf.module_used:
            
            cf.imod = imodule
            dc.reset_containers_trk()

            mod_time = time.time()
            t1 = time.time()

            reader.read_evt(ievent)
            deb.read_data[cf.imod] = time.time()-t1
            
            t1 = time.time()
            if(cf.n_sample[cf.imod] <= 0):
                print(' EVENT HAS NO CHARGE SAMPLE ...')
                cf.n_sample[cf.imod] = 0 #will be changed at the next event
                store.store_event(output)

            
            if(is_pulse==True):
                work.charge_pulsing()
                continue
            
            
            work.charge_signal_proc(deb)
            #fft_ps.append(ps)
            #dc.n_tot_hits  += np.sum(dc.evt_list[-1].n_hits[:,cf.imod])

            #if(detector == 'pdvd'):
            #    ''' temporary workflow for PDVD data '''
            #    work.charge_reco_pdvd(deb)                
            #else:

            work.charge_reco(deb)
                
            """ debugging tools """
            curr_mem = Process().memory_info().rss
            deb.memory_mod[cf.imod] = curr_mem
            deb.time_mod[cf.imod] = time.time()-mod_time
            
            #plot.plot_2dview_2dtracks([cf.imod], to_be_shown=True)

        #if(detector == 'pdhd'):
        work.charge_reco_whole()
    
        
    if(do_charge and do_pds):
        work.match_charge_and_pds()

                

    t1 = time.time()
    """ store the results """
    if(do_charge and cf.n_sample[cf.imod] > 0):
        store.store_event(output)
        store.store_pedestals(output)
        store.store_noisestudy(output)
        store.store_hits(output)
        #store.store_fft(output, fft_ps)

        
        if(is_pulse==True):                      
            store.store_event(output)
            store.store_pedestals(output)
            store.store_pulse(output)
            #store.store_avf_wvf(output)

        else:        
            #if(detector == 'pdvd'):
            #    store.store_hits_3d(output)
            #else:
            store.store_tracks2D(output)
            store.store_tracks3D(output)
            store.store_single_hits(output)
            store.store_ghost(output)
        
    if(do_pds and cf.n_pds_sample > 0):
        store.store_pds_event(output)
        store.store_pds_pedestals(output)
        store.store_pds_peak(output)

        if(do_charge and  cf.n_sample[cf.imod] > 0):
            store.store_pds_cluster(output)
    deb.output = time.time()-t1
        
    dc.n_tot_hits  += np.sum(dc.evt_list[-1].n_hits)
    dc.n_tot_pds_peaks += sum(dc.evt_list[-1].n_pds_peaks)    
    dc.n_tot_trk2d += sum(dc.evt_list[-1].n_tracks2D)
    dc.n_tot_trk3d += dc.evt_list[-1].n_tracks3D
    dc.n_tot_ghosts += dc.evt_list[-1].n_ghosts
    dc.n_tot_sh += dc.evt_list[-1].n_single_hits
    dc.n_tot_pds_clusters += dc.evt_list[-1].n_pds_clusters
    dc.n_tot_hits_clusters += dc.evt_list[-1].n_hits_clusters

    end_mem = Process().memory_info().rss
    deb.memory_tot = end_mem
    deb.time_tot = time.time()-t0

    #deb.dump()
    store.store_debug(output, deb)
    
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

print(f'it took {time.time()-tstart:.2f} s to run {nevent - evt_skip} events (average of {(time.time()-tstart)/(nevent - evt_skip):.2f} per event)')

