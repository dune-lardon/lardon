import sys
import argparse
import numpy as np
import tables as tab

from psutil import Process
import time as time
from datetime import datetime

import lardon.det_spec as det


def main():
    print("\nWelcome to LARDON !\n")

    tstart = time.time()

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-run', help='Run number to be processed', default=-1)#, required=True)
    parser.add_argument('-sub', help='Subfile to read', type=str, default=None)#, required=True)
    parser.add_argument('-n', '--nevent', type=int, help='number of events to process in the file [default (or -1) is all]', default=-1)

    parser.add_argument('-det', dest='detector', help='which detector is looked at [default is coldbox]', default='none', choices=['cb1top', 'cb1bot','dp', 'cbtop', 'cbbot', '50l', 'pdhd', 'pdvd'])

    parser.add_argument('-out', dest='outname', help='extra name on the output', default='')

    parser.add_argument('-skip', dest='evt_skip', type=int, help='nb of events to skip', default=0)
    parser.add_argument('-event', dest='single_event', type=int, help='Look at a specific event in the file', default=-1)
    
    parser.add_argument('-file', type=str, dest="custom_file_path", help="provide the raw file entire path", default=None)

    parser.add_argument('-pulse', dest='is_pulse', action='store_true', help='Used for charge pulsing data')

    parser.add_argument('-flow', type=str, default="-1", help="dataflow number", dest='dataflow')
    parser.add_argument('-writer', type=str, default="-1", help="datawriter number", dest='datawriter')
    parser.add_argument('-serv', type=str, default="-1", help="daq server nb", dest='daqserver')

    parser.add_argument('-job', dest='is_job', action='store_true', help='Flag that lardon is running on a job')

    parser.add_argument('-pds', dest='do_pds', action='store_true', help='Flag that lardon is reconstructing the PDS data')

    parser.add_argument('-trk', dest='do_charge', action='store_true', help='Flag that lardon is reconstructing the charge data')

    parser.add_argument('-hash', dest='hash_path', type=str, default='xx/xx', help='data hashed directories')

    parser.add_argument('-online', dest='online_mode', action='store_true', help='LARDONline mode')

    parser.add_argument('-gallery', type=str, dest='gallery', default='None', choices=['beam', 'top', 'bottom', 'both'], help='plot data in a nice way [warning: for PDVD only]')
    
    args = parser.parse_args()

    online_mode = args.online_mode
    if(online_mode):
        print('running in lardonline mode!\n')



    run = args.run
    sub = args.sub
    nevent = args.nevent
    detector = args.detector

    outname_option = args.outname
    evt_skip = args.evt_skip
    do_pds = args.do_pds
    do_charge = args.do_charge
    
    hash_path = args.hash_path

    
    """ high rate data-taking cases """
    dataflow = args.dataflow
    datawriter = args.datawriter
    daqserver = args.daqserver

    gallery = args.gallery
    is_gallery = False
    if(gallery != 'None'):
        do_charge = True
        is_gallery = True

    
    """ when file is in uncommon path (e.g. copied elsewhere) """
    custom_file_path = args.custom_file_path
    if(custom_file_path):
        print('---> Will read file ', custom_file_path)
        import lardon.utils.filenames as fname
        run, sub, dataflow, datawriter, daqserver, detector = fname.extract_info_form_file(custom_file_path)

        
    if(int(run) < 0 or detector == "none"):
        print('Please provide a run, subfile number and detector name to run LARDON')
        parser.print_help()
        exit()

    print('Looking at ', detector, ' data' )
        
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



    is_job = args.is_job


    import lardon.config as cf
    import lardon.data_containers as dc
    import lardon.decode_data as decoder
    import lardon.channel_mapping as cmap

    if(is_job == False):
        import lardon.plotting as plot
        plot.set_style()

    import lardon.reconstruction_parameters as params
    import lardon.workflow as work


    if(is_gallery):
        if(detector != "pdvd"):
            print("gallery plots only setup for PDVD data - sorry")
            exit()
        import lardon.gallery.pdvd as gal
        gal_eff = gal.effective_chan(gallery)


    """ Special case when data is written with multiple dataflow/datawriter app (e.g. high trigger rate, long window) """

    multipass_daqname = ""
    if(dataflow != "-1" or datawriter != "-1"):
        multipass_daqname = "_"

        if(dataflow !=  "-1"):
            multipass_daqname += dataflow
      
        if(datawriter != "-1"):
            multipass_daqname += datawriter
        if(daqserver != "-1"):
            multipass_daqname += "_s"+daqserver
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

    import lardon.store as store



    if(is_pulse):
        print('This is PULSING data reconstruction. Pulses will be found and fitted')
        print('WARNING : IT TAKES A LOT OF TIME')
        import lardon.pulse_waveforms as pulse
        dc.set_waveforms()
        store.create_tables_pulsing(output)

    
    else:
        store.create_tables(output)

    if(do_pds):
        store.create_tables_pds(output)
        cmap.get_pds_mapping(detector)




    """ set the channel mapping """
    cmap.get_mapping(detector)


    """ setup the decoder """
    reader = decoder.decoder(detector, run, str(sub), dataflow+"-"+datawriter, hash_path, custom_file_path)
    reader.open_file()
    nb_evt = reader.read_run_header()



    """ which events to read """
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
        dc.evt_list[-1].set_file_infos(dataflow, datawriter, daqserver)    
        
        
        ''' Workflow for PDS '''
        if(do_pds == True):       
            print('\n## Reading PDS ##')
            dc.reset_containers_pds()
            reader.read_pds_evt(ievent)

            if(cf.n_pds_stream_sample <=0 and cf.n_pds_trig_sample <= 0):
                print(' EVENT HAS NO PDS SAMPLE ...')
                cf.n_pds_stream_sample = 0
                cf.n_pds_trig_sample = 0
                store.store_pds_event(output)

            print('PDS streaming timestamp: ', dc.evt_list[-1].pds_stream_time)
            print('PDS self-trigger timestamp: ', dc.evt_list[-1].pds_trig_time)
            print('PDS nb of sample: stream mode', cf.n_pds_stream_sample, 'trigger mode', cf.n_pds_trig_sample)
            work.pds_signal_proc()
            work.pds_reco()            
                       

            #fft_ps = []
            #corr = []
    
        """ Workflow for charge """
        if(do_charge == True):
            print('\n## Reading TPC ##')
            for imodule in cf.module_used:

                cf.imod = imodule
                dc.reset_containers_trk()

                print('\nMODULE ', cf.imod)            
            
                mod_time = time.time()
                t1 = time.time()

                if(is_gallery):
                    if(gal_eff.read_module()):
                        reader.read_evt(ievent)                       
                        work.charge_signal_proc(deb, online_mode)
                        gal_eff.add_data()
                        gal_eff.draw()
                        continue
                    else:
                        continue
                    
                
                reader.read_evt(ievent)
                deb.read_data[cf.imod] = time.time()-t1
            
                t1 = time.time()
                if(cf.n_sample[cf.imod] <= 0):
                    print(' EVENT HAS NO CHARGE SAMPLE ...')
                    cf.n_sample[cf.imod] = 0 #will be changed at the next event
                    #store.store_event(output)

                print('1st sample timestamp: ', dc.evt_list[-1].charge_time[cf.imod])
                if(is_pulse==True):
                    work.charge_pulsing()
                    continue
            
                work.charge_signal_proc(deb, online_mode)

                #ps, cnr_corr = work.charge_signal_proc(deb, online_mode)
                #fft_ps.append(ps)
                #corr.append(cnr_corr)

            
                work.charge_reco(deb, online_mode)
                
                """ debugging tools """
                curr_mem = Process().memory_info().rss
                deb.memory_mod[cf.imod] = curr_mem
                deb.time_mod[cf.imod] = time.time()-mod_time        

            work.charge_reco_whole(online_mode)

        
        if(do_charge and do_pds):
            work.match_charge_and_pds()

                

        t1 = time.time()

        
        if(is_gallery):
            continue
        
        if(do_charge):
            store.store_event(output)
        """ store the results """
        if(do_charge and any(x>0 for x in cf.n_sample)):
            store.store_pedestals(output)
            store.store_noisestudy(output)
            store.store_hits(output)
            #store.store_fft(output, fft_ps)
            #store.store_corr(output, corr)

        
            if(is_pulse==True):                      
                store.store_event(output)
                store.store_pedestals(output)
                store.store_pulse(output)
                #store.store_avf_wvf(output)

            else:        
                store.store_tracks2D(output)
                store.store_tracks3D(output)
                store.store_single_hits(output)
                store.store_ghost(output)

        if(do_pds):
            store.store_pds_event(output)
        
        if(do_pds and (cf.n_pds_stream_sample > 0 or cf.n_pds_trig_sample > 0)):
            #store.store_pds_event(output)
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

if __name__ == "__main__":
    main()
