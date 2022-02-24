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
parser.add_argument('-elec', help='Which electronics are used [tde, top, bde, bot]', required=True, choices=["bot", "bde", "top", "tde"])
parser.add_argument('-run', help='Run number to be processed', required=True)
parser.add_argument('-sub', help='Subfile to read', type=int, required=True)
parser.add_argument('-n', '--nevent', type=int, help='number of events to process in the file [default (or -1) is all]', default=-1)
parser.add_argument('-det', dest='detector', help='which detector is looked at [default is coldbox]', default='coldbox', choices=['coldbox',])
parser.add_argument('-period', help='which detector period is looked at [default is 1]', default='1')
parser.add_argument('-out', dest='outname', help='extra name on the output', default='')
parser.add_argument('-skip', dest='evt_skip', type=int, help='nb of events to skip', default=0)
parser.add_argument('-f', '--file', help="Override derived filename")
parser.add_argument('-conf','--config',dest='conf', help='Analysis configuration ID', default='1')
parser.add_argument('-pulse', dest='is_pulse', action='store_true', help='Used for pulsing data')

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
import analysis_parameters as params
import track_3d as trk3d



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
pars = params.params()
pars.read(config=args.conf,elec=elec)
pars.dump()

""" set the channel mapping """

cmap.get_mapping(elec)
cmap.set_unused_channels()



""" setup the decoder """
reader = (read.top_decoder if elec == "top" else read.bot_decoder)(run, str(sub), args.file)
reader.open_file()
nb_evt = reader.read_run_header()

if(nevent > nb_evt or nevent < 0):
    print(f"WARNING: Requested {nevent} events from a file containing only {nb_evt} events.")
    nevent = nb_evt

print(f" --->> Will process {nevent - evt_skip} events [out of {nb_evt}] of run {run}")

""" store basic informations """
store.store_run_infos(output, int(run), int(sub), elec, nevent, time.time())
store.store_chan_map(output)


print('checking : ', cf.e_drift)

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
    print('time to read %.3f'%(time.time()-t0))
    #plot.plot_sticky_finder_daqch(to_be_shown=True)


    """ mask the unused channels """
    dc.mask_daq = dc.alive_chan

    """ compute the raw pedestal """
    """ produces a rough mask estimate """
    ped.compute_pedestal(noise_type='raw', pars=pars)

    """ update the pedestal """
    ped.compute_pedestal(noise_type='filt')


    if(is_pulse==True):
        """ pulse analysis does not need noise filtering """

        pulse.find_pulses()
    
        store.store_event(output)
        store.store_pedestals(output)
        store.store_pulse(output)
        continue

    """ low pass FFT cut """
    tf = time.time()
    ps = noise.FFT_low_pass(pars.noise_fft_lcut,pars.noise_fft_freq)


    """ WARNING : DO NOT STORE ALL FFT PS !! """
    #store.store_fft(output, ps)
    #plot.plot_FFT_vch(ps, to_be_shown=True)


    for n_iter in range(2):
        ped.compute_pedestal(noise_type='filt')
        #ped.update_mask(pars.ped_amp_sig_oth)
        ped.refine_mask(pars, n_pass=1)



    plot.plot_noise_vch(noise_type='filt', vrange=[0,20],to_be_shown=True)
    plot.event_display_per_view_noise([-40,40],[-10, 150], option='noise', to_be_shown=True)


    #plot.plot_wvf_current_vch([(0,188),(1,466),(2,291)], to_be_shown=True)

    """ CNR """
    tcoh = time.time()
    if(pars.noise_coh_per_view):
        noise.coherent_noise_per_view(pars.noise_coh_group, pars.noise_coh_capa_weight, pars.noise_coh_calibrated)
    else :
        noise.coherent_noise(pars.noise_coh_group)

    print("coherent noise : ", time.time()-tcoh)


    ped.compute_pedestal(noise_type='filt')
    ped.refine_mask(pars, n_pass=2)
    #ped.update_mask(pars.ped_amp_sig_oth)
    ped.compute_pedestal(noise_type='filt')

    plot.plot_noise_vch(noise_type='filt', vrange=[0,20],to_be_shown=True)

    #plot.plot_wvf_current_vch([(0,188),(1,466),(2,291)], to_be_shown=True)
    #plot.plot_noise_vch(noise_type='filt', vrange=[0,100],option='coh_nocapa',to_be_shown=False)

    
    #plot.plot_correlation_daqch(option='filtered',to_be_shown=True)
    #plot.plot_correlation_globch(option='filtered', to_be_shown=True)

    plot.event_display_per_view_roi([-40,40],[-10, 150], option='roi', to_be_shown=True)

    

    th = time.time()
    hf.find_hits(pars.hit_pad_left,
                 pars.hit_pad_right,
                 pars.hit_dt_min[0],
                 pars.hit_amp_sig[0],
                 pars.hit_amp_sig[1],
                 pars.hit_amp_sig[2])

    print("hit %.2f s"%(time.time()-th))
    print("Number Of Hits found : ", dc.evt_list[-1].n_hits)


    # plot.plot_2dview_hits(to_be_shown=True)


    #plot.plot_wvf_current_hits_roi_vch([(0,212),(0,211),(0,210),(0,209)],to_be_shown=True)
    plot.plot_wvf_current_hits_roi_vch([(1,513),(1,514),(1,515),(1,516)],to_be_shown=True)
    plot.plot_wvf_current_hits_roi_vch([(1,470),(1,471),(1,472),(1,475)],to_be_shown=True)
    #plot.plot_wvf_current_hits_roi_vch([(2,308),(2,307),(2,306),(2,305)],to_be_shown=True)
    #plot.plot_wvf_current_hits_roi_vch([(2,58),(2,59),(2,60),(2,61)],to_be_shown=True)
    #plot.plot_wvf_diff_vch([(0,212),(1,514),(2,305)],to_be_shown=True)
    #plot.plot_2dview_hits(to_be_shown=True)
    #plot.plot_track_wvf_vch([[(0,x) for x in range(30,63)],[(1,x) for x in range(30,70)],[(2,x) for x in range(35,80)]], tmin=1200, tmax=2000, to_be_shown=True, option='1')    
    
    plot.event_display_per_view_hits_found([-40,40],[-10, 150],option='hits', to_be_shown=True)
    """
    plot.plot_track_wvf_vch([[(0,x) for x in range(30,63)],[(1,x) for x in range(30,70)],[(2,x) for x in range(35,80)]], tmin=1200, tmax=2000, to_be_shown=True, option='1')

    plot.plot_track_wvf_vch([[(0,x) for x in range(190,220)],[(1,x) for x in range(174,184)],[(1,x) for x in range(502,520)],[(2,x) for x in range(268,323)]], tmin=9200, tmax=10000, to_be_shown=True,option='2')
    """
                
    

    trk2d.find_tracks_rtree(pars.trk2D_nhits,
                            pars.trk2D_rcut,
                            pars.trk2D_chi2cut,
                            pars.trk2D_yerr,
                            pars.trk2D_slope_err,
                            pars.trk2D_pbeta)

    #[t.mini_dump() for t in dc.tracks2D_list]

    # plot.plot_2dview_2dtracks(to_be_shown=True)


    trk3d.find_tracks_rtree(pars.trk3D_ztol,
                            pars.trk3D_qfrac,
                            pars.trk3D_len_min,
                            pars.trk3D_dx_tol, 
                            pars.trk3D_dy_tol,
                            pars.trk3D_dz_tol)

    # plot.plot_3d(to_be_shown=True)
    print("Number of 3D tracks found : ", len(dc.tracks3D_list))

    print('  %.2f s to process '%(time.time()-t0))

    store.store_event(output)
    store.store_pedestals(output)
    store.store_hits(output)
    store.store_tracks2D(output)
    store.store_tracks3D(output)

#store.store_avf_wvf(output)
reader.close_file()
output.close()
print('it took %.2f s to run'%(time.time()-tstart))
