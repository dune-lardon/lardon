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
parser.add_argument('-det', dest='detector', help='which detector is looked at [default is coldbox]', default='coldbox', choices=['coldbox',])
parser.add_argument('-period', help='which detector period is looked at [default is 1]', default='1')
parser.add_argument('-out', dest='outname', help='extra name on the output', default='')
parser.add_argument('-skip', dest='evt_skip', type=int, help='nb of events to skip', default=0)
parser.add_argument('-f', '--file', help="Override derived filename")
parser.add_argument('-conf','--config',dest='conf', help='Analysis configuration ID', default='1')
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
store.create_tables(output)

""" set analysis parameters """
pars = params.params()
pars.read(config=args.conf,elec=elec)
pars.dump()

""" set the channel mapping """
print(" will use ", cf.channel_map)
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

    #plot.plot_sticky_finder_daqch(to_be_shown=True)

    #continue
    """ mask the unused channels """
    dc.mask_daq = dc.alive_chan

    if(ievent==-1):
        ki1=[18, 231, 230, 229, 228, 227, 226, 225, 224, 235, 234, 233, 232, 247, 199, 198, 197, 196, 195, 194, 193, 192, 207, 207, 206, 206, 205, 205, 204, 204, 203, 203, 202, 202, 201, 201, 200, 215, 215, 214, 214, 213, 213, 212, 212, 211, 211, 210, 210, 209, 209, 208, 208, 221, 220, 218, 218, 217, 217, 216, 216, 191, 190, 189, 188, 107, 115, 327, 328]
        ki2=[22, 23, 9, 34, 125, 248, 249, 250, 251, 252, 253, 254, 255, 240, 241, 242, 243, 244, 245, 246, 247, 232, 233, 234, 235, 236, 237, 238, 239, 231, 280, 281, 282, 283, 284, 285, 286, 287, 264, 265, 266, 267, 268, 268, 269, 256, 257, 258, 259, 260, 261, 262, 263, 296, 297, 298, 288, 289, 290, 291, 292, 293, 294, 295, 585, 584, 544, 436, 414]
        ki3=[101, 70, 33, 50, 25, 567, 542, 502, 500, 500, 467, 430, 389, 394, 402, 353, 295, 289, 289, 317, 316, 263, 262, 261, 260, 259, 258, 257, 256, 231, 230, 229, 228, 228, 227, 226, 225, 224, 239, 238, 237, 236, 235, 234, 233, 232, 247, 247, 246, 245, 244, 243, 242, 241, 240, 240, 255, 255, 255, 254, 254, 253, 253, 252, 252, 251, 251, 250, 250, 249, 249, 248, 248, 248, 199, 198, 197, 196, 195, 194, 194, 193, 192, 207, 206, 205, 204, 203, 202, 201, 200, 215, 214, 213, 212, 211, 211, 210, 209, 208, 223, 222, 221, 220, 219, 218, 217, 216, 167, 161, 175, 174, 173, 173, 172, 171, 170, 169, 168, 183, 182, 181, 180, 179, 178, 177, 176, 191, 190, 189, 188, 187, 186, 185, 184, 155, 154, 152]
        ki1=[39, 43, 42, 41, 40, 40, 198, 197, 197, 136, 154, 153, 152, 292]
        #ki2=[36, 90, 93, 94, 95, 96, 195, 197, 197, 198, 240, 241, 242, 243, 232, 233, 236, 237, 229, 231, 483, 444, 358]
        #ki3=[110, 35, 50, 7, 7, 6, 5, 5, 2, 2, 2, 1, 0, 13, 12, 11, 10, 10, 9, 9, 22, 544, 516, 537, 536, 495, 488, 502, 510, 478, 389, 388, 395, 304, 269, 269, 269, 268, 268, 268, 268, 268, 276, 217, 135, 135, 134, 134, 133, 132, 131, 131, 130, 129, 128, 139, 137, 136]
        plot.plot_wvf_current_vch([(0,i) for i in ki1[6:8]], to_be_shown=True)
        #plot.plot_wvf_current_vch([(1,i) for i in ki2[0:10]], to_be_shown=True)
        #plot.plot_wvf_current_vch([(2,i) for i in ki3[0:10]], to_be_shown=True)

    """ compute the raw pedestal """
    ped.compute_pedestal(noise_type='raw', pars=pars)


    #plot.plot_noise_vch(noise_type='raw', option='raw', vrange=[0., 50.],to_be_shown=True)
    #plot.event_display_per_daqch(pars.plt_evt_disp_daqch_zrange,option='raw',to_be_shown=False)

    #cmap.arange_in_view_channels()
    #plot.event_display_per_view([-50,50],[-10,150],option='raw', to_be_shown=True)
    #continue

    tcoh = time.time()
    if(pars.noise_coh_per_view):
        noise.coherent_noise_per_view(pars.noise_coh_group, pars.noise_coh_capa_weight)
    else :
        noise.coherent_noise(pars.noise_coh_group)

    print("coherent noise : ", time.time()-tcoh)


    ped.compute_pedestal(noise_type='filt')
    #tp = time.time()
    #ped.refine_mask(pars)
    ped.update_mask(pars.ped_amp_sig_oth)

    #plot.plot_noise_vch(noise_type='filt', vrange=[0,100],option='coh_nocapa',to_be_shown=False)
    #plot.plot_noise_vch(noise_type='filt', vrange=pars.plt_noise_zrange,option='coh_w',to_be_shown=False)
    #continue
    #cmap.arange_in_view_channels()
    #plot.event_display_per_view([-10,10],[-10,50],option='coh_nocapa', to_be_shown=True)
    #plot.event_display_per_view([-100,100],[-50,50],option='coh_nocapa', to_be_shown=False)

    #plot.plot_noise_vch(noise_type='filt', vrange=pars.plt_noise_zrange,option='coh_w',to_be_shown=True)
    #continue
    tf = time.time()
    ps = noise.FFT_low_pass(pars.noise_fft_lcut,pars.noise_fft_freq)


    """ DO NOT STORE ALL FFT PS !! """
    #store.store_fft(output, ps)



    ped.compute_pedestal(noise_type='filt')
    ped.refine_mask(pars)
    #ped.update_mask(pars.ped_amp_sig_oth)



    #cmap.arange_in_view_channels()
    #plot.event_display_per_view([-80,80],[-10, 180],option='filt', to_be_shown=True)

    #plot.plot_correlation_daqch(option='filtered',to_be_shown=True)
    #plot.plot_correlation_globch(option='filtered', to_be_shown=False)




    #tcoh = time.time()
    #noise.coherent_noise(pars.noise_coh_group)
    #print("coherent time took ", time.time()-tcoh)

    #plot.plot_noise_vch(noise_type='filt', vrange=pars.plt_noise_zrange,option='coherent',to_be_shown=False)




    th = time.time()
    hf.find_hits(pars.hit_pad_left,
                 pars.hit_pad_right,
                 pars.hit_dt_min[0],
                 pars.hit_amp_sig[0],
                 pars.hit_amp_sig[1],
                 pars.hit_amp_sig[2])

    # print("hit %.2f s"%(time.time()-th))
    # print("Number Of Hits found : ", dc.evt_list[-1].n_hits)

    if(ievent==-1):
        print([i.channel for i in dc.hits_list if i.view==0])
        print([i.channel for i in dc.hits_list if i.view==1])
        print([i.channel for i in dc.hits_list if i.view==2])
        plot.plot_wvf_current_vch([(0,i) for i in ki1[6:8]], to_be_shown=True)
        #plot.plot_wvf_current_vch([(1,i) for i in ki2[0:10]], to_be_shown=True)
        #plot.plot_wvf_current_vch([(2,i) for i in ki3[0:10]], to_be_shown=True)

        #plot.plot_wvf_current_vch([(0,i.channel) for i in dc.hits_list if i.view==0], to_be_shown=True)
        #plot.plot_wvf_current_vch([(1,i.channel) for i in dc.hits_list if i.view==1], to_be_shown=True)
        #plot.plot_wvf_current_vch([(2,i.channel) for i in dc.hits_list if i.view==2], to_be_shown=True)

    # plot.plot_2dview_hits(to_be_shown=True)






    trk2d.find_tracks_rtree(pars.trk2D_nhits,
                            pars.trk2D_rcut,
                            pars.trk2D_chi2cut,
                            pars.trk2D_yerr,
                            pars.trk2D_slope_err,
                            pars.trk2D_pbeta)

    [t.mini_dump() for t in dc.tracks2D_list]

    # plot.plot_2dview_2dtracks(to_be_shown=True)


    trk3d.find_tracks_rtree(pars.trk3D_ztol,
                            pars.trk3D_qfrac,
                            pars.trk3D_len_min,
                            pars.trk3D_dtol)

    # plot.plot_3d(to_be_shown=True)
    print("Number of 3D tracks found : ", len(dc.tracks3D_list))

    print('  %.2f s to process '%(time.time()-t0))

    store.store_event(output)
    store.store_pedestals(output)
    store.store_hits(output)
    store.store_tracks2D(output)
    store.store_tracks3D(output)
reader.close_file()
output.close()
print('it took %.2f s to run'%(time.time()-tstart))
