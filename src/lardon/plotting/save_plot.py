import lardon.config as cf
import lardon.data_containers as dc
import lardon.utils.enum_type as et
import time as time
import matplotlib.pyplot as plt
#from matplotlib import rc


#rc('text', usetex=False)
def details(fig, is3D):
    run_nb = str(dc.evt_list[-1].run_nb)
    sub_nb = str(dc.evt_list[-1].sub)
    evt_nb = str(dc.evt_list[-1].evt_nb)
    trig_nb = str(dc.evt_list[-1].trigger_nb)
    det    = dc.evt_list[-1].det
    evt_time = dc.evt_list[-1].time_s
    trig_type = dc.evt_list[-1].trig_type
    
    sub_nb = sub_nb.replace('_','')
    run_nb = run_nb.replace('_','')


    infos = f"[{det}] Run {run_nb}-{sub_nb} event {evt_nb} (trigger {trig_nb})"
    infos_trig = f"{et.TRIGGER_TYPES[trig_type]}"
    infos_time = time.ctime(evt_time)
    
    ax = fig.gca()

    if(is3D==False):
        ax.text(0.005, 0.99, infos, transform=plt.gcf().transFigure, fontsize='x-small', va='top',ha='left', fontweight='bold')
        ax.text(0.005, 0.97, infos_trig, transform=plt.gcf().transFigure, fontsize='x-small', va='top',ha='left')
        ax.text(0.005, 0.95, infos_time, transform=plt.gcf().transFigure, fontsize='x-small', va='top',ha='left', fontstyle='italic')
    else:
        ax.text2D(0.005, 0.99, infos, transform=plt.gcf().transFigure, fontsize='x-small', va='top',ha='left')
        ax.text2D(0.005, 0.97, infos_trig, transform=plt.gcf().transFigure, fontsize='x-small', va='top',ha='left')
        ax.text2D(0.005, 0.95, infos_time, transform=plt.gcf().transFigure, fontsize='x-small', va='top',ha='left', fontstyle='italic')

def save(fig, option, out):

    run_nb = str(dc.evt_list[-1].run_nb)
    sub    = str(dc.evt_list[-1].sub)
    evt_nb = str(dc.evt_list[-1].evt_nb)
    trig_nb = str(dc.evt_list[-1].trigger_nb)
    #elec   = dc.evt_list[-1].elec
    det    = dc.evt_list[-1].det
    flow_nb = str(dc.evt_list[-1].dataflow)
    writer_nb = str(dc.evt_list[-1].datawriter)
    serv_nb = str(dc.evt_list[-1].daqserver)

    if(option):
        option = "_"+option
    else:
        option = ""


    out = f"{cf.plot_path}/{out}{option}_{det}_run{run_nb}-{sub}_f{flow_nb}w{writer_nb}s{serv_nb}_evt_{evt_nb}_trig_{trig_nb}.png"
    fig.savefig(out, dpi=200)


def save_with_details(fig, option, out, is3D=False):
    details(fig,is3D)
    save(fig,option,out)
    
