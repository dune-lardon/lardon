import config as cf
import data_containers as dc
import time as time
import matplotlib.pyplot as plt
from matplotlib import rc


rc('text', usetex=True)

def details(fig, is3D):
    run_nb = str(dc.evt_list[-1].run_nb)
    sub_nb = str(dc.evt_list[-1].sub)
    evt_nb = str(dc.evt_list[-1].evt_nb)
    trig_nb = str(dc.evt_list[-1].trigger_nb)
    det    = dc.evt_list[-1].det
    elec   = dc.evt_list[-1].elec
    evt_time = dc.evt_list[-1].time_s

    sub_nb = sub_nb.replace('_','')
    run_nb = run_nb.replace('_','')

    infos = '['+det+elec+r'] \textbf{Run '+run_nb+'-'+sub_nb+' event '+evt_nb+' (trigger '+trig_nb+')}'

    infos += '\n'
    infos += r'\textit{'+time.ctime(evt_time)+'}'

    ax = fig.gca()

    if(is3D==False):
        ax.text(0.005, 0.99, infos, transform=plt.gcf().transFigure, fontsize='x-small', va='top',ha='left')
    else:
        ax.text2D(0.005, 0.99, infos, transform=plt.gcf().transFigure, fontsize='x-small', va='top',ha='left')


def save(fig, option, out):

    run_nb = str(dc.evt_list[-1].run_nb)
    evt_nb = str(dc.evt_list[-1].trigger_nb)
    elec   = dc.evt_list[-1].elec
    det    = dc.evt_list[-1].det

    if(option):
        option = "_"+option
    else:
        option = ""

    fig.savefig(cf.plot_path+'/'+out+option+'_'+det+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png', dpi=200)


def save_with_details(fig, option, out, is3D=False):
    details(fig,is3D)
    save(fig,option,out)
