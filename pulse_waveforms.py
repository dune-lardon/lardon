from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


import config as cf
import data_containers as dc
import time as time

                

def resp_bde(x, t, A, tau):
    A *= 10*1.012

    A1 = 4.31054*A
    A2 = 2.6202*A
    A3 = 0.464924*A
    A4 = 0.762456*A
    A5 = 0.32768*A
    
    E1 = np.exp(-2.94809*(x-t)/tau)
    E2 = np.exp(-2.82833*(x-t)/tau)
    E3 = np.exp(-2.40319*(x-t)/tau)
    
    L1 = 1.19361*(x-t)/tau
    L2 = 2.38722*(x-t)/tau
    L3 = 2.5928*(x-t)/tau
    L4 = 5.18561*(x-t)/tau

    val = A1*E1 - A2*E2*(np.cos(L1) + np.cos(L1)*np.cos(L2) + np.sin(L1)*np.sin(L2)) + A3*E3*(np.cos(L3) + np.cos(L3)*np.cos(L4) + np.sin(L3)*np.sin(L4)) + A4*E2*(np.sin(L1) - np.cos(L2)*np.sin(L1) + np.cos(L1)*np.sin(L2)) - A5*E3*(np.sin(L3) - np.cos(L4)*np.sin(L3) + np.cos(L3)*np.sin(L4))
    
    out = np.piecewise(x, [x < t, x >= t], [0., 1])
    return out*val



def fit_pulses(t, data, tstart, ispos, debug=False):
    gmin, gmax = 1, 2000
    if(ispos == False):
        gmin, gmax = -2000, -1

    try : 
        popt, pcov = curve_fit(resp_bde, t, data, bounds=([t[0], gmin, .1], [t[-1], gmax,5.]))
        perr = np.sqrt(np.diag(pcov))
                
        ts, g, tau = popt[0], popt[1], popt[2]
        ts += tstart


        if(ispos==True):
            tm = np.argmax(data)
            vm = data[tm]
        else:
            tm = np.argmin(data)
            vm = data[tm]

        tm += tstart

        fit = resp_bde(t, *popt)
        sig = np.where(fit>0.1)
        n = np.count_nonzero(sig)
        fit_area = np.sum(fit)
        area = np.sum(data[sig])
        
        chi_square = ((data[sig] - fit[sig])**2).sum()
        rchi_square = chi_square/(n-3)

        if(debug):
            ptype = 'POS' if ipos else 'NEG'
            print(ptype+' PULSE ')
            print('CHI2 ', chi_square, 'n ', n, ' rchi2 = ', rchi_square)
            print('area ', area, ' fitted area ', fit_area, ' g ', g, ' tau ', tau)
            print(popt)
            print(perr)
            print('\n')
            
        res = [ts, tm, vm, g, tau, area, fit_area, chi_square]
    
    except RuntimeError:
        res = []


    return res



def find_pulses():
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    
    tfunction = time.time()
    
    for idaq in range(2):#cf.n_tot_channels):
        
        tw = time.time()
        view, vchan = dc.chmap[idaq].view, dc.chmap[idaq].vchan
        if(view >= cf.n_view or view < 0):
            continue

        wvf = dc.data_daq[idaq,:]
            
        tmax = np.argmax(wvf[:250])
        vmax = wvf[tmax]
        if(vmax < 90):
            tmax = 250 + np.argmax(wvf[250:500])
            vmax = wvf[tmax]

        fit_p = []
        npulses_p = 0
        

        i = 0
        while(tmax + 50 < cf.n_sample or i > 20):

            start = tmax - 100 if tmax >= 100 else 0
            stop  = tmax + 100 if tmax < cf.n_sample-100 else cf.n_sample

            """ positive pulse """

            tmax = start + np.argmax(wvf[start:stop+1])
            vmax = wvf[tmax]
            if(vmax < 90):
                tmax+=200
                continue

            i += 1            
            start = tmax - 10#25
            stop  = tmax + 50#25

            if(start >=0 and stop < cf.n_sample):
                dc.wvf_pos[idaq].append(wvf[start:stop].copy())
        
            if(start < 0):
                start = 0
        
            if(stop >= cf.n_sample):
                stop = cf.n_sample-1   

            
            tdata = np.linspace(start/cf.sampling, stop/cf.sampling, (stop-start)+1)
            res = fit_pulses(tdata, wvf[start:stop+1], tmax, ispos=True)
            
            if(len(res)>1):
                fit_p.append(res)
                npulses_p += 1
            
            
          
            tmax += 500

            
            ''' plot pulse+fit '''
            if(i<0):#idaq==10):
                ax.clear()
                ax.step(tdata, wvf[start:stop+1], where='mid',c='k', 
                        label = 'Peak = %.2f'%(vmax)
                        +'\n'
                        +'Area = %.2f'%(area))
                
                tplot = np.linspace(start/cf.sampling, stop/cf.sampling, 1000)
                ax.step(tdata, fit,c='orange', ls='dashed', where='mid',
                        label=r'start = %.2f $\pm$ %.2f'%(popt[0], perr[0])
                        +'\n'
                        +r'$\tau$ = %.2f $\pm$ %.2f'%(popt[2], perr[2])
                        +'\n'
                        +r'G=%.2f $\pm$ %.2f'%(popt[1], perr[1])
                        +'\n'
                        +'area = %.2f'%(fit_area)
                        +'\n'
                        +r'r$chi^2$ = %.2f'%(rchi_square))
                
                ax.plot(tplot, resp_bde(tplot, *popt),'r-')
                ax.set_title('Peak Number '+str(i))
                ax.set_xlabel(r'Time [$\mu$s]')
                ax.set_ylabel(r'ADC')
                ax.legend(loc='upper right')
                run_nb = str(dc.evt_list[-1].run_nb)
                evt_nb = str(dc.evt_list[-1].trigger_nb)
                
                fig.savefig('results/pulse_'+run_nb+'_'+evt_nb+'_'+str(idaq)+'_pos_pulse_'+str(i)+'.png')




        """ negative pulse """
                    
        tmin = np.argmin(wvf[:250])
        vmin = wvf[tmin]

        if(vmin > -90):
            tmin = np.argmin(wvf[250:500])
            vmin = wvf[tmin]
        
        fit_n = []
        npulses_n = 0

        i = 0
        while(tmin + 50 < cf.n_sample or i > 20):

            start = tmin - 100 if tmin >= 100 else 0
            stop  = tmin + 100 if tmin < cf.n_sample-100 else cf.n_sample

            tmin = start + np.argmin(wvf[start:stop])
            vmin = wvf[tmin]

            
            if(vmin > -90):
                tmin += 200
                continue
            i += 1

            start = tmin - 10#25
            stop  = tmin + 50#25
        

            if(start >=0 and stop < cf.n_sample):
                dc.wvf_neg[idaq].append(wvf[start:stop].copy())


            if(start < 0):
                start = 0
        
            if(stop >= cf.n_sample):
                stop = cf.n_sample-1   



        
            tdata = np.linspace(start/cf.sampling, stop/cf.sampling, (stop-start)+1)
            res = fit_pulses(tdata, wvf[start:stop+1], tmin, ispos=False)
            #res = [ts, tmin, vmin, g, tau, area, fit_area, chi_square]

            if(len(res)>1):
                fit_n.append(res)
                npulses_n += 1


            tmin += 500

            if(i<0):#idaq==10):
                ax.clear()
                ax.step(tdata, wvf[start:stop+1], where='mid',c='k', 
                        label = 'Peak = %.2f'%(vmin)
                        +'\n'
                        +'Area = %.2f'%(area))
                
                
                tplot = np.linspace(start/cf.sampling, stop/cf.sampling, 1000)
                ax.step(tdata, fit,c='orange', ls='dashed', where='mid',
                        label=r'start = %.2f $\pm$ %.2f'%(popt[0], perr[0])
                        +'\n'
                        +r'$\tau$ = %.2f $\pm$ %.2f'%(popt[2], perr[2])
                        +'\n'
                        +r'G=%.2f $\pm$ %.2f'%(popt[1], perr[1])
                        +'\n'
                        +'area = %.2f'%(fit_area)
                        +'\n'
                        +r'r$chi^2$ = %.2f'%(rchi_square))
                
                ax.plot(tplot, resp_bde(tplot, *popt),'r-')
                ax.set_title('Peak Number '+str(i))
                ax.set_xlabel(r'Time [$\mu$s]')
                ax.set_ylabel(r'ADC')
                ax.legend(loc='lower right')
                run_nb = str(dc.evt_list[-1].run_nb)
                evt_nb = str(dc.evt_list[-1].trigger_nb)
                
                fig.savefig('results/pulse_'+run_nb+'_'+evt_nb+'_'+str(idaq)+'_neg_pulse_'+str(i)+'.png')
                
            
            
        dc.pulse_fit_res.append(dc.fit_pulse(idaq, view, vchan, npulses_p, npulses_n, fit_p, fit_n)) 

    print('DONE it took ', time.time()-tfunction, ' to run')

