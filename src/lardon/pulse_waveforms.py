import lardon.config as cf
import lardon.data_containers as dc


from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


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



def fit_pulses(t, data, tstart, ispos, debug=False, show=False):
    if(len(t)!=len(data)):
        return []
    gmin, gmax = 50, 16000
    if(ispos == False):
        gmin, gmax = -16000, -50

    try : 
        popt, pcov = curve_fit(resp_bde, t, data, bounds=([t[0], gmin, .1], [tstart, gmax,4.]))
        perr = np.sqrt(np.diag(pcov))
                
        ts, g, tau = popt[0], popt[1], popt[2]
        ts_err, g_err, tau_err = perr[0], perr[1], perr[2]

        if(ispos==True):
            tm = np.argmax(data)
            vm = data[tm]
        else:
            tm = np.argmin(data)
            vm = data[tm]

        fit = resp_bde(t, *popt)
        if(ispos==True):
            sig = np.where(fit>0.1)
        else:
            sig = np.where(fit<-0.1)
        n = np.count_nonzero(sig)
        fit_area = np.sum(fit)
        area = np.sum(data[sig])
        
        chi_square = ((data[sig] - fit[sig])**2).sum()
        rchi_square = chi_square/(n-3)

        if(debug):
            ptype = 'POS' if ispos else 'NEG'
            print(ptype+' PULSE ')
            print('CHI2 ', chi_square, 'n ', n, ' rchi2 = ', rchi_square)
            print('area ', area, ' fitted area ', fit_area, ' g ', g, ' tau ', tau)
            print('Fit max : ', max(fit))
            for n, v, e in zip(["ts", "g", "tau"], popt, perr):
                print(n,": ", v," +/- ",e)
            print('\n')

        """
        if(show):

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.step(t, data, where='mid',c='k', 
                        label = 'Peak = %.2f ADC'%(max(data))
                        +'\n'
                        +'Area = %.2f ADC-tick'%(area))
                
            tplot = np.linspace(t[0], t[-1], 1000)

            ax.step(t, fit,c='orange', ls='dashed', where='mid',
                    label=r'start = %.2f $\pm$ %.2f $\mu$s'%(ts, ts_err)
                    +'\n'
                    +r'$\tau$ = %.2f $\pm$ %.2f $\mu$s'%(tau, tau_err)
                    +'\n'
                    +r'G=%.2f $\pm$ %.2f ADC'%(g, g_err)
                    +'\n'
                    +'area = %.2f ADC-tick'%(fit_area)
                    +'\n'
                    +r'r$\chi^2$ = %.2f'%(rchi_square))
                
            ax.plot(tplot, resp_bde(tplot, *popt),'r-')

            #ax.set_title('Peak Number '+str(i))
            ax.set_xlabel(r'Time [$\mu$s]')
            ax.set_ylabel(r'ADC')
            ax.legend(loc='upper right')
            run_nb = str(dc.evt_list[-1].run_nb)
            evt_nb = str(dc.evt_list[-1].trigger_nb)
 
            plt.show()
        """
        res = [ts, tm, vm, g, g_err, tau, tau_err, area, fit_area, rchi_square]
    
    except RuntimeError:
        res = []


    return res



def find_pulses():    
    #delta_t = 200
    pulse_dt = 2300
    adc_thr = 200

    h_delta_t = int(2300/2)

    
    tfunction = time.time()
    
    for idaq in range( cf.n_tot_channels):
        tw = time.time()
        view, vchan = dc.chmap[idaq].view, dc.chmap[idaq].vchan
        if(view >= cf.n_view or view < 0 or idaq in cf.broken_channels):
            #print(idaq, " skipped")
            continue

        wvf = dc.data_daq[idaq,:]

        fit_p = []
        npulses_p = 0

        i=0
        """ positive pulse """
        while(i*pulse_dt < cf.n_sample and i <10):
            start = i*pulse_dt
            stop = (i+1)*pulse_dt if (i+1)*pulse_dt < cf.n_sample else cf.n_sample
            pos_above = np.where(wvf[start:stop]>adc_thr)[0]

            if(len(pos_above) == 0):
                i+=1
                continue
            t_first = start + pos_above[0]
            search_start = t_first
            search_stop  = t_first+50 if t_first+50 < cf.n_sample else cf.n_sample
            tmax = t_first + np.argmax(wvf[search_start:search_stop])
            vmax = wvf[tmax]
            
            if(tmax < 10 or tmax > cf.n_sample-20):
                i += 1
                continue

            start = tmax - 10 if tmax-10 >=0 else 0
            stop  = tmax + 20 if tmax+20 < cf.n_sample else cf.n_sample

            #if(start >=0 and stop < cf.n_sample):
            """
            if(tmax >=30 and tmax < cf.n_sample-30):
                dc.wvf_pos[idaq].append(wvf[tmax-30:tmax+30].copy())
            """

            #print(idaq, view, vchan, "pulse", i, " at ", tmax, " = ", tmax/cf.sampling, " max ", vmax)
            tdata = np.linspace(start/cf.sampling, stop/cf.sampling, (stop-start)+1)
            #print("pos", start, stop, len(tdata), len(wvf[start:stop+1]))
            res = fit_pulses(tdata, wvf[start:stop+1], tmax/cf.sampling, ispos=True, debug=False, show=False)#(idaq==2693))#False)
            
            if(len(res)>1):
                fit_p.append(res)
                npulses_p += 1
            i+=1



        fit_n = []
        npulses_n = 0
            
        i=0
        """ negative pulse """
        while(i*pulse_dt < cf.n_sample and i <10):
            start = i*pulse_dt
            stop = (i+1)*pulse_dt if (i+1)*pulse_dt < cf.n_sample else cf.n_sample
            pos_below = np.where(wvf[start:stop]<-adc_thr)[0]

            if(len(pos_below) == 0):
                i+=1
                continue
            t_first = start + pos_below[0]
            search_start = t_first
            search_stop  = t_first+50 if t_first+50 < cf.n_sample else cf.n_sample
            tmin = t_first + np.argmin(wvf[search_start:search_stop])
            vmin = wvf[tmin]

            if(tmin < 10 or tmin > cf.n_sample-20):
                i += 1
                continue

            #if(i==0 and idaq < 50):
                #print(idaq, ": ", dc.evt_list[-1].noise_raw.ped_mean[idaq])
                #print(vmin, "---> ", vmin + dc.evt_list[-1].noise_raw.ped_mean[idaq])

            """ skip saturated waveforms """ 
            if(np.fabs(vmin+dc.evt_list[-1].noise_raw.ped_mean[idaq]) < 2):
                i+=1
                continue

            start = tmin - 10 if tmin-10 >=0 else 0
            stop  = tmin + 20 if tmin+20 < cf.n_sample else cf.n_sample

            #if(start >=0 and stop < cf.n_sample):
            """
            if(tmin >=30 and tmin < cf.n_sample-30):
                dc.wvf_neg[idaq].append(wvf[tmin-30:tmin+30].copy())
            """
                #dc.wvf_neg[idaq].append(wvf[start:stop].copy())

            tdata = np.linspace(start/cf.sampling, stop/cf.sampling, (stop-start)+1)
            #print("neg", start, stop, len(tdata), len(wvf[start:stop+1]))
            res = fit_pulses(tdata, wvf[start:stop+1], tmin/cf.sampling, ispos=False, debug=False, show=False)#(idaq==2693))#False)#(i==0))
            
            if(len(res)>1):
                fit_n.append(res)
                npulses_n += 1

            i+=1
        dc.pulse_fit_res.append(dc.fit_pulse(idaq, view, vchan, npulses_p, npulses_n, fit_p, fit_n)) 

    print('DONE it took ', time.time()-tfunction, ' to run')
'''
def find_pulses_prev():    
    delta_t = 2300
    h_delta_t = 1150
    if(False):
        if(vmax < 90):
            #tmax = 250 + np.argmax(wvf[250:500])
            tmax = h_delta_t + np.argmax(wvf[h_delta_t:h_delta_t+delta_t])
            vmax = wvf[tmax]

        fit_p = []
        npulses_p = 0
        

        i = 0
        while(tmax + h_delta_t < cf.n_sample or i< 10):#i > 20):

            start = tmax - h_delta_t if tmax >= h_delta_t else 0
            stop  = tmax + h_delta_t if tmax < cf.n_sample-100 else cf.n_sample

            """ positive pulse """

            tmax = start + np.argmax(wvf[start:stop+1])
            vmax = wvf[tmax]
            print("- - - - - - - - - - - - - - - - - - - - - - -\n",i, start, stop, tmax, vmax,"\n- - - - - - - - - - - - - - - - - - - - - - -\n")
            if(vmax < 90):
                #tmax+=200
                tmax+=delta_t
                continue

            i += 1            
            start = tmax - 10 #25
            stop  = tmax + 20 #25

            #this is debugging no ?
            if(start >=0 and stop < cf.n_sample):
                dc.wvf_pos[idaq].append(wvf[start:stop].copy())
        
            if(start < 0):
                start = 0
        
            if(stop >= cf.n_sample):
                stop = cf.n_sample-1   

            print("\n\n----->>>>>>>>>>>>>>>> found pulse at ", tmax, " with ", vmax, "\n -> start = ", start, " stop = ", stop)
            print(" in times = ", tmax/cf.sampling, start/cf.sampling, stop/cf.sampling)
            tdata = np.linspace(start/cf.sampling, stop/cf.sampling, (stop-start)+1)
            res = fit_pulses(tdata, wvf[start:stop+1], tmax, ispos=True, debug=True)
            
            if(len(res)>1):
                fit_p.append(res)
                npulses_p += 1
            
            
          
            #tmax += 500
            tmax += delta_t#500

            #plot pulse+fit
            if(True):#i<0):#idaq==10):
                ax.clear()
                print(res)
                #res = [ts, tm, vm, g, tau, area, fit_area, chi_square]
                f_tstart = res[0]
                f_tmax =  res[1]
                f_vmax = res[2]
                f_gain =  res[3]
                f_tau =  res[4]
                area = res[5]
                fit_area = res[6]
                chi_sq = res[7]
                ax.step(tdata, wvf[start:stop+1], where='mid',c='k', 
                        label = 'Peak = %.2f'%(f_vmax)
                        +'\n'
                        +'Area = %.2f'%(area))
                
                tplot = np.linspace(start/cf.sampling, stop/cf.sampling, 1000)

                ax.step(tdata, resp_bde(tdata, f_tstart, f_gain, f_tau),c='orange', ls='dashed', where='mid',
                        label=r'start = %.2f'%(f_tstart)
                        +'\n'
                        +r'$\tau$ = %.2f'%(f_tau)
                        +'\n'
                        +r'G=%.2f'%(f_gain)
                        +'\n'
                        +'area = %.2f'%(fit_area))
                        #+'\n'
                        #+r'r$chi^2$ = %.2f'%(rchi_square))
                
                ax.plot(tplot, resp_bde(tplot, f_tstart, f_gain, f_tau),'r-')

                ax.set_title('Peak Number '+str(i))
                ax.set_xlabel(r'Time [$\mu$s]')
                ax.set_ylabel(r'ADC')
                ax.legend(loc='upper right')
                run_nb = str(dc.evt_list[-1].run_nb)
                evt_nb = str(dc.evt_list[-1].trigger_nb)
                
                fig.savefig('results/pulse_'+run_nb+'_'+evt_nb+'_'+str(idaq)+'_pos_pulse_'+str(i)+'.png')
                plt.show()



        """ negative pulse """
        continue
        #tmin = np.argmin(wvf[:250])
        tmin = np.argmin(wvf[:h_delta_t])
        vmin = wvf[tmin]

        if(vmin > -90):
            #tmin = np.argmin(wvf[250:500])
            tmin = h_delta_t + np.argmin(wvf[h_delta_t:delta_t+h_delta_t])
            vmin = wvf[tmin]
        
        fit_n = []
        npulses_n = 0

        i = 0
        while(tmax + h_delta_t < cf.n_sample or i > 20):

            start = tmin - h_delta_t if tmin >= h_delta_t else 0
            stop  = tmin + h_delta_t if tmin < cf.n_sample-100 else cf.n_sample

            tmin = start + np.argmin(wvf[start:stop])
            vmin = wvf[tmin]

            
            if(vmin > -90):
                tmin += h_delta_t
                continue
            i += 1

            start = tmin - 100#10#25
            stop  = tmin + 250#50#25
        

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


            tmin += delta_t #500

            if(False):#i<0):#idaq==10):
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
'''
