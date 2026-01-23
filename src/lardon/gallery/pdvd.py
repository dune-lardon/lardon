import matplotlib.pyplot as plt
import colorcet as cc
import lardon.channel_mapping as chmap
import lardon.plotting.save_plot as sp
import matplotlib.gridspec as gridspec
import lardon.config as cf
import lardon.data_containers as dc
import numpy as np
import lardon.utils.enum_type as et

cmap_coll = cc.cm.linear_tritanopic_krjcw_5_95_c24_r
cmap_ind  = cc.cm.diverging_tritanopic_cwr_75_98_c20

class effective_chan:
    def __init__(self, gallery):
        self.gallery = gallery
        self.is_tde = False if (gallery == "bottom" or gallery == "both") else True
        self.is_beam = gallery == "beam"
        
        self.conference_style = True#False
        self.no_data = False        
        self.nsamp = -1
    
    def setup(self):
        self.nsamp = cf.n_sample[cf.imod]
        if(self.nsamp <= 0):
            return
        self.data_eff_ind = np.zeros((2, 955, self.nsamp))
        self.data_eff_coll = np.zeros((584, self.nsamp))
        
    def read_module(self):
        if(cf.imod < 2): #== bottom crps
            if(self.gallery == "bottom" or self.gallery == "both"):
                self.is_tde = False
                return True
            else:
                return False
        else: #==top crps
            if(self.gallery == "bottom"):
                return False
            else:
                self.is_tde = True
                return True
                
    def add_data(self):
        print("module ", cf.imod, self.nsamp, " and ", cf.n_sample[cf.imod])
        chmap.arange_in_view_channels()
        if(self.nsamp < 0 and cf.n_sample[cf.imod] < 0):#
            return
        if(self.nsamp < 0 and cf.n_sample[cf.imod] > 0):#
            self.setup()
        
        if(self.is_tde):
            if(cf.imod == 2):
                self.data_eff_ind[0, :476]   += dc.data[0, :476]
                self.data_eff_ind[0, 97:573] += dc.data[0, 476:952]
                self.data_eff_ind[1, :476]   += dc.data[1, 476:952]
                self.data_eff_ind[1, 97:573] += dc.data[1, :476]

            elif(cf.imod == 3):
                self.data_eff_ind[0, 382:858]  += dc.data[0, :476]
                self.data_eff_ind[0, 479:] += dc.data[0, 476:952]
                self.data_eff_ind[1, 382:858]   += dc.data[1, 476:952]
                self.data_eff_ind[1, 479:] += dc.data[1, :476]
            else:
                print('MODULE ',imod, 'is not top CRP')
        else:
            if(cf.imod == 0):
                self.data_eff_ind[0, :476]   += dc.data[0, 476:952]
                self.data_eff_ind[0, 97:573] += dc.data[0, :476]
                self.data_eff_ind[1, :476]   += dc.data[1, :476]
                self.data_eff_ind[1, 97:573] += dc.data[1, 476:952]

            elif(cf.imod == 1):
                self.data_eff_ind[1, 382:858]  += dc.data[1, :476]
                self.data_eff_ind[1, 479:]     += dc.data[1, 476:952]
                self.data_eff_ind[0, 382:858]  += dc.data[0, 476:952]
                self.data_eff_ind[0, 479:]     += dc.data[0, :476]
            else:
                print('MODULE ',imod, 'is not bottom CRP')
            
             
        if(not self.is_beam):
            self.data_eff_coll[:292] += dc.data[2, :292]
            self.data_eff_coll[:292] += dc.data[2, 292:584]
            self.data_eff_coll[292:] += dc.data[2, 584:876]
            self.data_eff_coll[292:] += dc.data[2, 876:1168]
        else:
            if(cf.imod == 2):
                self.data_eff_coll[:292] += dc.data[2, 292:584]
                self.data_eff_coll[292:] += dc.data[2, 876:]
            elif(cf.imod == 3):
                self.data_eff_coll[:292] += dc.data[2, :292]
                self.data_eff_coll[292:] += dc.data[2, 584:876]
            else:
                print('nope')


    def draw(self):
        if(self.nsamp <= 0):
            return
        if(cf.imod%2 == 0):
            return
        if(cf.imod < 2 and (self.gallery == "top" or self.gallery == "beam")):
            return
        if(cf.imod >= 2 and self.gallery == "bottom"):
            return
        
        if(self.is_beam):
            trigger_tick =  int((dc.evt_list[-1].event_time - dc.evt_list[-1].charge_time[cf.imod])*1e6/0.5)+3000

            tmin = trigger_tick - 500
            tmax = trigger_tick + 1250
            if(tmin <0): tmin=0
            if(tmax>self.nsamp): tmax=self.nsamp

            
            show_beam_event(self.data_eff_ind, self.data_eff_coll,self.conference_style, tmin=tmin, tmax=tmax)                            
            #
            tmin = trigger_tick - 800
            tmax = trigger_tick + 1500
            if(tmin <0): tmin=0
            if(tmax>self.nsamp): tmax=self.nsamp


            show_beam_event_coll_only(self.data_eff_coll,self.conference_style, tmin=tmin, tmax=tmax)
        else:
            show_all_event(self.data_eff_ind, self.data_eff_coll, self.conference_style, is_top=self.is_tde, tmin=0, tmax=self.nsamp)
            #show_all_event(self.data_eff_ind, self.data_eff_coll, 2400, 0, "top" if self.is_tde else "bottom")
        self.nsamp = -1

def show_all_event(data_eff_ind, data_eff_coll,conf_style, is_top, tmin, tmax):
    
    adc_ind_min, adc_ind_max = -150, 150
    adc_coll_min, adc_coll_max = -50, 350

    fig = plt.figure(figsize=(12,5.5))


    gs = gridspec.GridSpec(nrows=2, 
                           ncols=3,
                           height_ratios=[1, 20])

    ax_col_ind = fig.add_subplot(gs[0, :2])
    ax_col_coll = fig.add_subplot(gs[0, 2])
    ax = []
    [ax.append(fig.add_subplot(gs[1,i]) if i==0 else fig.add_subplot(gs[1,i], sharey=ax[0])) for i in range(3)]
    
    if(is_top):
        origin='upper'
        #tmin, tmax = tmax, tmin
    else:
        origin = 'lower'

    for i in range(2):        
        im = ax[i].imshow(data_eff_ind[i].transpose(), 
                       origin = origin, 
                       aspect = 'auto', 
                       interpolation='none',
                       cmap   = cmap_ind,    
                       vmin   = adc_ind_min, 
                       vmax   = adc_ind_max)

    im = ax[2].imshow(data_eff_coll.transpose(), 
                      origin = origin, 
                      aspect = 'auto', 
                      interpolation='none',
                      cmap   = cmap_coll,    
                      vmin   = adc_coll_min, 
                      vmax   = adc_coll_max)

    

    ax[0].set_ylabel('Time')
    ax[-1].set_ylabel('Time')

    
    for a,v in zip(ax, range(3)):
        a.set_xlabel(f'View {v} channel')
        ta, tb = tmin, tmax
        if(is_top):
            ta, tb = tmax, tmin            
        a.set_ylim(ta, tb)#tmin, tmax)
        #a.set_xticks([])
        #a.set_yticks([])
    ax[-1].yaxis.set_label_position("right")


    ax_col_coll.set_title('Collected Charge [ADC]')
    cb = fig.colorbar(ax[2].images[-1], cax=ax_col_coll, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    ax_col_ind.set_title('Induced Charge [ADC]')
    cb = fig.colorbar(ax[0].images[-1], cax=ax_col_ind, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    if(conf_style==False):

        ax[-1].yaxis.tick_right()    
        ax[1].tick_params(labelleft=False)

        plt.subplots_adjust(hspace=0.16, wspace=0.1, top=0.8, bottom=0.12, left=0.1, right=0.9)

        
    else:
        for a,v in zip(ax, range(3)):
            a.set_xticks([])
            a.set_yticks([])


        
        minch = [25, 25, 25]
        offch = [5, 5, 5]
        offt = 300        
        if(is_top):
            offt=200
        else:
            offt=300
        
        for i in range(2):
            ax[i].plot([minch[i], minch[i]+102],[tmax-offt, tmax-offt], c='k', lw=1)
            ax[i].text(minch[i]+51, tmax-offt, "78 cm", c='k', ha='center',va='bottom', size='x-small')

        ax[2].plot([minch[2], minch[2]+147],[tmax-offt, tmax-offt], c='k', lw=1)
        
        ax[2].text(minch[2], tmax-offt, "75 cm", c='k', ha='left',va='bottom', size='x-small')
        
        for i in range(2):
            ax[i].plot([minch[i], minch[i]],[tmax-offt, tmax-offt-1000], c='k', lw=1)
            ax[i].text(minch[i]-offch[i], tmax-offt-500, "80 cm", c='k', ha='center',va='center', size='x-small', rotation='vertical')

        ax[2].plot([minch[2], minch[2]],[tmax-offt, tmax-offt-1000], c='k', lw=1)
        ax[2].text(minch[2]-offch[2], tmax-offt-500, "80 cm", c='k', ha='center',va='center', size='x-small', rotation='vertical')


        plt.subplots_adjust(hspace=0.16, wspace=0.05, top=0.8, bottom=0.08, left=0.04, right=0.96)



    trig_type = dc.evt_list[-1].trig_type
    trigg = f"{et.TRIGGER_TYPES[trig_type]}"

    drift = "top" if is_top else "bottom"
    dc.evt_list[-1].det = "pdvd-"+drift
    sp.save_with_details(fig, '', 'ED_'+drift+'_gallery_3view_'+trigg)
    dc.evt_list[-1].det = "pdvd"
    

    #plt.show()

    plt.close()



    
def show_beam_event(data_eff_ind, data_eff_coll, conf_style, tmin, tmax):

    adc_ind_min, adc_ind_max = -200, 200
    adc_coll_min, adc_coll_max = -50, 450

    chan_min = [350,170,0]#[480, 380, 0] #[view0, view1, view2]
    chan_max = [740,840,584]#[530, 650, 584] #[view0, view1, view2]

    
    fig = plt.figure(figsize=(9.5,4.5))


    gs = gridspec.GridSpec(nrows=2, 
                           ncols=3,
                           height_ratios=[1, 20])

    ax_col_ind = fig.add_subplot(gs[0, :2])
    ax_col_coll = fig.add_subplot(gs[0, 2])
    ax = []
    [ax.append(fig.add_subplot(gs[1,i]) if i==0 else fig.add_subplot(gs[1,i], sharey=ax[0])) for i in range(3)]
    
    
    origin='upper'

    for i in range(2):        
        im = ax[i].imshow(data_eff_ind[i].transpose(), 
                       origin = origin, 
                       aspect = 'auto', 
                       interpolation='none',
                       cmap   = cmap_ind,    
                       vmin   = adc_ind_min, 
                       vmax   = adc_ind_max)
        
    im = ax[2].imshow(data_eff_coll.transpose(), 
                      origin = origin, 
                      aspect = 'auto', 
                      interpolation='none',
                      cmap   = cmap_coll,    
                      vmin   = adc_coll_min, 
                      vmax   = adc_coll_max)
                      #extent = [ch_min_coll, ch_max_coll, tmin,tmax])    

    
    ax[0].set_ylabel('Time')
    ax[-1].set_ylabel('Time')
    ax[-1].yaxis.set_label_position("right")
    
    
    ax_col_coll.set_title('Collected Charge [ADC]')
    cb = fig.colorbar(ax[2].images[-1], cax=ax_col_coll, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    ax_col_ind.set_title('Induced Charge [ADC]')
    cb = fig.colorbar(ax[0].images[-1], cax=ax_col_ind, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')


    
    for a,v in zip(ax, range(3)):
        a.set_xlabel(f'View {v} channel')
        a.set_ylim(tmax, tmin)
        a.set_xlim(chan_min[v], chan_max[v])
        

    if(conf_style == False):
        ax[-1].yaxis.tick_right()    
        ax[1].tick_params(labelleft=False)

        plt.subplots_adjust(hspace=0.16, wspace=0.1, top=0.8, bottom=0.12, left=0.11, right=0.89)
    else:
        for a,v in zip(ax, range(3)):
            a.set_xticks([])
            a.set_yticks([])
        
        minch = [25, 25, 25]
        offch = [5, 6, 6]
        offt = 50

        for i in range(2):
            ax[i].plot([chan_min[i]+minch[i], chan_min[i]+minch[i]+68],[tmax-offt, tmax-offt], c='k', lw=1)
            ax[i].text(chan_min[i]+minch[i]+3, tmax-offt, "52 cm", c='k', ha='left',va='bottom', size='x-small')
        ax[2].plot([chan_min[2]+minch[2], chan_min[2]+minch[2]+98],[tmax-offt, tmax-offt], c='k', lw=1)
        ax[2].text(chan_min[2]+minch[2]+3, tmax-offt, "50 cm", c='k', ha='left',va='bottom', size='x-small')

        for i in range(2):
            ax[i].plot([chan_min[i]+minch[i], chan_min[i]+minch[i]],[tmax-offt, tmax-offt-400], c='k', lw=1)
            ax[i].text(chan_min[i]+minch[i]-offch[i], tmax-offt-200, "32 cm", c='k', ha='center',va='center', size='x-small', rotation='vertical')

        ax[2].plot([chan_min[2]+minch[2], chan_min[2]+minch[2]],[tmax-offt, tmax-offt-400], c='k', lw=1)
        ax[2].text(chan_min[2]+minch[2]-offch[2], tmax-offt-200, "32 cm", c='k', ha='center',va='center', size='x-small', rotation='vertical')
        
        plt.subplots_adjust(hspace=0.16, wspace=0.05, top=0.8, bottom=0.12, left=0.05, right=0.95)



        
    trig_type = dc.evt_list[-1].trig_type
    trigg = f"{et.TRIGGER_TYPES[trig_type]}"
    dc.evt_list[-1].det = "pdvd-top"
    sp.save_with_details(fig, '', 'ED_beam_gallery_3view_'+trigg)
    dc.evt_list[-1].det = "pdvd"

    plt.show()
    
    plt.close()


def show_beam_event_coll_only(data_eff, conf_style, tmin, tmax):
    

    adc_min, adc_max = -50, 450

    
    fig = plt.figure(figsize=(6,5))
    gs = gridspec.GridSpec(nrows=2, 
                           ncols=1,
                           height_ratios=[1,20])

    ax_col = fig.add_subplot(gs[0,0])
    ax = fig.add_subplot(gs[1,0])


    origin='upper'

    im = ax.imshow(data_eff.transpose(), 
                   origin = origin, 
                   aspect = 'auto', 
                   interpolation='none',
                   cmap   = cmap_coll,    
                   vmin   = adc_min, 
                   vmax   = adc_max)

    ax.set_ylabel('Time [tick]')        
    ax.set_xlabel('Effective Channel')

    ax.set_ylim(tmax, tmin)

    
    #ax.plot([10, 59],[cf.n_sample[cf.imod]-50, cf.n_sample[cf.imod]-50], c='w', lw=2)
    #ax.text(34.5, cf.n_sample[cf.imod]-50, "25 cm", c='w', ha='center',va='bottom', size='small')
    
    ax_col.set_title('Collected Charge [ADC]')

    cb = fig.colorbar(ax.images[-1], cax=ax_col, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    
    plt.subplots_adjust(hspace=0.1, top=0.82, bottom=0.12, left=0.18, right=0.98)
    trig_type = dc.evt_list[-1].trig_type
    trigg = f"{et.TRIGGER_TYPES[trig_type]}"

    dc.evt_list[-1].det = "pdvd-top"
    sp.save_with_details(fig, '', 'ED_beam_gallery_coll_'+trigg)    
    dc.evt_list[-1].det = "pdvd"


    #plt.show()

    plt.close()
