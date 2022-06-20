# Liquid Argon Reconstruction Done in PythON
![Logo](figs/lardon_logo_text.png)


## Librairies needed to run lardon
You need miniconda installed :

https://docs.conda.io/en/latest/miniconda.html#linux-installers

and then get the librairies as stated in **lardenv.yml** :

`conda env create -f lardenv.yml`

 :warning: It'll take about 2-3 GB of space!

then : `conda activate lardenv`
 
## Before running lardon
Check and modify **config.py** and files in `settings/` :
* *store_path* : your directory where the output file will be stored
* *plot_path*  : your directory where control plots will be stored
* Update the detector & runs configuration files in `settings/` if needed, and the path to the data in `config.py`


## To run lardon on data
To launch lardon, type `python lardon.py` with the following arguments:<br/>
**Mandatory**:<br/>
* `-elec <top/tde/bot/bde>` which electronics is used
* `-run <run nb>` which run number
* `-sub <subfile name>` which subfile (*e.g.* 1)<br/>
*Optional*:<br/>
* `-n <nb of events>` how many events to process, default is -1 = all file
* `-out <output file option>` optional extra name for the output
* `-skip <nb to skip>` number of events to skip
* `-det <your_detector>` which detector to reconstruct, default is cb1 (coldbox 1st CRP) [one can also look at the np02 data]


*e.g.* : To run the first 10 events of top electronics run 455 subfile 5, type :

`python lardon.py -elec top -run 1415 -sub 5 -n 10 -out example`

the output h5file will be **store_path/top_455_5_example.h5**


## lardon Convention
# Coldbox 1st period
![convention](figs/coldbox_1.png)

* electrons drift along z axis
* the origin of the (x,y,z) system is at the center of the detector
* all distance are in cm


## Control Plots
:warning: The data is structured in `daq_channel` ordering, which can have a mix of views<br/>

By default, no control plots should be produced, but you can call the plotting functions in **lardon.py** anywhere in the reconstruction loop.


All plot functions have the two options :<br/>
* option="extra_output_name_if_you_want" [default is none]<br/>
* to_be_shown=True/False if you want to see the plot live [default is False]


### To plot the current event display:<br/>
Seen in `daq_channel` ordering:<br/>
`plot.plot_event_display_per_daqch()`<br/>
Seen in view channel ordering (more natural):<br/>
`cmap.arange_in_view_channels()` <- to be called first, might change in near future\
`plot.plot_event_display_per_view()`<br/>


### To plot the current waveform(s):<br/>
`plot.plot_wvf_current_daqch([daq_ch_1, daq_ch_2, ...])`<br/>
`plot.plot_wvf_current_vch([(view,ch1),(view,ch2),(...)])`<br/>

### To plot the noise RMS<br/>
`plot.plot_noise_daqch(noise_type='noise_type')`<br/>
`plot.plot_noise_vch(noise_type='noise_type')`<br/>
where `noise_type` is either `raw` or `filt`<br/>


### To check the signal/noise separation on the event display:<br/>
The signal (ROI) : `plot.event_display_per_view_roi()`<br/>
The noise : `plot.event_display_per_view_noise()`<br/>

### To plot hits found :<br/>
`plot.plot_2dview_hits()`<br/>
To see the hits found on top of the event display:<br/>
`plot.event_display_per_view_hits_found()`<br/>

### To plot 2D tracks (and hits):<br/>
`plot.plot_2dview_2dtracks()`<br/>

### To plot 3D tracks:<br/>
`plot.plot_2dview_hits_and_3dtracks()` <- see the 3D tracks projected in 2D<br/>
`plot.plot_3d()` <- see the 3D tracks in 3D<br/>

