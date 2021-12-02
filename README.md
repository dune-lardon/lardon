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
To launch lardon, type `python lardon.py` with the following arguments:
**Mandatory**: 
* `-elec <top/tde/bot/bde>` which electronics is used
* `-run <run nb>` which run number
* `-sub <subfile name>` which subfile (*e.g.* 1)
*Optional*:
* `-n <nb of events>` how many events to process, default is -1 = all file
* `-out <output file option>` optional extra name for the output
* `-skip <nb to skip>` number of events to skip
* `-det <your_detector>` which detector to reconstruct, default is coldbox


*e.g.* : To run the first 10 events of top electronics run 455 subfile 5, type :

`python lardon.py -elec top -run 1415 -sub 5 -n 10 -out example`

the output h5file will be **store_path/top_455_5_example.h5**


## lardon Convention
# Coldbox 1st period
![convention](figs/coldbox_1.png)

* electrons drift along z axis
* the origin of the (x,y,z) system is at the center of the detector
* all distance are in cm

## Reconstruction Parameters
The file **default_reco.yaml** contains all parameters needed for the reconstruction. If you want to change/add parameters, you can create your own **.yaml** file, following the `name: value` format. You don't have to copy all variables, the default one will be taken. 

:warning: The data is structured in `daq_channel` ordering

## Control Plots
By default, no control plots should be produced, but you can call the plotting functions in **lardon.py** anywhere in the reconstruction loop.


All plot functions have the two options :
* option="extra_output_name_if_you_want" [default is none] 
* to_be_shown=True/False if you want to see the plot live [default is False]


### To plot the current event display:
Seen in `daq_channel` ordering:
`plot.plot_event_display_per_daqch()`
Seen in view channel ordering (more natural):
`cmap.arange_in_view_channels()` <- to be called first, might change in near future
`plot.plot_event_display_per_view()`


### To plot the current waveform(s):
`plot.plot_wvf_current_daqch([daq_ch_1, daq_ch_2, ...])`
`plot.plot_wvf_current_vch([(view,ch1),(view,ch2),(...)])`

### To plot the noise RMS
`plot.plot_noise_daqch(noise_type='noise_type')`
`plot.plot_noise_vch(noise_type='noise_type')`
where `noise_type` is either `raw` or `filt`

### To plot hits found :
`plot.plot_2dview_hits()`

### To plot 2D tracks (and hits):
`plot.plot_2dview_2dtracks()` 

### To plot 3D tracks:
`plot.plot_2dview_hits_and_3dtracks()` <- see the 3D tracks projected in 2D
`plot.plot_3d()` <- see the 3D tracks in 3D

