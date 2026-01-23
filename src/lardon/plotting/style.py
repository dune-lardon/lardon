import matplotlib as mpl

def set_style():    
    params = {'figure.figsize': (12, 6),
              'legend.fontsize':'x-large',              
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize':'x-large',
              'ytick.labelsize':'x-large',
              "font.family": "serif",
              "font.serif": ["CMU Serif", "STIX", "DejaVu Serif"],  # Try in order
}
              #"font.serif": ["Computer Modern Roman"],
              #"mathtext.fontset": "cm"}
              
    mpl.rcParams.update(params)
    
