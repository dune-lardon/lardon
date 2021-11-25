import json as json

class params:
    def __init__(self):
       self.ped_amp_sig_fst = 1          # default amplitude trigger threshold for 1st pass signal mask - in RMS 
       self.ped_amp_sig_oth = 1          # default amplitude trigger threshold for other pass signal mask - in RMS 

       self.hit_amp_sig     = [3,6,2]    # default amplitude trigger threshold for hit search - in RMS
       self.hit_dt_min      = [10,10,10] # minimal delta t for hit search - in bins
       self.hit_pad_left    = 6
       self.hit_pad_right   = 10

    def read(self,config="1"):
       try:
          with open('settings/analysis_parameters.json','r') as f:
             data = json.load(f)
             if(config not in data):
                print("WARNING: Thresholds configuration ",config," not found.")
                print("         Default thresholds will be applied.")
             else:
                self.ped_amp_sig_fst = data[config]['pedestal']['first_pass_thrs']
                self.ped_amp_sig_oth = data[config]['pedestal']['other_pass_thrs']

                self.hit_amp_sig     = data[config]['hit_finder']['amp_sig_thrs']
                self.hit_dt_min      = data[config]['hit_finder']['dt_min']
                self.hit_pad_left    = data[config]['hit_finder']['pad']['left']
                self.hit_pad_right   = data[config]['hit_finder']['pad']['right']
       except:
            print("WARNING: Thresholds setting file (./settings/analysis_parameters.json) not found.")
            print("         Default thresholds will be applied.")

    # setters and getters (potentially not useful now)
    def set_ped_amp_sig_fst(self,value):
      self.ped_amp_sig_fst = values 

    def set_ped_amp_sig_oth(self,value):
      self.ped_amp_sig_oth = values 

    def get_ped_amp_fst(self):
      return self.ped_amp_sig_fst 

    def get_ped_amp_oth(self):
      return self.ped_amp_oth_fst 

    def set_hit_amp_sig(self,values):
      self.hit_amp_sig = values

    def set_hit_dt_min(self,values):
      self.hit_dt_min = values

    def set_hit_pad_left(self,value):
      self.hit_pad_left = value

    def set_hit_pad_right(self,value):
      self.hit_pad_right = value
    
    def get_hit_amp_sig_thrs(self):
      return self.hit_amp_sig  
    
    def get_hit_dt_min_thrs(self):
      return self.hit_dt_min

    def get_hit_pad_left(self):
      return self.hit_pad_left

    def get_hit_pad_right(self):
      return self.hit_pad_right


