import numpy as np
import math
import re

class WELLAPI:
    def show(self):
        print(self.str)
        print(self.int)
        
    def str2num(self):
        str_in = self.str
        if (str_in.upper() == 'NONE'):
            return None
        if str(self.int).upper() == 'NONE':
            str_in = str(str_in)
            str_in = str_in.strip()
            str_in = re.sub(r'[-−﹣−–—−]','-',str_in)
            c = len(re.findall('-',str_in))
            if c>1:
                val = re.sub(r'[^0-9\.]','',str(str_in))
            else:
                val = re.sub(r'[^0-9-\.]','',str(str_in))
            if val == '':
                return None
            try:
                val = np.floor(float(val))
            except:
                val = None
        else:
            val = self.int 
        return val


    def API2INT(self,length = 10):
        val_in = self.str
        
        if str(val_in).upper() == 'NONE':
            return 0
        
        try:
            if math.isnan(val_in):
                return 0
        except:
            pass
        val = self.str2num()
        
        if (val == None) or (val == 0):
            return 0
        
        lim = 10**length-1
        highlim = 10**length-1 #length digits
        lowlim =10**(length-2) #length -1 digits
        while val > highlim:
            val = math.floor(val/100)
        while val < lowlim:
            val = val*100
        val = int(val)
        return(val)

    def STRING(self,length = 10, dashed = False):
        #val_in = self.str
        val_in = self.API2INT(length)
        if val_in == None:
            return None
        val = str(val_in)
        val = val.zfill(length)
        if dashed:
            val = val[0:2]+'-'+val[2:5]+'-'+val[5:length]
            if length > 10:
                val2 = val[0:12] + '-' + '-'.join(val[i:i+2] for i in range(10+2, length+2, 2))
                val = val2
        return(val)
    
    def __init__(self, str_name):
        self.str = str(str_name)
        self.int = None  
    
        if self.str.upper() == 'NAN':
            self.str = 'None'
            self.int = None  
        elif isinstance(self,(int,float)) == True:
            self.int = int(np.floor(str_name))
        else:
            self.int = int(self.str2num())


