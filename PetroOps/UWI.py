import numpy as np
import math

class UWI:
    def __init__(self, str_name):
        self.str = str(str_name)
        if isinstance(str_name,(int,float)) == True:
            self.int = int(np.floor(str_name))
        else:
            self.int = None

    def show(self):
        print(self.str)
        print(self.int)

    def str2num(self):
        str_in = self.str
        try:
            if (str_in.upper() == 'NONE'):
                return None
            if self.int == None:
                str_in = str(str_in)
                str_in = str_in.strip()
                str_in = re.sub(r'[-−﹣−–—−]','-',str_in)
                c = len(re.findall('-',str_in))
                if c>1:
                    val = re.sub(r'[^0-9\.]','',str(str_in))
                else:
                    val = re.sub(r'[^0-9\.-]','',str(str_in))
                if val == '':
                    return None
                try:
                    val = int(val)
                except:
                    val = None
            else:
                val = self.int 
            return val
        except:
            print("CANNOT CONVERT STRING TO NUMBER: " + str(str_in))
            return None

    def API2INT(self,length = 10):
        val_in = self.str
        if val_in == None or str(val_in).upper() == 'NONE':
            return None
        try:
            if math.isnan(val_in):
                return None
        except:
            pass
        val = self.str2num()

        lim = 10**length-1
        highlim = 10**length-1 #length digits
        lowlim =10**(length-2) #length -1 digits
        while val > highlim:
            val = math.floor(val/100)
        while val < lowlim:
            val = val*100
        val = int(val)
        return(val)

    def dashed(self,length = 10):
        #val_in = self.str
        val_in = self.API2INT(length)
        val = str(val_in)
        val = val.zfill(length)
        val = val[0:2]+'-'+val[2:5]+'-'+val[5:length]
        if length > 10:
            val2 = val[0:12] + '-' + '-'.join(val[i:i+2] for i in range(10+2, length+2, 2))
            val = val2
        return(val)
