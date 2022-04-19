#import jax
#import jax.numpy as np
#import modin.pandas as pd
#import ray
#ray.init()
import pandas as pd
import numpy as np
import sys, re, math
from os import path, listdir, remove, makedirs, rename
from math import ceil,isnan,floor
import multiprocessing, concurrent.futures
from functools import partial

def read_excel(file):
    outdf = None
    xl = {}
    # read excel as a dictionary of dataframes
    try:
        xl = pd.read_excel(file,None)
    except:
        xl = pd.read_excel(file)
        pass
    if len(xl)==0:
        print('FILE XL READ ERROR IN: '+ file)
        outdf = 'FILE XL READ ERROR IN: '+ file
    
    if isinstance(xl,dict): # test if file read delivered a dictionary
        for k in xl.keys(): # for each sheet
            df_s = xl[k].copy(deep=True)
            df_s = df_s.dropna(how='all',axis=0).dropna(how='all',axis=1)
            #print(outdf)
            if isinstance(outdf,pd.DataFrame):
                return outdf
        
    if isinstance(outdf,pd.DataFrame):
        outdf = outdf.dropna(how='any',axis=0)
        outdf = outdf.dropna(how='all',axis=1)
    return outdf

def filelist(SUBDIR = None,EXT = None, BEGIN = None):
    pathname = path.dirname(sys.argv[0])

    if SUBDIR != None:
        pathname = os.path.join(pathname, SUBDIR)
        
    FLIST = list()
    if (EXT == None) & (BEGIN == None):
        FLIST = listdir(pathname)
    else:
        for f in listdir(pathname):
            if filetypematch(f, filetypes=EXT, prefix = BEGIN):
                FLIST.append(f)
    return FLIST

def tupelize(x):
    if isinstance(x,(str,float,int)):
        out = tuple([x])
    else:
        out = tuple(x)
    return out

def filetypematch(fname, filetypes,prefix = None):
    filetypes = tupelize(filetypes)
    output = fname.lower().endswith(filetypes)
    if prefix != None:
        prefix = tupelize(prefix)
        output = output * fname.lower().startswith(prefix)
    return output

if __name__=='__main__':
    pathname = path.dirname(sys.argv[0])
    adir = path.abspath(pathname)
    #if 1==1:
    FLIST = filelist(EXT='.csv',BEGIN = 'frac')
    FracFocus = pd.DataFrame()
    for f in FLIST:
        freg_df = pd.read_csv(f,low_memory=False)
        #freg_df = freg_df.drop_duplicates()
        FracFocus = pd.concat([FracFocus,freg_df],axis=0,join='outer',ignore_index=True)
    FracFocus = FracFocus.drop_duplicates()

    FracFocus.APINumber = FracFocus.APINumber.astype(str).str.replace(' ','').astype(int)
    
    FracFocus.to_json('FracFocusTables.JSON')
    FracFocus.to_parquet('FracFocusTables.PARQUET')
    
    #pd.pivot_table(FracFocus.loc[FracFocus.Purpose.astype('str').str.contains(r'divert|friction|gel',flags=re.IGNORECASE,regex=True)],values = ['MassIngredient'],index = ['APINumber','Purpose'], aggfunc={'MassIngredient':np.sum}).to_csv('Api_Purpose_Mass.csv')    

