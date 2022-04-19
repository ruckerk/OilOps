#import ray
#ray.init(num_cpus=12)
# jax
# modin

import numpy as np
import pandas as pd
import sys, re, math, shutil
from os import path, listdir, remove, makedirs, rename
from math import ceil,isnan,floor
import multiprocessing, concurrent.futures
from functools import partial
from datetime import datetime


# BUG FIXES
#v101 as_type(float) errors fixed with pd.to_numeric(errors='coerce')
#v106 looks for API col
#v107 checks file lists and only appends new files
#v200 new file name handling, modin, JSON



# NEED TO HANDLE UNLABELE COLUMNS


# needs to be updated to populate SQL db
# parallelize


def str2num(str_in):
    try:
        if (str(str_in).upper() == 'NONE'):
            return None
        if isinstance(str_in,(int,float)) == False:
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
                val = float(val)
            except:
                val = None
        else:
            val = str_in
        return val
    except:
        print("CANNOT CONVERT STRING TO NUMBER: " + str(str_in))
        return None

def str2num_noalpha(str_in):
    if str_in == None:
        return None
    str_in = str(str_in)
    #regexp = re.compile(r'[a-z]')
    #if regexp.search(str_in,re.IGNORECASE):
    if re.search(r'[a-z]',str_in,re.IGNORECASE):
        return(None)
    else:
        return(str2num(str_in))
    

def API2INT(val_in,length = 10):

    val_in = str2num(val_in)

    try:
        if math.isnan(val_in):
            return None
    except:
        pass
    

    if val_in == None or str(val_in).upper() == 'NONE':
        return None


    lim = 10**length-1
    highlim = 10**length-1 #length digits
    lowlim =10**(length-2) #length -1 digits
    while val_in > highlim:
        val_in = math.floor(val_in/100)
    while val_in < lowlim:
        val_in = val_in*100

    val_in = int(val_in)
    
    return(val_in)


def Find_API_Col(df_inAPI):

    # NOT WORKING RIGHT returning datestrings
    
    #if 1==1:
    APIterms = ['API','UWI']
    
    df2 = df_inAPI.copy(deep = True)
    lowlim = 10**(8)
    highlim = 10**14
    
    df2 = df2.applymap(str2num_noalpha)
    df2 = df2[(df2>lowlim) & (df2<highlim)].dropna(axis=0,how='all').dropna(axis=1,how='all')

    if df2.empty:
        return (None,None)
    keys = df2.keys()
    
    #df_in = df_in.applymap(int)
    keylist=[]
    UWIlist = pd.Series(data=None,dtype = int)

    knum = None
    UWI = None
    
    for k in keys:        
        # check for GT 100 rows per value
        if df2[k].shape[0]/len(df2[k].unique()) > 100:
            keylist.append(k)
        UWIlist = UWIlist.append(pd.Series((df2[k].dropna().unique()).tolist()),ignore_index=True)

    if len(keylist) > 0:
        longest = 0
        fav_k = keylist[0]
        
        for k in keylist:
            test = False
            # check for API/UWI key
            if any(x.upper() in k.upper() for x in APIterms):
                test=True           
            # confirm numbers are > 10 and less than 14 digits
            df2[k] = df2[k].apply(str2num)
            if (df2.loc[(df2[k]<highlim) & (df2[k]>lowlim),k].dropna().shape[0] > longest):
                fav_k = k
                longest = df2.loc[(df2[k]<highlim) & (df2[k]>lowlim),k].dropna().shape[0]
        knum = df_inAPI.keys().get_loc(fav_k)

    if UWIlist.empty == False:
        UWIlist = np.array([API2INT(xi,length=14) for xi in UWIlist])
        #API2INT(i,length=14)
        #UWIlist = UWIlist.apply(str2num).apply(API2INT,length=14)
        UWI = np.unique(UWIlist)
        if len(UWI)>1:
            return (None,None)
            #raise Exception('Found more than one UWI in Find_API_Col')
        UWI = int(UWI)
        
    return(UWI,knum)

def APIfromFilename(ffile,UWIlen=10):
    lst = re.findall(r'UWI[0-9]{9,}',ffile, re.I)
    if len(lst)==0:
        lst = re.findall(r'[0-9]{9,}',ffile)
    else:
        lst[0] = re.sub('UWI','',lst[0],re.I)
    if len(lst)>0:        
        UWI = API2INT(lst[0],length=UWIlen)
    else:
        UWI = None
    return UWI

def APIfromFrame(df_in):
    terms = list()
    if isinstance(df_in,dict):
        for k in df_in:
            terms.append(APIfromString(df_in[k].to_string(),ListResult=True))
        term = max(set(terms), key = terms.count)
    if isinstance(df_in,pd.DataFrame):
        term = APIfromString(df_in.to_string())
    return term

def APIfromString(STRING,ListResult=False,BlockT2=False):
    STRING = re.sub(r'[-−﹣−–—−]','-',STRING)
    t1 = re.compile(r'(?:UWI|API)\s*(?:[#:])*\s*([0-9\-\s]*)',re.I)
    t2 = re.compile(r'[0-9]{1,2}(?:\-)[0-9]{3}(?:\-)[0-9]{5,9}\-{0,1}[0-9]{0,4}')
    term = None
    try:
        terms = re.findall(t1,STRING)
        terms = list(map(str.strip, terms))
        term = max(set(terms), key = terms.count)
        if term == '':
            term = None
    except:
        pass
    
    if BlockT2 == False and term == None:
        try:
            terms = re.findall(t2,STRING)
            terms = list(map(str.strip, terms))
        except:
            pass
    if ListResult==False:
        try:
            term = max(set(terms), key = terms.count)
        except:
            pass
    return(term)
    
def COGCC_SURVEY_CLEANUP(df_in):#if True:
    skeys = df_in.keys().str.upper().str.strip().str.replace(' ','')
    mask = skeys.str.contains('HEADER')
    skey = list(df_in.keys()[mask])
    # Look for normal Col 1 template terms
    if len(skey) == 1:
        df_dummy = df_in[skey[0]].str.upper().str.strip().str.replace(' ','')
        m1 = df_dummy.str.contains('OPERATORNAME').fillna(False)
        m2 = df_dummy.str.contains('OPERATORNUMBER').fillna(False)
        m3 = df_dummy.str.contains('NORTHREFERENCE').fillna(False)

        # If all strings above are found, rename columes
        if df_in.loc[m1 + m2 + m3,skey[0]].size == 3:
            col1 = df_in.keys().get_loc(skey[0])
            default_keys = ['measured depth\n(ft)', 'inclination (°)', 'azimuth (°)', 'true vertical depth\n(ft)', 'northing \n+N/-S  (ft)', 'easting \n+E/-W  (ft)']
            
            ncol = len(skey) + len(default_keys)

            newkeys = pd.Series(df_in.keys())
            newkeys.iloc[col1:ncol]= skey+default_keys
            newkeys = list(newkeys)

            df_in.columns = newkeys
            return df_in
    else:
        return None 

def ExtractSurveyWrapper(df_in):
    OUT=pd.DataFrame()
    adf = df_in.copy()
    try:
        OUT = ExtractSurvey(adf)
        OUT = pd.DataFrame(OUT)
        if not OUT.empty:
            OUT.rename(columns = SurveyCols(OUT),inplace=True)
    except:
        try:
            df_in = COGCC_SURVEY_CLEANUP(adf)
            df_in = pd.DataFrame(df_in)
            if not df_in.empty:
                raise Exception('No survey found in dataframe')
            else:
                OUT = ExtractSurvey(df_in)
                OUT.rename(columns = SurveyCols(OUT),inplace=True)
            return outdf_in
        except:
            raise Exception('No survey found in dataframe')
    if not isinstance(OUT,pd.DataFrame):
        raise Exception('No survey found in dataframe')

    if OUT is None:
        OUT = pd.DataFrame()
    return OUT


def ExtractSurvey(df_in):
    outdf_in = pd.DataFrame()
    ReadUWI = APIfromFrame(df_in)

    adf_in=df_in.copy(deep=True)

    try: 
        SurveyCols(df_in) # is first row survey header?
        if df_in[list(SurveyCols(df_in))].dropna(how='all',axis = 1).dropna(how='all',axis = 0).shape[0]>5:
            cols = list(SurveyCols(df_in))
            outdf_in = df_in[cols].copy(deep=True)
                    
            outdf_in['UWI'] = ReadUWI
##                (DEAD,APICOL) = Find_API_Col(df_in)
##                if APICOL != None:
##                    cols = cols + [df_in.keys()[APICOL]]
##                    outdf_in = df_in[cols].copy(deep=True)
##                    outdf_in.rename(columns ={df_in.keys()[APICOL]:'UWI'},inplace=True)
##                else:
##                    outdf_in = df_in[cols].copy(deep=True)
##                    if (ReadUWI != None):
##                        outdf_in['UWI'] = ReadUWI
                
            outdf_in = outdf_in.applymap(str2num)
            outdf_in = outdf_in.dropna(how='all',axis = 1)
            outdf_in = outdf_in.dropna(how='all',axis = 0)
            if ('UWI' in outdf_in.keys()) == False:
                outdf_in['UWI']=None
            #outdf_in.rename(columns = SurveyCols(outdf_in),inplace=True)
            return outdf_in
    except:
        # test for strings
        test = re.findall(r'MD|TVD|DEPTH|INC|AZ',adf_in.to_string(),re.I)
        if len(test)>=3:
            for n in [1,2,3,4]:
                drow = -1
                for i in range(0,100): # is survey header in 1st 15 rows?
                    try:
                        df_in=adf_in.copy()
                        N=min(i,n)
                        concat_vals = df_in.iloc[i:i+N,:].apply(lambda row:'_'.join(row.values.astype(str)),axis=0)
                        #SurveyCols(concat_vals)

                        df_in = df_in.iloc[i+N:,:]
                        df_in.columns = concat_vals
                        cols = list(SurveyCols(df_in))

                        df_in.reset_index(drop=True, inplace= True)

                        outdf_in = df_in[cols].copy(deep=True)
                        outdf_in.rename(columns ={df_in.keys()[APICOL]:'UWI'},inplace=True)

                        outdf_in['UWI'] = ReadUWI
    ##                    
    ##                    (DEAD,APICOL) = Find_API_Col(df_in)
    ##                    if APICOL != None:
    ##                        cols = cols + [df_in.keys()[APICOL]]
    ##                        outdf_in = df_in[cols].copy(deep=True)
    ##                        outdf_in.rename(columns ={df_in.keys()[APICOL]:'UWI'},inplace=True)
    ##                    else:
    ##                        outdf_in = df_in[cols].copy(deep=True)
    ##                        if ReadUWI != None:
    ##                            outdf_in['UWI'] = ReadUWI

                        # WAS GOING TO BUILD A CHECK THAT LAST ROW IN KEY COLUMNS DOES NOT CONTAIN VALUES
                        #keycols = list()
                        #for c in cols:
                        #    keycols.append(outdf_in.keys().get_loc(c))
                        
                        #outdf_in = outdf_in.copy(deep=True)
                        outdf_in = outdf_in.applymap(str2num)

                        #.apply(pd.to_numeric,errors='coerce').dropna(axis=0,how='any').shape[0]
                        test = outdf_in.loc[:10,cols].dropna(how='any').shape[0]
                        if test<10:
                            continue

                        #for k in outdf_in.keys():
                            # GETTING SLICE ERROR HERE
                            #outdf_in.loc[:,k] = np.array(outdf_in.loc[:,k].astype(str).str.replace(r'[^0-9\.]*','',regex=True))
                            # GETTING SLICE ERROR HERE
                            #outdf_in.loc[:,k] = np.array(pd.to_numeric(outdf_in.loc[:,k], errors='coerce'))
                        outdf_in = outdf_in.apply(pd.to_numeric, errors='coerce')
                        outdf_in = outdf_in.dropna(how='all',axis = 1)
                        outdf_in = outdf_in.dropna(how='all',axis = 0)
                        if outdf_in.shape[0] > 5:
                            outdf_in = df_in[SurveyCols(df_in)]
                            outdf_in.rename(columns = SurveyCols(outdf_in),inplace=True)
                            if ('UWI' in outdf_in.keys()) == False:
                                outdf_in['UWI'] = None
                            return outdf_in
                    except: pass

def ERRORFILES(ERR_FOLDER='ERROR_FILES'):
    if not path.isdir(ERR_FOLDER):
        makedirs(ERR_FOLDER)
    return(ERR_FOLDER)

def CheckUWI(df_in):
    cols = SurveyCols(df_in)
    
    outdf.UWI = outdf.UWI.apply(str2num)

def survey_from_excel(file, ERRORS = True): #if True:
    ERR_FOLDER = None
    RUNERROR = False
    if ERRORS == True:
        ERR_FOLDER = ERRORFILES()

    outdf = pd.DataFrame()
    xl = {}
    # read excel as a dictionary of dataframes
    try:
        xl = pd.read_excel(file, sheet_name = None, engine = None)
    except:
        try:
            xl = pd.read_excel(file, sheet_name = None ,engine = 'openpyxl')
        except:
            print(file+': ERROR')
            RUNERROR = True
        
    if len(xl)==0:
        print('FILE XL READ ERROR IN: '+ file)
        outdf = 'FILE XL READ ERROR IN: '+ file
        if ERRORS == True:
            shutil.move(file, ERR_FOLDER)
        RUNERROR = True
        return None   
        
    READUWI = APIfromFrame(xl)
    FILENAMEUWI =  APIfromString(file,BlockT2 = True)
    READUWI2 = API2INT(READUWI)
    FILENAMEUWI2 = API2INT(FILENAMEUWI)
    if FILENAMEUWI2==READUWI2:
        UWI = READUWI
    l = [str(FILENAMEUWI),str(READUWI),str(None)]
    while 'None' in l:
        l.remove('None')
    if len(l)==1:
        UWI=l[0]
    else:
        UWI = None        
        
    if isinstance(xl,dict): # test if file read delivered a dictionary
        for k in xl.keys(): # for each sheet  #if True:
            df_s = xl[k].copy(deep=True)
            df_s = df_s.dropna(how='all',axis=0).dropna(how='all',axis=1)

            ext_df=pd.DataFrame()

            try:
                ext_df = (df_s)
                ext_df = pd.DataFrame(ext_df)
            except:
                continue

            #for kkey in SurveyCols().keys():
            #    list(SurveyCols(df_s).values())

            ext_df = ExtractSurveyWrapper(ext_df)

            ext_df

            if len(list((ext_df).values)) > 5:
                outdf = ext_df
            #else:
            #    UWI = set(list(outdf.UWI.apply(str2num)))
                
            #outdf = pd.concat([outdf,ext_df],axis=1,ignore_index=False)
            
            #print(ext_df.keys())
            
        if 'UWI' in outdf.keys():
            outdf['UWI'] = outdf.UWI.apply(str2num)
        else:
            outdf['UWI'] = UWI
            
            #pd.to_numeric(outdf.UWI, errors='coerce').dropna().to_list()
            #print('outdf dict done')
            #print(outdf)
            #if isinstance(outdf,pd.DataFrame):
            #    return outdf

    if isinstance(xl,pd.DataFrame):
        try:
            outdf = ExtractSurveyWrapper(xl)
            outdf = pd.DataFrame(outdf)
            if not 'UWI' in  outdf.keys():
                outdf['UWI'] = None
        except:
            pass
    
    if isinstance(outdf,pd.DataFrame):
        if outdf.empty:
            return None
        if not(UWI is None):            
            outdf['UWI'] = API2INT(UWI)
            outdf = outdf.applymap(str2num)
            outdf = outdf.apply(pd.to_numeric, errors='coerce')
            #outdf = outdf.loc[outdf.T.sum().index,:]
            outdf = outdf.dropna(thresh=3,axis=0)
    return outdf
             
def SurveyCols(df_s_in=None):
    
    
    sterms = {'MD':r'.*MEASURED.*DEPTH.*|.*MD.*|^\s*DEPTH\s*|(?:^|_)DEPTH(?:$|_)',
             'INC':r'.*INC.*|.*DIP.*',
             'AZI':r'.*AZI.*|.*AZM.*',
             'TVD':r'.*TVD.*|.*TRUE.*|.*VERTICAL.*DEPTH.*',
             'NORTH_Y':r'.*\+N.*|.*(?:\+){0,1}N(?:\+){0,1}(?:[\/\\]){0,1}(?:\-){0,1}S(?:\-){0,1}.*FT.*|.*N\+.*|^\s*N(?:[\/\\]){0,1}S\s*|.*NORTH(?!ING).*|(?:^|_)(?:\+){0,1}N(?:\+){0,1}(?:[\/\\]){0,1}(?:\-){0,1}S(?:\-){0,1}(?:$|_)',
             'EAST_X':r'.*\+E.*|.*(?:\+){0,1}E(?:\+){0,1}(?:[\/\\]){0,1}(?:\-){0,1}W(?:\-){0,1}.*FT.*|.*E\+.*|^\s*E(?:[\/\\]){0,1}W\s*|.*EAST(?!ING).*|(?:^|_)(?:\+){0,1}E(?:\+){0,1}(?:[\/\\]){0,1}(?:\-){0,1}W(?:\-){0,1}(?:$|_)'
        
           #  'NORTH_Y':r'.*ORTH.*|.*\+N.*|.*NS.*FT.*|.*N/S*',
           #  'EAST_X':r'.*EAST.*|.*\+E.*|.*EW.*FT.*|.*E/W.*'
        }
    if df_s_in is None:
        return(sterms)
    
    if isinstance(df_s_in,pd.Series):
        df_s_in=list(df_s_in)
    #if isinstance(df_s_in,pd.DataFrame):
    #    df_s_in=list(df_s_in.keys())
    for s in sterms:
        #print(sterms[s])
        if isinstance(df_s_in,pd.DataFrame):
            term = df_s_in.iloc[0,df_s_in.keys().str.contains(sterms[s], regex=True, case=False,na=False)].keys()
            if not isinstance(term, str) and len(term)>0:
                term = term[0]
            if len(term)>0:
                sterms[s] = term
            else:
                sterms[s] = None
            #sterms[s]=df_s_in.iloc[0,df_s_in.keys().str.contains(sterms[s], regex=True, case=False,na=False)].keys()[0]
            
            #sterms[s] = term
        if isinstance(df_s_in,list):
            sterms[s]= list(filter(re.compile('(?i)'+sterms[s]).match,df_s_in))[0]

    # sterms=dict((v, k) for k, v in sterms.iteritems())
    sterms = {v: k for k, v in sterms.items()}
    if None in list(sterms):
        raise Exception('Incomplete Survey Headers Found')
    return sterms

def Wrapped_Survey_Join(aa):
    return Survey_Join(*aa)

def SurveyCols_row(r_in):
    try: OUTPUT = SurveyCols(r_in.to_list())
    except: OUTPUT = None
    return OUTPUT

#def Survey_Join(FLIST:[list,str,int,pd.Series,pd.DataFrame,np.array], SAVEFILE:str):
def Survey_Join(SAVEFILE, FLIST, ERRORS = True): #if True:
    if SAVEFILE != None:
        SAVEFILE = re.sub(re.compile(r'\.[^.]+$'),'',SAVEFILE)
    if ERRORS == True:
        ERR_FOLDER = ERRORFILES()
        ERR_FILES = list()
    pd.set_option('mode.chained_assignment', None)
    # if 1==1:
    df=pd.DataFrame()
    if isinstance(FLIST,(str,int)):
        FLIST=[FLIST]
    if isinstance(FLIST,(np.ndarray,pd.DataFrame,pd.Series)):
        FLIST=list(FLIST)
    #ERROR_LIST = list()
        
    for ffile in FLIST:#if 1==1:
        try:
            #print(ffile)
            if ffile.lower().endswith(('.xls','xlsx','xlsm')):

                #if 1==1:
                rdf = survey_from_excel(ffile)
                if not isinstance(rdf,pd.DataFrame):
                    print('NO SURVEY FOUND IN: '+ ffile)
                    if ERRORS == True:
                        shutil.move(ffile, ERR_FOLDER)
                    continue

                if rdf.empty:
                    print('NO SURVEY FOUND IN: '+ ffile)
                    if ERRORS == True:
                        shutil.move(ffile, ERR_FOLDER)
                    continue                    
          
                #standardize column names
                rdf=rdf.rename(columns=SurveyCols(rdf))
                
                

                # all columns to numeric while catching decimals in strings
                rdf = rdf.applymap(str2num)

                if rdf.shape[0]==0 or rdf.empty:
                    continue

                rdf['FILE']=ffile
                
                rdf.UWI = pd.Series([API2INT(xi,length=14) for xi in rdf.UWI])
                
                #rdf.UWI = rdf.UWI.apply(API2INT,length=14)

                if rdf.UWI.dropna().empty:
                    rdf.loc[:,'UWI'] = APIfromFilename(ffile,UWIlen=14)
                    
                #rdf.UWI = rdf.UWI.astype(str).str.zfill(14)

                #print(rdf)
                
                if df.empty:
                    df=rdf
                    continue           
                #df.columns=rdf.columns
                try:
                    df=pd.concat([df, rdf], ignore_index=True)
                except:
                    print('ERROR IN: '+ ffile)
                    if ERRORS == True:
                        #specify full path to force overwrite
                        shutil.move(path.join(adir,ffile), path.join(ERR_FOLDER,ffile))
                    #ERROR_LIST.append(ffile)
        except OSError as err:
            print(err)
            print('GENERAL ERROR IN: '+ ffile)
            
    #df.loc[pd.to_numeric(df.iloc[:,2],errors='ignore').dropna().index,:].to_csv('JOINED_SURVEYS.csv')
    #df.loc[pd.to_numeric(df.iloc[:,2],errors='ignore').dropna().index,:].
    df = df.dropna(axis = 0, how='all').dropna(axis=1,how='all')
    if SAVEFILE != None:
    #    if path.exists(SAVEFILE+'.JSON'):
    #        df2 = pd.read_cjson(SAVEFILE+'.JSON', ignore)
    #        #df.to_csv(SAVEFILE+'.CSV', mode = 'a', header = False, index=False)
    #        df.to_json(SAVEFILE+'.JSON')
    #    else:
    #        df.to_csv(SAVEFILE+'.CSV', header = True, index = False)
    #        df.to_json(SAVEFILE,+'.JSON')
        df.to_json(SAVEFILE+'.JSON')
    return df

def DF_UNSTRING(df_IN):
    df_IN=df_IN.copy()
    df_IN=df_IN.replace('',np.nan)
    #df_IN=df_IN.loc[:,~df_IN.columns.duplicated()]

    #DATES
    DATECOLS = [col for col in df_IN.columns if 'DATE' in col.upper()]

    #for k in df_IN.keys():
    #print(k+" :: "+str(df_IN[k].dropna().iloc[0]))

    pattern = re.compile(r'[0-9]{4}[-_:/\\ ][0-9]{2}[-_:/\\ ][0-9]{2}[-_:/\\ ]*[0-9]{0,2}[-_:/\\ ]*[0-9]{0,2}[-_:/\\ ]*[0-9]{0,2}[-_:/\\ ]*')

    for k in df_IN.keys():
        #check if just strings
        if df_IN[k].astype(str).str.replace(r'[0-9\._\-\/\\]','',regex=True).str.len().mean() > 5:
            continue

        mask = df_IN[k].astype(str).str.count(pattern)>0
        mask = ~df_IN[k].isna() & mask

        if df_IN.loc[mask,k].count() == df_IN.loc[mask,k].count() & mask.sum()>0:
            DATECOLS.append(k)
    DATECOLS = list(set(DATECOLS))


    for k in DATECOLS:
        # common date problems
        # "REPORTED: "
        #  Prior to rule 205A.b.(2)(A)
        #pattern = re.compile(r'.*Prior to rule.*|reported:',re.I)

        #if pd.to_datetime(df_IN[k].str.replace(pattern,'').fillna(np.nan),errors='coerce').count() != df_IN[k].str.replace(pattern,'').count():
        #    df_IN.loc[pd.to_datetime(df_IN[k].str.replace(pattern,'').fillna(np.nan),errors='coerce').isna() != df_IN[k].str.replace(pattern,'').isna()][k].str.replace(pattern,'')
        #    DATECOLS.remove(k)
        #else:
        #    df_IN[k]=pd.to_datetime(df_IN[k].fillna(np.nan),errors='coerce')
        df_IN[k] = pd.to_datetime(df_IN[k],errors='coerce')

    #FLOATS
    FLOAT_MASK = (df_IN.apply(pd.to_numeric, downcast = 'float', errors = 'coerce').count() - df_IN.count())==0
    FLOAT_KEYS = df_IN.keys()[FLOAT_MASK]

    #INTEGERS
    INT_MASK = (df_IN[FLOAT_KEYS].apply(pd.to_numeric, downcast = 'float', errors = 'coerce').fillna(0.0).apply(np.floor)-df_IN[FLOAT_KEYS].apply(pd.to_numeric, downcast = 'float', errors = 'coerce')==0).fillna(0.0).max()
    #INT_MASK = (df_IN.apply(pd.to_numeric, downcast = 'integer', errors = 'coerce') - df_IN.apply(pd.to_numeric, downcast = 'float', errors = 'coerce') == 0).max()
    INT_KEYS = df_IN[FLOAT_KEYS].keys()[INT_MASK]

    #xx=(df_IN.apply(pd.to_numeric, downcast = 'integer', errors = 'coerce') - df_IN.apply(pd.to_numeric, downcast = 'float', errors = 'coerce') == 0)
    #for k in  df_IN.keys():
    #    df_IN.loc[xx[k]==False,k]

    #Force Unique Key Lists
    FLOAT_KEYS = list(set(FLOAT_KEYS)-set(INT_KEYS) - set(DATECOLS))
    INT_KEYS = list(set(INT_KEYS) - set(DATECOLS))

    df_IN[FLOAT_KEYS] = df_IN[FLOAT_KEYS].apply(pd.to_numeric, downcast = 'float', errors = 'coerce')
    df_IN[INT_KEYS] = df_IN[INT_KEYS].apply(pd.to_numeric, downcast = 'integer', errors = 'coerce')

    #Clean Up Strings?
    #df_IN.keys()[df_IN.dtypes=='O']

    return(df_IN)

if __name__ == "__main__":
    pathname = path.dirname(sys.argv[0])
    adir = path.abspath(pathname)

    df=pd.DataFrame()
    df1 = pd.DataFrame()
    JOINEDFILE = 'JOINED_SURVEY_FILE_V2_MERGE'
    
    
    #for file in listdir(pathname):
    #    if file.lower().endswith(('.json')) and ('surveys' in file.lower()):
    #        if df.empty:
    #            df = pd.read_json(file)
    #        else:
    #            df = pd.concat([df, pd.read_json(file)],axis=0,join='outer',ignore_index=True)
    #            df = df.drop_duplicates()

    PAT = re.compile(r'joined(?:[ -_]*)survey',re.I)

    OUTFILE = None #'JOINED_SURVEYS_XXXX.csv'    
    FLIST=list()
    for file in listdir(pathname):
        if file.lower().endswith(('.xls','xlsx','xlsm')):
            FLIST.append(file)

    print(str(len(FLIST))+' FILES CONSIDERED')

    for file in listdir(pathname):
        if file.lower().endswith(('.parquet')) and PAT.search(file) and ('abs' not in file.lower()) and ('xyz' not in file.lower())and ('3d' not in file.lower()):
            try:
                print(file)
                if df1.empty:
                    df1 = pd.read_parquet(file)
                else:
                    df1 = pd.concat([df1, pd.read_parquet(file)],axis=0,join='outer',ignore_index=True)
                    df1 = df1.drop_duplicates()
            except:
                print("ERROR IN FILE: " + str(file))
                
    #CLEAN UP API/UWI
    df1['UWI'] = df1.UWI.apply(API2INT,length=14)
    m1 = df1.UWI.isna()
    df1.loc[m1,'UWI'] = df1.loc[m1,'API'].apply(API2INT,length=14)
    
    
    if not df1.empty:
        df1 = df1.drop_duplicates()
        FLIST = pd.Series(FLIST)
        mask = FLIST.isin(df1.FILE)
        FLIST = FLIST.loc[~mask]
        FLIST = FLIST.drop_duplicates()
        FLIST = list(FLIST)
    
    #if path.exists(JOINEDFILE+'.PARQUET'):
    #    #df1 = pd.read_json('JOINED_SURVEY_FILE_V2'+'.JSON')
    #    df1 = pd.read_parquet('JOINED_SURVEY_FILE_V2'+'.PARQUET')
    #    df1.to_parquet(JOINEDFILE+'_'+datetime.now().strftime('%Y%M%d')+'.PARQUET')
    #    df1 = pd.concat([df1,df],axis=0,join='outer',ignore_index=True)
    #    df1 = df1.drop_duplicates()
    #    #df1.to_json(JOINEDFILE+'.JSON')
    #    df1.to_parquet(JOINEDFILE+'.PARQUET')
    #    FLIST = pd.Series(FLIST)
    #    mask = FLIST.isin(df1.FILE)
    #    FLIST = FLIST.loc[~mask]
    #    FLIST = FLIST.drop_duplicates()
    #    FLIST = list(FLIST)

    #initialize multiprocessing
    
    #if path.exists(JOINEDFILE):
    #    PreUsedFiles = pd.read_csv(JOINEDFILE,usecols = ['FILE'],squeeze=True)
    #    PreUsedFiles=PreUsedFiles.unique().tolist()
    #    FLIST2 = [x for x in FLIST if x not in PreUsedFiles]
    #    FLIST = FLIST2
    #    del FLIST2
    chunkmin = 200
    splits = max(floor(len(FLIST)/chunkmin),1)
    processors = multiprocessing.cpu_count()
    # if processor batches are nearly greater than min batch size
    if (len(FLIST)/processors/1.3) > chunkmin:
        splits = processors/1.3
    # if processor batches nearly > min batch size
    if (len(FLIST)/processors) > chunkmin:
        splits = processors
    splits = np.floor(splits)
    processors = min(processors,splits)
    data = np.array_split(FLIST,splits)
    del df
    
    func = partial(Survey_Join, None)

    #for f in FLIST:
    #    func(f)

    print(processors)
    if processors > 1:
        print('Concurrent Futures Starting')
        with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
            f = {executor.submit(func, a): a for a in data}
        RESULT=pd.DataFrame()
        for i in f.keys():
            RESULT=pd.concat([RESULT,i.result()],axis=0,join='outer',ignore_index=True)
    else:
        RESULT=Survey_Join(None,FLIST)
        
    if not(df1.empty):
        RESULT = pd.concat([RESULT,df1],axis=0,join='outer',ignore_index=True)


    R = RESULT.copy()

    RESULT = DF_UNSTRING(RESULT)
    RESULT.UWI = RESULT.UWI.apply(API2INT,args = (14,))
    RESULT = RESULT.drop_duplicates()

    m = RESULT.UWI.isna() * RESULT.API.notnull()
    RESULT.loc[m,'UWI'] = RESULT.loc[m,'API'].apply(API2INT,args = (14,))

    m = RESULT.UWI.isna()
    RESULT.loc[m,'UWI'] = RESULT.loc[m,'FILE'].apply(APIfromString,args = (True,)).fillna(np.nan)   

    RESULT = DF_UNSTRING(RESULT)
    RESULT['API'] = RESULT['API'].fillna('')
    
    RESULT.to_json(JOINEDFILE+'_'+datetime.now().strftime('%Y%M%d')+'.JSON')
    RESULT.to_parquet(JOINEDFILE+'_'+datetime.now().strftime('%Y%M%d')+'.PARQUET')
    #RESULT.to_csv(JOINEDFILE+'.csv',mode='a',header = False, index=False)
    #RESULT.to_json(JOINEDFILE+'.JSON')
    
    quit()
    
    m= RESULT[['MD','INC','AZI','TVD','NORTH_Y','EAST_X']].dropna(axis=0,how='any').index
    RESULT=RESULT.loc[m,:]

    RESULT = DF_UNSTRING(RESULT)    

    m = (RESULT.EAST_X**2+RESULT.NORTH_Y**2)**0.5>(4*5280)
    RESULT.loc[m,['FILE']].drop_duplicates()



    
    
##    if path.exists(JOINEDFILE+'.JSON'):
##        RESULT_OLD = pd.read_json(JOINEDFILE+'.json')
##        mask = ~RESULT_OLD.FILE.isin(RESULT.FILE)
##        RESULT = pd.concat([RESULT_OLD.loc[mask],RESULT],axis=0,join='outer',ignore_index=True)
##        RESULT.to_csv(JOINEDFILE+'.csv',mode='a',header = False, index=False)
##        RESULT.to_json(JOINEDFILE+'.JSON')
##    elif path.exists(JOINEDFILE+'.CSV'):
##        RESULT_OLD = pd.read_csv(JOINEDFILE+'.csv',squeeze=True)
##        mask = ~RESULT_OLD.FILE.isin(RESULT.FILE)
##        RESULT = pd.concat([RESULT_OLD.loc[mask],RESULT],axis=0,join='outer',ignore_index=True)
##        #RESULT.to_csv(JOINEDFILE+'.csv',mode='a',header = False, index=False)
##        RESULT.to_json
##    else:
##        #RESULT.to_csv(JOINEDFILE+'.csv',mode='a',header = False, index=False)
##        RESULT.to_json(JOINEDFILE+'.JSON')
                      

# validate API and overwrite with filname API if bad
#df.loc[df.UWI>9*10**12,'UWI']=df.loc[df.UWI>9*10**12,'FILE'].apply(APIfromFilename,UWIlen=14)
#df.loc[df.UWI<5*10**12,'UWI']=df.loc[df.UWI<5*10**11,'FILE'].apply(APIfromFilename,UWIlen=14)
#df.loc[df.UWI.isna(),'UWI']=df.loc[df.UWI.isna(),'FILE'].apply(APIfromFilename,UWIlen=14)



#ERROR IN: SURVEYDATA_2018_05_15_05123437820000.xlsx
#ERROR IN: SURVEYDATA_2018_09_27_05123435970000.xlsx
## ff
## fl

#for i in np.arange(0,len(FLIST),math.ceil(len(FLIST)/100)):
#    func(FLIST[i])

