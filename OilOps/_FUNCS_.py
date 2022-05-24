import pandas as pd
import numpy as np
import re #, datetime , math, 
#from os import path
# requires xlrd, openpyxl

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

def GetKey(df,key):
    # returns list of matches to <key> in <df>.keys() as regex search
    return df.iloc[0,df.keys().str.contains('.*'+key+'.*', regex=True, case=False,na=False)].keys().to_list()

def GetKeyRow(df_in,keys,regexparam = False):
    df_in = df_in.astype(str).apply(' '.join,axis=1)
    for k in keys:
        df_in=df_in.loc[df_in.str.contains(k,case=False,regex=regexparam)]
    if df_in.empty:
        out = None
    else:
        out = df_in.index.to_list()
    return out
  
def SurveyCols(df_in):
    sterms = {'MD':r'.*MEASURED.*DEPTH.*|.*MD.*',
             'INC':r'.*INC.*|.*DIP.*',
             'AZI':r'.*AZI.*|.*AZM.*',
             'TVD':r'.*TVD.*|.*TRUE.*|.*VERTICAL.*DEPTH.*',
             'NORTH_Y':r'.*NORTH.*|.*\+N.*|.*NS.*FT.*|.*N\+.*',
             'EAST_X':r'.*EAST.*|.*\+E.*|.*EW.*FT.*|.*E\+.*'
        }

    if df_in.keys().str.contains(r'XPATH|EAST_X',regex=True,case=False,na=False).max():
        sterms['NORTH_Y'] = r'YPATH|NORTH_Y'
        sterms['EAST_X'] = r'XPATH|EAST_X'

    if isinstance(df_in,pd.Series):
        df_in=list(df_in)
    for s in sterms:
        #print(sterms[s])
        if isinstance(df_in,pd.DataFrame):
            sterms[s]=df_in.iloc[0,df_in.keys().str.contains(sterms[s], regex=True, case=False,na=False)].keys()[0]
        if isinstance(df_in,list):
            sterms[s]= list(filter(re.compile('(?i)'+sterms[s]).match,df_in))[0]
    # sterms=dict((v, k) for k, v in sterms.iteritems())
    #sterms = {v: k for k, v in sterms.items()}
    return sterms
 
def CondenseSurveyCols(df_in):#if 1==1:
    df_in = df_in.astype(str).apply(' '.join,axis=1)
    df_in = df_in.str.split(pat=' ',n=-1,expand=True)
    new_header = df_in.iloc[0]
    df_in = df_in[1:]
    df_in.columns = new_header
    sdict = SurveyCols(df_in)
    df_in = df_in[sdict.values()]
    extra_cols = max(0,df_in.shape[1]-len(sdict)-1)
    df_in = df_in.apply(pd.to_numeric,errors='coerce').dropna(thresh=extra_cols,axis=0).dropna(how='any',axis=1)
    return df_in
