from tabula.io import read_pdf #pip install tabula-py
from os import path, listdir
import pandas as pd
import re, sys
from contextlib import suppress

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

def main():    
    skeys = ['MD','INC','TVD']

    pathname = path.dirname(sys.argv[0])

    pdf_list = []
    for file in listdir(pathname):
        if file.lower().endswith(".pdf"):
            pdf_list.append(file)

    SUMMARY = pd.DataFrame()
    for f in pdf_list:
        print(f)
        with suppress(Exception):
            df = read_pdf(f,pages='all')

        rtxt = r'[0-9]{2}[-]*[0-9]{3}[-]*[0-9]{5}'

        try:
            r=GetKeyRow(df[0],[rtxt],True)
            lst = (' '.join(df[0].iloc[r[0]].astype(str).to_list())).split(' ')
            r = re.compile(rtxt)
            API = list(filter(r.match, lst))[0]
        except:
            API = None

        for idf in df:
            x=idf.copy(deep=True)
            if idf.empty:
                continue
            r = GetKeyRow(idf,skeys)

            if r == None:
                continue
            r=r[0]
            idf = idf.loc[r:,:]
            idf = CondenseSurveyCols(idf)

            if idf.empty:
                continue

            idf['API'] = API
            idf['File'] = f


            SUMMARY=pd.concat([SUMMARY,idf],axis = 0, join='outer',ignore_index=True)
            SUMMARY.sort_values('MD',ignore_index=True).dropna(how='all',axis=0)

            #SUMMARY['API']=API

    SUMMARY.to_excel('Summary.xlsx')


