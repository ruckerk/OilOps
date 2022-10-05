# requires xlrd, openpyxl
from adjustText import adjust_text
from bs4 import BeautifulSoup as BS
from functools import partial
from io import StringIO, BytesIO
from math import floor, cos, sin, radians, atan2, degrees,atan2 , sqrt
from os import path, listdir, remove, makedirs, walk, mkdir, rename
from requests import Session
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from selenium import webdriver
from selenium.webdriver import Firefox, Chrome
from selenium.webdriver.firefox.options import Options
#from shapely.geometry import Polygon, Point, LineString
from sys import argv
from time import perf_counter, sleep
from tkinter import filedialog
from zipfile import ZipFile
import concurrent.futures
import csv
import datetime
import easygui
from glob import glob
from operator import itemgetter

import matplotlib.ticker as tkr
import matplotlib.pyplot as plt
import seaborn as sns

import multiprocessing
import numpy as np
import pandas as pd
import pylab
import re
import requests
import selenium
import shapefile as shp #pyshp
import sklearn as sk
import sqlalchemy
import sqlite3
import urllib
import warnings
import wget
import magic
import textract
import lasio

from scipy import signal
from scipy import interpolate
from scipy.optimize import curve_fit

import openpyxl
import xlsxwriter
import shutil
from urllib.request import urlopen 
import dateutil.parser

#MAPPING
import shapely
import shapely.wkt
from shapely.ops import unary_union
import pycrs
import pyproj
from pyproj import Transformer, CRS

import collections


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

def read_shapefile(sf):
    # https://towardsdatascience.com/mapping-with-matplotlib-pandas-geopandas-and-basemap-in-python-d11b57ab5dac
    #fetching the headings from the shape file
    fields = [x[0] for x in sf.fields][1:]
    #fetching the records from the shape file
    records = [list(i) for i in sf.records()]
    shps = [s.points for s in sf.shapes()]
    #converting shapefile data into pandas dataframe
    df = pd.DataFrame(columns=fields, data=records)
    #assigning the coordinates
    df = df.assign(coords=shps)
    return df

def IN_TC_AREA(well2,tc2):
    ln = None
    if len(well2.coords)>=2:
        try:
            ln = shapely.geometry.LineString(well2.coords)
        except:
            ln = shapely.geometry.LineString(well2.coords.values)
    elif len(well2.coords)==1:
        ln = shapely.geometry.Point(well2.coords[0])
    if ln == None:
        return(False)
    test = False
    for j in range(0,tc2.shape[0]):
        if test == False:
            poly = shapely.geometry.Polygon(tc2.coords.iloc[j])
            if ln.intersects(poly.buffer(15000)):
                test = True   
    return(test) 

def GROUP_IN_TC_AREA(tc,wells):
    out = pd.DataFrame()
    out['API'] = wells.API_Label.str.replace(r'[^0-9]','',regex=True)
    out['TEST'] = wells.apply(lambda x: IN_TC_AREA(x,tc),axis=1)
    return(out)

def requests_retry_session(
    retries=5,
    backoff_factor=0.4,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def Find_Str_Locs(df_in,string):
    # takes data table and finds index locations for all matches
    if len(string[0])<=1:
        string=[string]
    Output=pd.DataFrame({'Title':[],'Columns':[],'Rows':[]}).astype(object)
    Output.Title=pd.Series([w.replace(' ','_').replace('#','NUMBER').replace('^','') for w in string]).astype(str)
    #Output.astype({'Title': 'object','Columns': 'object','Rows': 'object'}).dtypes
    Output.astype(object)
    ii = -1
    for item in string:
        ii += 1
        Output.iloc[ii,1] = [(lambda x: df_in.index.get_loc(x))(i) for i in df_in.loc[(df_in.select_dtypes(include=[object]).stack().str.contains(f'.*{item}.*', regex=True, case=False,na=False).unstack()==True).any(axis='columns'),:].index.values ]
        Output.iloc[ii,2] = [(lambda x: df_in.index.get_loc(x))(i) for i in df_in.loc[:,(df_in.select_dtypes(include=[object]).stack().str.contains(f'.*{item}.*', regex=True, case=False,na=False).unstack()==True).any(axis='rows')].keys().values]
    Output.Title=pd.Series([w.replace(' ','_').replace('#','NUMBER').replace('^','') for w in string]).astype(str)
    return (Output)

def Summarize_Page(df_in,string):
    #build well as preliminary single row of data
    StringLocs = Find_Str_Locs(df_in,string)
    Summary=pd.DataFrame([],index=range(0,30),columns=StringLocs.Title)
    for item in StringLocs.Title:        
        colrow=StringLocs.loc[StringLocs.Title==item,['Columns','Rows',]].values.tolist()[0]
        itemlist = []
        for c in colrow[0]:
            for r in colrow[1]:
                itemlist.append(df_in.iloc[c,r+1])
            #itemlist=list(dict.fromkeys(itemlist))
        try:
            itemlist=itemlist.remove('')
        except: None
        itemlist=pd.Series(itemlist).dropna().sort_values().tolist()
        Summary.loc[:len(itemlist)-1,item]=itemlist

    Summary=Summary.dropna(axis=0,how='all')
    
    for item in StringLocs.Title:
        if len(Summary[item].dropna())==1:
            if ('LAT/LON' in item.upper()):
                pattern = re.compile('[a-zA-Z:]+')
                Summary.loc[0,item]=pattern.sub('',Summary[item].dropna().values[0])
            if ('DATE' in item.upper()) & (isinstance(Summary.loc[0,item],datetime.date)==False):
                pattern = re.compile('[a-zA-Z:]+')
                Summary.loc[0,item]=pd.to_datetime(pattern.sub('',Summary[item].dropna().values[0]),infer_datetime_format=True,errors='coerce')
            if pd.isna(Summary.loc[0,item]):
                continue
            try:
                Summary.loc[0,item]=Summary[item].dropna().values[0]
            except:
                continue
        elif len(set(Summary[item].dropna()))==1:
            Summary.loc[0,item]=list(set(Summary[item].dropna()))[0]
        elif (len(Summary[item].dropna())>1) & ('DATE' in item.upper()):
            Summary.loc[0,item]=pd.to_datetime(Summary[item],infer_datetime_format=True,errors='coerce').max()
        elif (len(Summary[item].dropna())>1) & ('TOP' in item.upper()):
            Summary.loc[0,item]=pd.to_numeric(Summary[item],errors='coerce').min()
        elif len(Summary[item].dropna())>1:
            Summary.loc[0,item]=pd.to_numeric(Summary[item],errors='coerce').max()
    return(Summary.loc[0,:])

def convert_shapefile(SHP_File,EPSG_OLD=3857,EPSG_NEW=3857,FilterFile=None,Label=''):
    #if 1==1:
    # Define CRS from EPSG reference frame number
    EPSG_OLD= int(EPSG_OLD)
    EPSG_NEW=int(EPSG_NEW)
    
    crs_old = CRS.from_user_input(EPSG_OLD)
    crs_new = CRS.from_user_input(EPSG_NEW)

    #read shapefile
    r = shp.Reader(SHP_File)   # THIS IS IN X Y COORDINATES!!!

    
    #define output filename
    out_fname = re.sub(r'(.*)(\.shp)',r'\1_EPSG'+str(EPSG_NEW)+Label+r'\2',SHP_File,flags=re.IGNORECASE)
    
    #if FilterFile != None:
    #    FILTERUWI=pd.read_csv(FilterFile,header=None,dtype=str).iloc[:,0].str.slice(start=1,stop=10)
    #    pdf = read_shapefile(r)
    #    SHP_APIS = pdf.API_Label.str.replace(r'[^0-9]','').str.slice(start=1,stop=10)
    #    SUBSET = SHP_APIS[SHP_APIS.isin(UWI)].index
    #else:
    #    SUBSET = np.arange(0,len(r.shapes()))

    # Speed, get subset of records
    if FilterFile == None:
        SUBSET=np.arange(0,len(r.shapes()))
    else:
        FILTERUWI=pd.read_csv(FilterFile,header=None,dtype=str).iloc[:,0].str.slice(start=1,stop=10)
        pdf=read_shapefile(r)
        pdf=pdf.API_Label.str.replace(r'[^0-9]','').str.slice(1,10)
        SUBSET=pdf[pdf.isin(FILTERUWI)].index.tolist()

    total = len(SUBSET)
    #compile converted output file    
    with shp.Writer(out_fname, shapeType=r.shapeType) as w:
        w.fields = list(r.fields)
        ct=0
        outpoints = []
        for i in SUBSET:
        #for shaperec in r.iterShapeRecords(): if 1==1:
            ct+=1
            if (floor(ct/20)*20) == ct:
                 print(str(ct)+" of "+str(total))
            shaperec=r.shapeRecord(i)
            Xshaperec=shaperec.shape            
            points = np.array(shaperec.shape.points).T
            
            TFORM = Transformer.from_crs(
                         pyproj.CRS.from_wkt(crs_old.to_ogc_wkt()),
                         pyproj.CRS.from_wkt(crs_old.to_ogc_wkt()),
                         always_xy=True).transform
            
            points_t= TFORM(crs_old, crs_new, points[0],points[1],always_xy=True)
            points[0:2]=points_t[0:2]
            #Xshaperec.points = list(map(tuple, points.T))
            json_shape = shaperec.shape.__geo_interface__
            json_shape['coordinates']=tuple(map(tuple, points.T))
            #outpoints = list(map(tuple, points))
            #Xshaperec.points=list(map(tuple, points))
##            if r.shapeType in [1,11,21]: ## "point" is used for point shapes
##                w.point(tuple(points))
##            if r.shapeType in [8,18,28]: ## "multipoint" is used for multipoint shapes
##                #outpoints=outpoints+(list(map(list, points.T)))
##                w.multipoint(list(map(list, points.T)))
##            if r.shapeType in [3,13,23]: ## "line" for lines
##                #outpoints.append(list(map(list, points.T)))
##                w.line([list(map(list, points.T))])
##            if r.shapeType in [5,15,25]: ## "poly" for polygons
##                #outpoints.append(list(map(list, points.T)))
##                w.poly(list(map(list, points.T)))
##            else: # "null" for null
##                w.null()
            w.record(*shaperec.record)
            w.shape(json_shape)
            #w.shape(Xshaperec)
            #del(Xshaperec)
    prjfile = re.sub(r'\.shp','.prj',out_fname,flags=re.IGNORECASE)
    tocrs = pycrs.parse.from_epsg_code(EPSG_NEW)
    with open(prjfile, "w") as writer:    
        _ = writer.write(tocrs.to_esri_wkt())
    #prjfile = open(re.sub(r'\.shp','.prj',out_fname,flags=re.IGNORECASE),'w')
    #prjfile.write(crs_old.to_wkt().replace(' ','').replace('\n',''))
    #prjfile.close()
       
def get_EPSG():
    msg = "Enter Shapefile EPSG codes"
    title = "Projection Definitions"
    fieldNames = ["Input projection EPSG code","Output projection code"]
    fieldValues = []  # we start with blanks for the values
    fieldValues = easygui.multenterbox(msg,title, fieldNames)
    while 1:
        CHECK,errmsg = check_2EPSG(fieldValues[0],fieldValues[1])
        if CHECK == True: break # no problems found
        fieldValues = easygui.multenterbox(errmsg, title, fieldNames, fieldValues)
    return(fieldValues)

def check_2EPSG(epsg1,epsg2):
    CRS1=CRS2=None
    OUTPUT = True
    MESSAGE = 'Valid EPSG codes'
    try:
        CRS1 = CRS.from_user_input(int(epsg1))
    except: pass
    try:
        CRS2 = CRS.from_user_input(int(epsg2))
    except: pass
    if CRS1==None:
        MESSAGE = 'Invalid input EPSG code'
        OUTPUT = False
    if CRS2 == None:
        MESSAGE = 'Invalid output EPSG code'
        OUTPUT = False
    return(OUTPUT,MESSAGE)

def check_EPSG(epsg1):
    CRS1=None
    OUTPUT = 'Validated EPSG code'
    CHECK = 1
    try:
        CRS1 = CRS.from_user_input(int(epsg1))
    except: pass
    if CRS1==None:
        OUTPUT = 'Invalid'
    return(OUTPUT)

def get_driver():
    ## initialize options
    #options = webdriver.ChromeOptions()
    ## pass in headless argument to options
    #options.add_argument('--headless')
    ## initialize driver
    #driver = webdriver.Chrome('\\\Server5\\Users\\KRucker\\chromedriver.exe',chrome_options=options)

    opts = Options()
    opts.headless = True
    driver = Firefox(options=opts)
    return driver

def listjoin(list_in, sep="_"):
    list_in = list(set(list_in))
    list_in = sorted(list_in)
    list_in = [s.strip() for s in list_in]
    str_out = sep.join(list_in)
    return str_out

def XYtransform(df_in, epsg1 = 4269, epsg2 = 2878):
    #2876
    df_in=df_in.copy()
    transformer = Transformer.from_crs(epsg1, epsg2,always_xy =True)
    df_in[['X','Y']]=df_in.apply(lambda x: transformer.transform(x.iloc[2],x.iloc[1]), axis=1).apply(pd.Series)
    #df_in[['X','Y']]=df_in.apply(lambda x: transform(epsg1,epsg2,x.iloc[2],x.iloc[1],always_xy=True), axis=1).apply(pd.Series)
    return df_in

def CheckDuplicate(fname):
    ct = ''
    while path.exists(fname):
        pattern = re.compile('.*_([0-9]*)\.(?:[0-9a-zA-Z]{3,4})',re.I)
        ct = re.search(pattern, fname)
        try:
            ct = ct.group(1)
            ct = int(ct)
            ct += 1
            ct = '_'+str(ct)
        except:
            ct = '_1'
        
        pattern = re.compile(r'(.*)(_[0-9]{1,4})(\.[a-z0-9]{3,4})',re.I)
        fname = re.sub(pattern,r'\1'+ct+r'\3',fname)
    return(fname)

def STIM_VALS_FROM_TXT(row): 
    TERMS = {'FLUID1':r'([0-9,\.]*) GAL',
             'FLUID2':r'([0-9,\.]*) BBL',
             'PROP`':r'([0-9,\.]*) *(?:#|LBS)',
             'PRESSURE':r'([0-9,\.]*) *PSI'}
    for k in TERMS.keys():
        # if 1==1:
        TERMS[k] = re.findall(TERMS[k],row,re.I)
        TERMS[k] = [re.sub(r',', '', i) for i in TERMS[k]]
        TERMS[k] = [re.sub('^$', '0', i) for i in TERMS[k]]
        TERMS[k] = np.array(TERMS[k])
        if TERMS[k].size == 0:
            TERMS[k] = 0
            continue            
        else:
            TERMS[k] = TERMS[k].astype(np.float)
        if k == 'PRESSURE':
            try:
                TERMS[k] = TERMS[k].max()
            except:
                pass
        else:
            try:
                TERMS[k] = TERMS[k].sum()
            except:
                pass
        if isinstance(TERMS[k],np.ndarray):
            TERMS[k] = 0
    return(TERMS)

def STIM_SUMMARY_TO_ARRAY(row):
    data = STIM_VALS_FROM_TXT(str(row))
    data = list(data.values())
    FLUID = data[0]+data[1]
    if FLUID == 0:
        FLUID = np.nan
    PROP = data[2]
    if PROP ==0:
        PROP = np.nan
    PSI = data[3]
    if PSI == 0:
        PSI = np.nan
        
    OUT = pd.Series([FLUID,PROP,PSI])
    return(OUT)

def TRYDICT(TERM,D):
    try:
        out = D[TERM]
    except:
        out = None
    return out

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
    pathname = path.dirname(argv[0])

    if SUBDIR != None:
        pathname = path.join(pathname, SUBDIR)
        
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

def FirstNumericRow(FILENAME,ROWCOUNT = 100):
    with open(FILENAME) as f:
        for pos, l_num in enumerate(f):
            if pos > ROWCOUNT:
                break
            if re.match(r'[\d\.\s]+',l_num.strip()):
                KEYROW = pos
                break
    return KEYROW

def COUNTER(n,tot,PCT_STEP=10):
    if floor(n/tot*PCT_STEP) == n/tot:
        return n/tot*PCT_STEP
    else:
        return

def FTYPE(FILELIST,COUNTER = False):
    if not isinstance(FILELIST,list):
        FILELIST = [FILELIST]
    ENDCNT = len(FILELIST)
    CNT = 0
    TYPES = list()
    for f in FILELIST:
        CNT+=1
        if COUNTER:
            COUNTER(CNT,ENDCNT,10)
        TYPES.append(magic.from_file(f))
    return(TYPES)

def FILETYPETEST(file_str,filetype_str):
    if not filetype_str.upper() in FTYPE(file_str)[0].upper():
        return False
    else:
        return True
