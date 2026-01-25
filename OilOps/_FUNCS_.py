 # requires xlrd, openpyxl
from __future__ import annotations

from dataclasses import dataclass, asdict


from bs4 import BeautifulSoup as BS
from functools import partial
from io import StringIO, BytesIO
from adjustText import adjust_text
from math import floor, cos, sin, tan, radians, atan, atan2, acos, asin, degrees, sqrt, ceil, isnan, pi, log10
import statistics
from os import path, listdir, remove, makedirs, walk, mkdir, rename, getlogin, getcwd, devnull, sep
from requests import Session
from requests.adapters import HTTPAdapter

from requests.packages.urllib3.util.retry import Retry
from selenium import webdriver
from selenium.webdriver import Firefox, Chrome
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import subprocess

#from shapely.geometry import Polygon, Point, LineString
import sys
from sys import argv, exec_prefix
from time import perf_counter, sleep
from tkinter import filedialog
from zipfile import ZipFile, BadZipfile
import zipfile
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

import futures3 as cfutures
import csv
import datetime
import easygui
from glob import glob
from operator import itemgetter
import base64
import itertools
import socket
import platform
import json
import time
import hashlib
import threading

import matplotlib.ticker as tkr
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.patches as mpatches

import seaborn as sns

import multiprocessing
import numpy as np
from numpy.linalg import lstsq

import pandas as pd
import pylab
import re
import requests
import selenium
import shapefile as shp #pyshp

import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.linear_model import LinearRegression, RidgeCV, MultiTaskElasticNetCV, LogisticRegression, HuberRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline   import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import check_random_state
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from sklearn import set_config
set_config(transform_output="pandas")

import sqlalchemy
import sqlite3
import urllib
import warnings
import wget
import magic
#import textract
import lasio
import psutil
import fnmatch
import random

from scipy import signal, stats, interpolate
from scipy.optimize import curve_fit, fmin_cobyla, least_squares
from scipy.stats.mstats import gmean, linregress
from scipy.stats import circmean
from scipy.ndimage import binary_dilation, gaussian_filter1d

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize, lsq_linear

from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

import ruptures as rpt

import openpyxl, xlrd
import xlsxwriter
import shutil
from urllib.request import urlopen 
from urllib.parse import urlparse, unquote, urljoin, parse_qs
from urllib3.util.retry import Retry

from pathlib import Path
import dateutil.parser

#MAPPING
import shapely
import shapely.wkt
from shapely.ops import unary_union
from shapely.geometry import shape, LineString, Point, MultiPoint

import pycrs
import pyproj
from pyproj import Transformer, CRS

import collections
from collections.abc import Iterable
from scipy.special import logsumexp

import base64 

import difflib
import wellpathpy as wp

import pymc as pm

from .WELLAPI import WELLAPI as WELLAPI

#import lightgbm as lgb
#from lightgbm import log_evaluation
#import xgboost as xgb
#from catboost import CatBoostRegressor
#from lightgbm import LGBMRegressor

from typing import Dict, Any, Tuple, List, Optional, Iterable

from tqdm import tqdm, trange

warnings.filterwarnings('ignore')
        
def sigmoid(x, L ,x0, k, b):
    # L: max val
    # x0: midpoint 
    # k: steepness
    # b: min val
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def exponential(x, A, K, C):
    y= A * np.exp(K * x) + C
    return (y)

def powfunc(x, a, c, d):
    return a*(x**(c+d))

def expfunc(x, a, b, c, d):
    return a*np.exp(-c*(x-b))+d

def exp_nct(depth, A, B):
    return A * np.exp(B * depth)
 
def stretch_exponential(x,c,tau,beta,y_offset):
    return c*(np.exp(-(x/tau)**beta))+y_offset

def linear(x,yint,slope):
    y = slope*x+yint
    return (y)
        
def curve_fitter(X,Y, funct, split = 0.2, plot = False, logx = False, logy = False, **modargs):
    m = np.isfinite(X)*np.isfinite(Y)
    if sum(m)<=1:
        return None
    X=X[m].copy()
    Y=Y[m].copy()

    if logx:
        X = np.log10(X)
    if logy:
        Y = np.log10(Y)
    
    if isinstance(split, (int, float)):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= split)
    else:
        X_train, X_test, y_train, y_train = X,X,Y,Y

    if len(modargs)>0:
        func = partial(funct, *ModelArgs)
    else:
        func = funct
        
    try:
        popt, pcov = curve_fit(func, X_train, y_train)
    except:
        return None

    x = np.linspace(min(X),max(X),100)
    y = func(x, *popt)
    if logx:
        x = 10**x
        X = 10**X        
    if logy:
        Y = 10**Y
        y = 10**y

    if plot:        
        plt.plot(X, Y, 'o', label='data')
        plt.plot(x,y, label='fit')
        plt.legend(loc='best')
       
        plt.xlim((0.9*min(X),1.1*max(X)))
        plt.ylim((0.9*min(Y),1.1*max(Y)))
        plt.show()
    return popt


def DF_UNSTRING(
    df_in: pd.DataFrame,
    *,
    sample_size: int = 800,
    date_success: float = 0.70,
    num_success: float = 0.90,
    date_cache: bool = True,
    coerce_empty_to_na: bool = True,
    datetime_to_utc_naive: bool = False,
    convert_bool: bool = True,
    bool_success: float = 0.98,
) -> pd.DataFrame:
    """
    Efficiently convert string/object columns to:
      - datetime64[ns] (dates)
      - integer / float (numbers)
      - boolean (optional)

    Designed for prepping data for JSON export (reduces numbers/dates stored as strings).
    Uses sampling for detection and converts each chosen column at most once.

    Notes for JSON:
      - pandas Timestamps aren't JSON-serializable by default; after this function,
        use df.to_json(date_format='iso') or convert datetimes to strings.
      - If you want datetimes as ISO strings in the dataframe, see the helper below.
    """

    # Precompiled regexes (fast + reused)
    _RE_EMPTY = re.compile(r"^\s*$")
    _RE_DATE_NAME = re.compile(r"(date|dt|time|timestamp)", re.I)
    _RE_DATE_LIKE = re.compile(
        r"""
        (?:\b\d{4}[-/_]\d{1,2}[-/_]\d{1,2}\b) |                 # YYYY-MM-DD (or / or _)
        (?:\b\d{1,2}[-/_]\d{1,2}[-/_]\d{2,4}\b) |               # MM-DD-YYYY (or DD-MM-YYYY)
        (?:\b\d{4}\d{2}\d{2}\b)                                 # YYYYMMDD
        """,
        re.VERBOSE,
    )

    if df_in is None or df_in.empty:
        return df_in

    df = df_in.copy()

    # Only consider object/string columns (leave numeric/datetime/categorical alone)
    obj_cols = df.columns[df.dtypes == "object"]
    if len(obj_cols) == 0:
        return df

    # Clean empty strings -> NA only on object cols (avoid scanning full df)
    if coerce_empty_to_na:
        df[obj_cols] = df[obj_cols].replace(_RE_EMPTY, pd.NA, regex=True)

    for col in obj_cols:
        s = df[col]
        nn = s.dropna()
        if nn.empty:
            continue

        # sample to decide type
        if len(nn) > sample_size:
            samp = nn.sample(sample_size, random_state=0)
        else:
            samp = nn

        # String view once
        ss = samp.astype("string")

        # ------------------------
        # Boolean detection (optional)
        # ------------------------
        if convert_bool:
            # canonical boolean tokens
            # (keep small + cheap)
            lower = ss.str.lower().str.strip()
            is_bool_token = lower.isin(
                ["true", "false", "t", "f", "yes", "no", "y", "n", "1", "0"]
            )
            if is_bool_token.mean() >= bool_success:
                # Full conversion (vectorized)
                full = df[col].astype("string").str.lower().str.strip()
                df[col] = full.map(
                    {
                        "true": True, "t": True, "yes": True, "y": True, "1": True,
                        "false": False, "f": False, "no": False, "n": False, "0": False,
                    }
                ).astype("boolean")
                continue

        # ------------------------
        # Date detection
        # ------------------------
        name_hint = bool(_RE_DATE_NAME.search(str(col)))
        regex_hint = ss.str.contains(_RE_DATE_LIKE, regex=True, na=False).mean() >= 0.20

        if name_hint or regex_hint:
            parsed = pd.to_datetime(samp, errors="coerce", cache=date_cache)
            if parsed.notna().mean() >= date_success:
                dt = pd.to_datetime(s, errors="coerce", cache=date_cache)

                # Optional: normalize timezone-ish datetimes to naive UTC
                # (Helpful when JSON consumers expect no tz)
                if datetime_to_utc_naive:
                    # If tz-aware, convert; if naive, leave
                    try:
                        if getattr(dt.dt, "tz", None) is not None:
                            dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
                    except Exception:
                        pass

                df[col] = dt
                continue

        # ------------------------
        # Numeric detection
        # ------------------------
        # Try numeric parse on sample
        num_samp = pd.to_numeric(samp, errors="coerce")
        if num_samp.notna().mean() >= num_success:
            full = pd.to_numeric(s, errors="coerce")

            # Integer-ness check on non-null values
            full_nn = full.dropna()
            if not full_nn.empty and (full_nn == np.floor(full_nn)).all():
                df[col] = pd.to_numeric(full, downcast="integer")
            else:
                df[col] = pd.to_numeric(full, downcast="float")
            continue

        # else: leave as object/string

    return df


def datetimes_to_iso_strings(
    df_in: pd.DataFrame,
    *,
    cols=None,
    utc_z: bool = False,
) -> pd.DataFrame:
    """
    Convert datetime64[ns] columns to ISO strings for JSON-friendly dataframe.
    If utc_z=True, outputs 'Z' suffix (assumes naive datetimes are UTC).
    """
    df = df_in.copy()
    if cols is None:
        cols = df.columns[pd.api.types.is_datetime64_any_dtype(df.dtypes)]
    for c in cols:
        s = df[c]
        if utc_z:
            # Ensure strings end with Z
            df[c] = s.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str.rstrip("0").str.rstrip(".") + "Z"
        else:
            df[c] = s.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str.rstrip("0").str.rstrip(".")
    return df
    

def run_sm_ols(df_in,YKEY =None):
    if YKEY == None:
        YKEY = df_in.keys()[0]
    XKEYS = df_in.keys().tolist()
    XKEYS.remove(YKEY)
        
    # Run statsmodel ols
    ols_results = sm.OLS(df_in.loc[:,YKEY], df_in.loc[:,XKEYS]).fit()
    
    # Unpack variables
    results = ols_results.params

    labels = [f'b{i}' for i in np.arange(0,len(results))] 
    labels.insert(0,'intercept')
    
    # Return to apply call as a series (3 separate columns)
    return pd.Series(results, index=labels)

    _RE_EMPTY = re.compile(r"^\s*$")
    _RE_DATE_NAME = re.compile(r"(date|dt|time|timestamp)", re.I)
    _RE_DATE_LIKE = re.compile(
        r"""
        (?:\b\d{4}[-/_]\d{1,2}[-/_]\d{1,2}\b) |                 # YYYY-MM-DD (or / or _)
        (?:\b\d{1,2}[-/_]\d{1,2}[-/_]\d{2,4}\b) |               # MM-DD-YYYY (or DD-MM-YYYY)
        (?:\b\d{4}\d{2}\d{2}\b)                                 # YYYYMMDD
        """,
        re.VERBOSE,
    )



def GetKey(df,key):
    # returns list of matches to <key> in <df>.keys() as regex search
    return df.keys()[df.keys().astype(str).str.contains(key, regex=True, case=False,na=False)].tolist()
    
def GetKeyRow(df_in,keys,regexparam = False):
    df_in = df_in.astype(str).apply(' '.join,axis=1)
    for k in keys:
        df_in=df_in.loc[df_in.str.contains(k,case=False,regex=regexparam)]
    if df_in.empty:
        out = None
    else:
        out = df_in.index.to_list()
    return out
  
def REMOVESurveyCols(df_in):
    sterms = {'MD':r'.*MEASURED.*DEPTH.*|.*MD.*',
             'INC':r'.*INC.*|.*DIP.*',
             'AZI':r'.*AZI.*|.*AZM.*',
             'TVD':r'.*TVD.*|.*TRUE.*|.*VERTICAL.*DEPTH.*',
             'NORTH_Y':r'.*NORTH.*|.*\+N.*|.*NS.*FT.*|.*N\+.*',
             'EAST_X':r'.*EAST.*|.*\+E.*|.*EW.*FT.*|.*E\+.*'
        }

    if df_in.keys().str.contains(r'X[ _]*PATH|EAST_X',regex=True,case=False,na=False).max():
        sterms['NORTH_Y'] = r'Y[ _]*PATH|NORTH_Y'
        sterms['EAST_X'] = r'X[ _]*PATH|EAST_X'

    if isinstance(df_in,pd.Series):
        df_in=list(df_in)
    for s in sterms:
        #print(sterms[s])
        if isinstance(df_in,pd.DataFrame):
            terms = df_in.iloc[0,df_in.keys().str.contains(sterms[s], regex=True, case=False,na=False)].keys().tolist()
            if len(terms)==0:
                sterms[s] = None
            else:
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
    # takes data table and finds index locations for all matches  #if True:
    if isinstance(string,str):
        string=[string]
    df_in = df_in.astype(object)
    Output=pd.DataFrame({'Title':[],'Columns':[],'Rows':[]}).astype(object)
    Output.Title=pd.Series([w.replace(' ','_').replace('#','NUMBER').replace('^','') for w in string]).astype(str)
    #Output.astype({'Title': 'object','Columns': 'object','Rows': 'object'}).dtypes
    Output.astype(object)
    for ii, item in enumerate(string):
        Output.at[ii,'Columns'] = [] 
        Output.at[ii,'Rows'] = []
        try:
            rows = [(lambda x: df_in.index.get_loc(x))(i) for i in df_in.loc[(df_in.select_dtypes(include=[object]).stack().str.contains(f'.*{item}.*', regex=True, case=False,na=False).unstack()==True).any(axis='columns').replace(False,np.nan).dropna().index,:].index.values ]
        except:
            continue
        for r in rows:
            #cols = [(lambda x: df_in.loc[r,:].index.get_loc(x))(i) for i in df_in.loc[r,(df_in.select_dtypes(include=[object]).stack().str.contains(f'.*{item}.*', regex=True, case=False,na=False).unstack()==True).any(axis='rows')].keys().values]
            cols = [(lambda x: df_in.loc[r,:].index.get_loc(x))(i) for i in df_in.loc[r,(df_in.astype(str).stack().str.contains(f'.*{item}.*', regex=True, case=False,na=False).unstack()==True).any(axis='rows')].keys().values]
            Output.at[ii,'Columns'] = Output.at[ii,'Columns'] + cols
            Output.at[ii,'Rows'] = Output.at[ii,'Rows'] + [r]*len(cols)
            # Output.at[ii,'Rows'] = [(lambda x: df_in.loc[:.c].index.get_loc(x))(i) for i in df_in.loc[:,(df_in.select_dtypes(include=[object]).stack().str.contains(f'.*{item}.*', regex=True, case=False,na=False).unstack()==True).any(axis='rows')].keys().values]
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
        except: 
            pass
        itemlist = pd.Series(itemlist).dropna().sort_values().tolist()
        itemlist = list(set(itemlist))
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
        elif ('TREAT' in item.upper()) & ('SUMMARY' in item.upper()):
            Summary.loc[0,item]=Summary[item].str.cat(sep=' ')
        elif len(Summary[item].dropna())>1:
            Summary.loc[0,item]=pd.to_numeric(Summary[item],errors='coerce').max()
    return(Summary.loc[0,:])

def convert_shapefile(SHP_File,EPSG_OLD=3857,EPSG_NEW=3857,FilterFile=None,Label=''):
    #if 1==1:
    # Define CRS from EPSG reference frame number
    EPSG_OLD= int(EPSG_OLD)
    EPSG_NEW=int(EPSG_NEW)
    
    #crs_old = CRS.from_user_input(EPSG_OLD)
    #crs_new = CRS.from_user_input(EPSG_NEW)
    crs_old = pycrs.parse.from_epsg_code(EPSG_OLD)
    crs_new = pycrs.parse.from_epsg_code(EPSG_NEW)
    
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
            
            points_t= TFORM(crs_old, crs_new, points[0],points[1])
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

def FullFileScan(FILE):
    L = [exec_prefix]
    ct = -1
    while (ct<2) or (L[0]!= L[min(len(L)-1,1)]):
        ct+=1
        L.insert(0,path.split(L[0])[0])
    OUT = []    
    for root, dirs, files in walk(L[0]):
        for file in files:
            if bool(re.findall(FILE,file,re.I)):
                OUT.append(path.join(root,str(file)))
    return OUT

def _linux_firefox_snap_binary() -> str | None:
    """
    If Firefox is installed via snap on Linux, return the snap firefox binary path if found.
    """
    try:
        # Example: /snap/firefox/current/usr/lib/firefox/firefox
        candidates = subprocess.getoutput("find /snap/firefox -name firefox 2>/dev/null").splitlines()
        return candidates[-1] if candidates else None
    except Exception:
        return None


def _linux_geckodriver_from_snap() -> str | None:
    """
    Try to locate geckodriver inside snap tree (some setups do this).
    """
    try:
        candidates = subprocess.getoutput("find /snap/firefox -name geckodriver 2>/dev/null").splitlines()
        return candidates[-1] if candidates else None
    except Exception:
        return None


def get_driver(
    download_dir: str | os.PathLike,
    *,
    headless: bool = True,
    geckodriver_path: str | None = None,
    firefox_binary: str | None = None,
    page_load_timeout: int = 60,
) -> webdriver.Firefox:
    """
    Firefox webdriver configured to auto-download common file types to download_dir.
    """
    download_dir = Path(download_dir).expanduser().resolve()
    download_dir.mkdir(parents=True, exist_ok=True)

    opts = Options()
    if headless:
        opts.add_argument("-headless")

    if firefox_binary:
        opts.binary_location = firefox_binary  # e.g., SNAP path or custom install

    # ---- Download behavior ----
    # 2 = use custom download directory
    opts.set_preference("browser.download.folderList", 2)
    opts.set_preference("browser.download.dir", str(download_dir))
    opts.set_preference("browser.download.useDownloadDir", True)

    
    # Don't show the download panel
    opts.set_preference("browser.download.manager.showWhenStarting", False)
    opts.set_preference("browser.download.alwaysOpenPanel", False)
    opts.set_preference("browser.download.panel.shown", False)

    opts.set_preference("browser.download.manager.focusWhenStarting", False)
    opts.set_preference("browser.helperApps.alwaysAsk.force", False)
 
    # IMPORTANT: disable built-in PDF viewer so PDFs download instead of opening
    opts.set_preference("pdfjs.disabled", True)

    # Don't ask what to do — just save
    mime_types = ",".join([
        "application/pdf",
        "application/zip",
        "application/octet-stream",
        "application/x-zip-compressed",
        "application/x-gzip",
        "application/x-tar",
        "text/plain",
        "text/csv",
        "text/xml",
        "application/xml",
        "application/json",
        # LAS files are often served as text/plain or octet-stream; include both above.
    ])
    opts.set_preference("browser.helperApps.neverAsk.saveToDisk", mime_types)
    opts.set_preference("browser.helperApps.neverAsk.openFile", mime_types)
    opts.set_preference("browser.helperApps.alwaysAsk.force", False)

    # Some sites behave differently if they detect automation; this can help slightly.
    opts.set_preference("dom.webdriver.enabled", False)

    service = Service(executable_path=geckodriver_path) if geckodriver_path else Service()

    driver = webdriver.Firefox(service=service, options=opts)
    driver.set_page_load_timeout(page_load_timeout)
    return driver
 
def XXget_driver():
    #pathname = path.dirname(argv[0])
    #adir = path.abspath(pathname)
    adir = getcwd()

    ## initialize options
    #options = webdriver.ChromeOptions()
    ## pass in headless argument to options
    #options.add_argument('--headless')
    ## initialize driver
    #driver = webdriver.Chrome('\\\Server5\\Users\\KRucker\\chromedriver.exe',chrome_options=options)
 
    opts = Options()
    opts.headless = True 
    #Options.add_argument("--headless=new")

    SNAP = False
    # Find local firefox program
    if bool(re.match(r'.*linux.*',sys.platform, re.I)):
        P = subprocess.run(['whereis','firefox'], capture_output = True, text = True)
        P = P.stdout.strip()
        P = re.split(r'\s+',P)[1:]
        Possible_Locations = P
        # specific solution for SNAP Firefox
        if 'snap' in '_'.join(Possible_Locations).lower():
            opts.binary_location = subprocess.getoutput("find /snap/firefox -name firefox").split("\n")[-1]
            SNAP = True
        
    if bool(re.match(r'.*win.*',sys.platform, re.I)):
        Possible_Locations = FullFileScan(r'firefox.exe$')
    
    opts.set_preference("browser.download.folderList", 2)
    opts.set_preference("browser.download.manager.showWhenStarting", False)
    opts.set_preference("browser.download.dir", adir)
    opts.set_preference("browser.helperApps.neverAsk.saveToDisk",
                        (
                            "application/pdf, application/zip, application/octet-stream, "
                            "text/csv, text/xml, application/xml, text/plain, "
                            "text/octet-stream, application/x-gzip, application/x-tar "
                            "application/"
                            "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        ),
                    )
    try:
        if SNAP:
            driver = webdriver.Firefox(service =
                     Service(executable_path = subprocess.getoutput(f"find {path.join('snap','firefox')} -name geckodriver").split("\n")[-1]),
                     options = opts)
        else:
            driver = Firefox(options=opts)
    except:
        drivers= []
        for f in Possible_Locations:
            opts.binary_location = f
            try:
                drivers.append(Firefox(options=opts))

                driver = drivers[-1]
                break
            except:
                try:
                    drivers[-1].quit()
                except:
                    pass
                pass
       
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

def filelist(SUBDIR = None,EXT = None, BEGIN = None, CONTAINS = None):
    pathname = path.dirname(argv[0])

    if SUBDIR != None:
        pathname = path.join(pathname, SUBDIR)
        
    FLIST = list()
    if (EXT == None) & (BEGIN == None) & (CONTAINS == None):
        FLIST = listdir(pathname)
    else:
        for f in listdir(pathname):
            if filetypematch(f, filetypes=EXT, prefix = BEGIN, contains = CONTAINS):
                FLIST.append(f)
    return FLIST

def tupelize(x):
    if isinstance(x,(str,float,int)):
        out = tuple([x])
    else:
        out = tuple(x)
    return out

def filetypematch(fname, filetypes = None, prefix = None, contains = None):
    output = True
    if bool(filetypes):
        filetypes = tupelize(filetypes)
        filetypes = tuple(x.lower() for x in filetypes)
        output = output * fname.lower().endswith(filetypes)
    if bool(prefix):
        prefix = tupelize(prefix)
        prefix = tuple(x.lower() for x in prefix)
        output = output * fname.lower().startswith(prefix)
    if bool(contains):
        output = output * bool(re.findall(contains, fname, re.I))
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

def APIfromFilename(ffile,UWIlen=10):
    lst = re.findall(r'UWI[0-9]{UWIlen-1,}',ffile, re.I)
    if len(lst)==0:
        lst = re.findall(r'[0-9]{UWIlen-1,}',ffile)
    else:
        lst[0] = re.sub('UWI','',lst[0],re.I)
    if len(lst)>0:        
        UWI = WELLAPI(lst[0]).API2INT(UWIlen)
    else:
        UWI = None
    return UWI

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

def ERRORFILES(ERR_FOLDER='ERROR_FILES'):
    if not path.isdir(ERR_FOLDER):
        makedirs(ERR_FOLDER)
    return(ERR_FOLDER)

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()
            
def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData

def load_surveyfile(conn, row, table = 'SURVEYFILES'):
    """
    Create a new survey as filename and blob 
    :param conn:
    :param row:
    :return:
    """
    sql = ''' INSERT INTO {0}(FILENAME, FILE) VALUES (?,?)'''.format(table)
    cur = conn.cursor()
    cur.execute(sql, row)
    conn.commit()
    return cur.lastrowid

def convert_to_list(x):
    try: x=x.tolist()
    except: pass
    if isinstance(x,(int,str)):
        x=[x]
    if isinstance(x,(np.ndarray,pd.DataFrame,pd.Series)):
        x=list(x)
    if not isinstance(x,list):
        x=list(x)
        x2 = []
        for i in x:
            if isinstance(i,list): x2.extend(flatten(i))
            else: x2.append(i)
        x=x2
    return(x)

def findfiles(which, where='.',latest = True):
    '''Returns list of filenames from `where` path matched by 'which'
       shell pattern. Matching is case-insensitive.'''
    # TODO: recursive param with walk() filtering
    rule = re.compile(fnmatch.translate(which), re.IGNORECASE)
    list_of_files = [path.join(where,name) for name in listdir(where) if rule.match(name)]
    if len(list_of_files)>0:
        latest_file = max(list_of_files, key=path.getctime)
    else:
        latest_file = None
    if latest == True:
        return latest_file
    else:
        return list_of_files

def SQL_UNDUPLICATE(CONN, TABLENAME, GROUPBY = ['MD','FILE','UWI']):
    # THIS IS VERY SLOW
    if isinstance(GROUPBY,str):
        GROUPBY = [GROUPBY]
    
    c = CONN.cursor()
    QRY = 'SELECT COUNT(*) FROM {}'.format(TABLENAME)
    OLDROWS = c.execute(QRY).fetchall()[0][0]

    QRY = 'delete from {} where rowid not in (select  min(rowid) from {} group by {})'.format(TABLENAME,TABLENAME,','.join(GROUPBY))
    c.execute(QRY)

    QRY = 'SELECT COUNT(*) FROM {}'.format(TABLENAME)
    NEWROWS = c.execute(QRY).fetchall()[0][0]

    print('{} DUPLICATES DROPPED'.format(OLDROWS-NEWROWS))
    CONN.commit()
          
    return(NEWROWS)

def INIT_SQL_TABLE(CONN,TABLENAME, FIELD_DICT= None):
    # DROP_COLS is list of fields to remove
    
    #DROP TABLE IF EMPTY
    TEST = -1
    try:
        TEST =  pd.read_sql('SELECT * FROM {0} LIMIT 100'.format(TABLENAME), CONN).shape[0]
    except:
        pass
    
    if TEST == 0:
        DROP_SQL_TABLE(CONN,TABLENAME)        
  
    if FIELD_DICT == None:
        FIELD_DICT = {}
    QRY = '''SELECT count(name) FROM sqlite_master WHERE type='table' AND name = '{0}' '''.format(TABLENAME)
    #QRY = re.sub('_TABLENAME_',TABLENAME,QRY)
    #print(QRY)

    c = CONN.cursor()
    c.execute(QRY)

    if c.fetchone()[0]==0:
        QRY = """ CREATE TABLE _TABLENAME_ (
                _FIELDS_
            ); """
        QRY = re.sub('_TABLENAME_',TABLENAME,QRY)
        FIELD_TEXT = ''
        for k in FIELD_DICT.keys():
            if isinstance(FIELD_DICT[k], (list, tuple)):
                #FIELD_TEXT = FIELD_TEXT + '"' + k + '" ' + ' '.join('\''+FIELD_DICT[k]+'\'')+', \n'
                if len(FIELD_TEXT) > 0:
                    FIELD_TEXT = '{0}, \'{1}\' {2}'.format(FIELD_TEXT, k, ' '.join(FIELD_DICT[k]))
                else:
                    FIELD_TEXT = '\'{0}\' {1}'.format(k, ' '.join(FIELD_DICT[k]))
            else:
                if len(FIELD_TEXT) > 0:
                    #FIELD_TEXT = FIELD_TEXT + '"'+ k + '" \'' + FIELD_DICT[k]+'\', \n'
                    FIELD_TEXT = '{0}, \'{1}\' {2}'.format(FIELD_TEXT, k, FIELD_DICT[k])
                else:
                    FIELD_TEXT = '\'{0}\' {1}'.format(k, FIELD_DICT[k])
        #FIELD_TEXT = FIELD_TEXT[:-4]
        #QRY = re.sub('_FIELDS_',FIELD_TEXT,QRY)
        QRY = 'CREATE TABLE {0} ({1}); '.format(TABLENAME,FIELD_TEXT)
                                                    
        print('1: '+QRY)
        c.execute(QRY)

    else:
        QRY = 'PRAGMA table_info('+TABLENAME+');'
        c.execute(QRY)
        COLS = c.fetchall()
        COLS = [X[1] for X in COLS]
        for k in FIELD_DICT.keys():
            if k in COLS:
                continue
            else:
                if isinstance(FIELD_DICT[k], (list, tuple)):
                    #FIELD_TEXT = '"' + k + '" ' + ' '.join(FIELD_DICT[k])
                    FIELD_TEXT = '\'{0}\' {1}'.format( k, ' '.join(FIELD_DICT[k]))                             
                else:
                    #FIELD_TEXT = '"' + k + '" ' + FIELD_DICT[k]
                    FIELD_TEXT = '\'{0}\' {1}'.format(k, FIELD_DICT[k])       
            print('FT: '+FIELD_TEXT)
            QRY = 'ALTER TABLE {0} ADD COLUMN {1}'.format(TABLENAME,FIELD_TEXT)
            #QRY = re.sub('_TABLENAME_',TABLENAME,QRY)
            #QRY = re.sub('_FIELDS_',FIELD_TEXT,QRY)
            print('2: '+QRY)
            c.execute(QRY)
    CONN.commit()
    return None

def DROP_SQL_TABLE(CONN, TABLE_NAME):
    c = CONN.cursor()
    QRY = 'DROP TABLE IF EXISTS \'{0}\''.format(str(TABLE_NAME))
    c.execute(QRY)
    CONN.commit()           

def READ_SQL_TABLE(CONN, TABLE_NAME):
    c = CONN.cursor()
    c.execute('SELECT * FROM ' + TABLE_NAME)
    OUT = c.fetchall()
    return OUT

def LIST_SQL_TABLES(CONN):
    QRY = '''SELECT name FROM sqlite_schema WHERE type = 'table' ORDER BY name'''
    c = CONN.cursor()
    c.execute(QRY)
    OUT = c.fetchall()
    OUT = list(itertools.chain.from_iterable(OUT))
    return OUT

def QUERY_SQL_TABLES(CONN,QUERY):
    c = CONN.cursor()
    c.execute(QUERY)
    OUT = c.fetchall()
    return OUT

def DTYPE_TO_SQL(STRING):
    STRING = str(STRING)
    STRING = STRING.upper()
    if any(x in STRING for x in ('OBJECT','DATE')):
        return 'TEXT'
    if any(x in STRING for x in ('INT','INT32')):
        return 'INTEGER'
    if any(x in STRING for x in ('REAL, FLOAT')):
        return 'REAL'
    return STRING

def FRAME_TO_SQL_TYPES(df_in):
    OUT = dict()
    for k in df_in.keys():
        T = df_in[k].dtype.name
        OUT[k] = DTYPE_TO_SQL(T)        
    return(OUT)    

def APPLY_DICT_TO_LIST(LIST_IN,DICT_IN):
    if isinstance(LIST_IN,type(None)):
        return LIST_IN
    elif isinstance(LIST_IN,(str, int, float)):
        LIST_IN = [LIST_IN]
    else:
        LIST_IN = list(LIST_IN)
    LIST_OUT = LIST_IN
    for i in LIST_IN:
        if not i in DICT_IN.keys():
            continue
        DICT_VAL = DICT_IN[i]
        if isinstance(DICT_VAL,(str, int, float)):
            DICT_VAL = [DICT_VAL]
        else:
            DICT_VAL = list(DICT_VAL)
        LIST_OUT = LIST_OUT + DICT_VAL
    return LIST_OUT

def AziFromLatLon(LON1,LAT1,LON2,LAT2):
    R = 6371
    dLAT = LAT2-LAT1  #phi
    dLON = LON2-LON1  #lambda
    a = sin(dLAT/2*pi/180)**2 + cos(LAT1*pi/180)*cos(LAT2*pi/180)*sin(dLON/2*pi/180)**2
    d = 2*R*atan2(sqrt(a),sqrt(1-a))
    theta = atan2(sin(dLON*pi/180)*cos(LAT2*pi/180), cos(LAT1*pi/180)*sin(LAT2*pi/180)-sin(LAT1*pi/180)*cos(LAT2*pi/180)*cos(dLON*pi/180))
    theta =  180/pi * theta
    while theta<0:
        theta += 360
    return theta

def superposition_time(x,y, stype = 'linear'):
    if not isinstance(x,np.ndarray):
        x = np.array(x)
    if not isinstance(y,np.ndarray):
        y = np.array(y)
    # ensure 0,0
    if abs(x[0]+y[0]) != 0:
        x = np.insert(x,0,0)
        y = np.insert(y,0,0)

    OUTPUT = []
    if stype == 'linear':
        dx = x[1:]-x[:-1]
        dy = y[1:]-y[:-1]
        for i,xi in enumerate(x):
            OUTPUT.append(sum((dy[:i]/y[i]* np.sqrt(x[i] - x[:i])))**2)

    return(OUTPUT)   

def asym_sigmoid(x, S, EC50, HillSlope, Top, Bottom):
    Denom = (1+(2**(1/S)-1)*((EC50/x)**HillSlope))**S
    Num = Top-Bottom
    Y = Bottom + (Num / Denom)
    return Y
 
        
def UrlDownload(url:str, savefile:str):
    #response=requests.get(url)
    #exfile=response.content
    #egfile=open(savefile,'wb')
    #egfile.write(exfile)
    #egfile.close()
    r=requests.get(url).content
    with open(savefile,'wb') as f:
        f.write(r)

def Items_in_Polygons(ITEM_SHAPEFILE:str, POLYGON_SHAPEFILE:str, BUFFER :(int,float) = None, EPSG4POLY = None, NameIndex:int = None):
    ITEMS = shp.Reader(ITEM_SHAPEFILE)
    ITEMS = read_shapefile(ITEMS)
    CRS_ITEMS = CRS_FROM_SHAPE(ITEM_SHAPEFILE)

    POLYS = shp.Reader(POLYGON_SHAPEFILE)
    POLYS = read_shapefile(POLYS)
    CRS_POLYS = CRS_FROM_SHAPE(POLYGON_SHAPEFILE)

    TFORMER = pyproj.Transformer.from_crs(pyproj.CRS.from_wkt(CRS_POLYS.to_ogc_wkt()),
                                          pyproj.CRS.from_wkt(CRS_ITEMS.to_ogc_wkt()),
                                          always_xy = True)

    ITEMS['coords_old'] = ITEMS['coords']
    POLYS['coords_old'] = POLYS['coords']

    if NameIndex == None:
        NAMES = POLYS.applymap(lambda x:isinstance(x,str)).sum(axis=0).replace(0,np.nan).dropna()
        NAMES = POLYS[list(NAMES.index)].nunique(axis=0).sort_values(ascending=False).index[0]
    else:
        NAMES = POLYS.keys()[NameIndex]

    for i in POLYS.index:
        NAME = POLYS.loc[i,NAMES]
        NAME = str(NAME)

        converted = TFORMER.transform(pd.DataFrame(POLYS.coords_old[0])[0],
                          pd.DataFrame(POLYS.coords_old[0])[1])
        POLYS.at[i,'coords'] = list(map(tuple, np.array(converted).T))
        POLY_SHAPE = shapely.geometry.Polygon(POLYS.loc[i,'coords'] )

        RESULT = GROUP_IN_TC_AREA(POLYS.loc[[i],:],ITEMS)
        ITEMS[f'IN_{NAME}'] = RESULT.TEST.values

    return ITEMS

def pd_find_regex(dataframe, regex_pattern):
    """
    Find all cells in a DataFrame that match a given regex pattern.

    Args:
        dataframe (pd.DataFrame): The input DataFrame to search.
        regex_pattern (str): The regex pattern to search for.

    Returns:
        list: A list of tuples containing (row_index, column_name) of matches.
    """
    matches = []
    pattern = re.compile(regex_pattern)

    for col in dataframe.columns:  # Iterate over all columns
        for idx, value in dataframe[col].items():  # Iterate over rows in each column
            if pd.notna(value) and pattern.search(str(value)):
                matches.append((idx, col))
    
    return matches

def zipwalk(zip_file_path):
    """
    Generates a tuple (dirname, dirnames, filenames) for each directory
    within a ZIP file, similar to os.walk().
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zf:
        # Get a list of all names within the zip file
        all_names = zf.namelist()

        # Organize names into a dictionary representing the directory structure
        # Key: directory path, Value: tuple of (subdirectories, files)
        structure = {}

        for name in all_names:
            parts = name.split('/')
            
            # Handle directories
            if name.endswith('/'):
                dir_path = '/'.join(parts[:-1])
                parent_dir = '/'.join(parts[:-2]) if len(parts) > 2 else ''
                
                if parent_dir not in structure:
                    structure[parent_dir] = ([], [])
                if parts[-2] not in structure[parent_dir][0]:
                    structure[parent_dir][0].append(parts[-2])
                
                if dir_path not in structure:
                    structure[dir_path] = ([], [])
            
            # Handle files
            else:
                dir_path = '/'.join(parts[:-1])
                filename = parts[-1]
                
                if dir_path not in structure:
                    structure[dir_path] = ([], [])
                structure[dir_path][1].append(filename)
        
        # Yield results in a similar fashion to os.walk
        for dir_path, (subdirs, files) in structure.items():
            yield dir_path, sorted(subdirs), sorted(files)
