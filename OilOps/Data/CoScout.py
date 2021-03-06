import datetime,re,io,csv,sys,requests,selenium,math, os
#urllib,
import multiprocessing,warnings,concurrent.futures
from os import path, listdir, remove, makedirs
import pandas as pd
#from urllib.request import urlopen 
import numpy as np
from bs4 import BeautifulSoup as BS
from math import ceil
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from time import sleep
from functools import partial

# v200 adds coordinate conversion

import shapefile as shp #pyshp
from shapely.geometry import Polygon, Point, LineString
from pyproj import Proj, transform, CRS
import easygui
from tkinter import filedialog

import sqlite3
from pyproj import Transformer

# no perf interval data in Emily's table for 05123505910000


# COORDINATE REFERENCES
# EPSG 4267
##GEOGCS["NAD27",
##    DATUM["North_American_Datum_1927",
##        SPHEROID["Clarke 1866",6378206.4,294.9786982138982,
##            AUTHORITY["EPSG","7008"]],
##        AUTHORITY["EPSG","6267"]],
##    PRIMEM["Greenwich",0,
##        AUTHORITY["EPSG","8901"]],
##    UNIT["degree",0.01745329251994328,
##        AUTHORITY["EPSG","9122"]],
##    AUTHORITY["EPSG","4267"]]

# EPSG 4269
##GEOGCS["NAD83",
##    DATUM["North_American_Datum_1983",
##        SPHEROID["GRS 1980",6378137,298.257222101,
##            AUTHORITY["EPSG","7019"]],
##        AUTHORITY["EPSG","6269"]],
##    PRIMEM["Greenwich",0,
##        AUTHORITY["EPSG","8901"]],
##    UNIT["degree",0.01745329251994328,
##        AUTHORITY["EPSG","9122"]],
##    AUTHORITY["EPSG","4269"]]

# EPSG

def UWI10(num):
    num=int(num)
    while num > 9e9:
        num = math.floor(num/100)
    num = int(num)
    return num

def XYtransform(df_in, epsg1 = 4269, epsg2 = 2878):
    #2876
    df_in=df_in.copy()
    transformer = Transformer.from_crs(epsg1, epsg2,always_xy =True)
    df_in[['X','Y']]=df_in.apply(lambda x: transformer.transform(x.iloc[2],x.iloc[1]), axis=1).apply(pd.Series)
    #df_in[['X','Y']]=df_in.apply(lambda x: transform(epsg1,epsg2,x.iloc[2],x.iloc[1],always_xy=True), axis=1).apply(pd.Series)
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
    try:
        ln = LineString(well2.coords)
    except:
        ln = LineString(well2.coords.values)
    test = False
    for j in range(0,tc2.shape[0]):
        if test == False:
            poly = Polygon(tc2.coords.iloc[j])
            if ln.intersects(poly.buffer(10000)):
                test = True   
    return(test) 

def GROUP_IN_TC_AREA(tc,wells):
    out = pd.DataFrame()
    out['API'] = wells.API_Label.str.replace(r'[^0-9]','',regex=True)
    out['TEST'] = wells.apply(lambda x: IN_TC_AREA(x,tc),axis=1)
    return(out)

def requests_retry_session(
    retries=6,
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

#url = 'https://cogcc.state.co.us/cogis/FacilityDetail.asp?facid=12346331&type=WELL'
#URL_BASE = 'https://cogcc.state.co.us/cogis/FacilityDetail.asp?facid=XNUMBERX&type=WELL'

#df=pd.read_html(html)
#df[0] # looks like this fully reads COGCC url
      # parse out table with alias book

# [Row, Column]

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

#TopPZ_FTG =[df.index.get_loc(df.loc[(df.select_dtypes(include=[object]).stack().str.contains('.*TOP PZ.*', regex=True, case=False,na=False).unstack()==True).any(axis='columns'),:].index.values[0]),
#          df.keys().get_loc(df.loc[:,(df.select_dtypes(include=[object]).stack().str.contains(r'.*FOOT.*', regex=True, case=False,na=False).unstack()==True).any(axis='rows')].keys().values[0])]
#BHL_FTG =[df.index.get_loc(df.loc[(df.select_dtypes(include=[object]).stack().str.contains('.*BOTTOM HOLE LOCATION.*', regex=True, case=False,na=False).unstack()==True).any(axis='columns'),:].index.values[0]),
#          df.keys().get_loc(df.loc[:,(df.select_dtypes(include=[object]).stack().str.contains(r'.*FOOT.*', regex=True, case=False,na=False).unstack()==True).any(axis='rows')].keys().values[0])]
#
# For i in API list, get ith data
    # Read table
    # break up multiple completions and completed intervals
    # API
    # Keep simple for V1.0
    # Summarize frac size, footages
    # Frac Date, Prod Date, Test Date
    # Formation footages

# Read pages "sections"
# Drop na datapoints and assign data to intervals
#df=pd.read_html(html)[0]
#XPATH = df.get_loc(df.iloc[df.iloc[:,0].str.contains('.*sidetrack.*',regex=True,case=False,na=False),0])
#Completions_index = df.ix[df.iloc[:,0].str.contains('.*completed information.*',regex=True,case=False,na=False),:].index
#TSummary_index = df.ix[df.iloc[:,0].str.contains('.*treatment.*summary.*',regex=True,case=False,na=False),:].index
# TSummary needs to be parsed into depth intervals with labels

def Get_Scouts(UWIs,db=None):
    #if 1==1:
    Strings = ['WELL NAME/NO', 'OPERATOR', 'STATUS DATE','FACILITYID','COUNTY','LOCATIONID','LAT/LON','ELEVATION',
               'SPUD DATE','JOB DATE','JOB END DATE','TOP PZ','BOTTOM HOLE LOCATION',#r'COMPLETED.*INFORMATION.*FORMATION',
               'TOTAL FLUID USED','MAX PRESSURE','TOTAL GAS USED','FLUID DENSITY','TYPE OF GAS',
               'NUMBER OF STAGED INTERVALS','TOTAL ACID USED','MIN FRAC GRADIENT','RECYCLED WATER USED',
               'TOTAL FLOWBACK VOLUME','PRODUCED WATER USED','TOTAL PROPPANT USED',
               'TUBING SIZE','TUBING SETTING DEPTH','# OF HOLES','INTERVAL TOP','INTERVAL BOTTOM','^HOLE SIZE','FORMATION NAME','1ST PRODUCTION DATE',
               'BBLS_H2O','BBLS_OIL','CALC_GOR', 'GRAVITY_OIL','BTU_GAS','TREATMENT SUMMARY']

    status_pat = re.compile(r'Status:([\sA-Z]*)[0-9]{1,2}/[0-9]{1,2}/[0-9]{1,2}', re.I)
                
    OUTPUT=[]
    pagedf=[]
    xSummary = None
    URL_BASE = 'https://cogcc.state.co.us/cogis/FacilityDetail.asp?facid=XNUMBERX&type=WELL'
    pathname = path.dirname(sys.argv[0])
    adir = path.abspath(pathname)
    warnings.simplefilter("ignore")
    if isinstance(UWIs,list) == False:
        UWIs=[UWIs]
    for UWI in UWIs:
        #if 1==1:
        UWI = str(UWI)
        if len(UWI)%2 == 1:
            UWI = UWI.zfill(len(UWI)+1)
            
        print(UWI+" "+datetime.datetime.now().strftime("%d/%m/%Y_%H:%M:%S"))
        docurl = None
        connection_attempts = 4 
        #Screen for Colorado wells
        userows=pd.DataFrame()
        if UWI[:2] == '05':
            #Reduce well to county and well numbers
            docurl=re.sub('XNUMBERX',UWI[2:10],URL_BASE)
            RETRY=0
            while RETRY<8:
                try:
                    pagedf=pd.read_html(docurl)[0]
                    RETRY=60
                except:
                    pagedf=[]
                    RETRY += 1
                    sleep(10)
            
        if len(pagedf)>0:
            xSummary = Summarize_Page(pagedf,Strings)
            xSummary['UWI']=UWI

            # Status code
            STAT_CODE = None
            try:
                status = status_pat.search(pagedf.iloc[1,0])
                status = status.group(1)
                STAT_CODE = status.strip()
            except:
                print('status error')
                pass

            xSummary['WELL_STATUS'] = STAT_CODE
            
            xSummary = pd.DataFrame([xSummary.values],columns= xSummary.index.tolist())

            if type(OUTPUT)==list:
                OUTPUT=xSummary
            else:
                OUTPUT=OUTPUT.append(xSummary,ignore_index=True)

    FILENAME = str(UWIs[0])+'_'+str(UWIs[-1])+"_"+datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    FILENAME = path.join(dir_add,FILENAME) 
    DF_UNSTRING(OUTPUT).to_json(FILENAME+'.JSON')
    DF_UNSTRING(OUTPUT).to_parquet(FILENAME+'.PARQUET')
    
    if db != None:
        DATECOLS = [col for col in OUTPUT.columns if 'DATE' in col.upper()]
        for k in DATECOLS:
            OUTPUT.loc[:,k]=pd.to_datetime(OUTPUT.loc[:,k]).fillna(np.nan)
            OUTPUT.loc[OUTPUT.loc[:,k],k]
        conn = sqlite3.connect(db)
        c = conn.cursor()
        TABLE_NAME = 'CO_SCOUT'
        SQL_COLS = list()
        # NEEDS CONVERSION OF PYTHON TYPES TO SQL TYPES
        for k,v in OUTPUT.dtypes.to_dict().items():    
            SQL_COLS=SQL_COLS+'['+str(k)+'] '+str(v)+','
        #c.execute('CREATE TABLE IF NOT EXISTS ' + TABLE_NAME + ' ')
        #sql = "select * from %s where 1=0;" % table_name
        #c.execute(sql)
        #TBL_COLS = [d[0] for d in curs.description]
        #ADD_COLS = list(set(SQL_COLS).difference(TBL_COLS))
        OUTPUT.to_sql(TABLE_NAME,conn,if_exists='append',index=False)

        #OUTPUT.to_csv(FILENAME)

    return(OUTPUT)

##def read_shapefile(sf):
##    # https://towardsdatascience.com/mapping-with-matplotlib-pandas-geopandas-and-basemap-in-python-d11b57ab5dac
##    #fetching the headings from the shape file
##    fields = [x[0] for x in sf.fields][1:]
##    #fetching the recordsi from the shape file
##    records = [list(i) for i in sf.records()]
##    shps = [s.points for s in sf.shapes()]
##    #converting shapefile data into pandas dataframe
##    df = pd.DataFrame(columns=fields, data=records)
##    #assigning the coordinates
##    df = df.assign(coords=shps)
##    return df

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
            print(str(ct)+" of "+str(total))
            shaperec=r.shapeRecord(i)
            Xshaperec=shaperec.shape            
            points = np.array(shaperec.shape.points).T
            points_t= transform(crs_old, crs_new, points[0],points[1],always_xy=True)
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

def CheckDuplicate(fname):
    ct = ''
    while os.path.exists(fname):
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

def DF_UNSTRING(df_IN):
    df_IN=df_IN.copy()

    #DATES
    DATECOLS = [col for col in df_IN.columns if 'DATE' in col.upper()]
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
    FLOAT_MASK = (df_IN.apply(pd.to_numeric, downcast = 'float', errors = 'coerce').count() - df_IN.count()==0)   
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
    
    return(df_IN)

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
        

#df = pd.read_csv(path.join(dir_add,'SCOUT_PULL_10082021 (copy).csv'))

# xx['TREATMENT_SUMMARY'].apply(STIM_SUMMARY_TO_ARRAY).rename(columns={0:'FluidBBLS',1:'PropLBS',2:'MaxPSI'})
# x = xx['TREATMENT_SUMMARY'].apply(STIM_SUMMARY_TO_ARRAY).rename(columns={0:'FluidBBLS',1:'PropLBS',2:'MaxPSI'}).reset_index().rename(columns={'UWI':'UWI10'})

if __name__ == "__main__":
    #if 1==1:
    # Initialize constants
    URL_BASE = 'https://cogcc.state.co.us/cogis/ProductionWellMonthly.asp?APICounty=123&APISeq=XNUMBERX&APIWB=00&Year=All'
    DL_BASE = 'http://cogcc.state.co.us/weblink/XLINKX'
    pathname = path.dirname(sys.argv[0])
    adir = path.abspath(pathname)
    dir_add = path.join(path.abspath(path.dirname(sys.argv[0])),"SCOUTS")

    if not path.exists(dir_add):
        makedirs(dir_add)

    outfile_key = 'SCOUT_PULL_SUMMARY_'
    outfile = path.join(dir_add,outfile_key+datetime.datetime.now().strftime("%d%m%Y"))

    DATES = dict()
    try:
        lastfile = sorted(glob.glob(path.join(dir_add,outfile_key+'*arquet')),key = os.path.getctime)[0]
        df = pd.read_parquet(lastfile)
        DATES = dict(zip(df.UWI,df.STATUS_DATE))
        del df
    except: pass
        
    #Read UWI files and form UWI list
    UWIlist=[]
##    for file in listdir(adir):
##        if file.lower().endswith(".uwi"):       
##            with open(file, 'r') as f:
##                for line in f:
##                    UWIlist.append(line[:-1])

    # shapefile UWIlist if 1==1:
    sfile = 'Directional_Lines.shp'
    sdf = shp.Reader(sfile)
    sdf = read_shapefile(sdf)
    ddf = pd.DataFrame(sdf.coords.to_list(), index=sdf.index)
    for i in ddf.index:
        x1 = ddf.iloc[i,0][0]
        y1 = ddf.iloc[i,0][1]
        x2 = ddf.iloc[i,:].dropna().iloc[-1][0]
        y2 = ddf.iloc[i,:].dropna().iloc[-1][1]
        sdf.loc[i,'Delta']=((x2-x1)**2+(y2-y1)**2)**0.5
    sdf['UWI10'] = sdf.API_Label.str.replace(r'[^0-9]','',regex=True).apply(UWI10)
    SHPUWIlist = list(sdf.loc[sdf.Delta>2000,'UWI10'].unique())
    #SHPUWIlist=['05' + s for s in SHPUWIlist]
    del ddf

    #filter shapefile to scouts within buffer of TC areas
    polygons_file = 'Type_Curve_Areas_2020.SHP' #EPSG 4267 (directional lines in EPSG 26913)
    epsg1 = 4267
    epsg2 = 26913
    transformer = Transformer.from_crs(epsg1, epsg2,always_xy =True)
    p_shp = shp.Reader(polygons_file)
    poly_df = read_shapefile(p_shp)
    
    del p_shp
    for i in range(0,poly_df.shape[0]):
        pts = poly_df.coords.iloc[i]
        pts = pd.DataFrame(pts)
        pts[['X','Y']]=pts.apply(lambda x: transformer.transform(x.iloc[0],x.iloc[1]), axis=1).apply(pd.Series)
        poly_df.coords.iloc[i] = list(pts[['X','Y']].to_records(index=False))        
    
    # remember
    # line.intersection(polygon.buffer(0.2))

    sdf['Test'] = False
    # tc = poly_df
    # sdf.apply(lambda x: IN_TC_AREA(x,poly_df),axis=1)
    processors = max(1,multiprocessing.cpu_count())
    data=np.array_split(sdf,processors)

    func = partial(GROUP_IN_TC_AREA,poly_df)
    quit()
    if processors > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
            f = {executor.submit(func, a): a for a in data}        
        RESULT=pd.DataFrame()
        for i in f.keys():
            RESULT=pd.concat([RESULT,i.result()],axis=0,join='outer',ignore_index=True)
    else:
        RESULT = GROUP_IN_TC_AREA(poly_df,sdf)

    UWIlist = (RESULT.loc[RESULT.TEST==True,'API'].apply(UWI10)*10000).astype(str).str.zfill(14).to_list()
    UWIlist = np.array_split(UWIlist,2)[0]
##    # SQL DB
##    sqldb = 'CO_3_2.1.sqlite'
##    with sqlite3.connect(sqldb) as conn:
##        dfSQL = pd.read_sql_query('SELECT * FROM WELL',conn)
##    transformer = Transformer.from_crs(4269, 2878,always_xy =True)
##    x=transformer.transform(list(dfSQL['Longitude']),list(dfSQL['Latitude']))
##    dfSQL['SHLX']=x[0]
##    dfSQL['SHLY']=x[1]
##    x=transformer.transform(list(dfSQL['BottomHolelongitude']),list(dfSQL['BottomHoleLatitude']))
##    dfSQL['BHLX']=x[0]
##    dfSQL['BHLY']=x[1]
##    dfSQL['Delta'] = ((dfSQL.SHLX-dfSQL.BHLX)**2+(dfSQL.SHLY-dfSQL.BHLY)**2)**0.5
##
##    SQL_UWIlist = list(dfSQL.loc[dfSQL.Delta>2000,'API'].apply(UWI10).unique())
##    del dfSQL,x
##
##    UWIlist = SHPUWIlist + SQL_UWIlist
##    UWIlist = list((pd.Series(UWIlist)*10000).astype(str).str.zfill(14).unique())
##    
    print ("read UWI file(s)")
    #UWIlist = list(set(UWIlist))
    #CoList = [idx for idx in UWIlist if idx.startswith('05')]
    # Parallel Execution
    #CoList=CoList[3000:3050]
    print ("starting map function")
    processors = max(1,multiprocessing.cpu_count())
    
    #quit()

if 1==1:
##    #lst = pd.read_csv('PullScoutSummary.csv')
##    if path.exists('PullScoutSummary.PARQUET'):
##        lst = pd.read_parquet('PullScoutSummary.PARQUET')
##        lst=lst.UWI10.astype(str).zfill(10)
##        lst=lst.to_list()
##    else:
##        lst=[]

    
    #batch = max(int(len(CoList)/2000),processors)
    data=np.array_split(UWIlist,processors)

##    chunksize = min(1000,int(len(CoList)/processors))
##    batch = int(len(CoList)/chunksize)
##    processors = min(processors,batch)
##    data = np.array_split(CoList,batch)
    print ("starting map function")
    # outfile = "BTU_API_PULL_"+datetime.datetime.now().strftime("%d%m%Y")+".csv"

    #processors = 0
    if processors > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
            f = {executor.submit(Get_Scouts, a): a for a in data}
        RESULT=pd.DataFrame()
        for i in f.keys():
            try:
                RESULT=pd.concat([RESULT,i.result()],axis=0,join='outer',ignore_index=True)
            except:
                pass
    else:
        RESULT = Get_Scouts(UWIlist)

    for f in listdir(dir_add):
        if 'parquet' in f.lower():
            a=pd.read_parquet(path.join(dir_add,f))
            b=pd.concat([a,b],axis=0, join='outer',ignore_index=True)
            b.dropna(axis=1,how='all',inplace=True)
            b = DF_UNSTRING(b) # BACKUP VERSION FOR RESULT
    
    #DF_UNSTRING TESTING if True:
##    a = DF_UNSTRING(b)
##    test = (b.groupby('UWI').FACILITYID.count()>1)
##    test.loc[test==True].index.to_list()
##    u2 = test.loc[test==True].index.to_list()
##    #b.loc[b.UWI == u2[0],:].T
##
##    for u in u2:
##        (b.loc[b.UWI == u,:].astype(str).T.apply(list,axis=1)).apply(set).apply(len).to_list()
##
##    lst = list()
##    for k in a.keys():
##        if (a[k]==b[k]).unique().size >1:
##            lst.append(k)
##            print (a[k].dtypes)
##            print (b[k].dtypes)
##
##    for k in lst:
##        m = (a[k]!=b[k])
##        pd.concat([a.loc[m,k],b.loc[m,k]], axis = 1)
    
    Problem_Columns = ['LAT/LON','TREATMENT_SUMMARY','STATUS_DATE','1ST_PRODUCTION_DATE']
    RESULT = RESULT.loc[RESULT.drop(Problem_Columns, axis=1).drop_duplicates().index]

    
    RESULT[['TREAT_FLUID','TREAT_PROPPANT','TREAT_PRESSURE']] = RESULT.TREATMENT_SUMMARY.apply(STIM_SUMMARY_TO_ARRAY)
    

##    if 1==1:
##        cts = R.FACILITYID.value_counts()
##        dbl = cts.loc[cts>1].index.values
##        R.loc[R.FACILITYID.isin(dbl)]
    
    #RESULT = Get_Scouts(UWIlist,SQLDB)
    #outfile = path.join(dir_add,'SCOUT_PULL_SUMMARY_'+datetime.datetime.now().strftime("%d%m%Y"))
    #RESULT.to_csv(outfile+'.csv',sep=',')
    RESULT.to_json(outfile+'.json')
    RESULT.to_parquet(outfile+'.parquet')
    
if 1==1:
    #SQLDB = '\\\Server5\\Verdad Resources\\Operations and Wells\\Geology and Geophysics\\WKR\\Decline_Parameters\\DeclineParameters_v200\\prod_data.db'
    SQLDB = 'prod_data.db'
    pd_sql_types={'object':'TEXT',
                  'int8':'INTEGER',
                  'int16':'INTEGER',
                  'int32':'INTEGER',
                  'int64':'INTEGER',                  
                  'float32':'REAL',
                  'float64':'REAL',
                  'bool':'TEXT',
                  'datetime64[ns]':'TEXT',
                  'timedelta[ns]':'REAL',
                  'category':'TEXT'
        }
    df_dtypes = RESULT.dtypes.astype('str').map(pd_sql_types).to_dict()
    with sqlite3.connect(SQLDB) as conn:
        TABLE_NAME = 'CO_SCOUT'
        RESULT.to_sql(TABLE_NAME,conn,
                      if_exists='replace',
                      index=False,
                      dtype=df_dtypes)


    

    
#np.array(transform(4269, 2876,-104.892422,39.999196,always_xy=True))
#np.array(transform(4269, 2876,-104.890175,39.999069,always_xy=True))
########################
#### END OF ROUTINE ####
########################
#
