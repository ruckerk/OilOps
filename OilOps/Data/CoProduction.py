import urllib,datetime,re,io,csv,sys,requests,selenium,math,time
import multiprocessing,warnings,concurrent.futures
from os import path, listdir, remove, makedirs
import pandas as pd
from urllib.request import urlopen 
import numpy as np
from bs4 import BeautifulSoup as BS
from math import ceil, floor
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from time import sleep
from functools import partial
import shapefile as shp #pyshp
from shapely.geometry import Polygon, Point, LineString
from pyproj import Proj, transform, CRS
import easygui
from tkinter import filedialog

import sqlite3
from pyproj import Transformer

# v004 pulls production formation

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


# DEFINE FUNCTION FOR ZONE DETECTION
# def IsZone(ZoneArray,ZoneName(s),OnlyFlag)
# parse array into list of all zones
# find all zones matching criteria
# if only flag = 1, then return fields with only search zone
# if only flag = 0, then return fields with any search zone
# "_".join(OUTPUT.Production_Formation.drop_duplicates())
#list(set("_".join(OUTPUT.Production_Formation.drop_duplicates()).split("_")))
# TIMPAS, NIOBRARA, FORT HAYS, CODELL, CARLILE
# J SAND
# COLUMN: NIO-COD: r"NIO|COD|(FORT or FT) H|CARL|TIMPAS
# COLUMN: J :"J"

##FM_DICT = {re.compile('NIO[BRA]*',re.I):'NIOBRARA',
##           re.compile('SHARON[ \-_]*SPRINGS',re.I):'NIOBRARA',
##           re.compile('F[OR]*T[ \-_]*H[AYS]*',re.I):'CODELL',
##           re.compile('TIMPAS',re.I):'CODELL',
##           re.compile('COD[DEL]*',re.I):'CODELL',
##           re.compile('CARLILE',re.I):'CODELL',
##           re.compile('J[ _\-0-9]*S[A]*ND',re.I):'JSAND',
##           re.compile('(^|[ \-])(J[ SAND\-]*)($|\-)'):r'\1JSAND\3'
##           }
##OUTPUT.Production_Formation.replace(FM_DICT,regex=True)
##FM_SEARCH = {}


def UWI10(num):
    if isinstance(num,str):
        num = re.sub(r'^0+','',num.strip())
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
    ln = None
    if len(well2.coords)>=2:
        try:
            ln = LineString(well2.coords)
        except:
            ln = LineString(well2.coords.values)
    elif len(well2.coords)==1:
        ln = Point(well2.coords[0])
    if ln == None:
        return(False)
    test = False
    for j in range(0,tc2.shape[0]):
        if test == False:
            poly = Polygon(tc2.coords.iloc[j])
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
            if (math.floor(ct/20)*20) == ct:
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

def Get_ProdData(UWIs,file='prod_data.db',SQLFLAG=0):
    #if 1==1:
    #URL_BASE = 'https://cogcc.state.co.us/cogis/ProductionWellMonthly.asp?APICounty=XCOUNTYX&APISeq=XNUMBERX&APIWB=XCOMPLETIONX&Year=All'
    URL_BASE = 'https://cogcc.state.co.us/production/?&apiCounty=XCOUNTYX&apiSequence=XNUMBERX'
    pathname = path.dirname(sys.argv[0])
    adir = path.abspath(pathname)
    warnings.simplefilter("ignore")
    OUTPUT=pd.DataFrame(columns=['BTU_MEAN','BTU_STD'
                                 ,'API_MEAN','API_STD'
                                 ,'Peak_Oil_Date','Peak_Oil_Days','Peak_Oil_CumOil','Peak_Oil_CumGas','Peak_Oil_CumWtr'
                                 ,'Peak_Gas_Date','Peak_Gas_Days','Peak_Gas_CumOil','Peak_Gas_CumGas','Peak_Gas_CumWtr'
                                 ,'OWR_PrePeakOil','OWR_PostPeakGas'
                                 ,'GOR_PrePeakOil','GOR_PeakGas','GOR_PostPeakGOR'
                                 ,'WOC_PostPeakOil','WOC_PostPeakGas'
                                 ,'Peak_Oil_CumWtr','Peak_Gas_CumWtr'
                                 ,'Month1'
                                 ,'GOR_MO2-4','GOR_MO5-7','GOR_MO11-13','GOR_MO23-25','GOR_MO35-37','GOR_MO47-49'
                                 ,'OWR_MO2-4','OWR_MO5-7','OWR_MO11-13','OWR_MO23-25','OWR_MO35-37','OWR_MO47-49'
                                 ,'Production_Formation'])
    MonthArray = np.arange(3,49,3)
    for i in MonthArray:
        OUTPUT[str(i)+'Mo_CumOil'] = np.nan
        OUTPUT[str(i)+'Mo_CumGas'] = np.nan
        OUTPUT[str(i)+'Mo_CumWtr'] = np.nan
                        
    if len(UWIs[0])<=1:
        UWIs=[UWIs]
        print(UWIs[0])
    ct = 0
    t1 = time.perf_counter()
    for UWI in UWIs:
        if (math.floor(ct/20)*20) == ct:
            print(str(ct)+' of '+str(len(UWIs)))
        ct+=1
        html = soup = pdf = None 
        #print(UWI)
        #if 1==1:
        ERROR=0
        while ERROR == 0: #if 1==1:
            connection_attempts = 4 
            #Screen for Colorado wells
            userows=pd.DataFrame()
            if UWI[:2] == '05':
                #print(UWI)
                #Reduce well to county and well numbers
                COWELL=UWI[5:10]
                if len(UWI)>=12:
                    COMPLETION=UWI[10:12]
                else:
                    COMPLETION="00"
                docurl=re.sub('XNUMBERX',COWELL,URL_BASE)
                docurl=re.sub('XCOUNTYX',UWI[2:5],docurl)
                docurl=re.sub('XCOMPLETIONX',COMPLETION,docurl)
                #try:
                #    html = urlopen(docurl).read()
                #except Exception as ex:
                #    print(f'Error connecting to {docurl}.')
                #    ERROR=1
                #    continue
                #soup = BS(html, 'lxml')
                #try:
                #    parsed_table = soup.find_all('table')[1]
                #except:
                #    print(f'No Table for {UWI}.')
                #    ERROR=1
                #    continue
                if time.perf_counter() - t1 < 0.5:
                    time.sleep(0.5)
                t1 = time.perf_counter()
                try:
                    #pdf = pd.read_html(docurl,encoding='utf-8', header=0)[1]
                    content = requests_retry_session().get(docurl).content
                    rawData = pd.read_html(io.StringIO(content.decode('utf-8')))
                    pdf = rawData[1]
                except:
                    print(f'Error connecting to {docurl}.')
                    ERROR=1
                    continue
                #pdf=pd.read_html('https://cogcc.state.co.us/cogis/ProductionWellMonthly.asp?APICounty=123&APISeq=42282&APIWB=00&Year=All')[1]
                try:
                    SEQ      = pdf.iloc[:,pdf.keys().str.contains('.*SEQUENCE.*', regex=True, case=False,na=False)].keys()[0]
                except:
                    for i in range(0,pdf.shape[1]):
                        newcol = str(pdf.keys()[i])+'_'+'_'.join(pdf.iloc[0:8,i].astype('str'))
                        pdf=pdf.rename({pdf.keys()[i]:newcol},axis=1)
                    try:    
                        SEQ      = pdf.iloc[:,pdf.keys().str.contains('.*SEQUENCE.*', regex=True, case=False,na=False)].keys()
                        x = min(np.array(pdf.loc[pdf[SEQ].astype(str) == COWELL,SEQ].index))-1   # non-value indexes
                        xrows = list(range(0, x))
                        for i in range(0,pdf.shape[1]):
                            newcol = str(pdf.keys()[i])+'_'+'_'.join(pdf.iloc[0:x,i].astype('str'))
                            pdf = pdf.rename({pdf.keys()[i]:newcol},axis=1)
                            pdf = pdf.drop(xrows,axis=0)
                    except:
                        print(f'Cannot parse tabels 1 at: {docurl}.')
                        ERROR = 1
                        continue
                   
##                DATE     =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*FIRST.*MONTH.*', regex=True, case=False,na=False)].keys()[0])
##                DAYSON   =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*DAYS.*PROD.*', regex=True, case=False,na=False)].keys()[0])
##                OIL      =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*OIL.*PROD.*', regex=True, case=False,na=False)].keys()[0])
##                GAS      =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*GAS.*PROD.*', regex=True, case=False,na=False)].keys()[0])
##                WTR      =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*WATER.*VOLUME.*', regex=True, case=False,na=False)].keys()[0])
##                API      =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*OIL.*GRAVITY.*', regex=True, case=False,na=False)].keys()[0])
##                BTU      =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*GAS.*BTU.*', regex=True, case=False,na=False)].keys()[0])
                try: 
                    DATE     = pdf.iloc[:,pdf.keys().str.contains('.*FIRST.*MONTH.*', regex=True, case=False,na=False)].keys()[0]
                    DAYSON   = pdf.iloc[0,pdf.keys().str.contains('.*DAYS.*PROD.*', regex=True, case=False,na=False)].keys()[0]
                    OIL      = pdf.iloc[0,pdf.keys().str.contains('.*OIL.*PROD.*', regex=True, case=False,na=False)].keys()[0]
                    GAS      = pdf.iloc[0,pdf.keys().str.contains('.*GAS.*PROD.*', regex=True, case=False,na=False)].keys()[0]
                    WTR      = pdf.iloc[0,pdf.keys().str.contains('.*WATER.*VOLUME.*', regex=True, case=False,na=False)].keys()[0]
                    API      = pdf.iloc[0,pdf.keys().str.contains('.*OIL.*GRAVITY.*', regex=True, case=False,na=False)].keys()[0]
                    BTU      = pdf.iloc[0,pdf.keys().str.contains('.*GAS.*BTU.*', regex=True, case=False,na=False)].keys()[0]
                    FM       = pdf.iloc[0,pdf.keys().str.contains('.*Formation.*', regex=True, case=False,na=False)].keys()[0]
                except:
                    print(f'Cannot parse tabels 2 at: {docurl}.')
                    ERROR = 1
                    continue                    

                # Date is date formatted                
                pdf[DATE]=pd.to_datetime(pdf[DATE]).dt.date
                # Sort on earliest date first
                pdf.sort_values(by = [DATE],inplace = True)
                pdf.index = range(1, len(pdf) + 1)
               
                pdf['OIL_RATE'] = pdf[OIL]/pdf[DAYSON]
                pdf['GAS_RATE'] = pdf[GAS]/pdf[DAYSON]
                pdf['WTR_RATE'] = pdf[WTR]/pdf[DAYSON]
                pdf['PROD_DAYS'] = pdf[DAYSON].cumsum()
                
                pdf['GOR'] = pdf[GAS]*1000/pdf[OIL]
                pdf['OWR'] = pdf[OIL]/pdf[WTR]
                pdf['WOR'] = pdf[WTR]/pdf[OIL]
                pdf['OWC'] = pdf[OIL]/(pdf[WTR]+pdf[OIL])
                pdf['WOC'] = pdf[WTR]/(pdf[WTR]+pdf[OIL])

                if pdf[[API]].dropna(how='any').shape[0]>3:
                    OUTPUT.at[UWI,'API_MEAN']         = pdf[API].astype('float').describe()[1]
                    OUTPUT.at[UWI,'API_STD']          = pdf[API].astype('float').describe()[2]
                    
                if pdf[[BTU]].dropna(how='any').shape[0]>3:
                    OUTPUT.at[UWI,'BTU_MEAN']         = pdf[BTU].astype('float').describe()[1]
                    OUTPUT.at[UWI,'BTU_STD']          = pdf[BTU].astype('float').describe()[2]

                if pdf[[OIL,GAS]].dropna(how='any').shape[0]>3:
                    OUTPUT.at[UWI,'Peak_Oil_Date']   = pdf[DATE][pdf[OIL].idxmax()]
                    OUTPUT.at[UWI,'Peak_Oil_Days']   = pdf['PROD_DAYS'][pdf[OIL].idxmax()]
                    OUTPUT.at[UWI,'Peak_Oil_CumOil'] = pdf[OIL][0:pdf[OIL].idxmax()].sum()
                    OUTPUT.at[UWI,'Peak_Oil_CumGas'] = pdf[GAS][0:pdf[OIL].idxmax()].sum()

                    OUTPUT.at[UWI,'Peak_Gas_Date']   = pdf[DATE][pdf[GAS].idxmax()]
                    OUTPUT.at[UWI,'Peak_Gas_Days']   = pdf['PROD_DAYS'][pdf[GAS].idxmax()]
                    OUTPUT.at[UWI,'Peak_Gas_CumOil'] = pdf[OIL][0:pdf[GAS].idxmax()].sum()
                    OUTPUT.at[UWI,'Peak_Gas_CumGas'] = pdf[GAS][0:pdf[GAS].idxmax()].sum()

                    PREPEAKOIL  = pdf.loc[(pdf['PROD_DAYS']-pdf['PROD_DAYS'][pdf[OIL].idxmax()]).between(-100,0),:].index
                    POSTPEAKOIL = pdf.loc[(pdf['PROD_DAYS'][pdf[OIL].idxmax()]-pdf['PROD_DAYS']).between(0,100),:].index
                    POSTPEAKGAS = pdf.loc[(pdf['PROD_DAYS'][pdf[GAS].idxmax()]-pdf['PROD_DAYS']).between(0,100),:].index
                    PEAKGAS = pdf.loc[(pdf['PROD_DAYS'][pdf[GAS].idxmax()]-pdf['PROD_DAYS']).between(-50,50),:].index
                    
                    OUTPUT.at[UWI,'GOR_PrePeakOil']  = pdf.loc[PREPEAKOIL,GAS].sum() * 1000 / pdf.loc[PREPEAKOIL,OIL].sum()
                    OUTPUT.at[UWI,'GOR_PeakGas']     = pdf.loc[PEAKGAS,GAS].sum() * 1000 / pdf.loc[PEAKGAS,OIL].sum()

                    if pdf[[WTR,OIL,GAS]].dropna(how='any').shape[0]>3:
                        OUTPUT.at[UWI,'OWR_PrePeakOil']  = pdf.loc[PREPEAKOIL,OIL].sum()/pdf.loc[PREPEAKOIL,WTR].sum()
                        OUTPUT.at[UWI,'OWR_PostPeakGas'] = pdf.loc[POSTPEAKGAS,OIL].sum()/pdf.loc[POSTPEAKGAS,WTR].sum()                    
                        OUTPUT.at[UWI,'WOC_PostPeakOil'] = pdf.loc[POSTPEAKOIL,WTR].sum() / (pdf.loc[POSTPEAKOIL,WTR].sum()+pdf.loc[POSTPEAKOIL,OIL].sum())
                        OUTPUT.at[UWI,'WOC_PostPeakGas'] = pdf.loc[POSTPEAKGAS,WTR].sum() / (pdf.loc[POSTPEAKGAS,WTR].sum()+pdf.loc[POSTPEAKGAS,OIL].sum())        
                        OUTPUT.at[UWI,'Peak_Oil_CumWtr'] = pdf[WTR][0:pdf[OIL].idxmax()].sum()
                        OUTPUT.at[UWI,'Peak_Gas_CumWtr'] = pdf[WTR][0:pdf[GAS].idxmax()].sum()

                    # Emily uses Month 1 begins at 1st month w/ +14days oil prod
                    if len(pdf[DATE].dropna())>10:
                        MONTH1 = pdf.loc[(pdf[DAYSON]>14) & (pdf[OIL]>0),DATE].min()
                        OUTPUT.at[UWI,'Month1'] = MONTH1

                        if not isinstance(MONTH1,float):
                            pdf['EM_PRODMONTH'] = (pd.to_datetime(pdf[DATE]).dt.year - MONTH1.year)*12+(pd.to_datetime(pdf[DATE]).dt.month - MONTH1.month)+1

##                            OUTPUT.at[UWI,'GOR_MO2-4']  = pdf.loc[(pdf['EM_PRODMONTH']>=2) & (pdf['EM_PRODMONTH']<=4),GAS].sum()*1000 / pdf.loc[(pdf['EM_PRODMONTH']>=2) & (pdf['EM_PRODMONTH']<=4),OIL].sum()
##                            OUTPUT.at[UWI,'GOR_MO5-7']  = pdf.loc[(pdf['EM_PRODMONTH']>=5) & (pdf['EM_PRODMONTH']<=7),GAS].sum()*1000 / pdf.loc[(pdf['EM_PRODMONTH']>=5) & (pdf['EM_PRODMONTH']<=7),OIL].sum()
##                            OUTPUT.at[UWI,'GOR_MO11-13']  = pdf.loc[(pdf['EM_PRODMONTH']>=11) & (pdf['EM_PRODMONTH']<=13),GAS].sum()*1000 / pdf.loc[(pdf['EM_PRODMONTH']>=11) & (pdf['EM_PRODMONTH']<=13),OIL].sum()
##                            OUTPUT.at[UWI,'GOR_MO23-25']  = pdf.loc[(pdf['EM_PRODMONTH']>=23) & (pdf['EM_PRODMONTH']<=25),GAS].sum()*1000 / pdf.loc[(pdf['EM_PRODMONTH']>=23) & (pdf['EM_PRODMONTH']<=25),OIL].sum()
##                            OUTPUT.at[UWI,'GOR_MO35-37']  = pdf.loc[(pdf['EM_PRODMONTH']>=35) & (pdf['EM_PRODMONTH']<=37),GAS].sum()*1000 / pdf.loc[(pdf['EM_PRODMONTH']>=35) & (pdf['EM_PRODMONTH']<=37),OIL].sum()
##                            OUTPUT.at[UWI,'GOR_MO47-49']  = pdf.loc[(pdf['EM_PRODMONTH']>=47) & (pdf['EM_PRODMONTH']<=49),GAS].sum()*1000 / pdf.loc[(pdf['EM_PRODMONTH']>=47) & (pdf['EM_PRODMONTH']<=49),OIL].sum()

##                            if pdf[[WTR,OIL,GAS]].dropna(how='any').shape[0]>3:
##                                OUTPUT.at[UWI,'OWR_MO2-4']  = pdf.loc[(pdf['EM_PRODMONTH']>=2) & (pdf['EM_PRODMONTH']<=4),OIL].sum() / pdf.loc[(pdf['EM_PRODMONTH']>=2) & (pdf['EM_PRODMONTH']<=4),WTR].sum()
##                                OUTPUT.at[UWI,'OWR_MO5-7']  = pdf.loc[(pdf['EM_PRODMONTH']>=5) & (pdf['EM_PRODMONTH']<=7),OIL].sum() / pdf.loc[(pdf['EM_PRODMONTH']>=5) & (pdf['EM_PRODMONTH']<=7),WTR].sum()
##                                OUTPUT.at[UWI,'OWR_MO11-13']  = pdf.loc[(pdf['EM_PRODMONTH']>=11) & (pdf['EM_PRODMONTH']<=13),OIL].sum() / pdf.loc[(pdf['EM_PRODMONTH']>=11) & (pdf['EM_PRODMONTH']<=13),WTR].sum()
##                                OUTPUT.at[UWI,'OWR_MO23-25']  = pdf.loc[(pdf['EM_PRODMONTH']>=23) & (pdf['EM_PRODMONTH']<=25),OIL].sum() / pdf.loc[(pdf['EM_PRODMONTH']>=23) & (pdf['EM_PRODMONTH']<=25),WTR].sum()
##                                OUTPUT.at[UWI,'OWR_MO35-37']  = pdf.loc[(pdf['EM_PRODMONTH']>=35) & (pdf['EM_PRODMONTH']<=37),OIL].sum() / pdf.loc[(pdf['EM_PRODMONTH']>=35) & (pdf['EM_PRODMONTH']<=37),WTR].sum()
##                                OUTPUT.at[UWI,'OWR_MO47-49']  = pdf.loc[(pdf['EM_PRODMONTH']>=47) & (pdf['EM_PRODMONTH']<=49),OIL].sum() / pdf.loc[(pdf['EM_PRODMONTH']>=47) & (pdf['EM_PRODMONTH']<=49),WTR].sum()
##                                                            
##                                OUTPUT.at[UWI,'OWC_MO3']  = pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<3),OIL].sum() / (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=3),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=3),WTR].sum())
##                                OUTPUT.at[UWI,'OWC_MO6']  = pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<6),OIL].sum() / (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=6),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=6),WTR].sum())
##                                OUTPUT.at[UWI,'OWC_MO12']  = pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<12),OIL].sum() / (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=12),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=12),WTR].sum())
##                                OUTPUT.at[UWI,'OWC_MO24']  = pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<24),OIL].sum() / (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=24),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=24),WTR].sum())
##                                OUTPUT.at[UWI,'OWC_MO36']  = pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<36),OIL].sum() / (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=36),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=36),WTR].sum())
##                                OUTPUT.at[UWI,'OWC_MO48']  = pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<48),OIL].sum() / (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<48),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=48),WTR].sum())

                            for i in MonthArray:
                                if max(pdf['EM_PRODMONTH']) >= i:
                                    i_dwn = i-1
                                    i_up = i+1
                                    OUTPUT[str(i)+'Mo_CumOil'] = pdf.loc[(pdf['EM_PRODMONTH']<=i),OIL].sum()
                                    OUTPUT[str(i)+'Mo_CumGas'] = pdf.loc[(pdf['EM_PRODMONTH']<=i),GAS].sum()
                                    OUTPUT[str(i)+'Mo_CumWtr'] = pdf.loc[(pdf['EM_PRODMONTH']<=i),WTR].sum()
                                    if pdf.loc[pdf['EM_PRODMONTH']>=i,[OIL,GAS]].dropna(how='any').shape[0]>=1:
                                        OUTPUT.at[UWI,'GOR_MO'+str(i_dwn)+'-'+str(i_up)]  = pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),GAS].sum()*1000 / pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),OIL].sum()
                                    if pdf.loc[pdf['EM_PRODMONTH']>=i,[OIL,WTR]].dropna(how='any').shape[0]>=1:
                                        OUTPUT.at[UWI,'OWC_MO'+str(i)] = pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),OIL].sum() / (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),WTR].sum())
                                        OUTPUT.at[UWI,'OWR_MO'+str(i_dwn)+'-'+str(i_up)]  = pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),OIL].sum() / pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),WTR].sum()
                                           
                OUTPUT.at[UWI,'Production_Formation'] = '_'.join(pdf[FM].unique())
                    
                
            ERROR = 1
            OUTPUT=OUTPUT.dropna(how='all')
            OUTPUT.index.name = 'UWI'   

            SQL_COLS = '''([UWI] INTEGER PRIMARY KEY
                 ,[BTU_MEAN] REAL
                 ,[BTU_STD] REAL
                 ,[API_MEAN] REAL
                 ,[API_STD] REAL
                 ,[Peak_Oil_Date] DATE
                 ,[Peak_Oil_Days] INTEGER
                 ,[Peak_Oil_CumOil] REAL
                 ,[Peak_Oil_CumGas] REAL
                 ,[Peak_Gas_Date] DATE
                 ,[Peak_Gas_Days] INTEGER
                 ,[Peak_Gas_CumOil] REAL
                 ,[Peak_Gas_CumGas] REAL
                 ,[OWR_PrePeakOil] REAL
                 ,[OWR_PostPeakGas] REAL
                 ,[WOC_PrePeakOil] REAL
                 ,[WOC_PostPeakOil] REAL
                 ,[WOC_PostPeakGas] REAL
                 ,[Peak_Oil_CumWtr] REAL
                 ,[Peak_Gas_CumWtr] REAL
                 ,[Month1] DATE
                 ,[GOR_MO2-4] REAL
                 ,[GOR_MO5-7] REAL
                 ,[GOR_MO11-13] REAL
                 ,[GOR_MO23-25] REAL
                 ,[GOR_MO35-37] REAL
                 ,[GOR_MO47-49] REAL
                 ,[OWR_MO2-4] REAL
                 ,[OWR_MO5-7] REAL
                 ,[OWR_MO11-13] REAL
                 ,[OWR_MO23-25] REAL
                 ,[OWR_MO35-37] REAL
                 ,[OWR_MO47-49] REAL
                 ,[OWC_MO3] REAL
                 ,[OWC_MO6] REAL
                 ,[OWC_MO12] REAL
                 ,[OWC_MO24] REAL
                 ,[OWC_MO36] REAL
                 ,[OWC_MO48] REAL
                 ,[Production_Formation] TEXT
                 ,[3Mo_CumOil] REAL
                 ,[6Mo_CumOil] REAL
                 ,[9Mo_CumOil] REAL
                 ,[12Mo_CumOil] REAL
                 ,[15Mo_CumOil] REAL
                 ,[18Mo_CumOil] REAL
                 ,[21Mo_CumOil] REAL
                 ,[24Mo_CumOil] REAL
                 ,[27Mo_CumOil] REAL
                 ,[30Mo_CumOil] REAL
                 ,[33Mo_CumOil] REAL
                 ,[36Mo_CumOil] REAL
                 ,[39Mo_CumOil] REAL
                 ,[42Mo_CumOil] REAL
                 ,[45Mo_CumOil] REAL
                 ,[48Mo_CumOil] REAL
                 ,[3Mo_CumGas] REAL
                 ,[6Mo_CumGas] REAL
                 ,[9Mo_CumGas] REAL
                 ,[12Mo_CumGas] REAL
                 ,[15Mo_CumGas] REAL
                 ,[18Mo_CumGas] REAL
                 ,[21Mo_CumGas] REAL
                 ,[24Mo_CumGas] REAL
                 ,[27Mo_CumGas] REAL
                 ,[30Mo_CumGas] REAL
                 ,[33Mo_CumGas] REAL
                 ,[36Mo_CumGas] REAL
                 ,[39Mo_CumGas] REAL
                 ,[42Mo_CumGas] REAL
                 ,[45Mo_CumGas] REAL
                 ,[48Mo_CumGas] REAL
                 ,[3Mo_CumWtr] REAL
                 ,[6Mo_CumWtr] REAL
                 ,[9Mo_CumWtr] REAL
                 ,[12Mo_CumWtr] REAL
                 ,[15Mo_CumWtr] REAL
                 ,[18Mo_CumWtr] REAL
                 ,[21Mo_CumWtr] REAL
                 ,[24Mo_CumWtr] REAL
                 ,[27Mo_CumWtr] REAL
                 ,[30Mo_CumWtr] REAL
                 ,[33Mo_CumWtr] REAL
                 ,[36Mo_CumWtr] REAL
                 ,[39Mo_CumWtr] REAL
                 ,[42Mo_CumWtr] REAL
                 ,[45Mo_CumWtr] REAL
                 ,[48Mo_CumWtr] REAL
                 )
                 '''

            TABLE_NAME = "PROD_SUMMARY"
            
    if (OUTPUT.shape[0] > 0) & (SQLFLAG != 0):
##                if path.exists(file):
##                    #OUTPUT.to_csv(file, mode='a', header=False)
##                    SQL_CMD = 'CREATE TABLE IF NOT EXISTS PROD_SUMMARY'+SQL_CREATETABLE
##                else:
##                    #OUTPUT.to_csv(file, mode='w', header=True)
##                    SQL_CMD = 'CREATE TABLE PROD_SUMMARY'+SQL_CREATETABLE
        #try:
        conn = sqlite3.connect(file)
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS ' + TABLE_NAME + ' ' + SQL_COLS)
        tmp = str(OUTPUT.index.max())
        OUTPUT.to_sql(tmp, conn, if_exists='replace', index = True)
        SQL_CMD='DELETE FROM '+TABLE_NAME+' WHERE [UWI] IN (SELECT [UWI] FROM \''+tmp+'\');'
        c.execute(SQL_CMD)
        SQL_CMD ='INSERT INTO '+TABLE_NAME+' SELECT * FROM \''+tmp+'\';'
        c.execute(SQL_CMD)
        conn.commit()
        
        SQL_CMD = 'DROP TABLE \''+tmp+'\';'
        c.execute(SQL_CMD)
        conn.commit()
        conn.close()
               # except: conn.close()
    try:
        conn.close()
    except:
        pass
    return(OUTPUT)

#if __name_ == "__main__":
if 1==1:
    # Initialize constants
    URL_BASE = 'https://cogcc.state.co.us/cogis/ProductionWellMonthly.asp?APICounty=123&APISeq=XNUMBERX&APIWB=00&Year=All'
    DL_BASE = 'http://cogcc.state.co.us/weblink/XLINKX'
    pathname = path.dirname(sys.argv[0])
    adir = path.abspath(pathname)
    dir_add = path.abspath(path.dirname(sys.argv[0]))+"/PROD"
    
    if not path.exists(dir_add):
        makedirs(dir_add)
        
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
    #SHPUWIlist = list(sdf.loc[sdf.Delta>2000,'UWI10'].unique())
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

    if processors > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
            f = {executor.submit(func, a): a for a in data}        
        RESULT=pd.DataFrame()
        for i in f.keys():
            RESULT=pd.concat([RESULT,i.result()],axis=0,join='outer',ignore_index=True)
    else:
        RESULT = GROUP_IN_TC_AREA(poly_df,sdf)

    UWIlist = (RESULT.loc[RESULT.TEST==True,'API'].apply(UWI10)*10000).astype(str).str.zfill(14).to_list()

    # ADD VERTICAL DATA
    wfile = 'Wells.shp'
    wdf = shp.Reader(wfile)
    wdf = read_shapefile(wdf)
    wdf['UWI10'] = wdf.API_Label.str.replace(r'[^0-9]','',regex=True).apply(UWI10)
    wdf['Test'] = False
    
    wwdf = wdf.loc[(~wdf.UWI10.isin(pd.Series(UWIlist).apply(UWI10))) & (wdf.Max_MD >= 5000) & (wdf.Spud_Date > datetime.date(1900,1,1))]
    
    data=np.array_split(wwdf,processors)
    if processors > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
            f = {executor.submit(func, a): a for a in data}        
        RESULT=pd.DataFrame()
        for i in f.keys():
            RESULT=pd.concat([RESULT,i.result()],axis=0,join='outer',ignore_index=True)
    else:
        RESULT = GROUP_IN_TC_AREA(poly_df,wwdf)
    UWIlist2 = (RESULT.loc[RESULT.TEST==True,'API'].apply(UWI10)*10000).astype(str).str.zfill(14).to_list()
    UWIlist = UWIlist+UWIlist2


    #UWIlist = np.array_split(UWIlist,2)[0]
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
            f = {executor.submit(Get_ProdData, a): a for a in data}
        RESULT=pd.DataFrame()
        for i in f.keys():
            RESULT=pd.concat([RESULT,i.result()],axis=0,join='outer',ignore_index=False)
    else:
        RESULT = Get_ProdData(UWIlist)
  
    #RESULT = Get_Scouts(UWIlist,SQLDB)

    
    outfile = dir_add+'/PROD_PULL_'+datetime.datetime.now().strftime("%m%d%Y")+".csv"
    RESULT.to_csv(outfile,sep=',')

##    SQLDB = '\\\Server5\\Verdad Resources\\Operations and Wells\\Geology and Geophysics\\WKR\\Decline_Parameters\\DeclineParameters_v200\\prod_data.db'
##    pd_sql_types={'object':'TEXT',
##                  'int64':'INTEGER',
##                  'float64':'REAL',
##                  'bool':'TEXT',
##                  'datetime[64]':'TEXT',
##                  'timedelta[ns]':'REAL',
##                  'category':'TEXT'
##        }
##    df_dtypes = RESULT.dtypes.astype('str').map(pd_sql_types).to_dict()
##    with sqlite3.connect(SQLDB) as conn:
##        TABLE_NAME = 'CO_SCOUT'
##        RESULT.to_sql(TABLE_NAME,conn,
##                      if_exists='replace',
##                      index=False,
##                      dtype=df_dtypes)


    

    
#np.array(transform(4269, 2876,-104.892422,39.999196,always_xy=True))
#np.array(transform(4269, 2876,-104.890175,39.999069,always_xy=True))
########################
#### END OF ROUTINE ####
########################
#

def listjoin(list_in, sep="_"):
    list_in = list(set(list_in))
    list_in = sorted(list_in)
    list_in = [s.strip() for s in list_in]
    str_out = sep.join(list_in)
    return str_out
    
#"_".join(OUTPUT.Production_Formation.unique())

FM_DICT = {re.compile('NIO[BRA]*',re.I):'NIOBRARA',
           re.compile('SHARON[ \-_]*SPRINGS',re.I):'NIOBRARA',
           re.compile('F[OR]*T[ \-_]*H[AYS]*',re.I):'CODELL',
           re.compile('TIMPAS',re.I):'CODELL',
           re.compile('COD[DEL]*',re.I):'CODELL',
           re.compile('CARLILE',re.I):'CODELL',
           re.compile('J[ _\-0-9]*S[A]*ND',re.I):'JSAND',
           re.compile('(^|[ \-])(J[ SAND\-]*)($|\-)'):r'\1JSAND\3',
           re.compile('D[ _\-0-9]*S[A]*ND',re.I):'DSAND',
           re.compile('(^|[ \-])(D[ SAND\-]*)($|\-)'):r'\1DSAND\3',
           re.compile('(^|_)J_',re.I):r'\1JSAND_',
           re.compile('(^|_)D[ &_]+',re.I):r'\1DSAND_'
           }

#if 1==1:
#    RESULT.Production_Formation = backup.copy()
RESULT.Production_Formation = RESULT.Production_Formation.str.replace("-","_")
RESULT.Production_Formation = RESULT.Production_Formation.replace(FM_DICT,regex=True)
    
RESULT.Production_Formation = RESULT.Production_Formation.str.split("_")
RESULT.Production_Formation = RESULT.Production_Formation.apply(listjoin)

RESULT['ProdFmList'] = RESULT.Production_Formation.str.split("_")

FMLIST = ['NIOBRARA','CODELL','JSAND']
RESULT['ProdFmList'] = RESULT.Production_Formation
for i in FMLIST:
    RESULT[i+'_PRODUCTION']=0
    RESULT.loc[RESULT.Production_Formation.str.contains(i),i+'_PRODUCTION']=1
    RESULT['ProdFmList'] = RESULT.ProdFmList.str.replace(i,'')

RESULT['OTHER_PRODUCTION'] = 0
RESULT.loc[RESULT['ProdFmList'].str.contains(r'[A-Z]',case=False,regex=True),'OTHER_PRODUCTION'] = 1
RESULT = RESULT.drop(columns=['ProdFmList'])

outfile = 'PROD_PULL_'+datetime.datetime.now().strftime("%m%d%Y")+".csv"
RESULT.to_csv(dir_add+'/'+outfile)

### list of fm names
##fmlist=[]
##for x in RESULT['ProdFmList']:
##    fmlist.extend(x)
##    fmlist = list(set(fmlist))

        


# DEFINE FUNCTION FOR ZONE DETECTION
# def IsZone(ZoneArray,ZoneName(s),OnlyFlag)
# parse array into list of all zones
# find all zones matching criteria
# if only flag = 1, then return fields with only search zone
# if only flag = 0, then return fields with any search zone
# "_".join(OUTPUT.Production_Formation.drop_duplicates())
#list(set("_".join(OUTPUT.Production_Formation.drop_duplicates()).split("_")))
# TIMPAS, NIOBRARA, FORT HAYS, CODELL, CARLILE
# J SAND
# COLUMN: NIO-COD: r"NIO|COD|(FORT or FT) H|CARL|TIMPAS
# COLUMN: J :"J"


#dir_add = '\\\\Server5\\Verdad Resources\\Operations and Wells\\Geology and Geophysics\\WKR\\Decline_Parameters\\DeclineParameters_v200\\PROD'
#RESULT=pd.DataFrame()
#for f in listdir(dir_add):
#    df = pd.read_csv(dir_add+'\\'+f)
#    RESULT = pd.concat([RESULT,df],axis=0,join='outer',ignore_index=True)


