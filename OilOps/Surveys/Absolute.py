from pyproj import Proj, transform, CRS
from pyproj import Transformer
import sys, math, datetime, multiprocessing, concurrent.futures, math, re
import pandas as pd
import numpy as np
import sqlite3
from os import path


# v003 fixed chunksize = 0 for small well sets

def UWI10(num):
    num=int(num)
    while num > 9e9:
        num = math.floor(num/100)
    num = int(num)
    return num


def Find_Str_Locs(df_in,string):
    # takes data table and finds index locations for all matches
    if isinstance(string,str):
        string=[string]
    Output=pd.DataFrame({'Title':[],'Columns':[],'Rows':[]}).astype(object)
    Output.Title=pd.Series([w.replace(' ','_').replace('#','NUMBER').replace('^','') for w in string]).astype(str)
    #Output.astype({'Title': 'object','Columns': 'object','Rows': 'object'}).dtypes
    Output.astype(object)
    for ii, item in enumerate(string):
        Output.iloc[ii,1] = [(lambda x: df_in.index.get_loc(x))(i) for i in df_in.loc[(df_in.select_dtypes(include=[object]).stack().str.contains(f'.*{item}.*', regex=True, case=False,na=False).unstack()==True).any(axis='columns'),:].index.values ]
        Output.iloc[ii,2] = [(lambda x: df_in.index.get_loc(x))(i) for i in df_in.loc[:,(df_in.select_dtypes(include=[object]).stack().str.contains(f'.*{item}.*', regex=True, case=False,na=False).unstack()==True).any(axis='rows')].keys().values]
    Output.Title=pd.Series([w.replace(' ','_').replace('#','NUMBER').replace('^','') for w in string]).astype(str)
    return (Output)

def str2num(str_in):
    if not isinstance(str_in, (int, float)):
        val = re.sub(r'[^0-9\.]','',str(str_in))
        if val == '':
            return None
        try:
            val = float(val)
        except:
            val = None
    else:
        val = str_in
    return val

def API2INT(val_in,length = 10):
    if val_in is None:
        return None
    try:
        if math.isnan(val_in):
            return None
    except:
        pass
    val = str2num(val_in)
    lim = 10**length-1
    highlim = 10**length-1 #length digits
    lowlim =10**(length-2) #length -1 digits
    while val > highlim:
        val = math.floor(val/100)
    while val < lowlim:
        val = val*100
    val = int(val)
    return(val)

def APIfromFilename(ffile,UWIlen=10):
    lst = re.findall(r'UWI[0-9]{9,}',ffile, re.I)
    if len(lst)==0:
        lst = re.findall(r'[0-9]{9,}',ffile)
    else:
        lst[0] = re.sub('UWI','',lst[0],re.I)
    return API2INT(lst[0],length=UWIlen) if len(lst)>0 else None

pathname = path.dirname(sys.argv[0])
adir = path.abspath(pathname)


#survey file
SFile = '\\\\Server5\\Verdad Resources\\Operations and Wells\\Geology and Geophysics\\WKR\\Decline_Parameters\\DeclineParameters_v200\\SURVEYS\\JOINED_SURVEY_FILE_V2.csv'
SFile = 'JOINED_SURVEY_FILE_V2_MERGE_20225906'
#sdf = pd.read_json(SFile+'.JSON')
sdf = pd.read_parquet(f'{SFile}.PARQUET')

# REMOVE DUPLICATE DATA SOURCED FROM SEPARATE FILES
mask = sdf.sort_values(by=['FILE','UWI','MD'])[['MD', 'INC', 'AZI', 'TVD', 'NORTH_Y', 'EAST_X', 'UWI']].drop_duplicates().index
#sdf = sdf.loc[mask,:]
FILES = sdf.loc[mask].groupby(by='FILE').UWI.count().loc[sdf.loc[mask].groupby(by='FILE').UWI.count()>5].index
sdf = sdf.loc[sdf.FILE.isin(FILES)]

sqldb = '\\\\Server5\Verdad Resources\\Operations and Wells\\Geology and Geophysics\\WKR\\Decline_Parameters\\DeclineParameters_v200\\CO_3_2.1.sqlite'
sqldb = path.join(path.dirname(adir),'CO_3_2.1.sqlite')
#sqldb = 'CO_3_2.1.sqlite'

with sqlite3.connect(sqldb) as conn:
##    cursor = conn.cursor()
##    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
##    print(cursor.fetchall())
    df = pd.read_sql_query('SELECT API,Latitude,Longitude FROM WELL',conn)
    #c = conn.cursor()	
    #df = c.execute('SELECT API,Latitude,Longitude FROM WELL ')

#df[['Y','X']]=df.apply(lambda x: transform(4269,2876,x.iloc[2],x.iloc[1],always_xy=True), axis=1).apply(pd.Series)

k1 = df.keys()[df.keys().str.contains('.*API|UWI.*', regex=True, case=False,na=False)]
k1=k1[0]
k1_num = df.keys().get_loc(k1)
df[k1]=df[k1].astype('int64')

k2 = sdf.keys()[sdf.keys().str.contains('.*API|UWI.*', regex=True, case=False,na=False)]
k2=k2[0]
k2_num = sdf.keys().get_loc(k2)
try:
    sdf.loc[sdf[k2].isna(),k2] = sdf.loc[sdf[k2].isna()].FILE.apply(APIfromFilename, UWIlen=14)
except:
    pass

mask = sdf[k2].isna()
sdf.loc[mask,k2] = sdf.loc[mask,'FILE'].apply(APIfromFilename)
sdf[k2] = sdf[k2].apply(API2INT,args = (14,))

sdf[k2]=sdf[k2].astype('float')
df[k1]=df[k1].astype('float')

df=df.loc[df[k1].isin(sdf[k2])]

processors = max(1,multiprocessing.cpu_count())
chunksize = min(5000,max(1,int(df.shape[0]/processors)))
print ("starting map function")
# outfile = "BTU_API_PULL_"+datetime.datetime.now().strftime("%d%m%Y")+".csv"
batch = int(df.shape[0]/chunksize)
processors = min(processors,batch)
data = np.array_split(df,batch)

def XYtransform(df_in, epsg1 = 4269, epsg2 = 2878):
    #2876
    df_in=df_in.copy()
    transformer = Transformer.from_crs(epsg1, epsg2,always_xy =True)
    df_in[['X','Y']]=df_in.apply(lambda x: transformer.transform(x.iloc[2],x.iloc[1]), axis=1).apply(pd.Series)
    #df_in[['X','Y']]=df_in.apply(lambda x: transform(epsg1,epsg2,x.iloc[2],x.iloc[1],always_xy=True), axis=1).apply(pd.Series)
    return df_in

with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
        f = {executor.submit(XYtransform, a): a for a in data}

RESULT=pd.DataFrame()
for i in f:
    RESULT=pd.concat([RESULT,i.result()],axis=0,join='outer',ignore_index=True)

#df[['Y','X']]=df.apply(lambda x: transform(4269,2876,x.iloc[2],x.iloc[1],always_xy=True), axis=1).apply(pd.Series)

ct = sdf.keys().get_loc('FILE')
sdf.iloc[:,0:(ct+1)]
sdf = sdf.iloc[:,0:(ct+1)].merge(RESULT,how = 'left', left_on = k2, right_on = k1)

sdf['XPATH']=sdf.EAST_X+sdf.X
sdf['YPATH']=sdf.NORTH_Y+sdf.Y

#sdf.to_csv('JOINED_ABS_SURVEYS_TC.csv', index=False)
sdf.to_json('JOINED_ABS_SURVEYS_TC.json')
sdf.to_parquet('JOINED_ABS_SURVEYS_TC.PARQUET')

# df[k].astype('int64')
# sdf.UWI.isin(df.API)
# x = sdf.merge(df,how = 'left', left_on = 'UWI', right_on = 'API')
# df[['Y','X']]=df.iloc[[3,5,20],:].apply(lambda x: transform(4269,2876,x.iloc[2],x.iloc[1],always_xy=True), axis=1).apply(pd.Series)

# XY_Dict = dict()
# for i in range(0,df.shape[0]):
    # XY_Dict[df.iloc[i,k_num]] = list(transform(4269,2876,df.iloc[i,2],df.iloc[i,1],always_xy=True))
	
# a=pd.Series(sdf.iloc[:,1].astype('int64').unique())
# ldf.loc[ldf.API.astype('int64').isin(pd.Series(a))]
# # ERROR CHECK FOR ALL LOCATIONS
# a.loc[~a.isin(ldf.API.astype('int64'))].shape[0]
# # BUILD DICTIONARY OF API AND X,Y SHL'sdf
# # APPLY DICTIONARY TO ENTIRE SURVEY TABLE TO CREATE ABSOLUTE X,Y FROM RELATIVE

