import pandas as pd
import numpy as np
import math, re, datetime
from os import path
# requires xlrd, openpyxl


#v003 added formation assignments to production table

# DEFINE FUNCTION FOR ADD TABLE TO MAIN TABLE
# FUNC(TBL1,TBL2, LEFT/OUTER, ON

#DEFINE FUNCTION FOR FINDING COLUMN in KEYS BY PARTIAL STRING
def Regular_FM(df_series,FM_DICT):
    FM_LIST = list(FM_DICT.values())
    FM_LIST = list(set(pd.Series(FM_LIST).replace({r'\\[0-9]':''},regex=True).str.strip().to_list()))

    dS = df_series.copy(deep=True)
    dS = dS.replace(FM_DICT,regex=True)
    dS = dS.to_frame()
    dS['CLEARED_FM'] = dS.iloc[:,0]

    for FM in FM_LIST:
        TITLE = FM+'_PRODUCTION'
        dS[TITLE]=0
        dS.loc[dS.iloc[:,0].str.contains(FM),TITLE]=1
        dS['CLEARED_FM'] = dS['CLEARED_FM'].str.replace(FM,'',regex=False).str.strip()

    dS['OTHER_PRODUCTION']=0
    dS['CLEARED_FM'] = dS['CLEARED_FM'].str.replace(re.compile('[^A-Z0-9]',re.I),'',regex=True)
    dS.loc[dS['CLEARED_FM'].str.contains(r'[A-Z]',regex=True),'OTHER_PRODUCTION']=1

    del dS['CLEARED_FM']

    return dS

def API10(num):
    if isinstance(num,str):
        num=re.sub('[^0-9]+','',num)
        num = re.sub('^[0]+','',num)
        num = int(num)
    if math.isnan(num):
        return(None)
    num = int(num)
    while num>9e9:
        num = math.floor(num/100)
    num = int(num)
    return(num)

def UWI10(num):
    if isinstance(num,str):
        num = re.sub(r'^0+','',num.strip())
    if math.isnan(num):
        return(None)    
    num=int(num)
    while num > 9e9:
        num = math.floor(num/100)
    #while num < 1e8:
    #    num = math.floor(num*100)
    num = int(num)
    return num

def UWI12(num):
    if isinstance(num,str):
        num = re.sub(r'^0+','',num.strip())
    if math.isnan(num):
        return(None)  
    num = int(num)
    while num > 9e11:
        num = math.floor(num/100)
    #while num < 1e10:
    #    num = math.floor(num*100)
    num = int(num)
    return num

def UWI14(num):
    if isinstance(num,str):
        num = re.sub(r'^0+','',num.strip())
    if math.isnan(num):
        return(None)
    num=int(num)
    while num > 9e13:
        num = math.floor(num/100)
    #while num < 1e12:
    #    num = math.floor(num*100)
    num = int(num)
    return num


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

if 1==1:   
    SQLFILE = 'SQL_WELL_SUMMARY.PARQUET'
    SCOUTFILE = path.join('SCOUTS','SCOUT_PULL_SUMMARY_10122021.parquet')
    SPACINGFILE = path.join('SURVEYS','3D_Spacing_07042022_0824.PARQUET') #01/17/2022
    PRODSUMMARYFILE = path.join('PROD','PROD_PULL_SUMMARY_01172022.parquet') #01/17/2022
    FRACFOCUSFILE = 'FracFocusTables.PARQUET'
    PETRAFILE = 'PETRA_HZ_EXPORT_011822.CSV'
    PETRAFILE_VERT = 'PETRA_VERT_EXPORT_052621.CSV'
    EURFILE = 'TypeCurveReview_21Q3_Oneline_20211012.xlsx'

    #GORFILE = 'GOR_GLR.csv'

    #ADD SQL SCOUT if 1==1:
    #df_SQL   = pd.read_csv(SQLFILE,low_memory=False)   # MULTIPLE UWI10 REPEATES
    #df_SCOUT = pd.read_csv(SCOUTFILE,low_memory=False) # ONE UWI10 REPEAT
    df_SQL   = pd.read_parquet(SQLFILE)
    df_SCOUT   = pd.read_parquet(SCOUTFILE) 

    date_keys = list(df_SCOUT.keys()[df_SCOUT.keys().str.lower().str.contains('date')])
    #df_SCOUT   = pd.read_json(SCOUTFILE,convert_dates = date_keys)

    print('SQL & SCOUT')

    try:
        df_SQL.drop(columns = ('Unnamed: 0'), inplace = True)
    except:
        pass
    df_SQL['UWI10'] = df_SQL.API.apply(API10)
    df_SCOUT['UWI10']=df_SCOUT.UWI.apply(API10)    
    df_SQL = df_SQL.drop_duplicates()
    df_SCOUT = df_SCOUT.drop_duplicates()


    #df_SQL Repeat Repair
    mask = df_SQL.drop(['API','API14x'], axis=1).drop_duplicates().index
    df_SQL = df_SQL.loc[mask]

    keys = ['PeakOil_Date','PeakGas_Date','PeakWater_Date','TreatmentDate','SpudDate','FirstCompDate']
    mask = df_SQL[keys].dropna(how='all').index
    df_SQL = df_SQL.loc[mask]
##    
##    if 1==1:
##        cts = df_SQL.UWI10.value_counts()
##        dbl = cts.loc[cts>1].index.values
##        problem = df_SQL.loc[df_SQL.UWI10.isin(dbl)].copy()
##
##        for u in problem.UWI10.unique():
##            print(u)
##            x = df_SQL.loc[df_SQL.UWI10==u, df_SQL.loc[df_SQL.UWI10==u].iloc[0,:]!=df_SQL.loc[df_SQL.UWI10==u].iloc[1,:]].dropna(how='all',axis=1)
##            for i in range(0,x.shape[0]):
##            x.iloc[i,:].dropna().shape
##	


    OUT = pd.merge(df_SQL,df_SCOUT,how='outer',on='UWI10',suffixes =('_SQLSCOUT','_COSCOUT'))
    del df_SQL, df_SCOUT
  
    # ADD WELL XYZ
    #df_XYZ   = pd.read_csv(SPACINGFILE,low_memory=False)
    #df_XYZ   = pd.read_json(SPACINGFILE)
    df_XYZ   = pd.read_parquet(SPACINGFILE)
    #date_keys = list(df_XYZ.keys()[df_XYZ.keys().str.lower().str.contains('date')])
    #df_XYZ   = pd.read_json(SPACINGFILE,convert_dates = date_keys) 
    
    print('XYZ file')
    
    df_XYZ   =  df_XYZ.drop_duplicates()
    m = df_XYZ.UWI10.dropna().index
    OUT = pd.merge(OUT,df_XYZ.loc[m],how='outer',on='UWI10',suffixes = ('','_XYZ'))

    del df_XYZ

    # ADD PRODUCTION SUMMARY
    #df_PROD  = pd.read_csv(PRODSUMMARYFILE,low_memory=False) # NO UWI10 Repeats
    #df_PROD  = pd.read_json(PRODSUMMARYFILE, convert_dates = date_keys)
    df_PROD  = pd.read_parquet(PRODSUMMARYFILE)
    date_keys = list(df_PROD.keys()[df_PROD.keys().str.lower().str.contains('date')])
    if 'Month1' not in date_keys:
        date_keys = date_keys + ['Month1']

    try:
        df_PROD.drop(columns = ('Unnamed: 0'), inplace = True)
    except:
        pass
    df_PROD['UWI10'] = df_PROD.UWI.apply(API10)

    print('Production')
     
    FORMATION_DICT = {re.compile('NIO[BRA]*',re.I):'NIOBRARA',
           re.compile('SHARON[ \-_]*SPRINGS',re.I):'NIOBRARA',
           re.compile('F[OR]*T[ \-_]*H[AYS]*',re.I):'CODELL',
           re.compile('TIMPAS',re.I):'CODELL',
           re.compile('COD[DEL]*',re.I):'CODELL',
           re.compile('CARLILE',re.I):'CODELL',
           re.compile('J[ _\-0-9]*SA*ND',re.I):'JSAND',
           re.compile('(^|[ _\-&])(J[\-_ ])($|(?!S[A]*ND))',re.I):r'\1JSAND\3'}

    Prod_Fields = Regular_FM(df_PROD['Production_Formation'],FORMATION_DICT)
    df_PROD=pd.concat([df_PROD, Prod_Fields.iloc[:,1:]], axis=1)

    OUT = pd.merge(OUT,df_PROD,how = 'outer', on='UWI10', suffixes = ('','_XYZ'))

    del df_PROD

    # ADD FRAC FOCUS
    #df_FRACFOCUS = pd.read_csv(FRACFOCUSFILE,low_memory=False)
    df_FRACFOCUS = pd.read_parquet(FRACFOCUSFILE)
    
    print('Frac Focus')
  
    FF_Summary = pd.DataFrame()
    FF_Summary['APINumber'] = df_FRACFOCUS['APINumber'].unique()
    FF_Summary['UWI10'] = FF_Summary['APINumber'].apply(API10)

    Purpose_List = {'Gel Mass':'Gel','Friction Reducer Mass':'Friction','Diverter Mass':'divert'}
    for i in Purpose_List:
        if ('df' in locals()) or ('df' in globals()):
            del df
        term = Purpose_List[i]
        df_FRACFOCUS[i] = False
        df_FRACFOCUS.loc[df_FRACFOCUS.Purpose.astype('str').str.contains(term,flags=re.IGNORECASE,regex=True),i] = True
        df = df_FRACFOCUS.loc[df_FRACFOCUS[i]==True].groupby(by=['APINumber'])['MassIngredient'].sum().rename(i)
        FF_Summary = pd.merge(FF_Summary,df,how='outer',on='APINumber')
        FF_Summary = FF_Summary.fillna(0)
    OUT = pd.merge(OUT,FF_Summary,how = 'outer', on='UWI10', suffixes = ('','_FracFocus'))

    del FF_Summary

    df_FRACFOCUS['UWI10']=df_FRACFOCUS.APINumber.apply(API10)
    df_FF=df_FRACFOCUS[['UWI10','TotalBaseWaterVolume']].groupby('UWI10').max()
    df_FF = df_FF.merge(
        df_FRACFOCUS.loc[(df_FRACFOCUS.Purpose.str.contains(re.compile('gel',re.I),regex=True)==True) & (df_FRACFOCUS.MassIngredient>0),['UWI10','MassIngredient']].groupby('UWI10').sum()
        ,how = 'left'
        ,on = 'UWI10')

    df_FF = df_FF.rename(columns={'MassIngredient':'Gel_Mass'})
    df_FF=df_FF.drop_duplicates()
    df_FF = df_FF.fillna('')

    OUT = pd.merge(OUT,df_FF,how='left',on='UWI10',suffixes = ('','_FRACFOCUS'))

    del df_FF,df_FRACFOCUS

    # Add Petra
    #df_PETRA = pd.read_csv('Petra_HZ_Target.csv', header = [0,1])
    df_PETRA = pd.read_csv(PETRAFILE, header = [0,1],low_memory=False)
    df_PETRA.columns = df_PETRA.columns.map('_'.join)
    df_PETRA['UWI10']=df_PETRA.iloc[:,0].apply(API10)

    print('Petra')
    
    keys = df_PETRA.columns.to_list()
    pattern = re.compile(r'(_)UNNAMED.*', re.IGNORECASE)
    df_PETRA.columns = [re.sub(pattern, '', file) for file in keys]

    OUT = pd.merge(OUT,df_PETRA,how='outer',on='UWI10',suffixes = ('','_PETRA'))

    # Add Petra Vertical data
    df_PETRA_V = pd.read_csv(PETRAFILE_VERT, header = [0,1],low_memory=False)
    df_PETRA_V.columns = df_PETRA_V.columns.map('_'.join)
    df_PETRA_V['UWI10']=df_PETRA_V.iloc[:,0].apply(API10)
    keys = df_PETRA_V.columns.to_list()
    pattern = re.compile(r'(_)UNNAMED.*', re.IGNORECASE)
    df_PETRA_V.columns = [re.sub(pattern, '', file) for file in keys]
    OUT = pd.merge(OUT,df_PETRA_V,how='outer',on='UWI10',suffixes = ('','_PETRAV'))

    del df_PETRA
   
    # ADD EUR
    df_EUR = pd.read_excel(EURFILE)
    df_EUR['UWI10'] = df_EUR.API10.apply(API10)
    print('EUR file')
    OUT = pd.merge(OUT,df_EUR,how='left',on='UWI10',suffixes = ('','_EUR'))

    del df_EUR
    
    # ADD EXTRA FILE
    #GORFILE = 'GOR_GLR.csv'

    #df_GOR = pd.read_csv(GORFILE,index_col=0,low_memory=False)
    #df_GOR['UWI10']=df_GOR.index
    #df_GOR['UWI10']=df_GOR['UWI10'].apply(API10)
    #OUT = pd.merge(OUT,df_GOR,how='left',on='UWI10',suffixes = ('','_GOR'))

if 1==1:
    OUT = OUT.dropna(axis=0,how='all')
    OUT = OUT.dropna(axis=1,how='all')

    OUT.UWI10 = OUT[['StateProducingUnitKey','API14x','API','UWI','UWI10']].fillna(method='bfill', axis=1).iloc[:, 0].apply(API10)

    #remove repeated columns
    OUT = OUT=OUT.loc[:,~OUT.columns.duplicated()]

    #remove repeated rows
    OUT = OUT.drop_duplicates()

    #repair data types
    OUT = DF_UNSTRING(OUT)
    print('Unstring finished')
    
    #CALC USEABLE VALUES
    TC_OUT = OUT.copy()

#>>> TC_DF=pd.read_excel('TypeCurveWells_All.xlsx')
#>>> TC_UWIS = TC_DF.API_Label.apply(API10).copy()
#>>> TC_OUT = OUT.loc[OUT.UWI10.isin(TC_UWIS),:].copy()
#>>> TC_OUT=DF_UNSTRING(TC_OUT)

    dXY = ['dxy1','dxy2','dxy3','dxy4','dxy5']
    dZ = ['dz1','dz2','dz3','dz4','dz5']

##def LimitFilter(num,Limit_Type='LOW', limit = 0):
##    if upper(Limit_Type) == 'LOW':
##        if num<limit:
##            num = np.nan
##    else:
##        if num>limit:
##            num = np.nan


#OUT
if 1==1:
    OUT['Rel_Nearest'] = (abs((np.array(OUT[dXY])/500)**2 + (np.array(OUT[dZ])/200)**2).min(axis=1))**(1/2)

    OUT['dXY1_Left'] = abs(OUT[dXY].clip(upper=0)).replace(0,np.nan).min(axis=1)
    LEFT_MIN = abs(OUT[dXY].clip(upper=0)).replace(0,np.nan).idxmin(axis=1)

    OUT['dXY1_Right'] = abs(OUT[dXY].clip(lower=0)).replace(0,np.nan).min(axis=1)
    RIGHT_MIN = abs(OUT[dXY].clip(lower=0)).replace(0,np.nan).idxmin(axis=1)


    #COL = RIGHT_MIN.str.replace('xy','z')
    #mask = RIGHT_MIN.str.replace('xy','z').dropna().index
    #COL = LEFT_MIN.str.replace('xy','z')
    
    for i in OUT.index:
        if isinstance(LEFT_MIN[i],str):
            try:
                COL = re.sub('xy','z',LEFT_MIN[i])
                OUT.loc[i,'dZ1_Left'] = OUT.loc[i,COL]
            except: pass
        if isinstance(RIGHT_MIN[i],str):
            try:
                COL = re.sub('xy','z',RIGHT_MIN[i])
                OUT.loc[i,'dZ1_RIGHT'] = OUT.loc[i,COL]
            except: pass

    KeyData_Dict = {
        'Completion_Date':r'(?:Stim|Treat|Comp|Job).*Date',
        'FirstProduction_Date':r'First.*Prod|1st.*prod|month1',
        'Lateral_Length':r'perf.*int|lat.*len',
        'STIM_FLUID':r'STIM.*FLUID|TOTAL.*FLUID',
        'STIM_PROPPANT':r'TOTAL.*PROPPANT',
        'PeakOil_CumOil':r'peak.*oil.*cum.*oil',
        'PeakOil_CumGas':r'peak.*oil.*cum.*gas'
        }

    keys = OUT.keys()
  
    #PROBLEM HERE (date formatting getting lost, fixed with Parquet files)
    for k in KeyData_Dict:
        kkey = KeyData_Dict[k]
        pattern = re.compile(kkey,re.I)
        k_sub = list(filter(pattern.match, keys))
        #print(k_sub)
       # if 'date' in k.lower():
       #     OUT[k_sub].apply(lambda x: pd.to_datetime(x).dt.date).max(axis=1)
        OUT[k] = OUT[k_sub].max(axis=1)

    # PERF INTERVAL
    OUT['PerfInterval'] = None
    mask = (abs(OUT.INTERVAL_TOP - OUT.INTERVAL_BOTTOM) > 0)
    OUT.loc[mask,'PerfInterval'] = abs(OUT.loc[mask,'INTERVAL_TOP'] - OUT.loc[mask,'INTERVAL_BOTTOM'])
    
    # API10
    OUT['API10'] = OUT['UWI10'].apply(UWI10)    

    # API12 if True:
    keys = ['API', 'APINumber','UWI/API','UWI/API_PETRAV','API14x', 'UWI10']
    keys = list(set(OUT.keys().tolist()) & set(keys))
    mask = OUT.index
    for k in keys:
        if (pd.to_numeric(OUT.loc[mask,k], errors = 'coerce').dropna().min()>1e10) & (pd.to_numeric(OUT.loc[mask,k], errors = 'coerce').dropna().min()<(1e13-1)):
            OUT.loc[mask,'API12'] = OUT.loc[mask,k].apply(UWI12)
            mask = OUT['API12'].isna()

    # API14
    keys = ['API', 'APINumber','UWI/API','UWI/API_PETRAV','API14x', 'UWI10']
    keys = list(set(OUT.keys().tolist()) & set(keys))
    mask = OUT.index
    for k in keys:
        if (pd.to_numeric(OUT.loc[mask,k], errors = 'coerce').dropna().min()>1e12) & (pd.to_numeric(OUT.loc[mask,k], errors = 'coerce').dropna().min()<(1e14-1)):
            OUT.loc[mask,'API14'] = OUT.loc[mask,k].apply(UWI14)
            mask = OUT['API14'].isna()

    # WELL NAME
    keys = ['WellName','WELLNAME','WELL_NAME/NO', 'WELLNAME_PETRAV']
    keys = list(set(OUT.keys().tolist()) & set(keys))
    mask = OUT.index
    OUT['WELL_NAME']=None
    for k in keys:
        OUT.loc[mask,'WELL_NAME'] = OUT.loc[mask,k]
        mask = OUT['WELL_NAME'].isna()

    # WELL NUMBER if 1==1:
    keys = ['WellNumber', 'WELLNO', 'WellNumber']
    keys = list(set(OUT.keys().tolist()) & set(keys))
    mask = OUT.index
    OUT['WELL_NUMBER']=None
    for k in keys:
        OUT.loc[mask,'WELL_NUMBER'] = OUT.loc[mask,k]
        mask = OUT['WELL_NUMBER'].isna()

    # OPERATOR if 1==1:
    keys = ['OperatorName','OPERATOR','Operator']
    keys = list(set(OUT.keys().tolist()) & set(keys))
    mask = OUT.index
    OUT['OPERATOR_']=None
    for k in keys:
        OUT.loc[mask,'OPERATOR_'] = OUT.loc[mask,k]
        mask = OUT['OPERATOR_'].isna()

    # STIM_FLUID if True:
    OUT['STIM_FLUID']=OUT['TotalBaseWaterVolume']/42
    mask = (OUT['STIM_FLUID']>1000)==False
    OUT.loc[mask,'STIM_FLUID']=OUT.loc[mask,'TotalFluid']
    mask = (OUT['STIM_FLUID']>1000)==False
    OUT.loc[mask,'STIM_FLUID']=OUT.loc[mask,'TOTAL_FLUID_USED']
    #mask = (TC_OUT['STIM_FLUID']>1000)==False
    #TC_OUT.loc[mask,'STIM_FLUID']=TC_OUT.loc[mask,'TREAT_FLUID']

    # FLUID_INTENSITY & FLUID_INTENSITY_TYPE
    mask = OUT['PerfInterval'].isna()
    OUT.loc[~mask,'STIM_FLUID_INTENSITY'] = OUT.loc[~mask,'STIM_FLUID']/OUT.loc[~mask,'PerfInterval']
    OUT.loc[mask,'STIM_FLUID_INTENSITY'] = OUT.loc[mask,'STIM_FLUID'] / abs(OUT.loc[mask,'LatLen'])

    # STIM_PROPPANT
    keys = ['TotalAmount_PROPPANT','TOTAL_PROPPANT_USED','TREAT_PROPPANT']
    keys = list(set(keys).intersection(OUT.keys().to_list()))
    mask = OUT.index
    OUT['STIM_PROPPANT']=None
    for k in keys:
        OUT.loc[mask,'STIM_PROPPANT'] = OUT.loc[mask,k]
        mask = OUT['STIM_PROPPANT'].isna()

    # PROPPANT_INTENSITY
    mask = OUT['PerfInterval'].isna()
    OUT.loc[~mask,'STIM_PROPPANT_INTENSITY'] = OUT.loc[~mask,'STIM_PROPPANT']/OUT.loc[~mask,'PerfInterval']
    OUT.loc[mask,'STIM_PROPPANT_INTENSITY'] = OUT.loc[mask,'STIM_PROPPANT'] / abs(OUT.loc[mask,'LatLen'])

    # STIM_PROPPANT_LOADING
    OUT['STIM_PROPPANT_LOADING'] = OUT['STIM_PROPPANT']/OUT['STIM_FLUID']

    # MAX PRESSURE if True:
    keys = ['MAX_PRESSURE','TREAT_PRESSURE']
    keys = list(set(keys) & set(OUT.keys().to_list()))
    mask = OUT.index
    OUT['STIM_MAX_PRESSURE']=None
    for k in keys:
        OUT.loc[mask,'STIM_MAX_PRESSURE'] = OUT.loc[mask,k]
        mask = OUT['STIM_MAX_PRESSURE'].isna()
        
    OUT = OUT.dropna(how='all')
    OUT = OUT.drop_duplicates()

    # manage duplicate wells
    PROBLEM_KEYS = []
    if 1==1:
        cts = OUT.API14.value_counts()
        dbl = cts.loc[cts>1].index.values
        OUT.loc[OUT.API14.isin(dbl)]

    #OUT= DF_UNSTRING(OUT)
    OUT.WELL_NUMBER.fillna('',inplace=True)
    OUT.WELL_NUMBER = OUT.WELL_NUMBER.astype(str)
    
    outfile = 'Compiled_Well_Data_XYZ_'+datetime.datetime.now().strftime("%d%m%Y")
    OUT.to_parquet(outfile+'.PARQUET')

######################################################################################################
    
    quit()

######################################################################################################

#TC_OUT
if 1==1:
    TC_OUT['Rel_Nearest'] = (abs((np.array(TC_OUT[dXY])/450)**2 + (np.array(TC_OUT[dZ])/300)**2).min(axis=1))**(1/2)

    TC_OUT['dXY1_Left'] = abs(TC_OUT[dXY].clip(upper=0)).replace(0,np.nan).min(axis=1)
    LEFT_MIN = abs(TC_OUT[dXY].clip(upper=0)).replace(0,np.nan).idxmin(axis=1)

    TC_OUT['dXY1_Right'] = abs(TC_OUT[dXY].clip(lower=0)).replace(0,np.nan).min(axis=1)
    RIGHT_MIN = abs(TC_OUT[dXY].clip(lower=0)).replace(0,np.nan).idxmin(axis=1)

    for i in TC_OUT.index:
        try:
            COL = re.sub('xy','z',LEFT_MIN[i])
            TC_OUT.loc[i,'dZ1_Left'] = TC_OUT.loc[i,COL]
        except: pass
        try:
            COL = re.sub('xy','z',RIGHT_MIN[i])
            TC_OUT.loc[i,'dZ1_RIGHT'] = TC_OUT.loc[i,COL]
        except: pass

    KeyData_Dict = {
        'Completion_Date':r'(?:Stim|Treat|Comp|Job).*Date',
        'FirstProduction_Date':r'First.*Prod|1st.*prod|month1',
        'Lateral_Length':r'perf.*int|lat.*len',
        'STIM_FLUID':r'STIM.*FLUID|TOTAL.*FLUID',
        'STIM_PROPPANT':r'TOTAL.*PROPPANT',
        'PeakOil_CumOil':r'peak.*oil.*cum.*oil',
        'PeakOil_CumGas':r'peak.*oil.*cum.*gas'
        }

    keys = TC_OUT.keys()


    for k in KeyData_Dict:
        kkey = KeyData_Dict[k]
        pattern = re.compile(kkey,re.I)
        k_sub = list(filter(pattern.match, keys))
        print(k_sub)
        TC_OUT[k] = TC_OUT[k_sub].max(axis=1)


#####################
#####################
    #if 1==1:
    # PERF INTERVAL
    TC_OUT['PerfInterval'] = None
    mask = (abs(TC_OUT.INTERVAL_TOP - TC_OUT.INTERVAL_BOTTOM) > 0)
    TC_OUT.loc[mask,'PerfInterval'] = abs(TC_OUT.loc[mask,'INTERVAL_TOP'] - TC_OUT.loc[mask,'INTERVAL_BOTTOM'])

    # LATERAL_LENGTH -> 'LatLen

    # API10
    TC_OUT['API10'] = TC_OUT['UWI10'].apply(UWI10)

    # API12
    keys = ['API', 'APINumber','UWI/API','UWI/API_PETRAV','API14x', 'UWI10']
    mask = TC_OUT.index
    for k in keys:
        if (pd.to_numeric(TC_OUT.loc[mask,k], errors = 'coerce').dropna().min()>1e10) & (pd.to_numeric(TC_OUT.loc[mask,k], errors = 'coerce').dropna().min()<(1e13-1)):
            TC_OUT.loc[mask,'API12'] = TC_OUT.loc[mask,k].apply(UWI12)
            mask = TC_OUT['API12'].isna()

    # API14
    keys = ['API', 'APINumber','UWI/API','UWI/API_PETRAV','API14x', 'UWI10']
    mask = TC_OUT.index
    for k in keys:
        if (pd.to_numeric(TC_OUT.loc[mask,k], errors = 'coerce').dropna().min()>1e12) & (pd.to_numeric(TC_OUT.loc[mask,k], errors = 'coerce').dropna().min()<(1e14-1)):
            TC_OUT.loc[mask,'API14'] = TC_OUT.loc[mask,k].apply(UWI14)
            mask = TC_OUT['API14'].isna()

    # WELL NAME
    keys = ['WellName','WELLNAME','WELL_NAME/NO', 'WELLNAME_PETRAV']
    mask = TC_OUT.index
    TC_OUT['WELL_NAME']=None
    for k in keys:
        TC_OUT.loc[mask,'WELL_NAME'] = TC_OUT.loc[mask,k]
        mask = TC_OUT['WELL_NAME'].isna()

    # WELL NUMBER if 1==1:
    keys = ['WellNumber', 'WELLNO', 'WellNumber']
    mask = TC_OUT.index
    TC_OUT['WELL_NUMBER']=None
    for k in keys:
        TC_OUT.loc[mask,'WELL_NUMBER'] = TC_OUT.loc[mask,k]
        mask = TC_OUT['WELL_NUMBER'].isna()

    # OPERATOR if 1==1:
    keys = ['OperatorName','OPERATOR','Operator']
    mask = TC_OUT.index
    TC_OUT['OPERATOR_']=None
    for k in keys:
        TC_OUT.loc[mask,'OPERATOR_'] = TC_OUT.loc[mask,k]
        mask = TC_OUT['OPERATOR_'].isna()

    # STATUS ?

    # STIM_FLUID
    TC_OUT['STIM_FLUID']=TC_OUT['TotalBaseWaterVolume']/42
    mask = (TC_OUT['STIM_FLUID']>1000)==False
    TC_OUT.loc[mask,'STIM_FLUID']=TC_OUT.loc[mask,'TotalFluid']
    mask = (TC_OUT['STIM_FLUID']>1000)==False
    TC_OUT.loc[mask,'STIM_FLUID']=TC_OUT.loc[mask,'TOTAL_FLUID_USED']
    #mask = (TC_OUT['STIM_FLUID']>1000)==False
    #TC_OUT.loc[mask,'STIM_FLUID']=TC_OUT.loc[mask,'TREAT_FLUID']

    # FLUID_INTENSITY & FLUID_INTENSITY_TYPE
    mask = TC_OUT['PerfInterval'].isna()
    TC_OUT.loc[~mask,'STIM_FLUID_INTENSITY'] = TC_OUT.loc[~mask,'STIM_FLUID']/TC_OUT.loc[~mask,'PerfInterval']
    TC_OUT.loc[mask,'STIM_FLUID_INTENSITY'] = TC_OUT.loc[mask,'STIM_FLUID'] / abs(TC_OUT.loc[mask,'LatLen'])

    # STIM_PROPPANT
    keys = ['TotalAmount_PROPPANT','TOTAL_PROPPANT_USED','TREAT_PROPPANT']
    keys = list(set(keys).intersection(TC_OUT.keys().to_list()))
    mask = TC_OUT.index
    TC_OUT['STIM_PROPPANT']=None
    for k in keys:
        TC_OUT.loc[mask,'STIM_PROPPANT'] = TC_OUT.loc[mask,k]
        mask = TC_OUT['STIM_PROPPANT'].isna()

    # PROPPANT_INTENSITY
    mask = TC_OUT['PerfInterval'].isna()
    TC_OUT.loc[~mask,'STIM_PROPPANT_INTENSITY'] = TC_OUT.loc[~mask,'STIM_PROPPANT']/TC_OUT.loc[~mask,'PerfInterval']
    TC_OUT.loc[mask,'STIM_PROPPANT_INTENSITY'] = TC_OUT.loc[mask,'STIM_PROPPANT'] / abs(TC_OUT.loc[mask,'LatLen'])

    # STIM_PROPPANT_LOADING
    TC_OUT['STIM_PROPPANT_LOADING'] = TC_OUT['STIM_PROPPANT']/TC_OUT['STIM_FLUID']

    # MAX PRESSURE
    keys = ['MAX_PRESSURE','TREAT_PRESSURE']
    keys = list(set(keys).intersection(TC_OUT.keys().to_list()))
    mask = TC_OUT.index
    TC_OUT['STIM_MAX_PRESSURE']=None
    for k in keys:
        TC_OUT.loc[mask,'STIM_MAX_PRESSURE'] = TC_OUT.loc[mask,k]
        mask = TC_OUT['STIM_MAX_PRESSURE'].isna()


    keys = [
        'API10',
        'API12',
        'API14',
        'WELL_NAME',
        'WELL_NUMBER',
        'OPERATOR_',
        'PerfInterval',
        'LatLen',
        'STIM_FLUID',
        'STIM_FLUID_INTENSITY',
        'STIM_PROPPANT',
        'STIM_PROPPANT_INTENSITY',
        'STIM_PROPPANT_LOADING',
        'STIM_MAX_PRESSURE',
        'API_MEAN',
        'BTU_MEAN',
        'GOR_PrePeakOil',
        'WOC_PostPeakGas',
        'Gel Mass',
        'Friction Reducer Mass',
        'Diverter Mass',
        'Rel_Nearest',
        'dXY1_Left',
        'dXY1_Right',
        'dZ1_Left',
        'dZ1_RIGHT',
        'Completion_Date',
        'FirstProduction_Date',
        'NUMBER_OF_STAGED_INTERVALS',
        'TUBING_SIZE',
        'TUBING_SETTING_DEPTH',
        'NUMBER_OF_HOLES'
        ]

    TC_OUT = TC_OUT[keys].dropna(how='all')
    TC_OUT = TC_OUT.drop_duplicates()

    # manage duplicate wells
    PROBLEM_KEYS = []
    if 1==1:
        cts = TC_OUT.API14.value_counts()
        dbl = cts.loc[cts>1].index.values
        TC_OUT.loc[TC_OUT.API14.isin(dbl)]
        
    quit()

    outfile = 'Compiled_Well_Data_XYZ_'+datetime.datetime.now().strftime("%d%m%Y")
    TCoutfile = 'TC_Compiled_Well_Data_XYZ_'+datetime.datetime.now().strftime("%d%m%Y")

    # Show Duplicates
    if 1==1:
        cts = TC_OUTR.FACILITYID.value_counts()
        dbl = cts.loc[cts>1].index.values
        TC_OUT.loc[TC_OUT.FACILITYID.isin(dbl)]

    #TC_OUT.to_csv(TCoutfile+'.csv',index=False)
    #OUT.to_csv(outfile+'.csv',index=False)
    TC_OUT.to_json(TCoutfile+'.JSON')
    OUT.to_json(outfile+'.JSON')
