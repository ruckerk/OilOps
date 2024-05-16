from ._FUNCS_ import *
from .DATA import *
from .SURVEYS import *
from .MAP import convert_XY
from ._MAPFUNCS_ import *
from .MAP import *

__all__ = ['CONSTRUCT_DB',
          'UPDATE_SURVEYS',
          'UPDATE_PROD']

def ONELINE(DB_NAME = 'FIELD_DATA.db'):
    WELLLINE_LOC = read_shapefile(shp.Reader('Directional_Lines.shp'))
    WELLLINE_LOC['UWI10'] = WELLLINE_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))
    WELLLINE_LOC = WELLLINE_LOC.loc[~(WELLLINE_LOC['UWI10'] == 500000000)]
    WELLLINE_LOC['X'] = WELLLINE_LOC.coords.apply(lambda x:x[0][0])
    WELLLINE_LOC['Y'] = WELLLINE_LOC.coords.apply(lambda x:x[0][1])
    WELLLINE_LOC['XBHL'] = WELLLINE_LOC.coords.apply(lambda x:x[-1][0])
    WELLLINE_LOC['YBHL'] = WELLLINE_LOC.coords.apply(lambda x:x[-1][1])
    #UWIlist = WELLLINE_LOC.loc[~(WELLLINE_LOC['UWI10'].isin(SCOUT_UWI)), 'UWI10']
    UWIlist = WELLLINE_LOC['UWI10'].unique().tolist()
          
    CONN = sqlite3.connect(DB_NAME)
    PROD = pd.read_sql('SELECT * FROM PRODDATA', CONN, chunksize = 100000)

    p_old = pd.DataFrame()
    OUTPUT = pd.DataFrame()
    UWILIST = []
          
    for p in PROD:
        p['UWI10'] = p['UWI'].apply(lambda x: WELLAPI(x).API2INT(10))        
        ULIST = list(set(p['UWI10']).intersection(set(WELLLINE_LOC['UWI10'])))
        p = p.loc[p.UWI10.isin(ULIST)]
        
        p = DF_UNSTRING(p)
        if not p_old.empty:
            UWILIST = list(set(p_old.UWI10).intersection(set(p.UWI10)))
            m = p_old.index[p_old.UWI10.isin(UWILIST)]
        else:
            m = p_old.index
        p_use = pd.concat([p_old.loc[m],p],axis = 0, join = 'outer')
        
        mm = p_use[['Oil_Produced','Gas_Produced','Water_Volume']].astype(float, errors = 'ignore').replace(0.0,np.nan).dropna(how='all',axis = 0).index
        p_use = p_use.loc[mm,:].copy()
        p_use.shape
        if len(mm)>20:
            p_use['NORM_OIL'] = p_use['Oil_Produced']/p_use.groupby(['UWI10'])['Oil_Produced'].cummax(skipna=True)
            p_use['NORM_GAS'] = p_use['Gas_Produced']/p_use.groupby(['UWI10'])['Gas_Produced'].cummax(skipna=True)
            p_use['NORM_WTR'] = p_use['Water_Volume']/p_use.groupby(['UWI10'])['Water_Volume'].cummax(skipna=True)
            p_use['CUM_OIL'] = p_use.groupby(['UWI10'])['Oil_Produced'].cumsum(skipna=True)
            p_use['CUM_GAS'] = p_use.groupby(['UWI10'])['Gas_Produced'].cumsum(skipna=True)       
            p_use['CUM_WTR'] = p_use.groupby(['UWI10'])['Water_Volume'].cumsum(skipna=True)
            p_use['CUM_GOR'] = p_use['CUM_GAS'] * 1000 / p_use['CUM_OIL']
            p_use['CUM_WOC'] = p_use['CUM_WTR'] * 1000 / (p_use['CUM_OIL'] + p_use['CUM_WTR'])
            p_use['TMB_OIL'] = p_use['CUM_OIL']/p_use['Oil_Produced']
            p_use['TMB_GAS'] = p_use['CUM_GAS']/p_use['Gas_Produced']
            p_use['TMB_WTR'] = p_use['CUM_WTR']/p_use['Water_Volume']
            p_use['WOC'] = p_use['Water_Volume']/(p_use['Oil_Produced']+p_use['Water_Volume'])
            p_use['PROD_DAYS'] = p_use.groupby(['UWI10'])['Days_Produced'].cumsum(skipna=True)
            p_use = DF_UNSTRING(p_use)
            
            mm_o95 = p_use.index[p_use['NORM_OIL']<0.95]
            mm_g95 = p_use.index[p_use['NORM_GAS']<0.95]
            mm_w95 = p_use.index[p_use['NORM_WTR']<0.95]
            mm_otmb200 = p_use.index[p_use['TMB_OIL']<200]
            mm_gtmb200 = p_use.index[p_use['TMB_GAS']<200]
            mm_wtmb200 = p_use.index[p_use['TMB_WTR']<200]
            mm_o2080 = p_use.index[(p_use['NORM_OIL']<=0.80)*(p_use['NORM_OIL']>=0.20)]
 
            PAIRS = [('TMB_OIL','NORM_OIL', True, False, mm_o95.intersection(mm_otmb200), stretch_exponential, 'TMB_NORM_OIL'),
                     ('TMB_GAS','NORM_GAS', True, False, mm_g95.intersection(mm_gtmb200), stretch_exponential, 'TMB_NORM_GAS'),
                     ('TMB_WTR','NORM_WTR', True, False, mm_w95.intersection(mm_otmb200), stretch_exponential, 'TMB_NORM_WTR'),
                     ('TMB_GAS','CUM_GOR', True, False, mm_g95, sigmoid, 'TMBGAS_GOR'),
                     ('TMB_WTR','OWC', True, False, mm_w95, sigmoid, 'TMBWTR_OWC'),
                     ('TMB_OIL', 'NORM_OIL', True, True, p_use.index, linear,'TMB_NORM_SLOPE2080'),
                     ('TMB_OIL', 'PROD_DAYS', True, True, p_use.index, linear, 'TMBOIL_DAYS'),
                    ]
            OUTPUT = OUTPUT.loc[~OUTPUT.index.isin(UWILIST)]
            MODELS = pd.DataFrame()

            # add recompletion detection by TMBOIL vs DAYS
            for (Xkey, Ykey, logx_bool, logy_bool, mm, func, NAME) in PAIRS:
                try:
                    MODEL = p_use.loc[mm,['UWI10',Xkey,Ykey]].dropna(how='any',axis=0).groupby(['UWI10']).apply(lambda x: curve_fitter(x[Xkey],x[Ykey], funct = func, split = None, plot = False, logx = logx_bool, logy = logy_bool))
                    MODEL = pd.DataFrame(MODEL.tolist())
                    MODEL['FUNCTION'] =  func.__name__ 
                    #NAME = '_'.join([Xkey,Ykey])
                    MODELS[NAME] = MODEL.apply(list,axis =1)
                except:
                    pass
            OUTPUT = pd.concat([OUTPUT,MODELS], axis = 0, join = 'outer') # left_index = True, right_index = True, how= 'outer')
            print(f'OUTPUT SHAPE: {OUTPUT.shape}')
        p_old = p

    # split OUTPUT params

    return OUTPUT

def CONSTRUCT_DB(DB_NAME = 'FIELD_DATA.db', SURVEYFOLDER = 'SURVEYFOLDER'):
    #pathname = path.dirname(argv[0])
    #adir = path.abspath(pathname)
    adir = getcwd()
          
    connection_obj = sqlite3.connect(DB_NAME)
    c = connection_obj.cursor()
    SURVEY_FILE_FIELDS = {'FILENAME':['CHAR'],'FILE':['BLOB']}
    INIT_SQL_TABLE(connection_obj, 'SURVEYFILES', SURVEY_FILE_FIELDS)
    
    DATA_COLS = {'MD': 'REAL',
         'INC': 'REAL', 
         'AZI':'REAL',
         'TVD':'REAL',
         'NORTH_dY':'REAL',
         'EAST_dX':'REAL',
         'UWI':'INTEGER',
         'FILE': 'TEXT'
         }  
    INIT_SQL_TABLE(connection_obj,'SURVEYDATA', DATA_COLS)
    
    #c.execute(''' SELECT DISTINCT FILENAME FROM SURVEYFILES  ''')
    #LOADED_FILES = c.fetchall()
    #LOADED_FILES = list(set(itertools.chain(*LOADED_FILES)))
    LOADED_FILES = pd.read_sql('SELECT DISTINCT FILENAME FROM SURVEYFILES', connection_obj).iloc[:,0].tolist()
    SCANNEDFILES = pd.read_sql('SELECT DISTINCT FILE FROM SURVEYDATA', connection_obj).iloc[:,0].tolist()
    
    SURVEYFOLDER = path.join(adir,SURVEYFOLDER)
    XLSLIST = filelist(SURVEYFOLDER,None,None,'.xls')
    USELIST = list(set(XLSLIST).difference(set(LOADED_FILES)))

    print('{} FILES TO LOAD'.format(len(USELIST)))

    for F in USELIST:
        if '~' in F:
            continue
        B = convertToBinaryData(path.join(SURVEYFOLDER,F))
        data_tuple = (F,B)
        load_surveyfile(connection_obj,data_tuple, table  = 'SURVEYFILES')
    
    SQL_UNDUPLICATE(connection_obj, 'SURVEYFILES', ['FILENAME'])

    #DATA = READ_SQL_TABLE(connection_obj,'SURVEYFILES')
    #DATA_df = pd.DataFrame(DATA, columns = ['FILE','BLOB'])
    DATA_df = pd.read_sql('SELECT * FROM SURVEYFILES', connection_obj)

    #QRY = 'SELECT DISTINCT UPPER(FILE) FROM SURVEYDATA'
    #c.execute(QRY)
    #SCANNEDFILES = c.fetchall()
    #SCANNEDFILES = list(itertools.chain(*SCANNEDFILES))
    m = DATA_df.loc[~DATA_df.FILENAME.isin(SCANNEDFILES)].index

    S_KEYS = pd.read_sql('SELECT * FROM SURVEYDATA LIMIT 1', connection_obj).keys()
    OUT = pd.DataFrame(columns = S_KEYS)

    warnings.filterwarnings('ignore')

    batch = min(5000,len(m))
    chunksize = max(int(len(m)/batch),1)
    mm=np.array_split(m,chunksize)
    print(f'{len(m)} new survey files')
    for m1 in mm:
        OUT = pd.DataFrame(columns = S_KEYS)
        for D in DATA_df.loc[m1].values:
            if 'LOCK.' in D[0].upper():
                continue

            FILE = D[0]
    
            if D[0].upper() in SCANNEDFILES:
                continue
            dd = list()
            try:
                dd = survey_from_excel(tuple(D),ERRORS = False)
            except Exception as e:
                print(f'EXCEPTION:{FILE} :: {e}')
            if isinstance(dd,pd.DataFrame):
                dd['FILE'] = FILE
                OUT = pd.concat([OUT,dd])
                
        if not OUT.empty:
            #OUT.rename(columns = {'UWI10':'UWI'}, inplace =True)
            #OUT['UWI10'] = OUT.UWI.apply(lambda x: WELLAPI(x).API2INT(10))
            OUT.to_sql('SURVEYDATA',index = False, con = connection_obj, if_exists = 'append', chunksize = 5000)
    
    print('New survey processing complete')
    SQL_UNDUPLICATE(connection_obj,'SURVEYDATA')
    
    print('SQL survey duplicates removed')
    warnings.filterwarnings('default')

    #ALL_SURVEYS = pd.read_sql_query('SELECT * FROM SURVEYDATA',connection_obj)
    ALL_SURVEYS= pd.DataFrame()
    READ = pd.read_sql_query('SELECT * FROM SURVEYDATA',connection_obj, chunksize = 50000)
    for i in READ:
        i2 = i.dropna(axis=0,how='all').dropna(axis=1,how='all')
        ALL_SURVEYS = pd.concat([ALL_SURVEYS,i2],axis = 0, join = 'outer', ignore_index = True)
          
    ALL_SURVEYS['UWI10'] = ALL_SURVEYS.UWI.apply(lambda x:WELLAPI(x).API2INT(10))
           
    # OLD PREFERRED SURVEYS
    if 'FAVORED_SURVEYS' in LIST_SQL_TABLES(connection_obj):
        QRY = 'SELECT FILE, UWI10, FAVORED_SURVEY FROM FAVORED_SURVEYS' 
        OLD_PREF = pd.read_sql(QRY, connection_obj)   
    else:
        OLD_PREF = pd.DataFrame(columns=['UWI10','FILE','FAVORED_SURVEY'])

    # ALL UWI/SURVEY PAIRS NOT ALREADY CONSIDERED
    m_new = ALL_SURVEYS[['UWI10','FILE']].merge(OLD_PREF[['UWI10','FILE']].drop_duplicates(),indicator = True, how='left').loc[lambda x : x['_merge']!='both'].index
    m_old = ALL_SURVEYS[['UWI10','FILE']].merge(OLD_PREF[['UWI10','FILE']].drop_duplicates(),indicator = True, how='left').loc[lambda x : x['_merge']=='both'].index
         
     # SET FAVORED SURVEY to 1/0 binary including old assignments   
    ALL_SURVEYS['FAVORED_SURVEY'] = -1    

    if len(m_old)>0:
        ALL_SURVEYS.loc[m_old,'FAVORED_SURVEY'] = ALL_SURVEYS.loc[m_old,['UWI10','FILE']].merge(OLD_PREF,on=['UWI10','FILE'], how = 'left')['FAVORED_SURVEY']

    ALL_SURVEYS.loc[ALL_SURVEYS['FILE']==ALL_SURVEYS['FAVORED_SURVEY'],'FAVORED_SURVEY'] = 1
    #ALL_SURVEYS.loc[~ALL_SURVEYS['FAVORED_SURVEY'].isin([-1,0]),'FAVORED_SURVEY'] = 0

    ALL_SURVEYS.FAVORED_SURVEY = ALL_SURVEYS.FAVORED_SURVEY.astype(int)

    print('Favored survey assignment complete')

    #UWIs with new file or none assigned
    NEW_UWI = ALL_SURVEYS.loc[m_new,'UWI10'].unique().tolist()
    NEW_UWI.extend(ALL_SURVEYS.loc[(ALL_SURVEYS.FAVORED_SURVEY == -1),'UWI10'].unique().tolist())
    m = ALL_SURVEYS.index[ALL_SURVEYS.UWI10.isin(NEW_UWI)]
    
    if len(m)>0:
        CONDENSE_DICT = Condense_Surveys(ALL_SURVEYS.loc[m,['UWI10','FILE','MD', 'INC', 'AZI', 'TVD','NORTH_dY', 'EAST_dX']])
        ALL_SURVEYS.loc[m,'FAVORED_SURVEY'] = ALL_SURVEYS.loc[m,'UWI10'].apply(lambda x:CONDENSE_DICT[x])
        m1 = (ALL_SURVEYS['FAVORED_SURVEY']==ALL_SURVEYS['FILE']) + (ALL_SURVEYS['FAVORED_SURVEY'] == 1) 
        ALL_SURVEYS.loc[m,'FAVORED_SURVEY'] = 0
        ALL_SURVEYS.loc[m1,'FAVORED_SURVEY'] = 1
            
    #m = FAVORED_SURVEY.apply(lambda x:CONDENSE_DICT[x.UWI10] == x.FILE, axis=1)

    # CREATE ABSOLUTE LOCATION TABLE if True:
    WELL_LOC = read_shapefile(shp.Reader('Wells.shp'))
    
    WELL_LOC['UWI10'] = WELL_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))
    WELL_LOC = WELL_LOC.loc[~(WELL_LOC['UWI10'] == 500000000)]
    WELL_LOC['X'] = WELL_LOC.coords.apply(lambda x:x[0][0])
    WELL_LOC['Y'] = WELL_LOC.coords.apply(lambda x:x[0][1])
    WELL_LOC['XBHL'] = WELL_LOC.coords.apply(lambda x:x[-1][0])
    WELL_LOC['YBHL'] = WELL_LOC.coords.apply(lambda x:x[-1][1])
    WELL_LOC[['XFEET','YFEET']] = pd.DataFrame(convert_XY(WELL_LOC.X,WELL_LOC.Y,26913,2231)).T.values 
          
    WELLPLAN_LOC = read_shapefile(shp.Reader('Directional_Lines_Pending.shp'))
    WELLPLAN_LOC['UWI10'] = WELLPLAN_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))
    WELLPLAN_LOC = WELLPLAN_LOC.loc[~(WELLPLAN_LOC['UWI10'] == 500000000)]
    WELLPLAN_LOC['X'] = WELLPLAN_LOC.coords.apply(lambda x:x[0][0])
    WELLPLAN_LOC['Y'] = WELLPLAN_LOC.coords.apply(lambda x:x[0][1])
    WELLPLAN_LOC['XBHL'] = WELLPLAN_LOC.coords.apply(lambda x:x[-1][0])
    WELLPLAN_LOC['YBHL'] = WELLPLAN_LOC.coords.apply(lambda x:x[-1][1])

    WELLLINE_LOC = read_shapefile(shp.Reader('Directional_Lines.shp'))
    WELLLINE_LOC['UWI10'] = WELLLINE_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))
    WELLLINE_LOC = WELLLINE_LOC.loc[~(WELLLINE_LOC['UWI10'] == 500000000)]
    WELLLINE_LOC['X'] = WELLLINE_LOC.coords.apply(lambda x:x[0][0])
    WELLLINE_LOC['Y'] = WELLLINE_LOC.coords.apply(lambda x:x[0][1])
    WELLLINE_LOC['XBHL'] = WELLLINE_LOC.coords.apply(lambda x:x[-1][0])
    WELLLINE_LOC['YBHL'] = WELLLINE_LOC.coords.apply(lambda x:x[-1][1])
    
    LOC_COLS = ['UWI10','X','Y','XBHL','YBHL']
    LOC_DF = WELLLINE_LOC[LOC_COLS].drop_duplicates()
    m = WELLPLAN_LOC.index[~(WELLPLAN_LOC.UWI10.isin(LOC_DF.UWI10))]
    LOC_DF = pd.concat([LOC_DF,WELLPLAN_LOC.loc[m,LOC_COLS].drop_duplicates()])
    m = WELL_LOC.index[~(WELL_LOC.UWI10.isin(LOC_DF.UWI10))]
    LOC_DF = pd.concat([LOC_DF,WELL_LOC.loc[m,LOC_COLS].drop_duplicates()])
    LOC_DF.UWI10.shape[0]-len(LOC_DF.UWI10.unique())
    
    LOC_DF[['XFEET','YFEET']] = pd.DataFrame(convert_XY(LOC_DF.X,LOC_DF.Y,26913,2231)).T.values
    LOC_DF[['XBHLFEET','YBHLFEET']] = pd.DataFrame(convert_XY(LOC_DF.XBHL,LOC_DF.YBHL,26913,2231)).T.values
 
    LOC_DF['DELTA'] = ((LOC_DF['YBHLFEET'] - LOC_DF['YFEET'])**2 +  (LOC_DF['XBHLFEET'] - LOC_DF['XFEET'])**2)**0.5
    m = LOC_DF['DELTA']>2000
    VS_UWIS = LOC_DF.loc[m,'UWI10'].unique()

    LOC_COLS = {'UWI10': 'INTEGER',
                'X': 'REAL',
                'Y': 'REAL',
                'XFEET':'REAL',
                'YFEET':'REAL'}
    
    INIT_SQL_TABLE(connection_obj, 'SHL', LOC_COLS)
    LOC_DF[['UWI10','X','Y','XFEET','YFEET']].to_sql(name = 'SHL', con = connection_obj, if_exists='replace', index = False, dtype = LOC_COLS)
    connection_obj.commit()
    
    m = (ALL_SURVEYS.FAVORED_SURVEY == 1)
    
    #FLAG PREFERRED SURVEYS if True:
    ALL_SURVEYS = ALL_SURVEYS.merge(LOC_DF[['UWI10','XFEET','YFEET']],how = 'left', on = 'UWI10')
    ALL_SURVEYS['NORTH'] = ALL_SURVEYS[['NORTH_dY','YFEET']].sum(axis=1)
    ALL_SURVEYS['EAST'] = ALL_SURVEYS[['EAST_dX','XFEET']].sum(axis=1)
    ALL_SURVEYS.rename({'YFEET':'SHL_Y_FEET','XFEET':'SHL_X_FEET'}, axis = 1, inplace = True)

    #ALL_SURVEYS['FAVORED_SURVEY'] = ALL_SURVEYS.apply(lambda x: CONDENSE_DICT[x.UWI10], axis = 1).str.upper() == ALL_SURVEYS.FILE.str.upper()

    SCHEMA = {'UWI10': 'INTEGER', 'FILE':'TEXT', 'FAVORED_SURVEYS':'INTEGER'}
    INIT_SQL_TABLE(connection_obj,'FAVORED_SURVEYS', SCHEMA)
    
    ALL_SURVEYS.loc[:,['UWI10','FILE','FAVORED_SURVEY']].drop_duplicates().to_sql('FAVORED_SURVEYS',
                                                   connection_obj,
                                                   schema = SCHEMA,
                                                   index = False,
                                                   if_exists = 'replace')
    
    connection_obj.commit()

    # XYZ SPACING CALC
    # PARENT CUM AT TIME OF CHILD FRAC
    # PAD ASSIGNMENTS FROM SPACING GROUPS (SAME SHL)
    # UNIT ASSIGNMENTS: NEAREST IN DATE DIFF RANGE AND EITHER LATERAL IS 90% OVERLAPPING OTHER

    #QRY = 'SELECT CAST(max(S.UWI10, P.UWI10) as INT) AS UWI10, S.FIRST_PRODUCTION_DATE, S.JOB_DATE, S.JOB_END_DATE, P.FIRST_PRODUCTION FROM SCOUTDATA AS S LEFT JOIN PRODUCTION_SUMMARY AS P ON S.UWI10 = P.UWI10'
    QRY = 'SELECT MAX(S.UWI10, P.UWI10) AS UWI10, P.MONTH1, S.JOB_DATE, S.JOB_END_DATE, S.FIRST_PRODUCTION_DATE FROM SCOUTDATA AS S LEFT JOIN PRODUCTION_SUMMARY AS P ON S.UWI10 = P.UWI10'
          
    WELL_DF = pd.read_sql(QRY,connection_obj)
    WELL_DF=WELL_DF.dropna(how='all',axis = 0)
    WELL_DF = WELL_DF.loc[~WELL_DF.UWI10.isna()]
    WELL_DF = DF_UNSTRING(WELL_DF)
    WELL_DF.sort_values(by = 'FIRST_PRODUCTION_DATE',ascending = False, inplace = True)

    UWIlist = WELL_DF.sort_values(by = 'UWI10', ascending = False).UWI10.tolist()
          
    # TEST FOR LATERAL LENGTH      
    m = ALL_SURVEYS.INC>88
    LL_TEST = ALL_SURVEYS.loc[(ALL_SURVEYS.INC>=88) * (ALL_SURVEYS.FAVORED_SURVEY==1)].copy()
    LL_TEST['XY_DELTA'] = (LL_TEST['NORTH_dY']**2+LL_TEST['EAST_dX']**2).apply(sqrt)
    LL_TEST = LL_TEST.groupby(by='UWI10')['XY_DELTA'].agg(['min','max'])
    LL_TEST['LATLEN'] = LL_TEST.iloc[:,1]-LL_TEST.iloc[:,0]
    UWIlist_LAT = LL_TEST.index[LL_TEST['LATLEN'] > 3000].tolist()

    UWIlist = list(set(UWIlist).union(set(UWIlist_LAT)))

    # MAJOR UPGRADE FOR SPEED: XYZ ONLY FOR NEW SURVEYS, AND WELLS NEAR NEW SURVEYS
    # FOR WELL IN NEW_SURVEYS: ALL_SURVEYS[[NORTH,EAST]] - [[NORTH,EAST]] <= 10000
    # UWILIST = AGGREGATED RESULT
          
    # FIND ALL UNCHANGED UWI-FILE PAIRS & USE THE OTHERS
    #m = pd.merge(ALL_SURVEYS[['UWI10','FILE']], OLD_PREF[['UWI10','FILE']], on=['UWI10','FILE'], how='left', indicator='TEST').TEST!='both'
    XYZ_OLD = pd.DataFrame()
    if 'SPACING' in LIST_SQL_TABLES(connection_obj):
        #ALL_SURVEYS = pd.read_sql('SELECT s.*, f.UWI10,f.FILE, f.FAVORED_SURVEY FROM SURVEYDATA s JOIN FAVORED_SURVEYS f ON s.UWI=f.UWI10 AND s.FILE=f.FILE WHERE f.FAVORED_SURVEY=1 ', connection_obj)                    
        XYZ_OLD = pd.read_sql('SELECT * FROM SPACING', con = connection_obj)
        if 'FILE' in XYZ_OLD.keys():
            KK = XYZ_OLD.keys().tolist()
            KK.remove('FILE')
            XYZ_OLD = XYZ_OLD[KK]
            del KK
        XYZ_OLD.rename(columns = {'XYZFILE':'FILE'}, inplace = True)
        all_df = pd.merge(ALL_SURVEYS[['UWI10','FILE','FAVORED_SURVEY']], 
                          XYZ_OLD[['UWI10','FILE']].dropna(),
                          how='left', 
                          on = ['UWI10','FILE'],
                          indicator='TEST')
        UWIlist = all_df.loc[(all_df.TEST!='both')*(all_df.FAVORED_SURVEY==1),'UWI10'].unique()     
     
        PTS0 = ALL_SURVEYS.loc[(ALL_SURVEYS.UWI10.isin(UWIlist)) * (ALL_SURVEYS.INC>88)*(ALL_SURVEYS.FAVORED_SURVEY==1),['UWI10','FILE','NORTH','EAST']].groupby(by=['UWI10','FILE'], axis = 0)[['EAST','NORTH']].agg(['first','mean','last'])
        #PTS0.reset_index(drop=False, inplace= True)
        PTS0['PTLIST'] = PTS0.apply(lambda x: [[x[0],x[1]], [x[2],x[3]], [x[4],x[5]]], axis = 1)              
        #PTS0 = list(itertools.chain(*PTS0))
              
        #REMOVE NAN
        #PTS0 = shapely.geometry.MultiPoint(PTS0)      
        PTS0['MULTIPOINT'] = PTS0['PTLIST'].apply(shapely.geometry.MultiPoint)
        PTS0['BUFFER'] = PTS0['MULTIPOINT'].apply(lambda x: x.buffer(5000))
        BUFFER_GROUP = shapely.geometry.MultiPolygon(PTS0['BUFFER'].tolist())
        #PTS0 = PTS0.buffer(10000)    # THIS CRASHES CPU ON MEMORY LIMIT

        # INTERSECT BUFFER
        #is it fast to create linestrings of each well and intersect?
        ALL_SURVEYS['XY']  = ALL_SURVEYS[['EAST','NORTH']].apply(list, axis = 1)  
        TEST = ALL_SURVEYS.loc[(ALL_SURVEYS.INC>88)*(ALL_SURVEYS.FAVORED_SURVEY==1),['UWI10','FILE','XY']].groupby(by=['UWI10','FILE'], axis = 0)['XY'].apply(list)
        TEST = TEST[TEST.apply(len)>1]
        TEST = TEST.apply(lambda x: shapely.geometry.LineString(x))
        
        s = shapely.strtree.STRtree(TEST.tolist())
        r = s.query(BUFFER_GROUP)  
        TEST = pd.DataFrame(TEST)
        TEST['INTERSECTS_BUFFER'] = TEST.XY.apply(lambda x: x in r)
        TEST.reset_index(drop=False,inplace=True)
          
        # UWIlist for wells intersecting buffer  
        UWIlist = TEST.loc[TEST['INTERSECTS_BUFFER'],'UWI10'].unique().tolist()
        UWIlist = [x for x in UWIlist if x>1e6]

        #PTS1.intersects(shapely.geometry.Point(PTS0[34]))

        UWI_MEANS = ALL_SURVEYS.loc[(ALL_SURVEYS.INC>88)*(ALL_SURVEYS.FAVORED_SURVEY==1),['UWI10','FILE','NORTH','EAST']].groupby(by=['UWI10','FILE'], axis = 0).mean().reset_index(drop=False)
        
        # USE A SPATIAL FUNCTION HERE
        # BUFFER AROUND CHANGING SURVEY PTS
        # FIND ALL UWIS TOUCHING BUFFER AREAS

        ALL_SURVEYS.loc[(ALL_SURVEYS.UWI10.isin(UWIlist)) * (ALL_SURVEYS.INC>88) * (ALL_SURVEYS.FAVORED_SURVEY==1),['UWI10','FILE','NORTH','EAST']].groupby(by=['UWI10','FILE'], axis = 0)[['EAST','NORTH']].first()
                                
    else:
        UWIlist = ALL_SURVEYS.loc[ALL_SURVEYS.FAVORED_SURVEY==1,'UWI10'].unique()
        UWIlist = list(UWIlist)

    m = (ALL_SURVEYS.FAVORED_SURVEY == 1)
    processors = max(1,floor(multiprocessing.cpu_count()/1))
          
    func = partial(XYZSpacing,
            xxdf= ALL_SURVEYS.loc[m,['UWI10','FILE','MD', 'INC', 'AZI', 'TVD','NORTH', 'EAST']],
            df_UWI = WELL_DF,
            DATELIMIT = 360,
            SAVE = False)
    
    XYZ = pd.DataFrame()
    if len(UWIlist) >2000:
        chunksize = int(len(UWIlist)/processors)
        chunksize = min(1000, chunksize)
        batches = int(len(UWIlist)/chunksize)
        data=np.array_split(UWIlist,batches)

        with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
            f = {executor.submit(func,a): a for a in data}
        
        for i in f.keys():
            XYZ = XYZ.append(i.result(), ignore_index = True)
    elif len(UWIlist)<=2000:
        XYZ = func(UWIlist)
    
    if ~XYZ.empty:
        XYZ = DF_UNSTRING(XYZ)    
        
        
        if not XYZ_OLD.empty:
            XYZ = pd.concat([XYZ, XYZ_OLD.loc[~XYZ_OLD.UWI10.isin(XYZ.UWI10)]], axis = 0, join = 'outer', ignore_index = True)
          
        #XYZ = XYZ.loc[~(XYZ.iloc[:,:-1].isna().any(axis=1))]

        #FIX THIS SO IT UPDATES PROPERLY WITHOUT DUPLICATED UWIS  
        XYZ_COLS = FRAME_TO_SQL_TYPES(XYZ)
        XYZ.to_sql(name = 'SPACING', con = connection_obj, if_exists='replace', index = False, dtype = XYZ_COLS)
        connection_obj.commit()
    
                    
    # DROP TABLES WITH NAMES THAT ARE ONLY NUMBERS (INTERMEDIATE TABLES USED EARLIER)
    for T in LIST_SQL_TABLES(connection_obj):
        if bool(re.match(r'^\d*$',T)):
            DROP_SQL_TABLE(connection_obj,T)
         
    
    ###################
    # PRODUCTION DATA #
    ###################
    
    ##############
    # SCOUT DATA #
    ##############
    if True:
        WELL_LOC.Spud_Date=pd.to_datetime(WELL_LOC.Spud_Date)
        WELL_LOC.Stat_Date=pd.to_datetime(WELL_LOC.Stat_Date)
        m_recent = ((datetime.datetime.now()-WELL_LOC['Stat_Date'])/pd.Timedelta(days=1))<=(365*2)
        m_spud = WELL_LOC.Spud_Date.dt.year>1000
        UWIlist = WELL_LOC.loc[m_recent + m_spud,'UWI10']      
        
        # if path.exists('SCOUTS'):
            # pfiles = listdir('SCOUTS')
            # pfiles = [f for f in pfiles if f.upper().endswith('PARQUET')]
            # SCOUT_DATA = pd.DataFrame()
            # for f in pfiles:
                # x1=pd.read_parquet(path.join('SCOUTS',f))
                # SCOUT_DATA = pd.concat([SCOUT_DATA,x1],ignore_index=True)
            # SCOUT_DATA.drop_duplicates(inplace=True)
            # SCOUT_DATA['UWI10'] = SCOUT_DATA.UWI.apply(lambda x:WELLAPI(x).API2INT(10))
            # DUPLICATED = SCOUT_DATA.loc[SCOUT_DATA['UWI10'].duplicated(),'UWI10'].to_list()
            # for u in DUPLICATED:
                # IDX = -1
                # SCOUT_SUB = SCOUT_DATA.loc[SCOUT_DATA.UWI10 == u].copy()
                # SCOUT_SUB = DF_UNSTRING(SCOUT_SUB)
                # IDX_ALT = SCOUT_SUB.sort_values(by='STATUS_DATE', ascending = False).index[0]
                # NA_LIMIT = SCOUT_SUB.isna().sum(axis=1).min()
                # m = SCOUT_SUB.isna().sum(axis=1) == NA_LIMIT
                # SCOUT_SUB =SCOUT_SUB.loc[m]
                # m = (SCOUT_SUB == SCOUT_SUB.iloc[0]).min(axis=0)
                # SCOUT_SUB = SCOUT_SUB.loc[:,~m]
                # m = SCOUT_SUB.isna().min(axis=0)
                # SCOUT_SUB = SCOUT_SUB.loc[:,~m]
                # m=SCOUT_SUB.isna().min(axis=0)
                # SCOUT_SUB = SCOUT_SUB.loc[:,~m]
                # SCOUT_SUB.drop_duplicates(inplace=True)
                # if 'TREATMENT_SUMMARY' in SCOUT_SUB.keys():
                    # m = SCOUT_SUB['TREATMENT_SUMMARY'].str.len() == SCOUT_SUB['TREATMENT_SUMMARY'].str.len().max
                    # SCOUT_SUB= SCOUT_SUB.loc[m]   
                # if (SCOUT_SUB.shape[0]==1) | (SCOUT_SUB.shape[1]==0):
                    # IDX = SCOUT_SUB.index[0]
                # n = SCOUT_SUB.astype(float, errors='ignore').dtypes != object
                # if n.any():
                    # m = (SCOUT_SUB.loc[:,n].astype(float).std()/SCOUT_SUB.loc[:,n].astype(float).mean()).abs() > 0.1
                    # if SCOUT_SUB.loc[:,m].shape[1]==0:
                        # IDX = SCOUT_SUB.index[0]     
                # if IDX == -1:
                    # IDX = IDX_ALT
                # SCOUT_SUB = SCOUT_DATA.loc[SCOUT_DATA.UWI10 == u].copy()
                # DROP_IDX = SCOUT_SUB.index[SCOUT_SUB.index != IDX]
                # SCOUT_DATA.drop(DROP_IDX, axis=0, inplace= True)
        
        SCOUTTABLENAME = 'SCOUTDATA'  
        SCOUT_DATA = pd.read_sql('SELECT DISTINCT UWI FROM {}'.format(SCOUTTABLENAME),connection_obj)     
        SCOUT_DATA['UWI10'] = SCOUT_DATA.UWI.apply(lambda x:WELLAPI(x).API2INT(10)) 
        #UWIlist = list(set(list(UWIlist))-set(SCOUT_DATA.UWI10.tolist()))
        
        #UWIlist = list()
        func = partial(Get_Scouts,
                db = DB_NAME,
                TABLE_NAME = SCOUTTABLENAME)  
        SCOUT_df = pd.DataFrame()
        if len(UWIlist) >2000:
            chunksize = int(len(UWIlist)/processors)
            chunksize = min(2000, chunksize)
            batches = int(len(UWIlist)/chunksize)
            data=np.array_split(UWIlist,batches)
    
            with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
                f = {executor.submit(func,a): a for a in data}
            
            for i in f.keys():
                SCOUT_df = pd.concat([SCOUT_df,i.result()],ignore_index = True)
                #SCOUT_df = SCOUT_df.append(i.result(), ignore_index = True)
        elif len(UWIlist)>0:
            SCOUT_df =  Get_Scouts(UWIlist,DB_NAME)

    ###################
    # FRAC FOCUS DATA #
    ###################
    # CREATE FRAC FOCUS TABLE
    FF = Merge_Frac_Focus(DIR = 'FRAC_FOCUS', SAVE = False)
    FF = DF_UNSTRING(FF)
    FF_COLS = FRAME_TO_SQL_TYPES(FF)
    FF.to_sql(name = 'FRAC_FOCUS', con = connection_obj, if_exists = 'replace', index = False, dtype = FF_COLS)
    connection_obj.commit()    
    ###########################
    # ADD PROD VOLUMES TO XYZ #
    ###########################

    ###############################
    # DEVELOPMENT UNIT ASSIGNMENT #
    ###############################
    # TABLE OF UNIT/WELL

    # SUMMARIZE COGCC SQL TABLES
    # add direct copy of key tables
    SUMMARIZE_COGCC_SQL()
 
def UPDATE_SCOUT(DB_NAME = 'FIELD_DATA.db', FULL_UPDATE = False, FOLDER = 'SCOUTS'):
    pathname = path.dirname(argv[0])
    #adir = path.abspath(pathname)
    adir = getcwd()
    udir = path.dirname(adir)

    CONN = sqlite3.connect(path.join(adir,DB_NAME))
    
    SCOUT_OLD = pd.read_sql('select * from SCOUTDATA',CONN)
    SCOUT_OLD['UWI10'] = SCOUT_OLD.UWI.apply(lambda x: WELLAPI(x).API2INT(10))
    SCOUT_UWI = list(SCOUT_OLD['UWI10'].unique())
    WELLLINE_LOC = read_shapefile(shp.Reader('Directional_Lines.shp'))
    WELLLINE_LOC['UWI10'] = WELLLINE_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))
    WELLLINE_LOC = WELLLINE_LOC.loc[~(WELLLINE_LOC['UWI10'] == 500000000)]
    WELLLINE_LOC['X'] = WELLLINE_LOC.coords.apply(lambda x:x[0][0])
    WELLLINE_LOC['Y'] = WELLLINE_LOC.coords.apply(lambda x:x[0][1])
    WELLLINE_LOC['XBHL'] = WELLLINE_LOC.coords.apply(lambda x:x[-1][0])
    WELLLINE_LOC['YBHL'] = WELLLINE_LOC.coords.apply(lambda x:x[-1][1])
    UWIlist = WELLLINE_LOC.loc[~(WELLLINE_LOC['UWI10'].isin(SCOUT_UWI)), 'UWI10']
    len(UWIlist)
    if len(UWIlist) >2000:
        func = partial(Get_Scouts,
            db = DB_NAME,
            TABLE_NAME = 'SCOUTDATA')
        processors = max(1,floor(multiprocessing.cpu_count()/1))
        chunksize = int(len(UWIlist)/processors)
        chunksize = min(2000, chunksize)
        batches = int(len(UWIlist)/chunksize)
        data=np.array_split(UWIlist,batches)
        SCOUT_df = pd.DataFrame()
        with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
            f = {executor.submit(func,a): a for a in data}
        for i in f.keys():
            SCOUT_df = SCOUT_df.append(i.result(), ignore_index = True)
    return None
    
def UPDATE_SPACINGS(DB = 'FIELD_DATA.db'):
    CONN = sqlite3.connect(DB)
    QRY = '''SELECT DISTINCT
        F.FAVORED_SURVEY,
        SUR.*,
        SCOUT.FIRST_PRODUCTION_DATE,
        SCOUT.JOB_DATE,
        SCOUT.SPUD_DATE,
        XYZ.MeanAZI180,
        XYZ.MAX_MD,
        XYZ.LatLen,
        SHL.XFEET AS SHL_X,
        SHL.YFEET AS SHL_Y,
        P.MONTH
        FROM FAVORED_SURVEYS F
        LEFT JOIN SURVEYDATA SUR ON SUR.UWI=F.UWI10 AND SUR.FILE = F.FILE
        LEFT JOIN SCOUTDATA SCOUT ON F.UWI10 = SCOUT.UWI10
        LEFT JOIN SPACING XYZ ON F.UWI10 = XYZ.UWI10
        LEFT JOIN SHL ON F.UWI10 = SHL.UWI10
        LEFT JOIN (SELECT UWI10, MIN(DATE(First_of_Month)) AS MONTH FROM PRODDATA GROUP BY UWI10) P ON F.UWI10 = P.UWI10
        WHERE F.FAVORED_SURVEY = 1 AND SUR.INC > 85  '''
    XYZ_DF = pd.read_sql(QRY,CONN)
    XYZ_DF = XYZ_DF.groupby(by=['UWI','MD'],as_index=False).first()
    XYZ_DF = DF_UNSTRING(XYZ_DF)
    XYZ_DF['DATE'] = XYZ_DF[GetKey(XYZ_DF,'DATE')+['MONTH']].min(axis=1,skipna=True)
    XYZ_DF['VINTAGE'] = XYZ_DF['DATE'].dt.year.fillna(datetime.datetime.now().year)
    XYZ_DF['YFEET']= XYZ_DF['NORTH_dY']+XYZ_DF['SHL_Y']
    XYZ_DF['XFEET']= XYZ_DF['EAST_dX']+XYZ_DF['SHL_X']
    XYZ_DF['SHAPE'] = XYZ_DF[['XFEET','YFEET']].apply(lambda x: tuple(x),axis=1)
    m = XYZ_DF['DATE'].isna()
    XYZ_DF = XYZ_DF.loc[~m,:]
    
    #SCOUT DATA
    QRY = 'SELECT MAX(S.UWI10, P.UWI10) AS UWI10, P.MONTH1, S.JOB_DATE, S.JOB_END_DATE, S.FIRST_PRODUCTION_DATE FROM SCOUTDATA AS S LEFT JOIN PRODUCTION_SUMMARY AS P ON S.UWI10 = P.UWI10'
          
    WELL_DF = pd.read_sql(QRY,CONN)
    WELL_DF=WELL_DF.dropna(how='all',axis = 0)
    WELL_DF = WELL_DF.loc[~WELL_DF.UWI10.isna()]
    WELL_DF = DF_UNSTRING(WELL_DF)
    WELL_DF.sort_values(by = 'FIRST_PRODUCTION_DATE',ascending = False, inplace = True)
        
        
          
def UPDATE_SURVEYS(DB = 'FIELD_DATA.db', FULL_UPDATE = False, FOLDER = 'SURVEYFOLDER'):
    ###############
    # GET SURVEYS #
    ############### #if True:
    # Initialize constants
    URL_BASE = 'https://ecmc.state.co.us/weblink/results.aspx?id=XNUMBERX'
    DL_BASE = 'https://ecmc.state.co.us/weblink/XLINKX'
    adir = getcwd()
    dir_add = path.join(adir,FOLDER)

    if FULL_UPDATE:
        OLD_YEAR = 1900
    else:
        OLD_YEAR = datetime.datetime.now().year-4
    
    SHL_BHL_THRESH = 2000
          
    #Read UWI files and form UWI list
    WELL_LOC = read_shapefile(shp.Reader('Wells.shp'))
    WELLPLAN_LOC = read_shapefile(shp.Reader('Directional_Lines_Pending.shp'))
    WELLLINE_LOC = read_shapefile(shp.Reader('Directional_Lines.shp'))

    WELL_LOC['UWI10'] = WELL_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))
    WELLPLAN_LOC['UWI10'] = WELLPLAN_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))
    WELLLINE_LOC['UWI10'] = WELLLINE_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))

    # CREATE ABSOLUTE LOCATION TABLE if True:
    WELL_LOC = read_shapefile(shp.Reader('Wells.shp'))
    
    WELL_LOC['UWI10'] = WELL_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))
    WELL_LOC = WELL_LOC.loc[~(WELL_LOC['UWI10'] == 500000000)]
    WELL_LOC['X'] = WELL_LOC.coords.apply(lambda x:x[0][0])
    WELL_LOC['Y'] = WELL_LOC.coords.apply(lambda x:x[0][1])
    WELL_LOC['XBHL'] = WELL_LOC.coords.apply(lambda x:x[-1][0])
    WELL_LOC['YBHL'] = WELL_LOC.coords.apply(lambda x:x[-1][1])

    WELLLINE_LOC = read_shapefile(shp.Reader('Directional_Lines.shp'))
    WELLLINE_LOC['UWI10'] = WELLLINE_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))
    WELLLINE_LOC = WELLLINE_LOC.loc[~(WELLLINE_LOC['UWI10'] == 500000000)]
    WELLLINE_LOC['X'] = WELLLINE_LOC.coords.apply(lambda x:x[0][0])
    WELLLINE_LOC['Y'] = WELLLINE_LOC.coords.apply(lambda x:x[0][1])
    WELLLINE_LOC['XBHL'] = WELLLINE_LOC.coords.apply(lambda x:x[-1][0])
    WELLLINE_LOC['YBHL'] = WELLLINE_LOC.coords.apply(lambda x:x[-1][1])
    
    LOC_COLS = ['UWI10','X','Y','XBHL','YBHL']
    LOC_DF = WELLLINE_LOC[LOC_COLS].drop_duplicates()
    m = WELL_LOC.index[~(WELL_LOC.UWI10.isin(LOC_DF.UWI10))]
    LOC_DF = pd.concat([LOC_DF,WELL_LOC.loc[m,LOC_COLS].drop_duplicates()])
    LOC_DF.UWI10.shape[0]-len(LOC_DF.UWI10.unique())
    
    LOC_DF[['XFEET','YFEET']] = pd.DataFrame(convert_XY(LOC_DF.X,LOC_DF.Y,26913,2231)).T.values
    LOC_DF[['XBHLFEET','YBHLFEET']] = pd.DataFrame(convert_XY(LOC_DF.XBHL,LOC_DF.YBHL,26913,2231)).T.values
 
    LOC_DF['DELTA'] = ((LOC_DF['YBHLFEET'] - LOC_DF['YFEET'])**2 +  (LOC_DF['XBHLFEET'] - LOC_DF['XFEET'])**2)**0.5
          
    m = LOC_DF['DELTA']>SHL_BHL_THRESH
    SHP_UWIS = list(LOC_DF.loc[m,'UWI10'].unique())

    #SHP_UWIS = list(set(WELL_LOC['UWI10']).union(set(WELLPLAN_LOC['UWI10'])).union(set(WELL_LOC['UWI10'])))
          
    connection_obj = sqlite3.connect(DB)
          
    try:
        QRY = f'SELECT UWI10, YY AS YEAR FROM (SELECT UWI10, min(CAST(strftime("%Y",date(First_of_Month)) AS INT)) AS YY FROM PRODDATA GROUP BY UWI10) WHERE YY > {OLD_YEAR}'
        UWIPROD = pd.read_sql(QRY, connection_obj)
        UWIPROD = UWIPROD.UWI10.tolist()

        #df = pd.read_sql('SELECT * FROM PRODUCTION_SUMMARY', connection_obj)
        df = pd.read_sql('SELECT UWI10, min(date(First_of_Month)) as Month1 FROM PRODDATA GROUP BY UWI10', connection_obj)
        df = df.dropna()
              
        UWIKEY = GetKey(df,'UWI')
        UWIKEY = df[UWIKEY].dropna(how='all',axis=0).map(lambda x: len(str(x))).max(axis=0).sort_values(ascending=False).index.tolist()
        UWIKEY = df[UWIKEY].map(lambda x: bool(re.search(r'[a-zA-Z\\\/]',str(x)))).sum(axis=0).sort_values(ascending=True).index[0]
              
        df['UWI10'] = df[UWIKEY].apply(lambda x: WELLAPI(x).API2INT(10))
        
    except:
        UWIPROD = []
        df = pd.DataFrame()
     
    if not df.empty:      
        df = DF_UNSTRING(df)
        df.Month1 = pd.to_datetime(df.Month1)
        OLD_UWI = df.loc[df.Month1.dt.year<OLD_YEAR, 'UWI10'].tolist()
        NEW_UWI = df.loc[df.Month1.dt.year>OLD_YEAR, 'UWI10'].tolist()
    else:
        OLD_UWI = NEW_UWI = []       

    FLIST = listdir(dir_add)
    FLIST = [f for f in FLIST if f.lower().endswith(('.xls','xlsx','xlsm'))]

    # WORKAROUND TO REMOVE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<      
    FLIST = [f for f in FLIST if '_UWI' in f]               
    SURVEYED_UWIS = [int(re.search(r'.*_UWI(\d*)\.',F).group(1)) for F in FLIST]
    
    # shapefile with any production
    PROD_SHP_UWIS = set(SHP_UWIS).intersection(set(OLD_UWI).union(set(NEW_UWI)))
    # surveyed wells that aren't recent
    OLD_SURVEYED_UWIS = set(SURVEYED_UWIS) - set(NEW_UWI)

    # Producing wells not in list of good surveys
    SURVEYDATA = pd.read_sql('SELECT * FROM SURVEYDATA WHERE INC>88',connection_obj)
    SURVEYDATA['OFFSET'] = (SURVEYDATA['NORTH_dY']**2+SURVEYDATA['EAST_dX']**2)**0.5
    # # SHL-BHL offset passes threshold
    OFFSET_THRESH_UWIS = SURVEYDATA.loc[SURVEYDATA['OFFSET']>SHL_BHL_THRESH,'UWI'].unique()
    
    m = ((SURVEYDATA.INC-SURVEYDATA.INC.apply(floor))>0) * ((SURVEYDATA.AZI-SURVEYDATA.AZI.apply(floor))>0)
    #SURVEYDATA.loc[m]
    ACTUALS_UWI = []
    MISSING_SURVEY_UWIQ = []
          
    if FULL_UPDATE:
        #SURVEYED_UWIS = []
        #NEW_UWI = []
        UWIlist = list(PROD_SHP_UWIS)
    else:
        UWIlist = list(set(UWIPROD).intersection(PROD_SHP_UWIS) - set(OLD_SURVEYED_UWIS))
                       
    #if len(OLD_UWI)>0:
    #    UWIlist = list(set(NEW_UWI).union(SHP_UWIS) - set(OLD_UWI) - set(SURVEYED_UWIS)) 
    #else:
    #    UWIlist = list(set(SHP_UWIS) - set(SURVEYED_UWIS)) 
        
    UWIlist.sort(reverse=True)
    
    #UWIlist = list(set(UWIPROD) - set(OLD_UWI))

    # Create download folder
    if not path.exists(dir_add):
            makedirs(dir_add)
    
    print('{0} UWI\'s for surveys'.format(len(UWIlist)))
          
    func = partial(CO_Get_Surveys, 
                   URL_BASE=URL_BASE, 
                   DL_BASE=DL_BASE, 
                   FOLDER = dir_add)

    if len(UWIlist)>500:
        processors = max(1,floor(multiprocessing.cpu_count()/1))
        processors = min(8,processors)

        chunksize = int(len(UWIlist)/processors)
        chunksize = 500
        batch = int(len(UWIlist)/chunksize)
        #processors = max(processors,batch)
        data=np.array_split(UWIlist,batch)
        #print (f'batch = {batch}')

        with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
           #f = {executor.submit(CO_Get_Surveys,a): a for a in data
           f = {executor.submit(func,a): a for a in data}                
    elif len(UWIlist)>0:
        func(UWIlist)
        #CO_Get_Surveys(UWIlist)


def UPDATE_PROD(FULL_UPDATE = False, DB = 'FIELD_DATA.db'):
    pathname = path.dirname(argv[0])
    adir = path.abspath(pathname)
    dir_add = path.join(adir,'PRODFOLDER')
          
    connection_obj = sqlite3.connect(DB)
    
    if not 'PRODDATA' in LIST_SQL_TABLES(connection_obj):
          FULL_UPDATE = True

    if not FULL_UPDATE:
        QRY = 'SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY UWI10 ORDER BY FIRST_OF_MONTH DESC) AS RANK_NO FROM PRODDATA) P1 WHERE P1.RANK_NO=1 AND P1.WELL_STATUS IN (\'PA\',\'AB\')'
        df_prod = pd.read_sql(QRY, connection_obj)  
        df_prod.First_of_Month = pd.to_datetime(df_prod.First_of_Month)
          
    QRY = '''SELECT DISTINCT printf('%014d',APINumber) as API14 FROM FRAC_FOCUS WHERE SUBSTR(API14,1,2)='05' '''
    FF_LIST = pd.read_sql(QRY,connection_obj)
    FF_LIST = FF_LIST.API14.tolist()

    QRY = '''SELECT *
              FROM SCOUTDATA
              WHERE (JULIANDAY('now') - JULIANDAY(STATUS_DATE)) < 600 '''
          
    SCOUT_LIST = pd.read_sql(QRY,connection_obj)
    m = SCOUT_LIST.loc[SCOUT_LIST.isna().sum(axis=1) < 19].index
    SCOUT_LIST = SCOUT_LIST.loc[m,'UWI10']  
    SCOUT_LIST = list(set(SCOUT_LIST))
          
    FULL_SCOUT_LIST = pd.read_sql('SELECT DISTINCT UWI10 FROM SCOUTDATA', connection_obj).iloc[:,0].tolist()
   
    connection_obj.close()

    if not FULL_UPDATE:                    
        df_prod = DF_UNSTRING(df_prod)
        df_prod.First_of_Month = pd.to_datetime(df_prod.First_of_Month)      
        df_prod['DAYS_SINCE_LAST_PROD'] = (datetime.datetime.now()-df_prod.First_of_Month).dt.days

        if df_prod['DAYS_SINCE_LAST_PROD'].min() > 180:
            FULL_UPDATE = True

        NONPRODUCERS = df_prod.loc[(df_prod.DAYS_SINCE_LAST_PROD>(30*15)) * (df_prod.Well_Status.isin(['AB','PA'])),'UWI10'].tolist()    
        UWIlist = list(set(SCOUT_LIST) - set(NONPRODUCERS))
    else:
        UWIlist = FULL_SCOUT_LIST

    UWIlist = [WELLAPI(x).STRING(10) for x in UWIlist]
    UWIlist.sort(reverse=True)
    
    # Create download folder
    if not path.exists(dir_add):
        makedirs(dir_add)
    # Parallel Execution if 1==1:
    processors = max(1,floor(1+multiprocessing.cpu_count()/2))
        
    chunksize = int(len(UWIlist)/processors)
    chunksize = 1000
    batch = int(len(UWIlist)/chunksize)
    #processors = max(processors,batch)
    data=np.array_split(UWIlist,batch)
    data = [list(a) for a in data]
          
    #print (f'batch = {batch}')
    func = partial(Get_ProdData,file=DB,SQLFLAG=0, PROD_DATA_TABLE = 'PRODDATA', PROD_SUMMARY_TABLE = 'PRODUCTION_SUMMARY')
    with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
        f = {executor.submit(func,a): a for a in data}
    
    
