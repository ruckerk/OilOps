from ._FUNCS_ import *
from .WELLAPI import *
from .DATA import *
from .SURVEYS import *
from .MAP import convert_XY

__all__ = ['CONSTRUCT_DB',
          'UPDATE_SURVEYS',
          'UPDATE_PROD']

def CONSTRUCT_DB(DB_NAME = 'FIELD_DATA.db'):
    pathname = path.dirname(argv[0])
    adir = path.abspath(pathname)
    
    connection_obj = sqlite3.connect(DB_NAME)
    c = connection_obj.cursor()
    SURVEY_FILE_FIELDS = {'FILENAME':['CHAR'],'FILE':['BLOB']}
    INIT_SQL_TABLE(connection_obj, 'SURVEYFILES', SURVEY_FILE_FIELDS)
    
    c.execute(''' SELECT DISTINCT FILENAME FROM SURVEYFILES  ''')
    LOADED_FILES = c.fetchall()
    LOADED_FILES = list(set(itertools.chain(*LOADED_FILES)))
    
    SURVEYFOLDER = 'SURVEYFOLDER'
    XLSLIST = filelist(SURVEYFOLDER,None,None,'.xls')
    USELIST = list(set(XLSLIST).difference(set(LOADED_FILES)))

    print('{} FILES TO LOAD'.format(len(USELIST)))

    for F in USELIST:
        if '~' in F:
            continue
        B = convertToBinaryData(path.join(SURVEYFOLDER,F))
        
        SQL_QRY = ''' INSERT INTO SURVEYFILES(FILENAME, FILE) VALUES (?,?)'''
        data_tuple = (F,B)

        load_surveyfile(connection_obj,data_tuple)
        
    connection_obj.commit()

    #DATA = READ_SQL_TABLE(connection_obj,'SURVEYFILES')
    #DATA_df = pd.DataFrame(DATA, columns = ['FILE','BLOB'])
    DATA_df = pd.read_sql('SELECT * FROM SURVEYFILES', connection_obj)

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

    QRY = 'SELECT DISTINCT UPPER(FILE) FROM SURVEYDATA'
    c.execute(QRY)
    SCANNEDFILES = c.fetchall()
    SCANNEDFILES = list(itertools.chain(*SCANNEDFILES))
    m = DATA_df.loc[~DATA_df.FILENAME.str.upper().isin(SCANNEDFILES)].index

    S_KEYS = pd.read_sql('SELECT * FROM SURVEYDATA LIMIT 1', connection_obj).keys()
    OUT = pd.DataFrame(columns = S_KEYS)

    warnings.filterwarnings('ignore')

    batch = min(5000,len(m))
    chunksize = max(int(len(m)/batch),1)
    mm=np.array_split(m,chunksize)

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
         
    SQL_UNDUPLICATE(connection_obj,'SURVEYDATA')
    
    warnings.filterwarnings('default')

    ALL_SURVEYS = pd.read_sql_query('SELECT * FROM SURVEYDATA',connection_obj)
    ALL_SURVEYS['UWI10'] = ALL_SURVEYS.UWI.apply(lambda x:WELLAPI(x).API2INT(10))
    
    #ddf = CO_ABS_LOC(ALL_SURVEYS.UWI10,'CO_3_2.1.sqlite')
    
    # DROP OR FIX UWIS IN CO_ABS_LOC, NOT USED CURRENTLY
    #ALL_SURVEYS = pd.merge(ALL_SURVEYS, ddf,how = 'left',on='UWI10',left_index = False, right_index = False)
    #ALL_SURVEYS['X_PATH'] = ALL_SURVEYS[['EAST_dX',AL'X_FEET']].sum(axis=1)
    #ALL_SURVEYS['Y_PATH'] = ALL_SURVEYS[['NORTH_dY','Y_FEET']].sum(axis=1)
    
    # OLD PREFERRED SURVEYS
    QRY = 'SELECT FILE, UWI10, FAVORED_SURVEY FROM FAVORED_SURVEYS' 
    OLD_PREF = pd.read_sql(QRY, connection_obj)
    

    
    m = ALL_SURVEYS[['UWI10','FILE']].merge(OLD_PREF[['UWI10','FILE']],indicator = True, how='left').loc[lambda x : x['_merge']!='both'].index

    #assign favored file definitions from old table
    ALL_SURVEYS['FAVORED_SURVEY'] = -1
    ALL_SURVEYS.loc[~ALL_SURVEYS.index.isin(m),'FAVORED_SURVEY'] = ALL_SURVEYS.loc[~ALL_SURVEYS.index.isin(m),['UWI10','FILE']].merge(OLD_PREF,on=['UWI10','FILE'])['FAVORED_SURVEY']
    
    # SET FAVORED SURVEY to 1/0 binary
    m1 = ALL_SURVEYS['FAVORED_SURVEY']==ALL_SURVEYS['FILE']
    ALL_SURVEYS.loc[m1,'FAVORED_SURVEY'] = 1
    ALL_SURVEYS.loc[~m1,'FAVORED_SURVEY'] = 0     
    ALL_SURVEYS.FAVORED_SURVEY = ALL_SURVEYS.FAVORED_SURVEY.astype(int)

    #UWIs with new file
    NEW_UWI = ALL_SURVEYS.loc[m,'UWI10'].unique().tolist()
    m = ALL_SURVEYS.loc[ALL_SURVEYS.UWI10.isin(NEW_UWI)].index  

    if len(m)>0:
        CONDENSE_DICT = Condense_Surveys(ALL_SURVEYS.loc[m,['UWI10','FILE','MD', 'INC', 'AZI', 'TVD','NORTH_dY', 'EAST_dX']])
        ALL_SURVEYS.loc[m,'FAVORED_SURVEY'] = ALL_SURVEYS.loc[m,'UWI10'].apply(lambda x:CONDENSE_DICT[x])
    
    #m = FAVORED_SURVEY.apply(lambda x:CONDENSE_DICT[x.UWI10] == x.FILE, axis=1)
    m = (ALL_SURVEYS.FAVORED_SURVEY == 1)
    USE_SURVEYS = ALL_SURVEYS.loc[m]
    USE_SURVEYS['UWI10'] = USE_SURVEYS.UWI.apply(lambda x:WELLAPI(x).API2INT(10))

    # CREATE ABSOLUTE LOCATION TABLE if True:
    WELL_LOC = read_shapefile(shp.Reader('Wells.shp'))
    
    WELL_LOC['UWI10'] = WELL_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))
    WELL_LOC = WELL_LOC.loc[~(WELL_LOC['UWI10'] == 500000000)]
    WELL_LOC['X'] = WELL_LOC.coords.apply(lambda x:x[0][0])
    WELL_LOC['Y'] = WELL_LOC.coords.apply(lambda x:x[0][1])

    WELLPLAN_LOC = read_shapefile(shp.Reader('Directional_Lines_Pending.shp'))
    WELLPLAN_LOC['UWI10'] = WELLPLAN_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))
    WELLPLAN_LOC = WELLPLAN_LOC.loc[~(WELLPLAN_LOC['UWI10'] == 500000000)]
    WELLPLAN_LOC['X'] = WELLPLAN_LOC.coords.apply(lambda x:x[0][0])
    WELLPLAN_LOC['Y'] = WELLPLAN_LOC.coords.apply(lambda x:x[0][1])
    
    WELLLINE_LOC = read_shapefile(shp.Reader('Directional_Lines.shp'))
    WELLLINE_LOC['UWI10'] = WELLLINE_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))
    WELLLINE_LOC = WELLLINE_LOC.loc[~(WELLLINE_LOC['UWI10'] == 500000000)]
    WELLLINE_LOC['X'] = WELLLINE_LOC.coords.apply(lambda x:x[0][0])
    WELLLINE_LOC['Y'] = WELLLINE_LOC.coords.apply(lambda x:x[0][1])
    
    LOC_COLS = ['UWI10','X','Y']
    LOC_DF = WELLLINE_LOC[LOC_COLS].drop_duplicates()
    m = WELLPLAN_LOC.index[~(WELLPLAN_LOC.UWI10.isin(LOC_DF.UWI10))]
    LOC_DF = pd.concat([LOC_DF,WELLPLAN_LOC.loc[m,LOC_COLS].drop_duplicates()])
    m = WELL_LOC.index[~(WELL_LOC.UWI10.isin(LOC_DF.UWI10))]
    LOC_DF = pd.concat([LOC_DF,WELL_LOC.loc[m,LOC_COLS].drop_duplicates()])
    LOC_DF.UWI10.shape[0]-len(LOC_DF.UWI10.unique())
    
    LOC_DF[['XFEET','YFEET']] = pd.DataFrame(convert_XY(LOC_DF.X,LOC_DF.Y,26913,2231)).T.values

    LOC_COLS = {'UWI10': 'INTEGER',
                'X': 'REAL',
                'Y': 'REAL',
                'XFEET':'REAL',
                'YFEET':'REAL'}
    INIT_SQL_TABLE(connection_obj, 'SHL', LOC_COLS)
    LOC_DF.to_sql(name = 'SHL', con = connection_obj, if_exists='replace', index = False, dtype = LOC_COLS)
    connection_obj.commit()
    
    #FLAG PREFERRED SURVEYS if True:
    ALL_SURVEYS['UWI10'] = ALL_SURVEYS.UWI.apply(lambda x:WELLAPI(x).API2INT(10))
    ALL_SURVEYS = ALL_SURVEYS.merge(LOC_DF[['UWI10','XFEET','YFEET']],how = 'left', on = 'UWI10')
    ALL_SURVEYS['NORTH'] = ALL_SURVEYS[['NORTH_dY','YFEET']].sum(axis=1)
    ALL_SURVEYS['EAST'] = ALL_SURVEYS[['EAST_dX','XFEET']].sum(axis=1)
    ALL_SURVEYS.rename({'YFEET':'SHL_Y_FEET','XFEET':'SHL_X_FEET'}, axis = 1, inplace = True)

   
    #ALL_SURVEYS['FAVORED_SURVEY'] = ALL_SURVEYS.apply(lambda x: CONDENSE_DICT[x.UWI10], axis = 1).str.upper() == ALL_SURVEYS.FILE.str.upper()

    SCHEMA = {'UWI10': 'INTEGER', 'FILE':'TEXT', 'FAVORED_SURVEYS':'INTEGER'}
    #INIT_SQL_TABLE(connection_obj,'SURVEYDATA', SCHEMA)

    m = ALL_SURVEYS.FAVORED_SURVEY==1
    ALL_SURVEYS.loc[:,['UWI10','FILE','FAVORED_SURVEY']].drop_duplicates().to_sql('FAVORED_SURVEYS',
                                                   connection_obj,
                                                   schema = SCHEMA,
                                                   if_exists = 'replace')
    
    connection_obj.commit()

    # XYZ SPACING CALC
    # PARENT CUM AT TIME OF CHILD FRAC
    # PAD ASSIGNMENTS FROM SPACING GROUPS (SAME SHL)
    # UNIT ASSIGNMENTS: NEAREST IN DATE DIFF RANGE AND EITHER LATERAL IS 90% OVERLAPPING OTHER

    QRY = 'SELECT CAST(max(S.UWI10, P.UWI10) as INT) AS UWI10, S.FIRST_PRODUCTION_DATE, S.JOB_DATE, S.JOB_END_DATE, P.FIRST_PRODUCTION FROM SCOUTDATA AS S LEFT JOIN PRODUCTION_SUMMARY AS P ON S.UWI10 = P.UWI10'
    QRY = 'SELECT MAX(S.UWI10, P.UWI10) AS UWI10, P.MONTH1, S.JOB_DATE, S.JOB_END_DATE, S.FIRST_PRODUCTION_DATE FROM SCOUTDATA AS S LEFT JOIN PRODUCTION_SUMMARY AS P ON S.UWI10 = P.UWI10'
          
    WELL_DF = pd.read_sql(QRY,connection_obj)
    WELL_DF=WELL_DF.dropna(how='all',axis = 0)
    WELL_DF = WELL_DF.loc[~WELL_DF.UWI10.isna()]
    WELL_DF = DF_UNSTRING(WELL_DF)
    WELL_DF.sort_values(by = 'FIRST_PRODUCTION_DATE',ascending = False, inplace = True)

    UWIlist = WELL_DF.sort_values(by = 'UWI10', ascending = False).UWI10.tolist()

    # MAJOR UPGRADE FOR SPEED: XYZ ONLY FOR NEW SURVEYS, AND WELLS NEAR NEW SURVEYS
    # FOR WELL IN NEW_SURVEYS: ALL_SURVEYS[[NORTH,EAST]] - [[NORTH,EAST]] <= 10000
    # UWILIST = AGGREGATED RESULT
          
    # FIND ALL UNCHANGED UWI-FILE PAIRS & USE THE OTHERS
    m = pd.merge(ALL_SURVEYS[['UWI10','FILE']], OLD_PREF[['UWI10','FILE']], on=['UWI10','FILE'], how='left', indicator='TEST').TEST!='both'
    m = ALL_SURVEYS.index[(ALL_SURVEYS.FAVORED_SURVEY==1) *m]
    UWIlist = ALL_SURVEYS.loc[m,'UWI10'].unique()
    
    XYZ_OLD = pd.DataFrame()
    if 'SPACING' in LIST_SQL_TABLES(connection_obj):
        XYZ_OLD = pd.read_sql('SELECT * FROM SPACING', con = connection_obj)
        XYZ_OLD.rename(columns = {'XYZFILE':'FILE'}, inplace = True)
        all_df = pd.merge(ALL_SURVEYS[['UWI10','FILE','FAVORED_SURVEY']], XYZ_OLD[['UWI10','FILE']], how='left', indicator='TEST')
        UWIlist = all_df.loc[(all_df.TEST!='both')*(all_df.FAVORED_SURVEY==1),'UWI10'].unique()
          
        UWI_MEANS = ALL_SURVEYS.loc[(ALL_SURVEYS.INC>88)*(ALL_SURVEYS.FAVORED_SURVEY==1),['UWI10','FILE','NORTH','EAST']].groupby(by=['UWI10','FILE'], axis = 0).mean().reset_index(drop=False)
        
        # USE A SPATIAL FUNCTION HERE
        # BUFFER AROUND CHANGING SURVEY PTS
        # FIND ALL UWIS TOUCHING BUFFER AREAS            
                        
    else:
        UWIlist = ALL_SURVEYS.loc[ALL_SURVEYS.FAVORED_SURVEY==1,'UWI10'].unique()

   

    processors = max(1,floor(multiprocessing.cpu_count()/1))
    
    func = partial(XYZSpacing,
            xxdf= ALL_SURVEYS.loc[m,['UWI10','FILE','MD', 'INC', 'AZI', 'TVD','NORTH', 'EAST']],
            df_UWI = WELL_DF,
            DATELIMIT = 360,
            SAVE = False)
    
    XYZ = pd.DataFrame()
    if len(UWIlist) >1000:
        chunksize = int(len(UWIlist)/processors)
        chunksize = min(2000, chunksize)
        batches = int(len(UWIlist)/chunksize)
        data=np.array_split(UWIlist,batches)

        with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
            f = {executor.submit(func,a): a for a in data}
        
        for i in f.keys():
            XYZ = XYZ.append(i.result(), ignore_index = True)
    elif len(UWIlist)>0:
        XYZ = func(UWIlist)
    
    if ~XYZ.empty:
        XYZ = DF_UNSTRING(XYZ)    
        XYZ_COLS = FRAME_TO_SQL_TYPES(XYZ)
        
        if ~XYZ_OLD.empty:
            XYZ = pd.concat([XYZ, XYZ_OLD.loc[~XYZ_OLD.UWI10.isin(XYZ.UWI10)]], axis = 0, join = 'outer', ignore_index = True)
          
        XYZ.to_sql(name = 'SPACING', con = connection_obj, if_exists='replace', index = False, dtype = XYZ_COLS)
        connection_obj.commit()
          
    ###################
    # PRODUCTION DATA #
    ###################

    ##############
    # SCOUT DATA #
    ##############
    

    ###################
    # FRAC FOCUS DATA #
    ###################
    # CREATE FRAC FOCUS TABLE
    FF = DATA.Merge_Frac_Focus(DIR = 'FRAC_FOCUS', SAVE = False)
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
 
def UPDATE_SURVEYS(DB = 'FIELD_DATA.db'):
    ###############
    # GET SURVEYS #
    ############### #if True:
    # Initialize constants
    global URL_BASE
    URL_BASE = 'https://cogcc.state.co.us/weblink/results.aspx?id=XNUMBERX'
    global DL_BASE 
    DL_BASE = 'https://cogcc.state.co.us/weblink/XLINKX'
    global pathname
    pathname = path.dirname(argv[0])
    global adir
    adir = path.abspath(pathname)
    global dir_add
    dir_add = path.join(adir,'SURVEYFOLDER')

    #Read UWI files and form UWI list
    WELL_LOC = read_shapefile(shp.Reader('Wells.shp'))
    WELLPLAN_LOC = read_shapefile(shp.Reader('Directional_Lines_Pending.shp'))
    WELLLINE_LOC = read_shapefile(shp.Reader('Directional_Lines.shp'))

    WELL_LOC['UWI10'] = WELL_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))
    WELLPLAN_LOC['UWI10'] = WELLPLAN_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))
    WELLLINE_LOC['UWI10'] = WELLLINE_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))

    SHP_UWIS = list(set(WELL_LOC['UWI10']).union(set(WELLPLAN_LOC['UWI10'])).union(set(WELL_LOC['UWI10'])))
    
    OLD_YEAR = datetime.datetime.now().year-4
    
    try:
        connection_obj = sqlite3.connect(DB)
        UWIPROD = pd.read_sql("SELECT DISTINCT UWI10 FROM PRODDATA WHERE First_of_Month LIKE '2022%'", connection_obj)
        UWIPROD = UWIPROD.UWI10.tolist()

        df = pd.read_sql('SELECT * FROM PRODUCTION_SUMMARY', connection_obj)
        connection_obj.close()    
    
        UWIKEY = GetKey(df,'UWI')
        UWIKEY = df[UWIKEY].dropna(how='all',axis=0).applymap(lambda x: len(str(x))).max(axis=0).sort_values(ascending=False).index[0]
        df['UWI10'] = df[UWIKEY].apply(lambda x: WELLAPI(x).API2INT(10))
        
    except:
        UWIPROD = []
        df = pd.DataFrame()
     
    if not df.empty:      
        df = DF_UNSTRING(df)
        OLD_UWI = df.loc[df.Month1.dt.year<OLD_YEAR, 'UWI10'].tolist()
        NEW_UWI = df.loc[df.Month1.dt.year>OLD_YEAR, 'UWI10'].tolist()
    else:
        OLD_UWI =[]

    FLIST = list()
    for file in listdir(dir_add):
        if file.lower().endswith(('.xls','xlsx','xlsm')):
            FLIST.append(file)

 
    SURVEYED_UWIS = [int(re.search(r'.*_UWI(\d*)\.',F).group(1)) for F in FLIST]
    if len(OLD_UWI)>0:
        UWIlist = list(set(OLD_UWI).union(set(NEW_UWI)) - set(SURVEYED_UWIS)) 
    else:
        UWIlist = list(set(SHP_UWIS) - set(SURVEYED_UWIS)) 
        
    UWIlist.sort(reverse=True)
    
    #UWIlist = list(set(UWIPROD) - set(OLD_UWI))

    # Create download folder
    if not path.exists(dir_add):
            makedirs(dir_add)
    
    print('{0} UWI\'s for surveys'.format(len(UWIlist)))
          
    func = partial(CO_Get_Surveys, 
                   URL_BASE=URL_BASE, 
                   DL_BASE=DL_BASE, 
                   dir_add=dir_add)

    if len(UWIlist)>1000:
        processors = min(1,floor(multiprocessing.cpu_count()/2))

        chunksize = int(len(UWIlist)/processors)
        chunksize = 1000
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
          
    QRY = '''SELECT DISTINCT printf('%014d',APINumber) as API14 FROM FRAC_FOCUS WHERE SUBSTR(API14,1,2)='05' '''
    FF_LIST = pd.read_sql(QRY,connection_obj)
    FF_LIST = FF_LIST.API14.tolist()

    QRY = '''SELECT UWI10,
                    FIRST_PRODUCTION_DATE,
                    BBLS_H2O,
                    BBLS_OIL,
                    BTU_GAS,
                    CALC_GOR,
                    GRAVITY_OIL,
                    JOB_DATE,
                    JOB_END_DATE,
                    MAX_PRESSURE,
                    MIN_FRAC_GRADIENT,
                    PRODUCED_WATER_USED,
                    RECYCLED_WATER_USED,
                    SPUD_DATE,
                    STATUS_DATE,
                    STIM_FLUID,
                    STIM_PROPPANT,
                    TOTAL_FLUID_USED
                    TOTAL_PROPPANT_USED,
                    TREATMENT_SUMMARY,
                    TREAT_FLUID,
                    TREAT_PROPPANT,
                    TREAT_PRESSURE
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
    func = partial(Get_ProdData,file=DB,SQLFLAG=1, PROD_DATA_TABLE = 'PRODDATA', PROD_SUMMARY_TABLE = 'PRODUCTION_SUMMARY')
    with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
        f = {executor.submit(func,a): a for a in data}
    
    
