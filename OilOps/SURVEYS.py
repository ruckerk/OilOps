from ._FUNCS_ import *
from .WELLAPI import WELLAPI as WELLAPI
from .MAP import convert_XY

# ISSUES
# LOADED SURVEYS ARE BRCOMING INTEGERS! DONT LOSE DECIMAL
# MANY WELLS ARE CATCHING WRONG COLUMN eg. TVD for INC

# BUG FIXES
#v101 as_type(float) errors fixed with pd.to_numeric(errors='coerce')
#v106 looks for API col
#v107 checks file lists and only appends new filess
#v200 new file name handling, modin, JSON

# NEED TO HANDLE UNLABELED COLUMNS

# needs to be updated to populate SQL db
# parallxyz

__all__ = ['Find_API_Col',
          'ExtractSurvey',
          'survey_from_excel',
          'SurveyCols',
          'JoinSurveysInFolder',
          'Survey_Join',
          'COGCC_SURVEY_CLEANUP',
          'APIfromFrame',
          'Condense_Surveys',
          'XYZSpacing',
          'LeftRightSpacing']

def Find_API_Col(df_inAPI):
    # NOT WORKING RIGHT returning datestrings
          
    #if 1==1:
    APIterms = ['API','UWI']
    rAPIterms = '|'.join(APIterms)
    
    df2 = df_inAPI.copy(deep = True)
    lowlim = 10**(8)
    highlim = 10**14
    def STR2INT(INPUT):       
        if bool(re.findall(r'[a-z]',str(INPUT),re.I)):
            return None
        try:
            val = WELLAPI(INPUT).str2num()
        except:
            val = None
        return val
                  
    df2 = df2.map(lambda x:STR2INT(x))
    df2 = df2[(df2>lowlim) & (df2<highlim)].dropna(axis=0,how='all').dropna(axis=1,how='all')

    if df2.empty:
        return (None,None)
    keys = df2.keys()
    
    keylist=[]
    UWIlist = pd.Series(data=None,dtype = int)

    knum = None
    UWI = None
    
    for k in keys:        
        # check for GT 50 rows per value
        if df2[k].shape[0]/len(df2[k].unique()) > 50:
            keylist.append(k)
        UWIlist = UWIlist.add(pd.Series((df2[k].dropna().unique())))

    if len(keylist) > 0:
        longest = 0
        fav_k = keylist[0]
        
        for k in keylist:
            test = False
            # check for API/UWI key
            if any(x.upper() in k.upper() for x in APIterms):
                test=True   
            elif df.iloc[:,-3].astype(str).str.contains(f'{rAPIterms}',case = False, flags=re.I, regex=True).max():
                test = True

            # confirm numbers are > 10 and less than 14 digits
            df2[k] = df2[k].apply(lambda x:WELLAPI(x).str2num())
            if (df2.loc[(df2[k]<highlim) & (df2[k]>lowlim),k].dropna().shape[0] > longest):
                fav_k = k
                longest = df2.loc[(df2[k]<highlim) & (df2[k]>lowlim),k].dropna().shape[0]
        knum = df_inAPI.keys().get_loc(fav_k)

    if UWIlist.empty == False:       
        UWIlist = np.array([WELLAPI(xi).API2INT(14) for xi in UWIlist])

        UWI = np.unique(UWIlist)
        if len(UWI)>1:
            return (None,None)
            #raise Exception('Found more than one UWI in Find_API_Col')
        try:
            UWI = int(UWI[0])
        except:
            print('Could not find useable API')
            UWI = None        
    return(UWI,knum)


def APIfromFrame(df_in):
    terms = list()
    if isinstance(df_in,dict):
        for k in df_in:
            terms.append(APIfromString(df_in[k].to_string(),ListResult=True))
        term = max(set(terms), key = terms.count)
    if isinstance(df_in,pd.DataFrame):
        term = APIfromString(df_in.to_string())
    return term


    
def COGCC_SURVEY_CLEANUP(df_in):#if True:
    df_in.columns = df_in.keys().astype(str)      
    skeys = df_in.keys().str.upper().str.strip().str.replace(' ','')
    mask = skeys.str.contains('HEADER')
    skey = list(df_in.keys()[mask])
    # Look for normal Col 1 template terms
    if len(skey) == 1:
        df_dummy = df_in[skey[0]].str.upper().str.strip().str.replace(' ','')
        m1 = df_dummy.str.contains('OPERATORNAME').fillna(False)
        m2 = df_dummy.str.contains('OPERATORNUMBER').fillna(False)
        m3 = df_dummy.str.contains('NORTHREFERENCE').fillna(False)

        # If all strings above are found, rename columes
        if df_in.loc[m1 + m2 + m3,skey[0]].size == 3:
            col1 = df_in.keys().get_loc(skey[0])
            default_keys = ['measured depth\n(ft)', 'inclination (°)', 'azimuth (°)', 'true vertical depth\n(ft)', 'northing \n+N/-S  (ft)', 'easting \n+E/-W  (ft)']
            
            ncol = len(skey) + len(default_keys)

            newkeys = pd.Series(df_in.keys())
            newkeys.iloc[col1:ncol]= skey+default_keys
            newkeys = list(newkeys)

            df_in.columns = newkeys
            return df_in
    else:
        return None 

def ExtractSurveyWrapper(df_in):
    OUT=pd.DataFrame()
    adf = df_in.copy()
    try:
        OUT = ExtractSurvey(adf)
        OUT = pd.DataFrame(OUT)
        if not OUT.empty:
            OUT.rename(columns = SurveyCols(OUT,False),inplace=True)
        else:
            raise Exception('ExtractSurvey yielded empty result') 
    except:
        try:
            R = FIND_SURVEY_HEADER(adf,False)
            if R!=None:
                H = FIND_SURVEY_HEADER(adf,True)
                adf = adf.loc[R[-1]:,:].iloc[1:,:]
                adf.columns = H
            df_in = COGCC_SURVEY_CLEANUP(adf)            
            df_in = pd.DataFrame(df_in)
            if df_in.empty:
                raise Exception('No survey found in dataframe')
            else:
                OUT = ExtractSurvey(df_in)
                OUT.rename(columns = SurveyCols(OUT, False),inplace=True)
            return OUT
        except:
            raise Exception('No survey found in dataframe')
    if not isinstance(OUT,pd.DataFrame):
        raise Exception('No survey found in dataframe')

    if OUT is None:
        OUT = pd.DataFrame()
    return OUT


def ExtractSurvey(df_in): #if True:
    outdf_in = pd.DataFrame()
    ReadUWI = APIfromFrame(df_in)
    if len(Find_API_Col(df_in)) >0:
        ReadUWI = Find_API_Col(df_in)[0]
              
    adf_in=df_in.copy(deep=True)

    try: 
        SurveyCols(df_in,False) # is first row survey header?
        if df_in[list(SurveyCols(df_in,False))].dropna(how='all',axis = 1).dropna(how='all',axis = 0).shape[0]>5:
            key_dict = SurveyCols(df_in,False)
            cols = list(key_dict)
          
            outdf_in = df_in[cols].copy(deep=True)
            outdf_in.rename(columns = key_dict,inplace=True)      
                    
            outdf_in['UWI'] = ReadUWI
##                (DEAD,APICOL) = Find_API_Col(df_in)
##                if APICOL != None:
##                    cols = cols + [df_in.keys()[APICOL]]
##                    outdf_in = df_in[cols].copy(deep=True)
##                    outdf_in.rename(columns ={df_in.keys()[APICOL]:'UWI'},inplace=True)
##                else:
##                    outdf_in = df_in[cols].copy(deep=True)
##                    if (ReadUWI != None):
##                        outdf_in['UWI'] = ReadUWI
                
            outdf_in = outdf_in.map(str2num)
            #outdf_in = DF_UNSTRING(outdf_in)
            outdf_in = outdf_in.apply(pd.to_numeric, errors = 'coerce', axis=0)
                    
            outdf_in = outdf_in.dropna(how='all',axis = 1)
            outdf_in = outdf_in.dropna(how='all',axis = 0)
            if ('UWI' in outdf_in.keys()) == False:
                outdf_in['UWI']=None
            #outdf_in.rename(columns = SurveyCols(outdf_in),inplace=True)
            outdf_in =   MIN_CURVATURE(outdf_in)     
            return outdf_in
    except:
        # test for strings
        test = re.findall(r'MD|TVD|DEPTH|INC|AZ',adf_in.to_string(),re.I)
        if len(test)>=3:
            for n in [1,2,3,4]:
                drow = -1
                for i in range(0,100): # is survey header in 1st 15 rows?
                    try:
                        df_in=adf_in.copy()
                        N = max(1,min(i,n))
                        concat_vals = df_in.iloc[i:i+N,:].apply(lambda row:'_'.join(row.values.astype(str)),axis=0)
                        #SurveyCols(concat_vals)

                        df_in = df_in.iloc[i+N:,:]
                        df_in.columns = concat_vals
                        key_dict = SurveyCols(df_in,False)
                        cols = list(key_dict)
                        newcols = list(key_dict.values())

                        df_in.reset_index(drop=True, inplace= True)

                        outdf_in = df_in[cols].copy(deep=True)
                        outdf_in.rename(columns = key_dict, inplace = True)
    ##                  outdf_in.rename(columns ={df_in.keys()[APICOL]:'UWI'},inplace=True)
                        
                        outdf_in['UWI'] = ReadUWI
    ##                    
    ##                    (DEAD,APICOL) = Find_API_Col(df_in)
    ##                    if APICOL != None:
    ##                        cols = cols + [df_in.keys()[APICOL]]
    ##                        outdf_in = df_in[cols].copy(deep=True)
    ##                        outdf_in.rename(columns ={df_in.keys()[APICOL]:'UWI'},inplace=True)
    ##                    else:
    ##                        outdf_in = df_in[cols].copy(deep=True)
    ##                        if (ReadUWI != None) & (ReadUWI != 0):
    ##                            outdf_in['UWI'] = ReadUWI

                        # WAS GOING TO BUILD A CHECK THAT LAST ROW IN KEY COLUMNS DOES NOT CONTAIN VALUES
                        #keycols = list()
                        #for c in cols:
                        #    keycols.append(outdf_in.keys().get_loc(c))
                        
                        #outdf_in = outdf_in.copy(deep=True)
                        #outdf_in = outdf_in.map(lambda x:WELLAPI(x).str2num())
                        #outdf_in = DF_UNSTRING(outdf_in)
                        outdf_in = outdf_in.map(str2num)
                        outdf_in = outdf_in.apply(pd.to_numeric, errors = 'coerce', axis=0)
                              
                              
                        #.apply(pd.to_numeric,errors='coerce').dropna(axis=0,how='any').shape[0]
                        test = outdf_in.loc[:10,newcols].dropna(how='any').shape[0]
                        if test<10:
                            continue

                        #for k in outdf_in.keys():
                            # GETTING SLICE ERROR HERE
                            #outdf_in.loc[:,k] = np.array(outdf_in.loc[:,k].astype(str).str.replace(r'[^0-9\.]*','',regex=True))
                            # GETTING SLICE ERROR HERE
                            #outdf_in.loc[:,k] = np.array(pd.to_numeric(outdf_in.loc[:,k], errors='coerce'))
                        outdf_in = outdf_in.apply(pd.to_numeric, errors='coerce')
                        outdf_in = outdf_in.dropna(how='all',axis = 1)
                        outdf_in = outdf_in.dropna(how='all',axis = 0)
                        if outdf_in.shape[0] > 5:
                            outdf_in = df_in[SurveyCols(df_in, False)]
                            outdf_in.rename(columns = SurveyCols(outdf_in,False),inplace=True)
                            if ('UWI' in outdf_in.keys()) == False:
                                outdf_in['UWI'] = None
                            outdf_in =   MIN_CURVATURE(outdf_in)
                            return outdf_in
                    except: pass



def CheckUWI(df_in):
    cols = SurveyCols(df_in)    
    outdf.UWI = outdf.UWI.apply(lambda x:WELLAPI(x).str2num())

def survey_from_excel(file, ERRORS = True): #if True:
    TUPLE_TEST = isinstance(file, tuple)
    READUWI = None
    if TUPLE_TEST:
          FNAME = file[0]
          file = file[1]
          
    ERR_FOLDER = None
    RUNERROR = False
    if ERRORS == True:
        ERR_FOLDER = ERRORFILES()

    outdf = pd.DataFrame()
    xl = {}
    # read excel as a dictionary of dataframes
    try:
        xl = pd.read_excel(file, sheet_name = None, engine = None, header = None)
    except:
        try:
            xl = pd.read_excel(file, sheet_name = None ,engine = 'openpyxl', header = None)
        except:
            #print(file+': ERROR')
            RUNERROR = True
    
    # make headers strings
    if isinstance(xl,dict):
        for i in xl:
            xl[i].columns = [str(s) for s in xl[i].columns]
    elif isinstance(xl,pd.DataFrame):
        xl.columns = [str(s) for s in xl.columns]

    if len(xl)==0:
        #print('FILE XL READ ERROR IN: '+ file)
        outdf = None
        if ERRORS == True:
            shutil.move(file, ERR_FOLDER)
        RUNERROR = True
        return None   

    if isinstance(xl,dict):
        R_LIST = []
        for x in xl:
            R_LIST.append(APIfromFrame(xl[x]))
        R_LIST = [WELLAPI(x).API2INT(14) for x in R_LIST if x != None]
        R_LIST = list(set(R_LIST))
        if len(R_LIST) > 0:
            READUWI = R_LIST[0] 
    else:
        READUWI = APIfromFrame(xl)
        
    if TUPLE_TEST:
        FILENAMEUWI =  APIfromString(FNAME,BlockT2 = True)
    else:
        FILENAMEUWI =  APIfromString(file,BlockT2 = True)
              
    READUWI2 = WELLAPI(READUWI).API2INT()
    FILENAMEUWI2 = WELLAPI(FILENAMEUWI).API2INT()
    if (FILENAMEUWI2 == READUWI2) & (FILENAMEUWI2 > 1e8):
        UWI = READUWI2
    elif (READUWI2 == 0) & (FILENAMEUWI2 >1e8):
        UWI = FILENAMEUWI2
    elif (FILENAMEUWI2 == 0) & (READUWI2 >1e8):
        UWI = READUWI2    
    else:
        UWI = None     
        
    if isinstance(xl,dict): # test if file read delivered a dictionary
        for k in xl.keys(): # for each sheet  #if True:
            df_s = xl[k].copy(deep=True)
            df_s = df_s.dropna(how='all',axis=0).dropna(how='all',axis=1)

            ext_df=pd.DataFrame()

            try:
                ext_df = (df_s)
                ext_df = pd.DataFrame(ext_df)
                ext_df = ExtractSurveyWrapper(ext_df)      
            except:
                continue

            #for kkey in SurveyCols().keys():
            #    list(SurveyCols(df_s).values())

       #     if len(list((ext_df).values)) > 5:
       #         outdf = ext_df
       #         break
            #else:
            #    UWI = set(list(outdf.UWI.apply(str2num)))
                
            #outdf = pd.concat([outdf,ext_df],axis=1,ignore_index=False)
            
            #print(ext_df.keys())
            
        if 'UWI' in outdf.keys():
            outdf['UWI'] = outdf.UWI.apply(lambda x: WELLAPI(x).str2num())
        else:
            outdf['UWI'] = UWI
            
            #pd.to_numeric(outdf.UWI, errors='coerce').dropna().to_list()
            #print('outdf dict done')
            #print(outdf)
            #if isinstance(outdf,pd.DataFrame):
            #    return outdf

    if isinstance(xl,pd.DataFrame):
        try:
            outdf = ExtractSurveyWrapper(xl)
            outdf = pd.DataFrame(outdf)
            if not 'UWI' in  outdf.keys():
                outdf['UWI'] = UWI
        except:
            pass
    
    if isinstance(outdf,pd.DataFrame):
        if outdf.empty:
            return None
        if not(UWI is None):            
            outdf['UWI'] = outdf.UWI.apply(lambda x: WELLAPI(x).API2INT())
            outdf = outdf.map(str2num)
            outdf = outdf.apply(pd.to_numeric, errors='coerce')
            outdf['UWI'].fillna(0,inplace=True)
            UWI_df = outdf['UWI'].max()
            if UWI_df == 0:
               outdf['UWI'] = UWI
            else:
               outdf['UWI'] = UWI_df               
            #outdf = outdf.loc[outdf.T.sum().index,:]
            outdf = outdf.dropna(thresh=3,axis=0)
    return outdf

def FIND_SURVEY_HEADER(df_in, return_header = False):
    for n in np.arange(0,4):
        for i,j in enumerate(df_in.index[0:min(100, df_in.shape[0])]):
            try:
                HEADER = df_in.loc[j:j+n,:].fillna('').astype(str).apply('_'.join,axis=0).tolist()
                x=SurveyCols(HEADER, False)
                ROWS = np.arange(j,j+n+1)
                if return_header:
                    return HEADER
                else:
                    return ROWS
            except: 
                pass

def SurveyCols(df_s_in=None, INCLUDE_NS = True):      
    sterms = {'MD':r'.*MEASURED.*DEPTH.*|.*MD.*|^\s*DEPTH\s*|(?:^|_)DEPTH(?:$|_)',
             'INC':r'.*INC.*|.*DIP.*',
             'AZI':r'.*AZI.*|.*AZM.*',
             'TVD':r'.*TVD.*|.*TRUE.*|.*VERTICAL.*DEPTH.*',
             'NORTH_dY':r'.*\+N.*|.*(?:\+){0,1}N(?:\+){0,1}(?:[\/\\]){0,1}(?:\-){0,1}S(?:\-){0,1}.*FT.*|.*N\+.*|^\s*N(?:[\/\\]){0,1}S\s*|.*NORTH(?!ING).*|(?:^|_)(?:\+){0,1}N(?:\+){0,1}(?:[\/\\]){0,1}(?:\-){0,1}S(?:\-){0,1}(?:$|_)|NS.*ft',
             'EAST_dX':r'.*\+E.*|.*(?:\+){0,1}E(?:\+){0,1}(?:[\/\\]){0,1}(?:\-){0,1}W(?:\-){0,1}.*FT.*|.*E\+.*|^\s*E(?:[\/\\]){0,1}W\s*|.*EAST(?!ING).*|(?:^|_)(?:\+){0,1}E(?:\+){0,1}(?:[\/\\]){0,1}(?:\-){0,1}W(?:\-){0,1}(?:$|_)|EW.*ft'
        
        #     ,'NORTH_Y':r'.*ORTH.*|.*\+N.*|.*NS.*FT.*|.*N/S*'
        #     ,'EAST_X':r'.*EAST.*|.*\+E.*|.*EW.*FT.*|.*E/W.*'
        }

    if INCLUDE_NS == False:
        for k in ['NORTH_dY','EAST_dX','NORTH_Y','EAST_X']:
            try:
                sterms.pop(k)
            except:
                pass
        #sterms.pop('TVD')

    if df_s_in is None:
        return(sterms)
    
    if isinstance(df_s_in,pd.Series):
        df_s_in=list(df_s_in)
        df_s_in= [str(s) for x in df_s_in]
        
    #if isinstance(df_s_in,pd.DataFrame):
    #    df_s_in=list(df_s_in.keys())
    
    for s in sterms:
        #print(sterms[s])
        if isinstance(df_s_in,pd.DataFrame):
            term = df_s_in.iloc[0,df_s_in.keys().astype(str).str.contains(sterms[s], regex=True, case=False,na=False)].keys()
            if not isinstance(term, str) and len(term)>0:
                term = term[0]
            if len(term)>0:
                sterms[s] = term
            else:
                sterms[s] = None
            #sterms[s]=df_s_in.iloc[0,df_s_in.keys().astype(str).str.contains(sterms[s], regex=True, case=False,na=False)].keys()[0]
            
            #sterms[s] = term
        if isinstance(df_s_in,list):
            df_s_in = [str(x) for x in df_s_in]
            sterms[s] = list(filter(re.compile('(?i)'+sterms[s]).match,df_s_in))
            if len(sterms[s])>0:
                sterms[s] = sterms[s][0]
            else:
                sterms[s] = None
                
   
    # sterms=dict((v, k) for k, v in sterms.iteritems())
    sterms = {v: k for k, v in sterms.items()}
    if None in list(sterms):
        raise Exception('Incomplete Survey Headers Found')
    return sterms

def Wrapped_Survey_Join(aa):
    return Survey_Join(*aa)

def SurveyCols_row(r_in):
    try: OUTPUT = SurveyCols(r_in.to_list())
    except: OUTPUT = None
    return OUTPUT

def str2num(IN):
        str_in = str(IN)
        if (str_in.upper() == 'NONE'):
            return None
        if str(IN).upper() == 'NONE':
            str_in = str(str_in)
            str_in = str_in.strip()
            str_in = re.sub(r'[-−﹣−–—−]','-',str_in)
            c = len(re.findall('-',str_in))
            val = re.sub(r'[^\-\s\d\.]',r'',str(str_in))
            val = re.sub(r'[^\-\s\d\.]',r'',str(val))
            if float(val) == int(val):
                val = int(val)
            #if c>1:             
            #    val = re.sub(r'[^0-9\.]','',str(str_in))
            #else:
            #    val = re.sub(r'[^0-9-\.]','',str(str_in))
            if (val == '') | (bool(re.match(r'^\s*$',str(val)))):
                return None
        else:
            val = IN
        return val
    
def Survey_Join(SAVEFILE, FLIST, ERRORS = True): #if True:
    if SAVEFILE != None:
        SAVEFILE = re.sub(re.compile(r'\.[^.]+$'),'',SAVEFILE)
    if ERRORS == True:
        ERR_FOLDER = ERRORFILES()
        ERR_FILES = list()
    pd.set_option('mode.chained_assignment', None)
    # if 1==1:
    df=pd.DataFrame()
    if isinstance(FLIST,(str,int)):
        FLIST=[FLIST]
    if isinstance(FLIST,(np.ndarray,pd.DataFrame,pd.Series)):
        FLIST=list(FLIST)
    #ERROR_LIST = list()
        
    for ffile in FLIST:#if 1==1:
        try:
            #print(ffile)
            if ffile.lower().endswith(('.xls','xlsx','xlsm')):

                #if 1==1:
                rdf = survey_from_excel(ffile)
                if not isinstance(rdf,pd.DataFrame):
                    print('NO SURVEY FOUND IN: '+ ffile)
                    if ERRORS == True:
                        shutil.move(ffile, ERR_FOLDER)
                    continue

                if rdf.empty:
                    print('NO SURVEY FOUND IN: '+ ffile)
                    if ERRORS == True:
                        shutil.move(ffile, ERR_FOLDER)
                    continue                    
          
                #standardize column names
                rdf=rdf.rename(columns=SurveyCols(rdf))
            
                # all columns to numeric while catching decimals in strings
                rdf = rdf.map(str2num)

                if rdf.shape[0]==0 or rdf.empty:
                    continue

                rdf['FILE']=ffile
                rdf.UWI = pd.Series([WELLAPI(xi).API2INT(14) for xi in rdf.UWI])
                
                if rdf.UWI.dropna().empty:
                    rdf.loc[:,'UWI'] = APIfromFilename(ffile,UWIlen=14)
                    
                
                if df.empty:
                    df=rdf
                    continue           
                #df.columns=rdf.columns
                try:
                    df=pd.concat([df, rdf], ignore_index=True)
                except:
                    print('ERROR IN: '+ ffile)
                    if ERRORS == True:
                        #specify full path to force overwrite
                        shutil.move(path.join(adir,ffile), path.join(ERR_FOLDER,ffile))
                    #ERROR_LIST.append(ffile)
        except OSError as err:
            print(err)
            print('GENERAL ERROR IN: '+ ffile)
            
    #df.loc[pd.to_numeric(df.iloc[:,2],errors='ignore').dropna().index,:].to_csv('JOINED_SURVEYS.csv')
    #df.loc[pd.to_numeric(df.iloc[:,2],errors='ignore').dropna().index,:].
    df = df.dropna(axis = 0, how='all').dropna(axis=1,how='all')
    if SAVEFILE != None:
    #    if path.exists(SAVEFILE+'.JSON'):
    #        df2 = pd.read_cjson(SAVEFILE+'.JSON', ignore)
    #        #df.to_csv(SAVEFILE+'.CSV', mode = 'a', header = False, index=False)
    #        df.to_json(SAVEFILE+'.JSON')
    #    else:
    #        df.to_csv(SAVEFILE+'.CSV', header = True, index = False)
    #        df.to_json(SAVEFILE,+'.JSON')
        df.to_json(SAVEFILE+'.JSON')
    return df


def JoinSurveysInFolder(SAVE = True, FILESTRING = None):
          
    pathname = path.dirname(argv[0])
    adir = path.abspath(pathname)

    df=pd.DataFrame()
    df1 = pd.DataFrame()
          
    if FILESTRING == None:
        JOINEDFILE = 'JOINED_SURVEY_FILE_V2_MERGE'
    else:
        JOINEDFILE = FILESTRING
    
    #for file in listdir(pathname):
    #    if file.lower().endswith(('.json')) and ('surveys' in file.lower()):
    #        if df.empty:
    #            df = pd.read_json(file)
    #        else:
    #            df = pd.concat([df, pd.read_json(file)],axis=0,join='outer',ignore_index=True)
    #            df = df.drop_duplicates()

    PAT = re.compile(r'joined(?:[ -_]*)survey',re.I)

    FLIST=list()
    for file in listdir(adir):
        if file.lower().endswith(('.xls','xlsx','xlsm')):
            FLIST.append(file)

    print(str(len(FLIST))+' FILES CONSIDERED')

    for file in listdir(adir):
        if file.lower().endswith(('.parquet')) and PAT.search(file) and ('abs' not in file.lower()) and ('xyz' not in file.lower())and ('3d' not in file.lower()):
            try:
                print(file)
                if df1.empty:
                    df1 = pd.read_parquet(file)
                else:
                    df1 = pd.concat([df1, pd.read_parquet(file)],axis=0,join='outer',ignore_index=True)
                    df1 = df1.drop_duplicates()
            except:
                print("ERROR IN FILE: " + str(file))
                
    #CLEAN UP API/UWI
    if not df1.empty:
        df1['UWI'] = df1.UWI.apply(lambda x: WELLAPI(x).API2INT(14))
        m1 = df1.UWI.isna()
        df1.loc[m1,'UWI'] = df1.loc[m1,'API'].apply(lambda x: WELLAPI(x).API2INT(14))
   
    if not df1.empty:
        df1 = df1.drop_duplicates()
        FLIST = pd.Series(FLIST)
        mask = FLIST.isin(df1.FILE)
        FLIST = FLIST.loc[~mask]
        FLIST = FLIST.drop_duplicates()
        FLIST = list(FLIST)
    
    #if path.exists(JOINEDFILE+'.PARQUET'):
    #    #df1 = pd.read_json('JOINED_SURVEY_FILE_V2'+'.JSON')
    #    df1 = pd.read_parquet('JOINED_SURVEY_FILE_V2'+'.PARQUET')
    #    df1.to_parquet(JOINEDFILE+'_'+datetime.datetime.now().strftime('%Y%M%d')+'.PARQUET')
    #    df1 = pd.concat([df1,df],axis=0,join='outer',ignore_index=True)
    #    df1 = df1.drop_duplicates()
    #    #df1.to_json(JOINEDFILE+'.JSON')
    #    df1.to_parquet(JOINEDFILE+'.PARQUET')
    #    FLIST = pd.Series(FLIST)
    #    mask = FLIST.isin(df1.FILE)
    #    FLIST = FLIST.loc[~mask]
    #    FLIST = FLIST.drop_duplicates()
    #    FLIST = list(FLIST)

    #initialize multiprocessing
    
    #if path.exists(JOINEDFILE):
    #    PreUsedFiles = pd.read_csv(JOINEDFILE,usecols = ['FILE'],squeeze=True)
    #    PreUsedFiles=PreUsedFiles.unique().tolist()
    #    FLIST2 = [x for x in FLIST if x not in PreUsedFiles]
    #    FLIST = FLIST2
    #    del FLIST2
    chunkmin = 200
    splits = max(floor(len(FLIST)/chunkmin),1)
    processors = multiprocessing.cpu_count()
    # if processor batches are nearly greater than min batch size
    if (len(FLIST)/processors/1.3) > chunkmin:
        splits = processors/1.3
    # if processor batches nearly > min batch size
    if (len(FLIST)/processors) > chunkmin:
        splits = processors
    splits = np.floor(splits)
    processors = min(processors,splits)
    data = np.array_split(FLIST,splits)
    del df
    
    func = partial(Survey_Join, None, ERRORS = False)

    #for f in FLIST:
    #    func(f)

    print(processors)
    if processors > 1:
        print('Concurrent Futures Starting')
        with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
            f = {executor.submit(func, a): a for a in data}
        RESULT=pd.DataFrame()
        for i in f.keys():
            RESULT=pd.concat([RESULT,i.result()],axis=0,join='outer',ignore_index=True)
    else:
        RESULT=Survey_Join(None,FLIST)
        
    if not(df1.empty):
        RESULT = pd.concat([RESULT,df1],axis=0,join='outer',ignore_index=True)

    R = RESULT.copy()

    RESULT = DF_UNSTRING(RESULT)
    RESULT.UWI = RESULT.UWI.apply(lambda x: WELLAPI(x).API2INT(14))
    RESULT = RESULT.drop_duplicates()
    
    if 'API' in RESULT.keys():
        m = RESULT.UWI.isna() * RESULT.API.notnull()
        RESULT.loc[m,'UWI'] = RESULT.loc[m,'API'].apply(lambda x: WELLAPI(x).API2INT(14))

    m = RESULT.UWI.isna()
    RESULT.loc[m,'UWI'] = RESULT.loc[m,'FILE'].apply(APIfromString,args = (True,)).fillna(np.nan)   

    RESULT = DF_UNSTRING(RESULT)
    if 'API' in RESULT.keys():
        RESULT['API'] = RESULT['API'].fillna('')

    if SAVE == True:
        RESULT.to_csv(JOINEDFILE+'_'+datetime.datetime.now().strftime('%Y%M%d')+'.CSV')
        RESULT.to_json(JOINEDFILE+'_'+datetime.datetime.now().strftime('%Y%M%d')+'.JSON')
        RESULT.to_parquet(JOINEDFILE+'_'+datetime.datetime.now().strftime('%Y%M%d')+'.PARQUET')
    
    return(RESULT)

def CO_ABS_LOC(UWIS, SQLDB = 'CO_3_2.1.sqlite'):
    pathname = path.dirname(argv[0])
    adir = path.abspath(pathname)

    if isinstance(UWIS,(str, float, int)):
        UWIS=[UWIS]

    UWI10S = [WELLAPI(X).API2INT(10) for X in UWIS]
    #sqldb = path.join(path.dirname(adir),'CO_3_2.1.sqlite')
          
    with sqlite3.connect(SQLDB) as conn:
        df = pd.read_sql_query('SELECT API,Latitude,Longitude FROM WELL',conn)
    
    df['UWI10'] = df.API.apply(lambda x: WELLAPI(x).API2INT(10)) 
    m = df[['Longitude','Latitude']].dropna().index
    A = convert_XY(df.loc[m,'Longitude'],df.loc[m,'Latitude'], EPSG_OLD=4269, EPSG_NEW=2878)
    df.loc[m,['X_FEET','Y_FEET']] = pd.DataFrame({'X_FEET':A[0],'Y_FEET':A[0]}).values
    
    return(df.loc[m,['UWI10','X_FEET','Y_FEET']].drop_duplicates())
         

def CondenseSurvey(xdf,LIST_IN):
    INC_LIMIT = 85
    # if 1==1:
    if isinstance(LIST_IN,(pd.Series,np.ndarray)):
        UWIs=list(LIST_IN)
    if isinstance(LIST_IN,(str,int,np.integer)):
        UWIs=[LIST_IN]
    if isinstance(LIST_IN,list):
        UWIs=LIST_IN
        
    OUTPUT = dict()
    
    #UWICOL = xdf.keys().get_loc(xdf.iloc[0,xdf.keys().str.contains('.*UWI.*|.*API.*', regex=True, case=False,na=False)].keys()[0])
    UWIKEYS = xdf.iloc[0,xdf.keys().str.contains('.*UWI.*|.*API.*', regex=True, case=False,na=False)].keys()

    #xdf['UWI10'] = xdf[UWIKEYS].apply(lambda x: x.apply(API2INT,length = 14), axis =1).max(axis=1)
    
    if 'UWI10' in UWIKEYS:
        UWIKEY = 'UWI10'
    else:
        xdf['UWI10'] = None
        for k in UWIKEYS:
            #xdf['I'] = xdf[k].apply(API2INT)
            xdf['I'] = xdf[k].apply(lambda x: WELLAPI(x).API2INT(10))
            xdf['UWI10'] = xdf['I','UWI10'].max(axis=1)
        xdf.drop('I',inplace=True)
        UWIKEY = 'UWI10'

    UWIKEY = 'UWI10'
    UWICOL = xdf.keys().get_loc(UWIKEY)
    #UWIKEY = xdf.keys()[UWICOL]

    # slice dataframe to subset for function
    xdf = xdf.loc[xdf[UWIKEY].isin(UWIs)].copy(deep=True)

    # format UWI's
    xdf[UWIKEY] = xdf[UWIKEY].apply(lambda x: WELLAPI(x).API2INT(10))
    UWIs = [WELLAPI(x).API2INT(10) for x in UWIs]
    UWIs = list(set(UWIs))

    # Add date column
    xdf['FileDate'] = xdf.FILE.str.extract(r'SURVEYDATA_([0-9]{4}_[0-9]{2}_[0-9]{2}).*\.x*')
    xdf['FileDate'] = pd.to_datetime(xdf['FileDate'],errors = 'coerce', format = '%Y_%m_%d')

    # assign date to NaT values
    xdf.loc[xdf['FileDate'].isna(),'FileDate'] = datetime.datetime(1900,1,1)
    
    #SKEYS = list(SurveyCols(xdf).keys())
    #xdf = xdf.rename(columns = SurveyCols(xdf.head(5)))

    SKEYS = list(SurveyCols(xdf).keys())
    xdf = xdf.loc[xdf[SKEYS].drop_duplicates().index,:]

    # Make key columns numeric
    for k in SurveyCols(xdf):
        xdf[k] = pd.to_numeric(xdf[k],errors = 'coerce')

    # Add Azi decimal column
    xdf.loc[:,'AZI_DEC'] = xdf.loc[:,'AZI'] - np.floor(xdf.loc[:,'AZI'])

    tot = len(UWIs)
    ct = 0
    for UWI in UWIs:
        ct +=1
        if floor(ct/10)==ct/10:
            print(ct,'/',tot,': ',UWI)
        # while df.loc[df.groupby('FILE').MD.apply(lambda x: x-np.floor(x))==0,:]  
        # filter to UWI of interest
        xxdf = xdf.copy(deep=True)
        xxdf = xxdf.loc[xxdf[UWIKEY] == UWI,:]
        ftest = list(xxdf.FILE.unique())
        
        #print(ftest)
        if xxdf.FILE.unique().shape[0] == 1:
            file = xxdf.FILE.unique()[0]
            
        while xxdf.FILE.unique().shape[0] > 1:
            # drop duplicates   if 1==1:
            #xxdf = xxdf.loc[xxdf[SKEYS].drop_duplicates().index,:]

            # if MD is all integers then drop
            # df.groupby(FILE).MD.apply(lambda x:x-np.floor(x))
            # sdf = sdf.loc[sdf.groupby('FILE').MD.apply(lambda x:x-np.floor(x))>0]
            # HZ AZI filter
            if xxdf.loc[xxdf.INC > INC_LIMIT,:].shape[0]>5:
                #xxdf.loc[:,'AZI_DEC'] = xxdf.loc[:,'AZI'] - np.floor(xxdf.loc[:,'AZI'])

                #sdf['MD_DEC'] = sdf['MD'] - np.floor(sdf['MD'])
                ftest = xxdf.loc[xxdf.INC > INC_LIMIT,:].groupby('FILE')['AZI_DEC'].std()
                ftest = pd.DataFrame(ftest[ftest>0.1])
                #files = ftest.loc[ftest>0.1].index.to_list()
                if ftest.empty:
                    ftest = pd.DataFrame(xxdf.groupby('FILE')['AZI_DEC'].std())
                xxdf = xxdf.drop(['AZI_DEC'],axis=1)
            else:
                ftest = xxdf.groupby('FILE')['AZI'].nunique()/xxdf.groupby('FILE')['AZI'].count()
                ftest = pd.DataFrame(ftest.loc[ftest>0.4])
                if ftest.empty:
                    xxdf.loc[:,'AZI_DEC'] = xxdf.loc[:,'AZI'] - np.floor(xxdf.loc[:,'AZI'])
                    ftest = pd.DataFrame(xxdf.groupby('FILE')['AZI_DEC'].std())
                    xxdf = xxdf.drop(['AZI_DEC'],axis=1)

            if ftest.shape[0] > 1:
                for f in ftest.index:
                    ftest.loc[f,'DATE'] = datetime.datetime(1900,1,1)
                    try:
                        date = re.findall(r'SURVEYDATA_(.*)_[0-9]*.x*',f)[0]
                        ftest.loc[f,'DATE'] = datetime.datetime.strptime(date, '%Y_%m_%d')
                    except: pass
                # set column equal to index
                ftest['FILEUWI']=ftest.index
                # get UWI from filename

                #ftest['FILEUWI']=ftest.FILEUWI.apply(APIfromFilename).apply(lambda x: WELLAPI(x).API2INT(10))
                UWImask = (ftest.FILEUWI.apply(APIfromFilename).apply(lambda x:WELLAPI(x).API2INT(10)) )== WELLAPI(UWI).API2INT(10)
                Datemask1 = (ftest.DATE==ftest.loc[UWImask,'DATE'].max())
                Datemask2 = (ftest.DATE==ftest.DATE.max())
                #f1 = ftest.loc[UWImask & (ftest.DATE==ftest.DATE.max())].index
                #f2 = ftest.loc[ftest.DATE==ftest.DATE.max()].index
                if sum(Datemask1) > 20:
                    ftest = ftest.loc[Datemask1]
                elif sum(Datemask2) > 20:
                    ftest = ftest.loc[Datemask2]
            file = ftest.index.values[0]
            xxdf = xxdf.loc[xxdf.FILE == file]

        OUTPUT.update({UWI:file})
        
        #OUTPUT = pd.merge(OUTPUT,df,how='outer',on=None, left_index=False, right_index=False)
        #OUTPUT = pd.merge(OUTPUT,xxdf,how='outer',on=None, left_index=False, right_index=False)
        
    return OUTPUT

def Condense_Surveys(xdf):
    # xdf = pd.read_csv(FILE)
    # RESULT = pd.DataFrame(columns = xdf.columns.to_list())
    # if 1==1:
    #UWICOL = xdf.keys().get_loc(xdf.iloc[0,xdf.keys().str.contains('.*UWI.*|.*API.*', regex=True, case=False,na=False)].keys()[0])

    xdf.keys().get_loc(xdf.iloc[0,xdf.keys().str.contains('.*UWI.*|.*API.*', regex=True, case=False,na=False)].keys()[0])
    
    UWIKEYS = list(xdf.keys()[xdf.keys().str.contains('.*UWI.*|.*API.*', regex=True, case=False,na=False)])
    
    #xdf['UWI10'] = xdf[UWIKEYS].max(axis=1).apply(API2INT)
    xdf['UWI10'] = xdf[UWIKEYS].max(axis=1).apply(lambda x: WELLAPI(x).API2INT(10))
    #UWIlist = list(xdf.iloc[:,UWICOL].unique())
    UWICOL = xdf.keys().get_loc('UWI10')
    UWIKEY = 'UWI10'
    UWIlist = list(xdf.UWI10.unique())
    

    chunkmin = 100
    chunkmax = 1000
    batchmin = ceil(len(UWIlist)/chunkmax)
    batchmax = ceil(len(UWIlist)/chunkmin)

    processors = multiprocessing.cpu_count()
    processors = min(processors,batchmin)

    batches = max(min(batchmax,processors),batchmin)
    
    data = np.array_split(UWIlist,batches)

    func = partial(CondenseSurvey, xdf)
    print ('condensing surveys')

    if processors > 1:
        with cfutures.ThreadPoolExecutor(max_workers = processors) as executor:
            f = {executor.submit(func, a): a for a in data}
        #RESULT=pd.DataFrame()
        RESULT = dict()
        print('merging condense sessions')
        for i in f.keys():
            #RESULT=pd.concat([RESULT,i.result()],axis=0,join='outer',ignore_index=True)
            RESULT.update(i.result())
            print(len(RESULT.keys()))
    else:
        RESULT=CondenseSurvey(xdf,UWIlist)
    #RESULT.to_csv(OUTFILE,index=False)
    return RESULT
          
# Define function for nearest neighbors
def XYZSpacing(xxUWI10, xxdf, df_UWI, DATELIMIT, SAVE = False):
    INC_LIMIT = 87
          
    # condensed SURVEYS in xxdf
    # WELL DATA in df_UWI
    # xxUWI10 is list of UWI's to calc
    # DATELIMIT is number of days well 2 can be completed after well 1 and still treated as neighbor
          
    # if True:
    xxdf = xxdf.copy(deep=True)

    df_UWI = df_UWI.copy(deep=True)
    #if 1==1:
    #[xUWI10,args]=arg
    #[xdf,df_UWI,DATELIMIT]=args
    xxUWI10=convert_to_list(xxUWI10)
    
    #p = psutil.Process(os.getpid())
    #p.nice(psutil.HIGH_PRIORITY_CLASS)

    ix = -1

    xxUWI10=pd.DataFrame(xxUWI10).iloc[:,0].unique().tolist()
    
    col_type = {'UWI10':int(),
                'LatLen':int(),
                'MeanTVD':int(),
                'MeanX':float(),
                'MeanY':float(),
                'MAX_MD': float(),
                'MeanAZI': float(),
                'MeanAZI180': float(),
                'overlap1':float(),
                'dz1':float(),
                'dxy1':float(),
                'UWI1':int(),
                'Days1':int(),
                'overlap2':float(),
                'dz2':float(),
                'dxy2':float(),
                'UWI2':int(),
                'Days2':int(),
                'overlap3':float(),
                'dz3':float(),
                'dxy3':float(),
                'UWI3':int(),
                'Days3':int(),
                'overlap4':float(),
                'dz4':float(),
                'dxy4':float(),
                'UWI4':int(),
                'Days4':int(),
                'overlap5':float(),
                'dz5':float(),
                'dxy5':float(),
                'UWI5':int(),
                'Days5':int(),
                'XYZFILE':str()}
    
    OUTPUT = pd.DataFrame(col_type,index=[])

    COMPDATES = df_UWI.iloc[0,df_UWI.keys().str.contains('.*PROD.*DATE|.*JOB.*DATE.*|STIM.*DATE[^0-9].*|.*COMP.*DATE.*|.*FIRST.*(PROD|DATE).*|.*SPUD.*DATE.*', regex=True, case=False,na=False)].keys()
    df_UWI[COMPDATES] = DF_UNSTRING(df_UWI[COMPDATES])
    df_UWI['MAX_COMPLETION_DATE'] = df_UWI[COMPDATES].max(axis=1)

    # INILL WELLS WITH MISSING DATES
    xxdf.UWI10.isin(df_UWI['UWI10'])
    
    df_UWI['MAX_COMPLETION_DATE'].fillna(datetime.datetime.now(), inplace = True)
    COMPDATEdfd = df_UWI.keys().get_loc('MAX_COMPLETION_DATE')

    xxdf.rename(columns = SurveyCols(xxdf.head(5)), inplace = True)
    
    # MAKE KEY COLUMNS NUMERIC
    for k in SurveyCols(xxdf):
        xxdf[k]=pd.to_numeric(xxdf[k],errors='coerce')

    #filter to lateral 
    xxdf = xxdf.loc[xxdf.INC> INC_LIMIT]

    try:
        SCOLS = SurveyCols(xxdf.head(5))
        SCOLS.pop(list(SurveyCols(xxdf.head(5)).keys())[5])
        SCOLS.pop(list(SurveyCols(xxdf.head(5)).keys())[4])
    except: 
        pass
    XPATH_NAME = (xxdf[GetKey(xxdf,'EAST|XFEET')].abs().std()/xxdf[GetKey(xxdf,'EAST')].abs().mean()).sort_values(ascending=True).keys()[0]
    YPATH_NAME = (xxdf[GetKey(xxdf,'NORTH|YFEET')].abs().std()/xxdf[GetKey(xxdf,'NORTH')].abs().mean()).sort_values(ascending=True).keys()[0]
    XPATH = xxdf.keys().get_loc(XPATH_NAME)
    YPATH = xxdf.keys().get_loc(YPATH_NAME)

    if 'UWI10' in xxdf.keys():
        UWICOL = xxdf.keys().get_loc('UWI10')
    else:
        UWICOL      = xxdf.keys().get_loc(xxdf.iloc[0,xxdf.keys().str.contains('.*UWI.*', regex=True, case=False,na=False)].keys()[0])
    #XPATH     = xxdf.keys().get_loc(list(SCOLS)[5])
    #YPATH     = xxdf.keys().get_loc(list(SCOLS)[4])
    #XPATH       = xxdf.keys().get_loc(list(SCOLS)[5])
    #YPATH       = xxdf.keys().get_loc(list(SCOLS)[4])       
    TVD         = xxdf.keys().get_loc(list(SCOLS)[3])
    AZI         = xxdf.keys().get_loc(list(SCOLS)[2])
    DIP         = xxdf.keys().get_loc(list(SCOLS)[1])
    MD          = xxdf.keys().get_loc(list(SCOLS)[0])

    #if 'NORTH_Y_XX' in list(SCOLS):
    #    XPATH       = xxdf.keys().get_loc(SCOLS['EAST_X_XX'])
    #    YPATH       = xxdf.keys().get_loc(SCOLS['NORTH_Y_XX'])
    
    #XPATH_NAME  = xxdf.keys()[XPATH] #list(SurveyCols(xxdf.head(5)))[5]
    #YPATH_NAME  = xxdf.keys()[YPATH] #list(SurveyCols(xxdf.head(5)))[4]
    
    MD_NAME = xxdf.keys()[MD]

    xxdf['UWI10'] = xxdf.iloc[:,UWICOL].apply(lambda x: WELLAPI(x).API2INT(10))
    
    for ix,xUWI10 in enumerate(xxUWI10):
        xUWI10=WELLAPI(xUWI10).API2INT(10)
              
        xdf = xxdf.copy(deep=True)
        xdf = xdf.loc[xdf[list(SurveyCols(xdf).keys())].dropna().index,:]
        #print(str(xxUWI10.index(xUWI10)),' / ',str(len(xxUWI10)),' ')
        
        if ix/10 == floor(ix/10):
            print(str(ix) + '/' + str(len(xxUWI10)))
                  
        OUTPUT=pd.concat([OUTPUT,pd.Series(name=ix,dtype='int64')], axis= 0, ignore_index = True)
        
        # Check for lateral survey points for reference well
        if xdf.loc[(xdf['UWI10']==xUWI10),:].shape[0]<=5:
            continue
            
        xFILE = xdf.loc[xdf.UWI10==xUWI10,'FILE'].values[0]        
        
        # PCA is 2 vector components
        pca = PCA(n_components=2)
        
        # add comp date filter at step 1
        try: 
            datecondition=(df_UWI.loc[df_UWI['UWI10']==xUWI10][df_UWI.keys()[COMPDATEdfd]]+datetime.timedelta(days =DATELIMIT)).values[0]
        except:
            datecondition = datetime.datetime.now() 

        UWI10list=df_UWI[(df_UWI[df_UWI.keys()[COMPDATEdfd]])<=datecondition].UWI10
        # filter on dates
        xdf=xdf[xdf.UWI10.isin(UWI10list)]
        #isolate reference well
        refXYZ=xdf[xdf.keys()[[UWICOL,XPATH,YPATH,TVD,MD]]][(xdf.UWI10==xUWI10)]
        
        if refXYZ.shape[0]<5:
            continue
            
        # CALC DESCRIPTIVE PARAMETERS
        OUTPUT.loc[ix,'UWI10']     = xUWI10
        OUTPUT.loc[ix,['LatLen']]  = (refXYZ[[XPATH_NAME,YPATH_NAME]].iloc[[0,-1],-2:].diff(axis=0).dropna()**2).sum().sum()**0.5      
        OUTPUT.loc[ix,['MeanTVD']] = statistics.mean(refXYZ[SCOLS['TVD']])
        OUTPUT.loc[ix,['MeanX']]   = statistics.mean(refXYZ[XPATH_NAME])
        OUTPUT.loc[ix,['MeanY']]   = statistics.mean(refXYZ[YPATH_NAME])
        OUTPUT.loc[ix,'MeanAZI']   = (circmean(refXYZ.iloc[:,[AZI]]*pi/180) * 180/pi) % 180
        OUTPUT.loc[ix,'MAX_MD']    = max(refXYZ[MD_NAME].dropna())
        OUTPUT.loc[ix,'XYZFILE']   = xFILE


        #reference well TVD approximation
        # if 1==1:
        #refTVD = gmean(abs(xdf.iloc[:,TVD][xdf.UWI10==xUWI10]))*np.sign(statistics.mean(xdf.iloc[:,TVD][xdf.UWI10==xUWI10]))
        #refTVD = statistics.mean(xdf.iloc[:,TVD][xdf.UWI10==xUWI10])
        refTVD = statistics.mean(refXYZ.TVD)
        #remove self well to prevent 0' offsets
        xdf=xdf[xdf['UWI10']!=xUWI10]
        
        if xdf.shape[0]>5:
            # get projection vector from survey points to well of interest
            #for pt in set(UWI10):   
            #    fmin_cobyla(objective, x0=[0.5,0.5], cons=[c1])
            pca.fit(refXYZ.iloc[:,[1,2]])
            X_fit = pca.transform(refXYZ[xdf.keys()[[XPATH,YPATH]]])
            RefXMin=min(X_fit[:,0])
            RefXMax=max(X_fit[:,0])
            XY_fit = pca.transform(xdf[xdf.keys()[[XPATH,YPATH]]])
            xdf['Xfit']=XY_fit[:,0]
            xdf['Yfit']=XY_fit[:,1]
            
            # clip to overlapping wells
            m = xdf['Xfit'] >= RefXMin
            xdf = xdf.loc[m,:]
            m = xdf['Xfit'] <= RefXMax
            xdf = xdf.loc[m,:]

            # clip to 2000' overlap
            CLIPLIMIT = 2000
            #XY_fit = XY_fit[(XY_fit[:,0]<max(X_fit[:,0])) & (XY_fit[:,0]>min(X_fit[:,0]))]
            m = (xdf.loc[(xdf.Xfit<max(X_fit[:,0])) & (xdf.Xfit>min(X_fit[:,0])),['UWI10','Xfit']].groupby(by='UWI10').max()-xdf.loc[(xdf.Xfit<max(X_fit[:,0])) & (xdf.Xfit>min(X_fit[:,0])),['UWI10','Xfit']].groupby(by='UWI10').min())>=CLIPLIMIT
            m = list(m.loc[m.iloc[:,0].values].index )            
            xdf = xdf.loc[xdf.UWI10.isin(m)]
            
            # clip to 10000' offset
            xdf=xdf[abs(xdf.Yfit)<10000]
            
            # clip to overlapping segments
            #xdf=xdf[(xdf.Xfit>=RefXMin)&(xdf.Xfit<=RefXMax)]

            # ref completion date
            refdate = (df_UWI[df_UWI['UWI10']==xUWI10][df_UWI.keys()[COMPDATEdfd]]).values[0]
                                           
            df_calc = pd.DataFrame(columns = ['UWI10','overlap','dxy','dz','abs_dxy','DAYS'])
            j=-1

            # CULL TO NEARBY WELLS!!
            LOCAL_LIMIT = 505
            LOCAL_UWI = 1
            
            for j,well in enumerate(set(xdf.UWI10)):
                overlap = max(xdf.Xfit[xdf.UWI10==well])-min(xdf.Xfit[xdf.UWI10==well])
                gmeandistance = gmean(abs(xdf.Yfit[xdf.UWI10==well]))*np.sign(statistics.mean(xdf.Yfit[xdf.UWI10==well]))
                #gmeandepth = gmean(abs(xdf.iloc[:,TVD][xdf.UWI10==well]))*np.sign(statistics.mean(xdf.iloc[:,TVD][xdf.UWI10==well]))-refTVD
                meandepth = statistics.mean(xdf.iloc[:,TVD][(xdf.UWI10==well)])
                try:
                    deltadays =  np.timedelta64(refdate-(df_UWI[df_UWI['UWI10']==well][df_UWI.keys()[COMPDATEdfd]]).values[0],'D').astype(float)
                except: 
                    deltadays = None
                
                df_calc.loc[j,'UWI10']=well
                df_calc.loc[j,'overlap']=overlap
                df_calc.loc[j,'dxy']=gmeandistance
                df_calc.loc[j,'abs_dxy']=abs(gmeandistance)
                df_calc.loc[j,'dz'] =meandepth - refTVD
                df_calc.loc[j,'DAYS']=deltadays
                
            df_calc[df_calc.overlap>=2000]
            sort_list=df_calc.sort_values(by=['abs_dxy']).UWI10

            #add closest 5 wells passing conditions in order
            for j in range(0,min(5,len(sort_list))):
                ol='overlap'+str(j+1)
                dz='dz'+str(j+1)
                dxy='dxy'+str(j+1)
                uwi='UWI'+str(j+1)
                days = 'Days'+str(j+1)
                OUTPUT.loc[ix,[ol,dz,dxy,uwi,days]]=df_calc[df_calc.UWI10==sort_list.iloc[j]][['overlap','dz','dxy','UWI10','DAYS']].values[0]

    if SAVE:
        outfile = 'XYZ_'+str(int(xxUWI10[0]))+'_'+str(int(xxUWI10[-1]))
        adir = getcwd()
        if not path.exists(path.join(adir,'XYZ')):
            mkdir(path.join(adir,'XYZ'))
        #try:
        #    OUTPUT = DF_UNSTRING(OUTPUT)
        #except:
        #    pass
        #try:
        #    OUTPUT.to_parquet(path.join(adir,'XYZ',outfile+'.parquet'))
        #except:
        #    pass
        try:
            OUTPUT.to_csv(path.join(adir,'XYZ',outfile+'.csv'))
        except:
            pass

    return OUTPUT
                   
def MIN_CURVATURE(df_survey):
    d = SurveyCols(df_survey,False)
    keys = list(d.values())

    df = df_survey[keys]
    df.rename(columns = d, inplace=True)

    df['INC_RAD'] = df[keys[1]] * pi/180
    df['AZI_RAD'] = df[keys[2]] * pi/180

    df[['TVD','NORTH_dY','EAST_dX']] = np.nan

    MD = df.keys().get_loc(keys[0])
    INC = df.keys().get_loc('INC_RAD')
    AZI = df.keys().get_loc('AZI_RAD')

    for i in np.arange(0,df.shape[0]):
        idx0 = df.index[i-1]
        idx1 = df.index[i]

        if (i==0):
            df.loc[idx1, ['TVD','NORTH_dY','EAST_dX']] = 0
            continue
        if (df.iloc[i,:][keys[1:]].sum() == 0):
            df.loc[idx1, ['NORTH_dY','EAST_dX']] = 0
            df.loc[idx1, 'TVD']  = df.loc[idx0, 'TVD'] + df.loc[idx1, keys[0]] - df.loc[idx0, keys[0]]
            continue

        BETA = acos( cos(df.iloc[i,INC] - df.iloc[i-1,INC] ) - sin(df.iloc[i-1,INC])*sin(df.iloc[i,INC])*(1-cos(df.iloc[i,AZI] - df.iloc[i-1,AZI])))
        if BETA == 0:
            RF = 1
        else:
            RF = 2/BETA * tan(BETA/2)
        NORTH = (df.iloc[i,MD] - df.iloc[i-1,MD])/2 * ( sin(df.iloc[i-1,INC])*cos(df.iloc[i-1,AZI]) + sin(df.iloc[i,INC])*cos(df.iloc[i,AZI])) * RF + df.iloc[i-1,:]['NORTH_dY']
        EAST = (df.iloc[i,MD] - df.iloc[i-1,MD])/2 * ( sin(df.iloc[i-1,INC])*sin(df.iloc[i-1,AZI]) + sin(df.iloc[i,INC])*sin(df.iloc[i,AZI])) * RF + df.iloc[i-1,:]['EAST_dX']
        TVD = (df.iloc[i,MD] - df.iloc[i-1,MD])/2 * ( cos(df.iloc[i-1,INC]) + cos(df.iloc[i,INC]) ) * RF + df.iloc[i-1,:]['TVD']

        df.loc[idx1, ['TVD','NORTH_dY','EAST_dX']] = [TVD,NORTH,EAST]
    df.drop(['INC_RAD','AZI_RAD'], axis=1, inplace = True) 
    return df      

def LeftRightSpacing(df_in):
    
    for i in [1,2,3,4,5]:
        xykey = GetKey(df_in,f'dxy{i}')[0]
        zkey = GetKey(df_in,f'dz{i}')[0]
        df_in[f'dxyz{i}'] = (df_in[xykey]**2+df_in[zkey]**2)**0.5


    xy_keys  = GetKey(df_in,'dxy\d')
    z_keys   = GetKey(df_in,'dz\d')
    xyz_keys = GetKey(df_in,'dxyz\d')

    xy_keys.sort()
    z_keys.sort()

    df_in[['left_dxy','left_dz','right_dxy','right_dz']]=np.nan

    for idx in df_in.index:
        xyz_order  = df_in.loc[idx,xyz_keys].sort_values(ascending =True).index.tolist()
        xyz_order  = [xyz_keys.index(x) for x in xyz_order]

        left_cols  = (df_in.loc[idx,xy_keys]  < 0).values.tolist()
        right_cols = (df_in.loc[idx,xy_keys] > 0).values.tolist()

        for i in xyz_order:
            if left_cols[i]:
                df_in.loc[idx,'left_dxy'] = df_in.loc[idx,xy_keys[i]]
                df_in.loc[idx,'left_dz']  = df_in.loc[idx,z_keys[i]]
                break
        for i in xyz_order:
            if right_cols[i]:
                df_in.loc[idx,'right_dxy'] = df_in.loc[idx,xy_keys[i]]
                df_in.loc[idx,'right_dz']  = df_in.loc[idx,z_keys[i]]
                break
    return("finished")
