# update base files
from ._FUNCS_ import *

__all__ = ['TEMP_SUMMARY_LAS',
           'HTMLtoTXT',
           'ZIPtoTXT',
           'DOCtoTXT',
           'LAS_TEXTABORTED_FIX',
           'LASREPAIR',
           'List_LAS_Files_In_Folder',
           'FIND_SP_KEY',
           'LOG_DETREND',
           'ARRAY_CHANGEPOINTS',
           'Init_Futures',
           'SP_WORKFLOW']

def TEMP_SUMMARY_LAS(_LASFILES_):
    SUMMARY = pd.DataFrame(columns = ['API','MNEMONIC','VALUE','DESCR','DEPTH','DATE','FILE'])
    LASDATA = False

    for f in _LASFILES_:
        print(f)
        try:
            las = lasio.read(f)
            LASDATA = True
        except:
            try:
                las = lasio.read(f, ignore_data=True)
                LASDATA = False
            except:
                print('Error reading '+f)
                continue

        LOGDATE = np.nan
        LOGDATE2 = np.nan

        for i in [i for i in las.header.keys() if 'CURVE' not in i.upper()]:
            for j in las.header[i]:
                if isinstance(j,str):
                    continue
                if 'DATE' in j.descr.upper():
                    try:
                        LOGDATE2 = dateutil.parser.parse(j.value)
                        if isinstance(LOGDATE,datetime.datetime):
                            LOGDATE = max(LOGDATE,LOGDATE2)
                        else:
                            LOGDATE = LOGDATE2
                    except:
                        pass

        try:
            UWI = str(las.well["UWI"].value)
            UWI = WELLAPI(UWI).API2INT(10)
        except:
            try:
                UWI = str(las.well["API"].value)
                UWI = WELLAPI(UWI).API2INT(10)
            except:
                try:
                    UWI = re.search(r'[0-9]{10,}',f)[0]
                    UWI = WELLAPI(UWI).API2INT(10)
                except:
                    pass
        if UWI == '':
            print('NO UWI FOR '+f)
            continue
            
        UWI = str(UWI).zfill(10)
        MAXDEPTH = las.header['Well']['STOP'].value

        #MAXDEPTH = las.df().dropna(how='all', axis=0).index.max()

        # IGNORE: RMC, RMF

        if LASDATA:
            for c in las.curves:
                if ('TEMP' in c.descr.upper()) :
                    # needs alias list of preferred temp logs      
                    TVAL = las[c.mnemonic][-10:].mean()
                    DEPTH = las.df()[c.mnemonic].dropna().index.max()
                    SUMMARY.loc[len(SUMMARY.index)] = [UWI, c.mnemonic, TVAL, c.descr, DEPTH, LOGDATE,f]
                    
        for i in [i for i in las.header.keys() if 'CURVE' not in i.upper()]:
            # look for what values match true BHT values for wells with them
            if isinstance(las.header[i],str):
                continue
            for j in las.header[i]:
                if 'RMF' in j.mnemonic.upper() or 'RMC' in j.mnemonic.upper() or '@' in str(j.value):
                    continue
                if (j.value == '') and not isinstance(j,lasio.las_items.CurveItem):
                    continue
                if 'TEMP' in j.descr.upper():
                    SUMMARY.loc[len(SUMMARY.index)] = [UWI, j.mnemonic, j.value, j.descr, MAXDEPTH, LOGDATE, f]
                elif 'BHT' in j.mnemonic.upper():
                    SUMMARY.loc[len(SUMMARY.index)] = [UWI, j.mnemonic, j.value, j.descr, MAXDEPTH, LOGDATE, f]
                elif ('DEGF' in j.unit.upper()) & ('OHM' not in j.unit.upper()):
                    SUMMARY.loc[len(SUMMARY.index)] = [UWI, j.mnemonic, j.value, j.descr, MAXDEPTH, LOGDATE,f]
    return(SUMMARY)


    
def HTMLtoTXT(file,TYPECHECK=True):
    if TYPECHECK:
        if not FILETYPETEST(file,'HTML'):
            print('ERROR: Are you sure ' + str(file)+' is HTML?')
            return
    file2 = file.split('.')[0]+'.html'
    shutil.move(file,file2)
    with open(file2,'r') as F2:
        Fcontent = F2.read()
        soup = BS(index, 'lxml')
        TEXT = soup.get_text('\n')
        TEXT = TEXT.strip()
    with open(file,'w') as F1:
        F1.write(TEXT)
    return None
    
def ZIPtoTXT(file,TYPECHECK=True):
    if TYPECHECK:
        if not FILETYPETEST(file,'ZIP'):
            raise NameError('ERROR: Is ' + str(file)+' a ZIP file?')
            return
    usepath = path.dirname(f)
    if usepath == '':
        usepath=None
    file2 = file.split('.')[0]+'.zip'
    shutil.move(file,file2)
    with ZipFile(file2) as ZF:
        for z in [z for z in ZF.namelist() if z.upper().endswith('LAS')]:
            if path.exists(file):
                file = file.split('.')[0]+'_1'+'.las'
            ZF.extract(z,usepath)
            if usepath == None:
                shutil.move(z,file)
            else:
                shutil.move(path.join(usepath,z),file)    
    return None

def DOCtoTXT(file,TYPECHECK=True):
    if TYPECHECK:
        if not FILETYPETEST(file,'Microsoft Word'):
            raise NameError('ERROR: Are you sure ' + str(file)+' is MS DOC?')
            return
       
    file2 = file.split('.')[0]+'.html'
    rename(file,file2)

    with open(file2,'r') as F2:
        Fcontent = F2.read()
        soup = BS(index, 'lxml')
        TEXT = soup.get_text('\n')
        TEXT = TEXT.strip()
    with open(file,'w') as F1:
        F1.write(TEXT)
    return None

def LAS_TEXTABORTED_FIX(FILESIN):
    if isinstance(FILESIN,str):
        FILESIN = list(FILESIN)
    elif not isinstance(FILESIN,list):
        FILESIN = list(FILESIN)
    for FILEIN in FILESIN:
        with open (FILEIN,'rb+') as FILE:
            lines = FILE.readlines()
            ENDLINE = -1
            pattern = re.compile(b'Thread was being aborted',re.I)
            for i,l in enumerate(lines):
                for match in re.finditer(pattern,l):
                    ENDLINE = max(i,ENDLINE)
            if ENDLINE > -1:
                a = FILE.seek(0)
                a = FILE.truncate()
                a = FILE.writelines(lines[:ENDLINE])
    return None

def LASREPAIR(FILES):
    if isinstance(FILES,str):
        FILES = [FILES]
    if not isinstance(FILES,list):
        FILES = list(FILES)    
    LASACTIONS = {'TROFF':None,
    'UTF-16':None,
    'UTF-8':None,
    'ALGOL':None,
    'ASCII':None,
     'Rich Text':None,
     'data':None,
    'Zip':ZIPtoTXT,
    'HTML':HTMLtoTXT,
    'Microsoft Word':DOCtoTXT}
    _df = pd.DataFrame()
    _df['FILES'] = FILES
    _df['TYPES'] = FTYPE(FILES)
    print('end type')
    _df['SIMPLETYPE'] = _df['TYPES'].apply(lambda x: x.split(',')[0])
    for k in [k for k in LASACTIONS.keys() if LASACTIONS[k]!=None]:
        print(k)
        m = _df.index[_df.SIMPLETYPE.str.upper().str.contains(k.upper())]
        if len(m)>0 :
            LASACTIONS[k](_df.loc[m,'FILES'].tolist())
    LAS_TEXTABORTED_FIX(FILES)
    return None


def List_LAS_Files_In_Folder():
    pathname = path.dirname(argv[0])
    adir = path.abspath(pathname)
    FILES = [f for f in listdir(adir) if '.LAS' in f.upper()]
    return FILES

def FIND_SP_KEY(LAS):
    SP_KEYS = [k.mnemonic for k in LAS.curves if 'SP' in k.mnemonic.upper()]
    SP_KEYS = SP_KEYS + [k.mnemonic for k in LAS.curves if 'SP' in k.descr.upper()]
    SP_KEYS = list(set(SP_KEYS))
    
    if len(SP_KEYS)==1:
        KEY = SP_KEYS[0]
    else:
        KEY = LAS.df()[SP_KEYS].apply(pd.Series.nunique, axis= 0).sort_values(ascending=False).keys()[0]
        print(SP_KEYS+': Assumed '+KEY)
        
    return KEY

def LOG_DETREND(LOG,KEY):
    if isinstance(LOG, lasio.las.LASFile):
        df1 = LOG.df()
    elif isinstance(LOG, (pd.DataFrame,np.ndarray)):
        df1 = LOG.copy()
    else:
        raise Exception('Log is not an array or lasio type')

    IDX_KEY = df1.index.name
    df1.reset_index(drop=False, inplace = True)
    df1.sort_values(by ='DEPT',inplace=True)
    NEWKEY = KEY+'_DETREND'
    m = df1[KEY].dropna().index 
    df1[NEWKEY] = np.nan
    df1.loc[m,NEWKEY]   = signal.detrend(df1.loc[m,KEY])
    df1.set_index(keys = IDX_KEY,drop=True, inplace = True)
    df1.sort_index(axis=0, inplace= True)
    return(df1[NEWKEY])

def ARRAY_CHANGEPOINTS(ARRAY, COUNT):
    if not isinstance(ARRAY, np.ndarray):
        raise Exception('Data is not an array')
    algo = rpt.Dynp(model="l2").fit(ARRAY)
    RESULT = algo.predict(n_bkps=COUNT)
    return(RESULT)

def Init_Futures(APPLY_DATA = None, MAX_SIZE = 5000, MIN_SIZE = 10):
    processors = max(1,multiprocessing.cpu_count())
    chunksize = max(MIN_SIZE,min(MAX_SIZE,max(1,int(len(APPLY_DATA)/(processors*2)))))
    batch = max(1,int(len(APPLY_DATA)/chunksize))
    processors = min(processors,batch)
    SPLIT_DATA = np.array_split(APPLY_DATA,batch)
    return(processors, SPLIT_DATA)

def SP_WORKFLOW(LASFILES,OUTFOLDER = 'SP_OUT'):
    pathname =path.dirname(argv[0])
    adir = path.abspath(pathname)
    OUTFOLDER = path.join(adir,OUTFOLDER)
    if not path.exists(OUTFOLDER):
        mkdir(OUTFOLDER)
        
    if isinstance(LASFILES,str):
        LASFILES = [LASFILES]
    for F in LASFILES:
        las = lasio.read(F)
        df = las.df().copy()
        
        KKEY = FIND_SP_KEY(las)
        df['SP_DETREND'] = LOG_DETREND(df,KKEY)
        
        # Ruptures changepoint detection
        breakpoints = ARRAY_CHANGEPOINTS(df[KKEY].to_numpy(),10)
        
        breakpoints = [0] + breakpoints

        if not 'DEPT' in df.keys():
            IDX_KEY = df.index.name
            df[IDX_KEY] = df.index
            
        df['BreakPoints'] = 0
        for i in np.arange(1,len(breakpoints)):
            last_break = breakpoints[i-1]
            this_break = breakpoints[i]
            
            #handle break at last point
            this_break = min(this_break,df.shape[0]-1)

            last_break = df.index[last_break]
            this_break = df.index[this_break]
            
            depth0 = df.loc[last_break, 'DEPT']
            depth1 = df.loc[this_break,'DEPT']

            df.loc[this_break:,'BreakPoints'] += 1

        las.add_curve('SP_DETREND',df['SP_DETREND'].values, unit='mV', descr='SP log with trend removed')
        las.add_curve('BKPTS',df['BreakPoints'].values, unit='NA', descr='Grouped between 10 depth series changepoints')

        NEWFILENAME = F.replace('.','_DETREND.')
        las.write(path.join(OUTFOLDER,NEWFILENAME), version = 2.0)
    return(None)
