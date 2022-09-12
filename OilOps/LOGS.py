# update base files
from ._FUNCS_ import *


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
    move(file,file2)
    with open(file2,'r') as F2:
        Fcontent = F2.read()
        soup = BS(index, 'lxml')
        TEXT = S.get_text('\n')
        TEXT = TEXT.strip()
    with open(file,'w') as F1:
        F1.write(TEXT)
    return None
    
def ZIPtoTXT(file,TYPECHECK=True):
    if TYPECHECK:
        if not FILETYPETEST(file,'ZIP'):
            print('ERROR: Are you sure ' + str(file)+' is ZIP?')
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
            print('ERROR: Are you sure ' + str(file)+' is MS DOC?')
            return
       
    file2 = file.split('.')[0]+'.html'
    rename(file,file2)

    with open(file2,'r') as F2:
        Fcontent = F2.read()
        soup = BS(index, 'lxml')
        TEXT = S.get_text('\n')
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
            pattern=re.compile(b'Thread was being aborted',re.I)
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
