# update base files
from ._FUNCS_ import *

__all__ = ['TEMP_SUMMARY_LAS',
           'HTMLtoTXT',
           'ZIPtoTXT',
           'DOCtoTXT',
           'LAS_TEXTABORTED_FIX',
           'LASREPAIR',
           'Find_R0',
           'List_LAS_Files_In_Folder',
           'FIND_SP_KEY',
           'LOG_DETREND',
           'ARRAY_CHANGEPOINTS',
           'Init_Futures',
           'SP_WORKFLOW',
           'DLOGR',
           'GetAlias',
           'Alias_Dictionary',
           'LogListAlias',
           'R0_DLOGN',
           'Get_API',
           'Mechanics']


# Update clay model to consider Bhuyan and Passey, 1994 method + Nphi_fl and Rho_fl from RLOGR R0


def TRfromHI(HI,HIO=750):
    TR = ((HIO/100)-(HI*(60-5*HIO/100))/(6000-5*HI))/(HIO/100)
    return TR

def HIfromTR(TR,HIO=750):
    HI = 60*HIO*(TR-1)/(5*HIO/100-60+5*(TR-1)*(HIO/100))
    return HI
                   

def decompose_log(df_in, p_thresh = None):
    if p_thresh == None:
        p_thresh = np.floor( df_in.dropna().shape[0]/2 )
        p_thresh = int(p_thresh)
    
    decompose = seasonal_decompose(df_in.dropna(),model='additive', period= p_thresh)
    return decompose.trend

def RollingAvg(df_in, windowsize, values_key,window_key = None):
    windowsize = np.floor((windowsize-1)/2)

    df_in = df_in.copy()
 
    if window_key == None:
        df_in['WIN'] = df_in.index
    else: 
        df_in['WIN'] = df_in[window_key]

    min_val = df_in['WIN'].min()
    max_val = df_in['WIN'].max()

    df_in['ROLL'] = np.nan
    
    for i in df_in.index:
        val = df_in.loc[i,'WIN']
        win0 = max(val - windowsize,min_val)
        win1 = min(val + windowsize,max_val)
        m = ((df_in['WIN']>= win0) & (df_in['WIN']<= win1)).values
        df_in.loc[i,'ROLL'] = df_in.loc[m,values_key].interpolate().mean()

    return df_in['ROLL']                              
    
def decompose_log(df_in, p_thresh = None):
    if p_thresh == None:
        p_thresh = np.floor( df_in.dropna().shape[0]/2 )
        p_thresh = int(p_thresh)
    
    decompose = seasonal_decompose(df_in.dropna(),model='additive', period= p_thresh)
    return decompose.trend


def detrend_log(df,xkey='index',ykey='SP',return_model = False,model_index = [], log = False):
    df_in = df.copy()
    if len(model_index) == 0:
        model_index = df_in.index
    if isinstance(df_in,pd.Series):
        ykey = df_in.name
        df_in['IDX'] = df_in.index
        x = df_in.loc[model_index].dropna().index.values
        y = df_in.loc[model_index].dropna().values   
        xkey = 'IDX'
    elif isinstance(df_in,pd.DataFrame):
        m1 = df_in[[xkey,ykey]].dropna().index
        m = model_index.intersection(m1)
        df_in['IDX'] = df_in.index
        x=df_in.loc[m,xkey].values
        y=df_in.loc[m,ykey].values
    if log:
        y = np.log10(y)       
    model = np.polyfit(x, y, 1)
    pred=np.poly1d(model)
    df_in[ykey+'_TREND'] = df_in[xkey].apply(lambda x: pred(x))
    if log:
        df_in[ykey+'_TREND'] = 10**df_in[ykey+'_TREND']
    if return_model:
        return pred
    else:
        return df_in[ykey+'_TREND']

def rescale_data(data_series, target_min_percentile_value, target_max_percentile_value, min_percentile=5, max_percentile=95):
    """
    Rescales a data series linearly based on specified percentiles.

    Args:
        data_series (np.array or pd.Series): The input data series.
        target_min_percentile_value (float): The desired value for the min_percentile.
        target_max_percentile_value (float): The desired value for the max_percentile.
        min_percentile (int): The percentile to use as the lower bound (default is 5).
        max_percentile (int): The percentile to use as the upper bound (default is 95).

    Returns:
        np.array: The rescaled data series.
    """

    # Calculate the actual 5th and 95th percentiles of the data
    p5 = np.percentile(data_series, min_percentile)
    p95 = np.percentile(data_series, max_percentile)

    # Handle the edge case where the 5th and 95th percentiles are the same
    if p95 == p5:
        if target_max_percentile_value == target_min_percentile_value:
            return np.full_like(data_series, target_min_percentile_value)
        else:
            raise ValueError("Cannot rescale: 5th and 95th percentiles are identical, but target values are different.")

    # Calculate the scaling factor (slope)
    scale_factor = (target_max_percentile_value - target_min_percentile_value) / (p95 - p5)

    # Calculate the offset (intercept)
    offset = target_min_percentile_value - (p5 * scale_factor)

    # Apply the linear transformation
    rescaled_data = (data_series * scale_factor) + offset

    return rescaled_data

def robust_log_fit(x, y,x2):
    log_x = np.log(x).reshape(-1, 1)
    model = HuberRegressor().fit(log_x, y)
    return model.predict(np.log(x2).reshape(-1, 1))

def tune_eaton_exponent(modulus, nct, overburden, hydrostatic, min_exp=1.0, max_exp=6.0, steps=50, ratio_invert=False):

    modulus = modulus.copy()
    nct = nct.copy()
    overburden = overburden.copy()
    hydrostatic = hydrostatic.copy()

    best_exp = 2.0
    best_score = np.inf
    hydro_grad = hydrostatic / overburden  # Expect ~0.465

    mm=(~np.isnan(modulus+nct+overburden+hydrostatic))
    modulus = modulus[mm]
    nct = nct[mm]
    overburden = overburden[mm]
    hydrostatic = hydrostatic[mm]

    for e in np.linspace(min_exp, max_exp, steps):
        if ratio_invert:
            modulus_ratio = np.clip(modulus / nct, 0.01, 10)
        else:
            modulus_ratio = np.clip(nct / modulus, 0.01, 10)

        # ratio = np.clip(nct / modulus, 0.01, 10)
        pp = overburden - (overburden - hydrostatic) * (modulus_ratio ** e)

        # Penalties
        underpressured = pp < hydrostatic
        percent_under = underpressured.mean()
        underpressure_penalty = 50 * percent_under + 5 * np.std(pp[underpressured]) if np.any(underpressured) else 0
        
        # Sudden changes penalty
        smoothness_penalty = np.std(np.diff(pp))

        # Physically unrealistic values (negative or too high)
        unrealistic_penalty = 100 * np.mean((pp < 0) | (pp > overburden * 1.1))

        total_score = underpressure_penalty + smoothness_penalty + unrealistic_penalty

        if total_score < best_score:
            best_score = total_score
            best_exp = e

    return best_exp

def compute_eaton_pp(depth, log_values, log_type='density', overburden=None, hydrostatic=None, eaton_exponent=3.0, filter = [], quantile = None):
    """
    Compute Eaton pore pressure and normal compaction trend (NCT) from a given log.
    
    Parameters:
        depth: np.array of depth values
        log_values: np.array of measured log values
        log_type: 'density', 'neutron', or 'resistivity'
        overburden: np.array or scalar of overburden pressure (psi)
        hydrostatic: np.array or scalar of hydrostatic pressure (psi)
        eaton_exponent: default Eaton exponent

    Returns:
        nct_fit: np.array of NCT fit values
        pore_pressure: np.array of estimated pore pressures (psi)
    """

    if len(filter) == 0:
        filter = [True] * len(depth)
    
    
    depth = np.asarray(depth)
    log_values = np.asarray(log_values)

    if quantile == None:
        if log_type.lower().startswith('neu'):
            q = 0.2
        elif log_type.lower().startswith('res'):
            q = 0.9
            log_values = np.log10(log_values)
        elif log_type.lower().startswith('den'):
            q = 0.9
    else:
        q = quantile

    # Step 1: Fit Normal Compaction Trend using lower envelope
    # Use 10th percentile in a rolling window to approximate NCT
    window = 500  # adjust based on resolution
    nct_envelope = pd.Series(log_values).rolling(window=window, min_periods=10, center=True).quantile(q)
    mask = ~nct_envelope.isna() * (depth>0) * filter
    #popt, _ = curve_fit(exp_nct, depth[mask], nct_envelope[mask])

    #nct_fit = exp_nct(depth, *popt)
    
    nct_model = linregress( depth[mask], nct_envelope[mask] )
    log_nct = nct_model.intercept + nct_model.slope * np.log(depth)
    nct_fit = np.exp(log_nct)

    if False:
        plt.plot(rhob, depth)
        plt.plot(nct_fit,depth)
        plt.xlim(0,4)
        plt.show()


    # Step 2: Handle pressures
    if overburden is None:
        overburden = 1.0 * depth * 0.052 * 1000  # ppg to psi

    if hydrostatic is None:
        hydrostatic = 0.465 * depth * 0.052 * 1000  # typical gradient
 
    overburden = np.asarray(overburden)
    hydrostatic = np.asarray(hydrostatic)

    # Step 3: Compute Ratio (Resistivity is inverted)
    mm = (depth>3000) * (~np.isnan(log_values)) * filter

    if log_type.lower().startswith('r'):
        ratio = np.clip(log_values / nct_fit, 0.001, 10)
        eaton_exponent = tune_eaton_exponent(log_values[mm],
                                    nct_fit[mm],
                                    overburden[mm],
                                    hydrostatic[mm],
                                    min_exp = 0.2,
                                    max_exp = 4, 
                                    steps = 2000,
                                    ratio_invert = True)        
        


    elif log_type.lower().startswith('n'):
        ratio = np.clip(log_values/nct_fit, 0.001, 10)
        eaton_exponent = tune_eaton_exponent(nct_fit[mm],
                                            log_values[mm],
                                            overburden[mm],
                                            hydrostatic[mm],
                                            min_exp = 0.2,
                                            max_exp = 10, 
                                            steps = 2000,
                                            ratio_invert = False)
    elif log_type.lower().startswith('d'):
        ratio = np.clip(nct_fit / log_values, 0.01, 10)
        eaton_exponent = tune_eaton_exponent(nct_fit[mm],
                                            log_values[mm],
                                            overburden[mm],
                                            hydrostatic[mm],
                                            min_exp = 0.2,
                                            max_exp = 10, 
                                            steps = 2000,
                                            ratio_invert = True)

    # Step 4: Eaton Equation
    pore_pressure = overburden - (overburden - hydrostatic) * (ratio ** eaton_exponent)

    return nct_fit, pore_pressure

def fit_quantile_nct_surface(
    df,
    y_col,                # e.g., 'DT'  (for modulus later you'd build M_NCT from DT_NCT & RHOB_NCT)
    depth_col='DEPTH_FT',
    umaa_col='UMAA',
    tau=0.15,             # 0.10–0.20 for lower envelope (DT); 0.80–0.90 for upper envelope (RHOB, R0)
    df_depth=7,           # spline degrees of freedom for depth
    df_umaa=5,            # spline degrees of freedom for UMAA
    df_interact=4,        # complexity of the interaction (usually <= min(df_depth, df_umaa))
    weights=None          # optional: observation weights (e.g., kernel by depth) to localize fit
):
    data = df[[y_col, depth_col, umaa_col]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[y_col, depth_col, umaa_col])
    if data.empty:
        return pd.Series(np.nan, index=df.index, name=f'{y_col}_NCT')

    # Natural splines (depth) + B-splines (UMAA) + interaction.
    # Use ':' for pure interaction; use '+' to include main effects.
    # If you want full interaction (main effects + interaction), you can also use '*':
    #   cr(depth, df=df_depth) * bs(UMAA, df=df_umaa)
    formula = (
        f"{y_col} ~ cr({depth_col}, df={df_depth})"
        f" + bs({umaa_col}, df={df_umaa})"
        f" + cr({depth_col}, df={max(3, df_interact)}):bs({umaa_col}, df={max(3, df_interact)})"
    )

    model = smf.quantreg(formula, data=data)
    res = model.fit(q=tau, max_iter=10000, weights=weights)

    # Predict across the full index (handle NaNs gracefully)
    pred = res.predict(df)
    pred.name = f'{y_col}_NCT'
    return pred


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
        soup = BS(Fcontent, 'lxml')
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
    processors = max(1,multiprocessing.cpu_count()-1)
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

        las.append_curve('SP_DETREND',df['SP_DETREND'].values, unit='mV', descr='SP log with trend removed')
        las.append_curve('BKPTS',df['BreakPoints'].values, unit='NA', descr='Grouped between 10 depth series changepoints')

        NEWFILENAME = F.replace('.','_DETREND.')
        las.write(path.join(OUTFOLDER,NEWFILENAME), version = 2.0)
    return(None)

def Get_API(las):
    uwikeys=['API','UWI','APIN']
    uwis=list()
    for key in uwikeys:
        if key in las.well.keys():
             try:
                uwi = re.sub(r'[^0-9]','',str(las.well[key].value))
                if len(uwi)>=8:
                    uwis.insert(uwikeys.index(key)-1,uwi)
             except:
                pass
    uwis = list(set([x for x in uwis if (len(x)>=10 & len(x)<=14)]))
    if len(uwis)==0:
        Puwi="0000_NOAPI"
    else:
        Puwi=str(int(uwis[0]))
    return(Puwi)

def R0(phi,arw,m):
    return np.float32(arw)*np.float32(phi)**np.float32(-m)

def func(x, a, b): return a + x * b

def Find_R0(df, sigma = 500):
    df_in = df.copy()
    depth = df_in.index
    ALIAS = GetAlias(df_in)        

    # --- Required logs ---
    m_phi = df_in[ALIAS['NPHI']].copy().replace([np.inf, -np.inf], np.nan).interpolate(limit_area='inside').dropna().index
    m_resd = df_in[ALIAS['RDEEP']].copy().replace([np.inf, -np.inf], np.nan).interpolate(limit_area='inside').dropna().index

    phin = df_in[ALIAS['NPHI']].copy().replace([np.inf, -np.inf], np.nan).interpolate(limit_area='inside').values
    resd = df_in[ALIAS['RDEEP']].copy().replace([np.inf, -np.inf], np.nan).interpolate(limit_area='inside').values

    if np.nanmean(phin)>1:
        phin = phin/100
            
    # --- Clean up ---
    phin = np.clip(phin, 0.01, 0.7)
    resd = np.clip(resd, 0.2, 2000)

    phin[depth<2000] = np.nan
    resd[depth<2000] = np.nan

    log_phi = np.log10(phin)
    log_resd = np.log10(resd)

    # --- Initialize output ---
    r0_continuous = np.full_like(depth, np.nan)
    m_continuous = np.full_like(depth, np.nan)    

    # --- Moving weighted regression parameters ---
    sigma = 500  # controls window width (ft)

    for i in range(len(depth)):
        z = depth[i]
        weights = np.exp(-0.5 * ((depth - z)/sigma)**2)
        valid = (np.isfinite(log_phi)) & (np.isfinite(log_resd))
        
        if np.sum(weights[valid]) > 20:
            x = log_phi[valid]
            y = log_resd[valid]
            w = weights[valid]

            # Sort by resistivity to apply weighted quantile threshold
            sort_idx = np.argsort(y)
            y_sorted = y[sort_idx]
            w_sorted = w[sort_idx]
            x_sorted = x[sort_idx]

            # Compute cumulative weights
            cum_weights = np.cumsum(w_sorted)
            cum_weights /= cum_weights[-1]

            # Apply weighted quantile (e.g., 20th percentile)
            q_idx = np.searchsorted(cum_weights, 0.2)
            threshold = y_sorted[q_idx]

            mask = y <= threshold
            x_sub = x[mask]
            y_sub = y[mask]
            w_sub = w[mask]

            if len(x_sub) > 5:
                w_sum = np.sum(w_sub)
                xw = np.sum(w_sub * x_sub) / w_sum
                yw = np.sum(w_sub * y_sub) / w_sum
                slope = np.sum(w_sub * (x_sub - xw) * (y_sub - yw)) / np.sum(w_sub * (x_sub - xw) ** 2)
                intercept = yw - slope * xw

                m = -slope
                a = 10**intercept
                r0_continuous[i] = a * phin[i]**(-m)
                m_continuous[i] = m
                   
    #r0_smooth1 = gaussian_filter1d(r0_continuous, sigma=10, mode='nearest')
    r0_smooth2 = savgol_filter(r0_continuous, 51, 2, mode='nearest')

    df_in ['R0'] = r0_smooth2
    return df_in ['R0']

def R0_DLOGN(df,uwi,Archie_N,LABEL='0'):
    #if 1==1:
    #pathname = path.dirname(sys.argv[0])
    #dir = path.abspath(pathname)
    #dir_add = dir+"\\DLOGR"
    #alias = GetAlias(df)
    #df=las.df()[[alias["NPHI"],alias["RDEEP"]]].dropna()
    #Alias = GetAlias(las)
    LABEL='_'+str(LABEL)
    #dir_add = path.abspath(path.dirname(sys.argv[0]))+"\\DLOGR"
    dir_add = path.join(getcwd(),'DLOGR')  
    if not path.exists(dir_add):
        makedirs(dir_add)
        
    dfx=df.dropna()
    dfx["SW"]=None
    dfx["R0"]=None
    NPHI=dfx.keys()[0]      #1st column is NPHI track
    RD=dfx.keys()[1]        #2nd column is RDEEP track
    if dfx[NPHI].median(axis=0)>2:
        dfx[NPHI]=dfx[NPHI]/100
    dfx=dfx.loc[(dfx[NPHI]>-0.5) & (dfx[NPHI]<0.5)]
    if (len(dfx[NPHI])>200):
        if min(dfx[NPHI])<=0:
            Phidelta=0.005-min(dfx[NPHI])
        else: Phidelta=0
        dfx["NPHI_X"]=dfx[NPHI]+Phidelta
        PHI_0=np.float32(min(dfx["NPHI_X"]))
        PHI_F=np.float32(min(0.3,max(dfx["NPHI_X"])))
        if(PHI_0<=0):
            Phidelta=0.005-PHI_0
            dfx["NPHI_X"]=dfx[NPHI]+Phidelta
            PHI_0=np.float32(min(dfx["NPHI_X"]))
            PHI_F=np.float32(max(dfx["NPHI_X"]))
        bin=10**np.arange(log10(PHI_0),log10(PHI_F),((log10(PHI_F)-log10(PHI_0))/20))
        r0_est = np.zeros((len(bin),5))
        for i in range(0,len(bin)-1):
            r0_est[i,0] = (bin[i]+bin[i+1])/2
            r0_est[i,4] = log10((bin[i]+bin[i+1])/2)
            r0_est[i,1] = sum(((dfx["NPHI_X"]>=bin[i]) & (dfx["NPHI_X"]<bin[i+1])))
            #    bin_cent[i,1] = sum(np.nonzero((FM_NEU>=bin[i]) and (FM_NEU<bin[i+1])))
            data = dfx[RD].loc[((dfx["NPHI_X"]>=bin[i]) & (dfx["NPHI_X"]<bin[i+1]))]
            if len(data)>10:
                param = stats.gamma.fit(data)
                r0_est[i,2]=stats.gamma.ppf(0.05,*param)
                if r0_est[i,2] > 0:
                    r0_est[i,3]=log10(r0_est[i,2])
                else:
                    r0_est[i,2] = None
                    #r0_est[i,2]=(sorted(data)[0]*sorted(data)[1]*sorted(data)[2])**(1./3.)
                    #r0_est[i,3]=log10((sorted(data)[0]*sorted(data)[1]*sorted(data)[2])**(1./3.))
        r0_est=r0_est[np.all(r0_est != 0, axis=1)]  # remove rows containing zero
        r0_est=r0_est[~np.isnan(r0_est).any(axis=1)] # remove rows containing nan

        #plt.plot(las[NKEY][np.nonzero((las[0]>top) & (las[0] < base))],las[RKEY][np.nonzero((las[0]>top) & (las[0] < base))],'b.')

        try: 
            popt, pcov = scipy.optimize.curve_fit(func, r0_est[(r0_est[:,2])>0,4], r0_est[(r0_est[:,2])>0,3])
        except:
            popt = [-2,2]

        ARW=np.float32(10**popt[0])
        M=np.float32(-popt[1])

        #popt, pcov = scipy.optimize.curve_fit(func, r0_est[:,0], r0_est[:,2])
        #plt.plot(las[NKEY][np.nonzero((las[0]>top) & (las[0] < base))],las[RKEY][np.nonzero((las[0]>top) & (las[0] < base))],'b.')
        #plt.plot(r0_est[:,0], func(r0_est[:,0], *popt), 'r-')
        Puwi=str(0)*(14-len(str(int(uwi))))+str(int(uwi))

        dfx["R0"]=dfx["NPHI_X"].map(lambda x: R0(x,ARW,M),na_action=None)
        dfx["SW"]=((dfx["R0"]/dfx[RD])**(1/Archie_N)).clip(0,1)
        #if M<=0:
        #    dfx=None

        # Create plot
        dfx_mask = dfx[['NPHI_X',RD]].dropna().index
        if dfx.shape[0]>5:
            fig, ax = plt.subplots()
            plt.xlim(0.01,1)
            plt.ylim(0.1,100)
            ax.plot(dfx["NPHI_X"],dfx[RD],'b.')
            ax.plot(r0_est[:,0],r0_est[:,2],'ro')
            ax.plot(np.arange(PHI_0,PHI_F,(PHI_F-PHI_0)/100),ARW*(np.arange(PHI_0,PHI_F,(PHI_F-PHI_0)/100)**(-M)),'k-',linewidth=1)
            ax.set(xlabel=str(NPHI)+' [v/v]',
                    ylabel= str(RD) + ' [Ohmm]',
                    title= 'UWI: '+str(Puwi)+ '\n m= ' + str(M) +'    ARw=' + str(ARW))
            df["DEPTH"]=df.index.astype(float)
            figfile=str(dir_add)+"\R0_Plot_"+str(Puwi)+"_"+str(min(df.DEPTH.astype(int)))+"_"+str(max(df.DEPTH.astype(int)))+LABEL+".png"
            if path.isfile(figfile):
               remove(figfile)
            fig.savefig(figfile)
        #except: pass
        try: plt.close()
        except: pass
        #NCALC=[e for e in Ninit if isinstance(e, (int, float))]+Phidelta
        #dfx["Sw"]=np.clip((np.fromiter(map(lambda x: R0((x+Phidelta),ARW,M),dfx["NPHI_X"]),dtype=np.float32)/dfx["NPHI_X"])**(1/Archie_N),0,1)
        #r0[(~np.isnan(Ninit)) & (Ninit>-Phidelta) & (~np.isnan(Rinit))]=np.fromiter(map(lambda x: R0((x+Phidelta),ARW,M),Ninit[(~np.isnan(Ninit)) & (Ninit>-Phidelta) & (~np.isnan(Rinit))]),dtype=np.float32)
        dfx["R0"]=dfx["NPHI_X"].map(lambda x: R0(x,ARW,M),na_action=None)

    output=dfx[["SW","R0"]]
    output.index=output.index.astype(str)
    return output

def Alias_Dictionary():
    AliasDicts={'BIT':{"BS":2,"BIT":1},
                'CAL':{"CALI":1,"CAL":1,"CAL1":1,"C13":2,"C13A":2,
                       "C13-A":2,"C13H":2,"C13I":2,"C13L":2,"C13M":2,
                       "C13P":2,"C13Z":2,"CA1":2,"CA2":2,"CADE":2,
                       "CADF":2,"CALI":1,"CALX":1,"CALZ":1,"CANC":2,
                       "CAPD":2,"CAX":2,"CLXC":2,"DCAL":2,"HCALX":1,
                       "SA":2,"HHCA":2,"ACAL":3,"C1":3,"C2":3,"C24":3,
                       "C24A":3,"C24-A":3,"C24H":3,"C24I":3,"C24L":3,
                       "C24M":3,"C24P":3,"C24Z":3,"CA":3,"CAL2":3,
                       "CAL3":3,"CALA":3,"CAL-A":3,"CALD":3,"CALE":3,
                       "CALH":3,"CALI_SPCS":3,"CALL":3,"CALM":3,"CALN":3,
                       "CALP":3,"CALS":3,"CALT":3,"CALX-A":3,"CALXH":3,
                       "CALXM":3,"CALX-ML":3,"CALXQ8":3,"CALXQH":3,
                       "CALX-R":3,"CALY":3,"CALY-A":3,"CALYH":3,"CALYM":3,
                       "CALY-ML":3,"CALYQ8":3,"CALYQH":3,"CAY":3,"CLCM":3,
                       "CLDC":3,"CLDM":3,"CLLO":3,"CLTC":3,"CLYC":3,
                       "CQLI":3,"HCAL":1,"HCAL2":3,"HCALI":3,"HCALY":3,
                       "HD":3,"HD_1":3,"HD1":3,"HD2":3,"HD3":3,"HDAR":3,
                       "HDIA":3,"HDMI":3,"HDMN":3,"HDMX":3,"HLCA":3,
                       "LCAL":3,"MCAL":3,"MLTC":3,"TAC2":3,"C3":3,"CLS2":6,
                       "MBTC":6,"TACC":6},
                'DCORR':{"RHOC":1,"2DRH":1,"DC":1,"DCOR":1,"DECR":1,"DRH":1,"DRHO":1,
                       "HBDC":1,"HDRA":1,"HDRH":1,"HHDRA":1,"RPCL_DCOR":1,
                       "ZCOR":1,"ZCOR2":1,"ZCOR2QH":1,"ZCORQH":1,"HHDR":1,"Z-COR":1},
                'DTC':{"AC":1,"ACCO":2,"ACL":2,"ACN":2,"DT":1,"DT24":2,
                       "DT24-A":2,"DT24AQI":2,"DT24QI":2,"DT24SQA":2,
                       "DT34":2,"DT35":2,"DT41":1,"DT4P":1,"DT4P_C":1,
                       "DT5":2,"DTC":1,"DTC1":2,"DTC2":1,"DTCA":2,
                       "DTCM":1,"DTCO":1,"DTCR":2,"DTCX":2,"DTL":1,
                       "DTLF":1,"DTMN":1,"DTMX":1,"DTP":1,"DTSC":2,
                       "TTC":1,"AC1":1,"MSTT":5,"VEL":3,"MDT":3,"DXDT":4,
                       "DYDT":5},
                'DTS':{"D4S":2,"DT4S":1,"DTS":1,"DTSD":1,"DTSF":1,"DTSS":2,
                       "SHSL":1,"TTS":5,"DTSM":1,"DTSH_MST":3},
                'GR':{"HNGS":1,"CGR":1,"ECGR":1,"EHGR":1,"GGCE":1,"GR-A":1,"GRC":1,
                       "GRCO":1,"GRDE":1,"GRDI":1,"GRGC":1,"GRGM":1,"GRN":1,
                       "GRNC":1,"GRNP":1,"GRQH":1,"HCGR":1,"HGR":1,"HGR_STGC":1,
                       "HGR-1A":1,"HGR-5A":1,"GRS":1,"HEHG":1,"GAB":2,"GAM":2,
                       "GAM1":2,"GAPI":2,"GR":2,"GR_AR":2,"GR_STGC":2,"GR1":2,
                       "GR-1A":2,"GR2":2,"GR-5A":2,"GRA":2,"GRDA":2,"GRG":2,
                       "GRGS":2,"GRH":2,"GRHD":2,"GRI":2,"GR-I":2,"GRLL":2,
                       "GRML":2,"GR-ML":2,"GRP":2,"GRPD":2,"GRR":2,"GR-R":2,
                       "GRSD":2,"GRSL":2,"GRZ":2,"GRZD":2,"HHGR":2,"HSGR":2,
                       "MGR":2,"RGR":2,"SGR":2,"SGRD":2,"GCPS":2,"GRM1":2,
                       "P01LGR":3,"P02LGR":3,"P03LGR":3,"GAMMA":3},
                'SPEC_K':{"K":2,"POTA":1,"HPOT":1,"KCPS":1,"HFK":1},
                'SPEC_U':{"HURA":1,"U":4,
                          "URAN":1,"UCPS":1,"UZ":1,"URAN":2},
                'SPEC_TH':{"TH":1,"HTHO":1,"THOR":1,"TCPS":1},
                'NPHI':{"APDC":5,"APDU":5,"APLC":4,"APLU":4,"CN":2,
                       "CNC":2,"CNCC":4,"CNCF":4,"CNCL":4,"CNCLS":4,
                       "CNCLSQH":4,"CNCQH":4,"CNCS":1,"CNCSGS":1,
                       "ENPH":2,"FNPS":2,"FPLC":5,"FPMC":4,"FPSC":3,
                       "HADC":5,"HALC":4,"HASC":1,"HHNP":2,"HHNPO":2,
                       "HNPH":1,"HNPHI":2,"HNPLS":4,"HNPO":2,"HNPO_LIM":4,
                       "HNPS":1,"HTNP":1,"HTNP_LIM":4,"NCN":2,"NCNL":2,
                       "NPHI":2,"NPHI_LIM":4,"NPHIL":4,"NPHIS":1,"NPHL":4,
                       "NPHS":4,"NPLS":4,"NPOR":2,"NPOR_":2,"NPOR_1":2,
                       "NPOR_LIM":4,"NPORLS":4,"NPRL":4,"POL":4,"POS":2,
                       "PRON":3,"QNP-1A":3,"QNP-5A":3,"RPOR":2,"SNP":2,
                       "TNPH":2,"TNPH_LIM":4,"TPHC":4,"CNPOR":4,"NLIM":2,
                       "NPHI_100":2,"CNCFLS":5,"CNLS":5,"CNPOR1":4,"DNPH":3,"PHIN":5,
                       "NPRS":3,"NPRD":3,"NPES":2, "NPHI_LS": 5},
                'PE':{"HPRA":1,"PEF8":1,"PEFZ":1,"2PEF":2,"HPEDN":1,"HPEF8":1,
                       "PDPE":1,"PE":1,
                       "PE2":1,"PE2QH":1,"PEDF":1,"PEDN":1,"PEF":1,
                       "PEF_SLDT":1,"PEF_SLDT_HR":1,"PEF8":1,"PEFA":1,
                       "PEFSA":1,"PEFZ":0,"PEQH":1,"SPEF":1,"HPEF":1,
                       "HPEF":1,"PEFI":1,"PEFI":1,"PEFL":1,"PEFL":1,
                       "LPE":1,"PEFS":2,"PEF_":2},
                'DEN':{"2RHB":3,"CODE":4,"DEN":3,"DENB":3,"HDEN":3,"HNDP":1,
                       "HRHO":3,"HRHOB":2,"HROM":4,"LRHO":4,"NRHB":4,"NRHO":4,
                       "RHO8":2,"RHOB":2,"RHOB_SLDT":4,"RHOI":4,"RHOM":4,
                       "RHOS":3,"RHOZ":1,"ZDEN":3,"ZDEN2":3,"ZDEN2QH":4,
                       "ZDENQH":3,"ZDENS":3,"ZDNC":3,"ZDNCQH":3,"ZDNCS":3,
                       "RHL":3,"RHS3":3,"RHS4":3},
                'RDEEP':{"A22H":10,"A22H_":10,"A22H_UNC":10,"A28H":10,"A28H_":10,"A28H_UNC":10,
                       "A34H":10,"A34H_":10,"A34H_UNC":10,"AF60":7,"AF90":3,"AFRT":6,
                       "AHF60":8,"AHF90":5,"AHO60":8,"AHO90":6,"AHT60":6,"AHT90":5,
                       "AHTRT":3,"AILD":3,"AIT":3,"AIT120":1,"AIT60":4,"AIT90":1,
                       "AO12":8,"AO60":7,"AO90":5,"AORT":6,"ASF60":8,"ASF90":6,
                       "ASO60":7,"ASO90":6,"AST60":4,"AST90":1,"AT60":3,"AT90":1,
                       "ATRT":1,"DIPH":2,"DVR1":6,"DVR2":2,"DVR4":4,"FE2":2,
                       "HILD":4,"HLLD":4,"HRL4":6,"HRL5":4,"HRLD":1,"RT60":6,
                       "IDER":4,"IDPH":2,"IDVR":2,"IL":2,"ILD":0,"ILD2":1,"RT90":1,
                       "ILD4":4,"LL":4,"LL3":4,"LL7":4,"LLD":4,"LN":6,"M0R6":6,
                       "M2R9":4,"M2RX":6,"M4R6":6,"M4R9":4,"M4RX":1,"P10H":10,
                       "P10H_":10,"P10H_UNC":10,"P16H":10,"P16H_":10,"P16H_UNC":10,
                       "P22H":10,"P22H_":10,"P22H_UNC":10,"P28H":10,"P28H_":10,"P28H_UNC":10,
                       "P34H":10,"P34H_":10,"P34H_UNC":10,"RFOC":4,"RILD":4,"RIPD":4,
                       "RLA1":6,"RLA2":4,"RLA3":3,"RLA4":4,"RLA5":6,
                       "RLL4":4,"RLL5":4,"RT":6,"RT_HRLT":1,"VILD":4,"VP65":1,
                       "VRSD":6,"ID40":6,"AIT6":4,"DDLL":4,"EIRD":4,"ID10":6,
                       "ID20":4,"ID25":4,"ILD1":4,"LLG":4,"REID":4,"RLL":5,
                       "AHF6":8,"AHF9":6,"AHO6":8,"AHO9":6,"AHOR":6,"AHT6":4,
                       "AHT9":1,"AHTR":1,"AIT9":1,"AS60":4,"AS90":1,"ASF6":8,
                       "ASF9":6,"ASFR":6,"ASRT":4,"AST6":4,"AST9":1,"ASTR":1,
                       "HAT6":8,"HAT9":1,"HATR":1,"RLLD":1,"RT_H":4,"AC90":10,
                       "AF6C":10,"AF9C":10,"AFCO":10,"AHFC":10,"AHFCO":10,"AHFCO60":10,
                       "AHFCO90":10,"AHOC":10,"AHOCO60":9,"AHOCO90":8,"AHTC":9,
                       "AHTCO":10,"AHTCO60":10,"AHTCO90":8,"AO6C":9,"AO9C":10,
                       "AOCO":10,"AS9C":10,"ASCO":10,"ASTC":10,"ASFC":9,"ASFCO60":9,
                       "ASFCO90":5,"ASOCO90":10,"ASTCO90":8,"AT6C":9,"AT9C":9,
                       "ATCO":10,"CFOC":10,"CHTZ":10,"CIDP":10,"CIDQ":10,"CIDR":10,
                       "CIL":10,"CILD":11,"CIPD":10,"CLLD":8,"COND":10,"DVC1":10,
                       "DVC2":10,"DVC4":10,"FC":10,"M0CX":10,"M1CX":10,"M2CC9":10,
                       "M2CCX":10,"M2CX":10,"M4CCX":10,"M4CX":10,"MVC2":10,"P34H_COND":10,
                       "RO90":6,"RD":4,"RTAO":1,"AT90DS":2, "HART":6, "RESD":10, "RESD_90M2":10},
                'RMED':{"AF30":2,"AHF30":2,"AHO30":2,"AHT30":1,"AIT30":1,"AO30":3,
                       "ASF30":2,"ASO30":3,"AST30":1,"AT30":1,"HILM":1,
                       "HLLS":1,"HRL3":1,"ILM":1,"ILM2":1,"ILM4":2,"IM":1,
                       "IMER":1,"IMPH":1,"IMVR":1,"LLS":2,"M0R3":1,"M1R3":1,
                       "M2R3":1,"M4R3":1,"MVR1":3,"MVR2":1,"MVR4":2,"RILM":1,
                       "RIPM":1,"RPCL":5,"VILM":2,"VP25":1,"AE60":2,"ILD25":2,
                       "ILM1":2,"ILM25":2,"IM10":2,"IM20":2,"IM25":2,"IM40":2,
                       "PIRM":2,"REIM":2,"DSLL":5,"AIT3":2,"AHF3":3,"AHO3":3,
                       "AHT3":1,"AS30":1,"ASF3":3,"AST3":1,"HAT3":1,"RLLS":3,
                       "CILM":5,"CIMQ":5,"CIMR":5,"CIPM":5,"RMLL":5,"RO30":3,
                       "RM":4, "RESM":5, "RT30": 2},
                'RSH':{"AF10":4,"AF20":5,"AFRX":3,"AHF10":4,"AHF20":5,
                       "AHO10":4,"AHO20":5,"AHSFI":3,"AHT10":2,"AHT20":3,
                       "AHTRX":1,"AIT10":2,"AIT20":3,"AO10":4,"AO20":5,
                       "AORX":3,"ASF10":4,"ASF20":5,"ASFI":3,"ASFL":3,
                       "ASN":2,"ASO10":4,"ASO20":5,"AST10":2,"AST20":3,
                       "AT10":2,"AT12":3,"AT20":3,"ATRX":1,"FE1":2,
                       "FEFE":2,"FR":2,"FRA":2,"HRL1":1,"HRL2":2,
                       "HRLS":2,"HSFLU":2,"ILS":2,"LL8":2,"LLA":2,
                       "M0R1":2,"M0R2":3,"M1R1":2,"M1R2":3,"M2R1":2,
                       "M2R2":3,"M4R1":2,"M4R2":3,"MFR":2,"MLL":2,
                       "MSFL":2,"PSHG":2,"RES":2,"RESS":1,"RILS":2,
                       "RLL1":3,"RLL2":2,"RS":2,"RSFE":2,"RSFL":2,
                       "RSG":2,"RXO":1,"RXO_HRLT":1,"RXO1D":1,"RXO8":1,
                       "RXOI":1,"RXOZ":1,"SFBC":2,"SFL":2,"SFL4":2,
                       "SFLA":2,"SFLR":2,"SFLU":2,"SN":2,"AE20":5,
                       "AE30":5,"AHF1":3,"AHF2":4,"AHO1":3,"AHO2":4,
                       "AHT1":1,"AHT2":2,"AS10":1,"AS20":2,"ASF1":3,
                       "ASF2":4,"ASRX":2,"AST1":1,"AST2":2,"HAT1":1,"RO10":4,
                       "HAT2":2,"HSFL":1,"MSFF":1,"RXOH":1,"CSFL":5,
                       "MCON":5,"RLL3":3, "RESS":5, "RT10": 2,"RT20": 3},
                'SP':{"HSP":1,"PCH1":2,"SP":1,"SPBR":2,"SPC":2,"SPCG":2,
                       "SPD":2,"SPDF":2,"SPDH":1,"SPH":1,"SPP":2,"SPSB":1,
                       "SSPK":2,"SPS":3,"AHSC":2,"AHSF":1,"SPA_":2,"SP_S":1}
                }
    return AliasDicts
           
def GetAlias(las):
    # curve options = BIT,CALIPER,DEN_CORRECTION,SONIC_DTC
    # SONIC_DTS,GAMMA,SPECTRAL_K,SPECTRAL_U,SPECTRAL_TH,
    # NEUTRON_PHI,PHOTOELECTRIC,DENSITY,RDEEP,RMEDIUM,
    # RSHALLOW, SPONTANEOUS

    # Dictionaries
    AliasDicts= Alias_Dictionary()
           
    alias={}
    for i in set(AliasDicts): alias[i]="NULL"
    for i in AliasDicts:
        for j in range(min(AliasDicts[i].values()),1+max(AliasDicts[i].values())):
            ADicts = { key:value for key, value in AliasDicts[i].items() if value ==j }
            if i not in alias:
                try:
                    aliases=list((item for item in las.keys() if re.split(r':',item)[0] in sorted(ADicts.keys(), key=ADicts.__getitem__)))
                    for k in aliases:
                        k_len=0
                        k_keep=None
                        #print(k)
                        #(np.array([np.array(las[k])[~np.isnan(np.array(las[k]))])
                        x=np.array([np.array(las[k])])
                        if (x[(~np.isnan(x))&(x!=999)].shape[0])>k_len:
                            k_keep=k
                        alias[i]=k_keep
                #try: alias[i]=str(next(item for item in las.keys() if item[0:len(i)] in sorted(ADicts.keys(), key=ADicts.__getitem__)))
                except:
                    alias[i]="NULL"
                #str(next(item for item in las.keys() if re.split(r':',item)[0] in sorted(AliasDicts[i].keys(), key=AliasDicts[i].__getitem__)))
            if i in alias:
                if (alias[i]=="NULL"):
                    try:
                        aliases=list((item for item in las.keys() if re.split(r':',item)[0] in sorted(ADicts.keys(), key=ADicts.__getitem__)))
                        for k in aliases:
                            k_len=0
                            k_keep=None
                            #print(k)
                            #(np.array([np.array(las[k])[~np.isnan(np.array(las[k]))])
                            x=np.array([np.array(las[k])])
                            if (x[(~np.isnan(x))&(x!=999)].shape[0])>k_len:
                                k_keep=k
                            alias[i]=k_keep
                    except:
                        alias[i]="NULL"

                    
                #    try: alias[i]=str(next((item for item in las.keys() if re.split(r':',item)[0] in sorted(ADicts.keys(), key=ADicts.__getitem__)),"NULL"))
                #    except: alias[i]="NULL"
                #else:
                #    try: alias_x=str(next((item for item in las.keys() if re.split(r':',item)[0] in sorted(ADicts.keys(), key=ADicts.__getitem__)),"NULL"))
                #    except: continue
                #    if (np.array([np.array(las[alias_x])[~np.isnan(np.array(las[alias_x]))][~999]]).shape[0] < np.array([np.array(las[alias[i]])[~np.isnan(np.array(las[alias[i]]))][~999]]).shape[0]):
                #        alias[i]=alias_x
    return alias

def LogListAlias(Input:(str,list,tuple)):
    AliasDicts= Alias_Dictionary()
    AliasDict2 = {}
    for k1, v1 in AliasDicts.items():
        for k2 in v1.keys():
            AliasDict2.update({k2:k1})
    if isinstance(Input,str):
        OUT = AliasDict2.get(Input)
    if isinstance(Input,tuple):
        OUT = (AliasDict2.get(x) for x in Input)
    if isinstance(Input,list):
        OUT = [AliasDict2.get(x) for x in Input]
    return OUT

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2
    #print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            #print("******************************")
            #print("Column: ",col)
            #print("dtype before: ",props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)

            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if (result > -0.01) and (result < 0.01):
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            #print("dtype after: ",props[col].dtype)
            #print("******************************")

    # Print final result
    #print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2
    #print("Memory usage is: ",mem_usg," MB")
    #print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist

def DLOGR(LASfile):
    #if 1==1:
    exlas=lasio.LASFile()
    dir_add = path.join(getcwd(),'DLOGR')       
           
    try: las = lasio.read(LASfile)
    except: las=[[0]]

    Alias = GetAlias(las)
    if (len(las[0])>100):
        
        df = (las.df().astype(np.float32))
        df["DEPTH"] = las.df().index.astype(float)
        dfmask=df[[Alias['DEN'],Alias['NPHI']]].dropna(thresh=1).index

        df = df.loc[(df.index).isin(dfmask)]
        uwi=Get_API(las)

        Puwi=str(0)*(14-len(str(int(uwi))))+str(int(uwi))
        #print(str(uwi))
        if (Alias["NPHI"]!="NULL") and (Alias["DEN"]!="NULL") and (Alias["RDEEP"]!="NULL"):

            if (len(df[[Alias["NPHI"],Alias["DEN"],Alias["RDEEP"]]].dropna())>100):
                ####################
                # PETROPHYSICAL QC #
                ####################
                # NULL BADHOLE
                # LOG CORRECTIONS
                # SONIC DESPIKE
                
                if df[Alias["NPHI"]].median(axis=0)>2:
                    df[Alias["NPHI"]]=df[Alias["NPHI"]]/100

                df['BADHOLE']=np.nan
                if (Alias['DCORR'] != 'NULL'):
                    df.loc[np.absolute(df[Alias['DCORR']])>0.15,'BADHOLE']=1
                    

                ###########################
                # Initial Estimate Curves #
                ###########################
                # U_APPX = las[PE]*las[RHOB]
                #SW_DLOGN(las[NPHI],las[RD],las.well["UWI"].value,2)

                #try: x = R0_DLOGN(las.df()[[Alias["NPHI"],Alias["RDEEP"]]].dropna(),Puwi,2,'X')
                #except: x=pd.DataFrame()
                    
                # df_in = las.df()[[Alias["DEN"],Alias["RDEEP"]]]
                # df_in["DPHI"] = (2.75-df_in[Alias["DEN"]])/1.75
                    
                # try: y = R0_DLOGN(df_in[["DPHI",Alias["RDEEP"]]].dropna(),Puwi,2,'Y')
                # except: y=pd.DataFrame()
                # del df_in
                # xr0=0.01*las.df()[[Alias["NPHI"],Alias["RDEEP"]]].dropna()[0]**(-2)
                # xsw=((las.df()[[Alias["NPHI"],Alias["RDEEP"]]].dropna()[1])/(las.df()[[Alias["NPHI"],Alias["RDEEP"]]].dropna()[0]**(-2)))**(-2)
                # x=pd.DataFrame({"R0":xr0,"SW":xsw})
                # x=[0.01*las.df()[[Alias["NPHI"],Alias["RDEEP"]]].dropna()[0]**(-2)],
                # x=((las.df()[[Alias["NPHI"],Alias["RDEEP"]]].dropna()[1])/(las.df()[[Alias["NPHI"],Alias["RDEEP"]]].dropna()[0]**(-2)))**(-2)
                # if x.shape[1]>=1:
                #     x.columns = ["R0_Nappx","Sw_Nappx"]
                #     df[x.columns[0]]=np.nan
                #     df[x.columns[1]]=np.nan
                #     df.index=df.index.astype(str)
                #     x.index=x.index.astype(str)
                #     df.update(x)
                #     #df=pd.concat([df, x.iloc[:,0]], axis=1, join_axes=[df.index])
                #     #df=pd.concat([df, x.iloc[:,1]], axis=1, join_axes=[df.index])
                #     #del x
                # if y.shape[1]>=1:
                #     y.columns = ["R0_Dappx","Sw_Dappx"]
                #     df[y.columns[0]]=np.nan
                #     df[y.columns[1]]=np.nan
                #     df.index=df.index.astype(str)
                #     y.index=y.index.astype(str)
                #     df.update(y)
                #     #df=pd.concat([df, x.iloc[:,0]], axis=1, join_axes=[df.index])
                #     #df=pd.concat([df, x.iloc[:,1]], axis=1, join_axes=[df.index])
                #     #del y

                # if all (k in df.keys() for k in ("R0_Nappx","Sw_Nappx","R0_Dappx","Sw_Dappx")): 
                #     #####################
                #     # STATISTICAL UNITS #
                #     #####################
                #     A =pd.DataFrame({'DEPTH':df.DEPTH.astype(float),
                #                      'DEPTH':df.DEPTH.astype(float)+0.5,
                #                      'R0N/NPHI':(df["R0_Nappx"]*df[Alias["NPHI"]]**-2).rolling(15).mean(),
                #                      'R0N/RT':(df["R0_Nappx"]/df[Alias["RDEEP"]]).rolling(15).mean(),
                #                      'R0D/DPHI':(df["R0_Dappx"]*df[Alias["NPHI"]]**-2).rolling(15).mean(),
                #                      'R0D/RT':(df["R0_Dappx"]/df[Alias["RDEEP"]]).rolling(15).mean(),
                #                      }).dropna().astype(np.float32)
                #     #print(str(uwi),str(A.shape[0]))
                #     # HDBSCAN clusting association
                #     if len(A.iloc[:,0])>50:

                #         # Renumbered to label in depth order and associate outliers with nearest label
                #         # c_num = HDBSCAN(min_cluster_size=100).fit_predict(A)

                #         # c_num = 0+(A.DEPTH>4000)+(A.DEPTH>6000)

                #         #clusters = hdbscan.HDBSCAN(min_cluster_size=n).fit_predict(A);A["Labels"]=clusters;plt.plot(A["Labels"],-A.index);plt.title("Clusters="+str(1+max(clusters)));plt.show()
                #         if len(set(c_num))>0:
                #             # A["clusters"]=c_num
                #             # df.loc[df.index.isin(A.index),"cluster"]= c_num
                #             # if (min(c_num)<0) and (len(set(c_num))>1):
                #             #     def_clusters=dict(zip(list(set(A.clusters)),stats.rankdata(list(map(lambda x: min(A[A["clusters"]==x].index.astype(float)),list(set(A.clusters))))).astype("int")))
                #             #     A["clusters"].replace(def_clusters,inplace=True)
                #             #     #A.loc[A.clusters==1,'clusters']=list(map(lambda x: round(x).astype(int),np.interp(A[A["clusters"]==1].index.astype(float),A[A["clusters"]>1].index.astype(float),A.clusters[A.clusters>1])))
                #             #     #A['clusters']=list(map(lambda x: round(x).astype(int),np.interp(A.index.astype(float),A[A["clusters"]>1].index.astype(float),A.clusters[A.clusters>1])))
                #             #     A['clusters'] = list(map(lambda x: round(x), np.interp(A.index.astype(float),A[A["clusters"]>1].index.astype(float),A.clusters[A.clusters>1])))
                #             #df.LABEL=A.rename(index=str,columns={"clusters":"LABEL"}).LABEL
                #             df.index=df.index.astype(str)
                #             A.index=A.index.astype(str)
                #             df["LABEL"]=np.nan
                #             # df.update(A.rename(index=str,columns={"clusters":"LABEL"})[["LABEL"]])

                #             # Assign nulls between clusters assignments to new null sets so nulls group instead of a single null Label
                #             # step = df.DEPTH[10]-df.DEPTH[9]
                #             #step=round(scipy.stats.mstats.gmean(df.index.astype(float).to_series().diff()[df.index.astype(float).to_series().diff()!=0].dropna()),1)
                #             step=las.well.STEP.value
                #             # if max(df.LABEL.dropna())>0:
                #             #     count=-2
                #             #     while (df.loc[df.LABEL.isnull(),'LABEL'].shape[0])>1:
                #             #         startnull = min(df.loc[df.LABEL.isnull(),'DEPTH'])
                #             #         try: endnull = min(df.loc[df.DEPTH>startnull,'LABEL'].dropna().index.astype(float))-step
                #             #         except: endnull = max(df.DEPTH)
                #             #         #          min(df.loc[df.index.astype(float)>(min(df.LABEL.isnull().index.astype(float))),'LABEL'].dropna().index.astype(float))-step
                #             #         df.loc[((df.DEPTH>=startnull)&(df.DEPTH<=endnull)),'LABEL']=count
                #             #         count-=1
                #             # else: df.LABEL=-100

                #                 #df.loc[((df.DEPTH>=startnull)&(df.DEPTH<=endnull)),'LABEL']
                #                 #df.loc[df.index.astype(float)>(min(df.LABEL.isnull().index.astype(float))),'LABEL'].dropna()
                #             #if max(df.LABEL[:firstnull])>0:
                #             #    df.loc[firstnull:,'LABEL']+=1
                #             #    df.loc[firstnull,'LABEL']=min(df.LABEL[firstnull:])-1
                #             #else: df.loc[firstnull,'LABEL']=0

                #             # def_clusters=dict(
                #             #     zip(
                #             #         list(
                #             #             #set(df.LABEL[df.LABEL>=0])
                #             #             set(df.LABEL)
                #             #             ),stats.rankdata(
                #             #                 list(map(
                #             #                     lambda x: min(df[df["LABEL"]==x].index.astype(float)
                #             #                                   ),list(set(df.LABEL))
                #             #                     ))
                #             #             ).astype("int")
                #             #         )
                #             #     )
                #             # df["LABEL"].replace(def_clusters,inplace=True)

                #             #df.LABEL=list(map(lambda x: round(x).astype(int),np.interp(df[df.isnull()].index.astype(float),df[df["LABEL"]>=0].index.astype(float),df.LABEL[df.LABEL>=0])))
                #             #df.LABEL=1
                #             ############
                #             # BAD HOLE #
                #             ############
                #             df['BADHOLE']=np.nan
                #             if (Alias['DCORR'] != 'NULL'):
                #                 df.loc[np.absolute(df[Alias['DCORR']])>0.15,'BADHOLE']=1

                #             # ###############################
                #             # # Assign R0 & SW per interval #
                #             # ###############################
                #             df["R0"]=np.nan
                #             df["SW_N"]=np.nan
                #             # N=np.nan
                #             # df.index=df.index.astype(str)
                #             # for i in df.LABEL.unique():
                #             #     if len(df.loc[(df.LABEL==i)&(df.BADHOLE!=1),[Alias["NPHI"],Alias["RDEEP"]]].dropna())>10:
                #             #         try: x=R0_DLOGN(df[[Alias["NPHI"],Alias["RDEEP"]]][(df.LABEL==i)&(df.BADHOLE!=1)].dropna(),Puwi,2,i);
                #             #         except: continue;
                #             #         if len(x)>0:
                #             #             x.index=x.index.astype(str)
                #             #             df.update(x)
                #             #             del x
                #             #             # GET N
                #             #             #dfsub=df.loc[df.LABEL==i,[Alias['DEN'],Alias['NPHI'],Alias['RDEEP'],'R0']].dropna()
                #             #             #pd.concat([(2.69-df.loc[df.LABEL==i,Alias['DEN']])/1.69,df.loc[df.LABEL==i,"R0"]/df.loc[df.LABEL==i,Alias['RDEEP']]],axis=1).dropna()
                #             #             data=((2.69-df.loc[df.LABEL==i,Alias['DEN']])/1.69)*(df.loc[df.LABEL==i,"R0"]/df.loc[df.LABEL==i,Alias['RDEEP']]).dropna()
                #             #             data=(np.log10((2.69-df.loc[df.LABEL==i,Alias['DEN']])/1.69)*np.log10(df.loc[df.LABEL==i,"R0"]/df.loc[df.LABEL==i,Alias['RDEEP']])).dropna()

                #             #             #normrange(XX.iloc
                #             #             data=data.loc[data>0]

                #             #             #data=np.log10(data)
                #             #             data=data[np.absolute(data)<1000].dropna()

                #             #             if len(data)>10:
                #             #                 #print(data)
                #             #                 param = stats.gamma.fit(data)
                #             #                 limit=10**stats.gamma.ppf(0.4,*param) # probabilistic value from bin
                #             #                 #df.loc[((data.loc[:,0]*data.iloc[:,1])<limit)]
                #             #                 pca=PCA()
                #             #                 filterlist=((df.LABEL==i) &
                #             #                             (np.log10((2.69-df[Alias['DEN']])/1.69*np.log10(df['R0'])/df[Alias['RDEEP']])<limit) &
                #             #                             (df.BADHOLE!=1))
                #             #                 if len(filterlist==True)<20: continue
                #             #                 try: pca.fit(np.log10(pd.concat([(2.69-df.loc[filterlist,Alias['DEN']])/1.69,
                #             #                           df.loc[filterlist,'R0']/df.loc[filterlist,Alias['RDEEP']]],
                #             #                           axis=1).clip(0.00001,100)).dropna())
                #             #                 #try: pca.fit(np.log10(pd.concat([(2.69-df.loc[((dfx.iloc[:,0]*dfx.iloc[:,1])<limit) &(df.LABEL==i)&(df.BADHOLE!=1),Alias['DEN']])/1.69,df.loc[((dfx.iloc[:,0]*dfx.iloc[:,1])<limit) &(df.LABEL==i)&(df.BADHOLE!=1),"R0"]/df.loc[((dfx.iloc[:,0]*dfx.iloc[:,1])<limit) &(df.LABEL==i)&(df.BADHOLE!=1),Alias['RDEEP']]],axis=1).clip(0.00001,100)).dropna())
                #             #                 except: continue
                #             #                 N=-1/min(pca.components_[:,1]/pca.components_[:,0]) # positive eigenvector slope
                #             #                 #if Npca>0df:
                #             #                 #    df.loc[df.LABEL==i,'SWpca']=((df.loc[df.LABEL==i,'R0']/df.loc[df.LABEL==i,Alias['RDEEP']]).clip(0,10)**(1/Npca)).clip(lower=0,upper=1)
                #             #                 #N=2
                #             #                 if (N>0.4) and (N<25):
                #             #                     df.loc[df.LABEL==i,'SW_N']=((df.loc[df.LABEL==i,'R0']/df.loc[df.LABEL==i,Alias['RDEEP']]).clip(0,10)**(1/N)).clip(lower=0,upper=1)
                #             #                 else: continue
                #             # for i in df.LABEL.unique():
                #             #     if ((df.loc[df.LABEL==i,'R0']/df.loc[df.LABEL==i,Alias['RDEEP']])**0.5).dropna().quantile(q=0.9) < 0.5:
                #             #         df.loc[df.LABEL==i,'R0']=None
                           
                df['R0'] = Find_R0(df)
                df['SW']=(df['R0']/df[Alias['RDEEP']])**0.5
                #####################
                # Create Export LAS #
                #####################
                # Created file at function beginning

                # Set Header
                exlas.well.STRT.value=df.DEPTH.min()
                exlas.well.STOP.value=df.DEPTH.max()
                exlas.well.STEP.value=step
                exlas.well.Date=str(datetime.datetime.today())
                exlas.well["INTP"]=lasio.HeaderItem(mnemonic="INTP", value="William Rucker", descr="Analyst for equations and final logs")
                exlas.well["UWI"]=lasio.HeaderItem(mnemonic="UWI", value=str(uwi).zfill(14), descr="Unique well identifier")
                try: exlas.well["APIN"].value=str(uwi).zfill(14)
                except: pass
                try:
                    exlas.well["API"].value=str(Puwi).zfill(14)
                    las.well["UWI"]=lasio.HeaderItem(mnemonic="UWI", value=str(Puwi).zfill(14), descr="Unique well identifier")
                except: pass

                #str(las.well["UWI"].value).zfill(14)

                exlas.append_curve('DEPT',df.DEPTH , unit='ft')

                ##############
                # CLAY MODEL #
                ##############
                df["ND_DIFF"]=(df[Alias["NPHI"]]-(2.69-df[Alias["DEN"]])/1.69)
                df["WKR_VCLAY"]=(-1.59488+234.513*(df[Alias["NPHI"]]-(2.69-df[Alias["DEN"]])/1.69))/100

                ##################
                # Resource Model #
                ##################
                #df["WKR_RHOFL"]=0.7*(1-df["SW"])+1.05*(df["SW"]);df["WKR_DPHI_269"]=(2.69-df[Alias["DEN"]])/(2.69-df["WKR_RHOFL"]);df["WKR_BVW_269"]=(df["SW"])*df["WKR_DPHI_269"]
                df["WKR_RHOFL"]=0.7*(1-df.SW.clip(lower=0,upper=1))+1.05*(df["SW"])
                df["WKR_DPHI_269"]=(2.69-df[Alias["DEN"]])/(2.69-df["WKR_RHOFL"])
                df["WKR_DPHI_265"]=(2.65-df[Alias["DEN"]])/(2.65-df["WKR_RHOFL"])
                df["WKR_DPHI_271"]=(2.71-df[Alias["DEN"]])/(2.71-df["WKR_RHOFL"])
                df["WKR_DLOGRN"]=np.log10(df.loc[df['R0']>0,Alias['RDEEP']]/df.loc[df['R0']>0,'R0'])
                df["WKR_BVW_269"]=(df["SW"])*df["WKR_DPHI_269"]
                df["WKR_BVH_269"]=(1-df.SW.clip(lower=0,upper=1))*df["WKR_DPHI_269"]
                df['WKR_Kirr']=(250*df.loc[df["WKR_DPHI_269"]>0,"SW"]**3)/(df["SW"].clip(lower=0,upper=1))**2
                #    df["WKR_PHI_INV"]=df["Sw"]-((df["R0"]/df[Alias["RSHAL"]])**0.5)*df["WKR_Dphi_269"]

                # for i in df.LABEL.unique():
                #     N=(np.log10(df.loc[(df.LABEL==i) & (df.SW_N>0) & (df.SW_N<1),'R0']/df.loc[(df.LABEL==i) & (df.SW_N>0) & (df.SW<1),Alias['RDEEP']])/np.log10(df.loc[(df.LABEL==i) & (df.SW_N>0) & (df.SW_N<1),'SW'])).mean(axis=0)
                #     cmap = cm.get_cmap('bwr');
                #     normalize = colors.Normalize(vmin=0, vmax=1);
                #     swcolors = [cmap(normalize(value)) for value in (df.loc[df.LABEL==i,'SW_N'].dropna().clip(0,1))];
                #     cmap=cm.get_cmap('gist_rainbow')
                #     try:
                #         normalize = colors.Normalize(vmin=min(df.loc[(df.LABEL==i) & (df.SW_N>=0),'WKR_BVW_269'].dropna()),
                #                                             vmax=max(df.loc[(df.LABEL==i) & (df.SW_N>=0),'WKR_BVW_269'].dropna()))
                #         bvwcolors = [cmap(normalize(value)) for value in (df.loc[df.LABEL==i,'WKR_BVW_269'].dropna().clip(0,1))];
                #     except:
                #         print(str(i) + ' failed')
                #         continue


                #     fig, ax = plt.subplots(figsize=(12,7));

                #     plt.suptitle('UWI: '+str(Puwi)+ '\n N= ' + str(N))
                #     ax1=plt.subplot(121)
                #     ax2=plt.subplot(122)
                #     ax1.scatter(df.loc[(df.LABEL==i) & (df.SW_N>=0),Alias['RDEEP']].clip(0.001,100),
                #                     df.loc[(df.LABEL==i) & (df.SW_N>=0),Alias['NPHI']].clip(0.001,100),
                #                     edgecolors='none',
                #                     s=50,
                #                     color=swcolors);
                #     ax1.set(ylabel=str(Alias['NPHI'])+' [v/v]',
                #                     xlabel= str(Alias['RDEEP']) + ' [Ohmm]',
                #                     title= 'Pickett',
                #                     #xlim=(1,20),
                #                     ylim=(0.01,0.5),
                #                     yscale=('log'),
                #                     xscale=('log'))

                #     ax2.scatter(df.loc[(df.LABEL==i) & (df.SW_N>=0),'R0'].clip(0.001,100)/df.loc[(df.LABEL==i) & (df.SW_N>=0),Alias['RDEEP']],
                #                     df.loc[(df.LABEL==i) & (df.SW_N>=0),'WKR_DPHI_269'].clip(0.001,1),
                #                     edgecolors='none',
                #                     s=50,
                #                     color=bvwcolors);
                #     np.array
                #     ax2.plot(np.array([0.001,0.1,2,5]),10**(-1/N*(np.log10(np.array([0.001,0.1,2,5]))+10**(1-0.05/N))),'k-')
                #     ax2.set(ylabel=str('DPHI_269')+' [v/v]',
                #                     xlabel= 'R0/RT',
                #                     title= 'Buckles',
                #                     xlim=(0.001,100),
                #                     ylim=(0.01,0.5),
                #                     yscale=('log'),
                #                     xscale=('log'))
                #     plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
                #     fig.savefig(str(dir_add)+"\\_"+str(Puwi)+"+"+str(min(df.loc[df.LABEL==i,'DEPTH']))+"_"+str(max(df.loc[df.LABEL==i,'DEPTH']))+"_Pickett.png")
                #     try: plt.close()
                #     except: pass
                #     try: fig.close()
                #     except: pass

                #################
                # Mineral Model #
                #################
                if Alias["PE"] != "NULL":
                    sw=df["SW"].clip(0,1)
                    sw.loc[sw.isna()]=1
                    dphi=pd.DataFrame(df["WKR_DPHI_269"])
                    dphi.loc[dphi.WKR_DPHI_269.isna(),"WKR_DPHI_269"]=(2.69-df.loc[df.WKR_DPHI_269.isna(),Alias["DEN"]])/1.69
                    df['U_APPX']=df[Alias["PE"]]*df[Alias["DEN"]]
                    df["WKR_UMAA"]=(df['U_APPX'] - (1-sw)*dphi.WKR_DPHI_269*0.136 - (sw)*dphi.WKR_DPHI_269*0.8)/(1-dphi.WKR_DPHI_269)
                    #Vss=(las['U_APPX']-Vclay*8-13.8+13.8*Vclay)/(4.79-1)
                    #Vls=1-Vclay-Vss
                    #Mlog = (las['RHOZ']/1000*(304800/las['DTCO'])**2)/1000
                    #Mmod=120*Vls+60*Vss+Vclay*25
                    exlas.append_curve('U_APPX', df.U_APPX, unit='barns/e', descr='Simplified Matrix Cross Section PE*RHOB')
                    exlas.append_curve('WKR_UMAA', df.WKR_UMAA, unit='barns/e', descr='Apparent Matrix Cross Section')

                # set curves
                #exlas.append_curve('RHOB',df[Alias["DEN"]], unit='g/cc', descr='Density used for calculation')
                #exlas.append_curve('NPHI',df[Alias["NPHI"]], unit='v/v', descr='Nphi used for calculation')
                #exlas.append_curve('RDEEP',df[Alias["RDEEP"]], unit='ohm-m', descr='RDEEP used for calculation')

                # exlas.append_curve('WKR_LABEL',df.LABEL, unit='Int', descr='Clustering Group')
                exlas.append_curve('WKR_R0', df.R0, unit='Ohm-m', descr='Resistivity at Sw=1 from Nphi Crossplot')
                exlas.append_curve('WKR_DLOGRN', df.WKR_DLOGRN, unit='Ohm-m', descr='Log10 of excess resistivity by neutron method')
                exlas.append_curve('WKR_SW', df.SW, unit='v/v', descr='Water Saturation from N=2 & WKR_R0 for total porosity')
                # exlas.append_curve('WKR_SW_N', df.SW_N, unit='v/v', descr='Water Saturation from PCA & WKR_R0 for total porosity')
                exlas.append_curve('WKR_RHOFL', df.WKR_RHOFL, unit='g/cc', descr='Fluid density using WKR_Sw')
                exlas.append_curve('WKR_DPHI_269', df.WKR_DPHI_269, unit='v/v', descr='Porosity using 2.69g/cc matrix and WKR_Rhofl')
                exlas.append_curve('WKR_DPHI_265', df.WKR_DPHI_265, unit='v/v', descr='Porosity using 2.65g/cc matrix and WKR_Rhofl')
                exlas.append_curve('WKR_DPHI_271', df.WKR_DPHI_271, unit='v/v', descr='Porosity using 2.71g/cc matrix and WKR_Rhofl')
                exlas.append_curve('WKR_BVW_269', df.WKR_BVW_269, unit='v/v', descr='Bulk volume water as Sw*WKR_DPHI_269')
                exlas.append_curve('WKR_BVH_269', df.WKR_BVH_269, unit='v/v', descr='Bulk volume hydrocarbon as (1-Sw)*WKR_DPHI_269')
                exlas.append_curve('WKR_ND_DIFF', df.ND_DIFF, unit='v/v', descr='Porosity separation between NPHI and DPHI_2.69')
                exlas.append_curve('WKR_Kirr', df.WKR_Kirr, unit='mD', descr='Perm assuming Sw=Swirr and N=2')
                exlas.append_curve('WKR_VCLAY', df.WKR_VCLAY, unit='v/v', descr='Clay volume by Neutron-Density')
                filename = str(dir_add)+"\\"+str(Puwi)+"_WKR_DLOGR.las"
                #if path.isfile(filename):
                #    remove(filename)
                exlas.write(filename, version = 2.0)
            else: 0
    else: exlas="FALSE"

    return exlas           

def Mechanics(lasfile):
    exlas=lasio.LASFile()
    dir_add = path.join(getcwd(),'MECH')
    if not path.exists(dir_add):
        mkdir(dir_add)
                   
    try: las=lasio.read(lasfile)
    except: las=[[0]]
    Alias=GetAlias(las)
    if (len(las[0])>100) and (Alias["DTC"]!="NULL") and (Alias["DEN"]!="NULL"):
        df=las.df()
        df['Depth'] = df.index       
        df["Vp"]=304800/df[Alias["DTC"]]
        df["Zp"]=df[Alias["DEN"]]*df["Vp"]/1000
        df["UCS_WFD"]=150.79*(304.8*df[Alias["DTC"]])**3.5
        df["VpMod"] = 1000*df[Alias["DEN"]]*(df["Vp"]**2)*(10**(-9))
        
        if  (Alias["DTS"]!="NULL"):
            df["Vs"]=304800/df[Alias["DTS"]]
            df["Zs"]=df[Alias["DEN"]]*df["Vs"]/1000
            df["Lame1"]=1000*df[Alias["DEN"]]*(df["Vp"]**2-2*df["Vs"]**2)*(10**(-9))
            df["ShearMod"]=1000*df[Alias["DEN"]]*(df["Vs"]**2)*10**(-9)
            df["E_Youngs"]=1000*df[Alias["DEN"]]*(df["Vs"]**2)*(3*df["Vp"]**2-4*df["Vs"]**2)/(df["Vp"]**2-df["Vs"]**2)*10**(-9)
            df["K_Bulk"]=1000*df[Alias["DEN"]]*(df["Vp"]**2-4/3*df["Vs"]**2)*10**(-9)
            df["Poisson"]=((df["Vp"]/df["Vs"])**2-2)/(2*(df["Vp"]/df["Vs"])**2-2)
            df["RhoLambda"]=df["Lame1"]*df[Alias["DEN"]]/1000*10**9
            df["MuRho"]=df["ShearMod"]*df[Alias["DEN"]]/1000*10**9
            df["LambdaMu"]=df["Lame1"]*df["ShearMod"]*10**18
            df["LambdaRho"]=df["Lame1"]*df[Alias["DEN"]]/1000*10**9

        
        # INITIALIZE EXPORT LAS
        exlas.well=las.well
        exlas.well.Date=str(datetime.datetime.today())
        exlas.well["INTP"]=lasio.HeaderItem(mnemonic="INTP", value="William Rucker", descr="Analyst for equations and final logs")
        exlas.well["UWI"].value=str(las.well["UWI"].value).zfill(14)
        exlas.well["APIN"].value=str(las.well["APIN"].value).zfill(14)
        exlas.append_curve('DEPT',df.Depth , unit='ft')

        # POPULATE EXPORT LAS
        exlas.append_curve('WKR_Vp',df.Vp, unit='m/s', descr='Metric P Wave Velocity')
        exlas.append_curve('WKR_VpMod',df.VpMod, unit='GPa', descr='Metric Compression Modulus')
        exlas.append_curve('WKR_UCS_WFD',df.UCS_WFD, unit='MPa', descr='Weatherford UCS model from DTC')

        if  (Alias["DTS"]!="NULL"):
            exlas.append_curve('WKR_Vs',df.Vs, unit='m/s', descr='Metric S Wave Velocity')
            exlas.append_curve('WKR_ShearMod',df.ShearMod, unit='GPa', descr='Metric Shear Modulus')
            exlas.append_curve('WKR_E_Youngs',df.E_Youngs, unit='GPa', descr='Metric Youngs Modulus')
            exlas.append_curve('WKR_K_Bulk',df.K_Bulk, unit='GPa', descr='Metric Bulk Modulus')
            exlas.append_curve('WKR_Poisson',df.Poisson, unit='None', descr='Poissons Ratio')
            exlas.append_curve('WKR_MuRho',df.MuRho, unit='GPa*Kg/m3', descr='Metric Lame Mu * Den')
            exlas.append_curve('WKR_LambdaMu',df.LambdaMu, unit='GPa*Gpa', descr='Metric Lame Mu * Lame Lambda')
            exlas.append_curve('WKR_LambdaRho',df.LambdaRho, unit='GPa*Kg/m3', descr='Metric Lame Lambda * Den')    

        filename = str(dir_add)+"\\"+str(exlas.well.uwi.value)+"_WKR_MECH.las"
        exlas.write(filename, version = 2.0)
    else: exlas=False
    return exlas

def EatonPP(lasfile,ROLLINGWINDOW = 200, QUANTILE = 0.5, EATON_EXP = 2.5, PLOTS = False, DEGREE_VP = 1, DEGREE_MOD = 1, MINIMUM_PTS = 100):
    exlas=lasio.LASFile()
    dir_add = path.join(getcwd(),'EATON')
    if not path.exists(dir_add):
        mkdir(dir_add)    
    
    try: las=lasio.read(lasfile)
    except: las=[[0]]
    Alias=GetAlias(las)
    if (len(las[0])>(MINIMUM_PTS*3)) and (Alias["DTC"]!="NULL") and (Alias["DEN"]!="NULL") and (Alias["PE"]!="NULL") :
        df=las.df()
        df['Depth'] = df.index       
        df["Vp"]=304800/df[Alias["DTC"]]
        df["VpMod"] = 1000 * df[Alias["DEN"]] * (df["Vp"]**2) * (10**(-9))

        m = (df[Alias['DEN']]>1.7) * (df[Alias['DEN']]<3)
        df['RHOB2'] = np.nan
        df.loc[m,'RHOB2'] = df.loc[m,Alias['DEN']]
        m = (df['RHOB2'].isna()) * (df.index<1000)
        df.loc[m,'RHOB2'] = 1.7
        df['RHOB2'].interpolate(inplace=True)
        df['RHOB3'] = df['RHOB2'].ewm(span=10).mean()
        
        df['DPHI269'] = (2.69-df[Alias['DEN']])/1.69
        df['PHYD'] = df.TVD*0.433
        df['OVERBURDEN'] = (df.RHOB3 * df.TVD.diff()).cumsum() * 30.48 / 70.3070

        df["Vp"].interpolate(inplace=True) 
        df['VP_200'] = df['Vp'].rolling(ROLLINGWINDOW).quantile(QUANTILE)
        df['VpMod'].interpolate(inplace=True)
        VPMODMAX = df['VpMod'].max()*1.2

        df['VpMod_Trends'] = np.nan     
        df2 = pd.DataFrame()
        ct = -1
               
        if 'WKR_UMAA' in df.keys():
           U_KEY = 'WKR_UMAA'
        else:
           df['U_APPX'] = df[[Alias["DEN"],Alias["PE"]]].prod(axis=1, skipna=False).dropna()
           U_KEY = 'U_APPX'

        m2 = df['VpMod'].dropna().index
        for i in np.arange(4,12,1):
            m = df.index[(df[U_KEY]> i)*(df[U_KEY]<(0.5+i))]
            if len(m.intersect(m2))>MINIMUM_PTS:
                ct += 1
                mod = detrend_log(df,'Depth','VpMod', True, m, log= True, fit_deg = DEGREE_VP)
                df2.at[ct,'U'] = df.loc[m,U_KEY].mean()
                for j in np.arange(0,DEGREE_VP+1):
                    df2.at[ct,f'mod{j}'] = mod[j]
                df['TEST'] = 10**df['Depth'].apply(lambda x: mod(x))
                try:                  
                    df.loc[m,'VpMod_Trends'] = decompose_log(df.loc[m,'VpMod'])
                except:
                    pass
                       
        if df2.shape[0]<4:
            print('Too few passable points')
            return False         
                  
        mod = []
        for j in np.arange(0,DEGREE_VP+1):
            mod.append(detrend_log(df2,'U',f'mod{j}',return_model = True, log = False, fit_deg = DEGREE_MOD))
            detrend_log(df2,'U',f'mod{j}',return_model = False, log = False, fit_deg = DEGREE_MOD)
            df[f'mod{j}'] = df[U_KEY].apply(lambda x: mod[j](x))
                   
            if PLOTS:
                fig, ax = plt.subplots()
                ax.scatter(df2['U'], df2[f'mod{j}'], label = f'Model param {j}', color = 'saddlebrown')
                ax.scatter(df2['U'], df2[f'mod{j}_TREND'], label = f'Model param {j} Trend', color = 'dodgerblue')
                ax.set_xlim([0,20])
                ax.legend()
                ax.invert_yaxis()
                plt.show()
                       
        KEYS = [f'mod{j}' for j in np.arange(0,DEGREE_VP+1)]
        KEYS.append('Depth')
               
        df['VpMod_NPT'] = 10**df[KEYS].apply(lambda x: np.poly1d([x[j] for j in np.arange(0,DEGREE_VP+1)[::-1]])(x[-1]), axis =1).dropna()
        df['Vp_NPT'] = (df['VpMod_NPT']/df['RHOB2']/1000/(10**(-9)))**0.5
        df['Eaton_VpMod'] = (df['OVERBURDEN'] - (df['OVERBURDEN']-df['PHYD']))*(df['VP_200']/df['Vp_NPT'])**EATON_EXP

        # Mud Weight Scales
        df['OVERBURDEN_MW'] = df.OVERBURDEN/df.TVD/0.05194805
        df['PHYD_MW'] = df.PHYD/df.TVD/0.05194805
        df['Eaton_VpMod_Mw'] = df.Eaton_VpMod/df.TVD/0.05194805
        
        # INITIALIZE EXPORT LAS
        exlas.well=las.well
        exlas.well.Date=str(datetime.datetime.today())
        exlas.well["INTP"]=lasio.HeaderItem(mnemonic="INTP", value="William Rucker", descr="Analyst for equations and final logs")
        exlas.well["UWI"].value=str(las.well["UWI"].value).zfill(14)
        exlas.well["APIN"].value=str(las.well["APIN"].value).zfill(14)
    
        exlas.append_curve('DEPT',df.Depth , unit='ft')    

        # POPULATE EXPORT LAS
        exlas.append_curve('WKR_VpMod_NCT',df.VpMod_NPT, unit='GPa', descr='Metric Compression Modulus Normal Compaction Trend')
        exlas.append_curve('WKR_VpMod',df.VpMod, unit='GPa', descr='Metric Compression Modulus')
        exlas.append_curve('WKR_PHYD',df.PHYD, unit='psi', descr='Hydrostatic pressure')
        exlas.append_curve('WKR_PLITH',df.OVERBURDEN, unit='psi', descr='Lithostatic pressure')
        exlas.append_curve('WKR_PPEM',df.Eaton_VpMod, unit='psi', descr='Eaton Pore Pressure using VpMod NPT')    
        exlas.append_curve('WKR_PGHYD',df.PHYD_MW, unit='ppg', descr='Hydrostatic pressure gradient')
        exlas.append_curve('WKR_PGLITH',df.OVERBURDEN_MW, unit='ppg', descr='Lithostatic pressure gradient')
        exlas.append_curve('WKR_PPGEM',df.Eaton_VpMod_Mw, unit='ppg', descr='Eaton Pore Pressure gradient using VpMod NPT')    
        
        filename = str(dir_add)+"\\"+str(exlas.well.uwi.value)+"_EATON.las"
        exlas.write(filename, version = 2.0)
               
        if PLOTS:
            fig, ax = plt.subplots()
            ax.scatter(df['Vp'], df['Depth'], label = 'Vp Data', color = 'saddlebrown')
            ax.scatter(df['VP_200'], df['Depth'], label = 'VP rolling', color = 'dodgerblue')
            ax.scatter(df['Vp_NPT'], df['Depth'], label = 'Vp_Model', linestyle = 'dashed', color = 'firebrick')
            ax.set_xlim([0,10000])
            ax.legend()
            ax.invert_yaxis()
            plt.show()

        if PLOTS:
            fig, ax = plt.subplots()
            ax.scatter(df['OVERBURDEN_MW'], df['Depth'], label = 'OVERBURDEN', color = 'saddlebrown')
            ax.scatter(df['PHYD_MW'], df['Depth'], label = 'HYDROSTATIC', color = 'dodgerblue')
            ax.scatter(df['Eaton_VpMod_Mw'], df['Depth'], label = 'EST PORE PRESSURE (EATON)', linestyle = 'dashed', color = 'firebrick')
            ax.set_xlim([0,30])
            ax.legend()
            ax.invert_yaxis()
            plt.show()
           
    else: exlas=False
    return exlas
