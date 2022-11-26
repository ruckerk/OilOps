import sys, os.path, math, datetime, re
import pandas as pd
import numpy as np
from os import path, listdir, remove, makedirs
from scipy.optimize import fmin_cobyla
from sklearn.decomposition import PCA
from scipy.stats.mstats import gmean
import statistics
import multiprocessing
from functools import partial
import psutil, os
from math import ceil
import concurrent.futures
import sqlite3


## ERRORS TO DEBUG
# left right sign is inconsistent, possibly using orientation of offset rather than parent
# v118 fix issue with file: AZI decimal filter, and FILEDATE

AAA = pd.read_csv('/home/ruckerwk/Programming/rseg_well_completion-275434.csv')
################################
# REVISIONS FOR 3D TVD VERSION #
################################
## 1) UWI/Comp_Date table to df1
## 2) Read Petra surveys file to df2
## 3) to df2 add first XYZ for DIP>85
## 4) to df2 add XYZ at last MD
## 5) for each well in list
##     - filter API list to wells completed before 1yr after well of interest
##     - filter for XYZ within 30000 ft
##     - execute spatial function
## def spatial function
##     input two data series
##     data series A is reference well (well of interest in spacing loop)
##     data series B is well to compute (ith well to become a spacing point in spacing loop)
##     transform series so Y is Landing-BHL
##     clip B[Y] range overlapping A[Y]
##     interpolate both wells as Rbf (is this actually helpful?)
##     for pt in B, get shortest distance to A as vector (dX,dY,dZ)
##     add column XY_dist=(dX^2+dY)^0.5
##     add column XYZ_dist=(dX^2+dY^2+dZ^2)^0.5
##     return averages from (XY_dist,XYZ_dist,dZ) with a sign that is meaningful and consistent (+E/N/Shallow)

##if not sys.warnoptions:
##    import os, warnings
##    warnings.simplefilter("default") # Change the filter in this process
##    os.environ["PYTHONWARNINGS"] = "default" # Also affect subprocesses

#max(xdf.UWI10.isin([well]))    


#################
## SUBROUTINES ##
#################

def convert_to_list(x):
    try: x=x.tolist()
    except: pass
    if isinstance(x,(np.int,str)):
        x=[x]
    if isinstance(x,(np.ndarray,pd.DataFrame,pd.Series)):
        x=list(x)
    if not isinstance(x,list):
        x=list(x)
        x2 = []
        for i in x:
            if isinstance(i,list): x2.extend(flatten(i))
            else: x2.append(i)
        x=x2
    return(x)

def objective(X,P):
    x,y = X
    return np.sqrt((x - P[0])**2 + (y - P[1])**2)

def c1(X):
    x,y = X
    return f(x) - y

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

def str2num_noalpha(str_in):
    if str_in is None:
        return None
    str_in = str(str_in)
    #regexp = re.compile(r'[a-z]')
    #if regexp.search(str_in,re.IGNORECASE):
    return None if re.search(r'[a-z]',str_in,re.IGNORECASE) else (str2num(str_in))

def API2INT(val_in,length = 10):
    if val_in is None:
        return None
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
    lst = re.findall(r'[0-9]{9,}',ffile)
    return API2INT(lst[0],length=UWIlen) if len(lst)>0 else None

def SurveyCols(df_in):
    sterms = {'MD':r'.*MEASURED.*DEPTH.*|.*MD.*|.*^(?!t).*^(?!v).*Depth.*',
             'INC':r'.*INC.*|.*DIP.*',
             'AZI':r'.*AZI.*|.*AZM.*',
             'TVD':r'.*TVD.*|.*TRUE.*DEPTH.*|.*VERTICAL.*DEPTH.*',
             'NORTH_Y':r'.*(?:ORTH.*^(?!ING)|\+N|NS.*FT|N/S|N\+).*|(?:^ns$)|.*(?:ORTH.*)',
             'EAST_X':r'.*(?:EAST.*^(?!ING)|\+E|EW.*FT|E/W|E\+).*|(?:^ew$)|.*(?:EAST.*)'
        }

    if df_in.keys().str.contains(r'XPATH|EAST_X_XX',regex=True,case=False,na=False).max():
        sterms['NORTH_Y_XX'] = r'YPATH|NORTH_Y_XX'
        sterms['EAST_X_XX'] = r'XPATH|EAST_X_XX'

    if isinstance(df_in,pd.Series):
        df_in=list(df_in)
    for s in sterms:
        #print(sterms[s])
        if isinstance(df_in,pd.DataFrame):
            sterms[s]=df_in.iloc[0,df_in.keys().str.contains(sterms[s], regex=True, case=False,na=False)].keys()[0]
        if isinstance(df_in,list):
            sterms[s]= list(filter(re.compile('(?i)'+sterms[s]).match,df_in))[0]

    # sterms=dict((v, k) for k, v in sterms.iteritems())
    sterms = {v: k for k, v in sterms.items()}
     
    return sterms

def UWI10_(num):
    try:
        num=int(num)
        #out = pd.to_numeric(pd.Series(lst))
        while num > 9e9:
            num = math.floor(num/100)
        num = int(num)
    except:
        num = np.nan
    return(num)
    
    
def CondenseSurvey(xdf,LIST_IN):
    # if 1==1:
    if isinstance(LIST_IN,(pd.Series,np.ndarray)):
        UWIs=list(LIST_IN)
    if isinstance(LIST_IN,(str,int,np.uint,np.int64)):
        UWIs=[LIST_IN]
    if isinstance(LIST_IN,list):
        UWIs=LIST_IN

    OUTPUT = []

    UWICOL = xdf.keys().get_loc(xdf.iloc[0,xdf.keys().str.contains('.*UWI.*|.*API.*', regex=True, case=False,na=False)].keys()[0])
    UWIKEY = xdf.keys()[UWICOL]

    # slice dataframe to subset for function
    xdf = xdf.loc[xdf[UWIKEY].isin(UWIs)].copy(deep=True)

    # format UWI's
    xdf[UWIKEY] = xdf[UWIKEY].apply(UWI10_)
    UWIs = list(map(UWI10_,UWIs))

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
    for ct, UWI in enumerate(UWIs, start=1):
        if math.floor(ct/10)==ct/10:
            print(ct,'/',tot,': ',UWI)
        # while df.loc[df.groupby('FILE').MD.apply(lambda x: x-np.floor(x))==0,:]  
        # filter to UWI of interest
        xxdf = xdf.copy(deep=True)
        xxdf = xxdf.loc[xxdf[UWIKEY] == UWI,:]
        ftest = list(xxdf.FILE.unique())

        #print(ftest)

        while xxdf.FILE.unique().shape[0] > 1:
            # drop duplicates   if 1==1:
            #xxdf = xxdf.loc[xxdf[SKEYS].drop_duplicates().index,:]

            # if MD is all integers then drop
            # df.groupby(FILE).MD.apply(lambda x:x-np.floor(x))
            # sdf = sdf.loc[sdf.groupby('FILE').MD.apply(lambda x:x-np.floor(x))>0]
            # HZ AZI filter
            if xxdf.loc[xxdf.INC > 85,:].shape[0]>5:
                #xxdf.loc[:,'AZI_DEC'] = xxdf.loc[:,'AZI'] - np.floor(xxdf.loc[:,'AZI'])

                #sdf['MD_DEC'] = sdf['MD'] - np.floor(sdf['MD'])
                ftest = xxdf.loc[xxdf.INC > 85,:].groupby('FILE')['AZI_DEC'].std()
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

                #ftest['FILEUWI']=ftest.FILEUWI.apply(APIfromFilename).apply(UWI10_)
                UWImask = (ftest.FILEUWI.apply(APIfromFilename).apply(UWI10_) )==UWI10_(UWI)
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


        OUTPUT = OUTPUT+list(xxdf.FILE.unique())
            
            #OUTPUT = pd.merge(OUTPUT,df,how='outer',on=None, left_index=False, right_index=False)
            #OUTPUT = pd.merge(OUTPUT,xxdf,how='outer',on=None, left_index=False, right_index=False)

    return OUTPUT

def Condense_Surveys(xdf):
    # xdf = pd.read_csv(FILE)
    # RESULT = pd.DataFrame(columns = xdf.columns.to_list())
    # if 1==1:
    UWICOL = xdf.keys().get_loc(xdf.iloc[0,xdf.keys().str.contains('.*UWI.*|.*API.*', regex=True, case=False,na=False)].keys()[0])
    UWIlist = list(xdf.iloc[:,UWICOL].unique())

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
        with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
            f = {executor.submit(func, a): a for a in data}
        #RESULT=pd.DataFrame()
        RESULT = []
        print('merging condense sessions')
        for i in f:
            #RESULT=pd.concat([RESULT,i.result()],axis=0,join='outer',ignore_index=True)
            RESULT = RESULT + i.result()
    else:
        RESULT=CondenseSurvey(xdf,UWIlist)
    #RESULT.to_csv(OUTFILE,index=False)
    return RESULT
          
# Define function for nearest neighbors
def XYZSpacing(xxdf,df_UWI,DATELIMIT,xxUWI10):
    # SURVEYS in xxdf
    # WELL DATA in df_UWI
    # if 1==1:
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
                'Days5':int()}
    
    OUTPUT = pd.DataFrame(col_type,index=[])

    COMPDATES = df_UWI.iloc[0,df_UWI.keys().str.contains('.*JOB.*DATE.*|STIM.*DATE[^0-9].*|.*COMP.*DATE.*', regex=True, case=False,na=False)].keys()
    df_UWI['MAX_COMPLETION_DATE'] = df_UWI[COMPDATES].fillna(datetime.datetime(1900,1,1)).max(axis=1)
    COMPDATEdfd = df_UWI.keys().get_loc('MAX_COMPLETION_DATE')

    xxdf = xxdf.rename(columns = SurveyCols(xxdf.head(5)))

    
    # MAKE KEY COLUMNS NUMERIC
    for k in SurveyCols(xxdf):
        xxdf[k]=pd.to_numeric(xxdf[k],errors='coerce')

    #print(list(xxdf.keys()))
    #print(SurveyCols(xxdf.head(5)))
    SCOLS = SurveyCols(xxdf.head(5))
    
    UWICOL      = xxdf.keys().get_loc(xxdf.iloc[0,xxdf.keys().str.contains('.*UWI.*', regex=True, case=False,na=False)].keys()[0])
    XPATH       = xxdf.keys().get_loc(list(SCOLS)[5])
    YPATH       = xxdf.keys().get_loc(list(SCOLS)[4])
    TVD         = xxdf.keys().get_loc(list(SCOLS)[3])
    DIP         = xxdf.keys().get_loc(list(SCOLS)[1])
    MD          = xxdf.keys().get_loc(list(SCOLS)[0])

    if 'NORTH_Y_XX' in list(SCOLS):
        XPATH       = xxdf.keys().get_loc(SCOLS['EAST_X_XX'])
        YPATH       = xxdf.keys().get_loc(SCOLS['NORTH_Y_XX'])
    
    XPATH_NAME  = xxdf.keys()[XPATH] #list(SurveyCols(xxdf.head(5)))[5]
    YPATH_NAME  = xxdf.keys()[YPATH] #list(SurveyCols(xxdf.head(5)))[4]
    
    MD_NAME = xxdf.keys()[MD]
    
    for xUWI10 in xxUWI10:
        # if 1==1:
        ix+=1
        xdf = xxdf.copy(deep=True)
        xdf = xdf.loc[xdf[SurveyCols(xdf)].dropna().index,:]
        #print(str(xxUWI10.index(xUWI10)),' / ',str(len(xxUWI10)),' ')
        if ix/10 == math.floor(ix/10):
            print(str(ix) + '/' + str(len(xxUWI10)))
        OUTPUT=OUTPUT.append(pd.Series(name=ix,dtype='int64'))

        xUWI10=UWI10_(xUWI10)

        # Check for lateral survey points for reference well
        if xdf.loc[(xdf['UWI10']==xUWI10) & (xdf['INC']>85),:].shape[0]<=5:
            continue
        
        # PCA is 2 vector components
        pca = PCA(n_components=2)
        # add comp date filter at step 1
        try: 
            datecondition=(df_UWI.loc[df_UWI['UWI10']==xUWI10][df_UWI.keys()[COMPDATEdfd]]+DATELIMIT).values[0]
        except:
            continue

        UWI10list=df_UWI[(df_UWI[df_UWI.keys()[COMPDATEdfd]])<=datecondition].UWI10
        # filter on dates
        xdf=xdf[xdf.UWI10.isin(UWI10list)]
        #isolate reference well
        refXYZ=xdf[xdf.keys()[[UWICOL,XPATH,YPATH,TVD,MD]]][xdf.UWI10==xUWI10]
        
        if refXYZ.shape[0]<5:
            continue
        #reference well TVD approximation
        # if 1==1:
        #refTVD = gmean(abs(xdf.iloc[:,TVD][xdf.UWI10==xUWI10]))*np.sign(statistics.mean(xdf.iloc[:,TVD][xdf.UWI10==xUWI10]))
        refTVD = statistics.mean(xdf.iloc[:,TVD][xdf.UWI10==xUWI10])
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
            
            for well in set(xdf.UWI10):
                j+=1
                overlap = max(xdf.Xfit[xdf.UWI10==well])-min(xdf.Xfit[xdf.UWI10==well])
                gmeandistance = gmean(abs(xdf.Yfit[xdf.UWI10==well]))*np.sign(statistics.mean(xdf.Yfit[xdf.UWI10==well]))
                #gmeandepth = gmean(abs(xdf.iloc[:,TVD][xdf.UWI10==well]))*np.sign(statistics.mean(xdf.iloc[:,TVD][xdf.UWI10==well]))-refTVD
                meandepth = statistics.mean(xdf.iloc[:,TVD][xdf.UWI10==well])-refTVD
                try:
                    deltadays =  np.timedelta64(refdate-(df_UWI[df_UWI['UWI10']==well][df_UWI.keys()[COMPDATEdfd]]).values[0],'D').astype(float)
                except: deltadays = None
                
                df_calc.loc[j,'UWI10']=well
                df_calc.loc[j,'overlap']=overlap
                df_calc.loc[j,'dxy']=gmeandistance
                df_calc.loc[j,'abs_dxy']=abs(gmeandistance)
                df_calc.loc[j,'dz']=meandepth
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
       
        OUTPUT.loc[ix,'UWI10']=xUWI10
        
        #calc lat len 
        OUTPUT.loc[ix,['LatLen']] =abs(RefXMax-RefXMin)    
        OUTPUT.loc[ix,['MeanTVD']]=refTVD
        OUTPUT.loc[ix,['MeanX']]  =statistics.mean(refXYZ[XPATH_NAME])
        OUTPUT.loc[ix,['MeanY']]  =statistics.mean(refXYZ[YPATH_NAME])
        OUTPUT.loc[ix,'MAX_MD']   = max(refXYZ[MD_NAME].dropna())

        if OUTPUT.shape[0]<1:
            OUTPUT=OUTPUT.append({'UWI10':xUWI10,
                                  'LatLen':abs(RefXMax-RefXMin),
                                  'MeanTVD':refTVD,
                                  'MeanX':statistics.mean(refXYZ[XPATH_NAME]),
                                  'MeanY':statistics.mean(refXYZ[YPATH_NAME])},
                                 ignore_index=True)
    outfile = 'XYZ_'+str(int(xxUWI10[0]))+'_'+str(int(xxUWI10[-1]))

    OUTPUT.dropna(axis = 0, how='all')
    OUTPUT = OUTPUT.drop_duplicates()
    
    OUTPUT.to_json(outfile+'.JSON')
    OUTPUT.to_parquet(outfile+'.PARQUET')
    return(OUTPUT)

                ################################################
                ##                SCRIPT BODY                 ##
                ################################################
#if __name__ == "__main__":
    #Get local path of .py file

if 1==1:
    #TC_FILTER = pd.read_csv('INSIDE_TC_AREA.csv')

    pathname = path.dirname(sys.argv[0])
    adir = path.abspath(pathname)

    #Set filenames to read in local directory
    #fname='Petra_HZ_Export_10072020.txt'
    #fname = '\\\\Server5\Verdad Resources\\Operations and Wells\\Geology and Geophysics\\WKR\\Decline_Parameters\\DeclineParameters_v200\\Compiled_Well_Data.csv'
  #  sqldb = '\\\\Server5\Verdad Resources\\Operations and Wells\\Geology and Geophysics\\WKR\\Decline_Parameters\\DeclineParameters_v200\\prod_data.db'   
    sqldb = path.join(path.dirname(adir), 'prod_data.db')
    sfile = 'JOINED_ABS_SURVEYS.csv'
    sfile = 'JOINED_ABS_SURVEYS_TC.csv'
    sfile = 'JOINED_ABS_SURVEYS_TC.json'
    sfile = 'JOINED_ABS_SURVEYS_TC.PARQUET'
    #sfile = 'PEGGY_ABS_SURVEYS2.csv'
    # set outfiles
    outfile = "3D_Spacing_"+datetime.datetime.now().strftime("%d%m%Y_%H%M")
    
    #ADD COMPLETION DATE OR 1ST PROD DATA FILTER FOR PARENT/CHILD
    DATELIMIT = datetime.timedelta(days=365)

    #Lateral section inclination limit(defines start of completable lateral)
    INC_LIMIT = 89
    
##    if not path.exists(surveyfile3):
##        # One Survey per UWI
##        if not path.exists(surveyfile2):
##            files = Condense_Surveys(surveyfile)
##        #else surveyfile2 = surveyfile
##        else:
##            sfile = surveyfile2
##    else:
##        sfile = surveyfile3

    #Main Script
    if path.exists(sfile):
        #df = pd.read_csv(sfile, header=0, na_filter=False, low_memory=False)
        #df = pd.read_json(sfile)
        df = pd.read_parquet(sfile)
        df = df.rename(columns = SurveyCols(df.head(2)))
        
        UWICOL =df.keys().get_loc(df.iloc[0,df.keys().str.contains('.*UWI.*', regex=True, case=False,na=False)].keys()[0])
        df.iloc[:,UWICOL].dropna().apply(API2INT)
        mm = df.iloc[:,UWICOL].fillna(1).apply(API2INT).isin(AAA.Unformatted_API_UWI.apply(API2INT))
        df = df.loc[mm,:]

        
        if df.groupby('UWI').FILE.nunique().max() > 1:
            print('Condense Surveys Start')
            #df2 =  df[df.FILE.isin(FLIST[5000:10000]])]
            FILE_LIST = Condense_Surveys(df)
            print('Condense Surveys End')
            df = df.loc[df.FILE.isin(FILE_LIST)].copy(deep=True)

        # if index saved to CSV, drop it if 1==1:
        df = df.drop(columns=re.findall(r'Unnamed.*',df.keys()[0]))
        UWICOL =df.keys().get_loc(df.iloc[0,df.keys().str.contains('.*UWI.*', regex=True, case=False,na=False)].keys()[0])
        UWIS = list(df.iloc[:,UWICOL].unique())
			
        #df = pd.read_csv(surveyfile,header=0,na_filter=False)0
        XPATH       = df.keys().get_loc((list(SurveyCols(df.head(5)))[5]))
        YPATH       = df.keys().get_loc((list(SurveyCols(df.head(5)))[4]))
        TVD         = df.keys().get_loc((list(SurveyCols(df.head(5)))[3]))
        DIP         = df.keys().get_loc((list(SurveyCols(df.head(5)))[1]))
        MD          = df.keys().get_loc((list(SurveyCols(df.head(5)))[0]))

        df['UWI10']=df.iloc[:,UWICOL].map(lambda x: (x/10000))
        m = df['UWI10'].isna()
        df.loc[~m,'UWI10'] = df.loc[~m,'UWI10'].apply(int)
        
        for k in SurveyCols(df):
            df[k] = pd.to_numeric(df[k],errors='coerce')
            
    if path.exists(sqldb):
        print('SQL connection')
        with sqlite3.connect(sqldb) as conn:
            df_UWI = pd.read_sql_query('SELECT * FROM WELL_SUMMARY',conn)
        
        #df_UWI         = pd.read_csv(fname, header=[0], na_filter=False ,low_memory=False)
        
        # if index saved to CSV, drop it
        #df_UWI = df_UWI.drop(columns=re.findall(r'Unnamed.*',df_UWI.keys()[0]))
        
        # MAKE ALL DATE COLUMNS INTO DATE TYPES
        for k in df_UWI.iloc[0,df_UWI.keys().str.contains('.*DATE.*', regex=True, case=False,na=False)].keys():
            df_UWI[k] = pd.to_datetime(df_UWI[k],errors = 'coerce',infer_datetime_format=True)

        # SUMMARIZE COMPLETION DATESif 1==1:
        COMPDATES = df_UWI.iloc[0,df_UWI.keys().str.contains('.*JOB.*DATE.*|STIMDATE[^0-9].*|.*COMP.*DATE.*', regex=True, case=False,na=False)].keys()
        df_UWI['MAX_COMPLETION_DATE'] = df_UWI[COMPDATES].max(axis=1)
        COMPDATEdfd = df_UWI.keys().get_loc('MAX_COMPLETION_DATE')

    ##    try:
    ##        COMPDATEdfd = df_UWI.keys().get_loc(df_UWI.iloc[0,df_UWI.keys().str.contains('.*COMP.*DATE.*', regex=True, case=False,na=False)].keys()[0])
    ##    except:
    ##        df_UWI = pd.read_csv(fname, header=[0,1], na_filter=False )
    ##        df_UWI.columns =df_UWI.columns.map('_'.join)
    ##        COMPDATEdfd = df_UWI.keys().get_loc(df_UWI.iloc[0,df_UWI.keys().str.contains('.*COMP.*DATE.*', regex=True, case=False,na=False)].keys()[0])
        KEYS = df_UWI.iloc[0,df_UWI.keys().str.contains('.*API|UWI.*', regex=True, case=False,na=False)].keys()
        KEYS = KEYS[df_UWI[KEYS].count()==df_UWI[KEYS].count().max()][0]
        
        UWICOL = df_UWI.keys().get_loc(KEYS)   #df_UWI.keys().get_loc(df_UWI.iloc[0,df_UWI.keys().str.contains('.*API|UWI.*', regex=True, case=False,na=False)].keys()[0])
        try:
            df_UWI = df_UWI[df_UWI.ziloc[:,UWICOL]>0]
        except:
            df_UWI = df_UWI[df_UWI.iloc[:,UWICOL]!=None]

        #df_UWI[df_UWI.keys()[COMPDATEdfd]]=df_UWI[df_UWI.keys()[COMPDATEdfd]].astype('datetime64[D]')
        df_UWI[df_UWI.keys()[COMPDATEdfd]]=pd.to_datetime(df_UWI[df_UWI.keys()[COMPDATEdfd]])
        #df_UWI[df_UWI.keys()[COMPDATEdfd]]=df_UWI[df_UWI.keys()[COMPDATEdfd]].dt.strftime('%m/%d/%Y')

        # Set Null completion dates to today
        df_UWI.loc[df_UWI.iloc[:,COMPDATEdfd].isnull(),df_UWI.keys()[COMPDATEdfd]] = pd.to_datetime(datetime.datetime.now().strftime('%m/%d/%Y'))

        #df_UWI[df_UWI.keys()[COMPDATEdfd]]=pd.to_datetime(df_UWI[df_UWI.keys()[COMPDATEdfd]])
        df_UWI['UWI10'] = df_UWI.iloc[:,UWICOL].apply(UWI10_)
    
    # add dates to survey dataframe
    if 1==1:
        df=pd.merge(df,df_UWI[[df_UWI.keys()[COMPDATEdfd],'UWI10']],how = 'left', on='UWI10')   ## THIS IS DROPPING WELLS!

        COMPDATE = df.keys().get_loc(df.iloc[0,df.keys().str.contains('.*MAX_COMPLETION_DATE.*', regex=True, case=False,na=False)].keys()[0])
        #df[df.keys()[COMPDATE]]=df[df.keys()[COMPDATE]].astype('datetime64[D]')

        # filter down to lateral only using INC_LIMIT
        df=df[df.iloc[:,DIP]>INC_LIMIT]

        #pd.concat([x for x in map(lambda x:XYZSpacing([x,df,df_UWI,DATELIMIT]),uwis[0:20])],ignore_index=True)
        s1 = pd.DataFrame()
        
    #df_UWI = df_UWI.loc[df_UWI.UWI10.isin(list(df.UWI.apply(UWI10_).unique())]
    #RESULT = XYZSpacing(df,df_UWI,DATELIMIT,[5123401790000,5123401700000,5123401780000, 512340173,5123401740000,512340177])

    #df.groupby(by='FILE').MD.max() - df.groupby(by='FILE').TVD.max()
    
    # TEST FUNCTION!!!
    if 1==1:
        print('Start XYZ function')
        func=partial(XYZSpacing,df,df_UWI,DATELIMIT)
        
        UWIlist = list(df.UWI10.drop_duplicates())


        # CHECK THIS IS WORKING RIGHT
        chunkmin = 500
        chunkmax = 1000
        
        batchmax = ceil(len(UWIlist)/chunkmax)
        batchmin = ceil(len(UWIlist)/chunkmin)
    
        processors = multiprocessing.cpu_count()-1
        processors = min(processors,batchmin)
        
        batches = max(min(batchmax,processors),batchmin)
        data = np.array_split(UWIlist,batches)

        #data = np.array_split(UWIlist,max(processors,ceil(batch/1.8)))

        
        
# VERY SLOW
# DEBUG
# TRY MAKING LARGE TABLE GLOBAL

        if processors > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
                f = {executor.submit(func, a): a for a in data}
            RESULT=pd.DataFrame()
##            for i in f.keys():
##                try:
##                    pd.concat([RESULT,i.result()],axis=0,join='outer',ignore_index=True)
##                except:
##                    pass
            for i in f.keys():
                try:
                    RESULT = pd.concat([RESULT,i.result()],axis=0,join='outer',ignore_index=True)  # <<<<<<<<<<<<<
                except:
                    print("ERROR IN FUTURE: " + str)
                    pass
            
        else:
            RESULT = func(UWIlist)
            data = [UWIlist]

        RESULT = RESULT.drop_duplicates()
        RESULT = RESULT.dropna(axis = 0, how='all')
        #RESULT=pd.DataFrame()
        #for i in range(len(data)):
        #    f = 'XYZ_'+str(int(data[i][0]))+'_'+str(int(data[i][-1]))+'.json'
        #    df_XYZ = pd.read_json(f, index_col=False)
        #    print(df_XYZ.shape)
        #    RESULT=pd.concat([RESULT,df_XYZ],axis=0,join='outer',ignore_index=True)

        #RESULT.to_csv('XYZ_SPACING_PEGGY.csv',index=False)
        #RESULT.to_csv(outfile+'.csv',index=False)
        RESULT.to_json(outfile+'.json')
        RESULT.to_parquet(outfile+'.PARQUET')

    pd_sql_types={'object':'TEXT',
                  'int64':'INTEGER',
                  'float64':'REAL',
                  'bool':'TEXT',
                  'datetime64':'TEXT',
                  'timedelta[ns]':'REAL',
                  'category':'TEXT'
        }

    #sqldb = '''\\\\Server5\\Verdad Resources\\Operations and Wells\\Geology and Geophysics\\WKR\\Decline_Parameters\\DeclineParameters_v200\\prod_data.db'''
    #sqldb = 'prod_data.db'
    with sqlite3.connect(sqldb) as conn:
            TABLE_NAME = 'CO_SPACING'
            RESULT.to_sql(TABLE_NAME,conn,
                          if_exists='replace',
                          index=False,
                          dtype=RESULT.dtypes.astype('str').map(pd_sql_types).to_dict())

    
    ##if 1==1:
    ##    uwis1=uwis[0:50]
    ##    processors = multiprocessing.cpu_count()-1
    ##    chunksize = int(len(uwis1)/processors)
    ##    batch = int(len(uwis1)/chunksize)
    ##    processors = max(processors,batch)
    ##    data=np.array_split(uwis1,batch)d
    ##    func=partial(XYZSpacing,xxdf=df,df_UWI=df_UWI,DATELIMIT=DATELIMIT)

    #with multiprocessing.Pool(processes=processors) as pool:
    #    pool.map(func,data,1)
    #with multiprocessing.Pool(processes=thread) as pool:
        #Summary.append(pool.map(XYZSpacing,[uwis[0:10],df,df_UWI,DATELIMIT]))
        #s1=pool.map(XYZSpacing,[uwis[0:20],df,df_UWI,DATELIMIT])
    #pd.concat([x for x in map(lambda x:XYZSpacing([x,df,df_UWI,DATELIMIT]),uwis[0:20])],ignore_index=True)
    #s1=pd.concat([x for x in map(lambda x:XYZSpacing(x,df,df_UWI,DATELIMIT),uwis)],ignore_index=True)

    #uwis = pd.read_csv('FULLAPILIST.csv',header=0).iloc[:,0].unique().tolist()

    #uwis=uwis.values.tolist()
    #s2=XYZSpacing(df,df_UWI,DATELIMIT,uwis)

    #s1=s2.merge(df, how = 'outer', on = 'UWI10')
    #s1=s1.merge(df_UWI, how = 'outer', on = 'UWI10')

    #s2=[x for x in map(lambda x:XYZSpacing(x,df,df_UWI,DATELIMIT),uwis)]
    #s1=pd.concat(s2,ignore_index=True)
    #s2.to_csv(outfile,sep=',',index=False)

if True:
    RESULT=pd.DataFrame()
    for f in listdir(adir):
        if f.upper().endswith('.PARQUET') and f.upper().startswith('XYZ'):
            print(f)
            RESULT = pd.concat([RESULT,pd.read_parquet(f)],axis=0,join='outer',ignore_index=True)  # <<<<<<<<<<<<<
    RESULT = RESULT.drop_duplicates()
    RESULT = RESULT.dropna(axis = 0, how='all')
    RESULT.to_json(outfile+'.json')
    RESULT.to_parquet(outfile+'.PARQUET')

        
