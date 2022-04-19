import urllib,datetime,re,io,csv,sys,requests,selenium,multiprocessing,warnings,time,math,concurrent.futures
from os import path, listdir, remove, makedirs
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as BS
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import selenium.webdriver.chrome.service as service
#from selenium.webdriver.chrome.options import Options
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import shapefile as shp #pyshp
import sqlite3
from pyproj import Transformer

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# V006: Fix pages = 0 error 02/01/2021
# V007: Use COGCC shapefile for HZ well list
# V100: record COGCC filename, fix error in multi-page well doc tables

# v103: parquet
# NEEDS FILE NAME CHANGES FOR CLARIFICATION
# URL LIST, RAW SURVEY COMPILATION, X/Y ADD



#from multiprocessing import Pool, cpu_count

# Initialize constants
global URL_BASE
URL_BASE = 'https://cogcc.state.co.us/weblink/results.aspx?id=XNUMBERX'
global DL_BASE 
DL_BASE = 'https://cogcc.state.co.us/weblink/XLINKX'
global pathname
pathname = path.dirname(sys.argv[0])
global adir
adir = path.abspath(pathname)
global dir_add
dir_add = path.join(path.abspath(path.dirname(sys.argv[0])),"SURVEYS")

def UWI10(num,limit=10):
    high_val = 1*10**(limit)-1
    low_val = 1 * 10**(limit-2)
    if isinstance(num,str):
        num = re.sub(r'^0+','',num.strip())
    try:
        num=int(num)
    except:
        num=None
        return num
    while num > high_val:
        num = math.floor(num/100)
    while num < low_val:
        num = num * 100
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

def requests_retry_session(
    retries=4,
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


def GetKey(df,key):
    # returns list of matches to <key> in <df>.keys() as regex search
    return df.iloc[0,df.keys().str.contains('.*'+key+'.*', regex=True, case=False,na=False)].keys().to_list()



# trial well 05123421580000
##def get_driver():
##    # initialize options
##    options = webdriver.ChromeOptions()
##    # pass in headless argument to options
##    options.add_argument('--headless')
##    # initialize driver
##    driver = webdriver.Chrome('\\\Server5\\Users\\KRucker\\chromedriver.exe',chrome_options=options)
##    return driver


def get_driver():
##    # initialize options
##    options = webdriver.ChromeOptions()
##    # pass in headless argument to options
##    options.add_argument('--headless')
##    # initialize driver
##    #driver = webdriver.Chrome('\\\\server5\\Users\\KRucker\\Library\\PYPATH\\chromedriver.exe',chrome_options=options)
##
##    opts = webdriver.chrome.options.Options()
##    opts.headless = True
##    driver = webdriver.Chrome(options=opts, executable_path='\\\\server5\\Users\\KRucker\\Library\\PYPATH\\chromedriver.exe')
##    return driver
    opts = Options()
    opts.headless = True
    opts.add_argument("window-size=1280,800")
    opts.add_argument('--disable-blink-features=AutomationControlled')
    
    driver = Firefox(options=opts)
    driver.set_page_load_timeout(30)
    return driver

#Define Functions for multiprocessing iteration
def Get_Surveys(UWIx, DB = None):
    #if isinstance(UWIx, list):
    #    UWIx=UWIx
    SUMMARY = pd.DataFrame(columns = ['UWI','DOCID','DOCNUM','URL','FILE']);
    if isinstance(UWIx,(str,int,float)):
        UWIx=[UWIx]
    if isinstance(UWIx,(np.ndarray,pd.Series,pd.DataFrame)):
        UWIx=pd.DataFrame(UWIx).iloc[:,0].tolist()

    print('start: '+str(UWIx[0]) + '\n')
    ct = 0
    tot = len(UWIx)
    
    with get_driver() as browser:
        #browser.manage().timeouts().implicitlyWait(10, TimeUnit.SECONDS)
        for UWI in UWIx:
            ct += 1
            UWI = str(UWI10(UWI)).zfill(10)
            
            interval = math.floor(len(UWIx)/10)
            if interval == 0:
                interval = -1
            if np.floor(ct/interval) == ct/interval:
                print(str(ct)+' of '+str(tot) + ' for '+str(UWIx[0]) + ' : ' + str(UWI))

            #print(UWI)
            warnings.simplefilter("ignore")
            SUCCESS=TRYCOUNT=PAGEERROR=ERROR=0
            while ERROR == 0:
                while (ERROR==0) & (TRYCOUNT<10):
                    PAGEERROR = 0
                    TRYCOUNT+=1
                    #print(TRYCOUNT)
                    if TRYCOUNT>0:
                        time.sleep(5*(1+0.2)**TRYCOUNT)
                    #Screen for Colorado wells
                    surveyrows=pd.DataFrame();
                    if UWI[0:2] != '05':
                        ERROR = 1
                    #Reduce well to county and well numbers
                    COWELL=UWI[2:10]
                    docurl=re.sub('XNUMBERX',COWELL,URL_BASE)
                    #page = requests.get(docurl)
                    #if str(page.status_code)[0] == '2':
                    #ADD# APPEND ERROR CODE TO LOG FILE
                    #service = service.Service('\\\Server5\\Users\\KRucker\\chromedriver.exe')
                    #service.start()
                    #capabilities = {'chrome.binary': "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"}
                    #options = Options();
                    #options.setBinary("C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe")
                    
                    #options.setBinary("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe")
                    #option.add_argument(' — incognito')
                    #browser = webdriver.Chrome('\\\Server5\\Users\\KRucker\\chromedriver.exe')
                 
                    try:
                        browser.get(docurl)
                    except Exception as ex:                       
                        print(f'Error connecting to {docurl}.')
                        SUMMARY.loc[SUMMARY.shape[0]+1] = [UWI,None,None,docurl,None];
                        ERROR=1
                        continue
                    try:
                        # MAY BE AN ERROR HERE 
                        elem = WebDriverWait(browser, 30).until(EC.presence_of_element_located((By.LINK_TEXT, "Document Name")))
                        browser.find_element_by_link_text('Document Name').click()
                    except:
                        SUMMARY.loc[SUMMARY.shape[0]+1] = [UWI,None,None,docurl,None];
                        ERROR = 1
                      
                    soup = BS(browser.page_source, 'lxml');
                    
                    # PARSED TABLE MAY BE GIVING INDEX ERROR
                    try:
                        parsed_table = soup.find_all('table')[0]
                    except:
                        ERROR = 1
                        continue
                    links = [np.where(tag.has_attr('href'),tag.get('href'),"no link") for tag in parsed_table.find_all('a',string='Download')]
 
                    pdf = pd.read_html(str(parsed_table),encoding='utf-8', header=0)[0]
                    pdf['LINK']=None
                    pdf.loc[pdf.Download.str.lower()=='download',"LINK"]=links
                    pdf['DOCID'] = pdf.LINK.astype('str').str.split("=").str[-1]
   
                    surveyrows=pdf.loc[(pdf.iloc[:,3].astype(str).str.contains('DIRECTIONAL DATA' or 'DEVIATION SURVEY DATA' or 'DIRECTIONAL SURVEY')==True)]

                    # If another page, scan it too
                    # select next largest number
                    tables=len(soup.find_all('table'))
                    parsed_table = soup.find_all('table')[tables-1]
                    data = [[td.a['href'] if td.find('a') else
                             '\n'.join(td.stripped_strings)
                            for td in row.find_all('td')]
                            for row in parsed_table.find_all('tr')]

                    # ERROR: PAGES COUNT IS NOT LEN(DATA)
                    #pages=len(data[0])
                    try:
                        pages = len(soup.find_all('a', href=re.compile(".*page[$][0-9]*.*",re.I), text=True))
                    except:
                        pages = 0


                    #Major bugfix on not pulling multi-page doc tables
                    if pages>0:
                        for p in range(0,pages+2):
                            #print(p)
                            if p>0:
                                try:
                                    browser.find_element_by_link_text(str(1+p)).click()
                                except:
                                    if math.floor((p-1)/10) == (p-1)/10:
                                        browser.find_element_by_link_text('...').click()
                            #browser.page_source if True:
                            soup = BS(browser.page_source, 'lxml')
                            #check COGCC site didn't glitch
                            tables=len(soup.find_all('table'))
                            parsed_table = soup.find_all('table')[tables-1]
                            data = [[td.a['href'] if td.find('a') else
                                 '\n'.join(td.stripped_strings)
                                for td in row.find_all('td')]
                                for row in parsed_table.find_all('tr')]
                            newpages=len(data[0])
                            #if newpages>pages:
                            #    PAGEERROR=1
                            #    print('pageerror')
                            #    break
                            try:
                                parsed_table = soup.find_all('table')[0]
                            except:
                                continue
                            pdf = pd.read_html(str(parsed_table),encoding='utf-8', header=0)[0]
                            links = [np.where(tag.has_attr('href'),tag.get('href'),"no link") for tag in parsed_table.find_all('a',string='Download')]
                            pdf['LINK']=None
                            pdf.loc[pdf.Download.str.lower()=='download',"LINK"]=links
                            pdf['DOCID'] = pdf.LINK.astype('str').str.split("=").str[-1]
                            #dirdata=[s for s in data if any(xs in s for xs in ['DIRECTIONAL DATA','DEVIATION SURVEY DATA'])]
                            #surveyrows.append(dirdata)
                            surveyrows = surveyrows.append(pdf.loc[pdf.iloc[:,3].astype(str).str.contains('DIRECTIONAL DATA' or 'DEVIATION SURVEY DATA')==True])
                    elif (pages == 0) and (sum([len(i) for i in data]) > 10):
                        try:
                            parsed_table = soup.find_all('table')[0]
                        except:
                            ERROR=1
                            continue
                        pdf = pd.read_html(str(parsed_table),encoding='utf-8', header=0)[0]
                        links = [np.where(tag.has_attr('href'),tag.get('href'),"no link") for tag in parsed_table.find_all('a',string='Download')]
                        pdf['LINK']=None
                        pdf.loc[pdf.Download.str.lower()=='download',"LINK"]=links
                        pdf['DOCID'] = pdf.LINK.astype('str').str.split("=").str[-1]
                        #dirdata=[s for s in data if any(xs in s for xs in ['DIRECTIONAL DATA','DEVIATION SURVEY DATA'])]
                        #surveyrows.append(dirdata)
                        surveyrows = surveyrows.append(pdf.loc[pdf.iloc[:,3].astype(str).str.contains('DIRECTIONAL DATA' or 'DEVIATION SURVEY DATA')==True])
                    else:
                        print(f'No Tables for {UWI}')
                        SUMMARY.loc[SUMMARY.shape[0]+1] = [UWI,None,None,None,None]
                        PAGEERROR=ERROR=1
                        break
                    
                    surveyrows=pd.DataFrame(surveyrows)
                    if len(surveyrows)==0:
                        SUMMARY.loc[SUMMARY.shape[0]+1] = [UWI,None,None,None,None]
                        ERROR=1
                        break
                    
                    surveyrows.loc[:,'DateString']=None
                    surveyrows.loc[:,'DateString']=surveyrows['Date'].astype('datetime64').dt.strftime('%Y_%m_%d')
                    LINKCOL=surveyrows.columns.get_loc('LINK')
                    
                    for i in range(0,surveyrows.shape[0]):
                        #dl_url= re.sub('XLINKX', str(surveyrows.loc[surveyrows['Date'].astype('datetime64').idxmax(),'LINK']),DL_BASE)
                        #DocDate=str(surveyrows.loc[surveyrows['Date'].astype('datetime64').idxmax(),'DateString'])
                        dl_url= re.sub('XLINKX', str(surveyrows.iloc[i,LINKCOL]),DL_BASE)
                        DocDate = str(surveyrows.iloc[i,surveyrows.columns.get_loc('DateString')])
                        DocID   = str(surveyrows.iloc[i,surveyrows.columns.get_loc('DOCID')])
                        DocNum  = str(int(surveyrows.iloc[i,surveyrows.columns.get_loc('DocumentNumber')]))
                        #df=pd.DataFrame(surveyrows[0]).transpose()
                        #daterow=df[0].str.contains("DocDate")
                        ##df.loc[df[0].str.contains("DocDate"),:]
                        #df.loc[df[0].str.contains("DocDate"),:].replace({r'.*([0-9]{2}/[0-9]{2}/[0-9]{2,4}).*':r'\1'},regex=True).astype('datetime64')          
                        #with requests.get(dl_url) as r:
                        #    soup = BS(r.content, 'lxml')
                        #    parsed_table = soup.find_all('table')[0]
                        #    data = [[td.a['hrf'] if td.find('a') else
                        #             ''.join(td.stripped_strings)
                        #            for td in row.find_all('td')]
                        #            for row in parsed_table.find_all('tr')]
                        #    dirdata=[s for s in data if any(xs in s for xs in ['DIRECTIONAL DATA','DEVIATION SURVEY DATA'])]
                        #    surveyrows.append(dirdata)
                        #    df = pd.DataFrame(data[1:], columns=data[0])
                        #    # If another page, scan it too
                        #    parsed_table = soup.find_all('table')[0]
                        #Select most recent survey data and download   
                        #XX=df.loc[df[0].str.contains("DocDate"),:].replace({r'.*([0-9]{2}/[0-9]{2}/[0-9]{2,4}).*':r'\1'},regex=True).astype('datetime64').transpose().idxmax()
                        #dl_url=re.sub('XLINKX',df.loc[df[0].str.contains("Download"),int(XX)].to_string(index=False),DL_BASE)
                        r=requests.get(dl_url, allow_redirects=True)
                        
                        # THIS LINE CREATED AN ERROR
                        try:
                            filetype=path.splitext(re.sub(r'.*filename=\"(.*)\"',r'\1',r.headers['content-disposition']))[1]
                        except:
                            filetype = ".xxx"
                        f2 = '_'.join(['SURVEYDATA',DocDate,'DOCID'+DocID,'DOCNUM'+DocNum,'UWI'+str(UWI)]) + filetype
                        filename = path.join(dir_add,f2)
                        filename = CheckDuplicate(filename)
                        #urllib.request.urlretrieve(dl_url, filename)
                        SUMMARY.loc[SUMMARY.shape[0]+1] = [UWI,int(DocID),int(DocNum),dl_url,f2]                        
                    SUCCESS=1
                    if SUCCESS==1:
                        ERROR = 1
    #try:
    #    browser.quit() 
    #except Exception:
    #    pass
    fname = 'SURVEYS_'+str(UWIx[0])+'_'+str(UWIx[-1])
    fname = path.join(dir_add,fname)
    SUMMARY.to_json(fname+'.JSON')
    SUMMARY.to_parquet(fname+'.PARQUET')
                       
    print('end: '+str(UWIx[0]) + '\n')
                       
    return(SUMMARY)

def CheckDuplicate(fname):
    ct = ''
    while path.exists(fname):
        pattern = re.compile(r'.*_([0-9]{0,2})\.(?:[0-9a-zA-Z]{,4})',re.I)
        ct = re.search(pattern, fname)
        try:
            ct = ct.group(1)
            ct = int(ct)
            ct += 1
            ct = '_'+str(ct)
        except:
            ct = '_1'
        pattern = re.compile(r'(.*)(_[0-9]{1,4}){0,1}(\.[a-z0-9]{,4})',re.I)
        fname = re.sub(pattern,r'\1'+ct+r'\3',fname)
    return(fname)

def DF_UNSTRING(df_IN):
    df_IN=df_IN.copy()
    #DATES
    DATECOLS = [col for col in df_IN.columns if 'DATE' in col.upper()]
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
    FLOAT_MASK = (df_IN.apply(pd.to_numeric, downcast = 'float', errors = 'coerce').count() - df_IN.count()==0)   
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
    
    return(df_IN)

def str2num(str_in):
    if isinstance(str_in,(int,float)) == False:
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
    if val_in == None:
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
    if len(lst)>0:
        UWI = API2INT(lst[0],length=UWIlen)
    else:
        lst = re.findall(r'[0-9]{9,}',re.sub('\-','',ffile))
        if len(lst)>0:
            UWI = API2INT(lst[0],length=UWIlen)
        else:
            UWI = None
    return UWI

def DL_from_URL(df_in):
    #URLS = df_in.iloc[:,0]
    #FILENAMES = df_in.iloc[:,1]
    
    ERRORS = list()
    for r in df_in.iterrows():
        try:
            url = r[1][0]
            dl_file = path.join(dir_add,r[1][1])
            #dl_file = CheckDuplicate(dl_file)
            if path.exists(dl_file):
                continue            
            response = requests_retry_session().get(url)
            print(response.status_code)
            with open(dl_file,'wb') as writefile:
                writefile.write(response.content)
            #RETRY = 0
            #while RETRY <= 10:
            #    try:
            #        urllib.request.urlretrieve(url, dl_file)
            #        RETRY = 10
            #    except:
            #        RETRY += 1
            #if RETRY != 10:
            #    ERRORS.append(url)
        except:
            ERRORS.append(str(url))
    return(ERRORS)
        
                                  
if __name__ == "__main__":
    
    # Create download folder
    if not path.exists(dir_add):
        makedirs(dir_add)

        
    # summary survey filenlenames and links
    SurveySummary = path.join(dir_add,'SURVEY_DATA')
    

    #Read UWI files and form UWI list
    UWIlist=[]
    
##    for chunk in pd.read_csv('WellSummary.csv',chunksize=10000):
##        # chunk = chunk.loc[chunk['3MonthOil']>0]
##        chunk = chunk.loc[chunk.PerfInterval>3000]
##        
##        UWIlist=UWIlist+chunk.API.astype(str).str.zfill(14).str[:10].str.ljust(14,'0').tolist() 

    # List is missing surveys
    #xfile = path.join('Compiled_Well_Data_XYZ.csv')
    #df = pd.read_csv(xfile)
    xfile = path.join('Compiled_Well_Data_XYZ_06042022')
    
    if path.exists(xfile+'.PARQUET'):
        df = pd.read_parquet(xfile+'.PARQUET')
    elif path.exists(xfile+'.JSON'):
        df = pd.read_json(xfile+'.JSON')
    else:
        print('No file exists: '+xfile)
    
    #MISSING_UWIlist = df.loc[(df.dz1.isna()) & (df.PerfInterval>3000),'UWI'].astype(str).str.zfill(14)
    UWI_OK = df[['UWI10','dz1']].dropna().UWI10
    
    #del df

    # shapefile UWIlist
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
    SHPUWIlist = list(sdf.loc[sdf.Delta>2000,'UWI10'].unique())
    #SHPUWIlist=['05' + s for s in SHPUWIlist]

    # SQL DB
    sqldb = 'CO_3_2.1.sqlite'
    with sqlite3.connect(sqldb) as conn:
        dfSQL = pd.read_sql_query('SELECT * FROM WELL',conn)
    transformer = Transformer.from_crs(4269, 2878,always_xy =True)
    x=transformer.transform(list(dfSQL['Longitude']),list(dfSQL['Latitude']))
    dfSQL['SHLX']=x[0]
    dfSQL['SHLY']=x[1]
    x=transformer.transform(list(dfSQL['BottomHolelongitude']),list(dfSQL['BottomHoleLatitude']))
    dfSQL['BHLX']=x[0]
    dfSQL['BHLY']=x[1]
    dfSQL['Delta'] = ((dfSQL.SHLX-dfSQL.BHLX)**2+(dfSQL.SHLY-dfSQL.BHLY)**2)**0.5
    mask = dfSQL.Delta>2000
    idx = dfSQL.loc[mask,['FirstProdDate','FirstCompDate','SpudDate']].dropna(thresh=1).index
    SQL_UWIlist = list(dfSQL.loc[idx,'API'].apply(UWI10).unique())
    del x
    # list of collected API's
    FLIST = []
    UWIlist = pd.Series(SQL_UWIlist+SHPUWIlist)
    UWIlist = UWIlist.apply(UWI10)


#######################################################################################XXX

    mask = UWIlist.isin(UWI_OK)
    UWIlist = UWIlist.loc[~mask]

##    #for file in listdir('\\\\Server5\\Verdad Resources\\Operations and Wells\\Geology and Geophysics\\WKR\\Decline_Parameters\\DeclineParameters_v200\\SURVEYS'):
##    # This stops search for a well that has any survey already
##    if path.exists(path.join(adir,'SURVEYS')):
##        for file in listdir(path.join(adir,'SURVEYS')):
##            if file.lower().endswith(('.xls','xlsx','xlsm')):
##                FLIST.append(file)
##        #fAPI = pd.Series(FLIST).astype(str).str.extract(r'_([0-9]{14}).*').iloc[:,0].unique()
##        fAPI = pd.Series(FLIST).apply(APIfromFilename).unique()
##        fAPI = list(pd.Series(fAPI).apply(UWI10))
##    else:
##        fAPI=list()
##
##    UWIlist = UWIlist.loc[~UWIlist.isin(fAPI)]



# READ JOINED_ABS_SURVEYS
# LOOK FOR "REAL" SURVEYS
# OMIT LIST OF WELLS TO PULL SURVEYS

    #BestSurveys = 'JOINED_ABS_SURVEYS.csv'
    BestSurveys = path.join(dir_add,'JOINED_ABS_SURVEYS_TC.json')
  
    if path.exists(dir_add):
        # only look after landing
        LAND_INC = 88

        #BEST_Sdf = pd.read_csv(path.join(adir,'SURVEYS',BestSurveys))
        BEST_Sdf = pd.read_json(BestSurveys)
        
        LAND_MD_df = BEST_Sdf.loc[BEST_Sdf.INC >= LAND_INC].groupby(['FILE','UWI'],axis = 0)['MD'].min()
        LAND_MD_df = LAND_MD_df.reset_index()
        LAND_MD_df = LAND_MD_df.rename(columns = {'MD':'HEEL_MD'})
        
        BEST_Sdf = BEST_Sdf.merge(LAND_MD_df,on=['FILE','UWI'])
        BEST_Sdf = BEST_Sdf.loc[BEST_Sdf.MD>=BEST_Sdf.HEEL_MD]     
        
        BEST_Sdf['AZI_dec'] = (BEST_Sdf.AZI - BEST_Sdf.AZI.apply(np.floor))
        ftest = BEST_Sdf.groupby(['FILE','UWI'])['AZI_dec'].std()
        ftest = ftest[ftest>0.1].reset_index()

        fAPI = list(ftest.UWI.apply(UWI10))
        

    UWIlist = UWIlist.loc[~UWIlist.isin(fAPI)]
    #UWIlist = list((UWIlist*10000).astype(str).str.zfill(14).unique())
if True:

    FLIST = list()
    for file in listdir(dir_add):
        if (file.lower().endswith('.json')) and ('survey' in file.lower()) and not('join' in file.lower()) and not('data' in file.lower()):
            FLIST.append(file)
            
    R_df = pd.DataFrame()
    for f in FLIST:
        f = path.join(dir_add,f)
        if R_df.empty:
            R_df = pd.read_json(f)
        else:
            R_df = pd.concat([R_df,pd.read_json(f)],axis=0,join='outer',ignore_index=True)

    idx = R_df.URL.drop_duplicates().index
    R_df = R_df.loc[idx]

    #mask = pd.to_numeric(R_df['UWI'],errors = 'coerce').dropna().index
    #R_df = R_df.loc[mask].drop_duplicates()

    UWIlist = pd.Series(UWIlist).apply(UWI10)
    mask = UWIlist.isin(R_df.UWI.apply(UWI10))
    UWIlist = UWIlist.loc[~mask]
    UWIlist = UWIlist.drop_duplicates()
    UWIlist = list(UWIlist)  
    
    #UWIlist = pd.read_csv('RELOAD_SURVEY.csv')
    #UWIlist = list(set(UWIlist.UWI10.unique()))
    
    #UWIlist = list(dfxyz[~dfxyz.iloc[:,0].apply(UWI10).isin(UList)].iloc[:,0].astype(str).str.zfill(14).unique())
    #del sdf, ddf, fAPI, dfSQL, x, SHPUWIlist
    #quit()
    
    #UWIlist = pd.read_csv('surveys.uwi')
    #UWIlist = list(UWIlist.iloc[:,0].astype(str).str.zfill(14))
 
    # Parallel Execution if 1==1:

    processors = multiprocessing.cpu_count()-0
    processors = min(processors,processors-2)
    #quit()
    CoList = [idx for idx in UWIlist if str(idx).zfill(10).startswith('05')]
    chunksize = min(200,int(len(CoList)/processors))
    batch = int(len(CoList)/chunksize)
#    processors = min(processors,batch)

    data=np.array_split(CoList,batch)
    print (f'batch = {batch}')

    #if True:
    if processors > 1: 
        with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
            f={executor.submit(Get_Surveys, a): a for a in data}
        RESULT=pd.DataFrame()
        for i in f.keys():
            RESULT=pd.concat([RESULT,i.result()],axis=0,join='outer',ignore_index=True)
    else:
        RESULT = Get_Surveys(UWIlist)

##    #RESULT.to_json('SURVEY_DATA.JSON')
##    if path.exists('SURVEY_DATA.JSON'):
##        R2 = pd.read_json('SURVEY_DATA.JSON')
##        RESULT = pd.concat([RESULT,R2],axis = 0, join = 'outer', ignore_index = True)
##        RESULT.drop_duplicates(inplace = True)
##        RESULT.to_json('SURVEY_DATA.JSON')
##        del R2
##    else:
##        RESULT.to_json('SURVEY_DATA.JSON')

if True:
    FLIST = list()
    for file in listdir(dir_add):
        if (file.lower().endswith('.json')) and ('survey' in file.lower()) and not('join' in file.lower()) and not('data' in file.lower()):
            FLIST.append(file)
            
    R_df = pd.DataFrame()
    for f in FLIST:
        f=path.join(dir_add,f)
        if R_df.empty:
            R_df = pd.read_json(f)
        else:
            R_df = pd.concat([R_df,pd.read_json(f)],axis=0,join='outer',ignore_index=True)
            
    mask = pd.to_numeric(R_df['UWI'],errors = 'coerce').dropna().index
    R_df = R_df.loc[mask].drop_duplicates()
    idx = R_df.URL.drop_duplicates().index
    R_df = R_df.loc[idx]
    idx = R_df.URL.dropna().index
    R_df = R_df.loc[idx]
    R_df.dropna(axis=1,how='all',inplace=True)
        

    fname = path.join(dir_add,'SURVEY_DATA.JSON')
    if path.exists(fname):
        R_df2 = pd.read_json(fname)
        R_df = pd.concat([R_df,R_df2],axis = 0, join = 'outer', ignore_index = True)
        R_df.drop_duplicates(inplace = True)
        idx = R_df.URL.drop_duplicates().index
        R_df = R_df.loc[idx]
        idx = R_df.URL.dropna().index
        R_df = R_df.loc[idx]
        R_df.dropna(axis=1,how='all',inplace=True)
        R_df.to_json(fname)
        del R_df2
    else:
        R_df.to_json(fname)

if True:
    FLIST = list()
    for file in listdir(dir_add):
        if ('.xls' in file.lower()) and ('surveydata' in file.lower()):
            FLIST.append(file)
            
    R_df = pd.read_json('SURVEY_DATA.JSON')
    dl_df = R_df[['URL','FILE']].dropna()
    dl_df = dl_df.loc[~dl_df.FILE.isin(FLIST)]
    processors = 1
    chunksize = min(200,int(dl_df.shape[0]/processors))
    batch = int(dl_df.shape[0]/chunksize)
    data=np.array_split(dl_df,batch)
    
    if processors > 1: 
        with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
            f={executor.submit(DL_from_URL, a): a for a in data}
        ERRORS=list()
        for i in f.keys():
            ERRORS=ERRORS.append(i.result())
    else:
        ERRORS = DL_from_URL(dl_df)
        
    
    #with multiprocessing.Pool(processes=processors) as pool:
    #    pool.map(Get_Surveys,data,1)

    #05123453030000
    #CoList.index('05123453030000')
    #Get_Surveys(CoList[2373:])
        
    #count_per_iteration = len(UWIlist) / float(processors)
    #pool=multiprocessing.Pool(processors)
    #results = pool.map(Get_Surveys,UWIlist)

    #with multiprocessing.Pool(processes=processors) as pool:
    #    pool.map(Get_Surveys,UWIlist[2:6],1)
        
    #for i in range(0, processors):
    #    list_start = int(count_per_iteration * i)
    #    list_end = int(count_per_iteration * (i+1))
    #    results = pool.map(Get_Surveys, [UWIlist[list_start:list_end]])

    #with multiprocessing.Pool(processes=processors) as pool:
    #    pool.map(Get_Surveys,UWIlist[0:5])
    #UWIlist='05001100400000'
    #for i in UWIlist:
    #    try: Get_Surveys(i)
    #    except: continue

    ########################
    #### END OF ROUTINE ####
    ########################

