import urllib,datetime,re,io,csv,sys,requests,selenium,multiprocessing,warnings,time,math
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

# V006: Fix pages = 0 error 02/01/2021
# V007: Use COGCC shapefile for HZ well list

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
dir_add = path.abspath(path.dirname(sys.argv[0]))+"\\SURVEYS"

def UWI10(num):
    if isinstance(num,str):
        num = re.sub(r'^0+','',num.strip())
    try:
        num=int(num)
    except:
        num=None
        return num
    while num > 9e9:
        num = math.floor(num/100)
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
    
    return driver

#Define Functions for multiprocessing iteration
def Get_Surveys(UWIx):
    
    #if 1==1:
    #if isinstance(UWIx, list):
    #    UWIx=UWIx
    if isinstance(UWIx,(str,int,float)):
        UWIx=[UWIx]
    if isinstance(UWIx,(np.ndarray,pd.Series,pd.DataFrame)):
        UWIx=pd.DataFrame(UWIx).iloc[:,0].tolist()

    with get_driver() as browser:
        for UWI in UWIx:
            print(UWI)
            warnings.simplefilter("ignore")
            SUCCESS=TRYCOUNT=PAGEERROR=ERROR=0
            while ERROR == 0:
                while (ERROR==0) & (TRYCOUNT<10):
                    TRYCOUNT+=1
                    print(TRYCOUNT)
                    if TRYCOUNT>0:
                        time.sleep(30)
                    #Screen for Colorado wells
                    surveyrows=pd.DataFrame()
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
                        continue
                        print(f'Error connecting to {docurl}.')
                        ERROR=1
                    browser.find_element_by_link_text('Document Name').click()

                    soup = BS(browser.page_source, 'lxml')
                    parsed_table = soup.find_all('table')[0]
                
                    pdf = pd.read_html(str(parsed_table),encoding='utf-8', header=0)[0]
                    links = [np.where(tag.has_attr('href'),tag.get('href'),"no link") for tag in parsed_table.find_all('a',string='Download')]
                    pdf['LINK']=None
                    pdf.loc[pdf.Download.str.lower()=='download',"LINK"]=links

                    surveyrows=pdf.loc[(pdf.iloc[:,3].astype(str).str.contains('DIRECTIONAL DATA' or 'DEVIATION SURVEY DATA' or 'DIRECTIONAL SURVEY')==True)]

                    # If another page, scan it too
                    # select next largest number
                    tables=len(soup.find_all('table'))
                    parsed_table = soup.find_all('table')[tables-1]
                    data = [[td.a['href'] if td.find('a') else
                             '\n'.join(td.stripped_strings)
                            for td in row.find_all('td')]
                            for row in parsed_table.find_all('tr')]
                    pages=len(data[0])

                    if pages>0:
                        for p in range(1,pages):
                            browser.find_element_by_link_text(str(1+p)).click()
                            browser.page_source
                            soup = BS(browser.page_source, 'lxml')
                            #check COGCC site didn't glitch
                            tables=len(soup.find_all('table'))
                            parsed_table = soup.find_all('table')[tables-1]
                            data = [[td.a['href'] if td.find('a') else
                                 '\n'.join(td.stripped_strings)
                                for td in row.find_all('td')]
                                for row in parsed_table.find_all('tr')]
                            newpages=len(data[0])
                            if newpages>pages:
                                PAGEERROR=1
                                break
                            parsed_table = soup.find_all('table')[0]
                            pdf = pd.read_html(str(parsed_table),encoding='utf-8', header=0)[0]
                            links = [np.where(tag.has_attr('href'),tag.get('href'),"no link") for tag in parsed_table.find_all('a',string='Download')]
                            pdf['LINK']=None
                            pdf.loc[pdf.Download.str.lower()=='download',"LINK"]=links
                            #dirdata=[s for s in data if any(xs in s for xs in ['DIRECTIONAL DATA','DEVIATION SURVEY DATA'])]
                            #surveyrows.append(dirdata)
                            surveyrows.append(pdf.loc[pdf.iloc[:,3].astype(str).str.contains('DIRECTIONAL DATA' or 'DEVIATION SURVEY DATA')==True])
                    elif (pages == 0) and (sum([len(i) for i in data]) > 10):
                        parsed_table = soup.find_all('table')[0]
                        pdf = pd.read_html(str(parsed_table),encoding='utf-8', header=0)[0]
                        links = [np.where(tag.has_attr('href'),tag.get('href'),"no link") for tag in parsed_table.find_all('a',string='Download')]
                        pdf['LINK']=None
                        pdf.loc[pdf.Download.str.lower()=='download',"LINK"]=links
                        #dirdata=[s for s in data if any(xs in s for xs in ['DIRECTIONAL DATA','DEVIATION SURVEY DATA'])]
                        #surveyrows.append(dirdata)
                        surveyrows.append(pdf.loc[pdf.iloc[:,3].astype(str).str.contains('DIRECTIONAL DATA' or 'DEVIATION SURVEY DATA')==True])
                    else:
                        print(f'No Tables for {UWI}')
                        PAGEERROR=ERROR=1
                        break
                    
                    surveyrows=pd.DataFrame(surveyrows)
                    if len(surveyrows)==0:
                        ERROR=1
                        break
                    surveyrows.loc[:,'DateString']=None
                    surveyrows.loc[:,'DateString']=surveyrows['Date'].astype('datetime64').dt.strftime('%Y_%m_%d')
                    LINKCOL=surveyrows.columns.get_loc('LINK')
                    for i in range(0,surveyrows.shape[0]):
                        #dl_url= re.sub('XLINKX', str(surveyrows.loc[surveyrows['Date'].astype('datetime64').idxmax(),'LINK']),DL_BASE)
                        #DocDate=str(surveyrows.loc[surveyrows['Date'].astype('datetime64').idxmax(),'DateString'])
                        dl_url= re.sub('XLINKX', str(surveyrows.iloc[i,LINKCOL]),DL_BASE)
                        DocDate=str(surveyrows.iloc[i,surveyrows.columns.get_loc('DateString')])
                        #df=pd.DataFrame(surveyrows[0]).transpose()
                        #daterow=df[0].str.contains("DocDate")
                        ##df.loc[df[0].str.contains("DocDate"),:]
                        #df.loc[df[0].str.contains("DocDate"),:].replace({r'.*([0-9]{2}/[0-9]{2}/[0-9]{2,4}).*':r'\1'},regex=True).astype('datetime64')          
                        #with requests.get(dl_url) as r:
                        #    soup = BS(r.content, 'lxml')
                        #    parsed_table = soup.find_all('table')[0]
                        #    data = [[td.a['href'] if td.find('a') else
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
                        filetype=path.splitext(re.sub(r'.*filename=\"(.*)\"',r'\1',r.headers['content-disposition']))[1]
                        filename=dir_add+'\\SURVEYDATA_'+DocDate+'_'+str(UWI)+filetype
                        if path.exists(filename):
                            #remove(filename)
                            filename=dir_add+'\\SURVEYDATA_'+DocDate+'_'+str(UWI)+'_1'+filetype
                        urllib.request.urlretrieve(dl_url, filename)
                    SUCCESS=1
                    if PAGEERROR==1:
                         TRYCOUNT+=1
                         PAGEERROR=0
                    if SUCCESS==1:
                        ERROR = 1
    try: browser.quit()
    except Exception:
        None
        
if __name__ == "__main__":


    #Read UWI files and form UWI list
    UWIlist=[]
    
    for chunk in pd.read_csv('WellSummary.csv',chunksize=10000):
        # chunk = chunk.loc[chunk['3MonthOil']>0]
        chunk = chunk.loc[chunk.PerfInterval>3000]
        
        UWIlist=UWIlist+chunk.API.astype(str).str.zfill(14).str[:10].str.ljust(14,'0').tolist() 

    # List is missing surveys
    xfile = "\\\\Server5\\Verdad Resources\\Operations and Wells\\Geology and Geophysics\\WKR\\Decline_Parameters\\DeclineParameters_v200\\Compiled_Well_Data.csv"
    df = pd.read_csv(xfile)
    UWIlist = df.loc[(df.dz1.isna()) & (df.PerfInterval>3000),'UWI'].astype(str).str.zfill(14)
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

    SQL_UWIlist = list(dfSQL.loc[dfSQL.Delta>2000,'API'].apply(UWI10).unique())

    # list of collected API's
    FLIST = []
    for file in listdir('\\\\Server5\\Verdad Resources\\Operations and Wells\\Geology and Geophysics\\WKR\\Decline_Parameters\\DeclineParameters_v200\\SURVEYS'):
        if file.lower().endswith(('.xls','xlsx','xlsm')):
            FLIST.append(file)
    fAPI = pd.Series(FLIST).astype(str).str.extract(r'_([0-9]{14}).*').iloc[:,0].unique()
    fAPI = list(pd.Series(fAPI).apply(UWI10))

    UWIlist = pd.Series(SQL_UWIlist+SHPUWIlist)
    UWIlist = UWIlist.apply(UWI10)
    UWIlist = UWIlist.loc[~UWIlist.isin(fAPI)]
    UWIlist = list((UWIlist*10000).astype(str).str.zfill(14).unique())
    
    
    #quit()
    #UWIlist = list(dfxyz[~dfxyz.iloc[:,0].apply(UWI10).isin(UList)].iloc[:,0].astype(str).str.zfill(14).unique())
    #del sdf, ddf, fAPI, dfSQL, x, SHPUWIlist
    #quit()
    
    #UWIlist = pd.read_csv('surveys.uwi')
    #UWIlist = list(UWIlist.iloc[:,0].astype(str).str.zfill(14))
    
    # Create download folder
    if not path.exists(dir_add):
            makedirs(dir_add)
    # Parallel Execution if 1==1:
    processors = multiprocessing.cpu_count()-0
    #quit()
    CoList = [idx for idx in UWIlist if str(idx).startswith('05')]
    chunksize = int(len(CoList)/processors)
    batch = int(len(CoList)/chunksize)
    processors = max(processors,batch)
    data=np.array_split(CoList,batch)
    print (f'batch = {batch}')
    #quit()
    with multiprocessing.Pool(processes=processors) as pool:
        pool.map(Get_Surveys,data,1)

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

