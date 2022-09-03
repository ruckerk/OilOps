import urllib,datetime,re,io,csv,sys,requests,selenium,multiprocessing,warnings,concurrent.futures
from os import path, listdir, remove, makedirs
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as BS
from selenium import webdriver
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
#from multiprocessing import Pool, cpu_count

def get_driver():
    ## initialize options
    #options = webdriver.ChromeOptions()
    ## pass in headless argument to options
    #options.add_argument('--headless')
    ## initialize driver
    #driver = webdriver.Chrome('\\\Server5\\Users\\KRucker\\chromedriver.exe',chrome_options=options)

    opts = Options()
    opts.headless = True
    driver = Firefox(options=opts)
    return driver

#Define Functions for multiprocessing iteration

def Get_LAS(UWIS):
    #if 1==1:
    URL_BASE = 'http://cogcc.state.co.us/weblink/results.aspx?id=XNUMBERX'
    DL_BASE = 'http://cogcc.state.co.us/weblink/XLINKX'
    #pathname = path.dirname(sys.argv[0])
    #adir = path.abspath(pathname)
    #dir_add = path.join(adir,"\\LOGS")
    warnings.simplefilter("ignore")
    
    if isinstance(UWIS,(str,float,int)):
        UWIS = [UWIS]
    else:
        UWIS = list(UWIS)
    BADLINKS = []
    with get_driver() as browser:
        for UWI in UWIS:
            print(UWI)
            ERROR=0
            while ERROR == 0: #if 1==1:
                connection_attempts = 0 
                #Screen for Colorado wells
                userows=pd.DataFrame()
                if UWI[:2] == '05':
                    #Reduce well to county and well numbers
                    COWELL=UWI[2:10]
                    docurl=re.sub('XNUMBERX',COWELL,URL_BASE)
                    #page = requests.get(docurl)
                    #if str(page.status_code)[0] == '2':
                    #ADD# APPEND ERROR CODE TO LOG FILE
                    #option = webdriver.ChromeOptions()
                    #option.add_argument(' — incognito')
                    #browser = webdriver.Chrome('\\\Server5\\Users\\KRucker\\chromedriver.exe')
                    
                    try:
                        browser.get(docurl)
                    except Exception as ex:
                        print(f'Error connecting to {base_url}.')
                        ERROR=1

                    browser.find_element_by_link_text('Class').click()    
                    soup = BS(browser.page_source, 'lxml')
                    parsed_table = soup.find_all('table')[0]

                    pdf = pd.read_html(str(parsed_table),encoding='utf-8', header=0)[0]
                    links = [np.where(tag.has_attr('href'),tag.get('href'),"no link") for tag in parsed_table.find_all('a',string='Download')]
                    pdf['LINK']=None
                    pdf.loc[pdf.Download.str.lower()=='download',"LINK"]=links

                    userows=pdf.loc[(pdf.Class.astype(str).str.contains('Well Logs')==True)]
                    
                    # If another page, scan it too
                    # select next largest number
                    tables=len(soup.find_all('table'))
                    parsed_table = soup.find_all('table')[tables-1]
                    data = [[td.a['href'] if td.find('a') else
                             '\n'.join(td.stripped_strings)
                            for td in row.find_all('td')]
                            for row in parsed_table.find_all('tr')]
                    pages=len(data[0])
                    
                    if pages>1:
                        for p in range(1,pages):
                            try:
                                page_link = browser.find_element_by_partial_link_text(str(1+p))
                                page_link.click()
                            except:
                                browser.get(docurl)
                                page_link = browser.find_element_by_partial_link_text(str(1+p))
                                page_link.click()

                            browser.page_source
                            soup = BS(browser.page_source, 'lxml')
                            parsed_table = soup.find_all('table')[0]
                            pdf = pd.read_html(str(parsed_table),encoding='utf-8', header=0)[0]
                            links = [np.where(tag.has_attr('href'),tag.get('href'),"no link") for tag in parsed_table.find_all('a',string='Download')]
                            pdf['LINK']=None
                            pdf.loc[pdf.Download.str.lower()=='download',"LINK"]=links
                            
                            #dirdata=[s for s in data if any(xs in s for xs in ['DIRECTIONAL DATA','DEVIATION SURVEY DATA'])]
                            #surveyrows.append(dirdata)

                            userows.append(pdf.loc[(pdf.Class.astype(str).str.contains('Well Logs')==True)])
                            
                    #browser.quit()
                    userows=pd.DataFrame(userows)
                    LINKCOL=userows.columns.get_loc('LINK')
                    if userows.empty:
                        ERROR = 1
                        continue
                    userows.loc[:,'DateString']=None
                    userows.loc[:,'DateString']=userows['Date'].astype('datetime64').dt.strftime('%Y_%m_%d')
                        
                    for i in range(0,userows.shape[0]):
                        dl_url = re.sub('XLINKX', str(userows.iloc[i,LINKCOL]),DL_BASE)
                        r=requests.get(dl_url, allow_redirects=True)
                        filetype=path.splitext(re.sub(r'.*filename=\"(.*)\"',r'\1',r.headers['content-disposition']))[1]
                        #if not ("PDF" and 'TIF') in filetype.upper():
                        if 'LAS' in filetype.upper():    
                            filename=dir_add+'\\LOGDATA_'+str(userows.DateString.iloc[i])+'_'+str(UWI)+filetype
                            while path.exists(filename):
                                filename = re.sub(filetype,'_1'+filetype,filename)
                            try:
                                urllib.request.urlretrieve(dl_url, filename)
                            except:
                                print('ERROR: '+dl_url)
                                BADLINKS = BADLINKS.append(dl_url)
                                
                ERROR = 1

if __name__ == "__main__":
    # Initialize constants
    global URL_BASE
    URL_BASE = 'http://cogcc.state.co.us/weblink/results.aspx?id=XNUMBERX'
    global DL_BASE 
    DL_BASE = 'http://cogcc.state.co.us/weblink/XLINKX'
    global pathname
    pathname = path.dirname(sys.argv[0])
    global adir
    adir = path.abspath(pathname)
    global dir_add
    dir_add = path.abspath(path.dirname(sys.argv[0]))+"\\LOGS"

    #Read UWI files and form UWI list
    UWIlist=[]
    for file in listdir(adir):
        if file.lower().endswith(".uwi"):       
            with open(file, 'r') as f:
                for line in f:
                    UWIlist.append(line[:-1])
    print ("read UWI file(s)")
    #UWIlist=UWIlist[40464:]
    # Create download folder
    if not path.exists(dir_add):
        makedirs(dir_add)
        
    # Parallel Execution
    processors = multiprocessing.cpu_count()-1
    #count_per_iteration = len(UWIlist) / float(processors)
    #pool=multiprocessing.Pool(processors)
    #results = pool.map(Get_Surveys,UWIlist)
    print ("starting map function")
           
    #with multiprocessing.Pool(processes=processors) as pool:
    #    pool.map(Get_Surveys,UWIlist[0:10],1)
        
    #for i in range(0, processors):
    #    list_start = int(count_per_iteration * i)
    #    list_end = int(count_per_iteration * (i+1))
    #    results = pool.map(Get_Surveys, [UWIlist[list_start:list_end]])

    data=np.array_split(UWIlist,processors*2)

    if processors > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
            f = {executor.submit(Get_LAS, a): a for a in data}
        RESULT=pd.DataFrame()
        for i in f.keys():
            RESULT=pd.concat([RESULT,i.result()],axis=0,join='outer',ignore_index=True)
    else:
        RESULT = Get_LAS(UWIlist)
        
    with multiprocessing.Pool(processes=processors) as pool:
        pool.map(Get_LAS,UWIlist)

#    pool.map(Get_Surveys,UWIlist[10:13])
#for i in UWIlist[2:10]:
#    try: Get_Surveys(i)
#    except: continue

########################
#### END OF ROUTINE ####
########################
