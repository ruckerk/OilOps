# update base files
from ._FUNCS_ import *

#Define Functions for multiprocessing iteration

def CO_BASEDATA():
    pathname = path.dirname(sys.argv[0])
    adir = path.abspath(pathname)

    #Frac Focus
    #https://www.fracfocus.org/index.php?p=data-download
    url = 'https://fracfocusdata.org/digitaldownload/FracFocusCSV.zip'
    filename = wget.download(url)
    with ZipFile(filename, 'r') as zipObj:
       # Extract all the contents of zip file in current directory
       zipObj.extractall('FRAC_FOCUS')


    # COGCC SQLITE
    # https://dnrftp.state.co.us/
    url = 'https://dnrftp.state.co.us/COGCC/Temp/Gateway/CO_3_2.1.zip'
    filename = wget.download(url)
    with ZipFile(filename, 'r') as zipObj:
       # Extract all the contents of zip file in current directory
       zipObj.extractall('COOGC_SQL')

    files = []
    start_dir = path.join(adir,'COOGC_SQL')
    pattern   = r'CO_3_2.*'

    for dir,_,_ in walk(start_dir):
        files.extend(glob(path.join(dir,pattern)))

    shutil.move(files[0], path.join(adir, path.basename(files[0])))
    shutil.rmtree(path.join(adir,'COOGC_SQL'))

    # COGCC shapefiles
    url = 'https://cogcc.state.co.us/documents/data/downloads/gis/DIRECTIONAL_LINES_SHP.ZIP'
    filename = wget.download(url)
    with ZipFile(filename, 'r') as zipObj:
       # Extract all the contents of zip file in current directory
       zipObj.extractall()
    remove(filename)

    url = 'https://cogcc.state.co.us/documents/data/downloads/gis/DIRECTIONAL_LINES_PENDING_SHP.ZIP'
    filename = wget.download(url)
    with ZipFile(filename, 'r') as zipObj:
       # Extract all the contents of zip file in current directory
       zipObj.extractall()
    remove(filename)

    url = 'https://cogcc.state.co.us/documents/data/downloads/gis/WELLS_SHP.ZIP'
    filename = wget.download(url)
    with ZipFile(filename, 'r') as zipObj:
       # Extract all the contents of zip file in current directory
       zipObj.extractall()
    remove(filename)

def Get_LAS(UWIS):
    #if 1==1:
    URL_BASE = 'http://cogcc.state.co.us/weblink/results.aspx?id=XNUMBERX'
    DL_BASE = 'http://cogcc.state.co.us/weblink/XLINKX'
    #pathname = path.dirname(sys.argv[0])
    adir = path.abspath(pathname)
    dir_add = path.join(adir,"LOGS")
    if path.isdir(dir_add) == False:
        mkdir(dir_add)
        
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
 
def Get_ProdData(UWIs,file='prod_data.db',SQLFLAG=0):
    #if 1==1:
    #URL_BASE = 'https://cogcc.state.co.us/cogis/ProductionWellMonthly.asp?APICounty=XCOUNTYX&APISeq=XNUMBERX&APIWB=XCOMPLETIONX&Year=All'
    URL_BASE = 'https://cogcc.state.co.us/production/?&apiCounty=XCOUNTYX&apiSequence=XNUMBERX'
    pathname = path.dirname(sys.argv[0])
    adir = path.abspath(pathname)
    warnings.simplefilter("ignore")
    OUTPUT=pd.DataFrame(columns=['BTU_MEAN','BTU_STD'
                                 ,'API_MEAN','API_STD'
                                 ,'Peak_Oil_Date','Peak_Oil_Days','Peak_Oil_CumOil','Peak_Oil_CumGas','Peak_Oil_CumWtr'
                                 ,'Peak_Gas_Date','Peak_Gas_Days','Peak_Gas_CumOil','Peak_Gas_CumGas','Peak_Gas_CumWtr'
                                 ,'OWR_PrePeakOil','OWR_PostPeakGas'
                                 ,'GOR_PrePeakOil','GOR_PeakGas','GOR_PostPeakGOR'
                                 ,'WOC_PostPeakOil','WOC_PostPeakGas'
                                 ,'Peak_Oil_CumWtr','Peak_Gas_CumWtr'
                                 ,'Month1'
                                 ,'GOR_MO2-4','GOR_MO5-7','GOR_MO11-13','GOR_MO23-25','GOR_MO35-37','GOR_MO47-49'
                                 ,'OWR_MO2-4','OWR_MO5-7','OWR_MO11-13','OWR_MO23-25','OWR_MO35-37','OWR_MO47-49'
                                 ,'Production_Formation'])
    MonthArray = np.arange(3,49,3)
    for i in MonthArray:
        OUTPUT[str(i)+'Mo_CumOil'] = np.nan
        OUTPUT[str(i)+'Mo_CumGas'] = np.nan
        OUTPUT[str(i)+'Mo_CumWtr'] = np.nan
                        
    if len(UWIs[0])<=1:
        UWIs=[UWIs]
        print(UWIs[0])
    ct = 0
    t1 = perf_counter()
    for UWI in UWIs:
        if (floor(ct/20)*20) == ct:
            print(str(ct)+' of '+str(len(UWIs)))
        ct+=1
        html = soup = pdf = None 
        #print(UWI)
        #if 1==1:
        ERROR=0
        while ERROR == 0: #if 1==1:
            connection_attempts = 4 
            #Screen for Colorado wells
            userows=pd.DataFrame()
            if UWI[:2] == '05':
                #print(UWI)
                #Reduce well to county and well numbers
                COWELL=UWI[5:10]
                if len(UWI)>=12:
                    COMPLETION=UWI[10:12]
                else:
                    COMPLETION="00"
                docurl=re.sub('XNUMBERX',COWELL,URL_BASE)
                docurl=re.sub('XCOUNTYX',UWI[2:5],docurl)
                docurl=re.sub('XCOMPLETIONX',COMPLETION,docurl)
                #try:
                #    html = urlopen(docurl).read()
                #except Exception as ex:
                #    print(f'Error connecting to {docurl}.')
                #    ERROR=1
                #    continue
                #soup = BS(html, 'lxml')
                #try:
                #    parsed_table = soup.find_all('table')[1]
                #except:
                #    print(f'No Table for {UWI}.')
                #    ERROR=1
                #    continue
                if perf_counter() - t1 < 0.5:
                    sleep(0.5)
                t1 = perf_counter()
                try:
                    #pdf = pd.read_html(docurl,encoding='utf-8', header=0)[1]
                    content = requests_retry_session().get(docurl).content
                    rawData = pd.read_html(io.StringIO(content.decode('utf-8')))
                    pdf = rawData[1]
                except:
                    print(f'Error connecting to {docurl}.')
                    ERROR=1
                    continue
                #pdf=pd.read_html('https://cogcc.state.co.us/cogis/ProductionWellMonthly.asp?APICounty=123&APISeq=42282&APIWB=00&Year=All')[1]
                try:
                    SEQ      = pdf.iloc[:,pdf.keys().str.contains('.*SEQUENCE.*', regex=True, case=False,na=False)].keys()[0]
                except:
                    for i in range(0,pdf.shape[1]):
                        newcol = str(pdf.keys()[i])+'_'+'_'.join(pdf.iloc[0:8,i].astype('str'))
                        pdf=pdf.rename({pdf.keys()[i]:newcol},axis=1)
                    try:    
                        SEQ      = pdf.iloc[:,pdf.keys().str.contains('.*SEQUENCE.*', regex=True, case=False,na=False)].keys()
                        x = min(np.array(pdf.loc[pdf[SEQ].astype(str) == COWELL,SEQ].index))-1   # non-value indexes
                        xrows = list(range(0, x))
                        for i in range(0,pdf.shape[1]):
                            newcol = str(pdf.keys()[i])+'_'+'_'.join(pdf.iloc[0:x,i].astype('str'))
                            pdf = pdf.rename({pdf.keys()[i]:newcol},axis=1)
                            pdf = pdf.drop(xrows,axis=0)
                    except:
                        print(f'Cannot parse tabels 1 at: {docurl}.')
                        ERROR = 1
                        continue
                   
##                DATE     =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*FIRST.*MONTH.*', regex=True, case=False,na=False)].keys()[0])
##                DAYSON   =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*DAYS.*PROD.*', regex=True, case=False,na=False)].keys()[0])
##                OIL      =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*OIL.*PROD.*', regex=True, case=False,na=False)].keys()[0])
##                GAS      =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*GAS.*PROD.*', regex=True, case=False,na=False)].keys()[0])
##                WTR      =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*WATER.*VOLUME.*', regex=True, case=False,na=False)].keys()[0])
##                API      =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*OIL.*GRAVITY.*', regex=True, case=False,na=False)].keys()[0])
##                BTU      =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*GAS.*BTU.*', regex=True, case=False,na=False)].keys()[0])
                try: 
                    DATE     = pdf.iloc[:,pdf.keys().str.contains('.*FIRST.*MONTH.*', regex=True, case=False,na=False)].keys()[0]
                    DAYSON   = pdf.iloc[0,pdf.keys().str.contains('.*DAYS.*PROD.*', regex=True, case=False,na=False)].keys()[0]
                    OIL      = pdf.iloc[0,pdf.keys().str.contains('.*OIL.*PROD.*', regex=True, case=False,na=False)].keys()[0]
                    GAS      = pdf.iloc[0,pdf.keys().str.contains('.*GAS.*PROD.*', regex=True, case=False,na=False)].keys()[0]
                    WTR      = pdf.iloc[0,pdf.keys().str.contains('.*WATER.*VOLUME.*', regex=True, case=False,na=False)].keys()[0]
                    API      = pdf.iloc[0,pdf.keys().str.contains('.*OIL.*GRAVITY.*', regex=True, case=False,na=False)].keys()[0]
                    BTU      = pdf.iloc[0,pdf.keys().str.contains('.*GAS.*BTU.*', regex=True, case=False,na=False)].keys()[0]
                    FM       = pdf.iloc[0,pdf.keys().str.contains('.*Formation.*', regex=True, case=False,na=False)].keys()[0]
                except:
                    print(f'Cannot parse tabels 2 at: {docurl}.')
                    ERROR = 1
                    continue                    

                # Date is date formatted                
                pdf[DATE]=pd.to_datetime(pdf[DATE]).dt.date
                # Sort on earliest date first
                pdf.sort_values(by = [DATE],inplace = True)
                pdf.index = range(1, len(pdf) + 1)
               
                pdf['OIL_RATE'] = pdf[OIL]/pdf[DAYSON]
                pdf['GAS_RATE'] = pdf[GAS]/pdf[DAYSON]
                pdf['WTR_RATE'] = pdf[WTR]/pdf[DAYSON]
                pdf['PROD_DAYS'] = pdf[DAYSON].cumsum()
                
                pdf['GOR'] = pdf[GAS]*1000/pdf[OIL]
                pdf['OWR'] = pdf[OIL]/pdf[WTR]
                pdf['WOR'] = pdf[WTR]/pdf[OIL]
                pdf['OWC'] = pdf[OIL]/(pdf[WTR]+pdf[OIL])
                pdf['WOC'] = pdf[WTR]/(pdf[WTR]+pdf[OIL])

                if pdf[[API]].dropna(how='any').shape[0]>3:
                    OUTPUT.at[UWI,'API_MEAN']         = pdf[API].astype('float').describe()[1]
                    OUTPUT.at[UWI,'API_STD']          = pdf[API].astype('float').describe()[2]
                    
                if pdf[[BTU]].dropna(how='any').shape[0]>3:
                    OUTPUT.at[UWI,'BTU_MEAN']         = pdf[BTU].astype('float').describe()[1]
                    OUTPUT.at[UWI,'BTU_STD']          = pdf[BTU].astype('float').describe()[2]

                if pdf[[OIL,GAS]].dropna(how='any').shape[0]>3:
                    OUTPUT.at[UWI,'Peak_Oil_Date']   = pdf[DATE][pdf[OIL].idxmax()]
                    OUTPUT.at[UWI,'Peak_Oil_Days']   = pdf['PROD_DAYS'][pdf[OIL].idxmax()]
                    OUTPUT.at[UWI,'Peak_Oil_CumOil'] = pdf[OIL][0:pdf[OIL].idxmax()].sum()
                    OUTPUT.at[UWI,'Peak_Oil_CumGas'] = pdf[GAS][0:pdf[OIL].idxmax()].sum()

                    OUTPUT.at[UWI,'Peak_Gas_Date']   = pdf[DATE][pdf[GAS].idxmax()]
                    OUTPUT.at[UWI,'Peak_Gas_Days']   = pdf['PROD_DAYS'][pdf[GAS].idxmax()]
                    OUTPUT.at[UWI,'Peak_Gas_CumOil'] = pdf[OIL][0:pdf[GAS].idxmax()].sum()
                    OUTPUT.at[UWI,'Peak_Gas_CumGas'] = pdf[GAS][0:pdf[GAS].idxmax()].sum()

                    PREPEAKOIL  = pdf.loc[(pdf['PROD_DAYS']-pdf['PROD_DAYS'][pdf[OIL].idxmax()]).between(-100,0),:].index
                    POSTPEAKOIL = pdf.loc[(pdf['PROD_DAYS'][pdf[OIL].idxmax()]-pdf['PROD_DAYS']).between(0,100),:].index
                    POSTPEAKGAS = pdf.loc[(pdf['PROD_DAYS'][pdf[GAS].idxmax()]-pdf['PROD_DAYS']).between(0,100),:].index
                    PEAKGAS = pdf.loc[(pdf['PROD_DAYS'][pdf[GAS].idxmax()]-pdf['PROD_DAYS']).between(-50,50),:].index
                    
                    OUTPUT.at[UWI,'GOR_PrePeakOil']  = pdf.loc[PREPEAKOIL,GAS].sum() * 1000 / pdf.loc[PREPEAKOIL,OIL].sum()
                    OUTPUT.at[UWI,'GOR_PeakGas']     = pdf.loc[PEAKGAS,GAS].sum() * 1000 / pdf.loc[PEAKGAS,OIL].sum()

                    if pdf[[WTR,OIL,GAS]].dropna(how='any').shape[0]>3:
                        OUTPUT.at[UWI,'OWR_PrePeakOil']  = pdf.loc[PREPEAKOIL,OIL].sum()/pdf.loc[PREPEAKOIL,WTR].sum()
                        OUTPUT.at[UWI,'OWR_PostPeakGas'] = pdf.loc[POSTPEAKGAS,OIL].sum()/pdf.loc[POSTPEAKGAS,WTR].sum()                    
                        OUTPUT.at[UWI,'WOC_PostPeakOil'] = pdf.loc[POSTPEAKOIL,WTR].sum() / (pdf.loc[POSTPEAKOIL,WTR].sum()+pdf.loc[POSTPEAKOIL,OIL].sum())
                        OUTPUT.at[UWI,'WOC_PostPeakGas'] = pdf.loc[POSTPEAKGAS,WTR].sum() / (pdf.loc[POSTPEAKGAS,WTR].sum()+pdf.loc[POSTPEAKGAS,OIL].sum())        
                        OUTPUT.at[UWI,'Peak_Oil_CumWtr'] = pdf[WTR][0:pdf[OIL].idxmax()].sum()
                        OUTPUT.at[UWI,'Peak_Gas_CumWtr'] = pdf[WTR][0:pdf[GAS].idxmax()].sum()

                    # Emily uses Month 1 begins at 1st month w/ +14days oil prod
                    if len(pdf[DATE].dropna())>10:
                        MONTH1 = pdf.loc[(pdf[DAYSON]>14) & (pdf[OIL]>0),DATE].min()
                        OUTPUT.at[UWI,'Month1'] = MONTH1

                        if not isinstance(MONTH1,float):
                            pdf['EM_PRODMONTH'] = (pd.to_datetime(pdf[DATE]).dt.year - MONTH1.year)*12+(pd.to_datetime(pdf[DATE]).dt.month - MONTH1.month)+1

##                            OUTPUT.at[UWI,'GOR_MO2-4']  = pdf.loc[(pdf['EM_PRODMONTH']>=2) & (pdf['EM_PRODMONTH']<=4),GAS].sum()*1000 / pdf.loc[(pdf['EM_PRODMONTH']>=2) & (pdf['EM_PRODMONTH']<=4),OIL].sum()
##                            OUTPUT.at[UWI,'GOR_MO5-7']  = pdf.loc[(pdf['EM_PRODMONTH']>=5) & (pdf['EM_PRODMONTH']<=7),GAS].sum()*1000 / pdf.loc[(pdf['EM_PRODMONTH']>=5) & (pdf['EM_PRODMONTH']<=7),OIL].sum()
##                            OUTPUT.at[UWI,'GOR_MO11-13']  = pdf.loc[(pdf['EM_PRODMONTH']>=11) & (pdf['EM_PRODMONTH']<=13),GAS].sum()*1000 / pdf.loc[(pdf['EM_PRODMONTH']>=11) & (pdf['EM_PRODMONTH']<=13),OIL].sum()
##                            OUTPUT.at[UWI,'GOR_MO23-25']  = pdf.loc[(pdf['EM_PRODMONTH']>=23) & (pdf['EM_PRODMONTH']<=25),GAS].sum()*1000 / pdf.loc[(pdf['EM_PRODMONTH']>=23) & (pdf['EM_PRODMONTH']<=25),OIL].sum()
##                            OUTPUT.at[UWI,'GOR_MO35-37']  = pdf.loc[(pdf['EM_PRODMONTH']>=35) & (pdf['EM_PRODMONTH']<=37),GAS].sum()*1000 / pdf.loc[(pdf['EM_PRODMONTH']>=35) & (pdf['EM_PRODMONTH']<=37),OIL].sum()
##                            OUTPUT.at[UWI,'GOR_MO47-49']  = pdf.loc[(pdf['EM_PRODMONTH']>=47) & (pdf['EM_PRODMONTH']<=49),GAS].sum()*1000 / pdf.loc[(pdf['EM_PRODMONTH']>=47) & (pdf['EM_PRODMONTH']<=49),OIL].sum()

##                            if pdf[[WTR,OIL,GAS]].dropna(how='any').shape[0]>3:
##                                OUTPUT.at[UWI,'OWR_MO2-4']  = pdf.loc[(pdf['EM_PRODMONTH']>=2) & (pdf['EM_PRODMONTH']<=4),OIL].sum() / pdf.loc[(pdf['EM_PRODMONTH']>=2) & (pdf['EM_PRODMONTH']<=4),WTR].sum()
##                                OUTPUT.at[UWI,'OWR_MO5-7']  = pdf.loc[(pdf['EM_PRODMONTH']>=5) & (pdf['EM_PRODMONTH']<=7),OIL].sum() / pdf.loc[(pdf['EM_PRODMONTH']>=5) & (pdf['EM_PRODMONTH']<=7),WTR].sum()
##                                OUTPUT.at[UWI,'OWR_MO11-13']  = pdf.loc[(pdf['EM_PRODMONTH']>=11) & (pdf['EM_PRODMONTH']<=13),OIL].sum() / pdf.loc[(pdf['EM_PRODMONTH']>=11) & (pdf['EM_PRODMONTH']<=13),WTR].sum()
##                                OUTPUT.at[UWI,'OWR_MO23-25']  = pdf.loc[(pdf['EM_PRODMONTH']>=23) & (pdf['EM_PRODMONTH']<=25),OIL].sum() / pdf.loc[(pdf['EM_PRODMONTH']>=23) & (pdf['EM_PRODMONTH']<=25),WTR].sum()
##                                OUTPUT.at[UWI,'OWR_MO35-37']  = pdf.loc[(pdf['EM_PRODMONTH']>=35) & (pdf['EM_PRODMONTH']<=37),OIL].sum() / pdf.loc[(pdf['EM_PRODMONTH']>=35) & (pdf['EM_PRODMONTH']<=37),WTR].sum()
##                                OUTPUT.at[UWI,'OWR_MO47-49']  = pdf.loc[(pdf['EM_PRODMONTH']>=47) & (pdf['EM_PRODMONTH']<=49),OIL].sum() / pdf.loc[(pdf['EM_PRODMONTH']>=47) & (pdf['EM_PRODMONTH']<=49),WTR].sum()
##                                                            
##                                OUTPUT.at[UWI,'OWC_MO3']  = pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<3),OIL].sum() / (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=3),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=3),WTR].sum())
##                                OUTPUT.at[UWI,'OWC_MO6']  = pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<6),OIL].sum() / (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=6),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=6),WTR].sum())
##                                OUTPUT.at[UWI,'OWC_MO12']  = pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<12),OIL].sum() / (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=12),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=12),WTR].sum())
##                                OUTPUT.at[UWI,'OWC_MO24']  = pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<24),OIL].sum() / (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=24),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=24),WTR].sum())
##                                OUTPUT.at[UWI,'OWC_MO36']  = pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<36),OIL].sum() / (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=36),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=36),WTR].sum())
##                                OUTPUT.at[UWI,'OWC_MO48']  = pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<48),OIL].sum() / (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<48),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=48),WTR].sum())

                            for i in MonthArray:
                                if max(pdf['EM_PRODMONTH']) >= i:
                                    i_dwn = i-1
                                    i_up = i+1
                                    OUTPUT[str(i)+'Mo_CumOil'] = pdf.loc[(pdf['EM_PRODMONTH']<=i),OIL].sum()
                                    OUTPUT[str(i)+'Mo_CumGas'] = pdf.loc[(pdf['EM_PRODMONTH']<=i),GAS].sum()
                                    OUTPUT[str(i)+'Mo_CumWtr'] = pdf.loc[(pdf['EM_PRODMONTH']<=i),WTR].sum()
                                    if pdf.loc[pdf['EM_PRODMONTH']>=i,[OIL,GAS]].dropna(how='any').shape[0]>=1:
                                        OUTPUT.at[UWI,'GOR_MO'+str(i_dwn)+'-'+str(i_up)]  = pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),GAS].sum()*1000 / pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),OIL].sum()
                                    if pdf.loc[pdf['EM_PRODMONTH']>=i,[OIL,WTR]].dropna(how='any').shape[0]>=1:
                                        OUTPUT.at[UWI,'OWC_MO'+str(i)] = pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),OIL].sum() / (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),WTR].sum())
                                        OUTPUT.at[UWI,'OWR_MO'+str(i_dwn)+'-'+str(i_up)]  = pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),OIL].sum() / pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),WTR].sum()
                                           
                OUTPUT.at[UWI,'Production_Formation'] = '_'.join(pdf[FM].unique())
                    
                
            ERROR = 1
            OUTPUT=OUTPUT.dropna(how='all')
            OUTPUT.index.name = 'UWI'   

            SQL_COLS = '''([UWI] INTEGER PRIMARY KEY
                 ,[BTU_MEAN] REAL
                 ,[BTU_STD] REAL
                 ,[API_MEAN] REAL
                 ,[API_STD] REAL
                 ,[Peak_Oil_Date] DATE
                 ,[Peak_Oil_Days] INTEGER
                 ,[Peak_Oil_CumOil] REAL
                 ,[Peak_Oil_CumGas] REAL
                 ,[Peak_Gas_Date] DATE
                 ,[Peak_Gas_Days] INTEGER
                 ,[Peak_Gas_CumOil] REAL
                 ,[Peak_Gas_CumGas] REAL
                 ,[OWR_PrePeakOil] REAL
                 ,[OWR_PostPeakGas] REAL
                 ,[WOC_PrePeakOil] REAL
                 ,[WOC_PostPeakOil] REAL
                 ,[WOC_PostPeakGas] REAL
                 ,[Peak_Oil_CumWtr] REAL
                 ,[Peak_Gas_CumWtr] REAL
                 ,[Month1] DATE
                 ,[GOR_MO2-4] REAL
                 ,[GOR_MO5-7] REAL
                 ,[GOR_MO11-13] REAL
                 ,[GOR_MO23-25] REAL
                 ,[GOR_MO35-37] REAL
                 ,[GOR_MO47-49] REAL
                 ,[OWR_MO2-4] REAL
                 ,[OWR_MO5-7] REAL
                 ,[OWR_MO11-13] REAL
                 ,[OWR_MO23-25] REAL
                 ,[OWR_MO35-37] REAL
                 ,[OWR_MO47-49] REAL
                 ,[OWC_MO3] REAL
                 ,[OWC_MO6] REAL
                 ,[OWC_MO12] REAL
                 ,[OWC_MO24] REAL
                 ,[OWC_MO36] REAL
                 ,[OWC_MO48] REAL
                 ,[Production_Formation] TEXT
                 ,[3Mo_CumOil] REAL
                 ,[6Mo_CumOil] REAL
                 ,[9Mo_CumOil] REAL
                 ,[12Mo_CumOil] REAL
                 ,[15Mo_CumOil] REAL
                 ,[18Mo_CumOil] REAL
                 ,[21Mo_CumOil] REAL
                 ,[24Mo_CumOil] REAL
                 ,[27Mo_CumOil] REAL
                 ,[30Mo_CumOil] REAL
                 ,[33Mo_CumOil] REAL
                 ,[36Mo_CumOil] REAL
                 ,[39Mo_CumOil] REAL
                 ,[42Mo_CumOil] REAL
                 ,[45Mo_CumOil] REAL
                 ,[48Mo_CumOil] REAL
                 ,[3Mo_CumGas] REAL
                 ,[6Mo_CumGas] REAL
                 ,[9Mo_CumGas] REAL
                 ,[12Mo_CumGas] REAL
                 ,[15Mo_CumGas] REAL
                 ,[18Mo_CumGas] REAL
                 ,[21Mo_CumGas] REAL
                 ,[24Mo_CumGas] REAL
                 ,[27Mo_CumGas] REAL
                 ,[30Mo_CumGas] REAL
                 ,[33Mo_CumGas] REAL
                 ,[36Mo_CumGas] REAL
                 ,[39Mo_CumGas] REAL
                 ,[42Mo_CumGas] REAL
                 ,[45Mo_CumGas] REAL
                 ,[48Mo_CumGas] REAL
                 ,[3Mo_CumWtr] REAL
                 ,[6Mo_CumWtr] REAL
                 ,[9Mo_CumWtr] REAL
                 ,[12Mo_CumWtr] REAL
                 ,[15Mo_CumWtr] REAL
                 ,[18Mo_CumWtr] REAL
                 ,[21Mo_CumWtr] REAL
                 ,[24Mo_CumWtr] REAL
                 ,[27Mo_CumWtr] REAL
                 ,[30Mo_CumWtr] REAL
                 ,[33Mo_CumWtr] REAL
                 ,[36Mo_CumWtr] REAL
                 ,[39Mo_CumWtr] REAL
                 ,[42Mo_CumWtr] REAL
                 ,[45Mo_CumWtr] REAL
                 ,[48Mo_CumWtr] REAL
                 )
                 '''

            TABLE_NAME = "PROD_SUMMARY"
            
    if (OUTPUT.shape[0] > 0) & (SQLFLAG != 0):
##                if path.exists(file):
##                    #OUTPUT.to_csv(file, mode='a', header=False)
##                    SQL_CMD = 'CREATE TABLE IF NOT EXISTS PROD_SUMMARY'+SQL_CREATETABLE
##                else:
##                    #OUTPUT.to_csv(file, mode='w', header=True)
##                    SQL_CMD = 'CREATE TABLE PROD_SUMMARY'+SQL_CREATETABLE
        #try:
        conn = sqlite3.connect(file)
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS ' + TABLE_NAME + ' ' + SQL_COLS)
        tmp = str(OUTPUT.index.max())
        OUTPUT.to_sql(tmp, conn, if_exists='replace', index = True)
        SQL_CMD='DELETE FROM '+TABLE_NAME+' WHERE [UWI] IN (SELECT [UWI] FROM \''+tmp+'\');'
        c.execute(SQL_CMD)
        SQL_CMD ='INSERT INTO '+TABLE_NAME+' SELECT * FROM \''+tmp+'\';'
        c.execute(SQL_CMD)
        conn.commit()
        
        SQL_CMD = 'DROP TABLE \''+tmp+'\';'
        c.execute(SQL_CMD)
        conn.commit()
        conn.close()
               # except: conn.close()
    try:
        conn.close()
    except:
        pass
    return(OUTPUT)
 
def Get_Scouts(UWIs,db=None):
    #if 1==1:
    Strings = ['WELL NAME/NO', 'OPERATOR', 'STATUS DATE','FACILITYID','COUNTY','LOCATIONID','LAT/LON','ELEVATION',
               'SPUD DATE','JOB DATE','JOB END DATE','TOP PZ','BOTTOM HOLE LOCATION',#r'COMPLETED.*INFORMATION.*FORMATION',
               'TOTAL FLUID USED','MAX PRESSURE','TOTAL GAS USED','FLUID DENSITY','TYPE OF GAS',
               'NUMBER OF STAGED INTERVALS','TOTAL ACID USED','MIN FRAC GRADIENT','RECYCLED WATER USED',
               'TOTAL FLOWBACK VOLUME','PRODUCED WATER USED','TOTAL PROPPANT USED',
               'TUBING SIZE','TUBING SETTING DEPTH','# OF HOLES','INTERVAL TOP','INTERVAL BOTTOM','^HOLE SIZE','FORMATION NAME','1ST PRODUCTION DATE',
               'BBLS_H2O','BBLS_OIL','CALC_GOR', 'GRAVITY_OIL','BTU_GAS','TREATMENT SUMMARY']

    status_pat = re.compile(r'Status:([\sA-Z]*)[0-9]{1,2}/[0-9]{1,2}/[0-9]{1,2}', re.I)
                
    OUTPUT=[]
    pagedf=[]
    xSummary = None
    URL_BASE = 'https://cogcc.state.co.us/cogis/FacilityDetail.asp?facid=XNUMBERX&type=WELL'
    pathname = path.dirname(sys.argv[0])
    adir = path.abspath(pathname)
    
    dir_add = path.join(adir,'SCOUTS')
    if path.isdir(dir_add) == False:
        mkdir(dir_add)
    
    warnings.simplefilter("ignore")
    if isinstance(UWIs,list) == False:
        UWIs=[UWIs]
    for UWI in UWIs:
        #if 1==1:
        UWI = str(UWI)
        if len(UWI)%2 == 1:
            UWI = UWI.zfill(len(UWI)+1)
            
        print(UWI+" "+datetime.datetime.now().strftime("%d/%m/%Y_%H:%M:%S"))
        docurl = None
        connection_attempts = 4 
        #Screen for Colorado wells
        userows=pd.DataFrame()
        if UWI[:2] == '05':
            #Reduce well to county and well numbers
            docurl=re.sub('XNUMBERX',UWI[2:10],URL_BASE)
            RETRY=0
            while RETRY<8:
                try:
                    pagedf=pd.read_html(docurl)[0]
                    RETRY=60
                except:
                    pagedf=[]
                    RETRY += 1
                    sleep(10)
            
        if len(pagedf)>0:
            xSummary = Summarize_Page(pagedf,Strings)
            xSummary['UWI']=UWI

            # Status code
            STAT_CODE = None
            try:
                status = status_pat.search(pagedf.iloc[1,0])
                status = status.group(1)
                STAT_CODE = status.strip()
            except:
                print('status error')
                pass

            xSummary['WELL_STATUS'] = STAT_CODE
            
            xSummary = pd.DataFrame([xSummary.values],columns= xSummary.index.tolist())

            if type(OUTPUT)==list:
                OUTPUT=xSummary
            else:
                OUTPUT=OUTPUT.append(xSummary,ignore_index=True)

    FILENAME = str(UWIs[0])+'_'+str(UWIs[-1])+"_"+datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    FILENAME = path.join(dir_add,FILENAME) 
    DF_UNSTRING(OUTPUT).to_json(FILENAME+'.JSON')
    DF_UNSTRING(OUTPUT).to_parquet(FILENAME+'.PARQUET')
    
    if db != None:
        DATECOLS = [col for col in OUTPUT.columns if 'DATE' in col.upper()]
        for k in DATECOLS:
            OUTPUT.loc[:,k]=pd.to_datetime(OUTPUT.loc[:,k]).fillna(np.nan)
            OUTPUT.loc[OUTPUT.loc[:,k],k]
        conn = sqlite3.connect(db)
        c = conn.cursor()
        TABLE_NAME = 'CO_SCOUT'
        SQL_COLS = list()
        # NEEDS CONVERSION OF PYTHON TYPES TO SQL TYPES
        for k,v in OUTPUT.dtypes.to_dict().items():    
            SQL_COLS=SQL_COLS+'['+str(k)+'] '+str(v)+','
        #c.execute('CREATE TABLE IF NOT EXISTS ' + TABLE_NAME + ' ')
        #sql = "select * from %s where 1=0;" % table_name
        #c.execute(sql)
        #TBL_COLS = [d[0] for d in curs.description]
        #ADD_COLS = list(set(SQL_COLS).difference(TBL_COLS))
        OUTPUT.to_sql(TABLE_NAME,conn,if_exists='append',index=False)

        #OUTPUT.to_csv(FILENAME)

    return(OUTPUT)

def Merge_Frac_Focus():
    pathname = path.dirname(sys.argv[0])
    adir = path.abspath(pathname)
    #if 1==1:
    FLIST = filelist(EXT='.csv',BEGIN = 'frac')
    FracFocus = pd.DataFrame()
    for f in FLIST:
        freg_df = pd.read_csv(f,low_memory=False)
        #freg_df = freg_df.drop_duplicates()
        FracFocus = pd.concat([FracFocus,freg_df],axis=0,join='outer',ignore_index=True)
    FracFocus = FracFocus.drop_duplicates()

    FracFocus.APINumber = FracFocus.APINumber.astype(str).str.replace(' ','').astype(int)
    
    FracFocus.to_json('FracFocusTables.JSON')
    FracFocus.to_parquet('FracFocusTables.PARQUET')
    return('FracFocusTables.PARQUET')
    

  
