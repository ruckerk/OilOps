# update base files
from ._FUNCS_ import *

__all__ = ['CO_BASEDATA',
           'Get_LAS',
           'Get_ProdData',
           'Get_Scouts',
           'Merge_Frac_Focus',
           'SUMMARIZE_COGCC']

#Define Functions for multiprocessing iteration

def CO_BASEDATA(FRACFOCUS = True, COGCC_SQL = True, COGCC_SHP = True):
    pathname = path.dirname(argv[0])
    adir = path.abspath(pathname)

    #Frac Focus
    if FRACFOCUS:
        #https://www.fracfocus.org/index.php?p=data-download
        url = 'https://fracfocusdata.org/digitaldownload/FracFocusCSV.zip'
        filename = wget.download(url)
        with ZipFile(filename, 'r') as zipObj:
           # Extract all the contents of zip file in current directory
           zipObj.extractall('FRAC_FOCUS')


    # COGCC SQLITE
    # https://dnrftp.state.co.us/
    if COGCC_SQL:
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
    if COGCC_SHP:
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
    pathname = path.dirname(argv[0])
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
    pathname = path.dirname(argv[0])
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
                    rawData = pd.read_html(StringIO(content.decode('utf-8')))
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
    pathname = path.dirname(argv[0])
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
    pathname = path.dirname(argv[0])
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
   
def SUMMARIZE_COGCC():
    # SQLite COMMMANDS
    # well data
    Q1 = """
        SELECT
                P.StateProducingZoneKey
                ,P.ProductionDate
                ,P.StateProducingUnitKey
                ,P.DaysProducing
                ,P.Oil
                ,P.Gas
                ,P.Water
                ,SUM(CASE WHEN P.DaysProducing + 0 = 0 THEN 28
                           ELSE P.DaysProducing 
                           END) 
                    OVER(PARTITION BY P.StateProducingUnitKey
                        ORDER BY P.ProductionDate ASC) as DaysOn
                ,CAST(SUM(CASE WHEN P.DaysProducing + 0 = 0 THEN 28
                           ELSE P.DaysProducing
                           END) 
                  OVER(PARTITION BY P.StateProducingUnitKey
                        ORDER BY P.ProductionDate ASC)/30.4 AS int) AS MonthsOn
                ,SUM(P.Oil+0) OVER 
                        (PARTITION BY P.StateProducingUnitKey
                        ORDER BY P.ProductionDate ASC) as CumOil
                ,SUM(P.Gas+0) OVER 
                        (PARTITION BY P.StateProducingUnitKey
                        ORDER BY P.ProductionDate ASC) as CumGas        
                ,SUM(P.Water+0) OVER 
                        (PARTITION BY P.StateProducingUnitKey
                        ORDER BY P.ProductionDate ASC) as CumWater
                ,P.GAS / P.OIL *1000 as GOR
                ,P.OIL / IIF(P.DaysProducing is null,28,P.DaysProducing) AS OilRate_BOPD
                ,P.Gas / IIF(P.DaysProducing is null = 0,28,P.DaysProducing) AS GasRate_MCFPD
                ,P.Water / IIF(P.DaysProducing is null = 0,28,P.DaysProducing) AS WaterRate_BWPD
                ,((P.GAS / P.OIL - LAG(P.GAS,2) OVER (
                                Partition By P.StateProducingUnitKey 
                                Order By P.ProductionDate)
                        / LAG(P.OIL,2)  OVER (
                                Partition By P.StateProducingUnitKey 
                                Order By P.ProductionDate)
                                ) +
                        (P.GAS / P.OIL - LAG(P.GAS,1) OVER (
                                Partition By P.StateProducingUnitKey 
                                Order By P.ProductionDate)
                        / LAG(P.OIL,1)  OVER (
                                Partition By P.StateProducingUnitKey 
                                Order By P.ProductionDate)
                                ))*1000
                        AS dGOR
                ,PM.StartDate
                ,PM.PeakOil
                ,PM.PeakGas
                ,PM.PeakWater
                ,P.Oil/P.DaysProducing / PM.PeakOil AS NormOilRate
                ,P.Gas/P.DaysProducing / PM.PeakGas AS NormGasRate
                ,P.Water/P.DaysProducing / PM.PeakWater AS NormWater
                ,P.Gas /P.DaysProducing /  PM.PeakGas / (P.Oil / P.DaysProducing / PM.PeakOil) AS NormGOR
                ,(0+STRFTIME('%Y',PM.StartDate)) as MINYEAR
                ,W.*
        FROM Production P
        LEFT JOIN Well AS W ON W.StateWellKey = P.StateProducingUnitKey
        LEFT JOIN (
                select StateProducingUnitKey
                        , MIN(ProductionDate) as StartDate
                        , MAX(Oil/DaysProducing) AS PeakOil
                        , MAX(Gas/DaysProducing) AS PeakGas
                        ,MAX(Water/DaysProducing) AS PeakWater 
                        from PRODUCTION 
                        WHERE DaysProducing < 600
                        GROUP BY StateProducingUnitKey
                ) PM
                ON PM.StateProducingUnitKey = P.StateProducingUnitKey
        WHERE (P.OIL + P.WATER + P.GAS) >0 
        --      AND (STRFTIME('%Y',PM.StartDate)+0) > 2015
        --      AND substr(P.StateProducingUnitKey,1,7)+0 >=  512334
        --  AND (0+STRFTIME('%Y',MIN(ProductionDate) OVER (PARTITION BY StateProducingUnitKey)))>=2015
        --      AND Trim(P.StateProducingZoneKey) = "NBRR"
        -- GROUP BY StateProducingUnitKey
        ORDER BY P.StateProducingUnitKey,P.ProductionDate DESC
    """

    Q1 = """
    SELECT
        StateProducingUnitKey
        ,MIN(ProductionDate) as StartDate
        ,MAX(Oil/DaysProducing) AS PeakOil
        ,MAX(Gas/DaysProducing) AS PeakGas
        ,MAX(Water/DaysProducing) AS PeakWater
        FROM PRODUCTION 
        WHERE DaysProducing < 800
        GROUP BY StateProducingUnitKey
    """

    # Completions Groups
    Q2_Drop = "DROP TABLE If EXISTS temp_table1;"
    Q2 = """
    CREATE TABLE IF NOT EXISTS temp_table1 AS
        SELECT DISTINCT
            StateProducingUnitKey
            ,CAST(StateProducingUnitKey AS INTEGER) AS INT_SPUKEY
            ,ProductionDate
            ,SUM(Oil + Gas + Water) OVER (PARTITION BY  StateProducingUnitKey
                                                ORDER BY ProductionDate
                                                ROWS BETWEEN UNBOUNDED PRECEDING    
                                                    AND 1 PRECEDING) 
                AS Cum_OilGasWtr
        FROM Production
        WHERE (Oil + Gas + Water)>=0
        GROUP BY StateProducingUnitKey,ProductionDate;
    SELECT 
        S.StateWellKey
        ,S.StateStimulationKey
        ,S.TreatmentDate
        ,TR2.PastTreatmentDate
        ,TR2.PastStateStimulationKey
        ,P1.Cum_OilGasWtr as CumTreatOne
        ,P2.Cum_OilGasWtr as CumTreatTwo
        ,P1.Cum_OilGasWtr - P2.Cum_OilGasWtr as Cum_Difference
        , DATE(TR.TreatmentDate,'start of month') AS DATE1
        , DATE(TR2.PastTreatmentDate,'start of month') AS DATE2
        ,TR.TreatmentRank as TreatmentRank
    FROM Stimulation S
    LEFT JOIN (
        SELECT 
            StateStimulationKey
            ,StateWellKey
            ,TreatmentDate
            ,Rank() OVER (
                PARTITION BY StateWellKey
                ORDER BY TreatmentDate ASC) TreatmentRank
            ,COUNT(TreatmentDate) Over (PARTITION BY StateWellKey) as TreatmentCount
        FROM Stimulation 
        ) TR
            ON TR.StateStimulationKey = S.StateStimulationKey
    LEFT JOIN (
        SELECT 
            StateStimulationKey AS PastStateStimulationKey
            ,StateWellKey
            ,TreatmentDate As PastTreatmentDate
            ,Rank() OVER (
                PARTITION BY StateWellKey
                ORDER BY TreatmentDate ASC) PastTreatmentRank
        FROM Stimulation ) TR2
    ON  (TR2.PastTreatmentRank + 1)= TR.TreatmentRank AND TR2.StateWellKey=TR.StateWellKey          
    LEFT JOIN temp_table1 P1
        ON  P1.INT_SPUKEY = CAST(S.StateWellKey AS INTEGER)
            AND DATE(P1.ProductionDate,'start of month') = DATE(TR.TreatmentDate,'start of month')
    LEFT JOIN temp_table1 P2
        ON  P2.INT_SPUKEY = CAST(S.StateWellKey AS INTEGER)
            AND DATE(P2.ProductionDate,'start of month') = DATE(TR2.PastTreatmentDate,'start of month')
    ORDER BY Cum_Difference DESC
    """

    #Completion Volumes
    Q3 = """
    SELECT 
            S.*
            ,SF.Type
            ,SF.TotalAmount
            ,SF.QC_CountUnits
        FROM Stimulation S
        LEFT JOIN(
            SELECT 
                StateStimulationKey
                ,FluidType as Type
                ,SUM(Volume) as TotalAmount
                ,COUNT(DISTINCT FluidUnits) as QC_CountUnits
            FROM StimulationFluid
            GROUP BY StateStimulationKey,FluidType
            ) SF
            ON SF.StateStimulationKey = S.StateStimulationKey
    UNION   
    SELECT 
            S.*
            ,SP.Type
            ,SP.TotalAmount
            ,SP.QC_CountUnits
        FROM Stimulation S      
            LEFT JOIN(
            SELECT
                StateStimulationKey
                ,'PROPPANT' as Type
                ,SUM(ProppantAmount) as TotalAmount
                ,COUNT(DISTINCT ProppantUnits) as QC_CountUnits
            FROM StimulationProppant
            GROUP BY StateStimulationKey
            ) SP
            ON SP.StateStimulationKey = S.StateStimulationKey   
    """

    # 3,6,9,12,15,18,21,24 month cum
    Q4="""
    SELECT 
        PP.StateProducingUnitKey
        ,Min(CASE
        WHEN PP.Oil + PP.Gas + PP.Water >=0
        THEN PP.ProductionDate  
        Else NULL
        END)  as FirstProduction 
        ,Min(CASE 
            WHEN PP.OilRank=1
            THEN PP.CumOil
            Else NULL
            END
            ) AS PeakOil_CumOil
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.CumGas
            Else NULL
            END
            )  AS PeakOil_CumGas
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.CumWater
            Else NULL
            END
            )  AS PeakOil_CumWater 
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.ProductionDate
            Else NULL
            END
            )  AS PeakOil_Date
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.DaysOn
            Else NULL
            END
            )  AS PeakOil_DaysOn
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.OilRate_BBLPD
            Else NULL
            END
            )  AS PeakOil_OilRate
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.GasRate_MCFPD
            Else NULL
            END
            )  AS PeakOil_GasRate
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.WaterRate_BBLPD
            Else NULL
            END
            )  AS PeakOil_WaterRate
        ,Min(CASE 
            WHEN PP.WaterRank=1
            THEN PP.CumOil
            Else NULL
            END
            ) AS PeakWater_CumOil
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.CumGas
            Else NULL
            END
            )  AS PeakWater_CumGas
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.CumWater
            Else NULL
            END
            )  AS PeakWater_CumWater 
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.ProductionDate
            Else NULL
            END
            )  AS PeakWater_Date
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.DaysOn
            Else NULL
            END
            )  AS PeakWater_DaysOn
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.OilRate_BBLPD
            Else NULL
            END
            )  AS PeakWater_OilRate
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.GasRate_MCFPD
            Else NULL
            END
            )  AS PeakWater_GasRate
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.WaterRate_BBLPD
            Else NULL
            END
            )  AS PeakWater_WaterRate
        ,Min(CASE 
            WHEN PP.GasRank=1
            THEN PP.CumOil
            Else NULL
            END
            ) AS PeakGas_CumOil
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.CumGas
            Else NULL
            END
            )  AS PeakGas_CumGas
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.CumWater
            Else NULL
            END
            )  AS PeakGas_CumWater
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.ProductionDate
            Else NULL
            END
            )  AS PeakGas_Date
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.DaysOn
            Else NULL
            END
            )  AS PeakGas_DaysOn
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.OilRate_BBLPD
            Else NULL
            END
            )  AS PeakGas_OilRate
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.GasRate_MCFPD
            Else NULL
            END
            )  AS PeakGas_GasRate
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.WaterRate_BBLPD
            Else NULL
            END
            )  AS PeakGas_WaterRate    
         ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*3)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '3MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*6)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '6MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*9)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '9MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*12)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '12MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*15)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '15MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*18)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '18MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*21)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '21MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*24)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '24MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*3)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '3MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*6)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '6MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*9)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '9MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*12)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '12MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*15)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '15MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*18)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '18MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*21)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '21MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*24)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '24MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*3)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '3MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*6)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '6MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*9)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '9MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*12)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '12MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*15)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '15MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*18)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '18MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*21)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '21MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*24)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '24MonthWater'
    FROM (
    SELECT
        P.*
        ,RANK() OVER (PARTITION BY P.StateProducingUnitKey
                    ORDER BY P.OilRate_BBLPD DESC) as OilRank
        ,RANK() OVER (PARTITION BY P.StateProducingUnitKey
                    ORDER BY P.GasRate_MCFPD DESC) as GasRank
        ,RANK() OVER (PARTITION BY P.StateProducingUnitKey
                    ORDER BY P.WaterRate_BBLPD DESC) as WaterRank
    FROM(
        SELECT 
                    StateProducingUnitKey   
                    ,Max(ProductionDate) as ProductionDate
                    ,Max(DaysProducing) as DaysProducing
                    ,Sum(Oil) as Oil
                    ,Sum(Gas) as Gas
                    ,Sum(Water) as Water
                    ,Sum(CO2) as CO2
                    ,Sum(Sulfur) as Sulfur
                    ,Sum(Oil) OVER (Partition By StateProducingUnitKey 
                                    Order By ProductionDate
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND 0 PRECEDING) CumOil
                    ,Sum(Gas) OVER (Partition By StateProducingUnitKey 
                                    Order By ProductionDate
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND 0 PRECEDING) CumGas
                    ,Sum(Water) OVER (Partition By StateProducingUnitKey 
                                    Order By ProductionDate
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND 0 PRECEDING) CumWater      
                    ,Sum(MAX(DaysProducing)) OVER (Partition By StateProducingUnitKey 
                                    Order By ProductionDate
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND 0 PRECEDING) DaysOn
                    ,Sum(Oil) / MAX(DaysProducing) as OilRate_BBLPD
                    ,Sum(Gas) / MAX(DaysProducing) as GasRate_MCFPD
                    ,Sum(Water) / MAX(DaysProducing) as WaterRate_BBLPD
                    FROM PRODUCTION
    --                  WHERE CAST(StateProducingUnitKey as INTEGER) >= 5123449960000 AND CAST(StateProducingUnitKey as INTEGER) <= 5123450000000
                    GROUP BY StateProducingUnitKey, ProductionDate
                    ORDER BY  StateProducingUnitKey,ProductionDate ASC
                    ) P
    WHERE P.DaysOn <= 700
    ORDER BY  StateProducingUnitKey,ProductionDate ASC) PP
    -- WHERE CAST(PP.StateProducingUnitKey as INTEGER) >= 5123449960000 AND CAST(PP.StateProducingUnitKey as INTEGER) <= 5123450000000
    GROUP BY PP.StateProducingUnitKey
    """
    #quit()

    sqldb = "CO_3_2.1.sqlite"
    conn = sqlite3.connect(sqldb)
    c = conn.cursor()
    # c.execute(Query1)

    Well_df=pd.read_sql_query(Q1,conn)

    # Determine how to aggregate treatments
    c = conn.cursor()
    c.execute(Q2_Drop)
    c.execute(Q2.split(';')[0])
   # c.execute(Q2.split(';')[1)
   # df = pd.read_sql_query(Q2.split(';')[2],conn)
    df = pd.read_sql_query(Q2.split(';')[1],conn)
    df.loc[df.PastTreatmentDate!=None]
    df2 = pd.merge(df,pd.DataFrame(df).pivot_table(index=['StateWellKey'],aggfunc='size').rename('Count'),how='left',left_on='StateWellKey',right_on='StateWellKey')
    df2['TDelta']=(pd.to_datetime(df2.DATE1)-pd.to_datetime(df2.DATE2)).astype('timedelta64[M]')
    x=df2.loc[(df2.Cum_Difference<50) | ((df2.TDelta <6) & (df2.Cum_Difference<1000)) | (df2.TDelta < 2)].sort_values(by='Cum_Difference')
    df2['TreatKey2']=0
    df2['TreatKey2']=((df2.Cum_Difference<50) | ((df2.TDelta <6) & (df2.Cum_Difference<1000)) | (df2.TDelta < 2))*1+df2.TreatKey2
    # For Stimulation order A-B-C
    # Dictionary receives stimulation B and returns A
    StimDictPrev = df2.loc[df2.TreatKey2>0].set_index('StateStimulationKey').PastStateStimulationKey.to_dict()
    # Dictionary receives stimulation B and returns C
    StimDict = df2.loc[df2.TreatKey2>0].set_index('PastStateStimulationKey').StateStimulationKey.to_dict()                          
    # StimDict to map and group
    # df2.loc[df2.TreatKey2>0].PastStateStimulationKey.map(StimDict).dropna()
    del df,df2,x                              

    # Get Stimulation Quantities
    STIM_df=pd.read_sql_query(Q3,conn)
    t_df = STIM_df.pivot(index = ['StateStimulationKey'], columns = ['Type'],values = ['TotalAmount','QC_CountUnits']).dropna(axis=1,how='all').dropna(axis=0,how='all')
    t_df.columns = ['_'.join(col).strip() for col in t_df.columns.values]

    STIM_df[['StateStimulationKey','TreatmentDate','StateWellKey']].drop_duplicates()

    STIM_df = pd.merge(t_df
        ,STIM_df[['StateStimulationKey','TreatmentDate','StateWellKey']].drop_duplicates()
        ,how = 'left'
        ,on = ['StateStimulationKey','StateStimulationKey'])
    del t_df
    i = 1
    STIM_df=STIM_df.drop(list(filter(re.compile('StateStimulationKey[0-9]').match, STIM_df.keys().to_list())),axis=1)    
    STIM_df['StateStimulationKey0']=STIM_df['StateStimulationKey']
    while STIM_df['StateStimulationKey'+str(i-1)].map(StimDict).dropna().shape[0]>0:
        STIM_df['StateStimulationKey'+str(i)] = STIM_df['StateStimulationKey'+str(i-1)]
        #STIM_df['StateStimulationKey'+str(i-1)].map(StimDict)
        STIM_df.loc[STIM_df['StateStimulationKey'+str(i-1)].map(StimDict).dropna().index,STIM_df.columns=='StateStimulationKey'+str(i)]=STIM_df['StateStimulationKey'+str(i-1)].map(StimDict).dropna()
        i+=1
    # Stimulation Dates Dictionary
    StimDates=STIM_df[['StateStimulationKey','TreatmentDate']].drop_duplicates().set_index('StateStimulationKey').astype('datetime64').TreatmentDate.to_dict()
    # Count of stimulations per well


    ##################################################################################################      UPDATE ME
        # UPDATE THIS TO TRACK INTERVAL OF COMPLETION ACTIVITY
        # AND ALSO FOLLOW THESE INTERVALS IN PRODUCTION
    ##################################################################################################
    ## Not sure, but I think this whole section has been superceded by lower sections
    ##if 1==1:
    ##    # Sum stimulation volumes where multiple stimulations in a short time
    ##    sumdf = STIM_df[(list(filter(re.compile('Total.*').match, STIM_df.keys().to_list())))+['StateStimulationKey'+str(i-1)]].groupby('StateStimulationKey'+str(i-1)).sum()    
    ##    # Max for QC Counts
    ##    maxdf = STIM_df[(list(filter(re.compile('QC_CountUnit.*').match, STIM_df.keys().to_list())))+['StateStimulationKey'+str(i-1)]].groupby('StateStimulationKey'+str(i-1)).max()
    ##    #Stimulation Summary Table
    ##    SS_df=sumdf.join(maxdf, on='StateStimulationKey'+str(i-1))
    ##    # Add Well API
    ##    SS_df.join(STIM_df[['StateStimulationKey'+str(i-1),'StateWellKey']].set_index('StateStimulationKey'+str(i-1)).drop_duplicates())
    ##    del x,maxdf,sumdf


    STIM_df['LastStimDate']=STIM_df['StateStimulationKey'+str(i-1)].map(StimDates)
    STIM_df[['StateWellKey','StateStimulationKey'+str(i-1),'LastStimDate']].drop_duplicates()
    ranks = STIM_df[['StateWellKey','StateStimulationKey'+str(i-1),'LastStimDate']].drop_duplicates() \
      .groupby(['StateWellKey'])['LastStimDate'] \
      .rank(ascending = True, method = 'first')-1
    ranks = ranks.fillna(0).astype('int64')

    RankDict = pd.Series(ranks.values.astype('int64')
                         ,index=STIM_df[['StateWellKey'
                         ,'StateStimulationKey'+str(i-1)
                         ,'LastStimDate']].drop_duplicates()['StateStimulationKey'+str(i-1)].values).to_dict()

    #ranks.name = 'StimRank'
    STIM_df['StimRank'] = STIM_df['StateStimulationKey'+str(i-1)].map(RankDict)
    STIM_df['StimRank'] = STIM_df['StimRank'].fillna(0)
    STIM_df['API14x']= (STIM_df.StimRank.astype('int64') + STIM_df.StateWellKey.astype('int64')).astype('str').str.zfill(14)

     #   STIM_df['StimDate0'] = STIM_df['StateStimulationKey'+str(i-1)].map(paststim).map(StimDates)
     #   STIM_df['StimDate1'] = STIM_df['StateStimulationKey'+str(i-1)].map(nextstim).map(StimDates)
     #   STIM_df.loc[STIM_df.LastStimDate == STIM_df.StimDate0,'StimDate0']=datetime.date(1900, 1, 1)
     #   STIM_df.loc[STIM_df.StimDate1 == STIM_df.StimDate1,'StimDate1']=datetime.datetime.now()

    # STIM_df.loc[STIM_df.StimRank>0]
    # STIM_df.loc[STIM_df.StimRank>0].StateWellKey
    # 0512340288000
    tbl = STIM_df[['StateWellKey','StateStimulationKey'+str(i-1),'LastStimDate','StimRank']]
    xbl = tbl.loc[tbl.StimRank == tbl.StimRank.max()].set_index('StateWellKey')

    dict1 = {};dict2 = {};
    for j in reversed(range(0,tbl.StimRank.max())):
        #suffix for rank n-1 values
        sfx = '_prev'
        x = pd.merge(tbl.loc[tbl.StimRank == j+1].set_index('StateWellKey')
             ,tbl.loc[tbl.StimRank == j].set_index('StateWellKey')
             ,how = 'inner'
             ,left_index = True
             ,right_index = True
             ,suffixes =('',sfx))

        dict1.update(x[['StateStimulationKey'+str(i-1),'StateStimulationKey'+str(i-1)+sfx]].set_index('StateStimulationKey'+str(i-1))['StateStimulationKey'+str(i-1)+sfx].to_dict())
        dict2.update(x[['StateStimulationKey'+str(i-1)+sfx,'StateStimulationKey'+str(i-1)]].set_index('StateStimulationKey'+str(i-1)+sfx)['StateStimulationKey'+str(i-1)].to_dict())
        # dict2.update(dict1)

    STIM_df['StimDate0'] = STIM_df['StateStimulationKey'+str(i-1)].map(dict2).map(StimDates).fillna(datetime.date(1900, 1, 1))
    STIM_df['StimDate1'] = STIM_df['StateStimulationKey'+str(i-1)].map(dict1).map(StimDates).fillna(datetime.datetime.now())
    STIM_df.loc[STIM_df.LastStimDate == STIM_df.StimDate0,'StimDate0']=datetime.date(1900, 1, 1)
    STIM_df.loc[STIM_df.StimDate1 == STIM_df.StimDate1,'StimDate1']=datetime.datetime.now()

    # 3,6,9,12,15,18,21,24 month cum
    cumdf = pd.read_sql_query(Q4,conn)
    cumdf = cumdf.set_index('StateProducingUnitKey',drop=False)
    #cumdf['StateProducingUnitKey'] = cumdf.index

    # adds empty columns 
    #cumdf = cumdf.reindex(cumdf.columns.tolist() + STIM_df.keys().tolist(), axis=1)
    #cumdf[STIM_df.keys()]=np.nan

    # For speed
    # 1) Assign all wells to 0th conpletion
    # 2) For each well with multiple completions, get assignments
    well_list = cumdf.index.intersection(STIM_df.loc[STIM_df.StimRank>0]['StateWellKey'].unique())

    cumdf = pd.merge(cumdf,STIM_df.drop_duplicates(['StateWellKey']).set_index('StateWellKey'),how='outer',left_index = True, right_index=True)
    STIM_df.StimDate0 = pd.to_datetime(STIM_df.StimDate0)
    cumdf.PeakOil_Date = pd.to_datetime(cumdf.PeakOil_Date)
    for well in well_list:
        x = STIM_df.loc[(STIM_df.StateWellKey == well) \
            & (STIM_df.StimDate0 <= cumdf.loc[well,'PeakOil_Date'])]
        if x.shape[0] ==0:
            continue
        elif x.shape[0] >1:
            x=x.sort_values(by='StimDate0',ascending=False).iloc[0,:]
        else:
            pass
        cumdf.loc[well,STIM_df.keys()]=x.squeeze()

    #StimPivot = STIM_df.pivot_table(values = 'LastStimDate',index = 'StateWellKey',columns = 'StimRank',aggfunc='max')                                                                                                                                        
    #StimPivot.loc[StimPivot.index == '05125121260000'].dropna(axis=1)

    # Get Perforation Detail
    Perf_df = pd.read_sql_query("""SELECT * FROM WellPerf""",conn)
    cumdf=pd.merge(cumdf,Perf_df.drop_duplicates('StateWellKey').set_index('StateWellKey'),how='outer',left_index = True, right_index=True)
    #Perf Interval
    cumdf['PerfInterval'] = cumdf.Bottom-cumdf.Top
    # Frac Per Ft
    cumdf['TotalFluid'] = cumdf['TotalAmount_FRESH WATER']+cumdf['TotalAmount_RECYCLED WATER']
    cumdf['Stimulation_FluidBBLPerFt'] = cumdf['TotalFluid']/cumdf['PerfInterval']
    cumdf['Stimulation_ProppantLBSPerFt'] = cumdf['TotalAmount_PROPPANT']/cumdf['PerfInterval']
    cumdf['Stimulation_ProppantLBSPerBBL'] = cumdf['TotalAmount_PROPPANT']/cumdf['TotalFluid']
    conn.close()

    # MERGE WITH WELL HEADERS
    with sqlite3.connect('CO_3_2.1.sqlite') as conn:
        Well_df=pd.read_sql_query("""SELECT * FROM WELL""",conn)
    ULT = pd.merge(cumdf,Well_df.set_index('StateWellKey'), left_index=True, right_index=True,how='outer')
    for k in ULT.keys():
        if 'DATE' in k.upper():
            ULT[k] = pd.to_datetime(ULT['FirstCompDate'])

    ULT.to_json('SQL_WELL_SUMMARY.JSON')
    ULT.to_parquet('SQL_WELL_SUMMARY.PARQUET')


    # dtypes for SQLITE3
    pd_sql_types={'object':'TEXT',
                  'int64':'INTEGER',
                  'float64':'FLOAT',
                  'bool':'TEXT',
                  'datetime64':'TEXT',
                  'datetime64[ns]':'TEXT',
                  'timedelta[ns]':'REAL',
                  'category':'TEXT'
        }

    #dtypes for SQLALCHEMY
    pd_sql_types={'object':sqlalchemy.types.Text,
                  'int64':sqlalchemy.types.BigInteger,
                  'float64':sqlalchemy.types.Float,
                  'bool':sqlalchemy.types.Boolean,
                  'datetime64':sqlalchemy.types.Date,
                  'datetime64[ns]':sqlalchemy.types.Date,
                  'timedelta[ns]':sqlalchemy.types.Float,
                  'category':sqlalchemy.types.Text
        }

    df_typemap = ULT.dtypes.astype('str').map(pd_sql_types).to_dict()

    engine = sqlalchemy.create_engine('sqlite:///prod_data.db')
    TABLE_NAME = 'Well_Summary'
    with engine.begin() as connection:
        ULT.to_sql(TABLE_NAME,connection,
                          if_exists='replace',
                          index=False,
                          dtype=df_typemap)

