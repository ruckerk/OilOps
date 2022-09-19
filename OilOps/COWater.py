from . import _FUNCS_
from .MAP import Pt_Distance, Pt_Bearing,county_from_LatLon

def DWR_GEOPHYSWELLSUMMARY(LAT,LON, RADIUS = 1, RADIUS_UNIT = 'miles'):
    # EXPECTING WGS84 COORDINATES
    QTERMS = ['wellId',
        'wellName',
        'permit',
        'ogccId',
        'locnum',
        'wellDepth',
        'aquiferPicks',
        'latitude',
        'longitude',
        'elevation']
        
    COUNTY = county_from_LatLon(LAT,LON)
    
    URL_ROUTE = 'https://dwr.state.co.us/Rest/GET/api/v2/groundwater/geophysicallogs/wells/'
    QUERY = {'format':'jsonnforced',
            'county':COUNTY,
            'latitude':LAT,
            'longitude':LON,
            'radius':RADIUS,
            'units':RADIUS_UNIT}
    RESULT = _FUNCS_.requests.get(url = URL_ROUTE, params = QUERY)

    df = _FUNCS_.pd.DataFrame(columns = QTERMS)
    for r in RESULT.json()['ResultList']:
        i = df.shape[0]
        for k in QTERMS:
            df.loc[i,k] = r[k]
    return(df)


def DWR_GEOPHYSTOPS(WELLIDS):
    QTERMS = ['wellId',
        'wellName',
        'aquifer',
        'gLogTopDepth',
        'gLogBaseDepth',
        'gLogTopElev',
        'gLogBaseElev']
    
    URL_ROUTE = 'https://dwr.state.co.us/Rest/GET/api/v2/groundwater/geophysicallogs/geoplogpicks/'

    if not isinstance(WELLIDS, _FUNCS_.collections.Iterable) or isinstance(WELLIDS,str):
        WELLIDS = [WELLIDS]

    df = _FUNCS_.pd.DataFrame(columns = [QTERMS])
    for WELLID in WELLIDS:  
        QUERY = {'format':'jsonnforced',
                'wellID':WELLID}
        RESULT = _FUNCS_.requests.get(url = URL_ROUTE, params = QUERY)

        for r in RESULT.json()['ResultList']:
            i = df.shape[0]
            for k in QTERMS:
                df.loc[i,k] = r[k]
    return(df)    

def DWR_WATERPERMITS(LAT,LON, RADIUS = 1, RADIUS_UNIT = 'miles'):
    # EXPECTING WGS84 COORDINATES
    QTERMS = ['receipt',
        'permit',
        'permitCurrentStatusDescr',
        'locationType',
        'latitude',
        'longitude',
        'dateWellPlugged',
        'associatedAquifers',
        'elevation',
        'depthTotal',
        'topPerforatedCasing',
        'bottomPerforatedCasing',
        'staticWaterLevel',
        'staticWaterLevelDate',
        'wdid',
        'moreInformation']
        
    COUNTY = county_from_LatLon(LAT,LON)
    
    URL_ROUTE = 'https://dwr.state.co.us/Rest/GET/api/v2/wellpermits/wellpermit/'
    QUERY = {'format':'jsonforced',
            'county':COUNTY,
            'latitude':LAT,
            'longitude':LON,
            'radius':RADIUS,
            'units':RADIUS_UNIT}
    
    RESULT = _FUNCS_.requests.get(url = URL_ROUTE, params = QUERY)

    df = _FUNCS_.pd.DataFrame(columns = QTERMS)
    for r in RESULT.json()['ResultList']:
        i = df.shape[0]
        for k in QTERMS:
            df.loc[i,k] = r[k]
    return(df)

def DWR_WATERWELLLEVELS(LAT,LON, RADIUS = 1, RADIUS_UNIT = 'miles'):
    # EXPECTING WGS84 COORDINATES
    QTERMS = ['wellId',
        'locationNumber',
        'usgsSiteId',
        'wellName',
        'wellDepth',
        'elevation',
        'aquifers',
        'measurementDate',
        'waterLevelDepth',
        'waterLevelElevation',
        'latitude',
        'longitude',
        'moreInformation']
        
    COUNTY = county_from_LatLon(LAT,LON)
    
    URL_ROUTE = 'https://dwr.state.co.us/Rest/GET/api/v2/groundwater/waterlevels/wells/'
    QUERY = {'format':'jsonforced',
            'county':COUNTY,
            'latitude':LAT,
            'longitude':LON,
            'radius':RADIUS,
            'units':RADIUS_UNIT}
    
    RESULT = _FUNCS_.requests.get(url = URL_ROUTE, params = QUERY)

    df = _FUNCS_.pd.DataFrame(columns = QTERMS)
    for r in RESULT.json()['ResultList']:
        i = df.shape[0]
        for k in QTERMS:
            df.loc[i,k] = r[k]
    return(df)


def WaterDataPull(LAT=40.5832238,LON=-104.0990673,RADIUS=10):
    # Lat/Lon as WGS84
    # RADIUS in miles
    
    headers = {
        'Accept': 'application/zip',
    }

    params = (
        ('mimeType', 'xlsx'),
        ('zip', 'yes'),
    )

    json_data = {
        'within': 30,
        'lat': str(LAT),
        'long': str(LON),
        'siteType': [
            'Well',
            'Subsurface',
            'Facility',
            'Aggregate groundwater use',
            'Not Assigned',
        ],
        'sampleMedia': [
            'Water',
            'water',
            'Other',
            'No media',
        ],
        'characteristicName': [
            'Fixed dissolved solids',
            'Total dissolved solids',
            'Total solids',
            'Solids',
            'Dissolved solids',
            'Total fixed solids',
            'Fixed suspended solids',
            'Percent Solids',
            'Total suspended solids',
            'Salinity',
        ],
        'startDateLo': '01-01-1900',
        'startDateHi': _FUNCS_.datetime.datetime.now().strftime("%m-%d-%Y"),
        'dataProfile': 'resultPhysChem',
        'providers': [
            'NWIS',
            'STEWARDS',
            'STORET',
        ],
    }

    #response = _FUNCS_.requests.get('https://www.waterqualitydata.us/data/Result/search', headers=headers, params=params, stream = True)
    r_data = _FUNCS_.requests.post('https://www.waterqualitydata.us/data/Result/search', headers=headers, params=params, json=json_data)
    #r_station = _FUNCS_.requests.post('https://www.waterqualitydata.us/data/Station/search', headers=headers, params=params, json=json_data)

    r_station = _FUNCS_.requests.get("https://www.waterqualitydata.us/data/Station/search?within=" + str(RADIUS) + "&lat=" +  str(LAT)+'&long=' + str(LON) + "&siteType=Well&siteType=Subsurface&siteType=Facility&siteType=Aggregate%20groundwater%20use&siteType=Not%20Assigned&sampleMedia=Water&sampleMedia=water&sampleMedia=Other&sampleMedia=No%20media&characteristicName=Total%20dissolved%20solids&characteristicName=Dissolved%20solids&characteristicName=Total%20solids&characteristicName=Total%20suspended%20solids&characteristicName=Fixed%20dissolved%20solids&characteristicName=Fixed%20suspended%20solids&characteristicName=Solids&characteristicName=Percent%20Solids&characteristicName=Total%20fixed%20solids&characteristicName=Salinity&startDateLo=01-01-1900&startDateHi=01-01-2030&mimeType=xlsx&zip=yes&providers=NWIS&providers=STEWARDS&providers=STORET")
    
    #zipfile = ZipFile(BytesIO(r.content))
    #f = zipfile.namelist()[0]
    #pd.read_excel(zipfile.open(f,mode = 'r')).keys()
    
    
    # Note: original query string below. It seems impossible to parse and
    # reproduce query strings 100% accurately so the one below is given
    # in case the reproduced version is not "correct".
    #response = requests.post('https://www.waterqualitydata.us/data/Result/search?mimeType=csv&zip=yes', headers=headers, json=json_data)
    return r_data,r_station

def Summarize_WaterChem(r1,r2, LAT, LON):
    # ASSUMES NAD87 EPSG 4269 COORDINATES FOR USGS
    
    zf = _FUNCS_.ZipFile(BytesIO(r1.content))
    f = zf.namelist()[0]
    df = _FUNCS_.pd.read_excel(zf.open(f,mode = 'r'))

    zf = _FUNCS_.ZipFile(BytesIO(r2.content))
    f = zf.namelist()[0]
    df2 = _FUNCS_.pd.read_excel(zf.open(f,mode = 'r'))

    LOCATIONS = df2['MonitoringLocationIdentifier'].unique()
    
    df = df.loc[(df.CharacteristicName.str.contains('solid',case=False)) & (df.CharacteristicName.str.contains('dissolve',case=False))]
    df = df.loc[df['ResultMeasure/MeasureUnitCode'].str.contains('mg')==True]
    df = df.loc[df['ActivityMediaSubdivisionName'].str.contains('Ground',case=False)==True]

    LONG_KEYS = df.keys()[df.keys().str.contains('longitude',case=False)].to_list()
    LAT_KEYS = df.keys()[df.keys().str.contains('latitude',case=False)].to_list()
    
    df2['PTS']=list(df2[['LongitudeMeasure','LatitudeMeasure']].to_records(index=False))

    df2['Distance'] = df2['PTS'].apply(Pt_Distance,pt2=(LON,LAT))
    df2['Bearing'] = df2['PTS'].apply(Pt_Bearing,pt2=(LON,LAT))

    df3 = df.merge(df2[['MonitoringLocationIdentifier','WellDepthMeasure/MeasureValue','Distance','Bearing']],left_on='MonitoringLocationIdentifier',right_on='MonitoringLocationIdentifier',how='outer')
    df3 = df3.loc[df3['ResultMeasureValue'].dropna().index]

    df2 = df2.loc[df2['MonitoringLocationIdentifier'].isin(df3['MonitoringLocationIdentifier'])]

    #DATA = df2.loc[(df2[GetKey(df2,'depth.*value')].max(axis=1)<=MAXDEPTH) & (df2[GetKey(df2,'depth.*value')].max(axis=1)>=MINDEPTH)].index
    DATA = df2.loc[:,'MonitoringLocationIdentifier'].isin(df3['MonitoringLocationIdentifier']).index

    #DATA = df2.loc[DATA,'Distance'].nsmallest(50).index
    DATA = df2.loc[DATA].groupby(by='MonitoringLocationIdentifier')['Distance'].min().nsmallest(1000).index
    DATA = list(DATA)
   
    #DATA = df2.loc[DATA,'MonitoringLocationIdentifier'].unique()
    DATA_LOCS = df2.loc[df2['MonitoringLocationIdentifier'].isin(DATA) & df2['Distance']>0,['MonitoringLocationIdentifier','Distance','Bearing']]

    OUTCOLS = ['MonitoringLocationIdentifier'
               ,'ActivityStartDate'
               ,'CharacteristicName'
               ,'ResultMeasureValue'
               ,'ResultMeasure/MeasureUnitCode']
    OUTCOLS = OUTCOLS + GetKey(df3,'depth.*value')

    RESULT = df3.loc[(df3['MonitoringLocationIdentifier'].isin(DATA)),OUTCOLS]
    #RESULT = RESULT.drop(['Distance', 'Bearing'],axis=1)
    RESULT = RESULT.merge(DATA_LOCS,how='outer',on = 'MonitoringLocationIdentifier')
    return(RESULT)

def COWATER_SUMMARY(LAT=40.5832238,LON=-104.0990673,RADIUS=10):
    T = _FUNCS_.Transformer.from_crs('EPSG:4326', 'EPSG:4269',always_xy =True)
    LLON2,LAT2 = T.transform(LON,LAT)
    
    r1,r2 = WaterDataPull(LAT2 ,LLON2 ,RADIUS)
    df_OUT = Summarize_WaterChem(r1,r2,LAT2,LLON2)    
    return(df_OUT)
