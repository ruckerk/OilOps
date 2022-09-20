from ._FUNCS_ import *
from .MAP import Pt_Distance, Pt_Bearing,county_from_LatLon

__all__ = [
    'DWR_GEOPHYSWELLSUMMARY',
    'DWR_GEOPHYSTOPS',
    'DWR_WATERPERMITS',
    'DWR_WATERWELLLEVELS',
    'WaterDataPull',
    'Summarize_WaterChem',
    'COWATER_SUMMARY',
    'Co_WaterWell_Summary'
]

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
    RESULT = requests.get(url = URL_ROUTE, params = QUERY)

    df = pd.DataFrame(columns = QTERMS)
    for r in RESULT.json()['ResultList']:
        i = df.shape[0]
        for k in QTERMS:
            df.loc[i,k] = r[k]
    df.columns = df.keys().get_level_values(0)
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

    if not isinstance(WELLIDS, collections.Iterable) or isinstance(WELLIDS,str):
        WELLIDS = [WELLIDS]

    df = pd.DataFrame(columns = [QTERMS])
    for WELLID in WELLIDS:  
        QUERY = {'format':'jsonnforced',
                'wellID':WELLID}
        RESULT = requests.get(url = URL_ROUTE, params = QUERY)

        for r in RESULT.json()['ResultList']:
            i = df.shape[0]
            for k in QTERMS:
                df.loc[i,k] = r[k]
    df.columns = df.keys().get_level_values(0)
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
    
    RESULT = requests.get(url = URL_ROUTE, params = QUERY)

    df = pd.DataFrame(columns = QTERMS)
    for r in RESULT.json()['ResultList']:
        i = df.shape[0]
        for k in QTERMS:
            df.loc[i,k] = r[k]
    df.columns = df.keys().get_level_values(0)
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
    
    RESULT = requests.get(url = URL_ROUTE, params = QUERY)

    df = pd.DataFrame(columns = QTERMS)
    for r in RESULT.json()['ResultList']:
        i = df.shape[0]
        for k in QTERMS:
            df.loc[i,k] = r[k]
    df.columns = df.keys().get_level_values(0)
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
        'startDateHi': datetime.datetime.now().strftime("%m-%d-%Y"),
        'dataProfile': 'resultPhysChem',
        'providers': [
            'NWIS',
            'STEWARDS',
            'STORET',
        ],
    }

    #response = requests.get('https://www.waterqualitydata.us/data/Result/search', headers=headers, params=params, stream = True)
    r_data = requests.post('https://www.waterqualitydata.us/data/Result/search', headers=headers, params=params, json=json_data)
    #r_station = requests.post('https://www.waterqualitydata.us/data/Station/search', headers=headers, params=params, json=json_data)

    r_station = requests.get("https://www.waterqualitydata.us/data/Station/search?within=" + str(RADIUS) + "&lat=" +  str(LAT)+'&long=' + str(LON) + "&siteType=Well&siteType=Subsurface&siteType=Facility&siteType=Aggregate%20groundwater%20use&siteType=Not%20Assigned&sampleMedia=Water&sampleMedia=water&sampleMedia=Other&sampleMedia=No%20media&characteristicName=Total%20dissolved%20solids&characteristicName=Dissolved%20solids&characteristicName=Total%20solids&characteristicName=Total%20suspended%20solids&characteristicName=Fixed%20dissolved%20solids&characteristicName=Fixed%20suspended%20solids&characteristicName=Solids&characteristicName=Percent%20Solids&characteristicName=Total%20fixed%20solids&characteristicName=Salinity&startDateLo=01-01-1900&startDateHi=01-01-2030&mimeType=xlsx&zip=yes&providers=NWIS&providers=STEWARDS&providers=STORET")
    
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
    
    zf = ZipFile(BytesIO(r1.content))
    f = zf.namelist()[0]
    df = pd.read_excel(zf.open(f,mode = 'r'))

    zf = ZipFile(BytesIO(r2.content))
    f = zf.namelist()[0]
    df2 = pd.read_excel(zf.open(f,mode = 'r'))

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
    T = Transformer.from_crs('EPSG:4326', 'EPSG:4269',always_xy =True)
    LLON2,LAT2 = T.transform(LON,LAT)
    
    r1,r2 = WaterDataPull(LAT2 ,LLON2 ,RADIUS)
    df_OUT = Summarize_WaterChem(r1,r2,LAT2,LLON2)    
    return(df_OUT)

def Co_WaterWell_Summary():
    lon, lat = OilOps.MAP.convert_XY(-104.6665798,40.0283805,4269,4326)
    df_gwells = OilOps.COWater.DWR_GEOPHYSWELLSUMMARY(lat,lon,5,'miles')
    df_gtops = OilOps.COWater.DWR_GEOPHYSTOPS(df_gwells.wellId)
    df_permits = OilOps.COWater.DWR_WATERPERMITS(lat,lon,5,'miles')
    df_levels = OilOps.COWater.DWR_WATERWELLLEVELS(lat,lon,5,'miles')
    
    df_TOPS = df_gtops.merge(df_gwells, on = 'wellId')
    df_TOPS[['DISTANCE','AZIMUTH']] = pd.DataFrame(df_TOPS.apply(lambda row: OilOps.MAP.DistAzi(lat,lon,row['latitude'],row['longitude'],4326),axis=1).tolist())
    df_TOPS['DISTANCE'] = df_TOPS['DISTANCE'] / 0.3048
    
    m_radius = df_TOPS.index[df_TOPS['DISTANCE']<=5280]
    
    x, y = OilOps.MAP.convert_XY(lon, lat, 4326, 2232)
    
    lon0, lat0 = OilOps.MAP.convert_XY(x-6000,y-6000,2232,4269)
    lon1, lat1 = OilOps.MAP.convert_XY(x+6000,y+6000,2232,4269)
    
    ext = (min(lon0,lon1),max(lon0,lon1),min(lat0,lat1),max(lat0,lat1))
    
    grid_x, grid_y = np.meshgrid(np.arange(min(lon1,lon0),max(lon1,lon0),3e-5), np.arange(min(lat0,lat1),max(lat0,lat1),3e-5))
