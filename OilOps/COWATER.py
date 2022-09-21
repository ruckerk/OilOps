from ._FUNCS_ import *
from .MAP import Pt_Distance, Pt_Bearing, county_from_LatLon, convert_XY, DistAzi
from shapely.geometry import Point

# Upgrade Ideas:
# Incorporate a DEM to add elevations to water well depths and convert to subsea

__all__ = [
    'DWR_GEOPHYSWELLSUMMARY',
    'DWR_GEOPHYSTOPS',
    'DWR_WATERPERMITS',
    'DWR_WATERWELLLEVELS',
    'WaterDataPull',
    'Summarize_WaterChem',
    'COWATER_QUALITY',
    'CO_WATERWELL_SUMMARY'
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
    df['moreInformation'] = df['moreInformation'].str.strip()
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
    df['moreInformation'] = df['moreInformation'].str.strip()
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

def COWATER_QUALITY(LAT=40.5832238,LON=-104.0990673,RADIUS=10):
    T = Transformer.from_crs('EPSG:4326', 'EPSG:4269',always_xy =True)
    LLON2,LAT2 = T.transform(LON,LAT)
    
    r1,r2 = WaterDataPull(LAT2 ,LLON2 ,RADIUS)
    df_OUT = Summarize_WaterChem(r1,r2,LAT2,LLON2)    
    return(df_OUT)

def CO_WATERWELL_SUMMARY(LAT,LON,RADIUS = 1,UNITS = 'miles', EPSG_IN = 4269, DATA = False):

    lon, lat = convert_XY(LON,LAT,EPSG_IN,4326)

    if UNITS=='miles':
        REQUEST_RADIUS = RADIUS + 4
    elif UNITS == 'feet':
        REQUEST_RADIUS = RADIUS + 4*5280

    df_gwells = DWR_GEOPHYSWELLSUMMARY(lat,lon,REQUEST_RADIUS,UNITS)
    df_gtops = DWR_GEOPHYSTOPS(df_gwells.wellId)
    df_permits = DWR_WATERPERMITS(lat,lon,REQUEST_RADIUS,UNITS)
    df_levels = DWR_WATERWELLLEVELS(lat,lon,REQUEST_RADIUS,UNITS)

    df_permits['DEPTH'] = df_permits[['depthTotal','bottomPerforatedCasing']].max(axis=1)
    df_permits['loc']=df_permits[['latitude','longitude']].astype(str).apply('_'.join, axis = 1)
    df_permits['MAX_DEPTH'] = df_permits['loc'].map(df_permits.groupby('loc')['DEPTH'].max())
    m_max_depth = df_permits.loc[df_permits['MAX_DEPTH']==df_permits['DEPTH']].index
    df_permits.drop('loc',axis = 1, inplace = True)

    df_TOPS = df_gtops.merge(df_gwells, on = 'wellId')

    # calculate distance/azimuth from location to each well
    df_TOPS[['DISTANCE','AZIMUTH']] = pd.DataFrame(df_TOPS.apply(lambda row: DistAzi(lat,lon,row['latitude'],row['longitude'],4326),axis=1).tolist())
    df_permits[['DISTANCE','AZIMUTH']] = pd.DataFrame(df_permits.apply(lambda row: DistAzi(lat,lon,row['latitude'],row['longitude'],4326),axis=1).tolist())

    # convert meters to feet
    df_TOPS['DISTANCE'] = df_TOPS['DISTANCE'] * 3.28084
    df_permits['DISTANCE'] = df_permits['DISTANCE'] * 3.28084
    
    #tops inside 1 mile
    m_radius = df_TOPS.index[df_TOPS['DISTANCE']<=5280]

    #wells inside 1 mile
    m_permit_radius = df_permits.index[df_permits['DISTANCE']<=5280]
    m_permit_radius_plus = df_permits.index[(df_permits['DISTANCE']<=6400) & (df_permits['DISTANCE']>5280)]

    # convert to feet
    x, y = convert_XY(lon, lat, 4326, 2231)
    dummy = pd.DataFrame(shapely.geometry.Point(x,y).buffer(5280).exterior.xy).T
    
    df_1MileRing = pd.DataFrame(dummy.apply(lambda r: convert_XY(r[0],r[1],2231,4326), axis=1).to_list())
    df_1MileRing.columns = ['LON','LAT']
    
    # get extents in area
    # NEEDS UPDATE TO CALC FROM UNITS AND RADIUS INPUT, USE geom.fwd?
    lon0, lat0 = convert_XY(x-6000,y-6000,2231,4326)
    lon1, lat1 = convert_XY(x+6000,y+6000,2231,4326)

    ext = (min(lon0,lon1),max(lon0,lon1),min(lat0,lat1),max(lat0,lat1))
    grid_x, grid_y = np.meshgrid(np.arange(min(lon1,lon0),max(lon1,lon0),3e-5), np.arange(min(lat0,lat1),max(lat0,lat1),3e-5))

    FXHLLS = [a for a in df_TOPS['aquifer'].unique() if 'FOX' in a.upper()]

    PROJECTIONS = dict()
    PROJECTIONS['DEEPEST_WELL'] = df_permits.loc[m_permit_radius,'DISTANCE'].max()
  
    for AQ in df_TOPS['aquifer'].unique():
        m = df_TOPS.index[df_gtops['aquifer']==AQ]
        mtop = df_TOPS.loc[m, 'gLogTopElev'].dropna().index
        mbase = df_TOPS.loc[m, 'gLogBaseElev'].dropna().index

        pts_top = df_TOPS.loc[mtop, ['longitude','latitude']]
        pts_base = df_TOPS.loc[mbase, ['longitude','latitude']]
        vals_top = df_TOPS.loc[mtop, 'gLogTopElev']
        vals_base = df_TOPS.loc[mbase, 'gLogBaseElev']

        if len(mtop)>=3:
            z_well = interpolate.griddata(pts_top,vals_top, (lon, lat), method='linear').mean()
            z_well = round(z_well,1)

            PROJECTIONS[AQ+'_TOP'] = z_well

        if len(mbase)>=3:
            z_well = interpolate.griddata(pts_base,vals_base, (lon, lat), method='linear').mean()
            z_well = round(z_well,1)

            PROJECTIONS[AQ+'_BASE'] = z_well

            if (AQ in FXHLLS):
                #FOX_BASE_Z = interpolate.griddata(pts_base,vals_base, (grid_x, grid_y), method='linear')

                try:
                    if z_well >= max(PROJECTIONS[k] for k in PROJECTIONS.keys() if k.removesuffix('_BASE') in FXHLLS):
                         base_z = interpolate.griddata(pts_base,vals_base, (grid_x, grid_y), method='linear')
                         base_label = AQ+'_BASE'
                except:
                    base_z = interpolate.griddata(pts_base,vals_base, (grid_x, grid_y), method='linear')
                    base_label = AQ+'_BASE'
                    

    
    #if deepest well near 1mile limit, create plot
    #df_permits.loc[m_permit_radius,'DISTANCE'].max() < df_permits.loc[m_permit_radius_plus,'DISTANCE'].max()
    if True:
        fig = plt.figure(figsize=(10, 10))
        
        #create meshgrid for radius
        grid_x, grid_y = np.meshgrid(np.arange(min(lon1,lon0),max(lon1,lon0),3e-5), np.arange(min(lat0,lat1),max(lat0,lat1),3e-5))
        
        if 'base_z' in locals():
            surface = plt.pcolormesh(grid_x, grid_y, base_z, cmap = 'viridis')
        
        plt.plot(df_1MileRing['LON'],df_1MileRing['LAT'],'blue')

        cntrl1 = plt.scatter(df_permits.loc[m_max_depth.join(m_permit_radius, how='inner'),'longitude'],
                             df_permits.loc[m_max_depth.join(m_permit_radius, how='inner'),'latitude'],
                             marker = 'o', edgecolor='k', facecolor = 'white', alpha =0.5, label = 'Wells inside 1 mi.')
        cntrl2 = plt.scatter(df_permits.loc[m_max_depth.join(m_permit_radius_plus, how='inner'),'longitude'],
                             df_permits.loc[m_max_depth.join(m_permit_radius_plus, how='inner'),'latitude'],
                             marker = 's', edgecolor='k', facecolor = 'blue', label = 'Wells outside 1 mi.')

        pad = plt.scatter(lon,lat, marker='s', color ='red', label = 'Pad Location')

        for i in m_max_depth.join(m_permit_radius.join(m_permit_radius_plus, how='outer'), how='inner'):
                plt.annotate(df_permits.loc[i,'MAX_DEPTH'].astype(int), (df_permits.loc[i,'longitude'], df_permits.loc[i,'latitude']))
                
        if 'z_well' in locals():
            plt.annotate(z_well, (lon, lat), c='r')
            
        plt.xlim(min(lon0,lon1),max(lon0,lon1))
        plt.ylim(min(lat0,lat1),max(lat0,lat1))
        
        if 'base_z' in locals():
            cbar = plt.colorbar(surface)
            cbar.set_label(base_label+' ELEVATION')
        plt.legend(loc = 'lower right')
        plt.title('Nearby Water Wells')
        
        
    PROJECTIONS = pd.DataFrame(list(PROJECTIONS.items()),columns = ['TOP','ELEVATION'])
    
    if DATA:
        return (df_permits,df_TOPS,PROJECTIONS, fig)
    else:
        return (PROJECTIONS, fig)
