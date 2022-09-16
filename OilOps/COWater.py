from _FUNCS_ import *

def DWR_GEOPHYSWELLSUMMARY(LAT,LON, RADIUS = 1, RADIUS_UNIT = 'miles'):
    # EXPECTING WGS84 COORDINATES
    QTERMS = ['wellId',
        'aquiferPicks',
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
    return(df)


def DWR_GEOPHYSTOPS(WELLIDS):
    QTERMS = ['wellId',
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
    return(df)

def DWR_WATERWELLLEVELS(LAT,LON, RADIUS = 1, RADIUS_UNIT = 'miles'):
    # EXPECTING WGS84 COORDINATES
    QTERMS = ['ResultList',
        'wellId',
        'locationNumber',
        'wellName',
        'wellDepth',
        'elevation',
        'aquifers',
        'waterLevelDepth',
        'waterLevelElevation',
        'latitude',
        'longitude']
        
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
    return(df)
