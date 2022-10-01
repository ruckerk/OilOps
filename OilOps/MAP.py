
from ._FUNCS_ import *
from ._MAPFUNCS_ import *
from .WELLAPI import *

__all__ = ['EPSG_CODES',
    'shapely_to_pyshp',
    'reverse_geom',
    'CenterOfGeom',
    'DXF_to_geojson',
    'GEOJSONLIST_to_SHP',
    'DAT_to_GEOJSONLIST',
    'SHP_to_GEOJSONLIST',
    'GEOJSONLIST_to_SHAPELY',
    'CRS_FROM_SHAPE',
    'SHP_DISTANCES',
    'IN_TC_AREA',
    'GROUP_IN_TC_AREA',
    'convert_XY',
    'county_from_LatLon',
    'Pt_Distance',
    'Pt_Bearing',
    'DistAzi',
    'elevation_function',
    'Items_in_Polygons']

def EPSG_CODES():
    
    print('''COMMON EPSG CODES
      NAD83 GRS 80: 4269
      NAD27 CLARK 66: 4267
      WGS84 WGS84: 4326
      UTM CO 13N: 26913
      NAD83 STATE PLANE: 2232''')


def shapely_to_pyshp(geom, GEOJ = False):
    # first convert shapely to geojson
    #try:
    #    shapelytogeojson = shapely.geometry.mapping
    #except:
    #    import shapely.geometry
    #    shapelytogeojson = shapely.geometry.mapping
    #geoj = shapelytogeojson(shapelygeom)
    if GEOJ:
        geoj = geom
    else:
        geoj = geom.__geo_interface__
 
    # create empty pyshp shape
    #record = shp._Shape()
    record = shp.Shape
    
    # set shapetype
    if geoj["type"] == "Null":
        pyshptype = 0
    elif geoj["type"] == "Point":
        pyshptype = 1
    elif geoj["type"] == "LineString":
        pyshptype = 3
    elif geoj["type"] == "Polygon":
        pyshptype = 5
    elif geoj["type"] == "MultiPoint":
        pyshptype = 8
    elif geoj["type"] == "MultiLineString":
        pyshptype = 3
    elif geoj["type"] == "MultiPolygon":
        pyshptype = 5
    record.shapeType = pyshptype
    # set points and parts
    if geoj["type"] == "Point":
        record.points = geoj["coordinates"]
        record.parts = [0]
    elif geoj["type"] in ("MultiPoint","Linestring"):
        record.points = geoj["coordinates"]
        record.parts = [0]
    elif geoj["type"] in ("Polygon"):
        record.points = geoj["coordinates"][0]
        record.parts = [0]
    elif geoj["type"] in ("MultiPolygon","MultiLineString"):
        index = 0
        points = []
        parts = []
        for eachmulti in geoj["coordinates"]:
            points.extend(eachmulti[0])
            parts.append(index)
            index += len(eachmulti[0])
        record.points = points
        record.parts = parts
    return record

def reverse_geom(geom):
    def _reverse(x, y, z=None):
        if z:
            return x[::-1], y[::-1], z[::-1]
        return x[::-1], y[::-1]
    return shapely.ops.transform(_reverse, geom)

def CenterOfGeom(geom):
    [_x,_y] = geom.centroid.coords.xy
    return [_x[0],_y[0]]

def DXF_to_geojson(DXFFILE):
    DXFDOC = ezdxf.readfile(DXFFILE)
    MSP = DXFDOC.modelspace()
    OUTLIST = list()
    for el in MSP:
        #if "LINE" in el.dxftype():
        #    geo.proxy(el).__geo_interface__
    #    for point in el.vertices:
        #        point.dxf.location
        try:
            g=geo.proxy(el).__geo_interface__
            OUTLIST.append(g)
        except:
            pass
    return OUTLIST

def GEOJSONLIST_to_SHP(GEOJLIST,OUT_BASEFILENAME, FOLDER = False):
    if FOLDER == False:
        pathname = path.dirname(argv[0])
        adir = path.abspath(pathname)
    else:
        adir = FOLDER
        
    for t in set([l['type'] for l in GEOJLIST]):
       with shp.Writer(OUT_BASEFILENAME+'_'+t.upper()) as shapewriter:
           stype = shapely_to_pyshp([l for l in L if l['type'] == t][0],True).shapeType
           shapewriter.shapeType = stype
           shapewriter.field('FAULT_LABEL', 'C')
           for l in L:
               shapewriter.record('REGIONAL WRENCH')
               shapewriter.shape(l)

       with open(path.join(adir,OUT_BASEFILENAME+'_'+t.upper()+'.prj'), "w") as writer:
           try:
               ESRI_WKT = tocrs.to_esri_wkt()
               _ = writer.write(tocrs.to_esri_wkt())
           except:
               url = 'https://spatialreference.org/ref/epsg/EPSGCODE/prj/'
               url = re.sub('EPSGCODE',str(26753),url)
               r = requests.get(url, allow_redirects=True)
               writer.write(r.content.decode())
    return()

def DAT_to_GEOJSONLIST(DATFILE):
    SKIPROWS = FirstNumericRow(DATFILE)
    df = pd.read_csv(DATFILE, header = None, skiprows = SKIPROWS)
    if df.shape[1] == 1:
        df = df.iloc[:,0].str.strip().str.split(r' +',expand=True)
    PGONS = list()
    for i in df.iloc[:,-1].unique():
        m = df.index[df.iloc[:,-1] == i]
        poly = Polygon(df.loc[m,[0,1]].astype(float).values.tolist())
        PGONS.append(poly)
    polygons = [shapely.wkt.loads(p.wkt) for p in PGONS]
    return polygons

def SHP_to_GEOJSONLIST(SHPFILE):
    _SHP = shp.Reader(SHPFILE)
    GLIST = [s.__geo_interface__['geometry'] for s in _SHP]
    return GLIST

def GEOJSONLIST_to_SHAPELY(GEOJSON_IN):  
    GEOCOLLECTION = shapely.geometry.GeometryCollection([shapely.geometry.shape(x) for x in GEOJSON_IN])
    return(GEOCOLLECTION)

def CRS_FROM_SHAPE(SHAPEFILE):
    FPRJ = SHAPEFILE.split('.')[0]+'.prj'
    try:
        pyprojCRS = pycrs.load.from_file(FPRJ)
    except:
        try:
            FPRJ = SHAPEFILE.split('.')[0]+'.PRJ'
            pyprojCRS = pycrs.load.from_file(FPRJ)
        except:    
            warnings.showwarning('No pyprojCRS read from '+SHAPEFILE)
    return(pyprojCRS)

def SHP_DISTANCES(SHP1,SHP2,MAXDIST=10000,CALCEPSG = 26753):
    # delivers nearest distance for each item in SHP1 to any item in SHP2
    # optional MAXDIST for maximum distance of interest

    pyprojCRS1 = CRS_FROM_SHAPE(SHP1)
    pyprojCRS2 = CRS_FROM_SHAPE(SHP2)

    _S1 = SHP_to_GEOJSONLIST(SHP1)
    

    _GS1 = GEOJSONLIST_to_SHAPELY(_S1)
    _GS2 = GEOJSONLIST_to_SHAPELY(_S2)

    _MAINSHAPE = shp.Reader(SHP1)
    _MAIN_DF = read_shapefile(_MAINSHAPE)
        
    # S2 polygons = [shapely.wkt.loads(p.wkt) for p in PGONS]

    # _GS1   FAULTS_MULTIP = MultiPolygon(polygons)
    # FAULTS_8000 = FAULTS_MULTIP.buffer(8000)

    # make same coordinate system in units/projection of interest
    project1 = pyproj.Transformer.from_crs(
        pyproj.CRS.from_wkt(pyprojCRS1.to_ogc_wkt()),
        pyproj.CRS.from_epsg(CALCEPSG),
        always_xy=True).transform

    project2 = pyproj.Transformer.from_crs(
        pyproj.CRS.from_wkt(pyprojCRS2.to_ogc_wkt()),
        pyproj.CRS.from_epsg(CALCEPSG),
        always_xy=False).transform        

    _GS1_RPRJ = shapely.ops.transform(project1, _GS1)
    _GS2_RPRJ = shapely.ops.transform(project2, _GS2)
    
    WORKLIST = [geom if geom.is_valid else geom.buffer(0) for geom in _GS2_RPRJ.geoms]

    boundary = shapely.ops.unary_union([x.buffer(MAXDIST) for x in WORKLIST])
    boundary0 = shapely.ops.unary_union(WORKLIST)
    
    # MAINSHAPE  DLINES = shp.Reader('Directional_Lines.prj')
    # MAIN_DF   dfs = read_shapefile(DLINES)

    _MAIN_DF['GEOMETRY'] = _GS1_RPRJ.geoms
    _MAIN_DF['CALC_DISTANCE'] = MAXDIST
    _MAIN_DF['MidX'] = np.nan
    _MAIN_DF['MidY'] = np.nan

    _MAIN_DF[['MidX','MidY']] = pd.DataFrame(_MAIN_DF['GEOMETRY'].apply(CenterOfGeom).to_list())
    _MAIN_DF['NEARFAULT'] = _MAIN_DF['GEOMETRY'].apply(lambda x:x.intersects(boundary))
    m = _MAIN_DF.index[_MAIN_DF['NEARFAULT']]
    _MAIN_DF.loc[m,'CALC_DISTANCE'] = _MAIN_DF.loc[m,'GEOMETRY'].apply(lambda x:x.distance(boundary0))

    _MAIN_DF.drop('GEOMETRY',axis=1,inplace=True)
    return(_MAIN_DF)


################################################################3
def IN_TC_AREA(well2,tc2):
    # wells is read_shapefile dataframe
    # tc2 is read_shapefile dataframe
    
    if len(well2.coords)>=2:
        try:
            ln =  shapely.geometry.LineString(well2.coords)
        except:
            ln =  shapely.geometry.LineString(well2.coords.values)
    elif len(well2.coords)==1:
        ln =  shapely.geometry.Point(well2.coords[0])
        
    if ln == None:
        return(False)
    test = False
    
    for j in range(0,tc2.shape[0]):
        if test == False:
            poly =  shapely.geometry.Polygon(tc2.coords.iloc[j])
            if ln.intersects(poly.buffer(0)):
                test = True   
    return(test) 


def GROUP_IN_TC_AREA(tc,wells):
    out = pd.DataFrame()
    out['API'] = wells.API_Label.apply(lambda x: WELLAPI(x).API2INT(10))
    #out['API'] = wells.API_Label.str.replace(r'[^0-9]','',regex=True)
    out['TEST'] = wells.apply(lambda x: IN_TC_AREA(x,tc),axis=1)
    return(out)

def convert_XY(X_LON,Y_LAT,EPSG_OLD=4267,EPSG_NEW=4326):
    pyproj.CRS0 = pyproj.CRS.from_epsg(EPSG_OLD)
    pyproj.CRS1 = pyproj.CRS.from_epsg(EPSG_NEW)
    transformer = pyproj.Transformer.from_crs(pyproj.CRS0,pyproj.CRS1,always_xy =True)
    X2, Y2 =transformer.transform(X_LON,Y_LAT)
    return X2,Y2

def county_from_LatLon(LAT,LON):
    geolocator = Nominatim(user_agent="geoapiExercises")
    CNTY = geolocator.reverse(str(LAT)+","+str(LON)).raw['address'].get('county')
    CNTY = CNTY.upper()
    CNTY = CNTY.replace('COUNTY','')
    CNTY = CNTY.strip()
    return(CNTY)

def Pt_Distance(pt1,pt2):
    R = 6373*1000*3.28084
    lon1 = radians(pt1[0])
    lat1 = radians(pt1[1])
    lon2 = radians(pt2[0])
    lat2 = radians(pt2[1])
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return(distance)

def Pt_Bearing(pt1,pt2):
    #Bearing from pt1 to pt2
    R = 6373*1000*3.28084
    lon1 = radians(pt1[0])
    lat1 = radians(pt1[1])
    lon2 = radians(pt2[0])
    lat2 = radians(pt2[1])
    X = cos(lat2)*sin(lon2-lon1)
    Y = cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(lon2-lon1)
    B = atan2(X,Y)
    B = degrees(B)
    return B

def DistAzi(LAT1,LON1,LAT2,LON2, EPSG):
    crs = pyproj.CRS.from_epsg(EPSG)
    geod = crs.get_geod()
    RESULT = geod.inv(LON1,LAT1,LON2,LAT2)
    return(RESULT[2],RESULT[0])
        
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
    
def elevation_function(LAT83, LON83):
    url = r'https://nationalmap.gov/epqs/pqs.php?'
    """Query service using lat, lon. add the elevation values as a new column."""
    params = {
            'output': 'json',
            'x': LON83,
            'y': LAT83,
            'units': 'Feet'
        }
        # format query string and return query value
    result = requests.get((url + urllib.parse.urlencode(params)))
    if '<error>' in result.text:
        ELEVATION = np.nan
    else:
        ELEVATION = result.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation']
    return ELEVATION

def get_openelevation(lat, long, units = 'feet', epsg_in=4269):
    T = Transformer.from_crs('EPSG:'+str(epsg_in), 'EPSG:4326',always_xy =True)
    long,lat = T.transform(long,lat)
    
    query = ('https://api.open-elevation.com/api/v1/lookup'
             f'?locations={lat},{long}')
    
    r = requests.get(query).json()  # json object, various ways you can extract value
    # one approach is to use pandas json functionality:
    
    elevation = pandas.json_normalize(r, 'results')['elevation'].values[0]
    
    if units.upper()=='FEET':
        # meters to feet
        elevation = elevation * 3.28084
   
    return elevation

def Items_in_Polygons(ITEM_SHAPEFILE,POLYGON_SHAPEFILE, BUFFER = None, EPSG4POLY = None):
    ITEMS = shp.Reader(ITEM_SHAPEFILE)
    ITEMS = read_shapefile(ITEMS)
    CRS_ITEMS = CRS_FROM_SHAPE(ITEM_SHAPEFILE)

    POLYS = shp.Reader(POLYGON_SHAPEFILE)
    POLYS = read_shapefile(POLYS)
    CRS_POLYS = CRS_FROM_SHAPE(POLYGON_SHAPEFILE)
    
    TFORMER = pyproj.Transformer.from_crs(pyproj.CRS.from_wkt(CRS_POLYS.to_ogc_wkt()),
                                          pyproj.CRS.from_wkt(CRS_ITEMS.to_ogc_wkt()),
                                          always_xy = True)
    
    ITEMS['coords_old'] = ITEMS['coords']
    POLYS['coords_old'] = POLYS['coords']
    
    NAMES = POLYS.applymap(lambda x:isinstance(x,str)).sum(axis=0).replace(0,np.nan).dropna()
    NAMES = POLYS[list(NAMES.index)].nunique(axis=0).sort_values(ascending=False).index[0]
        
    for i in POLYS.index:   
        NAME = POLYS.loc[i,NAMES]
        
        converted = TFORMER.transform(pd.DataFrame(POLYS.coords_old[0])[0],
                          pd.DataFrame(POLYS.coords_old[0])[1])
        POLYS.at[i,'coords'] = list(map(tuple, np.array(converted).T))
        POLY_SHAPE = shapely.geometry.Polygon(POLYS.loc[i,'coords'] )
        
        RESULT = GROUP_IN_TC_AREA(POLYS.loc[[i],:],ITEMS)
        ITEMS['IN_'+NAME] = RESULT.TEST.values
        
    return ITEMS    
    
                                                              
def ItemsInPolygons(ITEM_SHAPEFILE,POLYGON_SHAPEFILE, BUFFER = None, EPSG4POLY = None):
    OUT_ITEMS = shp.Reader(ITEM_SHAPEFILE)
    OUT_ITEMS = read_shapefile(OUT_ITEMS)
    
    POLY = shp.Reader(POLYGON_SHAPEFILE)
    POLY = read_shapefile(POLYS)
    
    NAMES = POLY.applymap(lambda x:isinstance(x,str)).sum(axis=0).replace(0,np.nan).dropna()
    NAMES = POLY[list(NAMES.index)].nunique(axis=0).sort_values(ascending=False)
    MAXITEMS = NAMES.max()
    
    NAMES = NAMES.index[NAMES == MAXITEMS].to_list()
    NAMES_DF = POLY[NAMES].copy()
    
	ITEMS_GJ = SHP_to_GEOJSONLIST(ITEM_SHAPEFILE)
	ITEMS =  GEOJSONLIST_to_SHAPELY(ITEMS_GJ)
	ITEMS_C =  CRS_FROM_SHAPE(ITEM_SHAPEFILE)
    
	POLY_GJ = SHP_to_GEOJSONLIST(POLY_SHAPEFILE)
	POLY =  GEOJSONLIST_to_SHAPELY(POLY_GJ) 
	POLY_C =  CRS_FROM_SHAPE(POLY_SHAPEFILE)
           
    if EPSG4POLY != None:
        POLY_OLD = POLY
        project2 = pyproj.Transformer.from_crs(
                             pyproj.CRS.from_wkt(C.to_ogc_wkt()),
                             pyproj.CRS.from_epsg(EPSG4POLY),
                             always_xy=True).transform
        POLY = transform(project2, POLY_OLD)
        POLY_C = pyproj.CRS.from_epsg(EPSG4POLY)
    
    if BUFFER != None:
        POLY = POLY.buffer(BUFFER)       
    
	project = pyproj.Transformer.from_crs(
                         pyproj.CRS.from_wkt(POLY_C.to_ogc_wkt()),
                         pyproj.CRS.from_wkt(ITEMS_C.to_ogc_wkt()),
                         always_xy=True).transform
    
	POLY_USE = transform(project, POLY)
    
    for j in np.arange(0,len(POLY_USE.geoms)):
        p = POLY_USE.geoms[j]
        NAME = '__'.join(NAMES_DF.iloc[j,:].astype(str))
        NAME = NAME.replace(' ','_')
        OUT_ITEMS[NAME] = [l.intersects(p) for l in ITEMS.geoms]
    return OUT_ITEMS
