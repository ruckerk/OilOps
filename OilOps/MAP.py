from ._FUNCS_ import *

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
        pathname = path.dirname(sys.argv[0])
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
        _CRS = pycrs.load.from_file(FPRJ)
    except:
        try:
            FPRJ = SHAPEFILE.split('.')[0]+'.PRJ'
            _CRS = pycrs.load.from_file(FPRJ)
        except:    
            warnings.showwarning('No CRS read from '+SHAPEFILE)
    return(_CRS)

def SHP_DISTANCES(SHP1,SHP2,MAXDIST=10000,CALCEPSG = 26753):
    # delivers nearest distance for each item in SHP1 to any item in SHP2
    # optional MAXDIST for maximum distance of interest

    _CRS1 = CRS_FROM_SHAPE(SHP1)
    _CRS2 = CRS_FROM_SHAPE(SHP2)

    _S1 = SHP_to_GEOJSONLIST(SHP1)
    _S2 = SHP_to_GEOJSONLIST(SHP2)

    _GS1 = GEOJSONLIST_to_SHAPELY(_S1)
    _GS2 = GEOJSONLIST_to_SHAPELY(_S2)

    _MAINSHAPE = shp.Reader(SHP1)
    _MAIN_DF = read_shapefile(_MAINSHAPE)
        
    # S2 polygons = [shapely.wkt.loads(p.wkt) for p in PGONS]

    # _GS1   FAULTS_MULTIP = MultiPolygon(polygons)
    # FAULTS_8000 = FAULTS_MULTIP.buffer(8000)

    # make same coordinate system in units/projection of interest
    project1 = pyproj.Transformer.from_crs(
        pyproj.CRS.from_wkt(_CRS1.to_ogc_wkt()),
        pyproj.CRS.from_epsg(CALCEPSG),
        always_xy=True).transform

    project2 = pyproj.Transformer.from_crs(
        pyproj.CRS.from_wkt(_CRS2.to_ogc_wkt()),
        pyproj.CRS.from_epsg(CALCEPSG),
        always_xy=False).transform		

    _GS1_RPRJ = shapely.ops.transform(project1, _GS1)
    _GS2_RPRJ = shapely.ops.transform(project2, _GS2)
	
    WORKLIST = [geom if geom.is_valid else geom.buffer(0) for geom in _GS2_RPRJ]

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