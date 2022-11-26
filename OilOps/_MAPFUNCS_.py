#MAPPING
from ._FUNCS_ import *
import shapefile as shp #pyshp
import shapely
import shapely.wkt
from shapely.ops import unary_union, cascaded_union, transform
import pycrs
import pyproj
import collections
from geopy.geocoders import Nominatim
import geopandas as gpd

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

def IN_TC_AREA(well2,tc2):
    ln = None
    if len(well2.coords)>=2:
        try:
            ln = shapely.geometry.LineString(well2.coords)
        except:
            ln = shapely.geometry.LineString(well2.coords.values)
    elif len(well2.coords)==1:
        ln = shapely.geometry.Point(well2.coords[0])
    if ln is None:
        return(False)
    test = False
    for j in range(tc2.shape[0]):
        if test == False:
            poly = shapely.geometry.Polygon(tc2.coords.iloc[j])
            if ln.intersects(poly.buffer(15000)):
                test = True
    return(test) 

def GROUP_IN_TC_AREA(tc,wells):
    out = _FUNCS_.pd.DataFrame()
    out['API'] = wells.API_Label.str.replace(r'[^0-9]','',regex=True)
    out['TEST'] = wells.apply(lambda x: IN_TC_AREA(x,tc),axis=1)
    return(out)

def convert_shapefile(SHP_File,EPSG_OLD=3857,EPSG_NEW=3857,FilterFile=None,Label=''):
    #if 1==1:
    # Define CRS from EPSG reference frame number
    EPSG_OLD= int(EPSG_OLD)
    EPSG_NEW=int(EPSG_NEW)

    crs_old = CRS.from_user_input(EPSG_OLD)
    crs_new = CRS.from_user_input(EPSG_NEW)
    TFORMER =  pyproj.Transformer.from_crs(crs_old, crs_new, always_xy= True)

    #read shapefile
    r = shp.Reader(SHP_File)   # THIS IS IN X Y COORDINATES!!!


    #define output filename
    out_fname = re.sub(r'(.*)(\.shp)',r'\1_EPSG'+str(EPSG_NEW)+Label+r'\2',SHP_File,flags=re.IGNORECASE)

    #if FilterFile != None:
    #    FILTERUWI=pd.read_csv(FilterFile,header=None,dtype=str).iloc[:,0].str.slice(start=1,stop=10)
    #    pdf = read_shapefile(r)
    #    SHP_APIS = pdf.API_Label.str.replace(r'[^0-9]','').str.slice(start=1,stop=10)
    #    SUBSET = SHP_APIS[SHP_APIS.isin(UWI)].index
    #else:
    #    SUBSET = np.arange(0,len(r.shapes()))

    # Speed, get subset of records
    if FilterFile is None:
        SUBSET=_FUNCS_.np.arange(0,len(r.shapes()))
    else:
        FILTERUWI=_FUNCS_.pd.read_csv(FilterFile,header=None,dtype=str).iloc[:,0].str.slice(start=1,stop=10)
        pdf=read_shapefile(r)
        pdf=pdf.API_Label.str.replace(r'[^0-9]','').str.slice(1,10)
        SUBSET=pdf[pdf.isin(FILTERUWI)].index.tolist()

    total = len(SUBSET)
    #compile converted output file
    with shp.Writer(out_fname, shapeType=r.shapeType) as w:
        w.fields = list(r.fields)
        outpoints = []
        for ct, i in enumerate(SUBSET, start=1):
            if (floor(ct/20)*20) == ct:
                print(f"{ct} of {total}")
            shaperec=r.shapeRecord(i)
            Xshaperec=shaperec.shape
            points = _FUNCS_.np.array(shaperec.shape.points).T

            #points_t= transform(crs_old, crs_new, points[0],points[1],always_xy=True
            # NEEDS TO BE LON LAT ORDER
            points_t = TFORMER.transform(points[0],points[1])

            #Xshaperec.points = list(map(tuple, points.T))
            json_shape = shaperec.shape.__geo_interface__
            json_shape['coordinates']=tuple(map(tuple, points.T))
            #outpoints = list(map(tuple, points))
            #Xshaperec.points=list(map(tuple, points))
##            if r.shapeType in [1,11,21]: ## "point" is used for point shapes
##                w.point(tuple(points))
##            if r.shapeType in [8,18,28]: ## "multipoint" is used for multipoint shapes
##                #outpoints=outpoints+(list(map(list, points.T)))
##                w.multipoint(list(map(list, points.T)))
##            if r.shapeType in [3,13,23]: ## "line" for lines
##                #outpoints.append(list(map(list, points.T)))
##                w.line([list(map(list, points.T))])
##            if r.shapeType in [5,15,25]: ## "poly" for polygons
##                #outpoints.append(list(map(list, points.T)))
##                w.poly(list(map(list, points.T)))
##            else: # "null" for null
##                w.null()
            w.record(*shaperec.record)
            w.shape(json_shape)
                    #w.shape(Xshaperec)
                    #del(Xshaperec)
    prjfile = re.sub(r'\.shp','.prj',out_fname,flags=re.IGNORECASE)
    tocrs = pycrs.parse.from_epsg_code(EPSG_NEW)
    with open(prjfile, "w") as writer:    
        _ = writer.write(tocrs.to_esri_wkt())
    #prjfile = open(re.sub(r'\.shp','.prj',out_fname,flags=re.IGNORECASE),'w')
    #prjfile.write(crs_old.to_wkt().replace(' ','').replace('\n',''))
    #prjfile.close()
       
def get_EPSG():
    msg = "Enter Shapefile EPSG codes"
    title = "Projection Definitions"
    fieldNames = ["Input projection EPSG code","Output projection code"]
    fieldValues = []  # we start with blanks for the values
    fieldValues = easygui.multenterbox(msg,title, fieldNames)
    while 1:
        CHECK,errmsg = check_2EPSG(fieldValues[0],fieldValues[1])
        if CHECK == True: break # no problems found
        fieldValues = easygui.multenterbox(errmsg, title, fieldNames, fieldValues)
    return(fieldValues)

def check_2EPSG(epsg1,epsg2):
    CRS1=CRS2=None
    OUTPUT = True
    MESSAGE = 'Valid EPSG codes'
    try:
        CRS1 = CRS.from_user_input(int(epsg1))
    except: pass
    try:
        CRS2 = CRS.from_user_input(int(epsg2))
    except: pass
    if CRS1 is None:
        MESSAGE = 'Invalid input EPSG code'
        OUTPUT = False
    if CRS2 is None:
        MESSAGE = 'Invalid output EPSG code'
        OUTPUT = False
    return(OUTPUT,MESSAGE)

def check_EPSG(epsg1):
    CRS1=None
    CHECK = 1
    try:
        CRS1 = CRS.from_user_input(int(epsg1))
    except: pass
    return 'Invalid' if CRS1 is None else 'Validated EPSG code'

def XYtransform(df_in, epsg1 = 4269, epsg2 = 2878):
    #2876
    df_in=df_in.copy()
    transformer = pyproj.Transformer.from_crs(epsg1, epsg2,always_xy =True)
    df_in[['X','Y']]=df_in.apply(lambda x: transformer.transform(x.iloc[2],x.iloc[1]), axis=1).apply(_FUNCS_.pd.Series)
    #df_in[['X','Y']]=df_in.apply(lambda x: transform(epsg1,epsg2,x.iloc[2],x.iloc[1],always_xy=True), axis=1).apply(_FUNCS_.pd.Series)
    return df_in

def FindCloseList(UWI10LIST, shpfile='/home/ruckerwk/Programming/Directional_Lines.shp'):
    gp = gpd.read_file(shpfile)
    try:
        gp['UWI10'] = gp.API_Label.apply(lambda x: OilOps.WELLAPI(x).API2INT(10))
    except:
        gp = gp.loc[~gp.API.isna()]
        gp['UWI10'] = gp.API.str.replace('-','').str[:10]
        gp['UWI10'] = "5" + gp['UWI10']
        gp['UWI10'] = gp['UWI10'].astype(int)

    ULIST = UWI10LIST
    USELIST = ULIST.copy()

    SUB = gp.loc[gp.UWI10.isin(ULIST)]
    for g in SUB.index:
        G = SUB.loc[g,'geometry'].buffer(5000)
        #gpd.tools.sjoin(gp,G,how = 'left')

        # ROUGH FILTER
        m = gp.geometry.centroid.distance(G.centroid)<10000
        m[g] = False
        m = gp.loc[m].index
        m2 = gp.loc[m,'geometry'].distance(G)<10000
        m2 = m2.loc[m2].index
        USELIST = USELIST + gp.loc[(gp.index.isin(m2)) & (~gp.UWI10.isin(USELIST)),'UWI10'].tolist()
    return USELIST
