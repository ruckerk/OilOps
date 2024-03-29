# STAIR PLOT
import matplotlib.ticker as tkr
from ._FUNCS_ import *

__all__ = ['STAIR_PLOT']


def STAIR_PLOT(ULIST,df, ProdKey= None,ReverseY=True):
    # add element to include actual survey points
    
    # subset or wells in plotting order
    #XY_KEY = GetKey(df,'Mean.*TVD')[0]
    #m = df.loc[df.UWI10.isin(ULIST)][XY_KEY].dropna().sort_values(ascending=True).index

    #Select hz dimension
    XY_KEY = df.loc[df.UWI10.isin(ULIST),GetKey(df,'Mean.*(X|Y)')].diff().abs().mean(axis=0).sort_values(ascending=False).keys()[0]

    # subset or wells in plotting order
    m = df.loc[df.UWI10.isin(ULIST),XY_KEY].dropna().sort_values(ascending=True).index
    
    # normalize X axis
    minX = df.loc[m,XY_KEY].min()

    # normalize color scale
    if ProdKey:
        #CVALS = ((df.loc[m,ProdKey] - df.loc[m,ProdKey].min()) / (df.loc[m,ProdKey].max()-df.loc[m,ProdKey].min())).values
        CVALS = df.loc[m,ProdKey].values
        CVAL_STEP = 10**np.floor(np.log10(df.loc[m,ProdKey].max() - df.loc[m,ProdKey].min()))/2
        CVAL_RANGE = range(int(np.ceil(np.nanmin(CVALS)/CVAL_STEP)*CVAL_STEP),
                       int(np.ceil(np.nanmax(CVALS)/CVAL_STEP)*CVAL_STEP),
                       int(CVAL_STEP))
    else:
        CVALS = 0
    
    
    #plot gunbarrel and stairs
    fig, ax = plt.subplots(1, sharex = True, squeeze = True)

    fig.set_figheight(7)
    fig.set_figwidth(21)

    #adjust Y axis for label room
    if ReverseY:
        plt.ylim([df.loc[m,'MeanTVD'].max()+100, df.loc[m,'MeanTVD'].min()-100])
    else:
        plt.ylim([df.loc[m,'MeanTVD'].min()-100, df.loc[m,'MeanTVD'].max()+100])
    
    ax.step(df.loc[m,XY_KEY] - minX,
            df.loc[m,'MeanTVD'],
            where = 'post',
            linewidth=1,
            linestyle ='--',
            alpha = 0.8)

    # add color by EUR
    sc = ax.scatter(df.loc[m,XY_KEY] - minX,
            df.loc[m,'MeanTVD'],
            c = CVALS,
            marker = 'o',
            s = 150,
            alpha=0.95,
            cmap = 'viridis')

    # set ticks
    ax.xaxis.set_major_locator(tkr.MultipleLocator(500))
    ax.yaxis.set_major_locator(tkr.MultipleLocator(50))
    ax.xaxis.set_minor_locator(tkr.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(tkr.AutoMinorLocator(5))

    # plot grid
    ax.grid(visible = True, which = 'both', axis = 'both')

    # fill grid cell and label scale

    # plot horizontal distances
    X = ((df.loc[m,XY_KEY] - minX).iloc[:-1] + df.loc[m,XY_KEY].diff().dropna().values/2).values
    Y = (df.loc[m,'MeanTVD'].iloc[:-1]).values
    LAB = df.loc[m,XY_KEY].diff().dropna().abs().apply(np.floor).values.astype(int)

    for i in range(0,len(X)):
        plt.text(X[i],Y[i],LAB[i],
                 color = 'royalblue',
                 fontweight = 550,
                 fontsize = 17,
                 horizontalalignment = 'center',
                 verticalalignment = 'center');

    # plot vertical distances
    X = (df.loc[m,XY_KEY] - minX).iloc[1:].values
    Y = (df.loc[m,'MeanTVD'].iloc[1:]-df.loc[m,'MeanTVD'].diff().dropna()/2).values
    LAB = df.loc[m,'MeanTVD'].diff().dropna().abs().apply(np.floor).values.astype(int)

    for i in range(0,len(X)):
        plt.text(X[i],Y[i],LAB[i],
                 color = 'royalblue',
                 fontweight = 550,
                 fontsize = 17,
                 horizontalalignment = 'center',
                 verticalalignment = 'center') ;


    # Plot well data labels (add regex to determine EUR or X mo Oil/Gas/Water
    LABELFORM = "API:_API_ DATE:_DATE_\nNAME:_NAME_\nOPER:_OPER_\nFI:_FI_   PI:_PI_\nLL:_LATLEN_   PROD:_PROD_"

    WELLDATE = GetKey(df,r'FIRST.*PROD.*DATE|JOB.*DATE|SPUD.*DATE')
    for k in WELLDATE:
        df[k] = pd.to_datetime(df[k],errors='coerce')
    WELLDATE = df[WELLDATE].loc[:,df[WELLDATE].max().dt.year>2000].keys().tolist()
    #WELLDATE = df.loc[m,WELLDATE].keys()[(df.loc[m,WELLDATE].isna().sum(axis=0) == df.loc[m,WELLDATE].isna().sum(axis=0).min())].tolist()
    #WELLDATE = df.loc[m,WELLDATE].dropna().apply(lambda x:x[WELLDATE] == max(x[WELLDATE]), axis =1).max(axis=0).sort_values(ascending = False).keys()[0]
    WELL_LABEL = GetKey(df,r'WELL.*(NAME|LABEL|NO)')
    test = df.loc[m,WELL_LABEL].map(lambda x: len(str(x)))
    for k in test.keys():
        test[k] = (test[k] == test.max(axis=1))
    WELL_LABEL = test.sum(axis=0).sort_values(ascending =False).keys()[0]
    #WELL_LABEL = test.apply(lambda x: x==max(x), axis = 1).sum(axis=0).sort_values(ascending=False).keys()[0]
    OPERATOR =  GetKey(df,r'OPERATOR')[0]
    FLUID_INTENSITY = GetKey(df,r'(WATER|FLUID|INJ).*INTEN')[0]
    PROP_INTENSITY = GetKey(df,r'(PROP|SAND).*INTEN')[0]
    LATERAL_LENGTH = GetKey(df,r'LAT.*LEN')
    LATERAL_LENGTH = (df.loc[m,LATERAL_LENGTH].fillna(0) > 0).sum(axis=0).sort_values(ascending=False).keys()[0]
    
    for i in m:
        if i == m[0]:
            LABELS = list()
        
        LABEL_TEXT = LABELFORM.replace('_API_',str(df.loc[i,'UWI10']))
        try:
            LABEL_TEXT = LABEL_TEXT.replace('_DATE_',df.loc[i,WELLDATE].strftime('%Y-%m-%d'))
        except:
            LABEL_TEXT = LABEL_TEXT.replace('_DATE_','NA')
        LABEL_TEXT = LABEL_TEXT.replace('_NAME_',str(df.loc[i,WELL_LABEL]).strip())
        LABEL_TEXT = LABEL_TEXT.replace('_OPER_',str(df.loc[i,OPERATOR]).strip())
        LABEL_TEXT = LABEL_TEXT.replace('_FI_',str(np.floor(df.loc[i,FLUID_INTENSITY])).split('.')[0].strip() +' BBL/FT')
        LABEL_TEXT = LABEL_TEXT.replace('_PI_',str(np.floor(df.loc[i,PROP_INTENSITY])).split('.')[0].strip() +' #/FT')
        LABEL_TEXT = LABEL_TEXT.replace('_LATLEN_',str(np.floor(df.loc[i,LATERAL_LENGTH])).split('.')[0].strip() +' FT')
        if ProdKey:
            LABEL_TEXT = LABEL_TEXT.replace('_PROD_',str(np.floor(df.loc[i,ProdKey])).split('.')[0].strip() +'MBBL')
        else:     
            LABEL_TEXT = LABEL_TEXT.replace('_PROD_','None')
            
        #LABEL_TEXT = f"API:{str(df.loc[i,'UWI10'])} DATE:_DATE_\nNAME:_NAME_\nOPER:_OPER_\nFI:_FI_   PI:_PI_\nLL:_LATLEN_   PROD:_PROD_"
        LABELS.append(LABEL_TEXT)
        
    # manage overposting
    texts = [ax.text((df.loc[m[j],XY_KEY] - minX),
                     df.loc[m[j],'MeanTVD'],
                     LABELS[j].strip(),
             fontsize = 9,
             bbox = dict(facecolor='w', edgecolor = 'none', alpha = 0.6)) for j in range(len(LABELS))]
    
    adjust_text(texts,
                ha = 'left',
                va = 'top',
                lim = 1000,
                force_points = (1,1),
                ax=ax)

    # add titles and axis labels
    plt.xlabel("Horizontal Distance [feet]")
    plt.ylabel("Vertical Distance [feet]")
    if ProdKey:
        cbar = plt.colorbar(sc, ticks = CVAL_RANGE, label = ProdKey)
    #cbar.set_label(ProdKey, rotation=270)
    
    pylab.tight_layout()
    plt.savefig('STAIRPLOT_'+str(ULIST[0])+'_'+str(ULIST[-1])+'.PNG')
    plt.close()
