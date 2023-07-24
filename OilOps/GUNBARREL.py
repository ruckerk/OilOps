# STAIR PLOT
import matplotlib.ticker as tkr
from ._FUNCS_ import *

__all__ = ['STAIR_PLOT']


def STAIR_PLOT(ULIST,df, ProdKey= None):

    # subset or wells in plotting order
    m = df.loc[df.UWI10.isin(ULIST)].MeanX.dropna().sort_values(ascending=True).index
    #m = df.loc[m,'StateProducingUnitKey'].dropna().index

    # normalize X axis
    minX = df.loc[m,'MeanX'].min()

    # normalize color scale
    if ProdKey:
        CVALS = ((df.loc[m,ProdKey] - df.loc[m,ProdKey].min()) / (df.loc[m,ProdKey].max()-df.loc[m,ProdKey].min())).values
        CVALS = df.loc[m,ProdKey].values
        CVAL_STEP = 10**np.floor(np.log10(df.loc[m,ProdKey].max() - df.loc[m,ProdKey].min()))/2
        CVAL_RANGE = range(int(np.ceil(CVALS.min()/CVAL_STEP)*CVAL_STEP),
                       int(np.ceil(CVALS.max()/CVAL_STEP)*CVAL_STEP),
                       int(CVAL_STEP)) 
    else:
        CVALS = 0
    
    
    #plot gunbarrel and stairs
    fig, ax = plt.subplots(1, sharex = True, squeeze = True)

    fig.set_figheight(5)
    fig.set_figwidth(12)

    #adjust Y axis for label room
    plt.ylim([df.loc[m,'MeanTVD'].min()-100, df.loc[m,'MeanTVD'].max()+100])

    ax.step(df.loc[m,'MeanX'] - minX,
            df.loc[m,'MeanTVD'],
            where = 'post',
            linewidth=1,
            linestyle ='--',
            alpha = 0.8)

    # add color by EUR
    sc = ax.scatter(df.loc[m,'MeanX'] - minX,
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
    X = ((df.loc[m,'MeanX'] - minX).iloc[:-1] + df.loc[m,'MeanX'].diff().dropna().values/2).values
    Y = (df.loc[m,'MeanTVD'].iloc[:-1]).values
    LAB = df.loc[m,'MeanX'].diff().dropna().abs().apply(np.floor).values.astype(int)

    for i in range(0,len(X)):
        plt.text(X[i],Y[i],LAB[i],
                 color = 'royalblue',
                 fontweight = 550,
                 fontsize = 17,
                 horizontalalignment = 'center',
                 verticalalignment = 'center');

    # plot vertical distances
    X = (df.loc[m,'MeanX'] - minX).iloc[1:].values
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
    WELLDATE = df.loc[m,WELLDATE].keys()[(df.loc[m,WELLDATE].isna().count(axis=0) == df.loc[m,WELLDATE].isna().count(axis=0).min())].tolist()
    WELLDATE = df.loc[m,WELLDATE].dropna().apply(lambda x:x[WELLDATE] == max(x[WELLDATE]), axis =1).max(axis=0).sort_values(ascending = False).keys()[0]
    WELL_LABEL = GetKey(df,r'WELL.*(NAME|LABEL|NO)')
    test = df.loc[m,WELL_LABEL].applymap(lambda x: len(str(x)))
    WELL_LABEL = test.apply(lambda x: x==max(x), axis = 1).sum(axis=0).sort_values(ascending=False).keys()[0]
    OPERATOR =  GetKey(df,r'OPERATOR')[0]
    FLUID_INTENSITY = GetKey(df,r'(WATER|FLUID|INJ).*INTEN')[0]
    PROP_INTENSITY = GetKey(df,r'(PROP|SAND).*INTEN')[0]
    LATERAL_LENGTH = GetKey(df,r'LAT.*LEN')
    LATERAL_LENGTH = (df.loc[m,LATERAL_LENGTH].fillna(0) > 0).sum(axis=0).sort_values(ascending=False).keys()[0]
    
    for i in m:
        if i == m[0]:
            LABELS = list()
        
        LABEL_TEXT = LABELFORM.replace('_API_',str(df.loc[i,'UWI10']))
        LABEL_TEXT = LABEL_TEXT.replace('_DATE_',df.loc[i,WELLDATE].dt.date.astype(str).str.strip())
        LABEL_TEXT = LABEL_TEXT.replace('_NAME_',df.loc[i,WELL_LABEL].strip())
        LABEL_TEXT = LABEL_TEXT.replace('_OPER_',df.loc[i,OPERATOR].strip())
        LABEL_TEXT.replace('_FI_',df.loc[i,FLUID_INTENSITY].astype(int).astype(str).strip()+' BBL/FT')
        LABEL_TEXT = LABEL_TEXT.replace('_PI_',df.loc[i,PROP_INTENSITY].astype(int).astype(str).strip()+' #/FT')
        LABEL_TEXT = LABEL_TEXT.replace('_LATLEN_',df.loc[i,LATERAL_LENGTH].astype(int).astype(str).strip()+' FT')
        if ProdKey:
            LABEL_TEXT = LABEL_TEXT.replace('_PROD_',df.loc[i,ProdKey].astype(int).astype(str).strip()+'MBBL')
        else:     
            LABEL_TEXT = LABEL_TEXT.replace('_PROD_','None')
            
        #LABEL_TEXT = f"API:{str(df.loc[i,'UWI10'])} DATE:_DATE_\nNAME:_NAME_\nOPER:_OPER_\nFI:_FI_   PI:_PI_\nLL:_LATLEN_   PROD:_PROD_"
        LABELS.append(LABEL_TEXT)
        
    # manage overposting
    texts = [ax.text((df.loc[m[j],'MeanX'] - minX),
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
        cbar = plt.colorbar(sc, ticks = CVAL_RANGE, label = 'EUR [MBBL]')
    #cbar.set_label('EUR [MBBL]', rotation=270)
    
    pylab.tight_layout()
    plt.save('STAIRPLOT_'+str(ULIST[0])+'_'+str(ULIST[-1])+'.PNG')
    plt.close()
