import sqlite3, re, datetime, sqlalchemy

#import COGCCpy
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn as sk
# pip install --no-use-pep517 scikit-learn

from functools import partial
import urllib,datetime,re,io,csv,sys,requests,selenium,multiprocessing,warnings
from os import path, listdir, remove, makedirs
import pandas as pd
from urllib.request import urlopen 
import numpy as np
from bs4 import BeautifulSoup as BS

import concurrent.futures

def Get_ProdData(UWIs,file='prod_data.db'):
    #if 1==1:
    #URL_BASE = 'https://cogcc.state.co.us/cogis/ProductionWellMonthly.asp?APICounty=XCOUNTYX&APISeq=XNUMBERX&APIWB=XCOMPLETIONX&Year=All'
    URL_BASE = 'https://cogcc.state.co.us/production/?&apiCounty=XCOUNTYX&apiSequence=XNUMBERX'
    pathname = path.dirname(sys.argv[0])
    adir = path.abspath(pathname)
    warnings.simplefilter("ignore")
    OUTPUT=pd.DataFrame(columns=['BTU_MEAN','BTU_STD',
                                 'API_MEAN','API_STD',
                                 'Peak_Oil_Date','Peak_Oil_Days','Peak_Oil_CumOil','Peak_Oil_CumGas','Peak_Oil_CumWtr',
                                 'Peak_Gas_Date','Peak_Gas_Days','Peak_Gas_CumOil','Peak_Gas_CumGas','Peak_Gas_CumWtr',
                                 'Peak_GOR_Date','Peak_GOR_Days','Peak_GOR_CumOil','Peak_GOR_CumGas','Peak_GOR_CumWtr',
                                 'OWR_PrePeakOil','OWR_PostPeakGas',
                                 'GOR_PrePeakOil','GOR_PeakGas','GOR_PostPeakGOR',
                                 'WOC_PostPeakOil','WOC_PostPeakGas'])
                        
    if len(UWIs[0])<=1:
        UWIs=[UWIs]
        print(UWIs[0])
    ct=0
    for UWI in UWIs:
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
                try:
                    pdf = pd.read_html(docurl,encoding='utf-8', header=0)[1]
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
                    

                    OUTPUT.at[UWI,'Peak_GOR_Date']   = pdf[DATE][pdf['GOR'].idxmax()]
                    OUTPUT.at[UWI,'Peak_GOR_Days']   = pdf['PROD_DAYS'][pdf['GOR'].idxmax()]
                    OUTPUT.at[UWI,'Peak_GOR_CumOil'] = pdf[OIL][0:pdf['GOR'].idxmax()].sum()
                    OUTPUT.at[UWI,'Peak_GOR_CumGas'] = pdf[GAS][0:pdf['GOR'].idxmax()].sum()

                    PREPEAKOIL  = pdf.loc[(pdf['PROD_DAYS']-pdf['PROD_DAYS'][pdf[OIL].idxmax()]).between(-100,0),:].index
                    POSTPEAKOIL = pdf.loc[(pdf['PROD_DAYS'][pdf[OIL].idxmax()]-pdf['PROD_DAYS']).between(0,100),:].index
                    POSTPEAKGAS = pdf.loc[(pdf['PROD_DAYS'][pdf[GAS].idxmax()]-pdf['PROD_DAYS']).between(0,100),:].index
                    PEAKGAS = pdf.loc[(pdf['PROD_DAYS'][pdf[GAS].idxmax()]-pdf['PROD_DAYS']).between(-50,50),:].index
                    POSTPEAKGOR = pdf.loc[(pdf['PROD_DAYS'][pdf['GOR'].idxmax()]-pdf['PROD_DAYS']).between(0,100),:].index
                    
                    OUTPUT.at[UWI,'GOR_PrePeakOil']  = pdf.loc[PREPEAKOIL,GAS].sum() * 1000 / pdf.loc[PREPEAKOIL,OIL].sum()
                    OUTPUT.at[UWI,'GOR_PeakGas']     = pdf.loc[PEAKGAS,GAS].sum() * 1000 / pdf.loc[PEAKGAS,OIL].sum()
                    OUTPUT.at[UWI,'GOR_PostPeakGOR'] = pdf.loc[POSTPEAKGOR,GAS].sum() * 1000 / pdf.loc[POSTPEAKGOR,OIL].sum()

                    if pdf[[WTR,OIL,GAS]].dropna(how='any').shape[0]>3:
                        OUTPUT.at[UWI,'OWR_PrePeakOil']  = pdf.loc[PREPEAKOIL,OIL].sum()/pdf.loc[PREPEAKOIL,WTR].sum()
                        OUTPUT.at[UWI,'OWR_PostPeakGas'] = pdf.loc[POSTPEAKGAS,OIL].sum()/pdf.loc[POSTPEAKGAS,WTR].sum()                    
                        OUTPUT.at[UWI,'WOC_PrePeakOil']  = pdf.loc[POSTPEAKOIL,WTR].sum() / (pdf.loc[POSTPEAKOIL,WTR].sum()+pdf.loc[POSTPEAKOIL,OIL].sum())
                        OUTPUT.at[UWI,'WOC_PostPeakGas'] = pdf.loc[POSTPEAKGAS,WTR].sum() / (pdf.loc[POSTPEAKGAS,WTR].sum()+pdf.loc[POSTPEAKGAS,OIL].sum())        
                        OUTPUT.at[UWI,'Peak_Oil_CumWtr'] = pdf[WTR][0:pdf[OIL].idxmax()].sum()
                        OUTPUT.at[UWI,'Peak_Gas_CumWtr'] = pdf[WTR][0:pdf[GAS].idxmax()].sum()
                        OUTPUT.at[UWI,'Peak_GOR_CumWtr'] = pdf[WTR][0:pdf['GOR'].idxmax()].sum()
                        
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
                 ,[Peak_GOR_Date] DATE
                 ,[Peak_GOR_Days] INTEGER
                 ,[Peak_GOR_CumOil] REAL
                 ,[Peak_GOR_CumGas] REAL
                 ,[GOR_PrePeakOil] REAL
                 ,[GOR_PeakGas] REAL
                 ,[GOR_PostPeakGOR] REAL
                 ,[OWR_PrePeakOil] REAL
                 ,[OWR_PostPeakGas] REAL
                 ,[WOC_PrePeakOil] REAL
                 ,[WOC_PostPeakOil] REAL
                 ,[WOC_PostPeakGas] REAL
                 ,[Peak_Oil_CumWtr] REAL
                 ,[Peak_Gas_CumWtr] REAL
                 ,[Peak_GOR_CumWtr] REAL
                 )
                 '''

            TABLE_NAME = "PROD_SUMMARY"
            
    if OUTPUT.shape[0] > 0:
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

def SUMMARIZE_COGCC(SAVE = True, SAVEDB= 'FIELD_DATA.db',TABLE_NAME = 'CO_SQL_SUMMARY'):
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
    if 1==1:
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
    if 1==1:
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

    if 1==1:
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

    if 1==1:
        STIM_df['StimDate0'] = STIM_df['StateStimulationKey'+str(i-1)].map(dict2).map(StimDates).fillna(datetime.date(1900, 1, 1))
        STIM_df['StimDate1'] = STIM_df['StateStimulationKey'+str(i-1)].map(dict1).map(StimDates).fillna(datetime.datetime.now())
        STIM_df.loc[STIM_df.LastStimDate == STIM_df.StimDate0,'StimDate0']=datetime.date(1900, 1, 1)
        STIM_df.loc[STIM_df.StimDate1 == STIM_df.StimDate1,'StimDate1']=datetime.datetime.now()


    # 3,6,9,12,15,18,21,24 month cum
    if 1==1:
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

    if 1==1:
        # MERGE WITH WELL HEADERS
        with sqlite3.connect('CO_3_2.1.sqlite') as conn:
            Well_df=pd.read_sql_query("""SELECT * FROM WELL""",conn)
        ULT = pd.merge(cumdf,Well_df.set_index('StateWellKey'), left_index=True, right_index=True,how='outer')
        for k in ULT.keys():
            if 'DATE' in k.upper():
                ULT[k] = pd.to_datetime(ULT['FirstCompDate'])

        #ULT.to_csv('SQL_WELL_SUMMARY.csv')
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
        # ULT replace Nan with None
    ##    ULT2 = ULT.where(pd.notnull(ULT), None)
    ##    with sqlite3.connect('prod_data.db') as conn:
    ##        TABLE_NAME = 'Well_Summary'
    ##        ULT.to_sql(TABLE_NAME,conn,
    ##                      if_exists='replace',
    ##                      index=False,
    ##                      dtype=df_typemap)
        if SAVE:
            with sqlite3.connect(SAVEDB) as CONN:
                ULT.to_sql(TABLE_NAME,CONN,
                    if_exists='replace',
                    index=False,
                    dtype=df_typemap)
        
#!! not all rows have a state producing key




# ULT.to_csv('WellSummary.csv',sep=',')
#ULT=pd.read_csv('WellSummary.csv'); ULT.loc[:,ULT.keys()[0]]=ULT.loc[:,ULT.keys()[0]].astype('str').str.zfill(14); ULT=ULT.set_index(ULT.keys()[0]);
#UWIlist = ULT.index.unique().to_list()

##class Result():
##    def __init__(self):
##        self.val = None
##
##    def update_result(self, val):
##        self.val = val
##result = Result()
##
##def f(x):
##    return x*0.5+x
##
##if __name__ == '__main__':
##    with multiprocessing.Pool(processes = 2) as pool:
##            results = [pool.apply_async(f,args=(x,)) for x in range(1,10)]
##            
##

######        
########### PULL STATE PRODUCTION DATA
######if __name__ == "__main__":
######    processors = max(1,multiprocessing.cpu_count()-1)
######    chunksize = min(5000,int(len(UWIlist)/processors))
######    #UWIlist=UWIlist[1000:(2000+processors*chunksize)]
######    #xUWIlist=UWIlist
######    print ("starting map function")
######    # outfile = "BTU_API_PULL_"+datetime.datetime.now().strftime("%d%m%Y")+".csv"
######    batch = int(len(UWIlist)/chunksize)
######    processors = max(processors,batch)
######    data = np.array_split(UWIlist,batch)
######    with concurrent.futures.ThreadPoolExecutor(max_workers = processors) as executor:
######        f = {executor.submit(Get_ProdData, uwis): uwis for uwis in data}
######



##import asyncio                        # Gives us async/await
##import aiohttp                        # For asynchronously making HTTP requests
##import aiofiles                       # For asynchronously performing file I/O operations
##import concurrent.futures             # Allows creating new processes

    
#(df.groupby('StateProducingUnitKey')['OilRate_BOPD'].max()-df.groupby('StateProducingUnitKey')['OilRate_BOPD'].quantile(0.99)).sort_values(ascending=False)
##
##if 1==1:
##    from sklearn import svm
##    from sklearn.datasets import make_moons, make_blobs
##    from sklearn.covariance import EllipticEnvelope
##    from sklearn.ensemble import IsolationForest
##    from sklearn.neighbors import LocalOutlierFactor
##
##matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
##
### Example settings
##if 1==1:
##    n_samples = 300
##    outliers_fraction = 0.05
##    n_outliers = int(outliers_fraction * n_samples)
##    n_inliers = n_samples - n_outliers
##
### define outlier/anomaly detection methods to be compared
##anomaly_algorithms = [
##    ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
##    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
##                                      gamma=0.1)),
##    ("Isolation Forest", IsolationForest(behaviour='new',
##                                         contamination=outliers_fraction,
##                                         random_state=42)),
##    ("Local Outlier Factor", LocalOutlierFactor(
##        n_neighbors=35, contamination=outliers_fraction))]
##
### Define datasets
##blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
##datasets = [
##    make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5,
##               **blobs_params)[0],
##    make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5],
##               **blobs_params)[0],
##    make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3],
##               **blobs_params)[0],
##    4. * (make_moons(n_samples=n_samples, noise=.05, random_state=0)[0] -
##          np.array([0.5, 0.25])),
##    14. * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)]
##
##
##df_sub = df.loc[(df['MonthsOn']<=16) & (df['OilRate_BOPD']>0) & (df['NormOilRate']<1)]
##
##df_sub['NormOilRate'<1].groupby('StateProducingUnitKey')['NormOilRate'].max()
##.max()-df.groupby('StateProducingUnitKey')['OilRate_BOPD'].quantile(0.99)).sort_values(ascending=False)
##
##cutoff = 0.3; plt.hist(df_sub.groupby('StateProducingUnitKey')['NormOilRate'].max()[df_sub.groupby('StateProducingUnitKey')['NormOilRate'].max()<cutoff].sort_values(ascending=False), 50, facecolor='blue', alpha=0.5);plt.show()
##df_sub.groupby('StateProducingUnitKey')['NormOilRate'].max()[df_sub.groupby('StateProducingUnitKey')['NormOilRate'].max()<cutoff].sort_values(ascending=False).index
##
### 05123155800000 weird late peak
##df_sub = df.loc[(df['MonthsOn']<=16) & (df['OilRate_BOPD']>0) & (df['NormOilRate']<1)].sort_values(by='CumOil', ascending=True)
##api = '05125120130000';plt.loglog(df[(df['StateProducingUnitKey']==api)]['CumOil'],df[df['StateProducingUnitKey']==api]['OilRate_BOPD'])
##print(df_sub.loc[df_sub['StateProducingUnitKey']==api][['ProductionDate','MonthsOn','StateProducingZoneKey','Oil','OilRate_BOPD','CumOil','NormOilRate']].sort_values(by='ProductionDate',ascending=True).to_string())
##
##df_sub.loc[df_sub['StateProducingUnitKey']=='05123155800000'][['OilRate_BOPD','CumOil']]
##
##df.to_csv('CoProd.csv')
##
##
### get list of wells with 2nd NormOil value below 0.6
##df_sub = df.loc[(df['MonthsOn']<=16) & (df['OilRate_BOPD']>0) & (df['NormOilRate']<1)]
###df_sub.loc[df_sub.StateProducingUnitKey.isin( df_sub.loc[df_sub.NormOilRate>=0.6].StateProducingUnitKey.unique())==False]
##api_list = df_sub.loc[df_sub.StateProducingUnitKey.isin( df_sub.loc[df_sub.NormOilRate>=0.6].StateProducingUnitKey.unique())==False].StateProducingUnitKey.unique()
##print(df.loc[df['StateProducingUnitKey']==api][['ProductionDate','MonthsOn','StateProducingZoneKey','Oil','OilRate_BOPD','CumOil','NormOilRate']].sort_values(by='ProductionDate',ascending=True).to_string())
##api=api_list[0];plt.plot(df[(df['StateProducingUnitKey']==api)]['CumOil'],df[df['StateProducingUnitKey']==api]['OilRate_BOPD'])
##
##api=api_list[2];plt.plot(df[(df['StateProducingUnitKey']==api)]['NormOilRate'],df[df['StateProducingUnitKey']==api]['NormGasRate'],'b--');plt.show()
##
##import subprocess, pip, sys 
##with open ('C:\\Users\\KRucker\\Documents\\pip38packages_nov.txt')as f:
##    lines = f.read().splitlines()
##for package in lines:
##    if 'NUMPY' in package.upper():
##        continue
##    print(package)
##    subprocess.call([sys.executable, '-m', 'pip', 'install', package])
##
