from ._FUNCS_ import *
__all__ = ['dpl',
           'richards',
           'fit_dpl_with_cum',
           'dpl_residual',
           'ProductionToParams',
           'fit_sigmoid_dual',
           'interpolate_to_daily_by_prod_days']

def richards(x, A,K,B,M,nu):
    return A + (K-A) / (1 + np.exp(-B*(x-M)))**(1/nu)

def dpl(t, q0, alpha, b1, b2, tx, m):
    return (q0 / (1 + alpha*t)**b1) * (1 + (t/tx)**m)**(-b2/m)

# -----------------------------------------------------------------
#  residual that blends rate + cumulative errors
# -----------------------------------------------------------------
def dpl_residual(theta, t, q_obs,
                 beta_cum=1.2,
                 slope_gamma=10.0, N_tail=30):

    q_hat   = dpl(t, *theta)
    dt      = np.diff(np.r_[0,t])
    cum_obs = np.cumsum(q_obs*dt); cum_hat=np.cumsum(q_hat*dt)

    log_err = np.log(q_hat+1e-9)-np.log(q_obs+1e-9)
    cum_err = beta_cum*(cum_hat-cum_obs)/cum_obs.max()

    # ---- slope penalty last N_tail points ----
    mbt_hat = cum_hat/(q_hat+1e-9)
    log_q_tail   = np.log(q_hat[-N_tail:])
    log_m_tail   = np.log(mbt_hat[-N_tail:])
    slope_tail   = np.diff(log_q_tail)/np.diff(log_m_tail)
    slope_err    = slope_gamma*(slope_tail + 1.0)

    return np.r_[log_err, cum_err, slope_err]

def fit_dpl_with_cum(t, q, beta_cum=1.2, p0=None, bounds=None, plot=True, t_EUR = None):
    q0 = q.copy()
    t = np.asarray(t, float)
    q = np.asarray(q, float)

    # initial guess if not given
    if p0 is None:
        p0 = [q[0], 0.002, 0.2, 1.0, 300, 4.0]
    if bounds is None:
        bounds = ([0., 0., 0., 0.9, 10., 1.],
                  [np.inf, 10., 2., 1.1, 5000., 10.])

    res = least_squares(
            dpl_residual, p0,
            args=(t, q, beta_cum, True),
            bounds=bounds,
            loss='soft_l1',       # robust to outliers
            f_scale=0.3,          # “softness”; tune 0.1–1
            max_nfev=40000)

    pars = res.x

    if t_EUR != None:
        t_hat = np.arange(1, t_EUR+1, 1)
    else:
        t_hat = t

    q_hat = dpl(t_hat, *pars)

    # ------- diagnostics plot -------------------------------------
    if plot:
        dt = np.diff(np.r_[0, t])
        dt_hat = np.diff(np.r_[0, t_hat])

        cum_obs = np.cumsum(q*dt)
        cum_hat = np.cumsum(q_hat*dt_hat)

        mbt_hat = cum_hat / q_hat

        fig,ax = plt.subplots(2,2,figsize=(13,10))
        axs = ax.flatten()
        # MBT plot
        mbt_obs = cum_obs / (q+1e-9)
        axs[0].loglog(mbt_obs, q,'.',alpha=.3,color='gray',label='Data')
        axs[0].loglog(mbt_hat, q_hat,'b--',lw=2,label='Model'); axs[0].legend()
        axs[0].set_xlabel('MBT'); axs[0].set_ylabel('Rate (bbl/d)')
        axs[0].set_title('Rate vs MBT (log-log)')

        # rate vs time
        axs[1].scatter(t, q, s=6, c='gray', alpha=.4, label='Data')
        axs[1].plot(t_hat, q_hat,'b--',lw=2,label='Model'); axs[1].legend()
        axs[1].set_xlabel('Days'); axs[1].set_ylabel('Rate')
        axs[1].set_title('Rate vs Time')

        # cum vs time
        axs[2].scatter(t, cum_obs, s= 6, c='gray', label='Data')
        axs[2].plot(t_hat, cum_hat,'b--',lw=2, label='Model'); axs[2].legend()
        axs[2].set_xlabel('Days'); axs[2].set_ylabel('Cum')
        axs[2].set_title('Cum vs Time')

        # rate vs cum
        axs[3].scatter(cum_obs,q, s=6, c = 'gray', label='Data')
        axs[3].plot(cum_hat, q_hat,'b--',lw=2, label='Model'); axs[3].legend()
        axs[3].set_xlabel('Cum Production'); axs[2].set_ylabel('Cum')
        axs[3].set_title('Rate vs Cum')

        if isinstance(q0,pd.Series):
            plt.suptitle(f'DPL Fit for {q0.name}')

        plt.tight_layout(); plt.show()

    return pars


def interpolate_to_daily_by_prod_days(df_data_in, prod_col='Oil', days_col='Days On', plot=True):
    """
    Interpolates monthly data to daily data using producing days as the index.
    Produces a continuous daily rate, sum matches monthly totals.
    
    Args:
        df_data_in: DataFrame with columns [prod_col], [days_col]
        prod_col: Production volume column (e.g., 'Oil')
        days_col: Number of producing days column
        plot: Whether to plot the results

    Returns:
        daily_df: DataFrame with columns ['ProducingDay', 'DailyRate', 'MonthIndex']
    """
    df_data_in = df_data_in.copy()
    df_data_in = df_data_in.reset_index(drop=True)
    df_data_in = df_data_in.loc[df_data_in[prod_col].dropna().index]
    
    monthly_vol = df_data_in[prod_col].values.astype(float)
    days_on = df_data_in[days_col].values.astype(int)
    avg_daily_rate = np.where(days_on > 0, monthly_vol / days_on, 0)

    # Build producing day axis (e.g., 1,2,...,N)
    cum_days = np.concatenate(([0], np.cumsum(days_on)))
    prod_day_mid = (cum_days[:-1] + cum_days[1:]) / 2  # midpoint for each month
    valid = avg_daily_rate > 0

    # Spline interpolation through average daily rate at midpoint of each month
    x_pts = prod_day_mid[valid]
    y_pts = avg_daily_rate[valid]
    spline = interpolate.CubicSpline(x_pts, y_pts, bc_type='natural', extrapolate=True)

    # Build daily series
    daily_rows = []
    prod_day_counter = 1
    for month_idx, (n_days, total_vol) in enumerate(zip(days_on, monthly_vol)):
        if n_days == 0: continue
        days = np.arange(prod_day_counter, prod_day_counter + n_days)
        raw_daily = spline(days)
        raw_daily = np.clip(raw_daily, 0, None)
        # Normalize to monthly total
        factor = total_vol / raw_daily.sum() if raw_daily.sum() > 0 else 0
        daily_rates = raw_daily * factor
        for d, r in zip(days, daily_rates):
            daily_rows.append({'ProducingDay': d, 'DailyRate': r, 'MonthIndex': month_idx})
        prod_day_counter += n_days

    daily_df = pd.DataFrame(daily_rows)

    # Plot for inspection
    if plot:
        plt.figure(figsize=(12,6))
        plt.plot(daily_df['ProducingDay'], daily_df['DailyRate'], label='Interpolated Daily Rate', alpha=0.8)
        plt.scatter(prod_day_mid, avg_daily_rate, color='red', zorder=10, label='Monthly Avg Rate')
        for i, day in enumerate(cum_days[1:]):
            plt.axvline(day, color='k', ls='--', alpha=0.2)
        plt.xlabel('Cumulative Producing Day')
        plt.ylabel(f'Daily {prod_col} Rate')
        plt.title('Interpolated Daily Rate by Producing Day')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return daily_df


def ProductionToParams(UWI_List:list,
                       df_data_in:pd.DataFrame,
                       UWI_key:str = 'UWI10',
                       Time_key:str = 'ProducingDay',
                       OilKey:str = 'Oil',
                       GasKey:str = 'Gas',
                       WaterKey:str = 'Water'):
                                  
    ProdData = df_data_in.loc[df_data_in.UWI10.isin(UWI_List)].copy()
    ProdData.sort_values(by = [UWI_key,Time_key],inplace = True, ascending = True)
                                  
    ProdData.rename(columns = {UWI_key:'UWI10',Time_key:'Days On', OilKey:'Oil',GasKey:'Gas',WaterKey:'Water'}, inplace = True)
    ProdData = ProdData.loc[ProdData.UWI10.isin(UWI_List)]
                                  
    ProdData['NormOil'] = ProdData['Oil'] / ProdData.groupby(UWI_key)['Oil'].cummax()
    ProdData['NormGas'] = ProdData['Gas'] / ProdData.groupby(UWI_key)['Gas'].cummax()
    ProdData['NormWater'] = ProdData['Water'] / ProdData.groupby(UWI_key)['Water'].cummax()

    modelkeys = ['UWI10']

    primary = fit_dpl_with_cum(np.arange(0,1000), np.arange(0,1000)**0.8, beta_cum=1.8, p0=None, bounds=None, plot=False, t_EUR = 365*5)
    q2 = pd.DataFrame({'DailyRate':np.arange(0,1000)*3,'GasRate':np.arange(0,1000)**2,'ProducingDay':np.arange(1,1001)})
    secondary = fit_sigmoid_dual(q2,
                            w_time=1.0, w_mbt=1.0,
                            p0=None, bounds=None,
                            plot=False,
                            OilKey = 'DailyRate',
                            GasKey = 'GasRate',
                            TimeKey = 'ProducingDay')[0]

    col_names = ['UWI10']  + [f'pOil_{ix}' for ix,xx in enumerate(primary)] +  [f'pNormOil_{ix}' for ix,xx in enumerate(primary)] +  [f'pCumGOR_{ix}' for ix,xx in enumerate(secondary)] +  [f'pCumWOC_{ix}' for ix,xx in enumerate(secondary)]
    WellModels = pd.DataFrame(columns = col_names)

    pbar = tqdm(total=len(ProdData[UWI_key].unique()),  # or leave None for unknown length
            desc="Parsing wells",
            ncols=100,           # fixed width
            smoothing=0.1,       # faster reaction to slow items
            bar_format='{l_bar}{bar}')
                                  
    for iu, u in enumerate(UWI_List):
        print(f'{UWI_List[0]}: {iu}/{len(UWI_List)}')
        #u
        pbar.update() 

        m = ProdData.index[ProdData['UWI10'] == u]
        if ProdData.loc[m,['Oil']].replace(0,np.nan).dropna().shape[0] < 12:
            WellModels.loc[iu,'UWI10'] = u
            continue

        # Oil Model
        q_daily = interpolate_to_daily_by_prod_days(ProdData.loc[m],'Oil', 'Days On', plot = False)
        q_daily_norm = interpolate_to_daily_by_prod_days(ProdData.loc[m],'NormOil', 'Days On', plot = False)
        q_daily_norm.rename(columns={'DailyRate': 'NormRate'}, inplace=True)
        
        try:
            fit1 = fit_dpl_with_cum(q_daily['ProducingDay'], q_daily['DailyRate'], beta_cum=1.8, p0=None, bounds=None, plot=False, t_EUR = 365*50)
            fit_norm = fit_dpl_with_cum(q_daily_norm['ProducingDay'], q_daily_norm['NormRate'], beta_cum=1.2, p0=None, bounds=None, plot=False, t_EUR = 365*50)
        except:
            fit1 = [np.nan] * len(primary)
            fit_norm = [np.nan] * len(primary)  

        # Gas Model
        try:
            q_daily_gas = interpolate_to_daily_by_prod_days(ProdData.loc[m],'Gas', 'Days On', plot = False)
            q_daily_gas.rename(columns={'DailyRate': 'GasRate'}, inplace=True)  
        except:
            q_daily_gas = q_daily.copy()
            q_daily_gas.rename(columns={'DailyRate': 'GasRate'}, inplace=True)  
            q_daily_gas['GasRate'] = 0


        # Water Model
        try:
            q_daily_wtr = interpolate_to_daily_by_prod_days(ProdData.loc[m],'Water', 'Days On', plot = False)
            q_daily_wtr.rename(columns={'DailyRate': 'WaterRate'}, inplace=True)
        except:
            q_daily_wtr = q_daily.copy()
            q_daily_wtr.rename(columns={'DailyRate': 'WaterRate'}, inplace=True)  
            q_daily_wtr['WaterRate'] = 0

        q2 = q_daily.merge(q_daily_gas, on=['ProducingDay', 'MonthIndex'], how='left')
        q2 = q2.merge(q_daily_norm, on=['ProducingDay', 'MonthIndex'], how='left')
        q2 = q2.merge(q_daily_wtr, on=['ProducingDay', 'MonthIndex'], how='left')

        q2['DailyModel'] = dpl(q2['ProducingDay'], *fit1)

        q2['NormOil'] = q2['DailyRate']/q2['DailyRate'].cummax()
        q2['GOR'] = 1000 * q2['GasRate'] / q2['DailyRate']
        q2['MBT_Oil'] = q2['DailyRate'].cumsum() / q2['DailyRate'].replace(0, np.nan)

        m_gori = m[(ProdData.loc[m,'NormOil'] > 0.98) * (ProdData.loc[m,'ProducingDays'] < 200)]
        gori_days = ProdData.loc[m_gori,'ProducingDays'].max()
        GORi = q2.loc[q2['ProducingDay'] <= gori_days, 'GasRate'].sum() / q2.loc[q2['ProducingDay'] <= gori_days, 'DailyRate'].sum() * 1000
        m_lategor = q2.index[np.cumsum(q2['DailyRate']) / q2['DailyRate']  > 300]
        GORf = q2.loc[m_lategor,'GasRate'].sum() / q2.loc[m_lategor,'DailyRate'].sum()

        maxdays = ProdData.loc[m, 'ProducingDays'].max()
        
        try:
            mq = q2[['DailyRate','GasRate']].dropna().index
            cumgor_model = fit_sigmoid_dual(q2.loc[mq],
                            w_time=1.0, w_mbt=1.0,
                            p0=None, bounds=None,
                            plot=False,
                            OilKey = 'DailyRate',
                            GasKey = 'GasRate',
                            TimeKey = 'ProducingDay')
        except:
            cumgor_model = [[np.nan]*len(secondary)]

        try:
            mw = q2[['DailyRate','WaterRate']].dropna().index
            cumwoc_model = fit_cumWC_sigmoid(q2.loc[mw],
                            w_time=1.0, 
                            w_mbt=1.0,
                            p0=None, 
                            bounds=None,
                            plot=False,
                            OilKey = 'DailyRate',
                            WaterKey = 'WaterRate',
                            TimeKey = 'ProducingDay')
        except:
            cumwoc_model = [[np.nan]*len(secondary)]
                        
        #store Well Model parameters
        try:
            WellModels.loc[iu] = [u] + list(fit1) + list(fit_norm) + list(cumgor_model[0]) + list(cumwoc_model[0])
        except:
            WellModels.loc[iu,'UWI10'] = u

    return WellModels           

def fit_sigmoid_dual(df_daily,
                     w_time=1.0, w_mbt=1.0,
                     p0=None, bounds=None,
                     plot=True,
                     OilKey = 'Oil',
                     GasKey = 'Gas',
                     TimeKey = 'ProducingDays'):


    """
    df_daily must contain:  'Days' (int), 'Oil', 'Gas' (volumes for that day)
    Returns: parameter vector (A,K,B,M,nu) and callable predictors.
    """

    # --- build cumulative & MBT --------------------------------
    df = df_daily.copy()
    df.rename(columns = {OilKey:'Oil', GasKey:'Gas', TimeKey: 'Days'}, inplace = True)

    df['CumOil'] = df['Oil'].cumsum()
    df['CumGas'] = df['Gas'].cumsum()
    df['OilRate']= df['Oil']        # daily → already per-day volume
    df['NormOil'] = df['Oil']/df['Oil'].cummax()
    df['MBT_Oil'] = df['CumOil']/df['Oil'].replace(0,np.nan)
    mbt = df['CumOil'] / df['OilRate'].clip(1e-6)

    # targets
    t_days   = df['Days'].to_numpy(float)
    y_cumGas = df['CumGas'].to_numpy(float)
    x_mbt    = np.log10(mbt)
    y_cgor   = (df['CumGas']/df['CumOil'].clip(1e-6)).to_numpy(float)

    m_gori = df.index[df['NormOil']>0.98]
    GORi_est = df.loc[m_gori, 'Gas'].sum() / df.loc[m_gori, 'Oil'].sum()
    GORi_est = max(0.1, GORi_est)
    m_gorf = df.index[df['MBT_Oil'] > 200]
    if len(m_gorf) > 5:
        GORf_est = df.loc[m_gorf, 'Gas'].sum() / df.loc[m_gorf, 'Oil'].sum()
    else:
        GORf_est = 10
    # --- initial guesses ---------------------------------------
    if p0 is None:
        A0 = GORi_est
        K0 = GORf_est
        #A0, K0 = np.percentile(y_cgor, [5, 95])
        B0, M0 = 0.05, 15
        nu0    = 1.2
        p0 = [A0, K0, B0, M0, nu0]

    if bounds is None:
        lb = [GORi_est/2, 0, 0,   0,   0.3]
        ub = [GORi_est*2, 1000, 20,  100, 5.0]
        bounds = (lb, ub)

    # --- residual combining both domains -----------------------
    def residual(theta):
        A,K,B,M,nu = theta
        cgor_hat   = richards(x_mbt, *theta)              # CumGOR(t)
        cumGas_hat = df['CumOil'] * cgor_hat              # CumGas(t)

        # scale errors to dimensionless
        r_time = (cumGas_hat - y_cumGas) / y_cumGas.max()
        r_mbt  = (cgor_hat   - y_cgor)   / y_cgor.max()

        return np.r_[w_time*r_time, w_mbt*r_mbt]

    # --- robust fit --------------------------------------------
    res = least_squares(residual, p0, bounds=bounds,
                        loss='soft_l1', f_scale=0.3, max_nfev=40000)
    pars = res.x

    # predictors
    def cgor_hat(mbt_arr):
        return richards(np.log10(mbt_arr), *pars)

    def cumGas_hat(days_arr):
        # need CumOil(days)  – build simple interpolant from history
        oil_interp = np.interp(days_arr, df['Days'], df['Oil'])
        cumOil_int = np.interp(days_arr, df['Days'], df['CumOil'])
        return cumOil_int * cgor_hat(cumOil_int / oil_interp.clip(1e-6))

    # --- QC plot -----------------------------------------------
    if plot:
        fig,ax = plt.subplots(1,3,figsize=(15,4))
        ax[0].plot(df['Days'], y_cumGas/1e3,'k.',label='CumGas data')
        ax[0].plot(df['Days'], cumGas_hat(t_days)/1e3,'r-',lw=2,label='Fit')
        ax[0].set_title('Cum Gas vs Time'); ax[0].legend()

        ax[1].loglog(mbt, y_cgor,'k.',ms=4, label='CumGOR data')
        ax[1].loglog(mbt, cgor_hat(mbt),'r-',lw=2, label='Fit')
        ax[1].set_xlabel('MBT (days)'); ax[1].set_ylabel('Cum GOR')
        ax[1].set_title('Cum GOR vs MBT'); ax[1].legend()

        # derive smooth daily gas rate from finite diff
        gas_rate_fit = np.gradient(cumGas_hat(t_days), t_days, edge_order=2)
        ax[2].plot(t_days, gas_rate_fit,'r-',label='Gas rate fit')
        ax[2].scatter(df['Days'], df['Gas'], s=6, c='gray', alpha=.4, label='Gas raw')
        ax[2].set_title('Daily Gas'); ax[2].legend()
        plt.tight_layout(); plt.show()

    return pars, cgor_hat, cumGas_hat

def fit_cumWC_sigmoid(df_daily,
                      w_time=1.0, w_mbt=1.0,
                      p0=None, bounds=None,
                      plot=True,
                      OilKey='Oil', WaterKey='Water', TimeKey='ProducingDays'):
    """
    Fits a decreasing sigmoid to cumulative water-cut
    (starts near 1 ➜ drops to plateau).  Returns:
        pars   : (A, K, B, M, nu) of Richards on deficit (1-CumWC)
        wc_hat : callable Cum-WC(mbt_array)
        cumW_hat : callable Cum Water (days_array)
    """

    # ---------------- rename & cumulative --------------------
    df = df_daily.copy()
    df.rename(columns={OilKey:'Oil', WaterKey:'Water', TimeKey:'Days'}, inplace=True)

    df['CumOil']   = df['Oil'].cumsum()
    df['CumWater'] = df['Water'].cumsum()
    df['OilRate']  = df['Oil']                    # daily volumes → rate
    df['MBT_Oil']  = df['CumOil']/df['OilRate'].clip(1e-6)

    cum_total      = df['CumOil'] + df['CumWater']
    df['CumWC']    = df['CumWater'] / cum_total.clip(1e-6)
    deficit        = 1.0 - df['CumWC']           # starts 0, rises

    # axes
    x_mbt = np.log10(df['MBT_Oil'] )
    t     = df['Days'].to_numpy(float)

    # -------- initial & final WC estimates -------------------
    peak_rate = df['OilRate'].max()
    pre_decl  = df['OilRate'] >= 0.90*peak_rate
    pre_decl  = binary_dilation(pre_decl.to_numpy(bool), iterations=3)

    WC_init = np.average(df.loc[pre_decl, 'CumWC'],
                         weights=df.loc[pre_decl,'Oil'])

    tail     = df.loc[df['OilRate']>1].tail(15)   # last ~15 days with oil
    WC_final = np.average(tail['CumWC'], weights=tail['Oil'])

    # initial guesses for deficit (=1-CumWC)
    if p0 is None:
        A0 = 1-WC_init               # near 0
        K0 = 1-WC_final              # plateau value
        B0, M0, nu0 = 0.05, x_mbt.median(), 1.3
        p0 = [A0, K0, B0, M0, nu0]

    if bounds is None:
        lb = [0,               0,    0,   x_mbt.min(), 0.3]
        ub = [1-WC_init*0.5,   1,   20,   x_mbt.max(), 5]
        bounds = (lb, ub)

    # -------- residual (time + MBT) ---------------------------
    cumW_obs = df['CumWater'].to_numpy(float)

    def residual(theta):
        d_hat = richards(x_mbt, *theta)
        wc_hat = 1.0 - d_hat
        cumW_hat = wc_hat * (df['CumOil'] + df['CumWater'])
        r_time = (cumW_hat - cumW_obs) / cumW_obs.max()
        r_mbt  = (wc_hat - df['CumWC']) / df['CumWC'].max()
        return np.r_[w_time*r_time, w_mbt*r_mbt]

    # -------- robust fit -------------------------------------
    res  = least_squares(residual, p0, bounds=bounds,
                         loss='soft_l1', f_scale=0.3, max_nfev=40000)
    pars = res.x

    # -------- predictors -------------------------------------
    def wc_hat(mbt_arr):
        return 1.0 - richards(np.log10(mbt_arr), *pars)

    def cumW_hat(days_arr):
        oil_interp = np.interp(days_arr, df['Days'], df['Oil'])
        cumOil_int = np.interp(days_arr, df['Days'], df['CumOil'])
        mbt_arr    = cumOil_int / oil_interp.clip(1e-6)
        return (wc_hat(mbt_arr) * (cumOil_int +  # total fluids so far
                                   np.interp(days_arr, df['Days'],
                                             df['CumWater'])))

    # -------- plots ------------------------------------------
    if plot:
        fig,ax = plt.subplots(1,3,figsize=(15,4))
        ax[0].plot(df['Days'], cumW_obs/1e3,'k.',label='CumW data')
        ax[0].plot(df['Days'], cumW_hat(t)/1e3,'r-',lw=2,label='Fit')
        ax[0].set_title('Cum Water vs Time'); ax[0].legend()

        ax[1].plot(df['MBT_Oil'], df['CumWC'],'k.',ms=4,label='CumWC data')
        ax[1].plot(df['MBT_Oil'], wc_hat(df['MBT_Oil']),'r-',lw=2,label='Fit')
        ax[1].set_xscale('log')
        ax[1].set_title('Cum WC vs MBT'); ax[1].legend()

        wc_rate = np.gradient(cumW_hat(t), t, edge_order=2)
        ax[2].plot(t, wc_rate,'r-',label='Water rate fit')
        ax[2].scatter(df['Days'], df['Water'], s=8, c='gray', alpha=.4)
        ax[2].set_title('Daily Water'); ax[2].legend()
        plt.tight_layout(); plt.show()

    return pars, wc_hat, cumW_hat
