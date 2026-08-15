from ._FUNCS_ import *
__all__ = ['dpl',
           'dpl_cum',
           'richards',
           'fit_dpl_with_cum',
           'dpl_residual',
           'ProductionToParams',
           'fit_sigmoid_dual',
           'fit_cumWC_sigmoid',
           'forecast_well',
           'interpolate_to_daily_by_prod_days',
           'estimate_gor_endpoints',
           'estimate_wc_endpoints',
           'population_prior',
           'build_spatial_prior_index',
           'spatial_prior',
           'estimate_final_gor_trailing']

def richards(x, A,K,B,M,nu):
    return A + (K-A) / (1 + np.exp(-B*(x-M)))**(1/nu)

def richards5(x, A, K, B, M, nu):
    """
    5-parameter Richards curve in x (we'll pass x = log10(MBT)).
    """
    # Numerically stable: clip exponent
    z = -B*(x - M)
    z = np.clip(z, -60.0, 60.0)
    return A + (K - A) / (1.0 + np.exp(z))**(1.0/np.clip(nu, 1e-6, None))
           
def dpl(t, q0, alpha, b1, b2, tx, m):
    return (q0 / (1 + alpha*t)**b1) * (1 + (t/tx)**m)**(-b2/m)

def mbt_coverage(x_mbt_max, M, coverage_scale=1.0):
    """
    Diagnostic only: how far a well's own data reaches past a fitted
    Richards sigmoid's inflection point (M, in log10(MBT) units), as a 0-1
    fraction of coverage_scale decades. Not used to gate anything in
    fit_sigmoid_dual/fit_cumWC_sigmoid (a short-history well's own M is
    just as underidentified as K, so using this to weight a per-well
    correction is circular - confirmed empirically, see A_fixed/K_fixed's
    docstring). Returned alongside fitted params purely as an FYI for
    callers who want a quick read on how far into the asymptotic region a
    given fit's own data reached.
    """
    return float(np.clip((x_mbt_max - M) / coverage_scale, 0.0, 1.0))

def estimate_gor_endpoints(df_daily, OilKey='Oil', GasKey='Gas', TimeKey='ProducingDays',
                           flat_cv_threshold=0.15, mbt_threshold=200, min_tail_points=20):
    """
    Direct (non-fit) estimates of initial and final GOR - read off the data
    rather than extrapolated from a curve fit. "Mapped", not fit: two
    earlier attempts at fitting K freely then shrinking it toward a prior
    (one gated by MBT coverage, one by Jacobian-based parameter
    uncertainty) were tested on a 30-well holdout against 4+ years of real
    production and both made things worse or were a wash - fixing a
    genuinely-known endpoint and only fitting the transition shape avoids
    the identifiability problem instead of trying to patch around it after
    the fact. See fit_sigmoid_dual's A_fixed/K_fixed.

    Initial GOR (A): averaged over the near-peak-rate window (NormOil >
    0.98). Many wells sit above bubble point pressure there - no free gas
    has evolved yet, so GOR is genuinely flat and a simple average is a
    real measurement, not an extrapolation. A_confident checks that this
    window's own GOR coefficient of variation is below flat_cv_threshold -
    if GOR is already rising even at near-peak rate, this well never showed
    a clean pre-bubble-point regime and the estimate shouldn't be trusted
    blindly.

    Final GOR (K): the actual cumulative Gas/Oil ratio using only
    production after a material-balance-time threshold (mbt_threshold) -
    a direct measurement of the late-life regime, not a guess about where
    a sigmoid will asymptote. K_confident requires at least min_tail_points
    of data past that threshold.

    Returns a dict: A, A_confident, K, K_confident, n_tail_points. When a
    well's own confidence is False for either endpoint, fall back to an
    external estimate for that one - a population median (see
    population_prior()), a play/zone P10-P90 surface, or engineering
    judgment - and pass IT as A_fixed/K_fixed instead.
    """
    df = df_daily.copy()
    df.rename(columns={OilKey: 'Oil', GasKey: 'Gas', TimeKey: 'Days'}, inplace=True)
    df['CumOil'] = df['Oil'].cumsum()
    df['MBT_Oil'] = df['CumOil'] / df['Oil'].replace(0, np.nan)
    df['NormOil'] = df['Oil'] / df['Oil'].cummax()

    m_a = df.index[df['NormOil'] > 0.98]
    gor_a_window = (df.loc[m_a, 'Gas'] / df.loc[m_a, 'Oil'].replace(0, np.nan)).dropna()
    if len(gor_a_window) >= 5 and gor_a_window.mean() > 0:
        A = float(df.loc[m_a, 'Gas'].sum() / df.loc[m_a, 'Oil'].sum())
        cv = float(gor_a_window.std() / gor_a_window.mean())
        A_confident = cv < flat_cv_threshold
    else:
        A = float(max(0.1, df['Gas'].iloc[:5].sum() / max(df['Oil'].iloc[:5].sum(), 1e-6)))
        A_confident = False

    m_k = df.index[df['MBT_Oil'] > mbt_threshold]
    if len(m_k) >= min_tail_points and df.loc[m_k, 'Oil'].sum() > 0:
        K = float(df.loc[m_k, 'Gas'].sum() / df.loc[m_k, 'Oil'].sum())
        K_confident = True
    else:
        K = None
        K_confident = False

    return {'A': A, 'A_confident': A_confident, 'K': K, 'K_confident': K_confident,
            'n_tail_points': int(len(m_k))}

def population_prior(values, confident_flags):
    """
    Simple stand-in for a play/zone P10-P90 surface, until one exists: the
    median and variance of an endpoint (K from estimate_gor_endpoints, or
    the water-cut equivalent) among wells whose own direct estimate was
    confident. Same interface either way - a single (value, variance) pair
    per well - so a real spatial surface can drop in later as a richer
    source for exactly the same A_fixed/K_fixed inputs, without changing
    anything about how the fit itself is called.

    Returns (None, None) if no wells qualify.
    """
    values = np.asarray(values, float)
    confident_flags = np.asarray(confident_flags, bool)
    trusted = values[confident_flags]
    if len(trusted) == 0:
        return None, None
    median = float(np.median(trusted))
    variance = float(np.var(trusted, ddof=1)) if len(trusted) > 1 else 0.0
    return median, variance

def build_spatial_prior_index(ref_lats, ref_lons, ref_values):
    """
    Build a reusable spatial index for spatial_prior() - construct once per
    reference population (e.g. once per formation/field, from wells with a
    confident direct estimate - the same trusted-only filtering
    population_prior() expects), then pass the returned index into many
    spatial_prior() calls rather than rebuilding a KDTree per well in a
    fleet fit.

    Only pass already-confident reference values - filter before calling
    this, exactly as population_prior() expects only trusted values.
    """
    from scipy.spatial import cKDTree
    ref_lats = np.asarray(ref_lats, float)
    ref_lons = np.asarray(ref_lons, float)
    ref_values = np.asarray(ref_values, float)
    FT_PER_DEG_LAT = 364000.0
    lat0 = float(np.mean(ref_lats))
    lon0 = float(np.mean(ref_lons))
    x = (ref_lons - lon0) * FT_PER_DEG_LAT * np.cos(np.radians(lat0))
    y = (ref_lats - lat0) * FT_PER_DEG_LAT
    tree = cKDTree(np.column_stack([x, y]))
    return {'tree': tree, 'lat0': lat0, 'lon0': lon0, 'values': ref_values}

def spatial_prior(lat, lon, index, k=40, max_dist_ft=5 * 5280.0):
    """
    Local (k-nearest-neighbor) median and variance of a reference
    population's values near (lat, lon), using an index built by
    build_spatial_prior_index(). Same (value, variance) return shape as
    population_prior() - a drop-in, spatially-aware replacement for it as
    the k_prior/wc_final_prior fallback in fit_sigmoid_dual/
    fit_cumWC_sigmoid, for wells whose own direct estimate isn't
    confident.

    Returns (None, None) if fewer than k reference wells fall within
    max_dist_ft (don't extrapolate a value from wells too far away to be
    representative of local reservoir quality).

    Confirmed empirically on 9,078 Wattenberg Niobrara wells with
    confident direct initial-GOR measurements (leave-one-out: mask each
    well's own value, predict from its 40 nearest OTHER confident
    neighbors): local median predicts the held-out well's true value with
    roughly half the error of the flat field-wide median (25.6% MAPE vs
    54.3%), and beats the flat median for 70% of individual wells. Same
    pattern for late-life water cut (5.5% vs 9.0% MAPE, 67% of wells).
    Initial GOR and late-life water cut both vary spatially across a field
    in ways a single field-wide number can't capture.

    BUT: that isolated-prediction win does NOT carry through to the actual
    downstream forecast. Two independent end-to-end tests on Wattenberg
    holdout wells (fit with K_fixed from spatial vs flat priors, compare
    forecasted vs actual gas at 4+ years) both came back null:
      - 6mo cutoff, n=223: 91.8% MAPE (spatial) vs 89.1% (flat)
      - 7mo cutoff, n=264: 89.8% MAPE (spatial) vs 91.5% (flat), and the
        spatial version's median signed error was actually WORSE
        (-57.0% vs -48.6%)
    At the short-history maturities where a fallback prior is actually
    used, the oil decline model's own instability dominates forecast
    error and swamps whatever gain a more accurate GOR prior provides.
    Also worth knowing: the population that needs a fallback at all
    collapses fast - ~46% of wells at 6mo, ~19% at 7mo, ~3% at 8mo, ~0% by
    9-10mo - so this only ever matters in a narrow 6-8mo window in the
    first place. Net: do not use this as the default K_fixed/
    wc_final_fixed fallback based on current evidence - it's a real,
    correctly-built tool, kept available for other uses (e.g. building the
    spatial P10/P90 surfaces themselves), just not shown to improve the
    fitting pipeline it was originally built for.
    """
    FT_PER_DEG_LAT = 364000.0
    x = (lon - index['lon0']) * FT_PER_DEG_LAT * np.cos(np.radians(index['lat0']))
    y = (lat - index['lat0']) * FT_PER_DEG_LAT
    dists, idxs = index['tree'].query([x, y], k=k)
    if np.max(dists) > max_dist_ft:
        return None, None
    local = index['values'][idxs]
    median = float(np.median(local))
    variance = float(np.var(local, ddof=1)) if len(local) > 1 else 0.0
    return median, variance

def estimate_final_gor_trailing(monthly_df, OilKey='oil_bbl', GasKey='gas_mcf',
                                tol=0.05, min_trail=6, max_trail=48, step=3, n_confirm=3):
    """
    Alternative to estimate_gor_endpoints()'s K: measures final/terminal GOR
    by searching BACKWARD from the end of a well's history for the
    smallest trailing window (last K months) where cumulative GOR has
    genuinely converged, rather than averaging over all months past a
    fixed material-balance-time threshold. Symmetric with how initial GOR
    is identified by searching forward from the start for the earliest
    stable window - same idea, mirrored to the other end of the well's
    life.

    Unlike the other estimate_*_endpoints() functions in this module, this
    one takes raw MONTHLY production rows (one row per calendar month, not
    daily-interpolated) - min_trail/max_trail/step are counts of months,
    validated at that granularity. Interpolating to daily first and
    reusing the same K values would silently mean "last K days" instead of
    "last K months", a much shorter and noisier window - don't do that.

    "Converged" requires n_confirm CONSECUTIVE step increases in K to all
    land within tol of each other, not just one lucky match - a single
    comparison is too easy to pass by chance on noisy monthly data.

    Tested against estimate_gor_endpoints()'s MBT-threshold K on ~1,200
    Wattenberg Niobrara wells: the two only correlate at 0.69, and the
    MBT-threshold version reads a systematic ~41% LOWER median. That's
    expected, not noise - GOR keeps rising late into a well's life, and a
    broad "MBT > threshold" window blends moderately-late months in with
    truly-terminal ones, pulling the average down. This function isolates
    only the genuinely-plateaued tail, so it should be the more physically
    accurate measurement of the true terminal value.

    NOT yet validated for downstream forecast impact when used as
    fit_sigmoid_dual's K_fixed (only shown to be a more accurate isolated
    measurement) - kept as a separate function rather than replacing
    estimate_gor_endpoints()'s K outright, since that function's existing
    MBT-threshold value is what the earlier fixed-parameter-fitting
    validation (-47% to -36% median forecast error on a 30-well holdout)
    was actually run against. Swap in cautiously.

    Returns a dict: final_gor (scf/bbl, None if never converged),
    confident (bool), trail_months (K at convergence, None if not found).
    """
    oil = monthly_df[OilKey].fillna(0).to_numpy(float)
    gas = monthly_df[GasKey].fillna(0).to_numpy(float)
    n = len(oil)
    Ks = list(range(min_trail, min(max_trail, n) + 1, step))
    if len(Ks) < n_confirm + 1:
        return {'final_gor': None, 'confident': False, 'trail_months': None}

    vals = []
    for K in Ks:
        o_sum = oil[-K:].sum()
        vals.append(gas[-K:].sum() * 1000.0 / o_sum if o_sum > 0 else np.nan)
    vals = np.asarray(vals)

    for i in range(len(vals) - n_confirm):
        window = vals[i:i + n_confirm + 1]
        if np.any(np.isnan(window)) or window[0] <= 0:
            continue
        if np.all(np.abs(window - window[0]) / window[0] < tol):
            return {'final_gor': float(np.mean(window)), 'confident': True, 'trail_months': Ks[i]}

    return {'final_gor': None, 'confident': False, 'trail_months': None}

def estimate_wc_endpoints(df_daily, OilKey='Oil', WaterKey='Water', TimeKey='ProducingDays',
                          mbt_threshold=200, min_tail_points=15):
    """
    Direct (non-fit) estimates of initial and terminal water cut - the
    water-cut equivalent of estimate_gor_endpoints() (see its docstring
    for the full rationale for measuring rather than fitting-then-
    shrinking these). Feed into fit_cumWC_sigmoid's wc_init_fixed/
    wc_final_fixed.

    Initial WC: averaged over the near-peak-rate window (OilRate >= 90% of
    peak) - wells genuinely start near 100% water cut before oil breaks
    through, so this window is a direct measurement in practice, not an
    extrapolation; always confident given at least a few points in that
    window; low n is the only failure mode.

    Final WC: the actual cumulative water-cut using only production after
    a material-balance-time threshold - not extrapolated via a sigmoid
    asymptote. WC_final_confident requires at least min_tail_points of
    data past that threshold.

    Returns a dict: WC_init, WC_init_confident, WC_final, WC_final_confident.
    """
    df = df_daily.copy()
    df.rename(columns={OilKey: 'Oil', WaterKey: 'Water', TimeKey: 'Days'}, inplace=True)
    df['CumOil'] = df['Oil'].cumsum()
    df['CumWater'] = df['Water'].cumsum()
    df['MBT_Oil'] = df['CumOil'] / df['Oil'].clip(1e-6)
    cum_total = df['CumOil'] + df['CumWater']
    df['CumWC'] = df['CumWater'] / cum_total.clip(1e-6)

    peak_rate = df['Oil'].max()
    pre_decl = df['Oil'] >= 0.90 * peak_rate
    pre_decl = binary_dilation(pre_decl.to_numpy(bool), iterations=3)
    n_init = int(pre_decl.sum())
    if n_init >= 3 and df.loc[pre_decl, 'Oil'].sum() > 0:
        WC_init = float(np.average(df.loc[pre_decl, 'CumWC'], weights=df.loc[pre_decl, 'Oil']))
        WC_init_confident = True
    else:
        WC_init = 1.0
        WC_init_confident = False

    m_k = df.index[df['MBT_Oil'] > mbt_threshold]
    if len(m_k) >= min_tail_points and df.loc[m_k, 'Oil'].sum() > 0:
        WC_final = float(np.average(df.loc[m_k, 'CumWC'], weights=df.loc[m_k, 'Oil']))
        WC_final_confident = True
    else:
        WC_final = None
        WC_final_confident = False

    return {'WC_init': WC_init, 'WC_init_confident': WC_init_confident,
            'WC_final': WC_final, 'WC_final_confident': WC_final_confident,
            'n_tail_points': int(len(m_k))}

def dpl_cum(t_array, params, dt=1.0):
    """
    Cumulative oil volume implied by the DPL rate model at arbitrary t (days).
    dpl() has no closed-form integral, so integrate on a fine internal day-grid
    and interpolate. Unlike np.interp on historical data, this stays correct
    for t past the end of the fitted history, which is what makes forward
    forecasting (not just backfilling) possible.
    """
    t_array = np.atleast_1d(np.asarray(t_array, float))
    t_max = max(float(np.max(t_array)), dt)
    grid = np.arange(0.0, t_max + dt, dt)
    q_grid = dpl(grid, *params)
    cum_grid = np.concatenate(([0.0], np.cumsum(0.5*(q_grid[1:]+q_grid[:-1])*np.diff(grid))))
    return np.interp(t_array, grid, cum_grid)

def dpl_rate(t, q0, alpha, b1, b2, tx, m):
    """
    Double-Power-Law / Transient-Hyperbolic-like rate.
    """
    t = np.asarray(t, float)
    alpha = max(alpha, 0.0)
    tx    = max(tx, 1.0)
    m     = max(m, 0.05)
    log_q = (np.log(max(q0, 1e-12))
             - b1*np.log1p(alpha*t)
             - (b2/m)*np.log1p((t/tx)**m))
    return np.exp(np.clip(log_q, -50, 50))


# -----------------------------------------------------------------
#  residual that blends rate + cumulative errors
# -----------------------------------------------------------------
def dpl_residual(theta, t, q_obs,
                 beta_cum=1.2,
                 slope_gamma=10.0, N_tail=30, target_slope=-1.0,
                 tail_horizon_mult=10.0):

    q_hat   = dpl(t, *theta)
    dt      = np.diff(np.r_[0,t])
    cum_obs = np.cumsum(q_obs*dt); cum_hat=np.cumsum(q_hat*dt)

    log_err = np.log(q_hat+1e-9)-np.log(q_obs+1e-9)
    cum_err = beta_cum*(cum_hat-cum_obs)/cum_obs.max()

    # ---- tail-slope (BDF) constraint, evaluated on a FAR-FUTURE
    # extrapolation window - never on the fit window's own tail ----
    # target_slope=-1.0 pins late-time log(rate) vs log(MBT) to the classic
    # unit-slope boundary-dominated-flow signature.
    #
    # Earlier version of this constraint evaluated it on the last N_tail
    # points of the ACTUAL fit window (the observed data), which forces
    # whatever data happens to be most recent to already look like BDF.
    # Confirmed empirically (60 real CO Niobrara wells, truncated to
    # 6/12/18/24mo and validated against 4+ years of real production, plus
    # a direct weight sweep from 0-10) that this saturates almost
    # immediately: even a small weight drags the ENTIRE fit toward an
    # artificially steep decline, because it directly competes against
    # log_err/cum_err for the same data points, and there's no way to just
    # turn it down gently once a well is young - a weight of just 3 (vs the
    # 10.0 default) already produced most of the full-strength bias.
    #
    # This version constrains only the model's deep extrapolation - a
    # window centered at tail_horizon_mult * tx (tx is theta's own current
    # transition-time estimate, so this tracks wherever the model itself
    # believes the late-time regime lives), never the observed data. That
    # decouples "does this curve eventually look physically sane" from
    # "does this curve match history" entirely, so cum-vs-time and
    # rate-vs-time fit quality on real data is never traded off against it,
    # and a young well's fit is never forced into premature BDF just
    # because that's all the data it has.
    #
    # tail_horizon_mult controls how soon (in transition-time multiples,
    # not calendar time) the BDF constraint is enforced. Lower it (e.g. to
    # 2-3) to force BDF onset sooner - deliberately useful for conservative
    # cashflow-modeling forecasts where under-forecasting EUR is preferred
    # to over-forecasting it. Raise slope_gamma above 10.0 alongside it for
    # an even harder pull toward the downside.
    tx = theta[4]
    t_max = float(t[-1]) if len(t) else 1.0
    horizon = min(max(t_max, tail_horizon_mult * max(tx, 1.0)), 50 * 365.0)
    t_tail = np.linspace(horizon * 0.9, horizon * 1.1, N_tail)
    q_tail = dpl(t_tail, *theta)
    cum_tail = dpl_cum(t_tail, theta, dt=max(1.0, horizon / 500.0))
    mbt_tail = cum_tail / (q_tail + 1e-9)

    log_q_tail = np.log(np.clip(q_tail, 1e-12, None))
    log_m_tail = np.log(np.clip(mbt_tail, 1e-12, None))
    slope_tail = np.diff(log_q_tail) / np.diff(log_m_tail)
    slope_err  = slope_gamma * (slope_tail - target_slope)

    return np.r_[log_err, cum_err, slope_err]

def fit_dpl_with_cum(t, q, beta_cum=1.2, p0=None, bounds=None, plot=True, t_EUR = None,
                     slope_gamma=10.0, N_tail=30, target_slope=-1.0,
                     tail_horizon_mult=10.0, return_cov=False):
    """
    tail_horizon_mult controls how far beyond the fitted transition time
    (tx) the boundary-dominated-flow constraint is enforced - see
    dpl_residual's docstring comment. Default (10x tx) keeps it a pure
    long-run sanity constraint on the extrapolated forecast, decoupled from
    the observed historical fit, so short-history wells aren't forced into
    premature BDF and cum-vs-time match quality on real data is preserved.

    For conservative/cashflow-modeling forecasts where under-forecasting
    EUR is preferred to over-forecasting it, lower tail_horizon_mult (e.g.
    2-3) to force BDF onset sooner, and/or raise slope_gamma above 10.0 for
    an even harder pull toward the downside.
    """
    q0 = q.copy()
    t = np.asarray(t, float)
    q = np.asarray(q, float)

    # initial guess if not given - scale tx0/alpha0 to this well's own time
    # span instead of a fixed global default, since a well with 6 months of
    # history and one with 8 years shouldn't share the same transition-time
    # starting point.
    if p0 is None:
        t_span = max(float(t[-1] - t[0]), 30.0)
        tx0 = np.clip(t_span / 3.0, 10.0, 5000.0)
        alpha0 = 2.0 / t_span
        p0 = [q[0], alpha0, 0.2, 1.0, tx0, 4.0]
    if bounds is None:
        bounds = ([0., 0., 0., 0.9, 10., 1.],
                  [np.inf, 10., 2., 1.1, 5000., 10.])

    # p0's defaults (or a caller-supplied p0 combined with caller-supplied
    # bounds) can land outside custom bounds - e.g. the default b2 guess of
    # 1.0 is infeasible against any bounds with a b2 ceiling below that.
    # least_squares raises outright in that case rather than clipping, so
    # do it here instead of making every caller with custom bounds
    # rediscover this by hitting the same crash.
    p0 = np.clip(np.asarray(p0, float), bounds[0], bounds[1])

    res = least_squares(
            dpl_residual, p0,
            args=(t, q, beta_cum, slope_gamma, N_tail, target_slope,
                  tail_horizon_mult),
            bounds=bounds,
            loss='soft_l1',       # robust to outliers
            f_scale=0.3,          # “softness”; tune 0.1–1
            max_nfev=40000)

    pars = res.x

    if return_cov:
        # Approximate parameter covariance from the Jacobian at the solution
        # (standard least_squares asymptotic covariance estimate). Lets
        # callers flag wells whose fit is poorly constrained instead of
        # trusting every point estimate equally when regressing curve-shape
        # parameters against completion drivers downstream.
        try:
            J = res.jac
            dof = max(len(res.fun) - len(pars), 1)
            s_sq = 2.0 * res.cost / dof
            pcov = np.linalg.inv(J.T @ J) * s_sq
        except Exception:
            pcov = np.full((len(pars), len(pars)), np.nan)

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

    if return_cov:
        return pars, pcov
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
                       WaterKey:str = 'Water',
                       progress_bar = True):
                                  
    ProdData = df_data_in.loc[df_data_in.UWI10.isin(UWI_List)].copy()

    DateKey = GetKey(ProdData,'Date|Month')[0]
    ProdData.sort_values(by = [UWI_key,DateKey],inplace = True, ascending = True)
                                  
    ProdData.rename(columns = {UWI_key:'UWI10',Time_key:'Days On', OilKey:'Oil',GasKey:'Gas',WaterKey:'Water'}, inplace = True)
    ProdData = ProdData.loc[ProdData.UWI10.isin(UWI_List)]
                                  
    ProdData['NormOil'] = ProdData['Oil'] / ProdData.groupby(UWI_key)['Oil'].cummax()
    ProdData['NormGas'] = ProdData['Gas'] / ProdData.groupby(UWI_key)['Gas'].cummax()
    ProdData['NormWater'] = ProdData['Water'] / ProdData.groupby(UWI_key)['Water'].cummax()

    modelkeys = ['UWI10']

    # Fixed parameter counts for the two model families - dpl() always takes
    # 6 params (q0, alpha, b1, b2, tx, m), richards() always takes 5
    # (A, K, B, M, nu). Previously this ran two throwaway fits on synthetic
    # data purely to count len(params), which wasted a fit per call and
    # silently broke if either model's signature ever changed shape.
    primary = [np.nan] * 6
    secondary = [np.nan] * 5

    col_names = ['UWI10']  + [f'pOil_{ix}' for ix,xx in enumerate(primary)] +  [f'pNormOil_{ix}' for ix,xx in enumerate(primary)] +  [f'pCumGOR_{ix}' for ix,xx in enumerate(secondary)] +  [f'pCumWOC_{ix}' for ix,xx in enumerate(secondary)]
    WellModels = pd.DataFrame(columns = col_names)
    if progress_bar:
        pbar = tqdm(total=len(ProdData[UWI_key].unique()),  # or leave None for unknown length
                desc="Parsing wells",
                ncols=100,           # fixed width
                smoothing=0.4,       # faster reaction to slow items
                bar_format='{l_bar}{bar}')
                                  
    for iu, u in enumerate(UWI_List):
        print(f'{UWI_List[0]}: {iu}/{len(UWI_List)}')
        #u
        if progress_bar:
            pbar.update() 

        m = ProdData.index[ProdData['UWI10'] == u]
        if ProdData.loc[m,['Oil']].replace(0,np.nan).dropna().shape[0] < 12:
            WellModels.loc[iu,'UWI10'] = u
            continue

        # Oil Model
        q_daily = interpolate_to_daily_by_prod_days(ProdData.loc[m],'Oil', 'Days On', plot = False)
        q_daily_norm = interpolate_to_daily_by_prod_days(ProdData.loc[m],'NormOil', 'Days On', plot = False)
        q_daily_norm.rename(columns={'DailyRate': 'NormRate'}, inplace=True)

        m_qdaily_oil = q_daily.index[q_daily.DailyRate > 0]
        m_qdaily_normoil = q_daily_norm.index[q_daily_norm.NormRate > 0]
               
        try:
            fit1 = fit_dpl_with_cum(q_daily.loc[m_qdaily_oil, 'ProducingDay'], q_daily.loc[m_qdaily_oil, 'DailyRate'], beta_cum=1.8, p0=None, bounds=None, plot=False, t_EUR = 365*50)
            fit_norm = fit_dpl_with_cum(q_daily_norm.loc[m_qdaily_normoil, 'ProducingDay'], q_daily_norm.loc[m_qdaily_normoil, 'NormRate'], beta_cum=1.2, p0=None, bounds=None, plot=False, t_EUR = 365*50)
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

        m_gori = m[(ProdData.loc[m,'NormOil'] > 0.98) * (ProdData.loc[m,'Days On'] < 200)]
        gori_days = ProdData.loc[m_gori,'Days On'].max()
        GORi = q2.loc[q2['ProducingDay'] <= gori_days, 'GasRate'].sum() / q2.loc[q2['ProducingDay'] <= gori_days, 'DailyRate'].sum() * 1000
        m_lategor = q2.index[np.cumsum(q2['DailyRate']) / q2['DailyRate']  > 300]
        GORf = q2.loc[m_lategor,'GasRate'].sum() / q2.loc[m_lategor,'DailyRate'].sum()

        maxdays = ProdData.loc[m, 'Days On'].max()
        
        # fit1 (this well's own raw-oil DPL params) is passed through as
        # oil_params so cumGas_hat/cumW_hat can extrapolate past the end of
        # history using the DPL forecast, not just backfill observed months.
        fit1_valid = fit1 if np.all(np.isfinite(fit1)) else None

        try:
            mq = q2[['DailyRate','GasRate']].dropna().index
            cumgor_model = fit_sigmoid_dual(q2.loc[mq],
                            w_time=1.0, w_mbt=1.0,
                            p0=None, bounds=None,
                            plot=False,
                            OilKey = 'DailyRate',
                            GasKey = 'GasRate',
                            TimeKey = 'ProducingDay',
                            oil_params = fit1_valid)
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
                            TimeKey = 'ProducingDay',
                            oil_params = fit1_valid)
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
                     TimeKey = 'ProducingDays',
                     oil_params=None,
                     A_fixed=None, K_fixed=None):


    """
    df_daily must contain:  'Days' (int), 'Oil', 'Gas' (volumes for that day)
    oil_params: optional 6-param dpl() fit for this well (from fit_dpl_with_cum).
        When given, cumGas_hat can be evaluated at times past the end of the
        historical record - MBT is computed from the DPL model's own oil
        forecast instead of np.interp on history, which just holds the last
        observed value flat. Without it, cumGas_hat only backfills history.

    A_fixed / K_fixed: hold the initial/final GOR asymptotes constant
    instead of fitting them, and only fit the transition shape (B, M, nu).
    Use estimate_gor_endpoints() to get a direct, non-extrapolated estimate
    of either from this well's own data when its A_confident/K_confident
    come back True; when they don't (short-history wells whose data
    doesn't reach a clean pre-bubble-point window or far enough past the
    MBT threshold), supply an external value instead - population_prior()
    over a set of long-history wells' confident estimates, a play/zone
    P10-P90 surface, or engineering judgment.

    Two earlier designs tried to keep K a free-fit parameter and shrink it
    toward a prior after the fact (one gated by MBT coverage, one by
    Jacobian-based parameter uncertainty) - both tested on a 30-well
    holdout against 4+ years of real production, and both made things
    worse or were a wash: freely-fit K is generally underidentified from
    short data (routinely trades off against M in ways a marginal-variance
    estimate doesn't catch), so trying to correct it after fitting doesn't
    reliably work. Fixing a genuinely-known endpoint and only fitting the
    shape avoids the problem instead of trying to patch around it.

    Returns: parameter vector (A,K,B,M,nu), callable predictors, and this
    well's own MBT coverage fraction (0-1, see mbt_coverage() if imported
    separately) - only meaningful for long-history wells; not used
    internally here, kept for callers who want it.
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
        A0 = A_fixed if A_fixed is not None else GORi_est
        K0 = K_fixed if K_fixed is not None else GORf_est
        # M0=15 (fixed) was wildly off-scale: x=log10(MBT) for real wells
        # here runs roughly 0-4 (MBT of 1 to ~10,000 days) - anchor to the
        # data's own log10(MBT) median instead (fit_cumWC_sigmoid already
        # did this correctly).
        B0, M0 = 0.05, x_mbt.median()
        nu0    = 1.2
        p0 = [A0, K0, B0, M0, nu0]

    if bounds is None:
        lb = [GORi_est/2, 0, 0,   x_mbt.min(),   0.3]
        # K's upper bound must scale with the data-estimated terminal GOR
        # (K0=GORf_est is used as p0[1]) - a fixed 1000 scf/bbl cap made p0
        # infeasible and crashed least_squares outright for any well whose
        # late-life GOR climbs past 1000, which is routine for DJ Basin
        # Niobrara/Codell wells late in life.
        ub = [GORi_est*2, max(GORf_est*3, 1000), 20,  x_mbt.max(), 5.0]
        bounds = (lb, ub)

    # --- reduced parameter set when A/K are held fixed ----------
    fixed = {}
    if A_fixed is not None: fixed[0] = A_fixed
    if K_fixed is not None: fixed[1] = K_fixed
    free_idx = [i for i in range(5) if i not in fixed]

    def expand(theta_free):
        theta = np.empty(5)
        for i, v in fixed.items():
            theta[i] = v
        for i, v in zip(free_idx, theta_free):
            theta[i] = v
        return theta

    p0_free = [p0[i] for i in free_idx]
    bounds_free = ([bounds[0][i] for i in free_idx], [bounds[1][i] for i in free_idx])
    p0_free = list(np.clip(np.asarray(p0_free, float), bounds_free[0], bounds_free[1]))

    # --- residual combining both domains -----------------------
    def residual(theta_free):
        theta = expand(theta_free)
        cgor_hat   = richards(x_mbt, *theta)              # CumGOR(t)
        cumGas_hat = df['CumOil'] * cgor_hat              # CumGas(t)

        # scale errors to dimensionless
        r_time = (cumGas_hat - y_cumGas) / y_cumGas.max()
        r_mbt  = (cgor_hat   - y_cgor)   / y_cgor.max()

        return np.r_[w_time*r_time, w_mbt*r_mbt]

    # --- robust fit --------------------------------------------
    res = least_squares(residual, p0_free, bounds=bounds_free,
                        loss='soft_l1', f_scale=0.3, max_nfev=40000)
    pars = expand(res.x)
    coverage = mbt_coverage(x_mbt.max(), pars[3])

    # predictors
    def cgor_hat(mbt_arr):
        return richards(np.log10(mbt_arr), *pars)

    def cumGas_hat(days_arr):
        days_arr = np.atleast_1d(np.asarray(days_arr, float))
        if oil_params is not None:
            # extrapolates correctly beyond history via the fitted DPL model
            oil_interp = dpl(days_arr, *oil_params)
            cumOil_int = dpl_cum(days_arr, oil_params)
        else:
            # history-only fallback: clamps at the last observed value past
            # the end of df['Days'], so this branch cannot forecast forward
            oil_interp = np.interp(days_arr, df['Days'], df['Oil'])
            cumOil_int = np.interp(days_arr, df['Days'], df['CumOil'])
        return cumOil_int * cgor_hat(cumOil_int / np.clip(oil_interp, 1e-6, None))

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

    return pars, cgor_hat, cumGas_hat, coverage

def fit_cumWC_sigmoid(df_daily,
                      w_time=1.0, w_mbt=1.0,
                      p0=None, bounds=None,
                      plot=True,
                      OilKey='Oil', WaterKey='Water', TimeKey='ProducingDays',
                      oil_params=None,
                      wc_init_fixed=None, wc_final_fixed=None):
    """
    Fits a decreasing sigmoid to cumulative water-cut
    (starts near 1 ➜ drops to plateau).  Returns:
        pars     : (A, K, B, M, nu) of Richards on deficit (1-CumWC)
        wc_hat   : callable Cum-WC(mbt_array)
        cumW_hat : callable Cum Water (days_array)
        coverage : this well's own MBT coverage fraction (0-1, diagnostic
                   only - see mbt_coverage())

    oil_params: optional 6-param dpl() fit for this well. When given,
        cumW_hat extrapolates past history using the DPL oil forecast for
        MBT instead of clamping at the last observed value (see
        fit_sigmoid_dual's oil_params for the same rationale).

    wc_init_fixed / wc_final_fixed: hold the initial/terminal water-cut
        asymptotes constant (as water-cut fractions, 0-1) instead of
        fitting them, and only fit the transition shape. Same rationale
        and same two-pass-shrinkage designs abandoned in favor of this, as
        fit_sigmoid_dual's A_fixed/K_fixed - see that docstring. Use
        estimate_wc_endpoints() for a direct, non-extrapolated estimate
        from this well's own data when confident, else supply an external
        value (population_prior() over confident long-history wells, a
        play/zone P10-P90 surface, or engineering judgment).
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

    # initial guesses for deficit (=1-CumWC); A/K here are the deficit's
    # own asymptotes, so a fixed water-cut fraction wc gets passed through
    # as (1-wc).
    if p0 is None:
        A0 = (1 - wc_init_fixed) if wc_init_fixed is not None else (1 - WC_init)
        K0 = (1 - wc_final_fixed) if wc_final_fixed is not None else (1 - WC_final)
        B0, M0, nu0 = 0.05, x_mbt.median(), 1.3
        p0 = [A0, K0, B0, M0, nu0]

    if bounds is None:
        lb = [0,               0,    0,   x_mbt.min(), 0.3]
        ub = [1-WC_init*0.5,   1,   20,   x_mbt.max(), 5]
        bounds = (lb, ub)

    # -------- reduced parameter set when A/K are held fixed -----
    fixed = {}
    if wc_init_fixed is not None: fixed[0] = 1 - wc_init_fixed
    if wc_final_fixed is not None: fixed[1] = 1 - wc_final_fixed
    free_idx = [i for i in range(5) if i not in fixed]

    def expand(theta_free):
        theta = np.empty(5)
        for i, v in fixed.items():
            theta[i] = v
        for i, v in zip(free_idx, theta_free):
            theta[i] = v
        return theta

    p0_free = [p0[i] for i in free_idx]
    bounds_free = ([bounds[0][i] for i in free_idx], [bounds[1][i] for i in free_idx])
    p0_free = list(np.clip(np.asarray(p0_free, float), bounds_free[0], bounds_free[1]))

    # -------- residual (time + MBT) ---------------------------
    cumW_obs = df['CumWater'].to_numpy(float)

    def residual(theta_free):
        theta = expand(theta_free)
        d_hat = richards(x_mbt, *theta)
        wc_hat = 1.0 - d_hat
        cumW_hat = wc_hat * (df['CumOil'] + df['CumWater'])
        r_time = (cumW_hat - cumW_obs) / cumW_obs.max()
        r_mbt  = (wc_hat - df['CumWC']) / df['CumWC'].max()
        return np.r_[w_time*r_time, w_mbt*r_mbt]

    # -------- robust fit -------------------------------------
    res  = least_squares(residual, p0_free, bounds=bounds_free,
                         loss='soft_l1', f_scale=0.3, max_nfev=40000)
    pars = expand(res.x)
    coverage = mbt_coverage(x_mbt.max(), pars[3])

    # -------- predictors -------------------------------------
    def wc_hat(mbt_arr):
        return 1.0 - richards(np.log10(mbt_arr), *pars)

    def cumW_hat(days_arr):
        days_arr = np.atleast_1d(np.asarray(days_arr, float))
        if oil_params is not None:
            oil_interp = dpl(days_arr, *oil_params)
            cumOil_int = dpl_cum(days_arr, oil_params)
        else:
            oil_interp = np.interp(days_arr, df['Days'], df['Oil'])
            cumOil_int = np.interp(days_arr, df['Days'], df['CumOil'])
        mbt_arr = cumOil_int / np.clip(oil_interp, 1e-6, None)
        wc      = wc_hat(mbt_arr)
        # CumWC = CumWater/(CumOil+CumWater)  =>  CumWater = CumWC*CumOil/(1-CumWC)
        return wc * cumOil_int / np.clip(1.0 - wc, 1e-9, None)

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

    return pars, wc_hat, cumW_hat, coverage


def forecast_well(oil_params, gor_params, wc_params, t_array):
    """
    Reconstruct a full 3-phase (oil/gas/water) forecast at arbitrary times
    t_array (days since first production), spanning both history and future,
    from three already-fitted parameter sets alone - no dependence on raw
    production data past this point:

        oil_params : 6 dpl() params from fit_dpl_with_cum on raw daily oil rate
        gor_params : 5 richards() params from fit_sigmoid_dual (CumGOR vs log10 MBT)
        wc_params  : 5 richards() params from fit_cumWC_sigmoid (deficit vs log10 MBT)

    The oil DPL model is the spine: it generates OilRate(t) and CumOil(t),
    from which MBT(t) = CumOil(t)/OilRate(t) is derived. GOR and water-cut
    are then evaluated as sigmoids of that same MBT(t), and differentiated
    back to rates. This is the explicit form of the cascade that
    fit_sigmoid_dual/fit_cumWC_sigmoid's oil_params argument makes possible.

    Returns a DataFrame: Days, OilRate, CumOil, MBT_Oil, CumGOR, GasRate,
    CumGas, CumWC, WaterRate, CumWater.
    """
    t_array = np.atleast_1d(np.asarray(t_array, float))

    def _cums(t):
        oil_rate = dpl(t, *oil_params)
        cum_oil  = dpl_cum(t, oil_params)
        mbt      = cum_oil / np.clip(oil_rate, 1e-9, None)
        log_mbt  = np.log10(np.clip(mbt, 1e-9, None))
        cum_gor  = richards(log_mbt, *gor_params)
        cum_gas  = cum_oil * cum_gor
        deficit  = richards(log_mbt, *wc_params)
        cum_wc   = 1.0 - deficit
        # invert CumWC = CumWater/(CumOil+CumWater) for CumWater
        cum_water = cum_wc * cum_oil / np.clip(1.0 - cum_wc, 1e-9, None)
        return oil_rate, cum_oil, mbt, cum_gor, cum_gas, cum_wc, cum_water

    oil_rate, cum_oil, mbt, cum_gor, cum_gas, cum_wc, cum_water = _cums(t_array)

    # GasRate/WaterRate need a local time-derivative. np.gradient requires
    # >=3 points for edge_order=2 and raises outright on a single point -
    # but callers legitimately want a rate at one forecast date (e.g. "as
    # of today"), not just a curve. When too few points were requested,
    # evaluate on a small internal +/-1 day window instead and interpolate
    # the derivative back onto the caller's t_array, rather than requiring
    # every caller to pad their own input.
    if len(t_array) >= 3:
        gas_rate = np.gradient(cum_gas, t_array, edge_order=2)
        water_rate = np.gradient(cum_water, t_array, edge_order=2)
    else:
        t_dense = np.unique(np.concatenate([t_array - 1.0, t_array, t_array + 1.0]))
        t_dense = t_dense[t_dense > 0]
        _, _, _, _, cum_gas_d, _, cum_water_d = _cums(t_dense)
        gas_rate_d = np.gradient(cum_gas_d, t_dense, edge_order=1)
        water_rate_d = np.gradient(cum_water_d, t_dense, edge_order=1)
        gas_rate = np.interp(t_array, t_dense, gas_rate_d)
        water_rate = np.interp(t_array, t_dense, water_rate_d)

    return pd.DataFrame({
        'Days': t_array, 'OilRate': oil_rate, 'CumOil': cum_oil, 'MBT_Oil': mbt,
        'CumGOR': cum_gor, 'GasRate': gas_rate, 'CumGas': cum_gas,
        'CumWC': cum_wc, 'WaterRate': water_rate, 'CumWater': cum_water,
    })
