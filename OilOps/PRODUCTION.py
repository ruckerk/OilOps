__all__ = ['dpl',
           'fit_dpl_with_cum',
           'dpl_residual']


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
