import numpy as np
from scipy.optimize import minimize

# 根本没有办法优化...很多时候爆炸
def scale_inter_intra(observer_id_list, condition_id_list, quality_list, no_delta=True):
    if not (len(observer_id_list) == len(condition_id_list) == len(quality_list)):
        raise ValueError('observer_id, condition_id and quality must be column vectors of the same size')

    CONDs, cond_ind = np.unique(condition_id_list, return_inverse=True)
    Con_n = CONDs.size
    OBSs, obs_ind = np.unique(observer_id_list, return_inverse=True)
    Obs_n = OBSs.size

    phi_0 = np.ones(Con_n) * np.mean(quality_list)
    delta_0 = np.zeros(Obs_n)
    log_v_0 = np.ones(Obs_n) * -2
    log_w_0 = np.ones(Con_n) * -2

    phi_bounds_low = np.full(phi_0.shape, 0)
    phi_bounds_up = np.full(phi_0.shape, 5)
    log_v_0_bounds_low = - np.ones(Obs_n) * 100
    log_v_0_bounds_up = np.ones(Obs_n) * 100
    log_w_0_bounds_low = - np.ones(Con_n) * 100
    log_w_0_bounds_up = np.ones(Con_n) * 100
    Low_bound = np.concatenate((phi_bounds_low, log_v_0_bounds_low, log_w_0_bounds_low))
    Up_bound = np.concatenate((phi_bounds_up, log_v_0_bounds_up, log_w_0_bounds_up))
    bounds = np.concatenate((Low_bound[:,None], Up_bound[:,None]), axis=1)

    if no_delta:
        par_0 = np.concatenate((phi_0, log_v_0, log_w_0))

        def log_likelihood_no_delta(par):
            phi = par[:Con_n]
            log_v = par[Con_n:Con_n+Obs_n]
            log_w = par[Con_n+Obs_n:2*Con_n+Obs_n]
            # log_vw = np.log(np.exp(log_v[obs_ind]) + np.exp(log_w[cond_ind]))
            # L = np.sum((quality_list - phi[cond_ind])**2 / (2 * np.exp(log_vw)**2) + log_vw)
            log_vw = np.logaddexp(log_v[obs_ind], log_w[cond_ind])
            L = np.sum((quality_list - phi[cond_ind])**2 / (2 * np.exp(2 * log_vw)) + log_vw)
            return L

        # par_rec_NM = minimize(log_likelihood_no_delta, par_0, method='Nelder-Mead', bounds=bounds)
        par_rec_Powell = minimize(log_likelihood_no_delta, par_0, method='Powell', bounds=bounds)
        # par_rec_CG = minimize(log_likelihood_no_delta, par_0, method='CG', bounds=bounds)
        # par_rec_BFGS = minimize(log_likelihood_no_delta, par_0, method='BFGS', bounds=bounds)
        # par_rec_NewtonCG = minimize(log_likelihood_no_delta, par_0, method='Newton-CG')
        par_rec_LBFGSB = minimize(log_likelihood_no_delta, par_0, method='L-BFGS-B', bounds=bounds)
        # par_rec_TNC = minimize(log_likelihood_no_delta, par_0, method='TNC', bounds=bounds)
        # par_rec_COBYLA = minimize(log_likelihood_no_delta, par_0, method='COBYLA', bounds=bounds)
        # par_rec_SLSQP = minimize(log_likelihood_no_delta, par_0, method='SLSQP', bounds=bounds)
        par_rec_trust_constr = minimize(log_likelihood_no_delta, par_0, method='trust-constr', bounds=bounds)
        # par_rec_dogleg = minimize(log_likelihood_no_delta, par_0, method='dogleg')
        # par_rec_trust_ncg = minimize(log_likelihood_no_delta, par_0, method='trust-ncg')
        # par_rec_trust_krylov = minimize(log_likelihood_no_delta, par_0, method='trust-krylov')
        # par_rec_trust_exact = minimize(log_likelihood_no_delta, par_0, method='trust-exact')

        # par_rec_Powell = minimize(log_likelihood_no_delta, par_0, method='Powell', bounds=bounds)
        # par_rec_L_BFGS_B = minimize(log_likelihood_no_delta, par_0, method='L-BFGS-B', bounds=bounds)
        # par_rec_trust_constr = minimize(log_likelihood_no_delta, par_0, method='trust-constr', bounds=bounds)

        par_rec = par_rec_Powell
        if par_rec_LBFGSB['fun'] <= par_rec_Powell['fun'] and par_rec_LBFGSB['fun'] <= par_rec_trust_constr['fun']:
            par_rec = par_rec_LBFGSB
        elif par_rec_trust_constr['fun'] <= par_rec_Powell['fun'] and par_rec_trust_constr['fun'] <= par_rec_LBFGSB[
            'fun']:
            par_rec = par_rec_trust_constr

        phi_rec = par_rec['x'][:Con_n]
        v_rec = np.exp(par_rec['x'][Con_n:Con_n+Obs_n])
        w_rec = np.exp(par_rec['x'][Con_n+Obs_n:2*Con_n+Obs_n])
        delta_rec = np.full_like(v_rec, np.nan)

    else:
        par_0 = np.concatenate((phi_0, delta_0, log_v_0, log_w_0))

        def log_likelihood(par):
            phi = par[:Con_n]
            delta = par[Con_n:Con_n+Obs_n]
            log_v = par[Con_n+Obs_n:Con_n+2*Obs_n]
            log_w = par[Con_n+2*Obs_n:2*Con_n+2*Obs_n]
            log_vw = np.log(np.exp(log_v[obs_ind]) + np.exp(log_w[cond_ind]))
            L = np.sum((quality_list - phi[cond_ind] - delta[obs_ind])**2 / (2 * np.exp(log_vw)**2) + log_vw)
            return L

        par_rec = minimize(log_likelihood, par_0)

        phi_rec = par_rec[:Con_n]
        delta_rec = par_rec[Con_n:Con_n+Obs_n]
        v_rec = np.exp(par_rec[Con_n+Obs_n:Con_n+2*Obs_n])
        w_rec = np.exp(par_rec[Con_n+2*Obs_n:2*Con_n+2*Obs_n])

    return phi_rec, delta_rec, v_rec, w_rec, OBSs, CONDs

# if __name__ == "__main__":
#     observer_id = np.array([...])  # Fill in with your data
#     condition_id = np.array([...])  # Fill in with your data
#     quality = np.array([...])  # Fill in with your data
#     options = Options(no_delta=False)
#
#     phi_rec, delta_rec, v_rec, w_rec, OBSs, CONDs = scale_inter_intra(observer_id, condition_id, quality, options)
