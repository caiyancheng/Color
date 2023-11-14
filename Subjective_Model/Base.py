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
    v_0 = np.exp(np.ones(Obs_n))
    w_0 = np.exp(np.ones(Con_n))

    if no_delta:
        par_0 = np.concatenate((phi_0, v_0, w_0))

        def log_likelihood_no_delta(par):
            phi = par[:Con_n]
            log_v = par[Con_n:Con_n+Obs_n]
            log_w = par[Con_n+Obs_n:2*Con_n+Obs_n]
            log_vw = np.log(np.exp(log_v[obs_ind]) + np.exp(log_w[cond_ind]))
            L = np.sum((quality_list - phi[cond_ind])**2 / (2 * np.exp(log_vw)**2) + log_vw)
            return L

        par_rec = minimize(log_likelihood_no_delta, par_0)

        phi_rec = par_rec[:Con_n]
        v_rec = np.exp(par_rec[Con_n:Con_n+Obs_n])
        w_rec = np.exp(par_rec[Con_n+Obs_n:2*Con_n+Obs_n])
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
