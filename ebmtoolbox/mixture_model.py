# *=========================================================================
# *
# *  Copyright Erasmus MC Rotterdam and contributors
# *
# *  Licensed under the GNU GENERAL PUBLIC LICENSE Version 3;
# *  you may not use this file except in compliance with the License.
# *
# *  Unless required by applicable law or agreed to in writing, software
# *  distributed under the License is distributed on an "AS IS" BASIS,
# *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# *  See the License for the specific language governing permissions and
# *  limitations under the License.
# *
# *=========================================================================*/
from __future__ import print_function


def calculate_likelihood_gmm(param, data, _):
    # The  dummy "_" argument is a hack to overcome how single element tuple is handled
    # differently in windows and linux. While it remains a single element tuple in windows,
    # in linux, the data type changes to that of the tuple element.
    import scipy.stats
    import numpy as np

    param_mix = param[4]
    norm_pre = scipy.stats.norm(loc=param[0], scale=param[1])
    norm_post = scipy.stats.norm(loc=param[2], scale=param[3])
    invalid_indices = np.isnan(data)
    valid_indices = np.logical_not(invalid_indices)
    likeli_pre = norm_pre.pdf(data[valid_indices])
    likeli_post = norm_post.pdf(data[valid_indices])

    likeli = np.multiply(param_mix, likeli_pre) + np.multiply(1 - param_mix, likeli_post) + 1e-100
    loglikeli = -np.sum(np.log(likeli))
    return loglikeli


def gmm_ay(data_all, data_ad_raw, data_nc_raw):
    import numpy as np
    import scipy.optimize as opt
    print('Warning using Option ay in Mixture Model:')
    print('The GMM optimization is sub-optimal as compared to the MATLAB implementation of the same algorithm.')
    n_eve = data_ad_raw.shape[1]
    params = np.zeros((n_eve, 5, 1))
    for i in range(n_eve):
        d_cni = data_nc_raw[:, i, 0]
        params[i, 0, 0] = np.nanmean(d_cni)
        params[i, 1, 0] = np.nanstd(d_cni) + 0.001  # Useful when standard deviation is 0
        d_alli = data_all[:, i]
        d_adi = data_ad_raw[:, i, 0]
        params[i, 2, 0] = np.nanmean(d_adi)
        params[i, 3, 0] = np.nanstd(d_adi) + 0.001
        params[i, 4, 0] = 0.5  # initialization with equal likelihood
        bnds = np.zeros((5, 2))
        # bnds[:,1]=(np.max(Dalli_valid_c),2*params[i,1] ,np.max(Dalli_valid_c),2*params[i,3],1)
        event_sign = params[i, 0] < params[i, 2]
        if event_sign == 1:
            bnds[:, 0] = (np.nanmin(d_alli), 0, params[i, 2, 0], 0, 0)
            bnds[:, 1] = (params[i, 0, 0], params[i, 1, 0], np.nanmax(d_alli), params[i, 3, 0], 1)
        else:
            bnds[:, 0] = (params[i, 0, 0], 0, np.nanmin(d_alli), 0, 0)
            bnds[:, 1] = (np.nanmax(d_alli), params[i, 1, 0], params[i, 2, 0], params[i, 3, 0], 1)
        idx = bnds[:, 1] - bnds[:, 0] <= 0.001
        bnds[idx, 1] = bnds[idx, 0] + 0.001  # Upper bound should be greater
        tup_arg = (d_alli[:, 0], 0)

        try:
            res = opt.least_squares(calculate_likelihood_gmm, params[i, :, 0], args=tup_arg, method='trf',
                                    bounds=np.transpose(bnds))
            if max(np.isnan(res.x)) != 1:  # In case of convergence to a nan value
                params[i, :, 0] = res.x
        except ValueError:
            print('Warning: Error in Gaussian Mixture Model')

    return params


def calculate_likelihood_gmm_n(param, data, _):
    # The "_" dummy argument is a hack to overcome how single element tuple is handled
    # differently in windows and linux. While it remains a single element tuple in windows,
    # in linux, the data type changes to that of the tuple element.
    from scipy.stats import multivariate_normal
    import numpy as np

    m = np.shape(data)

    n_feats = m[1]
    param_mix = param[-1]
    cov_n = np.zeros((n_feats, n_feats))
    mean_n = np.zeros(n_feats)
    for j in range(n_feats):
        cov_n[j, j] = param[(j * 4) + 1] ** 2.
        mean_n[j] = param[(j * 4) + 0]
    invalid_indices = np.isnan(data)
    valid_indices = np.logical_not(invalid_indices)
    likeli_pre = multivariate_normal.pdf(data[valid_indices], mean=mean_n, cov=cov_n)

    cov_d = np.zeros((n_feats, n_feats))
    mean_d = np.zeros(n_feats)
    for j in range(n_feats):
        cov_d[j, j] = param[(j * 4) + 3] ** 2.
        mean_d[j] = param[(j * 4) + 2]

    likeli_post = multivariate_normal.pdf(data[valid_indices], mean=mean_d, cov=cov_d)

    likeli = np.multiply(param_mix, likeli_pre) + np.multiply(1 - param_mix, likeli_post) + 1e-100
    loglikeli = -np.sum(np.log(likeli))
    return loglikeli


def calculate_prob_mm(data, params, val_invalid=0.5):
    # Works for single dimensional features
    import numpy as np
    import scipy.stats
    m = np.shape(data)
    p_yes = np.zeros(m)
    likeli_pre_all = np.zeros(m)
    likeli_post_all = np.zeros(m)
    for i in range(0, m[1]):
        r = 1 - params[i, 4]
        datai = data[:, i]
        paramsi = params[i, :]
        p = np.zeros(np.shape(datai))
        invalid_indices = np.isnan(datai)
        p[invalid_indices] = val_invalid
        valid_indices = np.logical_not(invalid_indices)
        datai_valid = datai[valid_indices]
        norm_pre = scipy.stats.norm(loc=paramsi[0], scale=paramsi[1])
        likeli_pre = norm_pre.pdf(datai_valid)
        norm_post = scipy.stats.norm(loc=paramsi[2], scale=paramsi[3])
        likeli_post = norm_post.pdf(datai_valid)

        p[valid_indices] = np.divide(likeli_post * r, (likeli_post * r) + (1 - r) * likeli_pre + 1e-100)
        likeli_pre_all[valid_indices, i] = likeli_pre
        likeli_post_all[valid_indices, i] = likeli_post
        p_yes[:, i] = p
    p_no = 1 - p_yes
    return p_yes, p_no, likeli_pre_all, likeli_post_all


def calculate_prob_mm_n(data, params, val_invalid=0.5):
    # Works for N dimensional features
    import numpy as np
    from scipy.stats import multivariate_normal

    m = np.shape(data)
    n_feats = m[2]
    p_yes = np.zeros((m[0], m[1]))
    likeli_pre_all = np.zeros(m)
    likeli_post_all = np.zeros(m)
    for i in range(0, m[1]):
        r = 1 - params[i, 4, 0]
        datai = data[:, i, :]

        p = np.zeros(m[0])
        invalid_indices = np.isnan(datai)
        p[invalid_indices] = val_invalid
        valid_indices = np.logical_not(invalid_indices)
        datai_valid = datai[valid_indices]
        # This is where the modification ends           
        cov_n = np.zeros((n_feats, n_feats))
        mean_n = np.zeros(n_feats)
        for j in range(n_feats):
            cov_n[j, j] = params[i, 1, j] ** 2.
            mean_n[j] = params[i, 0, j]

        likeli_pre = multivariate_normal.pdf(datai_valid, mean=mean_n, cov=cov_n)

        cov_d = np.zeros((n_feats, n_feats))
        mean_d = np.zeros(n_feats)
        for j in range(n_feats):
            cov_d[j, j] = params[i, 3, j] ** 2.
            mean_d[j] = params[i, 2, j]

        likeli_post = multivariate_normal.pdf(datai_valid, mean=mean_d, cov=cov_d)

        p[valid_indices] = np.divide(likeli_post * r, (likeli_post * r) + (1 - r) * likeli_pre + 1e-100)
        likeli_pre_all[valid_indices, i] = likeli_pre
        likeli_post_all[valid_indices, i] = likeli_post
        p_yes[:, i] = p
    p_no = 1 - p_yes
    return p_yes, p_no, likeli_pre, likeli_post
