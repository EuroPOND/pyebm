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

from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.utils import resample

from ebmtoolbox import corealgo as ca
from ebmtoolbox import mixture_model as mm
from ebmtoolbox import util
from ebmtoolbox import weighted_mallows as wma


def fit(data_in,
        method_options=False,
        verbose_options=False,
        factors=None,
        labels=None,
        data_test=None):
    if factors is None:
        factors = ['Age', 'Sex', 'ICV']
    if labels is None:
        labels = ['CN', 'MCI', 'AD']
    if data_test is None:
        data_test = []
    # Default Options for the function call

    default_method_options = namedtuple('DefaultMethodOptions',
                                        ' MixtureModel Bootstrap PatientStaging NStartpoints Niterations N_MCMC')
    default_verbose_options = namedtuple('DefaultVerboseOptions',
                                         'Distributions Ordering PlotOrder WriteBootstrapData PatientStaging')
    default_method_options.JointFit = 0
    default_method_options.MixtureModel = 'vv2'
    default_method_options.Bootstrap = 0
    default_method_options.PatientStaging = ['exp', 'p']
    default_method_options.NStartpoints = 10
    default_method_options.Niterations = 1000
    default_method_options.N_MCMC = 10000
    default_verbose_options.Distributions = 0
    default_verbose_options.Ordering = 0
    default_verbose_options.PlotOrder = 0
    default_verbose_options.WriteBootstrapData = 0
    default_verbose_options.PatientStaging = 0

    # Replacing the Default Options with Options given by the user
    if type(method_options) != bool:
        for fld in method_options._fields:
            setattr(default_method_options, fld, getattr(method_options, fld))

    if type(verbose_options) != bool:
        for fld in verbose_options._fields:
            setattr(default_verbose_options, fld, getattr(verbose_options, fld))


    # Data Preparation for debm
    pd_data_all, unique_subj_ids = util.pd_read_data(data_in, 0, labels=labels)
    pd_data_test_all, unique_test_subj_ids = util.pd_read_data(data_test, 0, labels=labels)
    pd_data_all, pd_data_test_all, biomarkers_list = util.correct_confounders(pd_data_all, pd_data_test_all, factors)

    num_events = len(biomarkers_list)
    idx_cn = pd_data_all['Diagnosis'] == 1
    idx_ad = pd_data_all['Diagnosis'] == 3
    idx_mci = pd_data_all['Diagnosis'] == 2
    data_all = util.pd2mat(pd_data_all, biomarkers_list, 0)
    if len(pd_data_test_all) > 0:
        data_test_all = util.pd2mat(pd_data_test_all, biomarkers_list, 0)
    else:
        data_test_all = []
    idx_cn = np.where(idx_cn)
    idx_cn = idx_cn[0]
    idx_ad = np.where(idx_ad)
    idx_ad = idx_ad[0]
    idx_mci = np.where(idx_mci)
    idx_mci = idx_mci[0]
    data_ad_raw = data_all[idx_ad, :, :]
    data_cn_raw = data_all[idx_cn, :, :]
    data_mci_raw = data_all[idx_mci, :, :]
    ptid_ad_raw = pd_data_all.loc[idx_ad, 'PTID']
    ptid_cn_raw = pd_data_all.loc[idx_cn, 'PTID']
    ptid_mci_raw = pd_data_all.loc[idx_mci, 'PTID']
    # Data Preparation for Bootstrapping
    pi0_all = []
    data_ad_raw_list = []
    params_opt_all = []
    event_centers_all = []
    data_cn_raw_list = []
    data_all_list = []
    ptid_all_list = []
    for i in range(default_method_options.Bootstrap):
        bs_data_ad_raw = resample(data_ad_raw, random_state=i)
        bs_ptid_ad_raw = resample(ptid_ad_raw, random_state=i)
        bs_data_cn_raw = resample(data_cn_raw, random_state=i)
        bs_ptid_cn_raw = resample(ptid_cn_raw, random_state=i)
        bs_data_mci_raw = resample(data_mci_raw, random_state=i)
        bs_ptid_mci_raw = resample(ptid_mci_raw, random_state=i)
        bs_data_all = np.concatenate((bs_data_ad_raw, bs_data_cn_raw, bs_data_mci_raw))
        bs_ptid_all = np.concatenate((bs_ptid_ad_raw, bs_ptid_cn_raw, bs_ptid_mci_raw))
        labels_ad = np.zeros(len(bs_data_ad_raw)) + 3
        labels_cn = np.zeros(len(bs_data_cn_raw)) + 1
        labels_mci = np.zeros(len(bs_data_mci_raw)) + 2
        labels_all = np.concatenate((labels_ad, labels_cn, labels_mci))
        if type(default_verbose_options.WriteBootstrapData) == str:
            str_out = '%s_%s.csv' % (default_verbose_options.WriteBootstrapData, i)
            dbs = pd.DataFrame(bs_data_all[:, :, 0], columns=biomarkers_list)
            dbs['Diagnosis'] = labels_all
            dbs.to_csv(str_out, index=False)
        data_ad_raw_list.append(bs_data_ad_raw)
        data_cn_raw_list.append(bs_data_cn_raw)
        data_all_list.append(bs_data_all)
        ptid_all_list.append(bs_ptid_all)
    if default_method_options.Bootstrap == False:
        data_ad_raw_list.append(data_ad_raw)
        data_cn_raw_list.append(data_cn_raw)
        data_all_list.append(data_all)
        ptid_all_list.append(pd_data_all['PTID'])
    subj_train_all = []
    subj_test_all = []
    for i in range(len(data_ad_raw_list)):  # For each bootstrapped iteration
        # Reject possibly wrongly labeled data
        if len(data_ad_raw_list) > 1:
            print([i]),
        data_ad_raw = data_ad_raw_list[i]
        data_cn_raw = data_cn_raw_list[i]
        data_all = data_all_list[i]
        data_ad_pruned, data_cn_pruned, params_raw, params_pruned = ca.reject(data_ad_raw, data_cn_raw)
        # Bias Correction to get an unbiased estimate
        if default_method_options.MixtureModel == 'vv2':
            params_opt = params_pruned
            mixes_old = params_opt[:, 4, 0]
            flag_stop = 0
            while flag_stop == 0:
                params_optmix, bnds_all = ca.gmm_control(data_all, data_cn_pruned, data_ad_pruned, params_opt,
                                                         itvl=0.001)
                params_opt, bnds_all = ca.gmm_control(data_all, data_cn_pruned, data_ad_pruned, params_optmix,
                                                      type_opt=3, params_pruned=params_pruned)
                mixes = params_opt[:, 4, 0]
                if np.mean(np.abs(mixes - mixes_old)) < 10 ** -2:
                    flag_stop = 1
                mixes_old = np.copy(mixes)
        elif default_method_options.MixtureModel == 'vv1':
            params_opt, bnds_all = ca.gmm_control(data_all, data_cn_pruned, data_ad_pruned, params_pruned, type_opt=1)
        elif default_method_options.MixtureModel == 'ay':
            params_opt = mm.gmm_ay(data_all, data_ad_raw, data_cn_raw)

        # Get Posterior Probabilities
        p_yes, p_no, likeli_post, likeli_pre = ca.classify(data_all, params_opt)
        # Probabilistic Kendall's Tau based Generalized Mallows Model
        mix = np.copy(params_opt[:, 4, 0])
        params_opt[:, 4, 0] = 0.5
        pi0, event_centers = ca.mcmc(data_all[:, :, 0], params_opt, default_method_options.N_MCMC, params_opt[:, 4, 0],
                                     default_method_options.NStartpoints,
                                     default_method_options.Niterations)
        params_opt[:, 4, 0] = mix
        event_centers_all.append(event_centers)
        for ni in range(num_events):
            idx = np.isnan(p_yes[:, ni])
            p_yes[idx, ni] = 1 - params_opt[ni, 4, 0]

        if default_method_options.PatientStaging[1] == 'p':
            y = p_yes
            n = 1 - p_yes
        else:
            y = likeli_post
            n = likeli_pre

        subj_stages = ca.staging(pi0, event_centers, y, n, default_method_options.PatientStaging)
        if len(data_test_all) > 0:
            p_yes_test, p_no_test, likeli_post_test, likeli_pre_test = ca.classify(data_test_all, params_opt)
            for ni in range(num_events):
                idx = np.isnan(p_yes_test[:, ni])
                p_yes_test[idx, ni] = 1 - params_opt[ni, 4, 0]
            if default_method_options.PatientStaging[1] == 'p':
                y = p_yes_test
                n = 1 - p_yes_test
            else:
                y = likeli_post_test
                n = likeli_pre_test
            subj_stages_test = ca.staging(pi0, event_centers, y, n, default_method_options.PatientStaging)
        pi0_all.append(pi0)
        params_opt_all.append(params_opt)
        event_centers_all.append(event_centers)
        subj_train = pd.DataFrame(columns=['PTID', 'Orderings', 'Weights', 'Stages'])
        subj_test = pd.DataFrame(columns=['PTID', 'Orderings', 'Weights', 'Stages'])

        subj_train['PTID'] = ptid_all_list[i]
        so_list, weights_list = util.prob_2_list_and_weights(p_yes)
        subj_train['Orderings'] = so_list
        subj_train['Weights'] = weights_list
        subj_train['Stages'] = subj_stages

        if len(pd_data_test_all) > 0:
            so_list, weights_list = util.prob_2_list_and_weights(p_yes_test)
            subj_test['Weights'] = weights_list
            subj_test['Orderings'] = so_list
            subj_test['PTID'] = pd_data_test_all['PTID']
            subj_test['Stages'] = subj_stages_test
        subj_test_all.append(subj_test)
        subj_train_all.append(subj_train)
    # Get Mean Ordering of all the bootstrapped iterations.
    if len(data_ad_raw_list) > 1:
        wts = np.arange(num_events, 0, -1)
        wts_all = np.tile(wts, (len(data_ad_raw_list), 1)).tolist()
        (pi0_mean, bestscore, scores) = wma.WeightedMallows.consensus(num_events, pi0_all, wts_all, [])
    else:
        pi0_mean = pi0_all[0]

    model_output = namedtuple('ModelOutput',
                              'MeanCentralOrdering EventCenters CentralOrderings BiomarkerList BiomarkerParameters')
    model_output.BiomarkerList = biomarkers_list
    model_output.MeanCentralOrdering = pi0_mean
    model_output.EventCenters = event_centers_all
    model_output.CentralOrderings = pi0_all
    model_output.BiomarkerParameters = params_opt_all

    # Visualize Results
    if default_verbose_options.Ordering == 1:
        util.visualize_ordering(biomarkers_list, pi0_all, pi0_mean, default_verbose_options.PlotOrder)
    if default_verbose_options.Distributions == 1:
        params_all = [params_opt]
        util.visualize_biomarker_distribution(data_all, params_all, biomarkers_list)
    if default_verbose_options.PatientStaging == 1:
        util.visualize_staging(subj_stages, pd_data_all['Diagnosis'], labels)

    if not default_method_options.Bootstrap:
        pi0_all = pi0_all[0]

    return model_output, subj_train_all, subj_test_all
