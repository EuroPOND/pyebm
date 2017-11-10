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
import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import statsmodels.formula.api as sm
from matplotlib import rc


def prob_2_list_and_weights(p_yes):
    subjectwise_weights = []
    subjectwise_ordering = []
    for i in range(0, np.shape(p_yes)[0]):
        weights_reverse = np.sort(p_yes[i, :])
        valid_indices = np.logical_not(np.isnan(weights_reverse))
        weights_reverse = weights_reverse[valid_indices]
        subjectwise_weights.append(weights_reverse[::-1].tolist())
        ordering_reverse = np.argsort(p_yes[i, :])
        ordering_reverse = ordering_reverse[valid_indices].astype(int)
        subjectwise_ordering.append(ordering_reverse[::-1].tolist())

    return subjectwise_ordering, subjectwise_weights


def perminv(sigma):
    result = sigma[:]
    for i in range(len(sigma)):
        result[sigma[i]] = i
    return result


def adjswap(pi, i):
    pic = pi[:]
    (pic[i], pic[i + 1]) = (pic[i + 1], pic[i])
    return pic


def pd_read_data(str_data, flag_joint_fit=False, labels=['CN', 'MCI', 'AD']):

    if type(str_data) is str:
        data = pd.read_csv(str_data)
    elif len(str_data) == 0:
        data = []
        unique_subj_ids = []
        return data, unique_subj_ids
    else:
        data = str_data
    unique_subj_ids = pd.Series.unique(data['PTID'])
    labels_array = np.zeros(len(data['Diagnosis']), dtype=int)

    for i in range(len(labels)):
        idx_label = data['Diagnosis'] == labels[i]
        idx_label = np.where(idx_label.values)
        idx_label = idx_label[0]
        if i == 0:
            label_i = 1
        elif i == len(labels) - 1:
            label_i = 3
        else:
            label_i = 2
        labels_array[idx_label] = label_i
    idx_selectedsubjects = np.logical_not(labels_array == 0)
    labels_selected = labels_array[np.logical_not(labels_array == 0)]
    data = data[idx_selectedsubjects]
    data = data.drop('Diagnosis', axis=1)
    data = data.assign(Diagnosis=pd.Series(labels_selected, data.index))

    return data, unique_subj_ids


def correct_confounders(data_train, data_test, factors=None, flag_correct=1):
    if factors is None:
        factors = ['Age', 'Sex', 'ICV']
    flag_test = 1
    droplist = ['PTID', 'Diagnosis', 'EXAMDATE']
    if flag_correct == 0 or len(factors) == 0:
        data_train = data_train.drop(factors, axis=1)
        data_biomarkers = data_train.copy()
        h = list(data_biomarkers)
        for j in droplist:
            if any(j in f for f in h):
                data_biomarkers = data_biomarkers.drop(j, axis=1)
        biomarkers_list = list(data_biomarkers)
        if len(data_test) > 0:
            data_test = data_test.drop(factors, axis=1)
    else:
        # Change categorical value to numerical value
        if len(data_test) == 0:
            flag_test = 0
            data_test = data_train.copy()

        if any('Sex' in f for f in factors):
            count = -1
            for data in [data_train, data_test]:
                count = count + 1
                sex = np.zeros(len(data['Diagnosis']), dtype=int)
                idx_male = data['Sex'] == 'Male'
                idx_male = np.where(idx_male)
                idx_male = idx_male[0]
                idx_female = data['Sex'] == 'Female'
                idx_female = np.where(idx_female)
                idx_female = idx_female[0]
                sex[idx_male] = 1
                sex[idx_female] = 0
                data = data.drop('Sex', axis=1)
                data = data.assign(Sex=pd.Series(sex))
                if count == 0:
                    data_train = data.copy()
                else:
                    data_test = data.copy()

        # Separate the list of biomarkers from confounders and meta data
        count = -1
        for data in [data_train, data_test]:
            count = count + 1
            data_biomarkers = data
            data_biomarkers = data_biomarkers.drop(factors, axis=1)
            h = list(data_biomarkers)
            for j in droplist:
                if any(j in f for f in h):
                    data_biomarkers = data_biomarkers.drop(j, axis=1)
            biomarkers_list = list(data_biomarkers)
            biomarkers_listnew = []
            for i in range(len(biomarkers_list)):
                biomarkers_listnew.append(biomarkers_list[i].replace(' ', '_'))
                biomarkers_listnew[i] = biomarkers_listnew[i].replace('-', '_')

            for i in range(len(biomarkers_list)):
                data = data.rename(columns={biomarkers_list[i]: biomarkers_listnew[i]})
            # Contruct the formula for regression. Also compute the mean value of the confounding factors for correction
            if count == 0:  # Do it only for training set
                str_confounders = ''
                mean_confval = np.zeros(len(factors))
                for j in range(len(factors)):
                    str_confounders = str_confounders + '+' + factors[j]
                    mean_confval[j] = np.nanmean(data[factors[j]].values)
                str_confounders = str_confounders[1:]

                # Multiple linear regression
                betalist = []
                for i in range(len(biomarkers_list)):
                    str_formula = biomarkers_listnew[i] + '~' + str_confounders
                    result = sm.ols(formula=str_formula, data=data).fit()
                    betalist.append(result.params)

            # Correction for the confounding factors
            Deviation = (data[factors] - mean_confval)
            Deviation[np.isnan(Deviation)] = 0
            for i in range(len(biomarkers_list)):
                betai = betalist[i].values
                betai_slopes = betai[1:]
                correction_factor = np.dot(Deviation.values, betai_slopes)
                data[biomarkers_listnew[i]] = data[biomarkers_listnew[i]] - correction_factor
            data = data.drop(factors, axis=1)

            for i in range(len(biomarkers_list)):
                data = data.rename(columns={biomarkers_listnew[i]: biomarkers_list[i]})
            if count == 0:
                data_train = data.copy()
            else:
                data_test = data.copy()

    if flag_test == 0:
        data_test = []
    return data_train, data_test, biomarkers_list


def pd2mat(pd_data, biomarkers_list, flag_joint_fit):
    # Convert arrays from pandas dataframe format to the matrices (which are used in debm algorithms)
    num_events = len(biomarkers_list)
    if flag_joint_fit == 0:
        num_feats = 1
    num_subjects = pd_data.shape[0]
    mat_data = np.zeros((num_subjects, num_events, num_feats))
    for i in range(num_events):
        mat_data[:, i, 0] = pd_data[biomarkers_list[i]].values
    return mat_data


def exam_date_str2num(exam_date_series):
    timestamp = np.zeros(len(exam_date_series))
    for i in range(len(exam_date_series)):
        stre = exam_date_series.values[i]
        if len(stre) > 5:
            timestamp[i] = time.mktime(datetime.datetime.strptime(stre, "%Y-%m-%d").timetuple())
        else:
            timestamp[i] = np.nan
    timestamp_series = pd.Series(timestamp)
    return timestamp_series


def visualize_biomarker_distribution(data_all, params_all, biomarkers_list):

    m = np.shape(data_all)
    n = len(params_all)
    fig, ax = plt.subplots(int(round(1 + m[1] / 3)), 3, figsize=(13, 4 * (1 + m[1] / 3)))
    for i in range(m[1]):
        d_alli = data_all[:, i, 0]
        valid_data = np.logical_not(np.isnan(d_alli))
        d_allis = d_alli[valid_data].reshape(-1, 1)
        d_allis = d_allis[:, 0]
        x_grid = np.linspace(np.min(d_allis), np.max(d_allis), 1000)
        for j in range(n):
            i1 = int(round(i / 3))
            j1 = np.remainder(i, 3)
            paramsij = params_all[j][i, :]
            norm_pre = scipy.stats.norm(loc=paramsij[0], scale=paramsij[1])
            norm_post = scipy.stats.norm(loc=paramsij[2], scale=paramsij[3])
            h = np.histogram(d_allis, 50)
            maxh = np.nanmax(h[0])
            ax[i1, j1].hist(d_allis, 50, fc='blue', histtype='stepfilled', alpha=0.3, normed=False)
            # ylim=ax[i,j].get_ylim()
            likeli_pre = norm_pre.pdf(x_grid) * (paramsij[4])
            likeli_post = norm_post.pdf(x_grid) * (1 - paramsij[4])
            likeli_tot = likeli_pre + likeli_post
            likeli_tot_corres = np.zeros(len(h[1]) - 1)
            bin_size = h[1][1] - h[1][0]
            for k in range(len(h[1]) - 1):
                bin_loc = h[1][k] + bin_size
                idx = np.argmin(np.abs(x_grid - bin_loc))
                likeli_tot_corres[k] = likeli_tot[idx]

            max_scaling = maxh / np.max(likeli_tot)

            scaling_opt = 1
            opt_score = np.inf
            if max_scaling > 1:
                scale_range = np.arange(1, max_scaling + 1, max_scaling / 1000.)
            else:
                scale_range = np.arange(1, (1 / max_scaling) + 1, max_scaling / 1000.)
                scale_range = np.reciprocal(scale_range)

            for s in scale_range:
                l2norm = (likeli_tot_corres * s - h[0]) ** 2
                idx_nonzero = h[0] > 0
                l2norm = l2norm[idx_nonzero]
                score = np.sum(l2norm)
                if score < opt_score:
                    opt_score = score
                    scaling_opt = s
            likeli_pre = likeli_pre * scaling_opt
            likeli_post = likeli_post * scaling_opt
            likeli_tot = likeli_pre + likeli_post

            ax[i1, j1].plot(x_grid, likeli_pre, color='green', alpha=0.5, lw=3)
            ax[i1, j1].plot(x_grid, likeli_post, color='red', alpha=0.5, lw=3)
            ax[i1, j1].plot(x_grid, likeli_tot, color='black', alpha=0.5, lw=3)

            ax[i1, j1].set_title(biomarkers_list[i])
    for j in range(1, n):
        plt.setp([a.get_yticklabels() for a in fig.axes[j::n]], visible=False)
    plt.show()


def visualize_ordering(labels, pi0_all, pi0_mean, plotorder):
    columns = ['Features', 'Event Position', 'Count']
    datapivot = pd.DataFrame(columns=columns)
    for i in range(len(labels)):
        bb = [item.index(i) for item in pi0_all]
        for j in range(len(labels)):
            cc = pd.DataFrame([[bb.count(j), j, labels[i]]], index=[j], columns=['Count', 'Event Position', 'Features'])
            datapivot = datapivot.append(cc)
    datapivot = datapivot.pivot("Features", "Event Position", "Count")
    if plotorder:
        newindex = []
        for i in range(len(list(pi0_mean))):
            aa = labels[pi0_mean[i]]
            newindex.append(aa)
        datapivot = datapivot.reindex(newindex)
    xticks = np.arange(len(labels)) + 1
    datapivot = datapivot[datapivot.columns].astype(float)
    heatmap = sns.heatmap(datapivot, cmap='binary', xticklabels=xticks, vmin=0, vmax=len(pi0_all))
    fig = heatmap.get_figure()
    plt.title('Positional variance diagram of the central ordering')
    plt.yticks(rotation=0)
    plt.show()


def visualize_staging(subj_stages, diagnosis, labels):
    if np.max(subj_stages) > 1:
        nb = np.max(subj_stages) + 2
        freq, binc = np.histogram(subj_stages, bins=np.arange(np.max(subj_stages) + 1.01))
    else:
        nb = 50
        freq, binc = np.histogram(subj_stages, bins=nb)

    freq = (1. * freq) / len(subj_stages)
    maxfreq = np.max(freq)

    idx_cn = np.where(diagnosis == 1)
    idx_cn = idx_cn[0]
    idx_ad = np.where(diagnosis == len(labels))
    idx_ad = idx_ad[0]
    idx_mci = np.where(np.logical_and(diagnosis > 1, diagnosis < len(labels)))
    idx_mci = idx_mci[0]

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.style.use('seaborn-whitegrid')
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('mathtext', fontset='stixsans')
    c = ['#4daf4a', '#377eb8', '#e41a1c']
    count = -1
    freq_all = []
    for idx in [idx_cn, idx_mci, idx_ad]:
        count = count + 1
        freq, binc = np.histogram(subj_stages[idx], bins=binc)
        freq = (1. * freq) / len(subj_stages)
        if count > 0:
            freq = freq + freq_all[count - 1]
        freq_all.append(freq)
        bw = 1 / (2. * nb)
        ax.bar(binc[:-1], freq, width=bw, color=c[count], label=labels[count], zorder=3 - count)
    ax.set_xlim([-bw, bw + np.max([1, np.max(subj_stages)])])
    ax.set_ylim([0, maxfreq])
    if np.max(subj_stages) < 1:
        ax.set_xticks(np.arange(0, 1.05, 0.1))
        ax.set_xticklabels(np.arange(0, 1.05, 0.1), fontsize=14)
    else:
        ax.set_xticks(np.arange(0, np.max(subj_stages) + 0.05, 1))
        ax.set_xticklabels(np.arange(0, np.max(subj_stages) + 0.05, 1), fontsize=14)

    ax.set_yticks(np.arange(0, maxfreq, 0.1))
    ax.set_yticklabels(np.arange(0, maxfreq, 0.1), fontsize=14)
    ax.set_xlabel('Estimated Disease State', fontsize=16)
    ax.set_ylabel('Frequency of occurrences', fontsize=16)
    ax.legend(fontsize=16)
    plt.title('Patient Staging', fontsize=16)
    plt.show()
