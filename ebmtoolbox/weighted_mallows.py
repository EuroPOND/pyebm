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

import copy

import numpy

from ebmtoolbox import util


class WeightedMallows(object):
    def __init__(self, phi, sigma0):
        self.phi = phi[:]
        self.sigma0 = sigma0[:]

    @staticmethod
    def fit_mallows(p_yes, params_opt):
        p_yes_padded = numpy.concatenate((numpy.zeros((p_yes.shape[0], 1)), p_yes, numpy.ones((p_yes.shape[0], 1))),
                                         axis=1)
        so_list, weights_list = util.prob_2_list_and_weights(p_yes_padded)
        num_events = numpy.shape(params_opt)[0]
        pi0_init = 1 + numpy.argsort(params_opt[:, 4, 0])
        pi0_init = numpy.insert(pi0_init, 0, num_events + 1)
        pi0_init = numpy.append(pi0_init, 0)
        pi0, bestscore, scores = WeightedMallows.consensus(num_events + 2, so_list, weights_list, pi0_init)
        ordering_distances = scores - bestscore
        event_centers = numpy.cumsum(ordering_distances)
        event_centers = event_centers / numpy.max(event_centers)
        idx0 = pi0.index(0)
        del pi0[idx0]
        idx_last = pi0.index(num_events + 1)
        del pi0[idx_last]
        pi0[:] = [int(x - 1) for x in pi0]
        event_centers = event_centers[:-1]
        return pi0, event_centers

    @staticmethod
    def unweightedkendall(pi, pi0):
        return sum(WeightedMallows.__v(pi, pi0))
        # here D is assumed to be in inverse form (vertical bar form)

    # and represented as a dictionary indexed by tuples
    @staticmethod
    def consensus(n, d, prob, pi0_init):
        max_seq_evals = 10000
        if len(pi0_init) == 0:
            q, h, r = WeightedMallows.compute_q(n, d, prob)
            qns = numpy.sum(q, axis=0)
            a_qns = numpy.argsort(qns)
            sig0 = list(a_qns)
        else:
            sig0 = list(pi0_init)
        bestscore = WeightedMallows.totalconsensus(sig0, d, prob)
        sig_list = (max_seq_evals + n - 1) * [0]
        count = 0
        sig_list[count] = sig0
        while True:
            scores = (n - 1) * [0]
            for i in range(n - 1):
                count = count + 1
                sig = util.adjswap(sig0, i)
                sig_list[count] = sig
                scores[i] = WeightedMallows.totalconsensus(sig, d, prob)
            bestidx = scores.index(min(scores))
            bestsig = util.adjswap(sig0, bestidx)
            if bestscore > scores[bestidx]:
                sig0 = bestsig[:]
                bestscore = scores[bestidx]
            if bestscore <= scores[bestidx] or count >= max_seq_evals:
                break
        return sig0, bestscore, scores

    @staticmethod
    def compute_q(n, d, prob):
        q = numpy.zeros((n, n))
        r = numpy.zeros((n, n))
        h = numpy.zeros((1, n))
        for i in range(0, len(d)):
            x = d[i]
            for j in range(len(x)):
                h[0, x[j]] = h[0, x[j]] + 1
                for l in range(j + 1, len(x)):
                    q[x[j], x[l]] += prob[i][j] - prob[i][l]
                    r[x[j], x[l]] += 1
        for j in range(n):
            for l in range(n):
                q[j, l] = q[j, l] / (h[0, j])
        return q, h, r

    @staticmethod
    def totalconsensus(pi0, d, prob):
        score = (len(d)) * [0]
        for i in range(0, len(d)):
            s = copy.copy(d[i])
            p = copy.copy(prob[i])
            pi0c = copy.copy(pi0)
            pi0c, p_new = WeightedMallows.__remove_absent_events(pi0c, s, p)  # for NaN events and non-events
            score[i] = WeightedMallows.__kendall(pi0c, s, p)
            # score[i]=weighted_mallows.__kendall_alt(pi0c,s,p)

        tscore = numpy.mean(score)
        return tscore

    @staticmethod
    def __remove_absent_events(pi0c, seq, p):
        pi0c_new = []
        p_new = p
        # p_new = [x if x>0.5 else 0 for x in p]
        for j in range(len(pi0c)):
            e = pi0c[j]
            if WeightedMallows.__find(seq, e) != -1:
                pi0c_new.append(e)
        return pi0c_new, p_new

    # pi0 and pi1 are assumed to be written in inverse notation
    @staticmethod
    def __kendall(ordering1, ordering2, p):
        n = len(ordering1)
        weighted_distance = numpy.array((n - 1) * [0.0])
        for i in range(0, n - 1):
            e1 = ordering1[i]
            idx_e2 = WeightedMallows.__find(ordering2, e1)
            if idx_e2 > i:
                ordering2.insert(i, ordering2[idx_e2])
                ordering2 = ordering2[:idx_e2 + 1] + ordering2[idx_e2 + 2:]
                pn = numpy.asarray(p)
                # wd=sum(pn[i]-pn[i+1:idx_e2+1])
                dp = pn[i:idx_e2] - pn[idx_e2]
                wd = sum(dp)
                weighted_distance[i] = wd
                p.insert(i, p[idx_e2])
                p = p[:idx_e2 + 1] + p[idx_e2 + 2:]
        # denom = n*(n-1)/2
        return sum(weighted_distance)

    @staticmethod
    def __kendall_alt(pi0, pi, prob):
        v = (len(pi) - 1) * [0]
        piidx = util.perminv(list(pi))
        pi0idx = util.perminv(list(pi0))
        for i in range(0, len(pi) - 1):
            for j in range(i + 1, len(pi)):
                m = (piidx[i] - piidx[j]) * (pi0idx[i] - pi0idx[j])
                if m < 0:
                    v[i] = v[i] + abs(prob[piidx[i]] - prob[piidx[j]])
        return sum(v)

    @staticmethod
    def __kendall_alt_ind(pi0, pi, prob):
        v = (len(pi) - 1) * [0]
        piidx = util.perminv(list(pi))
        pi0idx = util.perminv(list(pi0))
        for i in range(0, len(pi) - 1):
            for j in range(i + 1, len(pi)):
                m = (piidx[i] - piidx[j]) * (pi0idx[i] - pi0idx[j])
                if m < 0:
                    v[i] = v[i] + abs(prob[piidx[i]] - prob[piidx[j]])
        return v

    @staticmethod
    def __find(pi, val):
        n = len(pi)
        for i in range(0, n):
            if pi[i] == val:
                return i
        return -1

    # pis here is assumed to be written in inverse notation
    # this is the insertion code (the interleaving at each stage of the mallows model)
    @staticmethod
    def __v(pi, pi0):
        v = (len(pi) - 1) * [0]
        piidx = util.perminv(list(pi))
        for j in range(len(pi) - 1):
            v[j] = [piidx[item] < piidx[pi0[j]] for item in pi0[(j + 1):]].count(True)
        return v
