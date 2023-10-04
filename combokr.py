import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances

# the code relies on some functionalities of the synergy package: https://github.com/djwooten/synergy
# the relevant parts have copied and adapted from the source code here, and the imports have been commented out
# from synergy.combination import BRAID
# from synergy.single import Hill
# import synergy.utils

"""
## ***********************************************************************/
##    This file contains the code for comboKR, an approach for predicting 
##    drug combination surfaces. 
##
##     MIT License
##     Copyright (c) 2023 KEPACO
##
##     Permission is hereby granted, free of charge, to any person obtaining
##     a copy of this software and associated documentation files (the
##     "Software"), to deal in the Software without restriction, including
##     without limitation the rights to use, copy, modify, merge, publish,
##     distribute, sublicense, and/or sell copies of the Software, and to
##     permit persons to whom the Software is furnished to do so, subject
##     to the following conditions:
##
##     The above copyright notice and this permission notice shall be
##     included in all copies or substantial portions of the Software.
##
##     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
##     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
##     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
##     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
##     CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
##     TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
##     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
## 
## ***********************************************************************/

@ Riikka Huusari, 2023
"""

import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=DeprecationWarning)

class ComboKR:

    """
    Class implementing the comboKR approach for drug interaction surface prediction
    The output kernel can either be ordinary RBF kernel, or normalised w.r.t. the concentrations
    """

    def __init__(self, monotherapies, normalised_output_kernel=True):

        """

        :param monotherapies: Dictionary of the hill function values, in order [zero_resp, max_resp, hill_slope, EC50]
        :param normalised_output_kernel: True/False, if output kernel is normalised with monotherapy concentration info
        """

        self.monotherapies = monotherapies
        self.normalise_output_kernel = normalised_output_kernel
        self.normalised_concentration_grid = np.arange(0, 1, 0.1)
        self.normalised_concentration_grid = np.append(self.normalised_concentration_grid, 0.99)  # with 1 one gets error

    def train(self, drug_drug_tr, Kx, lmbda, tr_surfaces):

        """
        Training amounts to calculating an inverse of (K_x + lmbda*I)

        :param drug_drug_tr: Numpy array of size n*2  Note: n is already assumed to be "doubled" if needed, i.e. for
                             drugs X and Z, it would contain both [X, Z] and [Z, X] rows
        :param Kx: n*n kernel matrix
        :param lmbda: KRR regularisation parameter
        :param tr_surfaces: BRAID function parameters for all surfaces in training set, size n*9.
                            The parameters are assumed to be in order (zero_resp is assumed to be 100)
                            [zero_resp, max1_resp, max2_resp, max_joint_respo, hill1_slope, hill2_slope, hill1_EC50, hill2_EC50, kappa]
        :return:
        """

        # IOKR approach yields closed-form solution, for which the computationally
        # heavy part is the inverse

        self.dd_tr = drug_drug_tr
        self.tr_surfaces = tr_surfaces

        self.inv_mat = np.linalg.pinv(np.eye(Kx.shape[0]) * lmbda + Kx)

    def _create_candidates(self, drug1, drug2, C, style="braid"):
        """
        Candidates with BRAID surface model; other styles could also be implemented

        :param drug1: drug1, key in monotherapies
        :param drug2: drug2, key in monotherapies
        :param C: concentrations on which the created candidate surfaces are sampled
        :param style:
        :return: sampled candidate surfaces, BRAID parameters
        """

        # BRAID style assumes hills are available as monotherapies

        h1 = self.monotherapies[drug1]
        h2 = self.monotherapies[drug2]

        braid_params = [100, h1[1], h2[1], 0,
                        h1[2], h2[2],
                        h1[3], h2[3], 0]

        kappas = [-1.99, -1.5, -1, -0.5, -0.1, 0.01, 0.1, 0.5, 1, 2, 10, 25, 50]
        start = np.maximum(np.minimum(h1[1], h2[1]) - 50, 0)
        end = np.minimum(120, np.maximum(h1[1], h2[1]) + 10)
        Emaxs = np.arange(start, end, (end - start) / 4)

        candidate_evals = []
        all_param_combinations = []
        for kappa in kappas:
            braid_params[8] = kappa
            for Emax in Emaxs:
                braid_params[3] = Emax
                candidate_evals.append(braid_model_with_raw_c_input(C, *braid_params))
                all_param_combinations.append(np.copy(braid_params))
        candidate_evals = np.array(candidate_evals)
        candidate_evals = self.fix_Ytr(candidate_evals, matsize=int(np.sqrt(candidate_evals.shape[1])))
        return candidate_evals, all_param_combinations

    def predict(self, drug_drug_tst, drug1_concentrations, drug2_concentrations, Kx_t, additional_monotherapies=None):

        """

        :param drug_drug_tst: ids of the drugs in an array; n_tst * 2
        :param drug1_concentrations: concentrations on which drug1 is queried in the test combination (length d*d)
        :param drug2_concentrations: concentrations on which drug2 is queried in the test combination
        :param Kx_t: kernel on input data of size n_tr*n_tst
        :param additional_monotherapies: for new drug predictive scenario, here new monotherapy information can be given
        :return: n*(d^2) 2d array of predictions, n*9 array of BRAID parameters for predicted surfaces
        """

        if additional_monotherapies is not None:
            self.monotherapies.update(additional_monotherapies)

        z = np.dot(self.inv_mat, Kx_t.T)

        # get the training data surfaces at the normalised grid
        if self.normalise_output_kernel:
            C2n, C1n = np.meshgrid(self.normalised_concentration_grid, self.normalised_concentration_grid)
            c1n = mat_to_vec(C1n)
            c2n = mat_to_vec(C2n)

            Ytr = []
            for tr in range(self.dd_tr.shape[0]):
                drug1, drug2 = self.dd_tr[tr, :]

                h1_params = self.monotherapies[drug1]
                h2_params = self.monotherapies[drug2]

                # if output kernel should be normalised,
                c1n_raw = c_from_normalized_c(c1n, *h1_params)
                c2n_raw = c_from_normalized_c(c2n, *h2_params)
                # sampled_braid_surfaces_n.append(braid_model_with_raw_c_input([c1n_raw, c2n_raw], *paramarray))
                C = [c1n_raw, c2n_raw]
                # query the braid surfaces at training at these concentrations
                Ytr.append(braid_model_with_raw_c_input(C, *self.tr_surfaces[tr, :].ravel()))
            Ytr = np.array(Ytr)
            if np.any(np.isnan(Ytr)):
                Ytr = self.fix_Ytr(Ytr, matsize=len(self.normalised_concentration_grid))
            sigma = np.mean(pairwise_distances(Ytr))
            gamma = 1 / (2 * sigma ** 2)

        preds = []
        pred_params = []

        for ii in range(drug_drug_tst.shape[0]):

            drug1, drug2 = drug_drug_tst[ii, :]

            h1_params = self.monotherapies[drug1]
            h2_params = self.monotherapies[drug2]

            if self.normalise_output_kernel:
                # if output kernel should be normalised,
                c1n_raw = c_from_normalized_c(c1n, *h1_params)
                c2n_raw = c_from_normalized_c(c2n, *h2_params)
                # sampled_braid_surfaces_n.append(braid_model_with_raw_c_input([c1n_raw, c2n_raw], *paramarray))
                C = [c1n_raw, c2n_raw]
            else:
                C = [drug1_concentrations[ii, :], drug2_concentrations[ii, :]]

            candidate_evals, all_param_combinations = self._create_candidates(drug1, drug2, C)

            # query the braid surfaces at training at these concentrations
            # if normalised this has already been done outside of this loop
            if not self.normalise_output_kernel:
                Ytr = []
                for tt in range(self.dd_tr.shape[0]):
                    Ytr.append(braid_model_with_raw_c_input(C, *self.tr_surfaces[tt, :].ravel()))
                Ytr = np.array(Ytr)
                # what if there are nans? need to fill in Ytr
                if np.any(np.isnan(Ytr)):
                    Ytr = self.fix_Ytr(Ytr)
                # assert not np.any(np.isnan(Ytr))
                sigma = np.mean(pairwise_distances(Ytr))
                gamma = 1 / (2 * sigma ** 2)

            try:
                Ky_c_tr = rbf_kernel(candidate_evals, Ytr, gamma)
            except ValueError:
                # ok where were the nans
                nanrows = np.where(np.isnan(candidate_evals))[0]
                otherrows = np.array(
                    list(set(list(np.arange(candidate_evals.shape[0]))).difference(set(list(nanrows)))))
                if len(otherrows) == 0:
                    fixed_candidates = self.fix_Ytr(candidate_evals, matsize=int(np.sqrt(candidate_evals.shape[1])))
                    Ky_c_tr = rbf_kernel(fixed_candidates, Ytr, gamma)
                else:
                    otherrows = otherrows.astype(int)
                    candidate_evals = candidate_evals[otherrows, :]
                    Ky_c_tr = rbf_kernel(candidate_evals, Ytr, gamma)

            scores = np.dot(Ky_c_tr, z[:, ii])

            best_indx = np.argmax(scores)

            if self.normalise_output_kernel:
                Cforpred = [drug1_concentrations[ii, :], drug2_concentrations[ii, :]]

                preds.append(braid_model_with_raw_c_input(Cforpred, *all_param_combinations[best_indx]))
                pred_params.append(all_param_combinations[best_indx])
            else:
                preds.append(candidate_evals[best_indx, :])
                pred_params.append(all_param_combinations[best_indx])

        return np.array(preds), np.array(pred_params)

    def fix_Ytr(self, Ytr, matsize=4):

        """
        Ytr is assumed to be a matrix, where each row is vectorised sampled surface

        This script "fixes" the surfaces in case there are nan values
        The fix is replicating the previous values in row/column to replace the nan values

        :param Ytr: each row in here is vectorised surface matrix (each matrix should be d*d)
        :param matsize: d
        :return:
        """

        # ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
        # rows that are in unique rows here and in inds_in_tr
        [rows, cols] = np.where(np.isnan(Ytr))
        # print("rows and cols with nans before:\n", rows, cols)
        rows_to_fix = np.unique(rows)
        # print("need to fix", len(rows_to_fix), "rows! ")

        bad_braids3 = []

        for rr, rowindx in enumerate(np.unique(rows_to_fix)):

            vec = Ytr[rowindx, :]
            cols_here = np.where(np.isnan(vec))[0]
            # print("row:", rowindx)

            # print(vec)
            # print(cols_here)

            assert np.any(np.isnan(vec))

            # three possibilities: (1) all are nan, (2) only some are, consecutive, or (3) some are but they jump
            if len(cols_here) == matsize ** 2:
                # print("*")
                bad_braids3.append(rowindx)
            elif len(cols_here) == 1:
                if cols_here[0] % (matsize) == matsize-1:  # at the end of a column
                    if cols_here[0] == matsize-1:  # first column
                        vec[cols_here[0]] = vec[cols_here[0]-1]
                    else:
                        vec[cols_here[0]] = np.nanmean([vec[cols_here[0]-1], vec[cols_here[0]-matsize]])
                elif cols_here[0] >= matsize*(matsize-1):
                    # last in a row
                    if cols_here[0] == matsize*(matsize-1):
                        # first row
                        vec[cols_here[0]] = vec[cols_here[0]-matsize]
                    else:
                        vec[cols_here[0]] = np.nanmean([vec[cols_here[0]-matsize], vec[cols_here[0]-1]])
                else:
                    # it's in the middle... take mean of values around
                    indices_around = []
                    # not in first col
                    if cols_here[0] >=matsize:
                        indices_around.append(cols_here[0]-matsize)
                    if cols_here[0] <matsize*(matsize-1): # not in last col
                        indices_around.append(cols_here[0]+matsize)
                    if cols_here[0] % matsize !=0: # not on first row
                        indices_around.append(cols_here[0]-1)
                    if cols_here[0] % matsize !=(matsize-1): # not on last row
                        indices_around.append(cols_here[0]+1)
                    vec[cols_here[0]] = np.nanmean(vec[indices_around])
            elif (cols_here[-1] - cols_here[0]) == (len(cols_here) - 1):
                # print("-")
                # consecutive indexing; column
                """  example
                0 4 8  12
                1 5 9  13
                2 6 10 14
                3 7 11 15
                """
                # what if this is the first column?

                # take tha last available column and repeat
                last_col = vec[cols_here[0] - matsize:cols_here[0]]
                indx_for_col_to_fill = cols_here[0]
                while indx_for_col_to_fill < matsize ** 2:
                    try:
                        vec[indx_for_col_to_fill:indx_for_col_to_fill + matsize] = last_col
                    except:
                        print(vec)
                        print(cols_here)
                        print(indx_for_col_to_fill)
                        print(last_col)
                        raise
                    indx_for_col_to_fill += matsize
                Ytr[rowindx, :] = vec
            else:
                # print("¤")
                # jumping; rows

                update_rowindx = np.arange(matsize) * matsize
                prev_rowindx = np.copy(update_rowindx)
                done = False
                while np.all(prev_rowindx < matsize ** 2):
                    new_rowindx = prev_rowindx + 1
                    if new_rowindx[0] in cols_here:
                        # nan, assign as previous row
                        vec[new_rowindx] = vec[update_rowindx]
                    else:
                        # update row that should be used to fill in other rows
                        update_rowindx = new_rowindx
                    prev_rowindx = new_rowindx
                    if matsize ** 2 - 1 in new_rowindx:
                        break
                Ytr[rowindx, :] = vec

                # it can be both! so test still for consecutive once more here
                if np.any(np.isnan(vec)):
                    cols_here = np.where(np.isnan(vec))[0]
                    if (cols_here[-1] - cols_here[0]) == (len(cols_here) - 1):
                        # print("-")
                        # consecutive indexing; column
                        """  example
                        0 3 6
                        1 4 7
                        2 5 8
                        """
                        # take tha last available column and repeat
                        last_col = vec[cols_here[0] - matsize:cols_here[0]]
                        indx_for_col_to_fill = cols_here[0]
                        while indx_for_col_to_fill < matsize ** 2:
                            vec[indx_for_col_to_fill:indx_for_col_to_fill + matsize] = last_col
                            indx_for_col_to_fill += matsize
                        Ytr[rowindx, :] = vec

            # finally, just try to fix them one-by-one:
            for iter in range(2):
                for ii in range(len(cols_here)):
                    indices_around = []
                    # not in first col
                    if cols_here[ii] >= matsize:
                        indices_around.append(cols_here[ii] - matsize)
                    if cols_here[ii] < matsize * (matsize - 1):  # not in last col
                        indices_around.append(cols_here[ii] + matsize)
                    if cols_here[ii] % matsize != 0:  # not on first row
                        indices_around.append(cols_here[ii] - 1)
                    if cols_here[ii] % matsize != (matsize - 1):  # not on last row
                        indices_around.append(cols_here[ii] + 1)
                    vec[cols_here[ii]] = np.nanmean(vec[indices_around])
            if np.any(np.isnan(vec)):
                bad_braids3.append(rowindx)

        return Ytr


# ----------------------------------------------------------
# helper functions

def vec_to_mat(m, shape=(4, 4)):
    return np.reshape(m, shape, order="F")


def mat_to_vec(M):
    return np.reshape(M, (-1, 1), order="F").squeeze()


def hill_equation(c_query, zero_resp, max_resp, hill_slope, halfway_c):

    dh = np.power(c_query,hill_slope)
    return zero_resp + (max_resp-zero_resp)*dh/(np.power(halfway_c,hill_slope)+dh)


def normalized_c_from_c(c_query, zero_resp, max_resp, hill_slope, halfway_c):
    transfer = lambda resp: (resp - zero_resp)/(max_resp-zero_resp)  # response into concentration
    from_c_to_transfer = lambda c: transfer(hill_equation(c, zero_resp, max_resp, hill_slope, halfway_c))  # response
    res = from_c_to_transfer(c_query)
    # case bigger than max or smaller than zero? not going to happen because this is from model, not measurement.
    # res[res < 0] = 0
    # res[res > 1] = 1
    return res


def c_from_normalized_c(c_query, zero_resp, max_resp, hill_slope, halfway_c):
    inverse_transfer = lambda resp: resp*(max_resp-zero_resp) + zero_resp

    tmp_query = np.copy(c_query)
    tmp_query[tmp_query==1] = 0.99

    def inverse_hill(resp_query):
        E_ratio = (resp_query-zero_resp)/(max_resp-resp_query)
        d = np.float_power(E_ratio, 1./hill_slope)*halfway_c
        return d

    return inverse_hill(inverse_transfer(tmp_query))


def tanimotok(X, Z):
    intersections = np.dot(X, Z.T)
    rowcounts_x = np.sum(X, axis=1)
    rowcounts_z = np.sum(Z, axis=1)
    unions = np.add.outer(rowcounts_x, rowcounts_z) - intersections
    return intersections/unions


def braid_model_with_raw_c_input(d, E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta=1, print_params=False):
    """
    Copied and adapted from https://github.com/djwooten/synergy/blob/master/src/synergy/combination/braid.py
    Mainly assumes that response at 0 concentration of drug is 100

    From the braidrm R package (https://rdrr.io/cran/braidrm/man/evalBRAIDrsm.html)
    The parameters of this equation must satisfy h1>0, h2>0, delta>0, kappa>-2, sign(E3-E0)=sign(E1-E0)=sign(E2-E0), |E3-E0|>=|E1-E0|, and |E3-E0|>=|E2-E0|.

    """

    d1 = d[0]
    d2 = d[1]

    if print_params:
        print("%7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e"
              % (E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta))

    delta_Es = [E1 - E0, E2 - E0, E3 - E0]
    # delta_Es = [E1 - E0, E2 - E0]  # , E3 - E0]   <- to make combination not dependent on combination info
    delta_Es = -np.abs(delta_Es)  # band-aid..
    max_delta_E_index = np.argmax(np.abs(delta_Es))
    max_delta_E = delta_Es[max_delta_E_index]

    deltaA_div_maxdelta = delta_Es[0] / max_delta_E
    deltaB_div_maxdelta = delta_Es[1] / max_delta_E

    h = np.sqrt(h1 * h2)
    power = 1 / (delta * h)

    pow1 = np.power(d1 / C1, h1)
    D1 = deltaA_div_maxdelta * pow1 / (
            1 + (1 - deltaA_div_maxdelta) * pow1)

    pow2 = np.power(d2 / C2, h2)
    D2 = deltaB_div_maxdelta * pow2 / (
            1 + (1 - deltaB_div_maxdelta) * pow2)

    pow11 = np.power(D1, power)
    pow22 = np.power(D2, power)
    D = pow11 + pow22 + kappa * np.sqrt(pow11 * pow22)

    Dpow = np.power(D, -delta * h)
    # # Dpow = np.sign(D)*np.power(np.abs(D), -delta*h)
    # print(Dpow, "\n", E0 + max_delta_E / (1 + Dpow))
    # print(-delta*h)
    res = E0 + max_delta_E / (1 + Dpow)
    # TypeError: 'numpy.float64' object does not support item assignment
    try:
        res[D == 0] = 100
    except TypeError:
        if D == 0:
            res = 100
    return res

