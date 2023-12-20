import numpy as np
import pickle
import pandas as pd
import time
from datetime import timedelta

import matplotlib.pyplot as plt


from comboKR import ComboKR, normalized_c_from_c, c_from_normalized_c, braid_model_with_raw_c_input, tanimotok

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


"""
Example of how to use comboKR for drug surface prediction

Cell line 786-0 from the almanac data (first in alphabetical order)
The BRAID surfaces have been fitted with additional NCI60 monotherapy information.

The experiment is in the "new combo" scenario.

- !remember to unzip the data files! - 

@ Riikka Huusari, 2023
"""

# configure experiment

cell_id = "786-0"

names = np.load("data_for_demo/drug_names.npy")
feats = np.load("data_for_demo/drug_features.npy")
print(len(names))

# the results in the supplementary material can be recovered by selecting below "3" or "4"
# print(repr(names[0::3]))
drug_ids = names[0::3]  # oops the last is never considered but whatever, I won't change the setting after many exp

print("Using this many drugs in experiments:", len(drug_ids))


# =====================================================================================================================
# LOAD DATA

# saved fits for the cell line
with open("data_for_demo/braids_" + cell_id + ".df.pkl", "rb") as f:
    braid_params = pickle.load(f)
with open("data_for_demo/hills_" + cell_id + ".df.pkl", "rb") as f:
    hill_equations_dict = pickle.load(f)

# load the combo data
with open("data_for_demo/combo_data_for_demo.pickle", "rb") as f:
    combo_data_in_cell = pickle.load(f)
drug_drug_all = combo_data_in_cell[['NSC1', 'NSC2']].drop_duplicates()
drug_drug_all_np = drug_drug_all.to_numpy().astype(str)

# select subset of the data according to the experiment: only those combinations where either of the drugs
# is in the drug_ids list
subset_combo_data = combo_data_in_cell[combo_data_in_cell["NSC1"].isin(drug_ids)]
subset_combo_data = subset_combo_data[subset_combo_data["NSC2"].isin(drug_ids)]
# list the two drugs in the combinations
drug_drug_all_subset = subset_combo_data[['NSC1', 'NSC2']].drop_duplicates()
drug_drug_all_subset_np = drug_drug_all_subset.to_numpy().astype(str)
n = drug_drug_all_subset_np.shape[0]
print("n (not doubled):", n)

# concentrations and responses also in easier format
# here are the surface response measurements and drug1 and drug2 concentrations in the full data
# (these are needed for the test set elements as groundtruth and to know where to sample the predicted surfaces)
surface_measurements = np.load("data_for_demo/gt_surface_measurements.npy")
drug1_concentrations_in_measurements = np.load("data_for_demo/gt_concentrations1.npy")
drug2_concentrations_in_measurements = np.load("data_for_demo/gt_concentrations2.npy")

# select those that are used in the smaller experiment
inds_of_this_exp_in_full_data = []
for ii in range(drug_drug_all_subset_np.shape[0]):
    ind_in_full = np.where((drug_drug_all_np[:, 0] == drug_drug_all_subset_np[ii, 0]) &
                           (drug_drug_all_np[:, 1] == drug_drug_all_subset_np[ii, 1]))[0][0]
    inds_of_this_exp_in_full_data.append(ind_in_full)
inds_of_this_exp_in_full_data = np.array(inds_of_this_exp_in_full_data).astype(int)
surface_measurements = surface_measurements[inds_of_this_exp_in_full_data, :]
drug1_concentrations_in_measurements = drug1_concentrations_in_measurements[inds_of_this_exp_in_full_data, :]
drug2_concentrations_in_measurements = drug2_concentrations_in_measurements[inds_of_this_exp_in_full_data, :]


# =====================================================================================================================
# DIVIDE INTO TRAINING AND TESTING
# scenario: new combo
# simplified cv used proof-of-concept - not full

n_tr = int(0.6*n)
n_val = int(0.2*n)
n_tst = n-n_tr-n_val

print(n_tr, "&", n_val, "&", n_tst)

np.random.seed(0)
order = np.random.permutation(n)
tr_inds = order[:n_tr]
val_inds = order[n_tr:n_tr+n_val]
tst_inds = order[n_tr+n_val:]
tr = drug_drug_all_subset_np[tr_inds, 0:]
val = drug_drug_all_subset_np[val_inds, 0:]
tst = drug_drug_all_subset_np[tst_inds, 0:]

# double manually here the training data (i.e. include both drugA-drugB and drugB-drugA)
tr_doubled = np.vstack((tr, tr[:, [1, 0]]))

# get indices of the drug features
tr_doubled_drug_inds = np.zeros((tr_doubled.shape[0], 2))
val_drug_inds = np.zeros((n_val, 2))
tst_drug_inds = np.zeros((n_tst, 2))
for ii in range(len(drug_ids)):
    tr_doubled_drug_inds[tr_doubled==drug_ids[ii]] = ii
    val_drug_inds[val==drug_ids[ii]] = ii
    tst_drug_inds[tst==drug_ids[ii]] = ii
tr_doubled_drug_inds = tr_doubled_drug_inds.astype(int)
val_drug_inds = val_drug_inds.astype(int)
tst_drug_inds = tst_drug_inds.astype(int)

# build the input kernels for comboKRR
# k((x,z), (x',z')) = k(x,x')k(z,z')

base_kernel = tanimotok(feats, feats)
# kernel on firsts multiplied elementwise by kernel on seconds

K1 = base_kernel[np.ix_(tr_doubled_drug_inds[:, 0], tr_doubled_drug_inds[:, 0])]
K2 = base_kernel[np.ix_(tr_doubled_drug_inds[:, 1], tr_doubled_drug_inds[:, 1])]
Ktr = K1*K2

K1 = base_kernel[np.ix_(val_drug_inds[:, 0], tr_doubled_drug_inds[:, 0])]
K2 = base_kernel[np.ix_(val_drug_inds[:, 1], tr_doubled_drug_inds[:, 1])]
Kval = K1*K2

trval_doubled_drug_inds = np.vstack((tr_doubled_drug_inds,
                                     val_drug_inds,
                                     val_drug_inds[:, [1, 0]]))
K1 = base_kernel[np.ix_(trval_doubled_drug_inds[:, 0], trval_doubled_drug_inds[:, 0])]
K2 = base_kernel[np.ix_(trval_doubled_drug_inds[:, 1], trval_doubled_drug_inds[:, 1])]
Ktr_with_val = K1*K2


K1 = base_kernel[np.ix_(tst_drug_inds[:, 0], trval_doubled_drug_inds[:, 0])]
K2 = base_kernel[np.ix_(tst_drug_inds[:, 1], trval_doubled_drug_inds[:, 1])]
Ktst = K1*K2

print("tr, val, tst:", n_tr, n_val, n_tst)
print("kernel shapes:", Ktr.shape, Kval.shape, Ktst.shape)


# extract braids for tr data, and double them
# braid parameters were saved as a pandas dataframe
# use pandas merge to get the values at the training samples
braids_tr_single_pd = pd.merge(pd.DataFrame({'NSC1': tr[:, 0], 'NSC2': tr[:, 1]}), braid_params, on=["NSC1", "NSC2"])
braids_tr_single = braids_tr_single_pd[["E0", "E1", "E2", "E3", "h1", "h2", "C1", "C2", "kappa"]].to_numpy()
braids_tr_doubled = np.vstack((braids_tr_single, braids_tr_single[:, [0, 2, 1, 3, 5, 4, 7, 6, 8]]))

# braids on validation
braids_val_single_pd = pd.merge(pd.DataFrame({'NSC1': val[:, 0], 'NSC2': val[:, 1]}), braid_params, on=["NSC1", "NSC2"])
braids_val_single = braids_val_single_pd[["E0", "E1", "E2", "E3", "h1", "h2", "C1", "C2", "kappa"]].to_numpy()
braids_trval_doubled = np.vstack((braids_tr_single,
                                  braids_tr_single[:, [0, 2, 1, 3, 5, 4, 7, 6, 8]],
                                  braids_val_single,
                                  braids_val_single[:, [0, 2, 1, 3, 5, 4, 7, 6, 8]]))

# for piicm2 I need all braids in order of cell_drug_drug_all
braids_all_pd = pd.merge(pd.DataFrame({'NSC1': drug_drug_all_np[:, 0], 'NSC2': drug_drug_all_np[:, 1]}), braid_params, on=["NSC1", "NSC2"])
braids_all = braids_all_pd[["E0", "E1", "E2", "E3", "h1", "h2", "C1", "C2", "kappa"]].to_numpy()

# tr, val, tst raw data
# ground truth surface measurements
gt_val = surface_measurements[val_inds, :]
gt_tst = surface_measurements[tst_inds, :]
# concentrations
# tr_concentrations1_doubled = drug1_concentrations_in_measurements[tr_inds].append(drug2_concentrations_in_measurements[tr_inds])
# tr_concentrations2_doubled = drug2_concentrations_in_measurements[tr_inds].append(drug1_concentrations_in_measurements[tr_inds])
val_concentrations1 = drug1_concentrations_in_measurements[val_inds]
val_concentrations2 = drug2_concentrations_in_measurements[val_inds]
tst_concentrations1 = drug1_concentrations_in_measurements[tst_inds]
tst_concentrations2 = drug2_concentrations_in_measurements[tst_inds]



# =====================================================================================================================
# run the algorithm


def train_and_test_with_combokr(hill_equations_dict, dd_tr, dd_tst, Kx, Kx_t, c_t_d1, c_t_d2, braids_tr, lmbda, lmbda_n=None):

    """
    The function runs comboKR and comboKR n. training and testing with given data

    :param hill_equations_dict: dictionary of hill equations, to use for monotherapies in building candidates when solving the pre-image problem
    :param dd_tr: numpy array, each row gives the drug ids for the drugs in combinations in training set
    :param dd_tst: numpy array, each row gives the drug ids for the drugs in combinations in test set
    :param Kx: full kernel matrix on inputs
    :param Kx_t: kernel matrix on inputs, between test and training sets
    :param c_t_d1: concentrations of drug 1 on which the predictions should be evaluated
    :param c_t_d2: concentrations of drug 2 on which the predictions should be evaluated
    :param braids_tr: the fitted braid functions of the training set surfaces
    :param lmbda: regularisation parameter for KRR in comboKR
    :param lmbda_n: regularisation parameter for KRR in comboKRn (if not given, defaults to same as lambda)
    :return: numpy arrays containing vectorised predictions evaluated on the concentrations given
    """

    if lmbda_n is None:
        lmbda_n = lmbda
    # cv made easier

    algo = ComboKR(hill_equations_dict, normalised_output_kernel=False)
    algo_normalised = ComboKR(hill_equations_dict, normalised_output_kernel=True)

    print("   training original..")
    algo.train(dd_tr, Kx, lmbda, braids_tr)
    print("   training normalised..")
    algo_normalised.train(dd_tr, Kx, lmbda_n, braids_tr)

    print("   predicting original..")
    preds, pred_params = algo.predict(dd_tst, c_t_d1, c_t_d2, Kx_t)
    print("   predicting normalised..")
    preds_normalised, pred_params_normalised = algo_normalised.predict(dd_tst, c_t_d1, c_t_d2, Kx_t)

    return preds, preds_normalised


def cv_combokr_both():

    """
    Run cross-validation over both comboKR variants (original and normalised output kernel)
    :return: final predictions on test set, for both comboKR and comboKR n.
    """

    lmbdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    best_corr = -1
    print("Training comboKR")
    for lmbda in lmbdas:
        print(".. with ", lmbda)

        preds, preds_n = train_and_test_with_combokr(hill_equations_dict, tr_doubled, val, Ktr, Kval, val_concentrations1, val_concentrations2, braids_tr_doubled, lmbda)

        corr = np.corrcoef(preds.ravel(), gt_val.ravel())[0, 1]
        corr_n = np.corrcoef(preds_n.ravel(), gt_val.ravel())[0, 1]
        if lmbda == lmbdas[0]:
            best_lmbda = lmbda
            best_lmbda_n = lmbda
            best_corr = corr
            best_corr_n = corr_n
        else:
            if corr > best_corr:
                best_lmbda = lmbda
                best_corr = corr
            if corr_n > best_corr_n:
                best_lmbda_n = lmbda
                best_corr_n = corr

    print("best lmbdas were", best_lmbda, best_lmbda_n)
    # single tr and val
    t0=time.process_time()
    preds, preds_n = train_and_test_with_combokr(hill_equations_dict, np.vstack((tr_doubled, val, val[:, [1, 0]])), tst, Ktr_with_val, Ktst,
                                                 tst_concentrations1, tst_concentrations2,
                                                 braids_trval_doubled, best_lmbda, best_lmbda_n)
    print("comboKR (both) took: ", timedelta(seconds=time.process_time() - t0))
    return preds, preds_n


def scatter_res(preds, true, name=""):
    corr = np.corrcoef(preds.ravel(), true.ravel())[0, 1]
    plt.figure(figsize=(5,4))
    plt.scatter(true.ravel(), preds.ravel())
    plt.xlabel("GT")
    plt.ylabel("pred")
    plt.xlim([-10, 110])
    plt.ylim([-10, 110])
    plt.title(name+": %5.3f"%corr)
    plt.tight_layout()


np.save("small_example_res/tst_gt_"+str(n)+".npy", gt_tst)

print("\n\n\n=== Running comboKR ===")
preds_combokrr, preds_combokrr_n = cv_combokr_both()
scatter_res(preds_combokrr, gt_tst, "comboKR")
scatter_res(preds_combokrr_n, gt_tst, "comboKRn")
plt.show()

print("all done :) ")

