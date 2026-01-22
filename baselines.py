import numpy as np
from collections import defaultdict
from sklearn.impute import SimpleImputer, KNNImputer

from tqdm import tqdm



def get_ccBaseline_data(train_nr_df, 
                        test_nr_df, test_gt_df, 
                        train_samps, test_samps):

    """
    process CC dfs to be ready for sklearn modules.
    Returns numpy arrays of shape (n_samples, n_features)
    """
    # may train on all or some samples of train data
    train_nr_data = train_nr_df[train_samps].T.to_numpy()
    test_nr_data = test_nr_df[test_samps].T.to_numpy()
    test_gt_data = test_gt_df[test_samps].T.to_numpy()
    
    return(train_nr_data, test_nr_data, test_gt_data)



def impute_constant(train_nr_data, test_nr_data, type_ = "zero"):

    """
    train_data: Simulated-missing WGBS data.
    test_data: Simulated missing WGBS or actual RRBS data.
    """

    if type_ == "zero":
        const_imputer = SimpleImputer(missing_values = np.nan, 
                                      strategy = "constant", 
                                      fill_value = 0, 
                                      keep_empty_features = True)
        
    elif type_ == "one":
        const_imputer = SimpleImputer(missing_values = np.nan, 
                                      strategy = "constant", 
                                      fill_value = 1, 
                                      keep_empty_features = True)
        
    elif type_ == "p5":
        const_imputer = SimpleImputer(missing_values = np.nan, 
                                      strategy = "constant", 
                                      fill_value = 0.5, 
                                      keep_empty_features = True)
        
    elif type_ == "mean":
        const_imputer = SimpleImputer(missing_values = np.nan, 
                                      strategy = "mean", 
                                      fill_value = 1, 
                                      keep_empty_features = True)
        
    
    const_imputer.fit(train_nr_data)
    preds_data = const_imputer.transform(test_nr_data) # numpy array

    return(preds_data)



def impute_knn(train_nr_data, test_nr_data, 
               k = 100, weight = "uniform"):

    """
    train_data: Simulated-missing WGBS data.
    test_data: Simulated missing WGBS or actual RRBS data.
    """

    knn_imputer = KNNImputer(n_neighbors = k, weights = weight, 
                             keep_empty_features = True)
    knn_imputer.fit(train_nr_data)
    preds_data = knn_imputer.transform(test_nr_data)

    return(preds_data)



def impute_patchWiseMeans(pcTrain_nr_map, train_samps,
                          pcTest_nr_map, test_samps,
                          samp_cpgMask_map = None, test_gt_df = None):
    
    """
    n_patches should remain the same between pcTrain and pcTest.
    """

    # "Fit" the model on the training data
    # print("Fitting patch-wise means model...")
    num_patches = len(pcTrain_nr_map[train_samps[0]])
    patch_data_map = defaultdict(list)

    for i in range(num_patches):
        for samp in train_samps:
            patch_id = pcTrain_nr_map[samp][i][0]
            patch_nr = pcTrain_nr_map[samp][i][1]
            patch_data_map[patch_id].extend(patch_nr) # accumulate data over all samples

    patch_means_map = {}
    for patch_id in patch_data_map.keys():

        patch_data = patch_data_map[patch_id]
        if np.sum(np.isnan(patch_data)) != len(patch_data):
            patch_means_map[patch_id] = np.nanmean(patch_data)
        else:
            patch_means_map[patch_id] = 0
    # print("Done!")

    # "Transform" the testing data + get_pcBasline_sampSeqMap functionality
    # separating functionality would require looping twice, this is faster
    # print("Transforming...")
    # samp_trueseq_map = {}
    samp_predseq_map = {}

    for samp in test_samps:
        # cpg_mask = samp_cpgMask_map[samp]

        # samp_trueseq = test_gt_df.loc[:, samp].values.copy()

        samp_patches = pcTest_nr_map[samp]
        samp_predseq = []

        for i in range(len(samp_patches)):
            patch_id = samp_patches[i][0]
            patch_nr = samp_patches[i][1]
            patch_mean = patch_means_map[patch_id]
            patch_data = np.nan_to_num(patch_nr, nan = patch_mean)
            samp_predseq.extend(patch_data)

        # samp_trueseq = samp_trueseq[~cpg_mask]
        # samp_trueseq = samp_trueseq
        # samp_trueseq_map[samp] = samp_trueseq
        
        # samp_predseq = np.array(samp_predseq)[~cpg_mask]
        samp_predseq = np.array(samp_predseq)
        samp_predseq_map[samp] = samp_predseq
    
    # print("Done!")

    # return(samp_trueseq_map, samp_predseq_map)
    return(samp_predseq_map)



# utility function
def get_ccBaseline_sampSeqMap(test_samps, test_gt_data, 
                              preds_data, samp_cpgMask_map):

    samp_trueseq_map = {}
    samp_predseq_map = {}

    for i, samp in enumerate(test_samps):
        
        # cpg_mask = samp_cpgMask_map[samp]
        # samp_trueseq_map[samp] = test_gt_data[i,:][~cpg_mask]
        # samp_predseq_map[samp] = preds_data[i,:][~cpg_mask]

        samp_trueseq_map[samp] = test_gt_data[i,:]
        samp_predseq_map[samp] = preds_data[i,:]

    return(samp_trueseq_map, samp_predseq_map)