import os
import numpy as np
import pandas as pd
from collections import defaultdict
from aruna.process_dataset import get_cc_gt


### ----- INFERENCE HELPERS ----- ###
def get_cpgmask(chrom_list, dataset):

    # for subsetting baselines to not include cpg_mask (missing in WGBS)
    cpgMask_map = {}
    for chrom in chrom_list:
        samp_cpgMask_map = {}
        cc_gt_df, _ = get_cc_gt(dataset, chrom)
        for samp in cc_gt_df.columns:
            cpg_mask = np.isnan(cc_gt_df.loc[:,samp].values) # wgbs missing = True
            samp_cpgMask_map[samp] = cpg_mask
        cpgMask_map[chrom] = samp_cpgMask_map

    return(cpgMask_map)



def get_spp_collapsed_maps(spp_map, chr):

    """
    Collect all of #spp predictions (specified repeats) per sample and return mean chromosome methylome.

    Arguments
    --------
    spp_map (dict): stores #spp methylomes per sample. 
                    Struct -> {chr: {samp_repeat1: [<#cpg in chrom>], ..., samp_repeatSPP: []}}.
    chr (str): chromosome id in form chrID.
    """

    temp_dict = defaultdict(list)

    for k in spp_map[chr].keys():
        samp_name = k.split("_")[0]
        temp_dict[samp_name].append(spp_map[chr][k])

    final_map = {}
    for samp in temp_dict.keys():
        final_map[samp] = np.mean(np.stack(temp_dict[samp], axis = 0), axis = 0)

    return(final_map)



def save_aruna_preds(preds_map, *, canonical_index, 
                     out_dir, test_regimes, folds, split_type, 
                     chrom, verbose = True):
    """
    Save Aruna prediction maps to CSV with canonical CpG index.

    Parameters
    ----------
    preds_map : dict
        Nested dict: preds_map[regime][fold] -> data convertible to DataFrame
    canonical_index : list-like
        Row-wise CpG identifiers (must align with DF rows)
    out_dir : str
        Output directory
    test_regimes : iterable of str
        e.g. ["rrbs_sim"]
    folds : iterable of str
        e.g. ["0", "1", "2"]
    split_type : str
        e.g. "donor_holdout"
    chrom : str
        e.g. "chr21"
    verbose : bool
        Print saved paths if True
    """

    os.makedirs(out_dir, exist_ok=True)

    for regime in test_regimes:
        if regime not in preds_map:
            raise KeyError(f"Regime '{regime}' not found in preds_map")

        for fold in folds:
            if fold not in preds_map[regime]:
                raise KeyError(f"Fold '{fold}' not found under regime '{regime}'")

            df = pd.DataFrame(preds_map[regime][fold])

            if len(df) != len(canonical_index):
                raise ValueError(
                    f"[{regime} fold {fold}] "
                    f"Length mismatch: df={len(df)} canonical_index={len(canonical_index)}"
                )

            # set CpG index directly
            df.index = pd.Index(canonical_index, name="start")

            fname = f"{regime}_{split_type}_fold{fold}_{chrom}.csv"
            fpath = os.path.join(out_dir, fname)

            df.to_csv(fpath, index=True)

            if verbose:
                print(f"Saved: {fpath}")
