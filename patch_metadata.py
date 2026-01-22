import os
import glob
import math
import pickle
import pathlib
import numpy as np
import pandas as pd

from read_utils import read_chromCentric_dataset

import multiprocessing
from tqdm import tqdm

import warnings



# NOTE: Importing this script makes available reference genome metadata
# NOTE: CpG start position metadata is 0-indexed, chrom lengths metadata is 1-indexed. Adjustments have been made.
# NOTE: Final data uses 0-indexing.
# Global namespace
CWD = "/home/js228/patch_recurrence"
patch_metadata_dir = os.path.join(CWD, "metadata")

# All these paths are subject to change
scratch_dir = "/srv/risotto.cs.rice.edu/scratch/js228/"

# everything reference genome associated
hg38_metadata_dir = os.path.join(CWD, "metadata/hg38_ref")
hg38_chrlen_path = os.path.join(hg38_metadata_dir, "hg38_chr_lengths.bedGraph")
# reference genome chrom lengths and CpG 0-indexed posns
hg38_chrlen_df = pd.read_csv(hg38_chrlen_path, header = None, sep = "\t", names = ["seqname", "start", "end"])
hg38_allcpg_df = pd.read_csv(os.path.join(hg38_metadata_dir, "hg38_cpg_py0idx.csv")) # 0-indexed CpGs
# hg38_metadata_dir = os.path.join(scratch_dir, "reference_genome/hg38_metadata") # equivalent scratch dir


### Patch-wise Position Metadata Creation [from reference genome] ###
def get_patchPosn_metadata(chrom, s):
    
    """
    Needs reference genome info only.
    Independent of dataset and samp_list.

    OUTPUT: Three files saved specific to Span and Chrom:
    1. patch_refCpG: Dict
    2. patch_genomicSpan: Dict
    3. patch_numCpG: DataFrame
    """

    curr_savedir = os.path.join(patch_metadata_dir, "span"+str(s), chrom)
    pathlib.Path(curr_savedir).mkdir(parents=True, exist_ok=True)

    print("Looking for existing Patch positional metadata for {}-span{}bp".format(chrom, s))
    refCpG_file = os.path.join(curr_savedir, "patch_refCpG_span"+str(s)+".pkl")
    genomicSpan_file = os.path.join(curr_savedir, "patch_genomicSpan"+str(s)+".pkl")

    if os.path.isfile(refCpG_file) and os.path.isfile(genomicSpan_file):
        print("Patch positions metadata found!\nRef Cpg at: {}\nGenomic Spans at: {}".format(refCpG_file, 
                                                                                            genomicSpan_file))
        with open(refCpG_file, "rb") as f:
            patch_refCpG_map = pickle.load(f)
        with open(genomicSpan_file, "rb") as f:
            patch_genomicSpan_map = pickle.load(f)

    else:
        print("Positional metadata not found, creating...")
        patch_refCpG_map, patch_genomicSpan_map = compute_patchPosn(chrom, s)

        print("Saving refCpG_map for {}-span{} to: {}".format(chrom, s, refCpG_file))
        with open(refCpG_file, 'wb') as f:
            pickle.dump(patch_refCpG_map, f)
        print("Done!")

        print("Saving genomicSpan_map for {}-span{} to: {}".format(chrom, s, genomicSpan_file))
        with open(genomicSpan_file, 'wb') as f:
            pickle.dump(patch_genomicSpan_map, f)
        print("Done!")

    print("Creating additional metadata (refNumCpG_map)...")
    patch_numCpG_map = {}
    for p in patch_refCpG_map.keys():
        patch_numCpG_map[p] = patch_refCpG_map[p].shape[0]
    patch_refNumCpG_df = pd.DataFrame({"patch_id": patch_numCpG_map.keys(), "cpg_count": patch_numCpG_map.values()})
    patch_refNumCpG_df.set_index("patch_id", inplace = True)

    print("All patch positional metadata created! Num Patches: {}".format(len(patch_refCpG_map)))
    
    return (patch_refCpG_map, patch_genomicSpan_map, patch_refNumCpG_df)



def compute_patchPosn(chrom, s):

    # collect required metadata information
    # get chrom start and end from chrom length metadata
    chrom_start = hg38_chrlen_df[hg38_chrlen_df.loc[:,"seqname"] == chrom].start.values[0]
    chrom_end = hg38_chrlen_df[hg38_chrlen_df.loc[:,"seqname"] == chrom].end.values[0]
    # get 0-indexed starts for all chrom CpGs
    chrom_allcpg_df = hg38_allcpg_df[hg38_allcpg_df.loc[:, "seqname"] == chrom]
    chrom_allcpg_df.set_index("start", inplace = True)

    print(chrom)
    print("Chromosome Range: ", chrom_start, chrom_end)
    print("Number of CpGs: ", chrom_allcpg_df.shape[0])

    # Fast implementation: 700x speedup
    # Various +/- 1s are to account for discrepancy between 1-indexed chr lengths file and 0-based bedGraphs and;
    # Pandas' inclusive indexing at both ends for index col (See: https://pandas.pydata.org/docs/user_guide/indexing.html)
    # For a given chromosome, collect reference CpG set per patch
    chrom_patch_refCpG_map = {}
    chrom_patch_genomicSpan_map = {}

    chrom_num_patches = math.ceil((chrom_end-chrom_start)/s)
    print("For Span {}bp, creating {} Reference Patches for {}...".format(s, 
                                                                          chrom_num_patches, 
                                                                          chrom))

    with tqdm(total = chrom_num_patches) as pbar: 
        for p in range(chrom_start,chrom_end): # p-th Patch, chr_start is 1 (1-based)
        
            # current patch spans from [lower, upper] <- closed interval
            lower = (p-1)*s
            upper = (p*s)-1

            if upper>=chrom_end:
                break
            else:
                # in pandas, endpoints are both inclusive for index col, "start" is 0-indexed
                patch_cpg_df = chrom_allcpg_df.loc[lower:upper] 
                chrom_patch_refCpG_map[p] = patch_cpg_df # allows for empty
                chrom_patch_genomicSpan_map[p] = (lower,upper)
                last_valid_upper = upper
                pbar.update(1)

        if last_valid_upper<chrom_end-1:
            patch_cpg_df = chrom_allcpg_df.loc[last_valid_upper+1:chrom_end-1] # +1 to avoid overlap with penultimate patch, -1 to account for 1-indexed chr length file
            chrom_patch_refCpG_map[p] = patch_cpg_df
            chrom_patch_genomicSpan_map[p] = (last_valid_upper+1, chrom_end-1)
            pbar.update(1)

    print("Reference Patch creation complete!")
    return(chrom_patch_refCpG_map, chrom_patch_genomicSpan_map)



### Patch-wise Statistics Computation [from provided sample pool] ###
def get_patchStats_metadata(dataset, s, chrom, samp_list, 
                            num_workers, overwrite = False):

    """
    Needs reference genome metadata (precomputed) and sample list.
    Stats computed are w.r.t. provided sample pool.
    Sample pool should be used for downstream training and not for Testing.

    OUTPUT: Two files saved specific to Span and Chrom:
    1. patch_avgBetas
    2. patch_avgCov
    """

    if num_workers > 16:
        warnings.warn("Using more than 16 cores! Cores used: ", num_workers)
    
    curr_savedir = os.path.join(patch_metadata_dir, dataset, "span"+str(s), chrom)

    print("Looking for existing Patch positional metadata for {}-span{}".format(chrom, s))
    avgBetas_file = os.path.join(curr_savedir, "patch_avgBetas_span"+str(s)+".csv")
    avgCov_file = os.path.join(curr_savedir, "patch_avgCov_span"+str(s)+".csv")

    if os.path.isfile(avgBetas_file) and os.path.isfile(avgCov_file) and not overwrite:
        print("Patch statistics metadata found!\nAvg. Betas at: {}\nAvg. Coverage at: {}".format(avgBetas_file, 
                                                                                               avgCov_file))
        patch_avgBetas_df = pd.read_csv(avgBetas_file, header = 0, index_col = 0)
        patch_avgCov_df = pd.read_csv(avgCov_file, header = 0, index_col = 0)

    else:
        print("Patch statistics metadata not found (or overwrite requested), creating...")
        print("Computing Prior Statistics using {} samples...".format(len(samp_list)))

        patch_refCpG_map, _, _ = get_patchPosn_metadata(chrom, s)
        fm_data, cov_data = read_chromCentric_dataset(CWD, dataset, chrom)
        
        # multiprocess over sample data
        pool = multiprocessing.Pool(processes = num_workers) 
        # Create a tqdm progress bar
        with tqdm(total=len(samp_list)) as pbar:
            def update(*a):
                pbar.update()
            # Apply compute_stats asynchronously to each filepath
            results = []
            for samp in samp_list:
                result = pool.apply_async(compute_patch_stats, args=(samp, fm_data, cov_data, patch_refCpG_map), callback = update)
                results.append(result)
            final_results = [result.get() for result in results]
        pool.close()
        pool.join()

        samp_names = [final_results[i][0] for i in range(len(final_results))]
        colnames = ["patch_id"] + samp_names

        # initialize dataframes
        patch_avgBetas_df = pd.DataFrame(columns = colnames)
        patch_avgCov_df = pd.DataFrame(columns = colnames)

        # populate dataframes
        for i in range(len(final_results)):
            curr_sampname = final_results[i][0]
            if i == 0:
                # this should be the same for betas and cov as well across samples
                all_patch_ids = list(final_results[i][1].keys())
                patch_avgBetas_df.loc[:,"patch_id"] = all_patch_ids
                patch_avgCov_df.loc[:,"patch_id"] = all_patch_ids

            patch_avgBetas_df.loc[:, curr_sampname] = list(final_results[i][1].values())
            patch_avgCov_df.loc[:, curr_sampname] = list(final_results[i][2].values())

        patch_avgBetas_df.set_index("patch_id", inplace = True)
        patch_avgCov_df.set_index("patch_id", inplace = True)

        # add a column containing patch-wise mean computed for dataset
        patch_avgBetas_df["dataset_avg"] = patch_avgBetas_df.mean(axis = 1)
        patch_avgCov_df["dataset_avg"] = patch_avgCov_df.mean(axis = 1)

        # save to disk
        pathlib.Path(curr_savedir).mkdir(parents=True, exist_ok=True)

        print("Saving avgBetas_df for {}-span{}, computed from {} samples to: {}".format(chrom, s, len(samp_list), avgBetas_file))
        
        patch_avgBetas_df.to_csv(avgBetas_file, header = True, index = True)
        print("Saving avgCov_df for {}-span{}, computed from {} samples to: {}".format(chrom, s, len(samp_list), avgCov_file))
        patch_avgCov_df.to_csv(avgCov_file, header = True, index = True)

                      
        print("Patch-wise avg. Betas and avg. Coverages are now available for each sample!")
    
    return(patch_avgBetas_df, patch_avgCov_df)



def compute_patch_stats(samp_name, fm_data, cov_data, patch_refCpG_map): 
    
    """ 
    For multiprocessing, return each sample's output and collate later.
    Each sample corresponds to 2 dicts of size chr_num_patches.
    """

    patch_avgBetas_map = {}
    patch_avgCov_map = {}

    for p in patch_refCpG_map.keys(): # p-th Patch

        curr_patch = patch_refCpG_map[p]

        if curr_patch.shape[0]==0:
            patch_avgBetas_map[p] = 0
            patch_avgCov_map[p] = 0

        else:

            samp_fm = fm_data.loc[:, samp_name].to_frame()
            samp_cov = cov_data.loc[:, samp_name].to_frame()

            # right join to keep num_CpG within patches constant over all samples (use reference patch-cpg sets)
            samp_patch_fm = samp_fm.merge(curr_patch, how = "right", on = "start").fillna(0)
            samp_patch_fm = samp_patch_fm.drop(columns = "seqname")

            samp_patch_cov = samp_cov.merge(curr_patch, how = "right", on = "start").fillna(0)
            samp_patch_cov = samp_patch_cov.drop(columns = "seqname")

            # Patch-wise means for current sample's FM and COV data
            patch_avgBetas_map[p] = np.mean(samp_patch_fm.iloc[:,0].values)
            patch_avgCov_map[p] = np.mean(samp_patch_cov.iloc[:,0].values)


    return(samp_name, patch_avgBetas_map, patch_avgCov_map)



def get_patch_noncpgMask(s, chrom):
    
    curr_savedir = os.path.join(patch_metadata_dir, "span"+str(s), chrom)
    pathlib.Path(curr_savedir).mkdir(parents=True, exist_ok=True)

    print("Looking for existing Patch Non-CpG masks for {}-span{}bp".format(chrom, s))
    noncpgMask_file = os.path.join(curr_savedir, "patch_noncpgMask_span"+str(s)+".pkl")

    if os.path.isfile(noncpgMask_file):
        print("Non-CpG masks found at: {}".format(noncpgMask_file))
        with open(noncpgMask_file, "rb") as f:
            patch_noncpgMask_map = pickle.load(f)
    else:
        print("Non-CpG masks not found! Creating...")
        patch_refCpG_map = get_patchPosn_metadata(chrom, s)[0]
        patch_noncpgMask_map = make_patch_noncpgMask(s, chrom, patch_refCpG_map)
        
        print("Saving noncpgMask_map for {}-span{}bp to: {}".format(chrom, s, noncpgMask_file))
        with open(noncpgMask_file, "wb") as f:
            pickle.dump(patch_noncpgMask_map, f)
        print("Done!")

    return(patch_noncpgMask_map)
                    


def make_patch_noncpgMask(s, chrom, patch_refCpG_map):

    """
    Non-CpG mask is a function of chromosome, s and patch id.
    Needs computation and storing only once.
    Stored as patch-chrom metadata just like patch position metadata.
    """
    print("Computing Non-CpG masks for all patches for {}-{}bp".format(chrom, s))

    patch_noncpgMask_map = {}
    for p in patch_refCpG_map.keys():

        cpg_posns = np.array(patch_refCpG_map[p].index)
        p_lower = (p-1)*s
        p_cpg_idx = cpg_posns - p_lower

        seq_vec = np.zeros(s)
        seq_vec[p_cpg_idx] = 1 # mark cytosine positions

        patch_noncpgMask_map[p] = seq_vec

    print("Done!")
    return(patch_noncpgMask_map)



def get_mPatchPosn_metadata(chrom, num_cpg, offset = 0):

    """
    offset: used when multiple chromosomes used for training/testing.
            add sum of lens for chr_{<i} for chr i.
    """
    
    hg38_chr_df = hg38_allcpg_df[hg38_allcpg_df.loc[:, "seqname"] == chrom].set_index("start")
    hg38_chr_df.index = hg38_chr_df.index + offset
    patch_refCpG_map = {}
    i = 0
    pid = 0
    while i < hg38_chr_df.shape[0]:
        patch_refCpG_map[pid] = hg38_chr_df.iloc[i:i+num_cpg]
        i += num_cpg
        pid += 1
    
    return(patch_refCpG_map)



### MINIMAL USAGE EXAMPLE
"""
import pandas as pd
from read_utils import read_chromCentric_dataset
from make_patch_metadata import get_patchPosn_metadata, get_patchStats_metadata

CWD = os.getcwd()
dataset = "gtex"
chrom = "chr21"
s = 5000
fm_data, cov_data = read_chromCentric_dataset(CWD, dataset, chrom)
samp_list = fm_data.columns[:5]
patch_avgBetas_df, patch_avgCov_df = get_patchStats_metadata(dataset, s, chrom, samp_list,
                                                             num_workers = 5, overwrite = True)
"""