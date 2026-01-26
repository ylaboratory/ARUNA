"""
Author: Janmajay Singh

Purpose: All modules in this file take a DIR with Sample SUBDIRS.
All "cc" functions turn these sample-wise subdirs and turn the dataset more chromosome-focused.
I.e., Saving 1 file per chrom, such as  chr1.fm storing beta values for 1 sample per column.

All "pc" functions take the .fm (and .mask for noise-simulations) and "patchify them".
For this patching process to complete, "pc" functions rely on pre-computed external metadata.
Metadata maps "patch ids" to CpG start posns that lie within that patch.
This metadata can be made using functions in make_patch_metadata.

"gt" -> ground truth and "noisy" -> noise simulations.
While "gt" functions typically only store fm (Beta value) files;
"noisy" functions store fm as well as "mask" files to refer to CpGs which were simulated to be missing.
Note that this differs from CpGs missing in the ground truth.

"get" functions are used to find and read the data or generate it (by calling make functions) if not found.
"make" functions can be explicity called as well if data is known not to exist in the data dir.
num_workers for multiprocessing can be assigned only when calling "make" funcs directly.
Otherwise a pre-asigned number of workers (typically 8) are used.
"""

import os
import glob
import pickle
import multiprocessing
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd

from aruna.patch_utils import get_patch_data
from aruna.read_utils import get_processed_df
from aruna.noise_simulators import get_sim_mask
from aruna.patch_metadata import get_mPatchPosn_metadata

from tqdm import tqdm
import warnings


CWD = os.getcwd()
# canonical cpg set from hg38 - zero indexed
hg38_metadata_dir = Path(__file__).resolve().parent.parent / "data" / "metadata"
hg38_all_df = pd.read_csv(os.path.join(hg38_metadata_dir, "hg38_cpg_py0idx.csv"))


# GET Functions
def get_cc_gt(dataset, chrom, data_dir = None):
    
    """
    Function to return Chrom-Centric Ground Truth data.
    If files are already on disk for provided dataset-chrom, they are read and returned.
    Otherwise, files are generated and returned.
    """

    hg38_chr_df = hg38_all_df[hg38_all_df.loc[:, "seqname"] == chrom].set_index("start")
    save_dir = os.path.join(CWD, "data", dataset, "chrom_centric/true")
    fm_save_path = Path(os.path.join(save_dir, "FractionalMethylation", chrom + ".fm"))
    cov_save_path = Path(os.path.join(save_dir, "ReadDepth", chrom + ".cov"))
    print("Looking for Ground-Truth FractionalMethylation and Coverage files...")
    if fm_save_path.is_file() and cov_save_path.is_file():
        print("All files found!\nFM at: {}\nCOV at: {}".format(fm_save_path,
                                                               cov_save_path))
        chr_fm_df = pd.read_csv(fm_save_path, sep = "\t", index_col= 0)
        chr_cov_df = pd.read_csv(cov_save_path, sep = "\t", index_col= 0)
    else:
        print("One or both files not found.")
        if data_dir == None:
            raise ValueError("Data directory with .cov files not provided!")
        chr_fm_df, chr_cov_df = make_cc_gt(dataset, chrom, 
                                           data_dir, hg38_chr_df, 
                                           fm_save_path, cov_save_path)
    
    return (chr_fm_df, chr_cov_df)



def get_cc_noisy(dataset, chrom, nr, data_dir = None):

    """
    Function to return Chrom-Centric Noise Simulation data.
    If files are already on disk for provided dataset-chrom, they are read and returned.
    Otherwise, files are generated and returned.
    """

    hg38_chr_df = hg38_all_df[hg38_all_df.loc[:, "seqname"] == chrom].set_index("start")   
    save_dir = os.path.join(CWD, "data", dataset, "chrom_centric", nr)
    fm_save_path = Path(os.path.join(save_dir, "FractionalMethylation", chrom + ".mask.fm"))
    mask_save_path = Path(os.path.join(save_dir, "SimulatedMask", chrom + ".mask"))
    print("Looking for Noise-Simulated FractionalMethylation and Coverage files...")
    if fm_save_path.is_file() and mask_save_path.is_file():
        print("All files exist!\nFM at: {}\nMASK at: {}".format(fm_save_path, 
                                                                mask_save_path))
        chr_simMiss_fm_df = pd.read_csv(fm_save_path, sep = "\t", index_col= 0)
        chr_simMask_df = pd.read_csv(mask_save_path, sep = "\t", index_col= 0)
    else:
        print("One or both files not found. Creating...")
        if data_dir == None:
            raise ValueError("Data directory with .cov files not provided!")
        # Read in GT Fractional Methylation Data
        chr_fm_df, _ = get_cc_gt(dataset, chrom, data_dir)
        chr_simMiss_fm_df, chr_simMask_df = make_cc_noisy(dataset, chrom, nr, 
                                                          hg38_chr_df, chr_fm_df, 
                                                          fm_save_path, mask_save_path)
        
    return(chr_simMiss_fm_df, chr_simMask_df)



def get_pc_gt(dataset, chrom, num_cpg = None, 
              data_dir = None, data_type = "fm", subset = None):
    
    """
    Function to return Patch-Centric Ground Truth data.
    If files are already on disk for provided dataset-chrom, they are read and returned.
    Otherwise, files are generated and returned.

    data_type (str): One of "fm" or "cov".
    """

    assert num_cpg, "Num CpGs need to be supplied!"
    save_dir = os.path.join(CWD, "data", dataset, "patch_centric", "numCpg"+str(num_cpg), "true")
    if not isinstance(chrom, (list, tuple)):
        chrom = [chrom] # works for single str chrom and multiple as list
    pc_map = defaultdict(list) # returned 
    for c in chrom:
        print("Current chromosome: ", c)
        # hg38_chr_df = hg38_all_df[hg38_all_df.loc[:, "seqname"] == c].set_index("start")
        if data_type == "fm":
            save_path = Path(os.path.join(save_dir, 
                                             "FractionalMethylation", 
                                             c + "_patches.fm.pkl"))
        elif data_type == "cov":
            save_path = Path(os.path.join(save_dir, 
                                             "ReadDepth", 
                                             c + "_patches.cov.pkl"))
        print("Looking for Ground-Truth Patchified {} files...".format(data_type.upper()))
        if save_path.is_file():
            print("Patchified data found at: {}".format(save_path))
            with open(save_path, "rb") as f:
                pc_gtfm_map = pickle.load(f)
        else:
            print("Patchified GT data not found!")
            if not data_dir:
                raise ValueError("Data directory with .cov files not provided!")
            pc_gtfm_map = make_pc_gt(dataset, c, 
                                    data_dir, save_path,
                                    num_cpg, data_type)
        if subset:
            for samp in subset:
                assert samp in list(pc_gtfm_map.keys()), "Samp not found in loaded data!"
                pc_map[samp].extend(pc_gtfm_map[samp])
        else:
            warnings.warn("Since sample subset is not provided, returning entire dataset!")
            for samp in list(pc_gtfm_map.keys()):
                pc_map[samp].extend(pc_gtfm_map[samp])

    return(pc_map)



def get_pc_noisy(dataset, chrom, 
                 num_cpg = None, nr = "rrbs_sim", 
                 data_dir = None, subset = None):
    
    """
    Function to return Patch-Centric Noise Simulated data.
    If files are already on disk for provided dataset-chrom, they are read and returned.
    Otherwise, files are generated and returned.
    """

    assert num_cpg, "Num CpGs need to be supplied!"
    save_dir = os.path.join(CWD, "data", dataset, "patch_centric", "numCpg"+str(num_cpg), nr)
    if not isinstance(chrom, (list, tuple)):
        chrom = [chrom] # works for single str chrom and multiple as list
    pc_nrfm_map = defaultdict(list)
    pc_nrmask_map = defaultdict(list)
    for c in chrom:
        print("Curent chromosome: ", c)
        fm_save_path = Path(os.path.join(save_dir, "FractionalMethylation", c + "_patches.mask.fm.pkl"))
        mask_save_path = Path(os.path.join(save_dir, "SimulatedMask", c + "_patches.mask.pkl"))
        print("Looking for Patchified Noise-Simulated FractionalMethylation and Coverage files...")
        if fm_save_path.is_file() and mask_save_path.is_file():
            print("All files exist!\nFM at: {}\nMASK at: {}".format(fm_save_path, 
                                                                    mask_save_path))
            with open(fm_save_path, "rb") as f:
                pc_simMiss_fm_map = pickle.load(f)
            with open(mask_save_path, "rb") as f:
                pc_simMiss_mask_map = pickle.load(f)
        else:
            print("One or both files not found!")
            if not data_dir:
                raise ValueError("Data directory with .cov files not provided!")
            pc_simMiss_fm_map, pc_simMiss_mask_map = make_pc_noisy(dataset, c, 
                                                                   data_dir, 
                                                                   fm_save_path, mask_save_path,
                                                                   num_cpg, nr)
        assert pc_simMiss_fm_map.keys() == pc_simMiss_mask_map.keys(), "Sample set mismatch found!"
        if subset:
            for samp in subset:
                assert samp in list(pc_simMiss_fm_map.keys()), "Samp not found in loaded FM!"
                assert samp in list(pc_simMiss_mask_map.keys()), "Samp not found in loaded Mask!"

                pc_nrfm_map[samp].extend(pc_simMiss_fm_map[samp])
                pc_nrmask_map[samp].extend(pc_simMiss_mask_map[samp])
        else:
            warnings.warn("Since sample subset is not provided, returning entire dataset!")
            for samp in list(pc_simMiss_fm_map.keys()):
                pc_nrfm_map[samp].extend(pc_simMiss_fm_map[samp])
                pc_nrmask_map[samp].extend(pc_simMiss_mask_map[samp])

    return(pc_nrfm_map, pc_nrmask_map)



# MAKE Functions
def make_cc_gt(dataset, chrom, 
               data_dir, hg38_chr_df,
               fm_save_path, cov_save_path, 
               num_workers = 8):

    """
    Function to generate Chrom-centric Ground Truth data.
    num_workers can only be controlled from here, by explicit calls.
    """
    print("Processing for {} and {}...".format(dataset, chrom))
    # get sample dirs and populate with sample pathnames
    merged_cpg_files = []
    if dataset == "gtex":
        samp_dirs = sorted(
                        d for d in os.listdir(data_dir) 
                        if d.startswith("GTEX-") and os.path.isdir(os.path.join(data_dir, d)))
        # samp_dirs = sorted(os.listdir(data_dir))[:-3] # last 3 are not sample dirs
    
    elif dataset == "encode_wgbs" or dataset == "encode_rrbs":
        samp_dirs = sorted(
                        os.path.basename(p) for p in glob.glob(os.path.join(data_dir, "ENCFF*")) 
                        if os.path.isdir(p))
    else:
        raise NotImplementedError("Sample list population logic for dataset unavailable!")
    # populate if no error raised
    for samp in samp_dirs:
        merged_cpg_files.extend(glob.glob(os.path.join(data_dir, samp, 
                                                       "*.cpgMerged.CpG_report.merged_CpG_evidence.cov")))
    print("cpgMerged.CpG_report.merged_CpG_evidence.cov files for {} samples found!".format(len(merged_cpg_files)))
    if len(samp_dirs) < num_workers:
        num_workers = len(samp_dirs)
    print("Creating FM and COV files with {} workers...".format(num_workers))
    pool = multiprocessing.Pool(processes = num_workers)
    with tqdm(total=len(merged_cpg_files)) as pbar:
        # Function to update the progress bar
        def update(*a):
            pbar.update()
        results = []
        for samp_file in merged_cpg_files:
            result = pool.apply_async(get_model_ready_data, args=(dataset, 
                                                                  samp_file,  chrom, 
                                                                  hg38_chr_df), callback=update)
            results.append(result)
        final_results = [result.get() for result in results]
    pool.close()
    pool.join()
    # Unpack results in desired format
    all_fm = {}
    all_cov = {}
    all_obs_perc = [] # for sanity stats

    for res in final_results:
        samp_name = res[0]
        all_fm[samp_name] = res[1]
        all_cov[samp_name] = res[2]
        all_obs_perc.append(res[3])
    # for sanity stats
    print("CpG Observation rates for {} - {} over ALL samples are: ".format(dataset, chrom))
    print("Mean: ", round(np.mean(all_obs_perc),3))
    print("Std: ", round(np.std(all_obs_perc),3))
    all_fm_df = pd.DataFrame(all_fm).set_index(hg38_chr_df.index)
    all_cov_df = pd.DataFrame(all_cov).set_index(hg38_chr_df.index)
    print("For {} - {}, saving Fractional Methylation Data for all samples to: {}...".format(dataset, chrom, fm_save_path))
    fm_save_path.parent.mkdir(parents=True, exist_ok=True) 
    all_fm_df.to_csv(fm_save_path, sep = "\t", header = True, index = True)
    print("For {} - {}, saving Read Coverage Data for all samples to: {}...".format(dataset, chrom, cov_save_path))
    cov_save_path.parent.mkdir(parents=True, exist_ok=True) 
    all_cov_df.to_csv(cov_save_path, sep = "\t", header = True, index = True)
    print("All files saved to disk!")

    return(all_fm_df, all_cov_df)



def make_cc_noisy(dataset, chrom, nr, 
                  hg38_chr_df, chr_fm_df, 
                  fm_save_path, mask_save_path):

    """
    Assumes that the Chrom-Centric Ground Truth file for chromosome is already available.
    Function to generate Chrom-centric Noise Simulated data if not found for provided dataset.
    Allowed noise regimes:
    1. mcar_[x]: where x is the integer % of missing CpGs needed.
    2. rrbs_sim: closest to RRBS-like missingness patterns.
    3. rrbs_ideal: simulates a setting where only CGIs are observed.

    Function saves 2 files for each setting:
    I) <chrom>.mask.fm with beta values (between 0 and 1) with NaNs to indicate missingness.
    II) <chrom>.mask with boolean values. These indicate CpGs which were simulated as missing.
        - This helps distinguish from CpGs missing in the original (WGBS) data.
    """

    print("Processing for {} - {} - {}...".format(dataset, chrom, nr))
    print("Creating Noise Simulation files...")
    # simulate missing masks for each sample independently
    all_simMiss_fm, all_simMiss_mask = get_noisy_data(nr, chrom, chr_fm_df, 
                                                      hg38_chr_df) # returns 2 dicts
    all_simMiss_fm_df = pd.DataFrame(all_simMiss_fm).set_index(hg38_chr_df.index)
    all_simMiss_mask_df = pd.DataFrame(all_simMiss_mask).set_index(hg38_chr_df.index)
    print("For {} - {}, saving Fractional Methylation Data for all samples to: {}...".format(dataset, chrom, fm_save_path))
    fm_save_path.parent.mkdir(parents=True, exist_ok=True) 
    all_simMiss_fm_df.to_csv(fm_save_path, sep = "\t", header = True, index = True)
    print("Saved!")
    print("For {} - {}, saving Simulated Mask Data for all samples to: {}...".format(dataset, chrom, mask_save_path))
    mask_save_path.parent.mkdir(parents=True, exist_ok=True) 
    all_simMiss_mask_df.to_csv(mask_save_path, sep = "\t", header = True, index = True)
    print("Saved!", end = "\n\n")
    print("All files saved to disk!")

    return(all_simMiss_fm_df, all_simMiss_mask_df)



def make_pc_gt(dataset, chrom, 
               data_dir, save_path, 
               num_cpg = None, 
               data_type = "fm", num_workers = 8):
    
    print("Processing for {} - {} and {} CpGs/Patch...".format(dataset, chrom, num_cpg))
    patch_refCpG_map = get_mPatchPosn_metadata(chrom, num_cpg, hg38_all_df)
    print("PatchID-CpG reference maps complete!")
    # Read in CC GT Fractional Methylation or ReadDepth Data to "patchify"
    if data_type == "fm":
        chr_df, _ = get_cc_gt(dataset, chrom, data_dir)
    elif data_type == "cov":
        _, chr_df = get_cc_gt(dataset, chrom, data_dir)
    if len(chr_df.columns) < num_workers:
        num_workers = len(chr_df.columns)
    print("Creating mappings such as sample: [patch_id: patch_betas] with {} workers...".format(num_workers))
    pool = multiprocessing.Pool(processes = num_workers)
    with tqdm(total=len(chr_df.columns)) as pbar:
        # Function to update the progress bar
        def update(*a):
            pbar.update()
        results = []
        for samp_name in chr_df.columns:
            result = pool.apply_async(get_samp_patchBetas_map, args=(samp_name, 
                                                                     chr_df, 
                                                                     patch_refCpG_map, 
                                                                     chrom), callback=update)
            results.append(result)
        final_results = [result.get() for result in results]
    pool.close()
    pool.join()
    pc_gt_map = {} # dict mapping sample names to list of tuples
    for samp_res in final_results:
        samp = list(samp_res.keys())[0]
        pc_gt_map[samp] = samp_res[samp]
    print("Patchification complete! Saving {} Data for all samples to: {}...".format(data_type.upper(), 
                                                                                     save_path))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(pc_gt_map, f)
    print("All files saved to disk!")

    return(pc_gt_map)



def make_pc_noisy(dataset, chrom, 
                  data_dir, 
                  fm_save_path, mask_save_path, 
                  num_cpg = None, 
                  nr = "rrbs_sim",  
                  num_workers = 8):

    print("Processing for {} - {} - {} and {} CpGs/Patch...".format(dataset, chrom, nr, num_cpg))
    patch_refCpG_map = get_mPatchPosn_metadata(chrom, num_cpg, hg38_all_df)
    print("PatchID-CpG reference maps complete!")

    # Read in CC Noisy Fractional Methylation Data to "Patchify"
    chr_simMiss_fm_df, chr_simMask_df = get_cc_noisy(dataset = dataset, 
                                                     chrom = chrom, nr = nr, 
                                                     data_dir = data_dir)
    if len(chr_simMiss_fm_df.columns) < num_workers:
        num_workers = len(chr_simMiss_fm_df.columns)
    print("Creating mappings such as sample: [patch_id: patch_betas] with {} workers...".format(num_workers))
    # -------- PROCESS DATA FOR FRACTIONAL METHYLATION -------- #
    pool = multiprocessing.Pool(processes = num_workers)
    with tqdm(total=len(chr_simMiss_fm_df.columns)) as pbar:
        # Function to update the progress bar
        def update(*a):
            pbar.update()
        results = []
        for samp_name in chr_simMiss_fm_df.columns:
            result = pool.apply_async(get_samp_patchBetas_map, args=(samp_name, 
                                                                     chr_simMiss_fm_df, 
                                                                     patch_refCpG_map, 
                                                                     chrom), callback=update)
            results.append(result)
        final_results = [result.get() for result in results]
    pool.close()
    pool.join()
    pc_simMiss_fm_map = {} # dict mapping sample names to list of tuples
    for samp_res in final_results:
        samp = list(samp_res.keys())[0]
        pc_simMiss_fm_map[samp] = samp_res[samp]

    print("Patchification complete! Saving FM Data for all samples to: {}...".format(fm_save_path))
    fm_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fm_save_path, "wb") as f:
        pickle.dump(pc_simMiss_fm_map, f)
    print("Saved!")
    # -------- PROCESS DATA FOR SIMULATED MASKING -------- #
    pool = multiprocessing.Pool(processes = num_workers)
    with tqdm(total=len(chr_simMask_df.columns)) as pbar:
        # Function to update the progress bar
        def update(*a):
            pbar.update()
        results = []
        for samp_name in chr_simMask_df.columns:
            result = pool.apply_async(get_samp_patchBetas_map, args=(samp_name, 
                                                                     chr_simMask_df, 
                                                                     patch_refCpG_map, 
                                                                     chrom), callback=update)
            results.append(result)
        final_results = [result.get() for result in results]
    pool.close()
    pool.join()
    pc_simMiss_mask_map = {} # dict mapping sample names to list of tuples
    for samp_res in final_results:
        samp = list(samp_res.keys())[0]
        pc_simMiss_mask_map[samp] = samp_res[samp]

    print("Patchification complete! Saving SimMask Data for all samples to: {} ...".format(mask_save_path))
    mask_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mask_save_path, "wb") as f:
        pickle.dump(pc_simMiss_mask_map, f)
    print("Saved!")

    print("All files saved to disk!")

    return(pc_simMiss_fm_map, pc_simMiss_mask_map)


# HELPER FUNCTIONS

def get_model_ready_data(dataset, samp_file, chrom, hg38_chrcpg_df):

    """
    Reads data (output of Bismark pipeline, see read_utils for details).
    Converts beta values to be in [0,1] and computes a total read depth column.
    NOTE: Turns the CpG set in read DF into a "Canonical form"
    """
    if dataset == "atlas": # special naming otherwise replicates override each other
        samp_name = "_".join(samp_file.split("/")[-1].split("_")[:2])
    else:
        samp_name = samp_file.split("/")[-1].split("_")[0]

    num_chr_cpg = hg38_chrcpg_df.shape[0] # to also calc obs rates in WGBS (it ain't 100%)
    
    samp_df = get_processed_df(samp_file, chrom) # read data with beta in [0,1] and total depth col
    std_samp_df = samp_df.merge(hg38_chrcpg_df, how = "right", on = "start") # canonical form
    
    # split up data into respective vars
    samp_fm = std_samp_df.loc[:, "beta"].values
    samp_cov = std_samp_df.loc[:, "read_depth"].values
    samp_obs_perc = (samp_df.shape[0]/num_chr_cpg) * 100

    return(samp_name, samp_fm, samp_cov, samp_obs_perc)



def get_noisy_data(noise_regime, chrom, 
                   chr_fm_df, hg38_chrcpg_df):

    """
    Noise simulated for each sample INDEPENDENTLY.
    """
    all_sim_miss_fm = {}
    all_sim_miss_mask = {}
    for samp in chr_fm_df.columns:
        samp_betas = deepcopy(chr_fm_df.loc[:, samp].values)
        sim_mask = get_sim_mask(noise_regime, chrom, hg38_chrcpg_df)
        samp_betas[sim_mask] = np.nan
        all_sim_miss_fm[samp] = samp_betas
        all_sim_miss_mask[samp] = sim_mask
    # Stats for sanity (and curiosity)
    print("Computing metrics for sanity checks...")
    all_simMiss_rates = []
    all_origMiss_rates = []
    all_aftMiss_rates = []
    for samp in all_sim_miss_fm.keys():
        all_simMiss_rates.append(np.sum(all_sim_miss_mask[samp])/all_sim_miss_mask[samp].shape[0] * 100)
        all_origMiss_rates.append(chr_fm_df.loc[:, samp].isna().sum()/chr_fm_df.shape[0] * 100)
        all_aftMiss_rates.append(np.sum(np.isnan(all_sim_miss_fm[samp]))/all_sim_miss_fm[samp].shape[0] * 100)
    print("Avg: {} Std: {} of Originally Missing CpG rates.\n".format(round(np.mean(all_origMiss_rates),3), 
                                                                      round(np.std(all_origMiss_rates),3)))
    print("Avg: {} Std: {} of Simulated Missing CpG rates.\n".format(round(np.mean(all_simMiss_rates),3), 
                                                                     round(np.std(all_simMiss_rates),3)))
    print("Avg: {} Std: {} of After-Simulation Missing CpG rates.\n".format(round(np.mean(all_aftMiss_rates),3), 
                                                                            round(np.std(all_aftMiss_rates),3)))

    return(all_sim_miss_fm, all_sim_miss_mask)



def get_samp_patchBetas_map(samp, chr_df, 
                            patch_refCpG_map, chrom = None):

    """
    Cuts up a sample into component patches and returns a dict: [list of patches].
    NOTE: Hides a (slow) merge operation in the get_patch_data function.
    """

    samp_allpatch_map = defaultdict(list)
    samp_chr_df = chr_df.loc[:, samp].to_frame() # otherwise a Series is returned
    for p in patch_refCpG_map.keys():
        if patch_refCpG_map[p].shape[0] > 0: # patch has >0 CpG
            patch_betas = get_patch_data(samp, samp_chr_df,
                                         p, patch_refCpG_map)
            # chrom last of tuple for backward comp with dataloader
            samp_allpatch_map[samp].append((p, patch_betas, chrom)) # MODIFIED to include chrom
    
    return(samp_allpatch_map)