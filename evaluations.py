
"""
NOTE: These are M-Patch Specific for now
NOTE: For G-Patch, see eval_utils.py

Five Core plots + 2 Optional:

1. Whole methylome
    - Distributions of all GT
    - Distributions of all Inferred
    - Distribution of True/Simulated RRBS [to help answer: how much variability across samples was captured in the RRBS which served as the starting point of the problem?]

    - WM analysis split by input/ground truth coverage (is variability coming from coverage?)

2. PCA (1v2 and 2v3)
    - All GT
    - All Inferred
    - All RRBS

3. Mean Absolute Error vs. Distance from last observed CpG
    - Averaged over test samples
    - Showing standard deviation shadow
    - With or without double y-axis
    - Highlighting at what point performance becomes a function of something simpler

4. Heatmap of GT vs. Inferred methylation ✓
    - Alternative to calibration plot
    - As shown in GimmeCpG

5. Boxplots (all CpGs, Ziller set etc)
    - This includes the 2 core quantitative metrics: MSE and R^2

6. Coverage Heatmaps
    6.1 Training data WGBS (sim_rrbs): CpG coverage bin (x-axis) vs. Absolute Error (color by numbers that fall within that coordinate-bin)
        - Shows intra-dataset error distribution as a function of the original coverage of the CpG
        - Helps answer: How does error depend on the true known coverage of the CpG when ** no batch effects exist **? 
        - This is important since the true coverage determines the possible beta values of the CpG.
    6.2 Test-train comparison (across dataset): CpG coverage bin (x-axis) from testing set, same CpGs coverage bin (y-axis) from training set, fill in the absolute error.

    6.1 [ALT.] CpG coverage bins (x-axis), true/inferred betas (y-axis), color by MAE or MSE.

7. If no variation, it is of interest to know where is the original variation coming from and how exactly the smoothing happens that it disappears.
    7.1 Plot histogram of AE per CpG, take out the CpG set with highest AE and see where they belong (annotation analysis).
    7.2 AUROC plots
"""

# TODO: This is GTEx specific for now

import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from utils import OrderedCounter
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib import cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


CWD = os.getcwd()
modelData_dir = os.path.join(CWD, "model_data/saved_models")
hg38_metadata_dir = os.path.join(CWD, "metadata/hg38_ref")
hg38_all_df = pd.read_csv(os.path.join(hg38_metadata_dir, "hg38_cpg_py0idx.csv"))
gtex_metadata_fpath = "/home/js228/patch_recurrence/metadata/GTEx_v9_WGBS_data_Bulk_fastq_metadata.csv"
gtex_metadata_df = pd.read_csv(gtex_metadata_fpath)


# ---------- DATA COLLECTION AND COLLATION FUNCTIONS ----------
def collate_mpatches(val_res): #val_res dict is from model_engine.valid_step_mpatch

    """
    Collates all MPatches for all supplied samples and chromosomes to form chr-methylomes.
    
    Arguments
    ------
    val_res: Output from model_engine.valid_step_mpatch.
    
    Returns
    ------
    Dict of List of 4-tuples (samp, true_seq, pred_seq, eval_mask)
    Each of the latter three should be of len = num_cpg in chrom 
    """

    all_collated_data = defaultdict(list)

    # lists of len #Patches
    val_samps = val_res["samples"]
    val_chrom = val_res["chrom"]
    # lists of len #Batches --> np.array of shape (#Patches,#CpG)
    val_gt = np.concatenate(val_res["gt"])
    val_preds = np.concatenate(val_res["preds"])
    val_evalMask = np.concatenate(val_res["evalMask"])

    samp_numPatches_map = OrderedCounter(val_samps) # maintains same order as shuffle = False in dataloader
    numPatches_perSamp = set(samp_numPatches_map.values())
    assert len(numPatches_perSamp) == 1, "Samples can not have different number of patches!"
    numPatches_perSamp = numPatches_perSamp.pop() #extract sole value

    # sample loop
    for i, samp in tqdm(enumerate(samp_numPatches_map.keys()), 
                        desc = "Sample: ",
                        total = len(samp_numPatches_map.keys())):
        samp_begin = i*numPatches_perSamp
        samp_end = (i+1)*numPatches_perSamp

        samp_subset = val_samps[samp_begin:samp_end] # till (end_idx-1)
        assert len(set(samp_subset)) == 1, "More than one sample extracted!"
        assert set(samp_subset).pop() == samp, "Sample mismatch!"

        samp_chrom_subset = val_chrom[samp_begin:samp_end]
        samp_gt_subset = val_gt[samp_begin:samp_end]
        samp_pred_subset = val_preds[samp_begin:samp_end]
        samp_evalMask_subset = val_evalMask[samp_begin:samp_end]

        if i == 0: #initialize only from first sample
            # diff chr have diff num patches (but identical over samples)
            chrom_numPatches_map = OrderedCounter(samp_chrom_subset)
            chrom_totcpg_map = defaultdict()
            for chrom in chrom_numPatches_map.keys():
                hg38_chr_df = hg38_all_df[hg38_all_df.loc[:, "seqname"] == chrom].set_index("start")
                tot_chrom_cpgs = hg38_chr_df.shape[0]
                chrom_totcpg_map[chrom] = tot_chrom_cpgs

        # chromosome loop
        j = 0
        for chrom in chrom_numPatches_map.keys():
            numPatches_chrom = chrom_numPatches_map[chrom]
            chrom_begin = j
            chrom_end = j + numPatches_chrom

            hg38_chr_df = hg38_all_df[hg38_all_df.loc[:, "seqname"] == chrom].set_index("start")
            tot_chrom_cpgs = hg38_chr_df.shape[0]
            
            chrom_subset = samp_chrom_subset[chrom_begin:chrom_end]
            assert len(set(chrom_subset)) == 1, "More than one chromosome extracted!"
            assert set(chrom_subset).pop() == chrom, "Chromosome mismatch!"

            chrom_gt_subset = samp_gt_subset[chrom_begin:chrom_end] # till (chrom_end-1)
            chrom_pred_subset = samp_pred_subset[chrom_begin:chrom_end]
            chrom_evalMask_subset = samp_evalMask_subset[chrom_begin:chrom_end]

            # subset from reference num cpgs to remove padding effect from data_engine
            true_seq = chrom_gt_subset.flatten()[:chrom_totcpg_map[chrom]]
            pred_seq = chrom_pred_subset.flatten()[:chrom_totcpg_map[chrom]]
            eval_mask = chrom_evalMask_subset.flatten()[:chrom_totcpg_map[chrom]]

            all_collated_data[chrom].append((samp, true_seq, pred_seq, eval_mask))
        
            j += numPatches_chrom
    
    return (all_collated_data)



def collate_mslices(val_res, val_spp, chrom):

    all_collated_data = defaultdict(list)

    samp_list = val_res["samples"]
    # pid_list = val_res["pids"]
    true_list = val_res["gt"]
    preds_list = val_res["preds"]
    evalMask_list = val_res["evalMask"]

    # infer these
    sps, num_cpg = true_list[0].shape[1], true_list[0].shape[2]

    slice_list = [j for i in samp_list for j in zip(*i)] # len is num_slices = patches * samples * spp
    num_samples = len(set([i[0] for i in slice_list]))
    num_patches = int(len(slice_list)//(num_samples*val_spp))

    slice_sampdata = np.array(slice_list).reshape(num_samples, num_patches, 
                                                val_spp, sps)
    # pid_data = np.array(pid_list).reshape(num_samples, num_patches, val_spp)

    # 0-th index since the first/top sample in the slice is our target for evals
    gt_data = np.concatenate(true_list)[:,0,:].reshape(num_samples, 
                                                       num_patches, 
                                                       val_spp, num_cpg)
    pred_data = np.concatenate(preds_list)[:,0,:].reshape(num_samples, 
                                                          num_patches, 
                                                          val_spp, num_cpg)
    evalMask_data = np.concatenate(evalMask_list)[:,0,:].reshape(num_samples, 
                                                                 num_patches, 
                                                                 val_spp, num_cpg)
    for samp_idx in range(num_samples):
        for repeat_idx in range(val_spp):

            if samp_idx == 0:
                hg38_chr_df = hg38_all_df[hg38_all_df.loc[:, "seqname"] == chrom].set_index("start")
                tot_cpgs = hg38_chr_df.shape[0]

            # slice_set = slice_sampdata[samp_idx,:,repeat_idx,:] # may use later
            samp, = np.unique(slice_sampdata[samp_idx,:,repeat_idx,:][:,0])

            # subset from reference num cpgs to remove padding effect from data_engine
            true_seq = gt_data[samp_idx,:,repeat_idx,:].flatten()[:tot_cpgs]
            pred_seq = pred_data[samp_idx,:,repeat_idx,:].flatten()[:tot_cpgs]
            eval_mask = evalMask_data[samp_idx,:,repeat_idx,:].flatten()[:tot_cpgs]

            all_collated_data[chrom].append((samp, true_seq, pred_seq, 
                                            eval_mask, repeat_idx))

    return (all_collated_data)



def collate_mslices_test(test_res, spp, chrom):

    all_collated_data = defaultdict(list)

    samp_list = test_res["samples"]
    preds_list = test_res["preds"]

    # infer these
    sps, num_cpg = preds_list[0].shape[1], preds_list[0].shape[2]

    slice_list = [j for i in samp_list for j in zip(*i)] # len is num_slices = patches * samples * spp
    num_samples = len(set([i[0] for i in slice_list]))
    num_patches = int(len(slice_list)//(num_samples*spp))

    slice_sampdata = np.array(slice_list).reshape(num_samples, num_patches, 
                                                  spp, sps)
    # 0-th index since the first/top sample in the slice is our target for evals
    pred_data = np.concatenate(preds_list)[:,0,:].reshape(num_samples, 
                                                          num_patches, 
                                                          spp, num_cpg)

    for samp_idx in range(num_samples):
        for repeat_idx in range(spp):

            if samp_idx == 0:
                hg38_chr_df = hg38_all_df[hg38_all_df.loc[:, "seqname"] == chrom].set_index("start")
                tot_cpgs = hg38_chr_df.shape[0]

            # slice_set = slice_sampdata[samp_idx,:,repeat_idx,:] # may use later
            samp, = np.unique(slice_sampdata[samp_idx,:,repeat_idx,:][:,0])
            # subset from reference num cpgs to remove padding effect from data_engine
            pred_seq = pred_data[samp_idx,:,repeat_idx,:].flatten()[:tot_cpgs]
            all_collated_data[chrom].append((samp, pred_seq, repeat_idx))

    return (all_collated_data)




def process_seq(all_collated_data, cpgMask_map, 
                type_ = "nothing", slice = False):

    """
    Method to modify the predicted sequence (and the GT sequence) depending on type_.

    Arguments
    -----
    all_collated_data: dict of list of 4-tuples: (samp, true_seq, pred_seq, eval_mask)
    type_: str
           One of ["standard", "learned", "recons". "nan_subs", "intermediate", 
                   "cgi", "cgi_shelf", "cgi_shore", "cgi_inter" "promoter", 
                   "3utr", "5utr", "exon", "intron", "ziller"].
            Takes a subset of the original seq (which is of len = #canonical CpGs) for evaluation.
            
            Modeling-specific types;
            standard: Replaces reconstructed CpGs (those observed in RRBS and thus during training) with GT value, keeps only those CpGs obs in WGBS
            imputed: Include only CpGs observed in WGBS but missing in RRBS
                     Makes sense only for rrbs_sim settings and not actual RRBS testing (?)
            recons: Include only CpGs observed in RRBS (i.e., not forming learning signal). Performance is expected near 0 error
            nan_subs: Include only CpGs missing in training WGBS. Performance is expected to be random

            Biologically-relevant types;
            intermediate: Include only CpGs with true beta between 0 and 1.
            cgi: CpG-islands
            cgi_shelf: CpG-shelves
            cgi_shore: CpG-shores
            cgi_inter: Open sea CpGs
            promoter: CpGs overlapping with known promoters. Likely to overlap with CGIs
            3utr: CpGs overlapping with 3' UTRs
            5utr: CpGs overlapping with 5' UTRs
            exon: CpGs overlapping with exons
            intron: CpGs overlapping with introns
            1to5kb: CpGs lying within a distance range of 1 kb to 5 kb of known genic regions, upstream or downstream
            ziller: A dynamic CpG set from Ziller et al. Nature 2013


    Returns
    -----
    Modifies sample pred_seq based on type_.


    Note: See bottom of this file for an example cpgMask creation.
    """

    biological_annots = ["intermediate", "cgi", "cgi_shelf", 
                         "cgi_shore", "cgi_inter", "promoter", "3utr", 
                         "5utr", "exon", "intron", "1to5kb", "ziller"]

    chrom_gt_map = {}
    chrom_pred_map = {}

    for chrom in all_collated_data.keys():
        print("Current chromosome: ", chrom)
        
        samp_gt_map = {}
        samp_pred_map = {}
        samp_procseq_lens = []
        model_obs_numcpg = []

        for samp_data in all_collated_data[chrom]:
            samp_name = samp_data[0]
            gt = samp_data[1]
            pred = samp_data[2]
            eval_mask = samp_data[3]
            cpg_mask = cpgMask_map[chrom][samp_name]
            recons_mask = (~eval_mask & ~cpg_mask) # only T when cpg_mask = F & sim_mask = F
            
            # numpy is pass by reference, modified sequence should not overwrite original
            gt_hat = gt.copy()
            pred_hat = pred.copy()

            if type_ == "nothing":
                pass

            elif type_ == "standard":
                pred_hat[recons_mask] = gt[recons_mask]
                gt_hat = gt_hat[~cpg_mask]
                pred_hat = pred_hat[~cpg_mask]
            
            elif type_ == "imputed":
                gt_hat = gt[eval_mask]
                pred_hat = pred[eval_mask]

            elif type_ == "recons":
                gt_hat = gt[recons_mask]
                pred_hat = pred[recons_mask]

            elif type_ == "nansubs": # missing in WGBS
                gt_hat = gt[cpg_mask]
                pred_hat = pred[cpg_mask]
            
            elif type_ in biological_annots:
                if type_ == "intermediate":
                    annot_mask = ~((gt == 0) | (gt == 1))
                else:
                    annot_mask = get_annot_mask(chrom, type_)
                
                # Subset: Consider only CpGs within annotation AND observed in GT
                subset_mask = (annot_mask & ~cpg_mask)
                gt_hat = gt[subset_mask]
                # Replace: reconstructed CpGs with GT data then Subset
                pred_hat[recons_mask] = gt[recons_mask]
                pred_hat = pred_hat[subset_mask]

                # CpGs that were reconstructed AND are in the subset 
                # ignores cpgs at GT nan_subs
                model_obs_numcpg.append(np.sum(recons_mask & annot_mask)) 

            samp_procseq_lens.append(len(gt_hat))

            if slice: # to demarcate repeats
                samp_name = samp_name + "_" + str(samp_data[4])
            
            samp_gt_map[samp_name] = gt_hat
            samp_pred_map[samp_name] = pred_hat
        
        print("#CpG stats under eval: {} +/- {}".format(np.mean(samp_procseq_lens), 
                                                        np.std(samp_procseq_lens)))
        if type_ in biological_annots:
            print("{} +/- {} CpGs were observed by the model".format(np.mean(model_obs_numcpg), 
                                                                     np.std(model_obs_numcpg)))
        
        chrom_gt_map[chrom] = samp_gt_map
        chrom_pred_map[chrom] = samp_pred_map
    
    return(chrom_gt_map, chrom_pred_map)


def process_seq_test(all_collated_data, slice = False):
    
    """
    Useful with true RRBS collates.
    """
    chrom_pred_map = {}

    for chrom in all_collated_data.keys():
        print("Current chromosome: ", chrom)
        samp_pred_map = {}

        for samp_data in all_collated_data[chrom]:
            samp_name = samp_data[0]
            pred = samp_data[1]
 
            if slice: # to demarcate repeats
                samp_name = samp_name + "_" + str(samp_data[2])
            
            samp_pred_map[samp_name] = pred
        
        chrom_pred_map[chrom] = samp_pred_map
    
    return(chrom_pred_map)



def get_annot_mask(chrom, annot):

    """
    Supports annot in [cgi, cgi_shelf, cgi_shore, cgi_inter, 
                       promoter, 3utr, 5utr, exon, intron, 1to5kb,
                       ziller, vmr]
    """

    chrom_metadata_dir = os.path.join(CWD, "metadata", chrom)
    if annot not in ["ziller", "vmr"]:
        annot_mask_fpath =  os.path.join(chrom_metadata_dir, 
                                        annot + "_hg38_mask.pkl")
    else:
        annot_mask_fpath =  os.path.join(chrom_metadata_dir, 
                                        annot + "_" + chrom + "_hg38_mask.pkl")
    with open(annot_mask_fpath, "rb") as f:
        annot_mask = pickle.load(f)

    if hasattr(annot_mask, "to_numpy"):
        annot_mask = annot_mask.to_numpy()

    return annot_mask



# ---------- [5] BOXPLOTS OF QUANT. PERFORMANCE, RESULT DFs ----------
def eval_sampwise_quant(samp_trueseq_map, 
                        samp_predseq_map,
                        print_avg = False):
    
    tot_mse = 0
    tot_r2 = 0
    samp_mse_map = {}
    samp_r2_map = {}
    num_samps = len(samp_trueseq_map.keys())

    for samp in samp_trueseq_map.keys():
        samp_mse = mean_squared_error(samp_trueseq_map[samp], 
                                      samp_predseq_map[samp])
        
        samp_r2 = r2_score(samp_trueseq_map[samp], 
                             samp_predseq_map[samp]) 
        
        samp_mse_map[samp] = samp_mse
        samp_r2_map[samp] = samp_r2

        tot_mse += samp_mse
        tot_r2 += samp_r2
    
    if print_avg:
        print("Avg. MSE of {} samples: {}".format(num_samps, tot_mse/num_samps))
        print("Avg. R2 of {} samples: {}".format(num_samps, tot_r2/num_samps))


    return(samp_mse_map, samp_r2_map)



# ---------- [4],[6] HEATMAPS ----------

def get_binidx(bins, x):
    """
    bins: array of bin edges, length n_bins+1
    returns: 0..n_bins-1 for values within [bins[0], bins[-1]]
             -1 for values outside range or NaN
    """
    x = np.asarray(x)
    n_bins = len(bins) - 1

    idx = np.full(x.shape, -1, dtype=np.int64)
    valid = np.isfinite(x) & (x >= bins[0]) & (x <= bins[-1])

    # right-closed bins: (bins[i], bins[i+1]] except first includes left edge
    tmp = np.searchsorted(bins, x[valid], side="right") - 1
    tmp = np.clip(tmp, 0, n_bins - 1)
    idx[valid] = tmp
    return idx

    

# -------------------------
# Heatmap matrix builder
# -------------------------
def get_heatmap_data(
    test_samps,
    samp_trueseq_map,
    samp_predseq_map,
    n_bins=20,
    normalize=True,
    drop_invalid=True,
    return_valid_mask=False,
):
    """
    Builds a 2D matrix where:
      - columns are ground-truth bins
      - rows are inferred bins
      - entry (y,x) counts CpGs with gt_bin=x and inf_bin=y

    normalize=True -> column-normalize to estimate P(inferred_bin | true_bin)
    """

    # ---- Gather data ----
    all_gt = []
    all_inf = []
    for samp in test_samps:
        all_gt.append(np.asarray(samp_trueseq_map[samp]))
        all_inf.append(np.asarray(samp_predseq_map[samp]))

    ground_truth = np.concatenate(all_gt)
    inferred = np.concatenate(all_inf)

    # ---- Bins / labels ----
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.round(bins, 6)  # mild rounding; searchsorted handles most issues anyway

    bin_labels = [f"[{bins[0]:.2f}, {bins[1]:.2f}]"] + \
                 [f"({bins[i]:.2f}, {bins[i+1]:.2f}]" for i in range(1, n_bins)]

    # ---- Bin indices ----
    gt_bin_idx  = get_binidx(bins=bins, x=ground_truth)
    inf_bin_idx = get_binidx(bins=bins, x=inferred)

    valid = (gt_bin_idx >= 0) & (inf_bin_idx >= 0)

    if drop_invalid:
        gt_bin_idx  = gt_bin_idx[valid]
        inf_bin_idx = inf_bin_idx[valid]
        ground_truth_valid = ground_truth[valid]
        inferred_valid     = inferred[valid]
    else:
        # keep originals for plotting marginals; heatmap will ignore invalid by masking below
        ground_truth_valid = ground_truth
        inferred_valid     = inferred

    # ---- Build heatmap counts (vectorized) ----
    heatmap_counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    np.add.at(heatmap_counts, (inf_bin_idx, gt_bin_idx), 1.0)

    if normalize:
        # Column-normalize: per GT bin
        col_sums = heatmap_counts.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            heatmap_norm = heatmap_counts / col_sums
        heatmap_norm = np.nan_to_num(heatmap_norm, nan=0.0, posinf=0.0, neginf=0.0)
        final_matrix = heatmap_norm
    else:
        final_matrix = heatmap_counts

    out = (final_matrix, bins, bin_labels, ground_truth_valid, inferred_valid)
    if return_valid_mask:
        out = out + (valid,)
    return out


# -------------------------
# Plotting
# -------------------------
def get_heatmap_plot(
    heatmap_data,
    bins,
    bin_labels,
    ground_truth=None,
    inferred=None,
    *,
    cmap="viridis",
    diag_linewidth=0.6,
    normalize_label=True,
    vmin=None,
    vmax=None):
    """
    If ground_truth and inferred are provided, adds marginal histograms.
    """
    n_bins = len(bin_labels)

    if (ground_truth is not None) and (inferred is not None):
        fig = plt.figure(figsize=(20, 20), dpi=300)

        gs = gridspec.GridSpec(
            2, 2, figure=fig,
            width_ratios=[7, 2],
            height_ratios=[2, 7],
            wspace=0.02, hspace=0.02
        )

        # Main heatmap
        ax_main = plt.subplot(gs[1, 0])
        sns.heatmap(
            heatmap_data,
            cmap=cmap,
            xticklabels=bin_labels,
            yticklabels=bin_labels,
            cbar=False,
            vmin=vmin,
            vmax=vmax,
            ax=ax_main
        )
        ax_main.invert_yaxis()
        ax_main.set_xlabel("Ground Truth", fontsize=35)
        ax_main.set_ylabel("Inferred", fontsize=35)
        ax_main.tick_params(axis="x", rotation=55, labelsize=24)
        ax_main.tick_params(axis="y", rotation=0, labelsize=24)

        # Diagonal outline boxes
        for i in range(n_bins):
            ax_main.add_patch(
                plt.Rectangle((i, i), 1, 1, fill=False,
                              edgecolor="white", linewidth=diag_linewidth)
            )

        # Top histogram (ground truth)
        ax_top = plt.subplot(gs[0, 0])
        sns.histplot(
            ground_truth,
            bins=bins,
            ax=ax_top,
            color=sns.color_palette("rocket")[0],
            fill=True
        )
        ax_top.set_xlim(bins[0], bins[-1])
        ax_top.axis("off")

        # Right histogram (inferred)
        ax_right = plt.subplot(gs[1, 1])
        sns.histplot(
            y=inferred,
            bins=bins,
            ax=ax_right,
            color=sns.color_palette("rocket")[0],
            fill=True
        )
        ax_right.set_ylim(bins[0], bins[-1])
        ax_right.axis("off")

        # Colorbar manually placed
        divider = make_axes_locatable(ax_right)
        cax = divider.append_axes("right", size="15%", pad=0.05)
        img = ax_main.collections[0]
        cbar = fig.colorbar(img, cax=cax)
        cbar.ax.tick_params(labelsize=20)

        # if normalize_label:
        #     cbar.set_label("P(inferred bin | true bin)", fontsize=16)
        # else:
        #     cbar.set_label("Fraction of CpGs", fontsize=16)

        for spine in cax.spines.values():
            spine.set_visible(False)

        return fig

    # Simple heatmap only
    fig = plt.figure(figsize=(20, 16), dpi=300)
    ax = plt.gca()
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        xticklabels=bin_labels,
        yticklabels=bin_labels,
        cbar_kws={"label": "CpG Count"},
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )

    for i in range(n_bins):
        ax.add_patch(
            plt.Rectangle((i, i), 1, 1, fill=False,
                          edgecolor="white", linewidth=diag_linewidth)
        )

    ax.invert_yaxis()
    ax.set_xlabel("Ground Truth", fontsize=16)
    ax.set_ylabel("Inferred", fontsize=16)
    ax.tick_params(axis="x", rotation=55, labelsize=14)
    ax.tick_params(axis="y", rotation=0, labelsize=14)

    return fig



def plot_coverage_vs_methylation(test_samps,
                                 samp_betaseq_map, 
                                 samp_covseq_map, 
                                 beta_n_bins = 20, normalize = True,
                                 ylabel = "Unknown", get_cb_params = False,
                                 vmin = None, vmax = None, pal = "crest"):
    
    # Data Setup
    all_betas = []
    all_cov = []
    for samp in test_samps:
        all_betas.append(samp_betaseq_map[samp])
        all_cov.append(samp_covseq_map[samp])

    beta_seq = np.concatenate(all_betas)
    coverage = np.concatenate(all_cov)

    assert np.all(beta_seq <= 1.0), "Beta values exceed 1"
    assert np.all(coverage >= 1), "Coverage must be >= 1"

    # -----------------------------
    # Define bins
    # -----------------------------

    cov_bins = [1, 5, 10, 15, 20, 25, 30, np.inf]
    cov_bin_labels = [f"[{cov_bins[0]}, {cov_bins[1]}]"] + \
    [f"({cov_bins[i]}, {cov_bins[i+1]}]" for i in range(1, len(cov_bins)-2)]  + \
        [f"({cov_bins[-2]}, \u221E)"]

    beta_bins = np.linspace(0, 1.0, beta_n_bins + 1)
    beta_bins = np.round(beta_bins, 3) # avoid weird floating point issues
    beta_bin_labels = [f"[{beta_bins[0]:.2f}, {beta_bins[1]:.2f}]"] + \
        [f"({beta_bins[i]:.2f}, {beta_bins[i+1]:.2f}]" for i in range(1, beta_n_bins)]

    # Digitize
    # cov_bin_idx = np.digitize(coverage, bins=cov_bins, right=True)
    # beta_bin_idx = np.digitize(beta_seq, bins=beta_bins, right=True)
    # cov_bin_idx = np.clip(cov_bin_idx - 1, 0, len(cov_bins) - 2)
    # beta_bin_idx = np.clip(beta_bin_idx - 1, 0, beta_n_bins - 1)

    cov_bin_idx = get_binidx(bins = cov_bins, x = coverage)
    beta_bin_idx = get_binidx(bins = beta_bins, x = beta_seq)

    # heatmap: rows = methylation bins, cols = coverage bins
    heatmap_data = np.zeros((beta_n_bins, len(cov_bins) - 1))
    for x, y in zip(cov_bin_idx, beta_bin_idx):
        heatmap_data[y, x] += 1

    if normalize:
        # Normalize by column (coverage bin)
        column_sums = heatmap_data.sum(axis=0, keepdims=True)
        normalized_data = heatmap_data / column_sums
        normalized_data = np.nan_to_num(normalized_data)
        final_matrix = normalized_data
    else:
        final_matrix = heatmap_data

    # -----------------------------
    # Plotting
    # -----------------------------
    fig = plt.figure(figsize=(15, 15), dpi=300)
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[7, 2],
                           height_ratios=[2, 7],
                           wspace=0.02, hspace=0.02)

    # Main heatmap
    ax_main = plt.subplot(gs[1, 0])

    if (vmin is not None) and (vmax is not None):
        sns.heatmap(
            final_matrix,
            vmin = vmin,
            vmax = vmax,
            cmap=pal,
            xticklabels=cov_bin_labels,
            yticklabels=beta_bin_labels,
            cbar=False,
            ax=ax_main
        )
    else:
        sns.heatmap(
        final_matrix,
        cmap=pal,
        xticklabels=cov_bin_labels,
        yticklabels=beta_bin_labels,
        cbar=False,
        ax=ax_main
    )
        
    ax_main.invert_yaxis()
    ax_main.set_xlabel("Coverage Bin", fontsize=16)
    ax_main.set_ylabel(ylabel, fontsize=16)
    ax_main.tick_params(axis='both', labelsize=14)
    ax_main.tick_params(axis='x', rotation=55)
    ax_main.tick_params(axis='y', rotation=0)

   
    # Top histogram: coverage
    ax_top = plt.subplot(gs[0, 0])
    sns.histplot(coverage, bins=[1, 5, 10, 15, 20, 25, 30, 35], ax=ax_top)
    ax_top.set_xlim(1, 35)
    ax_top.axis('off')

    # Right histogram: methylation
    ax_right = plt.subplot(gs[1, 1])
    sns.histplot(y=beta_seq, bins=beta_n_bins, ax=ax_right)
    ax_right.set_ylim(0, 1.001)
    ax_right.axis('off')

    # Colorbar
    divider = make_axes_locatable(ax_right)
    cax = divider.append_axes("right", size="15%", pad=0.05)
    img = ax_main.collections[0]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label("Fraction of CpGs (per coverage bin)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    cbar.outline.set_visible(False)

    if get_cb_params:
        vmin = final_matrix.min()
        vmax = final_matrix.max()
        return (fig, vmin, vmax)
    
    else:
        return (fig)



def plot_mae_map(test_samps,
                 samp_trueseq_map,
                 samp_predseq_map,
                 samp_covseq_map, 
                 beta_n_bins = 20, pal = "crest"):
    
    # Data Setup
    all_gt = []
    all_preds = []
    all_cov = []

    for samp in test_samps:
        all_gt.append(samp_trueseq_map[samp])
        all_preds.append(samp_predseq_map[samp])
        all_cov.append(samp_covseq_map[samp])

    ground_truth = np.concatenate(all_gt)
    inferred = np.concatenate(all_preds)
    coverage = np.concatenate(all_cov)

    assert np.all(ground_truth <= 1.0), "GT Beta values exceed 1"
    assert np.all(inferred <= 1.0), "Inferred Beta values exceed 1"
    assert np.all(coverage >= 1), "Coverage must be >= 1"

    # -----------------------------
    # Define bins
    # -----------------------------
    cov_bins = [1, 5, 10, 15, 20, 25, 30, np.inf]
    cov_bin_labels = [f"[{cov_bins[0]}, {cov_bins[1]}]"] + \
    [f"({cov_bins[i]}, {cov_bins[i+1]}]" for i in range(1, len(cov_bins)-2)]  + \
        [f"({cov_bins[-2]}, \u221E)"]


    beta_bins = np.linspace(0, 1.0, beta_n_bins + 1)
    beta_bins = np.round(beta_bins, 3) # avoid weird floating point issues
    beta_bin_labels = [f"[{beta_bins[0]:.2f}, {beta_bins[1]:.2f}]"] + \
        [f"({beta_bins[i]:.2f}, {beta_bins[i+1]:.2f}]" for i in range(1, beta_n_bins)]

    cov_bin_idx = get_binidx(bins = cov_bins, x = coverage)
    beta_bin_idx = get_binidx(bins = beta_bins, x = ground_truth)

    # -----------------------------
    # Compute MAE per bin
    # -----------------------------
    mae_matrix = np.full((beta_n_bins, len(cov_bins)-1), np.nan)

    for i in range(beta_n_bins):
        for j in range(len(cov_bins)-1):
            mask = (beta_bin_idx == i) & (cov_bin_idx == j)
            if np.any(mask):
                mae_matrix[i, j] = mean_absolute_error(ground_truth[mask], inferred[mask])

    # -----------------------------
    # Plotting
    # -----------------------------

    fig = plt.figure(figsize=(12, 10), dpi=300)
    
    sns.heatmap(
        mae_matrix,
        cmap=pal,  # or "viridis", "YlOrRd"
        xticklabels=cov_bin_labels,
        yticklabels=beta_bin_labels,
        cbar=True,
    )

    # Flip y-axis to go from low to high
    plt.gca().invert_yaxis()

    # Axis labels
    plt.xlabel("Coverage", fontsize=16)
    plt.ylabel("Ground Truth Betas", fontsize=16)
    # plt.title("Heatmap: Inferred vs. Ground Truth Methylation")
    plt.xticks(rotation=55, fontsize = 14)
    plt.yticks(rotation=0, fontsize = 14)

    return (fig)



"""
Example cpgMask creation

dataset = "gtex"
val_chr = chr_split["Validate"]
val_samps = fold_samps["Validate"]

cpgMask_map = {}
for chrom in val_chr:
    samp_cpgMask_map = {}
    cc_gt_df, _ = get_cc_gt(dataset, chrom)
    for samp in val_samps:
        cpg_mask = np.isnan(cc_gt_df.loc[:,samp].values) # wgbs missing = True
        samp_cpgMask_map[samp] = cpg_mask
    cpgMask_map[chrom] = samp_cpgMask_map

"""