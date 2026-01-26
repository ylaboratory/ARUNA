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