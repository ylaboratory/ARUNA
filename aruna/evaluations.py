import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import mean_squared_error, r2_score

# for plotting calibration 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


CWD = os.getcwd()
metadata_dir = Path(__file__).resolve().parent.parent / "data" / "metadata"
hg38_all_df = pd.read_csv(os.path.join(metadata_dir, "hg38_cpg_py0idx.csv"))


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