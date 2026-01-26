import os
import csv
import torch
import numpy as np
from model_utils import mvalTobeta
from collections import defaultdict
from process_dataset import get_cc_gt
from sklearn.metrics import mean_squared_error, r2_score
from evaluations import process_seq, eval_sampwise_quant



def log_quant_metrics(fold, epoch, all_collated_data, feat_type):
    
    """
    Designed for wandb logging.

    Logs the following in a chromosome-averaged manner:
    1. Sample average MSE 
    2. Sample average R2 
    3. GT and Pred avg. beta at CpGs [average over samples]
    4. GT and Pred variance over betas at CpGs [variance across a sample's methylome]
    5. GT and Pred variance over betas at CpGs [variance over a CpG across the dataset]

    Since the sequences are not processed with process_seq, eval_mask is used only for mse and r2 computation.
    
    NOTE: The mean and variances do not account for a missing mask, so are affected by choice of nan_sub.
    """
    
    log_dict = {"fold{}-epoch".format(fold): epoch}

    # collect data for all chromosomes and log average metrics
    for chrom in all_collated_data.keys():

        all_gt = []
        all_pred = []
        tot_mse = 0.0
        tot_r2 = 0.0

        chrom_res = all_collated_data[chrom]
        
        for samp_data in chrom_res:
            if feat_type == "mvals":
                gt = mvalTobeta(samp_data[1])
                pred = mvalTobeta(samp_data[2])
            else:
                gt = samp_data[1]
                pred = samp_data[2]

            eval_mask = samp_data[3]

            all_gt.append(gt)
            all_pred.append(pred)
            tot_mse +=  mean_squared_error(gt[eval_mask], pred[eval_mask])
            tot_r2 += r2_score(gt[eval_mask], pred[eval_mask])
        
        # logging 1-2
        avg_mse = tot_mse/len(chrom_res)
        avg_r2 = tot_r2/len(chrom_res)

        # logging 3-5
        gt = np.stack(all_gt)
        pred = np.stack(all_pred)

        refMu = np.mean(gt)
        predMu = np.mean(pred)

        refVar_samp = np.mean(np.var(gt, axis = 1))
        predVar_samp = np.mean(np.var(pred, axis = 1))

        refVar_cpg = np.mean(np.var(gt, axis = 0))
        predVar_cpg = np.mean(np.var(pred, axis = 0))

        log_dict.update({"fold{}-{}-mse".format(fold, chrom): avg_mse,
                         "fold{}-{}-r2".format(fold, chrom): avg_r2, 
                         "fold{}-{}-RefMu".format(fold, chrom): refMu,
                         "fold{}-{}-PredMu".format(fold, chrom): predMu,
                         "fold{}_{}-RefVarSamp".format(fold, chrom): refVar_samp,
                         "fold{}_{}-PredVarSamp".format(fold, chrom): predVar_samp,
                         "fold{}-{}-RefVarCpg".format(fold, chrom): refVar_cpg,
                         "fold{}-{}-PredVarCpg".format(fold, chrom): predVar_cpg})
    
    return(log_dict)



def log_quant_metrics_slice(fold, epoch, all_collated_data, feat_type):

    """
    Minor difference in functionality since each sample has #spp repeats.
    The repeats are averaged to provide the final prediction.
    The (averaged) CpG-wise variance within repeats is also tracked and again averaged across samples. This should be close to 0 as opposed to PredVarCpg which should approximate the reference value.
    """
    
    log_dict = {"fold{}-epoch".format(fold): epoch}

    # collect data for all chromosomes and log average metrics
    for chrom in all_collated_data.keys():

        chrom_res = all_collated_data[chrom]

        all_gt = []
        all_pred = []
        samp_pred_map = defaultdict(list) # list len would be test spp
        tot_mse = 0.0
        tot_r2 = 0.0

        for samp_data in chrom_res:
            samp_name = samp_data[0]
            if feat_type == "mvals":
                gt = mvalTobeta(samp_data[1])
                pred = mvalTobeta(samp_data[2])
            else:
                gt = samp_data[1]
                pred = samp_data[2]
            eval_mask = samp_data[3]

            if len(samp_pred_map[samp_name]) == 0:
                all_gt.append(gt)

            samp_pred_map[samp_name].append(pred)
             
        intraSampVar_cpg = 0 # cpg-wise variance across repeats for the same sample
        for samp_name in samp_pred_map:
            samp_preds = np.stack(samp_pred_map[samp_name], axis = 0)

            # store cpg-variance across #spp repeats per sample
            intraSampVar_cpg += np.mean(np.var(samp_preds, axis = 0))
            
            # generate final pred for each sample as mean of #spp predictions
            pred = np.mean(samp_preds, axis = 0)
            
            tot_mse +=  mean_squared_error(gt[eval_mask], pred[eval_mask])
            tot_r2 += r2_score(gt[eval_mask], pred[eval_mask])

            all_pred.append(pred)
        
        # logging 1-3
        avg_intraSampVar_cpg = intraSampVar_cpg/len(samp_pred_map.keys())
        avg_mse = tot_mse/len(samp_pred_map.keys())
        avg_r2 = tot_r2/len(samp_pred_map.keys())

        # logging 4-6
        gt = np.stack(all_gt)
        pred = np.stack(all_pred)

        refMu = np.mean(gt)
        predMu = np.mean(pred)

        refVar_samp = np.mean(np.var(gt, axis = 1))
        predVar_samp = np.mean(np.var(pred, axis = 1))

        refVar_cpg = np.mean(np.var(gt, axis = 0))
        predVar_cpg = np.mean(np.var(pred, axis = 0))

        log_dict.update({"fold{}-{}-IntraVarCpG".format(fold, chrom): avg_intraSampVar_cpg,
                         "fold{}-{}-mse".format(fold, chrom): avg_mse,
                         "fold{}-{}-r2".format(fold, chrom): avg_r2, 
                         "fold{}-{}-RefMu".format(fold, chrom): refMu,
                         "fold{}-{}-PredMu".format(fold, chrom): predMu,
                         "fold{}_{}-RefVarSamp".format(fold, chrom): refVar_samp,
                         "fold{}_{}-PredVarSamp".format(fold, chrom): predVar_samp,
                         "fold{}-{}-RefVarCpg".format(fold, chrom): refVar_cpg,
                         "fold{}-{}-PredVarCpg".format(fold, chrom): predVar_cpg})
    
    return(log_dict)



def log_to_csv(run, fold, dataset, 
               chr_list, samp_list, 
               all_collated_data, res_file, 
               slice = False):
    """
    Logs metrics using the "standard" setting to results csv once per fold
    """
    fieldnames = ["run_name", "chrom", "fold", 
                  "sampwise_mse", "sampwise_r2", 
                  "foldavg_mse", "foldavg_r2"]
    file_exists = os.path.isfile(res_file)
    with open(res_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        cpgMask_map = {}
        for chrom in chr_list:
            samp_cpgMask_map = {}
            cc_gt_df, _ = get_cc_gt(dataset, chrom)
            for samp in samp_list:
                cpg_mask = np.isnan(cc_gt_df.loc[:,samp].values) # wgbs missing = True
                samp_cpgMask_map[samp] = cpg_mask
            cpgMask_map[chrom] = samp_cpgMask_map

        chrom_gt_map, chrom_pred_map = process_seq(all_collated_data, 
                                                    cpgMask_map, 
                                                    type_ = "standard")
    
        for chrom in chrom_gt_map.keys():
        
            samp_gt_map = chrom_gt_map[chrom]
            samp_pred_map = chrom_pred_map[chrom]

            if not slice:
                samp_mse_map, samp_r2_map = eval_sampwise_quant(samp_gt_map, 
                                                                samp_pred_map)
            
            if slice:
                samp_spp_preds_map = defaultdict(list) # collect all inferences per sample
                new_samp_pred_map = defaultdict(list) # store mean of spp inferences
                
                for k in samp_pred_map.keys(): # keys are samp_sppIdx
                    samp_name = k.split("_")[0]
                    samp_spp_preds_map[samp_name].append(np.array(samp_pred_map[k]))
                
                for samp_name in samp_spp_preds_map.keys():
                    final_pred = np.mean(np.stack(
                                                samp_spp_preds_map[samp_name], 
                                                axis = 0), 
                                        axis = 0)
                    new_samp_pred_map[samp_name] = final_pred


                samp_mse_map, samp_r2_map = eval_sampwise_quant(samp_gt_map, 
                                                                new_samp_pred_map)
            
            row = {"run_name": run, 
                "chrom": chrom, 
                "fold": fold, 
                "sampwise_mse": list(samp_mse_map.values()), 
                "sampwise_r2": list(samp_r2_map.values()), 
                "foldavg_mse": np.mean(list(samp_mse_map.values())), 
                "foldavg_r2": np.mean(list(samp_r2_map.values()))}
            
            writer.writerow(row)
    
    return



def log_weights_wandb(arch, model):

    if arch != "dae":
        raise AssertionError("Model Architecture not supported for logging weight norms!")

    log_dict = {}

    for i, seq_layer in enumerate(model.children()):

        if i == 0: # Encoder Sequential Block
            lin_layer_idx = [idx for idx in range(0,11,2)] # TODO: Remove hardcoding
            for idx in lin_layer_idx:
                layer_weight = seq_layer[idx].weight
                layer_bias = seq_layer[idx].bias.reshape(-1,1)
                weight_norm = torch.linalg.matrix_norm(layer_weight)
                bias_norm = torch.linalg.matrix_norm(layer_bias)

                log_dict["enc_weights"+str(idx)] = weight_norm.item()
                log_dict["enc_bias"+str(idx)] = bias_norm.item()
    
        if i == 1: # Decoder Sequential Block
            lin_layer_idx = [idx for idx in range(0,11,2)] # TODO: Remove hardcoding
            for idx in lin_layer_idx:
                layer_weight = seq_layer[idx].weight
                layer_bias = seq_layer[idx].bias.reshape(-1,1)
                weight_norm = torch.linalg.matrix_norm(layer_weight)
                bias_norm = torch.linalg.matrix_norm(layer_bias)
                log_dict["dec_weights"+str(idx)] = weight_norm.item()
                log_dict["dec_bias"+str(idx)] = bias_norm.item()

    return(log_dict)



# wandb logging doesn't work with nested dicts
# more suited for plotting with pandas
def log_weights(arch, model, log_dict):

    if arch != "dae":
        raise AssertionError("Model Architecture not supported for logging weight norms!")

    if log_dict == None:
        log_dict = {"enc_weights": defaultdict(list),
                    "enc_bias": defaultdict(list),
                    "dec_weights": defaultdict(list),
                    "dec_bias": defaultdict(list)}

    for i, seq_layer in enumerate(model.children()):

        if i == 0: # Encoder Sequential Block
            lin_layer_idx = [idx for idx in range(0,11,2)] # TODO: Remove hardcoding
            for idx in lin_layer_idx:
                layer_weight = seq_layer[idx].weight
                layer_bias = seq_layer[idx].bias.reshape(-1,1)
                weight_norm = torch.linalg.matrix_norm(layer_weight)
                bias_norm = torch.linalg.matrix_norm(layer_bias)

                log_dict["enc_weights"][idx].append(weight_norm.item())
                log_dict["enc_bias"][idx].append(bias_norm.item())
    
        if i == 1: # Decoder Sequential Block
            lin_layer_idx = [idx for idx in range(0,11,2)] # TODO: Remove hardcoding
            for idx in lin_layer_idx:
                layer_weight = seq_layer[idx].weight
                layer_bias = seq_layer[idx].bias.reshape(-1,1)
                weight_norm = torch.linalg.matrix_norm(layer_weight)
                bias_norm = torch.linalg.matrix_norm(layer_bias)
                log_dict["dec_weights"][idx].append(weight_norm.item())
                log_dict["dec_bias"][idx].append(bias_norm.item())

    return(log_dict)