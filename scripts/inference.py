import os
import yaml
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from collections import defaultdict
from aruna.models import DCAE_MSLICE
from aruna.model_utils import get_peObj
from torch.utils.data import DataLoader
from aruna.process_dataset import get_cc_gt
from aruna.data_utils import get_mslice_dataset
from aruna.evaluations import process_seq
from aruna.model_engine import valid_step_mslice
from aruna.evaluations import collate_mslices


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



def save_aruna_preds(preds_map, *, 
                     canonical_index, 
                     out_dir, chrom, 
                     verbose = True):
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
    chrom : str
        e.g. "chr21"
    verbose : bool
        Print saved paths if True
    """

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(preds_map)
    if len(df) != len(canonical_index):
        raise ValueError(
            f"Length mismatch: df={len(df)} canonical_index={len(canonical_index)}")
    # set CpG index directly
    df.index = pd.Index(canonical_index, name="start")
    fname = f"{chrom}.csv"
    fpath = os.path.join(out_dir, fname)
    df.to_csv(fpath, index=True)
    if verbose:
        print(f"Saved: {fpath}")


#### ---- INFERENCE CORE ---- ####

def run_mslice_inference(test_data, model_path, 
                         config_path, samples, 
                         nr, cpgMask_map):


    print("Running Aruna inference...")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = DCAE_MSLICE(config = config["model"])
    model.load_state_dict(torch.load(model_path))

    config["data"]["test_dataset"] = test_data
    config["exp"]["test_nr"] = nr
    chr = config["data"]["chrom"]
    batch_dim = config["model"]["batch_dim"]

    valData_obj = get_mslice_dataset(config, samples, mode = "infer")
    valloader = DataLoader(valData_obj, 
                        batch_size = batch_dim, 
                        shuffle = False, num_workers = 4) # shuffle False during inference
    print("#Batches in Validation (batch_dim={}): {}".format(batch_dim, len(valloader)))

    if config["model"]["posn_embed"]:
        pe_type = config["model"]["posn_embed"].split("_")[0]
        if  pe_type== "type2":
            embed_dim = model.embed_dim
        else:
            embed_dim = None
        pe_obj = get_peObj(pe_type = pe_type, 
                            num_cpg = config["data"]["num_cpgs"], 
                            embed_dim = embed_dim,
                            chrom = chr)
        # this is done basically to simplify batching of PE along with dataloader
        valData_obj.pe_obj = pe_obj

    device = config["model"]["device"]
    criterion = config["model"]["criterion"]
    model = model.to(device)
    if criterion == "mse":
        loss_fn = nn.MSELoss()

    # 4-step processing
    # 1: Run inference on incomplete sequences
    val_res = valid_step_mslice(model, valloader, 
                                loss_fn, device)
    # 2: combine patches (but use top patch per slice only), add a spp_idx
    all_collated_data = collate_mslices(val_res, 
                                        config["data"]["test_spp"], 
                                        chr)
    # 3: do rudimentary reduction of whole methylome while demarcating spp repeats
    chrom_gtspp_map, chrom_predspp_map = process_seq(all_collated_data, 
                                                    cpgMask_map, 
                                                    type_ = "nothing", 
                                                    slice = True)
    # collapse spp repeats by averaging
    chrom_gt_map = get_spp_collapsed_maps(chrom_gtspp_map, chr) # does nothing harmful [verified check]
    chrom_preds_map = get_spp_collapsed_maps(chrom_predspp_map, chr)
    
    samp_evalMask_map = {}
    for samp_data in all_collated_data[chr]:
        samp_name = samp_data[0]
        if samp_name not in samp_evalMask_map:
            eval_mask = samp_data[3] # identical for all spp, doesnt matter which repeat you choose
            samp_evalMask_map[samp_name] = eval_mask

    return(chrom_gt_map, chrom_preds_map, samp_evalMask_map)