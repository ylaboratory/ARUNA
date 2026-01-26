import os
import yaml
import glob
import json
import time
import random
import logging
import itertools
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import Counter

CWD = os.getcwd()
modelData_dir = os.path.join(CWD, "model_data")

logger = logging.getLogger(__name__)



def get_config(config_name):

    """
    Searches for <config_name>.yaml in CWD/model_data/configs.
    Once found, looks for multiple specs for a parameter.
    If multiple specs are found, automatically returns 1 config dict per combination.
    This behavior mimics grid search.

    Always searches for multi params in "data" args.
    If "mpatch" is specified, also searches in "model" args.
    NOTE: The latter behavior is subject to change.


    Arguments
    -----
    config_name (str): Name for current configuration to search.

    Returns
    -----
    config (generator of dicts): each config is a dict mimicking the structure of the input YAML.
    
    """

    config_file = os.path.join(modelData_dir, "configs", config_name+".yaml")
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # detect if multiple values for a model param have been provided
    # TODO: doesnt work when grid over "s" and model 
    logger.info("Detecting multiple params for model field...")
    multi_params = {}
    for param,val in config["data"].items():
        if type(val) == list:
            param_group = "data"
            multi_params[param] = val
    if config["data"]["kind"] == "mpatch": # dcae is more flexible for mpatch setting
        for param, val in config["model"].items():
            if param in ["layers", "kernel_sizes", "out_channels", "strides"]:
                continue
            if type(val) == list:
                param_group = "model"
                multi_params[param] = val
    
    if len(multi_params) > 0:
        logger.info("Multiple values for {} detected!".format(list(multi_params.keys())))

        multiparam_order = list(multi_params.keys())
        multiparam_vals = [multi_params[p] for p in multiparam_order]
        # multiparam_combs = defaultdict(list)
        multiparam_combs = []

        for i in itertools.product(*multiparam_vals):
            multiparam_combs.append(i)
        logger.info("GridSearch space created! Yielding >1 config set.")

        for comb in multiparam_combs:
            proc_config = deepcopy(config)
            for p in range(len(multiparam_order)):
                # proc_config["model"][multiparam_order[p]] = comb[p]
                proc_config[param_group][multiparam_order[p]] = comb[p]
            yield(proc_config)
    else:
        logger.info("Multiple params for no model field detected!")
        yield(config)




def get_chrom_splits(split_dir, chrom_holdout):
    
    if chrom_holdout == "simple":
        chr_splitfile = os.path.join(split_dir, "simple_mulchrom.json")
    elif chrom_holdout == "holdout":
        chr_splitfile = os.path.join(split_dir, "holdout.json")

    with open(chr_splitfile, "rb") as f:
        chr_split = json.load(f)
    return(chr_split)



def get_sampCpgMask_map(data_df, samp_list):

    """
    data_df: should be GT data
    samp_list: should usually be list of testing or validation samples.
    """
    
    samp_cpgMask_map = {} # needed only for test samples
    for samp in data_df[samp_list].columns:
        # wgbs unobs CpGs = True
        cpg_mask = np.isnan(data_df.loc[:, samp].values) 
        samp_cpgMask_map[samp] = cpg_mask

    return (samp_cpgMask_map)



class OrderedCounter(Counter, dict):
    """
    Counter that remembers the order elements are first encountered.
    Recipe from python's official docs.
    """
    
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, dict(self))

    def __reduce__(self):
        return self.__class__, (dict(self),)



def validate_config(config, input_shape = None):
    """
    Validate experiment config.
    Raises ValueError if something is invalid.

    For MSlice setting;
    - Checks if CNN model will be valid.
    - Changes strides list of lists to be list of tuples.
    """

    # === Data checks ===
    if config["data"]["feat_type"] == "mvals":
        if config["model"]["last_activation"] != "identity":
            raise ValueError("feat_type 'mvals' requires last_activation='identity'")
        if config["data"]["nan_sub"] != 0:
            raise ValueError("feat_type 'mvals' requires nan_sub=0")

    if config["group"][3] == 2: #exp 2xx -> MSlice
        if config["data"]["chrom"] == "multiple":
            raise ValueError("Multiple chromosome training not yet supported for MSlice setting!")

    assert config["model"]["criterion"] in ["mse", "bce", "l1", "huber"], "Loss function not supported!"
    
    if config["model"]["criterion"] == "bce":
        if config["data"]["feat_type"] == "mvals":
            raise ValueError("M-values not compatible with BCE loss")


    # === Positional embedding / stem checks ===
    posn_embed = config["model"]["posn_embed"]
    stem_dict = config["model"]["stem_dict"]
    if posn_embed:
        if posn_embed.split("_")[0] == "type2":
            if not stem_dict:
                raise ValueError("Stem is required when Type2 PE is specified!")
        elif posn_embed.split("_")[0] == "type1":
            if stem_dict:
                raise ValueError("Stem not allowed with Type1 PE!")
    else:
        if stem_dict:
            raise ValueError("Stem not allowed without PE!")


    # === CNN shape validity checks for EXP 2XX only! ===
    if config["group"][3] == "2": #exp 2xx -> MSlice

        kernel_sizes = config["model"]["kernel_sizes"]
        config["model"]["strides"] = [tuple(s) for s in config["model"]["strides"]]  # yaml -> tuples
        strides = config["model"]["strides"]

        curr_h, curr_w = input_shape
        for i, (k, s) in enumerate(zip(kernel_sizes, strides)):
            # handle int vs tuple kernels
            if isinstance(k, int):
                k_h = k_w = k
            else:
                k_h, k_w = k

            s_h, s_w = s

            # SAME padding
            p_h, p_w = (k_h-1)//2, (k_w-1)//2

            out_h = (curr_h + 2*p_h - k_h) // s_h + 1
            out_w = (curr_w + 2*p_w - k_w) // s_w + 1

            if out_h <= 0 or out_w <= 0:
                raise ValueError(
                    f"Invalid CNN spec at layer {i}: "
                    f"in=({curr_h},{curr_w}), kernel=({k_h},{k_w}), stride=({s_h},{s_w}), "
                    f"padding=({p_h},{p_w}) → out=({out_h},{out_w})"
                )

            curr_h, curr_w = out_h, out_w

    logger.info("Config Validated!")
    return(config)



def sampname_to_alias(kfold_dict):

    """
    Because I forgot the data was made with aliases but the splits were made with sample names.
    """

    scratch_dir = "/srv/risotto.cs.rice.edu/scratch/js228"
    loyfer_meta_path = os.path.join(scratch_dir, "temp/loyfer_raw/metadata")
    samples_meta_df = pd.read_csv(os.path.join(loyfer_meta_path, "samples.csv"), sep = ",")

    new_kfold = {}
    for fold, data in kfold_dict.items():
        temp_dict = {}
        for key_name in data.keys():
            old_names = data[key_name]
            new_names = [samples_meta_df[samples_meta_df.loc[:, "accession_id"] == i]["alias"].values[0] for i in old_names]

            temp_dict[key_name] = new_names
        new_kfold[fold] = temp_dict
        
    return(new_kfold)


def add_validation_split(kfold_dict, val_frac=0.1, seed=42):
    """
    Add a 10% validation split (randomly sampled) within each fold's Train set.
    Meant for use with atlas experiments where an explicit validation is not specified.
    This provides a validation set per fold, for use with early stopping and general performance monitoring.
    
    Parameters
    ----------
    kfold_dict : dict
        Nested dict like {'0': {'Train': [...], 'Test': [...]}, '1': {...}, ...}
    val_frac : float, default=0.1
        Fraction of Train samples to use for validation
    seed : int
        Random seed for reproducibility
    """
    random.seed(seed)
    fold_samps = {}

    for fold, data in kfold_dict.items():
        train_all = data["Train"]
        n_val = max(1, int(len(train_all) * val_frac))
        val_subset = random.sample(train_all, n_val)
        train_subset = [x for x in train_all if x not in val_subset]

        fold_samps[fold] = {
            "Train": train_subset,
            "Validate": val_subset,
        }
        logger.info(f"Fold {fold}: Train={len(train_subset)}, Validate={len(val_subset)}")

    return fold_samps
