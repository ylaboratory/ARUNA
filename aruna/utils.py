import os
import glob
import json
import logging
import numpy as np
from collections import Counter
from typing import List, Sequence, Tuple, Optional

CWD = os.getcwd()
modelData_dir = os.path.join(CWD, "model_data")

logger = logging.getLogger(__name__)


def get_splits(split_dir, exp_type):
    
    avail_splits = glob.glob(os.path.join(split_dir, "*.json"))
    for f in avail_splits:
        if f.split("/")[-1].split(".")[0] == exp_type:
            split_file = f
    with open(split_file, "rb") as f:
            split_dict = json.load(f)
    return(split_dict)


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



def add_validation_split(samples: Sequence[str], 
                         val_frac: float = 0.1, 
                         seed: Optional[int] = None,
                        ) -> Tuple[List[str], List[str]]:
    """
    Split sample IDs into (train, val) by randomly selecting val_frac for validation.
    """
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0,1). Got {val_frac}")

    samples = list(samples)
    if len(samples) == 0:
        return [], []

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(samples))

    n_val = max(1, int(round(len(samples) * val_frac)))
    val = [samples[i] for i in perm[:n_val]]
    train = [samples[i] for i in perm[n_val:]]
    return train, val