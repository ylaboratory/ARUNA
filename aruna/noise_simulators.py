import os
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng()
noise_metadata_dir = Path(__file__).resolve().parent.parent / "data" / "metadata"



def get_sim_mask(noise_regime, chrom, hg38_chrcpg_df): # get simulated missing mask

    """
    Works purely using noise_regime, reference CpG set and Noise Metadata (for RRBS).
    No sample-specific info needed.
    """    
    
    all_ref_chr_cpg = hg38_chrcpg_df.index.values

    # MCAR Regimes
    if noise_regime.split("_")[0] == "mcar":
        miss_ratio = int(noise_regime.split("_")[1])/100
        sim_mask = rng.uniform(size = len(all_ref_chr_cpg)) < miss_ratio # True = Missing
    
    # RRBS_SIM Regime
    elif noise_regime == "rrbs_sim":
        rrbs_dir = "rrbs_chr_pobs"
        file_suffix = "_pobs"
        pobs_filepath = os.path.join(noise_metadata_dir, 
                                     rrbs_dir, "rrbs_" + chrom + file_suffix + ".tsv")
        pobs_df = pd.read_csv(pobs_filepath, sep = "\t", header = 0)
        temp_df = hg38_chrcpg_df.merge(pobs_df, on = "start", how = "left")
        # ASSUMPTION: CpGs in Reference, Not in RRBS metadata, assign P(Obs) = 0 in RRBS
        temp_df["prob_obs"] = temp_df["prob_obs"].fillna(value = 0)
        sim_mask = rng.binomial(np.ones(temp_df.shape[0], 
                                              dtype = "int"), 
                                              temp_df.loc[:, "prob_obs"])
        sim_mask = ~np.array(sim_mask, dtype = bool) # True = Missing

    else:
        raise ValueError("Invalid Noise Regime specified!")


    return(sim_mask)