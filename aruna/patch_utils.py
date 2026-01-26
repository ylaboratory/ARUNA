def get_patch_data(samp_name, samp_chr_df, 
                   patch_idx, patch_refCpG_map):
    
    """
    Performs a basic "merge" operation.
    
    Assumes that samp_chr_df already has a canonical CpG set.
    Assumes patch_refCpG_map was made from the same reference genome as the canonical set.

    If supplied DF is from ground truth, NaNs represent Motomoto missing CpG.
    If supplied DF is from noise sim data, sim_mask can be obtained from info in the associated SimulatedMask directory.

    Returns: np.array of len = #CpGs in patch with associated sample's betas.
    """
    
    samp_patch_df = samp_chr_df.merge(patch_refCpG_map[patch_idx],
                                  how = "right", 
                                  right_index = True,
                                  left_index = True)
    patch_betas =  samp_patch_df.loc[:, samp_name].values
    
    return(patch_betas)