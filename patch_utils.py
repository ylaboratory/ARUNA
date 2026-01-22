import math
import numpy as np
import torch



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
    # samp_patch_df = samp_chr_df.merge(patch_refCpG_map[patch_idx],
    #                                   how = "right", on = "start")
    
    # 16x speedup when explicitly providing index (CpG start posns) as merge key
    samp_patch_df = samp_chr_df.merge(patch_refCpG_map[patch_idx],
                                  how = "right", 
                                  right_index = True,
                                  left_index = True)
    patch_betas =  samp_patch_df.loc[:, samp_name].values
    
    return(patch_betas)



def get_patch_data_legacy(samp_chr_df, p, patch_refCpG_map):
# def get_patch_data(samp_chr_df, p, chr_patch_refCpG_map, cpg_mask = 0):

    """
    Returns
    -------
    patch_data (np.array, float): Beta values at observed CpGs in sample patch.
    patch_cpg_mask (np.array, bool): Binary values (True = Missing) for missing CpGs in sample patch.
                                     Missing relative to a reference genome in the Bioinfo pipeline.
    patch_cpgs (np.array, int): Zero-indexed Start positions of ALL CpGs in patch.
                               Invariant to sample since this is the reference set.
                               Positions from the Bismark merged_cpg pipeline with default params.
    """
    
    samp_patch_cpg_df = samp_chr_df.merge(patch_refCpG_map[p],
                                          how = "right", on = "start")
    samp_patch_cpg_df.drop(columns = ["seqname_x", "seqname_y"], inplace = True)
    
    patch_data = samp_patch_cpg_df.loc[:, "beta"].values
    patch_cpg_mask = np.isnan(patch_data)
    patch_cpg_starts = np.array(samp_patch_cpg_df.index)
    
    return(patch_data, patch_cpg_mask, patch_cpg_starts)


# LIKELY OBSOLETE CODE (std patch v1/2/3 now implemented in data engine as an on-the-fly transform)
# TODO: Modify to use patch genomic span (?) -> necessary? when easily inferred
def make_std_patch_v1(s, p, patch_beta_vec, 
                      patch_refCpG_map, nonCpg_mask = -1):
    
    """
    CpG = Beta Values, Non-CpG = Mask
    NOTE: Modified to NOT verify if patch num_cpg > 0 since this is to be checked in the outer loop.

    # patch_noncpg_mask can be directly used, True for positions that are not CpGs
    """
    cpg_posns = np.array(patch_refCpG_map[p].index)
    p_lower = (p-1)*s
    p_cpg_idx = cpg_posns - p_lower

    if nonCpg_mask == -1:
        std_patch = -np.ones(s)
    else:
        raise NotImplementedError("nonCpG_mask other than -1 not supported yet!")

    std_patch[p_cpg_idx] = patch_beta_vec
    patch_noncpg_mask = ~np.array([(i in set(p_cpg_idx)) for i in range(s)]) # True for Non-CpG

    return(std_patch, patch_noncpg_mask)



def make_std_patch_v2(s, p, patch_beta_vec, 
                      patch_refCpG_map, nonCpg_mask = -1):

    """
    NOTE: Sequence Vector does NOT change with data. Property of patch.
    
    In Vector 1: CpG = 1, Non-CpG = 0.
    In Vector 2: CpG = Beta Values, Non-CpG = Mask
    concat([CpG presence binary vector], [Beta vector])
    E.g. [1,0,0,0,0,0,1] + [0.3,-1,-1,-1,-1,-1,0.9]
    Final: [1,0,0,0,0,0,1, 0.3,-1,-1,-1,-1,-1,0.9]

    # patch_noncpg_mask is only for the latter half of the vector (patch[5000:]), True for positions that are not CpGs
    """

    cpg_posns = np.array(patch_refCpG_map[p].index)
    p_lower = (p-1)*s
    p_cpg_idx = cpg_posns - p_lower

    seq_vec = np.zeros(s)
    seq_vec[p_cpg_idx] = 1 # mark cytosine positions

    if nonCpg_mask == -1:
        beta_vec = -np.ones(s)
    else:
        raise NotImplementedError("nonCpG_mask other than -1 not supported yet!")

    beta_vec[p_cpg_idx] = patch_beta_vec

    std_patch = np.concatenate([seq_vec, beta_vec])
    patch_noncpg_mask = ~np.array([(i in set(p_cpg_idx)) for i in range(s)])
    return(std_patch, patch_noncpg_mask)



def make_std_patch_v3(s, p, patch_beta_vec, 
                      patch_refCpG_map, nonCpg_mask = -1):

    """
    In Vector 1: CpG = 1, Non-CpG = 0.
    In Vector 2: CpG = Beta Values, Non-CpG = Mask
    axis-concat([CpG presence binary vector], [Beta vector])
    E.g. [1,0,0,0,0,0,1] + [0.3,-1,-1,-1,-1,-1,0.9]
    Final: [[1,0,0,0,0,0,1], [0.3,-1,-1,-1,-1,-1,0.9]]

    # patch_noncpg_mask is only for the latter vector (dim1), True for positions that are not CpGs
    """
    cpg_posns = np.array(patch_refCpG_map[p].index)
    p_lower = (p-1)*s
    p_cpg_idx = cpg_posns - p_lower

    seq_vec = np.zeros(s)
    seq_vec[p_cpg_idx] = 1 # mark cytosine positions

    if nonCpg_mask == -1:
        beta_vec = -np.ones(s)
    else:
        raise NotImplementedError

    beta_vec[p_cpg_idx] = patch_beta_vec

    std_patch = np.stack([seq_vec, beta_vec], axis = 0)
    patch_noncpg_mask = ~np.array([(i in set(p_cpg_idx)) for i in range(s)])
    return(std_patch, patch_noncpg_mask)
# OBSOLETE CODE ENDS



def make_patch_stdbetas(patch_beta_vec, p, 
                        s, patch_noncpgMask_map, 
                        nonCpg_mask = -1):
    
    p_cpg_idx = np.nonzero(patch_noncpgMask_map[p])[0]
    if nonCpg_mask == -1:
        beta_vec = -np.ones(s)
    else:
        raise NotImplementedError
    beta_vec[p_cpg_idx] = patch_beta_vec
    return(beta_vec)



def get_nn(patch_idx, n_dist, patch_refCpG_map):

    """
    Creates a symmetric window of n_dist number of patches around the queries patch_idx.
    Edge cases include when symmetry is not possible (for queried patches near the ends of the chromosome).
    NOTE: Patch idx starts from 1.
    NOTE: For ODD n_dist, num upstream patches (towards 5') are greater by 1 than downstream patches.
    TODO: patch_refNumCpg_df can be derived from chr_patch_refCpG_map (redundant info?)
    """

    tot_patches = len(patch_refCpG_map)
    all_pids = list(patch_refCpG_map.keys())

    assert patch_idx < tot_patches, "Queried patch can not be greater than total number of available patches!"
    assert n_dist < tot_patches, "N_dist can not be greater than the total number of patches!"

    ss = int(math.ceil(n_dist/2)) # symmetric size
    neighbor_patch_idx = []
    patch_idx -= 1 # needed to use patch_idx as a python index (which starts from 0)


    # for below code, lf = left_flank and rf = right_flank of neighbors surrounding queried patch
    # Case 1: Query patch near left edge of chromsome
    if patch_idx < ss:

        lf = n_dist - abs(patch_idx-n_dist)
        lf = int(min(lf, ss))
        rf = n_dist - lf

        if lf == 0: # queried patch is the leftmost
            # add right flank
            neighbor_patch_idx.extend(all_pids[patch_idx+1:patch_idx+rf+1])

        else:
            # add left flank
            neighbor_patch_idx.extend(all_pids[patch_idx-lf:patch_idx])
            # add right flank
            neighbor_patch_idx.extend(all_pids[patch_idx+1:patch_idx+rf+1])


    # Case 2: Query patch somewhere in the middle (symmetric neighborhood possible)
    elif patch_idx + ss < tot_patches:
        
        lf = ss
        rf = n_dist - ss
        # add left flank
        neighbor_patch_idx.extend(all_pids[patch_idx-lf:patch_idx])
        # add right flank
        neighbor_patch_idx.extend(all_pids[patch_idx+1:patch_idx+rf+1])


    # Case 3: Query patch near right edge of chromsome
    else:
        rf = tot_patches - (patch_idx+1)
        lf = n_dist - rf

        if rf == 0:
            # add right flank
            neighbor_patch_idx.extend(all_pids[patch_idx-lf:patch_idx])

        else:
            # add left flank
            neighbor_patch_idx.extend(all_pids[patch_idx-lf:patch_idx])
            # add right flank
            neighbor_patch_idx.extend(all_pids[patch_idx+1:])

    return(neighbor_patch_idx)



# replaced by eval_utils.collate_data
# def get_samp_evalData(testloader, curr_samp, 
#                       num_patches, all_batch_preds):

#     """
#     Collects all patches for curr_samp.
#     Useful to compute sample-wise eval metrics.
#     Quite slow, a generator-based pattern might be better to avoid re-iterating over past sample's patches.

#     Len of all_batch_preds = num batches.
#     """
    
#     samp_numPatches = 0
#     samp_names = []
#     samp_gt = []
#     samp_preds = []
#     samp_evalMask_batches = []

#     batches_skipped = 0

#     while samp_numPatches < num_patches:

#         for batch_idx, batch_data in enumerate(testloader):       
                
#             batch_samps = batch_data["samp"]
#             sample_mask = [i==curr_samp for i in batch_samps]
            
#             samp_names.extend([batch_samps[i] for i in range(len(batch_samps)) if sample_mask[i]])

#             if np.sum(sample_mask) == 0: # sample data not found
#                 batches_skipped += 1
#                 continue

#             batch_gt = batch_data["true"][sample_mask][:,1,:]
#             batch_preds = all_batch_preds[batch_idx][sample_mask][:,1,:]
#             batch_noncpgMask = batch_data["noncpg_mask"][sample_mask]
            
#             samp_gt.append(batch_gt[~batch_noncpgMask].numpy())
#             samp_preds.append(batch_preds[~batch_noncpgMask])

#             # process batch_evalMask a bit to remove padding and convert to bool
#             batch_evalMask = batch_data["eval_mask"][sample_mask]
#             batch_evalMask = batch_evalMask[~torch.isnan(batch_evalMask)].type(torch.bool)
#             samp_evalMask_batches.append(batch_evalMask.numpy())        

#             samp_numPatches += np.sum(sample_mask)


#         #if samp_numPatches == num_patches:    

#         full_seq = np.hstack(samp_gt)
#         pred_seq = np.hstack(samp_preds)
#         eval_mask = np.hstack(samp_evalMask_batches)

#         # samp_names returned for sanity
#         return(set(samp_names), full_seq, pred_seq, eval_mask)