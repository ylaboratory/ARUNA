import math
import itertools
import numpy as np

import torch
from torch.utils.data import Dataset

from patch_utils import make_patch_stdbetas
# from model_utils import PositionEmbedding

import logging
logger = logging.getLogger(__name__)



class MSliceCentricDataset(Dataset):

    def __init__(self, samp_list, chrom, num_cpgs, 
                 gt_patches_map, nr_patches_map, 
                 simMask_patches_map, cov_patches_map = None, 
                 sps = 2, spp = 10, 
                 posn_embed = None, 
                 transform = None):
        
        """
        CHANGELOG (NOV 2025): read depth functionality was temporarily removed.
        
        Constrain: A slice contains only 1 PID.
                   - Posn embedding repeated in channel dim downstream.
                   - The indexed sample-pid patch will always be top, with random samplings in dim 2 (Height-wise).
                     
        sps (int): samples per slice, the "width" of a slice. Typically 2 or 3.
        spp (int): slices per patch, number of randomly sampled combinations per patch id. 
                   Goal is to have "enough" to provide model with both similar and dissimilar tissue/cell types.
                   Typically > 5.
        
        ~~Total number of elements returned by dataloader = spp * #Patches~~
        REVISED: Total number of elements returned by dataloader = spp * #Patches * #Samples
        REVISED: Explicitly only supports single chromosome.
        
        2-level randomness: Randomly sample a patch (as before), but also, randomly sample #sps-1 samples to make a slice.
            - Randomness over patches performed by dataloader
            - Randomness over samples performed here.
        
        TODO: [MULCHROM NOT SUPPORTED] Chromosome also needs to be fixed for a given PID to support mulchrom training. This is likely not computationally feasible for Slices.

        NOTE: Since randomness is inherent in the slice-construction; a dataset (of slices) is not entirely reconstructable, 
        unless a random seed is used. Even then, indexing a certain element multiple times will result in multiple slices, 
        and the same sequence will only be possible by reinitializing the dataset object.
        NOTE: Current sample-patch always row 0 of dim1.
        """

        assert math.comb(len(samp_list), sps) >= spp, "{} slices per patch can not be sampled - too few combinations!".format(spp)

        self.samples = samp_list
        self.chrom = chrom
        self.num_cpgs = num_cpgs
        self.sps = sps
        self.spp = spp
        self.pe = posn_embed
        self.transform = transform

        self.pids = list([i[0] for i in gt_patches_map[samp_list[0]]])
        sample_pids = list(itertools.product(self.samples, self.pids))
        self.pid_pool = [i for i in sample_pids for _ in range(self.spp)]

        self.gt_patches_map = gt_patches_map
        self.nr_patches_map = nr_patches_map
        self.simMask_patches_map = simMask_patches_map
        # self.cov_patches_map = cov_patches_map

        if self.pe:
            self.pe_obj = None

        
    def __len__(self):

        return(len(self.pid_pool))


    def __getitem__(self, idx):
        
        slice_samples, pid = [self.pid_pool[idx][0]], self.pid_pool[idx][1]
        slice_samples.extend(list(np.random.choice(self.samples, self.sps-1)))
        
        if self.pe:
            posn_vec = self.pe_obj.get_pe(self.chrom, pid) # sample agnostic
        else:
            posn_vec = False

        gt_slice = []
        nr_slice = []
        evalMask_slice = []
        posnVec_slice = []
        # cov_slice = []

        for samp in slice_samples:
            
            gt_patch = self.gt_patches_map[samp][pid][1]
            nr_patch = self.nr_patches_map[samp][pid][1]
            cpg_mask = np.isnan(gt_patch)
            sim_mask = self.simMask_patches_map[samp][pid][1]
            eval_mask = (~cpg_mask & sim_mask)
            # cov_patch = self.cov_patches_map[samp][pid][1]

            if gt_patch.shape[0] < self.num_cpgs: # for a sample's last patch
                gt_patch = pad_msequence(gt_patch, self.num_cpgs, pad_val = "zeros")
                nr_patch = pad_msequence(nr_patch, self.num_cpgs, pad_val = "zeros")
                eval_mask = pad_msequence(eval_mask, self.num_cpgs, pad_val = "false")
                # cov_patch = pad_msequence(eval_mask, self.num_cpgs, pad_val = "zeros")

            gt_slice.append(gt_patch)
            nr_slice.append(nr_patch)
            evalMask_slice.append(eval_mask)
            posnVec_slice.append(posn_vec) # repeated values since slice is constrained
            # cov_slice.append(cov_patch)
        
        gt_slice = np.stack(gt_slice, axis = 0)
        nr_slice = np.stack(nr_slice, axis = 0)
        evalMask_slice = np.stack(evalMask_slice, axis = 0)
        posnVec_slice = np.stack(posnVec_slice, axis = 0) # works for False case too
        # cov_slice = np.stack(cov_slice, axis = 0)

        # slice_data = {"samps": slice_samples,
        #               "patch_id": pid,
        #               "true": gt_slice,
        #               "noisy": nr_slice,
        #               "eval_mask": evalMask_slice,
        #               "posn_vec": posnVec_slice,
        #               "read_depth": cov_slice}

        slice_data = {"samps": slice_samples,
                      "patch_id": pid,
                      "true": gt_slice,
                      "noisy": nr_slice,
                      "eval_mask": evalMask_slice,
                      "posn_vec": posnVec_slice}
        
        if self.transform:
            if type(self.transform) == list:
                for t in self.transform:
                    slice_data = t(slice_data)
            else:
                slice_data = self.transform(slice_data)

        return(slice_data)



class MSliceCentricDatasetReduced(Dataset):

    """
    Useful for loading actual RRBS datasets. 
    """

    def __init__(self, samp_list, 
                 chrom, num_cpgs, 
                 gt_patches_map,
                 sps = 2, spp = 10, 
                 posn_embed = None, 
                 transform = None):

        assert math.comb(len(samp_list), sps) >= spp, "{} slices per patch can not be sampled - too few combinations!".format(spp)

        self.samples = samp_list
        self.chrom = chrom
        self.num_cpgs = num_cpgs
        self.sps = sps
        self.spp = spp
        self.pe = posn_embed
        self.transform = transform

        self.pids = list([i[0] for i in gt_patches_map[samp_list[0]]])
        sample_pids = list(itertools.product(self.samples, self.pids))
        self.pid_pool = [i for i in sample_pids for _ in range(self.spp)]

        self.gt_patches_map = gt_patches_map
        
        if self.pe:
            self.pe_obj = None

        
    def __len__(self):
        return(len(self.pid_pool))


    def __getitem__(self, idx):
        slice_samples, pid = [self.pid_pool[idx][0]], self.pid_pool[idx][1]
        slice_samples.extend(list(np.random.choice(self.samples, self.sps-1)))
        
        if self.pe:
            posn_vec = self.pe_obj.get_pe(self.chrom, pid) # sample agnostic
        else:
            posn_vec = False

        gt_slice = []
        posnVec_slice = []
    
        for samp in slice_samples:
            gt_patch = self.gt_patches_map[samp][pid][1]
            if gt_patch.shape[0] < self.num_cpgs: # for a sample's last patch
                gt_patch = pad_msequence(gt_patch, self.num_cpgs, pad_val = "zeros")
                
            gt_slice.append(gt_patch)
            posnVec_slice.append(posn_vec)
        
        gt_slice = np.stack(gt_slice, axis = 0)
        posnVec_slice = np.stack(posnVec_slice, axis = 0) 
       
        slice_data = {"samps": slice_samples,
                      "patch_id": pid,
                      "true": gt_slice,
                      "posn_vec": posnVec_slice}
        
        if self.transform:
            if type(self.transform) == list:
                for t in self.transform:
                    slice_data = t(slice_data)
            else:
                slice_data = self.transform(slice_data)

        return(slice_data)



def pad_sequence(raw_seq, out_shape, pad_val = np.nan):
    assert raw_seq.shape[0] <= out_shape, "Sequence shape mismatch while padding!"
    padded_seq = np.zeros(out_shape)
    padded_seq[:len(raw_seq)] = raw_seq
    padded_seq[len(raw_seq):] = pad_val
    return(padded_seq)



def pad_msequence(raw_sequence, out_shape, pad_val = "zeros"):

    """
    Different implementation from GPatch.
    """
    if pad_val == "zeros":
        padded_seq = np.zeros(out_shape)
       
    elif pad_val == "false":
        padded_seq = np.zeros(out_shape, dtype = bool)

    padded_seq[:len(raw_sequence)] = raw_sequence
    return(padded_seq)



class StandardRepr():

    def __init__(self, patch_noncpgMask_map,
                 repr_ = "v3"):

        self.patch_noncpgMask_map = patch_noncpgMask_map # has binary seq vec
        if repr_ != "v3":
            raise NotImplementedError("Patch representation only supported for std_v3!")
        self.repr_ = repr_

    def __call__(self, patch_data):
        
        pid = patch_data["patch_id"]
        gt_std_patch = patch_data["true"]
        nr_std_patch = patch_data["noisy"]
        
        seq_vec = self.patch_noncpgMask_map[pid]
        gt_stdv3_patch = np.stack([seq_vec, gt_std_patch], axis = 0)
        nr_stdv3_patch = np.stack([seq_vec, nr_std_patch], axis = 0)

        patch_data["true"] = gt_stdv3_patch
        patch_data["noisy"] = nr_stdv3_patch

        return(patch_data)
    


class ReplaceNaN():

    def __init__(self, replace_val = 0):
        
        """

        Replaces all Patch NaNs with replace_val.
        replace_val needs to be calclulated elsehere and can be:
        1. Integer Indicator
        2. Dataset Mean
        3. Dataset Median
        where dataset is typically all patches in the training data.
        We take the dataset statistics to account for MNAR effects and ensure unbiasedness.

        Handles coverage/read_depth info separately.
        - When available, missing coverage data is always replaced by 0 (irrespective of replace_val).
        """

        self.replace_val = replace_val  

    def __call__(self, data_):
        
        full_seq = data_["true"]
        full_seq = np.nan_to_num(full_seq, nan = self.replace_val)
        data_["true"] = full_seq

        if "noisy" in data_.keys():
            sparse_seq = data_["noisy"]
            sparse_seq = np.nan_to_num(sparse_seq, nan = self.replace_val)
            data_["noisy"] = sparse_seq

        if "read_depth" in data_.keys():
            cov_seq = data_["read_depth"]
            cov_seq = np.nan_to_num(cov_seq, nan = 0)
            data_["read_depth"] = cov_seq
        
        return(data_)



class AddNoise():
    """
    Add some Gaussian white noise at missing CpGs only (at eval_mask).
    This may prevent the model from collapsed inferences.
    Default params are chosen to keep CpG-wise variance at approx ~0.04.
    This is from GTEx empirical ground truth values and subject to change based on dataset.
    ** Noise is sampled each patch at a time and is iid over patches. **
    """

    def __init__(self, mu = 0.0, sigma = 0.2):
        
        self.rng = np.random.default_rng()
        self.mu = mu
        self.sigma = sigma

    def __call__(self, patch_data):
        
        sparse_seq = patch_data["noisy"]
        eval_mask = patch_data["eval_mask"]
        
        if np.isnan(sparse_seq).sum()>0:
            raise AssertionError("Looks like ReplaceNaN hasn't been called yet.\n\
                                 Please Replace NaNs with a valid value before adding noise!")

        if (np.sum(sparse_seq<0) > 0) or (np.sum(sparse_seq>1) > 0):
            raise AssertionError("Looks like M-value conversion was performed before noise addition,\n\
                                 Please add noise and then convert to M-values!")
        
        updated_seq = np.empty(sparse_seq.shape)      
        noise = self.rng.normal(self.mu, self.sigma, sparse_seq.shape)
        
        updated_seq[~eval_mask] = sparse_seq[~eval_mask]
        updated_seq[eval_mask] = sparse_seq[eval_mask] + noise[eval_mask]
        updated_seq = np.clip(updated_seq, a_min = 0, a_max = 1)

        patch_data["noisy"] = updated_seq
        return(patch_data)
        


class ToMValue():
    
    """
    By default, converts both: Full and Sparse sequences to M-values.
    Use M-Values instead of Beta values for (possibly) more stable training.
    Use default constructor.
    """

    def __init__(self, eps = 1e-4):
        self.eps = eps # for numerical stability in log
    
    def __call__(self, patch_data):

        full_seq = patch_data["true"]
        sparse_seq = patch_data["noisy"]

        if np.isnan(sparse_seq).sum()>0:

            raise AssertionError("Looks like ReplaceNaN hasn't been called yet.\n\
                                 Please Replace NaNs with a valid value before converting to M-values!")

        else:

            full_seq = np.clip(full_seq, 
                               a_min = self.eps, 
                               a_max = 1.0 - self.eps)
            sparse_seq = np.clip(sparse_seq, 
                                 a_min = self.eps, 
                                 a_max = 1.0 - self.eps)
        
            full_seq = np.log2(np.true_divide(full_seq, 
                                              np.subtract(1, full_seq)))
            
            sparse_seq = np.log2(np.true_divide(sparse_seq, 
                                                np.subtract(1, sparse_seq)))
        
        patch_data["true"] = full_seq
        patch_data["noisy"] = sparse_seq
        
        return(patch_data)



class ToTensor():
    
    """
    float32 needed (single precision) for pytorch models for cpu(?).
    """

    def __init__(self, prec):
        self.prec = prec
    
    def __call__(self, data_):

        full_seq = data_["true"]
        if "noisy" in data_:
            sparse_seq = data_["noisy"]
            if np.isnan(sparse_seq).sum()>0:
                raise AssertionError("Looks like ReplaceNaN hasn't been called yet.\n\
                                    Please Replace NaNs with a valid Beta before converting Tensors!")

        transform_pe = ("posn_vec" in data_) and isinstance(data_["posn_vec"], np.ndarray)

        if self.prec == "single":
            full_seq = torch.from_numpy(full_seq).type(torch.float32)
            if "noisy" in data_:
                sparse_seq = torch.from_numpy(sparse_seq).type(torch.float32)
            if transform_pe:
                data_["posn_vec"] = torch.from_numpy(data_["posn_vec"]).type(torch.float32)
        
        else:
            full_seq = torch.from_numpy(full_seq).type(torch.float64)
            if "noisy" in data_:
                sparse_seq = torch.from_numpy(sparse_seq).type(torch.float64)
            if transform_pe:
                data_["posn_vec"] = torch.from_numpy(data_["posn_vec"]).type(torch.float64)
                
        data_["true"] = full_seq
        if "noisy" in data_:
            data_["noisy"] = sparse_seq
        
        return(data_)