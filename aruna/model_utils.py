import os
import logging
import numpy as np
import pandas as pd
from aruna.patch_metadata import get_mPatchPosn_metadata

logger = logging.getLogger(__name__)



# globals
CWD = os.getcwd()
patch_metadata_dir = os.path.join(CWD, "metadata")
hg38_metadata_dir = os.path.join(patch_metadata_dir, "hg38_ref")
hg38_chrLens_df = pd.read_csv(os.path.join(hg38_metadata_dir, 
                                           "hg38_chr_lengths.bedGraph"), 
                                           sep = "\t", header = None, 
                                           names = ["seqname", "start", "end"])
hg38_chrLens_df['chr_num'] = hg38_chrLens_df['seqname'].str.extract(r'chr(\d+)', 
                                                                    expand=False).astype(int)
hg38_chrLens_df = hg38_chrLens_df.sort_values(by='chr_num')



class PositionEmbedding():
    def __init__(self, num_cpg, 
                  pe_type = "type1",
                  mode = "single"):

        """
        When mode = multiple, it is assumed multiple chrs are being used for training/testing.
        In this case, for chr i; position offset = len of sum of chr <i will be added to each cpg.
        """
        self.num_cpg = num_cpg
        self.pe_type = pe_type
        self.pe_map = {} # dict of dicts like {chr: {pid: np.array()}}
        self.all_chrom = ["chr"+str(i) for i in range (1,23)]

        self.patch_refCpG_map = {}
        logger.info("Populating patch-wise CpG positions mapping...")
        for chrom in self.all_chrom:
            chr_idx = int(chrom[3:])
            if mode == "multiple":
                offset = hg38_chrLens_df.loc[hg38_chrLens_df['chr_num'] < chr_idx, 'end'].sum()
            else:
                 offset = 0
            self.patch_refCpG_map[chrom] = get_mPatchPosn_metadata(chrom, 
                                                                   self.num_cpg,
                                                                   offset)
        logger.info("Patch-Position metadata mapping complete!")
        self.pe_dim = None # for type2


    def get_pe(self, chrom, pid):
        pe = self.pe_map[chrom][pid]
        return(pe)
        

    def store_t1_pe(self):
        for chrom in self.all_chrom:
            pid_pe_map = {}
            for pid in self.patch_refCpG_map[chrom].keys():
                pid_pe_map[pid] = compute_type1_pe(pid, 
                                                   self.patch_refCpG_map[chrom], 
                                                   self.num_cpg)

            self.pe_map[chrom] = pid_pe_map
        logger.info("Type 1 PEs computed!")
        return


    def store_t2_pe(self):
        
        if not self.pe_dim:
             raise ValueError("Model embedding dim needed for Type 2 PE!")
    
        for chrom in self.all_chrom:
            pid_pe_map = {}
            for pid in self.patch_refCpG_map[chrom].keys():
                pid_pe_map[pid] = compute_type2_pe(pid, 
                                                   self.patch_refCpG_map[chrom], 
                                                   self.num_cpg, self.pe_dim)

            self.pe_map[chrom] = pid_pe_map
        logger.info("Type 2 PEs computed!")
        return



# this is implemented outside the model class to reinforce that this is at the data (input) level
def compute_type1_pe(pid, pid_cpg_map, num_cpgs):


    """
    Only meant for use with MPatches!

    Trying sinusoidal positional embeddings
    PE(pos, 2i) = sin(pos/(10000 ** (2*i/d_model)))
    PE(pos, 2i+1) = cos(pos/10000 ** (2*i/d_model))
    Trial 1: 
            pos = CpG position on chr 
            i = CpG position on patch
            d_model = Hyperparameter num_cpgs
    """
    patch_posnvec = np.zeros(num_cpgs)

    # stays the same
    patch_idx = np.arange(1,num_cpgs+1) # all
    w = (10000 ** (2*patch_idx/num_cpgs))

    # for different sine/cosine computations
    patch_idx_even = np.arange(2,num_cpgs+1,2)-1
    patch_idx_odd = np.arange(1,num_cpgs+1,2)-1

    # depends on patch id
    patch_posns = list(pid_cpg_map[pid].index)

    if len(patch_posns) != num_cpgs: # edge case for the last patch
            patch_posns.extend([-1]*(num_cpgs - len(patch_posns)))

    patch_posns = np.array(patch_posns)
    even_posns = np.sin(patch_posns/w)
    odd_posns = np.cos(patch_posns/w)

    patch_posnvec[patch_idx_even] = even_posns[patch_idx_even]
    patch_posnvec[patch_idx_odd] = odd_posns[patch_idx_odd]

    return(patch_posnvec)



def compute_type2_pe(pid, pid_cpg_map, num_cpgs, num_features):

    """
    Applied after embedding patches.
    Per patch output of shape (num_cpgs, num_features)
    Trying sinusoidal positional embeddings
    PE(pos, 2i) = sin(pos/(10000 ** (2*i/d_model)))
    PE(pos, 2i+1) = cos(pos/10000 ** (2*i/d_model))
    Trial 2: 
            pos = CpG position on chr
            i = dim along feature dimension
            d_model = num_features post embedding
    Note that i should basically be an iterator along d_model.
    """
    
    patch_posnvec = np.zeros((num_cpgs, num_features))

    feature_idx = np.arange(num_features)+1
    # for different sine/cosine computations
    feature_idx_even = np.arange(2,num_features+1,2)-1
    feature_idx_odd = np.arange(1,num_features+1,2)-1

    w = (10000 ** (2*feature_idx/num_features))

    patch_posns = list(pid_cpg_map[pid].index)
    if len(patch_posns) != num_cpgs: # edge case for the last patch
            patch_posns.extend([-1]*(num_cpgs - len(patch_posns)))
    
    patch_posns = np.array(patch_posns)
    even_posns = np.sin(patch_posns[:,None]/w[None,:])
    odd_posns = np.cos(patch_posns[:,None]/w[None,:])

    patch_posnvec[:,feature_idx_even] = even_posns[:,feature_idx_even]
    patch_posnvec[:,feature_idx_odd] = odd_posns[:,feature_idx_odd]

    return(patch_posnvec.T)



def get_peObj(pe_type, num_cpg, embed_dim, chrom):

    """
    chrom: str or None type. 
    """

    if chrom:
        mode = "single"
    else:
        mode = "multiple"
    logger.info("Positional Embeddings {} with {} chrs selected.".format(pe_type, mode))
    pe_obj = PositionEmbedding(num_cpg,
                                pe_type, mode)
    if pe_type == "type2":
        pe_obj.pe_dim = embed_dim
        logger.info("Computing and storing Type 2 PE mapping for all chromosomes...")
        pe_obj.store_t2_pe()
    else:
        logger.info("Computing and storing Type 1 PE mapping for all chromosomes...")
        pe_obj.store_t1_pe()
    
    return(pe_obj)     



def mvalTobeta(mval_seq):
    
    # beta_seq = (2.0**(mval_seq))/(2.0**(mval_seq) + 1)
    # above formulation changed to sigmoid-like for numerical stability for large M-vals
    # issue first encountered in EXP010-Trial1 with Mvals.
    beta_seq = 1 / (1 + np.exp2(-mval_seq))
    return(beta_seq)