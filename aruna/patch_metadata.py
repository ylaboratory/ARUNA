def get_mPatchPosn_metadata(chrom, num_cpg, 
                            hg38_allcpg_df, offset = 0):

    """
    offset: used when multiple chromosomes used for training/testing.
            add sum of lens for chr_{<i} for chr i.
    """
    
    hg38_chr_df = hg38_allcpg_df[hg38_allcpg_df.loc[:, "seqname"] == chrom].set_index("start")
    hg38_chr_df.index = hg38_chr_df.index + offset
    patch_refCpG_map = {}
    i = 0
    pid = 0
    while i < hg38_chr_df.shape[0]:
        patch_refCpG_map[pid] = hg38_chr_df.iloc[i:i+num_cpg]
        i += num_cpg
        pid += 1
    
    return(patch_refCpG_map)