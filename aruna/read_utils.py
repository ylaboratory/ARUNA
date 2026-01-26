import pandas as pd

pd.options.mode.chained_assignment = None

import logging
logger = logging.getLogger(__name__)



def read_sample(filepath):

    """
    Utility function to read in data processed with the;
    bismark_methylation_extractor --bedGraph + coverage2cytosine --merge_cpg --zero_based 
    pipeline.
    
    Arguments:
    ---------

    filepath (str): A valid filepath to a sample's .
                    Input files are expected to have columns (in order):
                    Sequence | Start | End | Beta (%) | Methylated Reads | Unmethylated Reads.
    

    Returns:
    -------
    df: pd.DataFrame with named columns and data for sequences in target_chr.


    """
    # keep only chrs 1-22, remove X, Y and scaffolds.
    target_chr = [str(i) for i in range(1,23)] # str idx in bismark files
    chr_replace_map = {k:"chr"+k for k in target_chr}
    df = pd.read_csv(filepath,
                     header = None, sep = "\t", 
                     names = ["seqname", "start", "end", "beta",
                              "num_methylated", "num_unmethylated"], 
                     dtype = {"seqname":"str"}) # by default, seqnames are read as str "1" or int 1 (at non-overlapping parts) 

    df = df[df.loc[:, "seqname"].isin(target_chr)]
    df = df.replace(chr_replace_map) # to make amenable to joins with hg38 dfs
    df = df.reset_index().drop(columns = "index")
    
    return(df)



def get_processed_df(samp_file, curr_chr):

    """
    Reads and performs essential subsetting and processing of sample data.
    samp_file points to a file compatible with read_sample.
    Returns: pandas.DataFrame of samp-chr with betas converted to ratios and\
             a new read_depth column inferred from num_methylated and num_unmethylated columns.
    """
    
    df = read_sample(samp_file)
    chr_df = df[df.loc[:, "seqname"] == curr_chr]
    chr_df["beta"] = chr_df["beta"]/100
    chr_df["read_depth"] = chr_df[["num_methylated", "num_unmethylated"]].sum(axis=1)
    chr_df.set_index("start", inplace = True)
    
    return(chr_df)