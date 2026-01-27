from aruna.process_dataset import get_pc_gt, get_pc_noisy
from aruna.data_engine import ReplaceNaN, AddNoise, ToTensor, ToMValue
from aruna.data_engine import MSliceCentricDataset



def get_mpatches(dataset, chrom, patch_type, num_cpg, nr, seed, samp_list):

    """
    Gets MPatches for both - Ground Truth and Noise Regime data (w/ Simulated Mask) for the entire dataset (i.e., all samples).
    Supports one or more chromosomes.

    Independent of no. of chromosomes, the return structure is always dict like;
    {sample_name: [(3-tuple), ...]} where 3-tuple is (patch_id, patch_betas, chromosome).

    Simulated Mask is True where CpGs were simulated to be missing.
    For RRBS datasets, Simulated Mask is True wherever imputed values are desired.
    Usually for RRBS, simMask = cpg_mask (all missing in GT samples).

    Arguments
    -----
    dataset (str): Name of dataset.
    chrom (list or str): list or str of the form chr# where # is from 1-22.
    patch_type (str): One of "mpatch" or "gpatch".
    num_cpg (int): Number of CpGs per patch. Only used when "mpatch" is specified.
    nr (str): One of "rrbs_sim" or "mcar_XX" where XX can be 30,50,90 for percent missing CpGs in simulated data.
    """

    print(
    "Getting Patch Data for:\n"
    "Dataset: {}\n"
    "Chr(s): {}\n"
    "Patch Type: {}\n"
    "NR: {}\n"
    "#Samples: {}".format(dataset, chrom, patch_type, nr, len(samp_list)))
    
    gt_map = get_pc_gt(dataset = dataset, 
                       chrom = chrom, 
                       num_cpg = num_cpg, 
                       data_type = "fm", 
                       subset = samp_list)
    
    nr_map, simMask_map = get_pc_noisy(dataset = dataset, 
                                       chrom = chrom, 
                                       num_cpg = num_cpg,
                                       nr = nr, 
                                       subset = samp_list)

    cov_map = None
    
    return (gt_map, nr_map, simMask_map, cov_map) # cov_map placed last for backward compatibility



def get_mslice_dataset(config, samples, mode = "infer"):
    
     """
     mode: {"train", "infer"}
     """

     # resolve set of data transforms to apply based on provided config
     transforms_list = [ReplaceNaN(replace_val = config["data"]["nan_sub"])]
     if config["data"]["apply_noise"]:
          mu = config["data"]["apply_noise"]["mu"]
          sigma = config["data"]["apply_noise"]["sigma"]
          transforms_list.append(AddNoise(mu, sigma))
     if config["data"]["feat_type"] == "mvals":
          transforms_list.append(ToMValue())
     transforms_list.append(ToTensor("single"))

     if mode == "train":
          d = config["data"]["train_dataset"]
          nr = config["exp"]["train_nr"]
          repeats_ = config["data"]["train_spp"]
     else:
          d = config["data"]["test_dataset"]
          nr = config["exp"]["test_nr"]
          repeats_ = config["data"]["test_spp"]

     print("Loading Data...")
     gt, nr, simMask, cov = get_mpatches(d,
                                         config["data"]["chrom"],
                                         config["data"]["kind"],
                                         config["data"]["num_cpgs"],
                                         nr, 
                                         config["exp"]["sim_seed"], 
                                         samples)

     data_obj = MSliceCentricDataset(samples, 
                                     config["data"]["chrom"], 
                                     config["data"]["num_cpgs"], 
                                     gt, nr, 
                                     simMask, cov, 
                                     sps = config["data"]["sps"], 
                                     spp = repeats_, 
                                     posn_embed = config["model"]["posn_embed"], 
                                     transform = transforms_list)

     print("Loaded Data with {} samples and {} patches.".format(len(samples), 
                                                                len(data_obj)))
     return(data_obj)