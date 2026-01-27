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



def get_mslice_dataset(config, samp_split, inference = False):

     # resolve set of data transforms to apply based on provided config
    transforms_list = [ReplaceNaN(replace_val = config["data"]["nan_sub"])]
    if config["data"]["apply_noise"]:
         mu = config["data"]["apply_noise"]["mu"]
         sigma = config["data"]["apply_noise"]["sigma"]
         transforms_list.append(AddNoise(mu, sigma))
    if config["data"]["feat_type"] == "mvals":
         transforms_list.append(ToMValue())
    transforms_list.append(ToTensor("single"))

    train_samps = samp_split["Train"]
    val_samps = samp_split["Validate"]

    if not inference:
        print("Loading Training Data...")
        gt_train, nr_train, simMask_train, cov_train = get_mpatches(config["data"]["train_dataset"], 
                                                                    config["data"]["chrom"], 
                                                                    config["data"]["kind"], 
                                                                    config["data"]["num_cpgs"], 
                                                                    config["exp"]["train_nr"],  
                                                                    config["exp"]["sim_seed"],
                                                                    train_samps)
        trainData_obj = MSliceCentricDataset(train_samps, 
                                         config["data"]["chrom"], 
                                         config["data"]["num_cpgs"], 
                                         gt_train, nr_train, 
                                         simMask_train, cov_train, 
                                         sps = config["data"]["sps"], 
                                         spp = config["data"]["train_spp"], 
                                         posn_embed = config["model"]["posn_embed"], 
                                         transform = transforms_list)
        
        print("Loaded Training Data with {} samples and {} patches.".format(len(train_samps), 
                                                                        len(trainData_obj)))

    print("Loading Validation (or Testing) Data...")
    gt_val, nr_val, simMask_val, cov_val = get_mpatches(config["data"]["test_dataset"],
                                                        config["data"]["chrom"],
                                                        config["data"]["kind"],
                                                        config["data"]["num_cpgs"],
                                                        config["exp"]["test_nr"], 
                                                        config["exp"]["sim_seed"], 
                                                        val_samps)

    valData_obj = MSliceCentricDataset(val_samps, 
                                       config["data"]["chrom"], 
                                       config["data"]["num_cpgs"], 
                                       gt_val, nr_val, 
                                       simMask_val, cov_val, 
                                       sps = config["data"]["sps"], 
                                       spp = config["data"]["test_spp"], 
                                       posn_embed = config["model"]["posn_embed"], 
                                       transform = transforms_list)

    print("Loaded Validation Data with {} samples and {} patches.".format(len(val_samps), 
                                                                          len(valData_obj)))
    
    if not inference:
        return(trainData_obj, valData_obj)
    else:
        return(valData_obj)
    



def get_mslice_dataset_test(config, samples):

     # resolve set of data transforms to apply based on provided config
    transforms_list = [ReplaceNaN(replace_val = config["data"]["nan_sub"])]
    if config["data"]["apply_noise"]:
         mu = config["data"]["apply_noise"]["mu"]
         sigma = config["data"]["apply_noise"]["sigma"]
         transforms_list.append(AddNoise(mu, sigma))
    if config["data"]["feat_type"] == "mvals":
         transforms_list.append(ToMValue())
    transforms_list.append(ToTensor("single"))


    print("Loading Testing Data...")
    gt_val, nr_val, simMask_val, cov_val = get_mpatches(config["data"]["test_dataset"],
                                                        config["data"]["chrom"],
                                                        config["data"]["kind"],
                                                        config["data"]["num_cpgs"],
                                                        config["exp"]["test_nr"], 
                                                        config["exp"]["sim_seed"], 
                                                        samples)

    valData_obj = MSliceCentricDataset(samples, 
                                       config["data"]["chrom"], 
                                       config["data"]["num_cpgs"], 
                                       gt_val, nr_val, 
                                       simMask_val, cov_val, 
                                       sps = config["data"]["sps"], 
                                       spp = config["data"]["test_spp"], 
                                       posn_embed = config["model"]["posn_embed"], 
                                       transform = transforms_list)

    print("Loaded Validation Data with {} samples and {} patches.".format(len(samples), 
                                                                          len(valData_obj)))
    return(valData_obj)