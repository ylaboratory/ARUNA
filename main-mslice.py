import os
import yaml
import random
import logging
import argparse
import itertools
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing
from torch.utils.data import DataLoader

from torchinfo import summary

from model_utils import get_peObj
from data_utils import get_mslice_dataset
from log_utils import log_quant_metrics_slice, log_to_csv
from utils import get_config, validate_config, assign_namev2, get_splits
from utils import add_validation_split, sampname_to_alias # atlas-specific helpers

from models import DCAE_MSLICE
from model_engine import train_step_mslice, valid_step_mslice, EarlyStopper

from evaluations import collate_mslices
import wandb

torch.manual_seed(108)
np.random.seed(108)
random.seed(108)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')
logging.basicConfig(format = "[%(levelname)s] %(asctime)s %(message)s",
                    datefmt = "%d-%b-%y %H:%M:%S",
                    level = logging.INFO)
logger = logging.getLogger(__name__)


# globals
CWD = os.getcwd()
modelData_dir = os.path.join(CWD, "model_data")
resData_dir = os.path.join(CWD, "results")
res_file = os.path.join(resData_dir, "val_res.csv")


def main(config):

    # Initiate wandb run with custom unique name
    run_name, run_group, run_tags = assign_namev2(config)
    run = wandb.init(project = "Patch-Imputation",
                     name = run_name,
                     tags = run_tags, 
                     group = run_group,
                     config = config)

    # define model save path, create a Subdir per Run
    modeldir_name = run_name.replace(":", "-").replace(">", "") # modify for valid linux dir name
    model_rundir = os.path.join(modelData_dir, "saved_models", modeldir_name)
    logger.info("Trained models will be saved at: {}".format(model_rundir))
    Path(model_rundir).mkdir(parents = True, exist_ok = True)

    # Get run's config-related metadata
    split_dir = os.path.join(CWD, "splits", 
                             config["data"]["train_dataset"] + "_intra_exps")

    split_dict = get_splits(split_dir, config["exp"]["split"])

    # name modification and add validation per fold for atlas exps
    if config["data"]["train_dataset"] == "atlas":
        split_dict = sampname_to_alias(split_dict)
        split_dict = add_validation_split(split_dict, 
                                          val_frac = 0.1, seed = 42)
    
    # add a small validation split when training with all GTEx
    if config["data"]["train_dataset"] == "gtex" and config["exp"]["split"] == "all":
        split_dict = add_validation_split(split_dict, 
                                          val_frac = 0.1, seed = 42)
        
    folds = list(split_dict.keys())
    if "Test" in folds:
        folds.remove("Test")

    chr = config["data"]["chrom"]
    batch_dim = config["model"]["batch_dim"]

    logger.info("Starting training + validation on {} folds...".format(len(folds)))
    for k in folds:
        logger.info("Fold: {}".format(k))
        fold_samps = split_dict[k]

        # Model save-related meta info
        model_savepath = os.path.join(model_rundir, "Fold"+k+".pth")
        with open(os.path.join(model_rundir, "config.yaml"), "w") as conf_file:
            yaml.dump(config, conf_file, default_flow_style = False)
        
        trainData_obj, valData_obj = get_mslice_dataset(config, 
                                                samp_split = fold_samps)

        # initialize dataloaders
        trainloader = DataLoader(trainData_obj, 
                                    batch_size = batch_dim, 
                                    shuffle = True, num_workers = 4)
        logger.info("#Batches in Training (batch_dim={}): {}".format(batch_dim, 
                                                                        len(trainloader)))
        valloader = DataLoader(valData_obj, 
                                batch_size = batch_dim, 
                                shuffle = False, num_workers = 4) # shuffle False during inference
        logger.info("#Batches in Validation (batch_dim={}): {}".format(batch_dim, 
                                                                        len(valloader)))
        
        logger.info("Initializing model based on config...")
        model = DCAE_MSLICE(config = config["model"])

        # Opt. initialization for Positional Embedding
        if config["model"]["posn_embed"]:
            pe_type = config["model"]["posn_embed"].split("_")[0]
            if  pe_type== "type2":
                embed_dim = model.embed_dim
            else:
                embed_dim = None
            pe_obj = get_peObj(pe_type = pe_type, 
                                num_cpg = config["data"]["num_cpgs"], 
                                embed_dim = embed_dim,
                                chrom = chr)
            # this is done basically to simplify batching of PE along with dataloader
            trainData_obj.pe_obj = pe_obj
            valData_obj.pe_obj = pe_obj

        # SHAPE EXAMPLE: Print torchvision.summary with layer-wise dimensions
        noise_shape = (config["model"]["batch_dim"], 
                    config["data"]["sps"],
                    config["data"]["num_cpgs"])
        if config["model"]["stem_dict"]:
            stem_dict = config["model"]["stem_dict"]
            pe_dim2 = stem_dict[list(stem_dict.keys())[-1]]["out_channels"]
            pe_shape = (config["model"]["batch_dim"], 
                        config["data"]["sps"],
                        pe_dim2, 
                        config["data"]["num_cpgs"])
        else:
            pe_shape = noise_shape
        summary_ = summary(model, [noise_shape, pe_shape], verbose = 0)
        logger.info(str(summary_))
        # SHAPE EXAMPLE ENDS

        wandb.config["trainable_params".format(k)] = summary_.trainable_params

        device = config["model"]["device"]
        criterion = config["model"]["criterion"]
        num_epochs = config["model"]["num_epochs"]
        model = model.to(device)
        logger.info("Model initialized and loaded onto GPU!")

        if criterion == "mse":
            loss_fn = nn.MSELoss()
        elif criterion == "bce":
            loss_fn = nn.BCELoss()
        elif criterion == "l1":
            loss_fn = nn.L1Loss()
        elif criterion == "huber":
            loss_fn = nn.HuberLoss()

        optimizer = optim.Adam(model.parameters(), 
                               lr = config["model"]["learning_rate"], 
                               weight_decay = config["model"]["l2_penalty"])
        es = EarlyStopper(patience = config["model"]["es_steps"], 
                          delta = config["model"]["es_delta"],
                          save_fpath = model_savepath)

         # Start training model
        logger.info("Training Model for {} Epochs...".format(num_epochs))
        for epoch in range(num_epochs):        
            logger.info("Epoch {}/{}".format(epoch+1, num_epochs))

            # Train 1 Epoch        
            epoch_trainLoss = train_step_mslice(model, trainloader, 
                                                loss_fn, optimizer, 
                                                device)
            # Validate 1 Epoch
            val_res = valid_step_mslice(model, valloader, 
                                        loss_fn, device)
            
            epoch_valLoss = val_res["epoch_valLoss"]

            logger.info("Epoch Train {}/{} Loss: {}".format(epoch+1, num_epochs, epoch_trainLoss))
            logger.info("Epoch Validation {}/{} Loss: {}".format(epoch+1, num_epochs, epoch_valLoss))

            loss_logs = {"fold{}-epoch".format(k): epoch,
                         "fold{}-train_loss".format(k): epoch_trainLoss, 
                         "fold{}-validation_loss".format(k): epoch_valLoss}
            run.log(loss_logs)

             # Log metrics EVERY epoch
            # logger.info("Computing validation-set metrics...")
            # all_collated_data = collate_mslices(val_res, 
            #                                     config["data"]["test_spp"],
            #                                     chr)
            # quantlog_dict = log_quant_metrics_slice(k, epoch, all_collated_data, 
            #                                         config["data"]["feat_type"])

            # run.log(quantlog_dict)
            # logger.info("Metrics computed and logged.")

            if config["model"]["early_stop"]:
                if es.early_stop(epoch_valLoss, epoch, model):
                    run.summary["fold{}-min_validation_loss".format(k)] = es.min_validation_loss
                    run.summary["fold{}-best_epoch".format(k)] = es.min_val_epoch
                    logger.info("Early stopping activated!")
                    break
                    
        # In case of no EarlyStop, save model after all epochs finish
        if not config["model"]["early_stop"]:
            es.checkpoint(model, model_savepath)
        logger.info("Training Complete")

        # uses the last collated data
        # !! NOTE: This may not be the best performance!!
        # To get best performance, load saved model and re-run validation or test
        # log_to_csv(run_name, k, config["data"]["test_dataset"], 
        #            [chr,], fold_samps["Validate"], 
        #            all_collated_data, res_file, slice = True)
    
    logger.info("Trained model's fold-wise validation performance stored at: {}".format(res_file))
    run.finish()




if __name__ == "__main__":

    wandb.require("core") # better logging and more performant
    wandb.login()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help = "YAML file specifying run configuration.") 
    args = parser.parse_args()

    config_gen = get_config(args.config) # generator object
    for config in config_gen:
        
        config = validate_config(config, (config["data"]["num_cpgs"], 
                                          config["data"]["sps"]))
 
        main(config)
    
    logger.info("Run Complete!")