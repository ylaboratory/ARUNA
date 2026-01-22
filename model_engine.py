import numpy as np
import torch
import math

import logging
logger = logging.getLogger(__name__)

# TODO: For GPatch train and valid step, fix loss computation

def train_step(model, trainloader, 
               loss_fn, optimizer,
               device, arch = None, 
               s = None, per = 3):
    """
    per: "prints per epoch"
    """

    if arch == "dcae" and not s:
        raise AssertionError("Patch size needed when architecture is specified!")

    model.train()
    
    running_loss = 0.0
    log_denom = int(round(len(trainloader)/per, -2)) # 3 prints per epoch
    if log_denom == 0:
        log_denom = 1

    for i, batch_data in enumerate(trainloader):

        batch_true = batch_data["true"].to(device)
        batch_noisy = batch_data["noisy"].to(device)
        batch_noncpgMask = batch_data["noncpg_mask"].to(device)
        batch_evalMask = batch_data["eval_mask"].to(device)
        batch_evalMask = batch_evalMask[~torch.isnan(batch_evalMask)].type(torch.bool).to(device)

        # SKIPPING 0 eval_mask BATCHES:
        # sometimes, all batch-patches' cpgs can have False eval_mask
        # this can happen when non-cpg was observed in GT or none was sim masked
        # particularly true for small patch sizes (chr21 partculary, see karyoplot)
        # such a batch should be skipped during training and validation since
        # loss computation is not possible
        if torch.sum(batch_evalMask).item() == 0:
            continue

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            _, pred_seq = model(batch_noisy)
            if arch == "dcae":
                pred_seq = pred_seq[:,:,:s]

            # get data at eval_mask only, no for loop needed
            # use index 1 on dim 2 when train_on == "betas"
            # noisy_eval = batch_noisy[:,1,:][~batch_noncpgMask][batch_evalMask]
            true_eval = batch_true[:,1,:][~batch_noncpgMask][batch_evalMask]
            pred_eval = pred_seq[:,1,:][~batch_noncpgMask][batch_evalMask]

            loss = loss_fn(pred_eval, true_eval) # default reduction = "mean"
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * batch_true.shape[0]  # to account for reduction
        if i%log_denom == 0 and log_denom != 1:
                print("{}/{} batches complete".format(i, len(trainloader)))

    print("{}/{} batches complete".format(len(trainloader), len(trainloader)))
    return(running_loss)



def valid_step(model, valloader, 
               loss_fn, device,
               arch = None, s = None,
               get_embeds = False,
               get_preds = False,
               skip_noeval_batches = True):
    
    """
    Function should be used for validation and testing sets.
    To avoid unnecessary if-else checks, 
    if any/all of get_embeds and get_preds are False, return DS remains same
    but with empty lists.
    """
    
    if arch == "dcae" and not s:
        raise AssertionError("Patch size needed when architecture is specified!")
    
    model.eval()
    all_embeds = []
    all_batch_preds = []

    running_loss = 0.0
    for batch_data in valloader:
        batch_true = batch_data["true"].to(device)
        batch_noisy = batch_data["noisy"].to(device)
        batch_noncpgMask = batch_data["noncpg_mask"].to(device)
        batch_evalMask = batch_data["eval_mask"].to(device)
        batch_evalMask = batch_evalMask[~torch.isnan(batch_evalMask)].type(torch.bool).to(device)

        # see comment in train_step
        if skip_noeval_batches:
            if torch.sum(batch_evalMask).item() == 0:
                continue

        with torch.set_grad_enabled(False):
            
            if get_embeds:
                embed_seq, pred_seq = model(batch_noisy)
                all_embeds.append(embed_seq.detach().cpu())
            else:
                 _, pred_seq = model(batch_noisy)
                 if arch == "dcae":
                     pred_seq = pred_seq[:,:,:s]
            
            true_eval = batch_true[:,1,:][~batch_noncpgMask][batch_evalMask]
            pred_eval = pred_seq[:,1,:][~batch_noncpgMask][batch_evalMask]
            loss = loss_fn(pred_eval, true_eval) # default reduction = "mean"
        
        if get_preds:
            all_batch_preds.append(pred_seq.detach().cpu().numpy())
        
        running_loss += loss.item() * batch_true.shape[0]

    valid_loss = running_loss/len(valloader.sampler)

    return(valid_loss, all_embeds, all_batch_preds)



# Slightly different train and val steps (especially loss computation for MPatch)
def train_step_mpatch(model, trainloader, 
                      loss_fn, optimizer,
                      device = None, arch = None, per = 3):
    """
    per: prints per epoch
    """
   
    model.train()
    
    running_loss = 0.0
    log_denom = int(round(len(trainloader)/per, -2))
    if log_denom == 0:
        log_denom = 1

    evalMask_size = 0 # stores total eval_mask CpGs in the dataset, refreshes every epoch
    for i, batch_data in enumerate(trainloader):
        batch_true = batch_data["true"].to(device)
        batch_noisy = batch_data["noisy"].to(device)
        batch_evalMask = batch_data["eval_mask"].to(device)
        batch_pe = batch_data["posn_vec"].to(device)

        if torch.sum(batch_evalMask).item() == 0: # skip batches with no trainable CpGs
            continue

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            batch_preds = model(batch_noisy, batch_pe).squeeze() # (B, 1, #CpG) --> (B, #CpG)

            true_eval = batch_true[batch_evalMask]
            pred_eval = batch_preds[batch_evalMask]

            # loss with default reduction = "mean", average over eval_mask number of CpGs
            
            # special consideration for smooth-label BCE loss setting
            if loss_fn._get_name() == "BCELoss":
                pred_eval = pred_eval.clamp(min = 1e-7, max =(1-1e-7))

            loss = loss_fn(pred_eval, true_eval)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * batch_evalMask.sum().item() # to account for reduction
        evalMask_size += batch_evalMask.sum().item() # total # of trainable CpGs
        if i%log_denom == 0 and log_denom != 1:
                print("{}/{} batches complete".format(i, len(trainloader)))

    print("{}/{} batches complete".format(len(trainloader), len(trainloader)))
    epoch_trainLoss = running_loss/evalMask_size # should approximate MSE (on a per-CpG basis)
    return(epoch_trainLoss)



def valid_step_mpatch(model, valloader, 
                      loss_fn, device = None, 
                      arch = None):
    """
    Function should be used for validation and testing sets.
    To avoid unnecessary if-else checks, 
    if any/all of get_embeds and get_preds are False, return DS remains same but with empty lists.
    """

    model.eval()
    
    all_batch_preds = []
    
    # to help with patch collation and whole methylome evals
    samp_list = []
    chrom_list = []
    true_list = []
    evalmask_list = []
    
    running_loss = 0.0
    evalMask_size = 0 # stores total eval_mask CpGs in the dataset, refreshes every epoch
    
    for batch_data in valloader:
        samp_list.extend(batch_data["samp"])
        chrom_list.extend(batch_data["chrom"])
        true_list.append(batch_data["true"])
        evalmask_list.append(batch_data["eval_mask"])
    
        batch_true = batch_data["true"].to(device)
        batch_noisy = batch_data["noisy"].to(device)
        batch_evalMask = batch_data["eval_mask"].to(device)
        batch_pe = batch_data["posn_vec"].to(device)
    
        with torch.set_grad_enabled(False):
            batch_preds = model(batch_noisy, batch_pe).squeeze() # (B, 1, #CpG) --> (B, #CpG)

            # dont skip if evalMask is all False to avoid misalignment between returned lists
            # instead, only modify loss computation variables
            if torch.sum(batch_evalMask).item() > 0:
                true_eval = batch_true[batch_evalMask]
                pred_eval = batch_preds[batch_evalMask]
                loss = loss_fn(pred_eval, true_eval)
                running_loss += loss.item() * batch_evalMask.sum().item() # to account for reduction
                evalMask_size += batch_evalMask.sum().item() 

        all_batch_preds.append(batch_preds.detach().cpu().numpy())

    epoch_valLoss = running_loss/evalMask_size # should approx per-CpG MSE
    assert len(all_batch_preds) == len(true_list) == len(evalmask_list),\
          "#Batch mismatch in valid_step! Maybe due to empty eval_mask based skipping."
    
    return_struct = {"epoch_valLoss": epoch_valLoss, 
                     "samples": samp_list,
                     "chrom": chrom_list,
                     "gt": true_list,
                     "preds": all_batch_preds,
                     "evalMask": evalmask_list}
    
    return(return_struct)



def train_step_mslice(model, trainloader, 
                      loss_fn, optimizer, 
                      device = None, per = 3):
    """
    per: prints per epoch
    """

    model.train()
    running_loss = 0.0
    
    log_denom = int(round(len(trainloader)/per, -2))
    if log_denom == 0:
        log_denom = 1

    evalMask_size = 0 # stores total eval_mask CpGs in the dataset, refreshes every epoch
    for i, batch_data in enumerate(trainloader):
        batch_true = batch_data["true"].to(device)
        batch_noisy = batch_data["noisy"].to(device)
        batch_evalMask = batch_data["eval_mask"].to(device)
        batch_pe = batch_data["posn_vec"].to(device)

        if torch.sum(batch_evalMask).item() == 0: # skip batches with no trainable CpGs
            continue

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            batch_preds = model(batch_noisy, batch_pe).squeeze() # channel dim squeezed out

            true_eval = batch_true[batch_evalMask]
            pred_eval = batch_preds[batch_evalMask]
            loss = loss_fn(pred_eval, true_eval)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * batch_evalMask.sum().item() # to account for reduction
        evalMask_size += batch_evalMask.sum().item() # total # of trainable CpGs
        
        if i%log_denom == 0 and log_denom != 1:
                print("{}/{} batches complete".format(i, len(trainloader)))

    print("{}/{} batches complete".format(len(trainloader), len(trainloader)))
    epoch_trainLoss = running_loss/evalMask_size # should approximate MSE (on a per-CpG basis)
    return(epoch_trainLoss)



def valid_step_mslice(model, valloader, 
                      loss_fn, device = None):

    model.eval()

    samp_list = [] # len = len(valloader) = num_batches
    pid_list = [] # len = num_slices = #patches * #samples * spp
    true_list = [] # len = len(valloader) = num_batches
    evalmask_list = []
    all_batch_preds = []

    running_loss = 0.0
    evalMask_size = 0 # stores total eval_mask CpGs in the dataset, refreshes every epoch
    
    for batch_data in valloader:
        
        samp_list.append(batch_data["samps"]) # 2-tuples of batch_dim * num_batches
        pid_list.extend(batch_data["patch_id"])
        true_list.append(batch_data["true"])
        evalmask_list.append(batch_data["eval_mask"])

        batch_true = batch_data["true"].to(device)
        batch_noisy = batch_data["noisy"].to(device)
        batch_evalMask = batch_data["eval_mask"].to(device)
        batch_pe = batch_data["posn_vec"].to(device)

        with torch.set_grad_enabled(False):
            batch_preds = model(batch_noisy, batch_pe).squeeze() # squeezes channel dim

            if torch.sum(batch_evalMask).item() > 0:
                    true_eval = batch_true[batch_evalMask]
                    pred_eval = batch_preds[batch_evalMask]
                    loss = loss_fn(pred_eval, true_eval)
                    running_loss += loss.item() * batch_evalMask.sum().item() # to account for reduction
                    evalMask_size += batch_evalMask.sum().item() 

        all_batch_preds.append(batch_preds.detach().cpu().numpy())
    
    epoch_valLoss = running_loss/evalMask_size # should approx per-CpG MSE
    
    return_struct = {"epoch_valLoss": epoch_valLoss, 
                     "samples": samp_list,
                     "pids": pid_list,
                     "gt": true_list,
                     "preds": all_batch_preds,
                     "evalMask": evalmask_list}
    
    return(return_struct)




def test_step_mslice(model, testloader, 
                     device = None):
    
    """
    Meant for inference with GT RRBS data.
    """

    model.eval()

    samp_list = [] # len = len(valloader) = num_batches
    pid_list = [] # len = num_slices = #patches * #samples * spp
    all_batch_preds = []
    
    for batch_data in testloader:
        
        samp_list.append(batch_data["samps"]) # 2-tuples of batch_dim * num_batches
        pid_list.extend(batch_data["patch_id"])
       
        batch_true = batch_data["true"].to(device)
        batch_pe = batch_data["posn_vec"].to(device)

        with torch.set_grad_enabled(False):
            batch_preds = model(batch_true, batch_pe).squeeze() # squeezes channel dim

        all_batch_preds.append(batch_preds.detach().cpu().numpy())

    
    return_struct = {"samples": samp_list,
                     "pids": pid_list,
                     "preds": all_batch_preds}
    
    return(return_struct)




# class EarlyStopper:
    
#     def __init__(self, patience, delta, save_fpath):
        
#         self.patience = patience
#         self.delta = delta
#         self.counter = 0
#         self.min_validation_loss = float('inf')

#         self.max_plateau = patience
#         self.min_val_epoch = None
#         self.save_fpath = save_fpath # to checkpoint save model

#     def early_stop(self, validation_loss, epoch, model):
        
#         # if val loss reduced by more than 1e-3 [!!!WARNING: specific to MSE loss!!!]
#         if validation_loss < self.min_validation_loss:
#             self.min_validation_loss = validation_loss
#             self.min_val_epoch = epoch
#             logger.info("Model checkpoint at epoch: {}, validation loss: {}".format(epoch, 
#                                                                                     self.min_validation_loss))
#             self.checkpoint(model, self.save_fpath)
#             self.counter = 0

#         # if val loss diverged by more than delta
#         elif validation_loss > (self.min_validation_loss + self.delta):
#             self.counter += 1
#             if self.counter >= self.patience:
#                 logger.info("Validation loss diverged.")
#                 return True
       
#         # if val loss hasn't improved in a while 
#         if (epoch - self.min_val_epoch >= self.max_plateau):
#             logger.info("Validation loss plateaued.")
#             return True
#         return False


#     def checkpoint(self, model, save_fpath):
#         torch.save(model.state_dict(), save_fpath)
#         logger.info("New best model (as per validation loss) saved at: {}".format(save_fpath))



class EarlyStopper:

    def __init__(self, patience = 5, 
                 min_delta = 0.0005, save_fpath = None):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.save_fpath = save_fpath

        self.best = float("inf")
        self.best_epoch = -1
        self.counter = 0

    def early_stop(self, val_loss, epoch, model):
        if val_loss is None or not math.isfinite(val_loss):
            self.counter += 1
            logger.info(
                f"ES | epoch={epoch} val={val_loss} "
                f"(non-finite) counter={self.counter}/{self.patience}"
            )
            return self.counter >= self.patience

        prev_best = self.best
        is_new_best = val_loss < prev_best

        if is_new_best:
            improvement = prev_best - val_loss
            self.best = val_loss
            self.best_epoch = epoch

            logger.info(
                f"ES | epoch={epoch} NEW_BEST val={val_loss:.6f} "
                f"(improved by {improvement:.6f}, min_delta={self.min_delta})"
            )

            if self.save_fpath is not None:
                self.checkpoint(model, self.save_fpath)

            if improvement > self.min_delta:
                self.counter = 0
                logger.info(
                    f"ES | epoch={epoch} significant improvement "
                    f"(>{self.min_delta}), counter reset"
                )
                return False
            else:
                self.counter += 1
                logger.info(
                    f"ES | epoch={epoch} improvement {improvement:.6f} "
                    f"<= min_delta ({self.min_delta}); "
                    f"counter={self.counter}/{self.patience}"
                )
        else:
            self.counter += 1
            logger.info(
                f"ES | epoch={epoch} no improvement "
                f"(val={val_loss:.6f}, best={self.best:.6f}); "
                f"counter={self.counter}/{self.patience}"
            )

        if self.counter >= self.patience:
            logger.info(
                f"ES | STOP triggered after {self.patience} epochs without "
                f"significant improvement. "
                f"best={self.best:.6f} at epoch={self.best_epoch}"
            )
            return True

        return False


    def checkpoint(self, model, save_fpath):
        torch.save(model.state_dict(), save_fpath)
        logger.info(f"New best model saved at: {save_fpath}")
