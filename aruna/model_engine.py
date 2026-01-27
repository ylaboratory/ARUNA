import torch
import math


def train_step(model, trainloader, 
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



def valid_step(model, valloader, 
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




def test_step(model, testloader, 
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
            print(
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

            print(
                f"ES | epoch={epoch} NEW_BEST val={val_loss:.6f} "
                f"(improved by {improvement:.6f}, min_delta={self.min_delta})"
            )

            if self.save_fpath is not None:
                self.checkpoint(model, self.save_fpath)

            if improvement > self.min_delta:
                self.counter = 0
                print(
                    f"ES | epoch={epoch} significant improvement "
                    f"(>{self.min_delta}), counter reset"
                )
                return False
            else:
                self.counter += 1
                print(
                    f"ES | epoch={epoch} improvement {improvement:.6f} "
                    f"<= min_delta ({self.min_delta}); "
                    f"counter={self.counter}/{self.patience}"
                )
        else:
            self.counter += 1
            print(
                f"ES | epoch={epoch} no improvement "
                f"(val={val_loss:.6f}, best={self.best:.6f}); "
                f"counter={self.counter}/{self.patience}"
            )

        if self.counter >= self.patience:
            print(
                f"ES | STOP triggered after {self.patience} epochs without "
                f"significant improvement. "
                f"best={self.best:.6f} at epoch={self.best_epoch}"
            )
            return True

        return False


    def checkpoint(self, model, save_fpath):
        torch.save(model.state_dict(), save_fpath)
        print(f"New best model saved at: {save_fpath}")
