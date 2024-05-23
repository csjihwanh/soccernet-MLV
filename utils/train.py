import logging
import os
import time
import torch
import gc
from config.classes import INVERSE_EVENT_DICTIONARY
import json
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
from tqdm import tqdm
import torch.distributed as dist

def trainer(train_loader,
            val_loader2,
            test_loader2,
            model,
            optimizer,
            scheduler,
            criterion,
            best_model_path,
            epoch_start,
            model_name,
            path_dataset,
            max_epochs=1000,
            rank=None,
            world_size=None,
            fsdp=False,
            sampler = (None,None,None,None),
            ):
    
    train_sampler, valid_sampler, test_sampler, chall_sampler = sampler

    logging.info("start training")
    counter = 0


    for epoch in range(epoch_start, max_epochs):

        print(f"Epoch {epoch+1}/{max_epochs}")
    
        # Create a progress bar
        if rank == 0 or rank is None:
            pbar = tqdm(total=len(train_loader), desc="Training", position=0, leave=True)

        ###################### TRAINING ###################
        prediction_file, loss_action, loss_offence_severity = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=True,
            set_name="train",
            pbar=pbar,
            rank=rank,
            world_size=world_size,
            sampler = train_sampler 
        )

        results = evaluate(os.path.join(path_dataset, "Train", "annotations.json"), prediction_file)
        print("TRAINING")
        print(results)

        ###################### VALIDATION ###################
        prediction_file, loss_action, loss_offence_severity = train(
            val_loader2,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train = False,
            set_name="valid",
            rank=rank,
            world_size=world_size,
            sampler = valid_sampler
        )

        results = evaluate(os.path.join(path_dataset, "Valid", "annotations.json"), prediction_file)
        print("VALIDATION")
        print(results)


        ###################### TEST ###################
        prediction_file, loss_action, loss_offence_severity = train(
                test_loader2,
                model,
                criterion,
                optimizer,
                epoch + 1,
                model_name,
                train=False,
                set_name="test",
                rank=rank,
                world_size=world_size,
                sampler = test_sampler
            )

        results = evaluate(os.path.join(path_dataset, "Test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)
        

        scheduler.step()

        counter += 1

        #if counter > 3:
        state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
        }
        path_aux = os.path.join(best_model_path, str(epoch+1) + "_model.pth.tar")
        torch.save(state, path_aux)
        print(f"model saved at {path_aux}")
        
    pbar.close()    
    return

def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          model_name,
          train=False,
          set_name="train",
          pbar=None,
          GPU=0,
          rank=None,
          world_size=None,
          sampler=None,
        ):
    

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    loss_total_action = 0
    loss_total_offence_severity = 0
    total_loss = 0

    if not os.path.isdir(model_name):
        os.mkdir(model_name) 

    # path where we will save the results
    prediction_file = "predictions_" + set_name + "_epoch_" + str(epoch) + ".json"
    data = {}
    data["Set"] = set_name

    actions = {}

    if True:
        if sampler:
            sampler.set_epoch(epoch)

        for targets_offence_severity, targets_action, mvclips, action in dataloader:
            targets_offence_severity = targets_offence_severity.cuda()
            targets_action = targets_action.cuda()
            mvclips = mvclips.cuda().float()

            if rank:
                targets_offence_severity = targets_offence_severity.to(rank)
                targets_action = targets_action.to(rank)
                mvclips = mvclips.to(rank)

            if pbar is not None:
                pbar.update()

            # compute output
            outputs_offence_severity, outputs_action, _ = model(mvclips)
            
            if len(action) == 1:
                preds_sev = torch.argmax(outputs_offence_severity, 0)
                preds_act = torch.argmax(outputs_action, 0)

                values = {}
                values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
                if preds_sev.item() == 0:
                    values["Offence"] = "No offence"
                    values["Severity"] = ""
                elif preds_sev.item() == 1:
                    values["Offence"] = "Offence"
                    values["Severity"] = "1.0"
                elif preds_sev.item() == 2:
                    values["Offence"] = "Offence"
                    values["Severity"] = "3.0"
                elif preds_sev.item() == 3:
                    values["Offence"] = "Offence"
                    values["Severity"] = "5.0"
                actions[action[0]] = values       
            else:
                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), 1)
                preds_act = torch.argmax(outputs_action.detach().cpu(), 1)

                for i in range(len(action)):
                    values = {}
                    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
                    if preds_sev[i].item() == 0:
                        values["Offence"] = "No offence"
                        values["Severity"] = ""
                    elif preds_sev[i].item() == 1:
                        values["Offence"] = "Offence"
                        values["Severity"] = "1.0"
                    elif preds_sev[i].item() == 2:
                        values["Offence"] = "Offence"
                        values["Severity"] = "3.0"
                    elif preds_sev[i].item() == 3:
                        values["Offence"] = "Offence"
                        values["Severity"] = "5.0"
                    actions[action[i]] = values       

            
            if len(outputs_offence_severity.size()) == 1:
                outputs_offence_severity = outputs_offence_severity.unsqueeze(0)   
            if len(outputs_action.size()) == 1:
                outputs_action = outputs_action.unsqueeze(0)  
   
            #compute the loss
            loss_offence_severity = criterion[0](outputs_offence_severity, targets_offence_severity)
            loss_action = criterion[1](outputs_action, targets_action)

            loss = loss_offence_severity + loss_action

            if train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if param.grad is not None:
                            print(f"Parameter: {name}, Requires Grad: {param.requires_grad}, Gradient: {param.grad.shape}")
                        else:
                            print(f"Parameter: {name}, Requires Grad: {param.requires_grad}, Gradient is None (potential issue)")
                    else:
                        print(f"Parameter: {name} does not require gradients")

                optimizer.step()

            loss_total_action += float(loss_action)
            loss_total_offence_severity += float(loss_offence_severity)
            total_loss += 1
            
    
        gc.collect()
        torch.cuda.empty_cache()
        #torch.save(model.state_dict(), f'/hub_data1/intern/soccernet-pose/checkpoints/model{epoch}.pth')
    
    data["Actions"] = actions
    with open(os.path.join(model_name, prediction_file), "w") as outfile: 
        json.dump(data, outfile)  
    return os.path.join(model_name, prediction_file), loss_total_action / total_loss, loss_total_offence_severity / total_loss




# Evaluation function to evaluate the test or the chall set
def evaluation(dataloader,
          model,
          set_name="test",
        ):
    

    model.eval()

    prediction_file = "predicitions_" + set_name + ".json"
    data = {}
    data["Set"] = set_name

    actions = {}
           
    if True:
        for _, _, mvclips, action in dataloader:

            mvclips = mvclips.cuda().float()
            #mvclips = mvclips.float()
            outputs_offence_severity, outputs_action, _ = model(mvclips)

            if len(action) == 1:
                preds_sev = torch.argmax(outputs_offence_severity, 0)
                preds_act = torch.argmax(outputs_action, 0)

                values = {}
                values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
                if preds_sev.item() == 0:
                    values["Offence"] = "No offence"
                    values["Severity"] = ""
                elif preds_sev.item() == 1:
                    values["Offence"] = "Offence"
                    values["Severity"] = "1.0"
                elif preds_sev.item() == 2:
                    values["Offence"] = "Offence"
                    values["Severity"] = "3.0"
                elif preds_sev.item() == 3:
                    values["Offence"] = "Offence"
                    values["Severity"] = "5.0"
                actions[action[0]] = values       
            else:
                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), 1)
                preds_act = torch.argmax(outputs_action.detach().cpu(), 1)

                for i in range(len(action)):
                    values = {}
                    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
                    if preds_sev[i].item() == 0:
                        values["Offence"] = "No offence"
                        values["Severity"] = ""
                    elif preds_sev[i].item() == 1:
                        values["Offence"] = "Offence"
                        values["Severity"] = "1.0"
                    elif preds_sev[i].item() == 2:
                        values["Offence"] = "Offence"
                        values["Severity"] = "3.0"
                    elif preds_sev[i].item() == 3:
                        values["Offence"] = "Offence"
                        values["Severity"] = "5.0"
                    actions[action[i]] = values                    


        gc.collect()
        torch.cuda.empty_cache()
    
    data["Actions"] = actions
    with open(prediction_file, "w") as outfile: 
        json.dump(data, outfile)  
    return prediction_file
