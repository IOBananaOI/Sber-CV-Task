import torch
import neptune
import os
import numpy as np
import random

from anls import anls_score
from tqdm import tqdm

def get_gradient_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    return total_norm


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def relaxed_correctness(labels, predictions, threshold=0.05):
    if is_number(labels) and is_number(predictions):
        return float(labels) * (1 - threshold) <= float(predictions) <= float(labels) * (1 + threshold)
    
    return int(labels == predictions)


def compute_metrics(labels, predictions):
    metrics = {
        "EM" : [],
        "Relaxed correctness" : [],
        "ANLS" : [],
    }
    labels = labels.lower()
    predictions = predictions.lower()
    
    # Compute Exact match metric
    metrics['EM'] = int(labels == predictions)
    
    # Compute ANLS metric
    metrics['ANLS'] = anls_score(prediction=predictions, gold_labels=[labels], threshold=0.5)

    # Compute Relaxed accuracy
    metrics['Relaxed correctness'] = relaxed_correctness(labels, predictions)
    
    return metrics


def train_model(model, 
                optimizer, 
                train_dl,
                test_dl, 
                num_epochs, 
                processor,
                device,
                scheduler=None,
                neptune_tracking=True,
                model_name="",
                start_epoch=0):
    
    if neptune_tracking:
        run = neptune.init_run(
            api_token=os.environ.get("NEPTUNE_API_TOKEN"),
            project='bng215/Model-Collapse'
        )
    
    for epoch in range(start_epoch, num_epochs):
        
        model.train()
        with tqdm(train_dl) as tepoch:
            tepoch.set_description(f"Epoch: {epoch + 1}: Train stage")
            for step, batch in enumerate(tepoch):                
                labels = batch.pop("labels").to(device).squeeze(1)
                flattened_patches = batch.pop("flattened_patches").to(device).squeeze(1)
                attention_mask = batch.pop("attention_mask").to(device).squeeze(1)
                
                outputs = model(flattened_patches=flattened_patches,
                                attention_mask=attention_mask,
                                labels=labels)
            
                loss = outputs.loss
                
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                
                gnorm = get_gradient_norm(model)

                if scheduler is not None:
                    scheduler.step()
                
                if neptune_tracking:
                    run['Loss (Train)'].append(loss.item())
                    run['Gradient norm'].append(gnorm)
                    run['Learning rate'].append(optimizer.param_groups[0]['lr'])
                    
                if (step + 1) % 100 == 0:
                    model.eval()
                    with torch.no_grad():
                        test_metrics = {
                            "Loss (Test)" : [],
                            "EM" : [],
                            "Relaxed correctness" : [],
                            "ANLS" : [],
                        }
                        
                        with tqdm(test_dl) as val_stage:
                            val_stage.set_description(f"Epoch: {epoch + 1}: Val stage")
                            for batch in val_stage:
                                labels = batch.pop("labels").to(device).squeeze(1)
                                flattened_patches = batch.pop("flattened_patches").to(device).squeeze(1)
                                attention_mask = batch.pop("attention_mask").to(device).squeeze(1)
                                
                                outputs = model(flattened_patches=flattened_patches,
                                                attention_mask=attention_mask,
                                                labels=labels)
                                
                                loss = outputs.loss
                                test_metrics['Loss (Test)'].append(loss.item())
                                
                                predictions = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask)
                                decoded_prediction = processor.batch_decode(predictions, skip_special_tokens=True)[0]
                                decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)[0]
                                    
                                batch_metrics = compute_metrics(decoded_labels, decoded_prediction)
                                
                                for k, v in batch_metrics.items():
                                    test_metrics[k].append(v)
                                
                            if neptune_tracking:
                                for k, v in test_metrics.items():
                                    run[f"{k}"].append(np.mean(v))
                            else:
                                print(f"Loss (Test): {loss.item()}")
                                for k, v in test_metrics.items():
                                    print(f"{k}:", np.mean(v))
                                
                    state = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                            
                    torch.save(state, f'weights/{model_name}_step_{step+1}.pth')
                    model.train()
                        
        