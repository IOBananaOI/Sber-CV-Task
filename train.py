import torch
import neptune
import os

from tqdm import tqdm

def get_gradient_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    return total_norm


def compute_metrics(labels, predictions):
    metrics = {
        "EM" : [],
        "BLEU" : [],
        "Relaxed corectness" : [],
        "ANLS" : [],
        "CIDEr" : []
    }
    # Compute Exact match metric
    metrics['EM'] = int(labels == predictions)
    
    # Compute BLEU metric
    


def train_model(model, 
                optimizer, 
                train_dl,
                test_dl, 
                num_epochs, 
                processor,
                device,
                scheduler=None,
                neptune_tracking=True,
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
            for i, batch in enumerate(tepoch):                
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
                
                if neptune_tracking:
                    run['Loss (Train)'].append(loss.item())
                    run['Gradient norm'].append(gnorm)
                    run['Learning rate'].append(optimizer.param_groups[0]['lr'])
                        
        model.eval()
        with torch.no_grad():
            test_metrics = {
                "Loss (Test)" : [],
                "EM" : [],
                "BLEU" : [],
                "Relaxed corectness" : [],
                "ANLS" : [],
                "CIDEr" : []
            }
            
            for i, batch in enumerate(test_dl):
                labels = batch.pop("labels").to(device).squeeze(1)
                flattened_patches = batch.pop("flattened_patches").to(device).squeeze(1)
                attention_mask = batch.pop("attention_mask").to(device).squeeze(1)
                
                outputs = model(flattened_patches=flattened_patches,
                                attention_mask=attention_mask,
                                labels=labels)
                
                loss = outputs.loss
                test_metrics['Loss (Test)'].append(loss.item())
                
                
                
                
                if i < 3:
                    predictions = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask)        
                    print("Predictions:", processor.batch_decode(predictions, skip_special_tokens=True))
                    print("Ground-truth:", processor.batch_decode(labels, skip_special_tokens=True))
                    
                
                    
                    
                if neptune_tracking:
                    pass
                else:
                    print(f"Loss (Test): {loss.item()}")
                    
        torch.save(model.state_dict(), f'weights/ep_{epoch+1}.pth')