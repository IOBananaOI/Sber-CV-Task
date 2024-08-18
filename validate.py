import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from train import compute_metrics

def validate_model(model,
             test_dl,
             device,
             processor,
             save_path,
             file_suffix=''
             ):
    model.eval()
    with torch.no_grad():
        test_metrics = {
            "Loss (Test)" : [],
            "EM" : [],
            "Relaxed correctness" : [],
            "ANLS" : [],
        }
        
        pred_dict = {
            "predictions" : [],
            "labels" : []
        }
        
        metrics_df = pd.DataFrame(test_metrics)
        metrics_df.loc[0] = 0
        
        with tqdm(test_dl) as val_stage:
            val_stage.set_description(f"Validation in process")
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
                
                pred_dict['predictions'].append(decoded_prediction)
                pred_dict['labels'].append(decoded_labels)
                
                batch_metrics = compute_metrics(decoded_labels, decoded_prediction)
                
                for k, v in batch_metrics.items():
                    test_metrics[k].append(v)
        
        
        pred_df = pd.DataFrame(pred_dict)
        
        for k, v in test_metrics.items():
            metrics_df[k] = np.mean(v)
                
        pred_df.to_csv(save_path + f"predictions{file_suffix}.csv", index=False)
        metrics_df.to_csv(save_path + f"metrics{file_suffix}.csv", index=False)