import numpy as np
import torch
import os
import json

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class RealCQA(Dataset):
    def __init__(self, img_list, qa_json, img_path, processor, device):
        super().__init__()
        self.img_list = img_list
        self.qa_json = qa_json
        self.img_path = img_path
        self.processor = processor
        self.device = device
        
    def __len__(self):
        return len(self.img_list)
    

    def __getitem__(self, idx):
        item_id = self.img_list[idx][:-4]

        # Get image with following name
        image = Image.open(self.img_path + item_id + '.jpg')
        
        # Get corresponding information from json file
        qa = self.qa_json[item_id]
        
        # Since every image has a plethora of questions, select one from them randomly
        rnd_sample = np.random.randint(len(qa))

        # Take only question and corresponding answer from dict
        q, a = qa[rnd_sample]['question'], qa[rnd_sample]['answer']

        if isinstance(a, list):
            while isinstance(a[0], list):
                a = a[0]
            a = ', '.join([str(el) for el in a])
        
        # Process images and corresponding questions
        inputs = self.processor(images=image, text=q, return_tensors="pt", max_patches=768).to(self.device)
    
        inputs['labels'] = str(a)
        
        return inputs
    

    def collator(self, batch):
        new_batch = {"flattened_patches":[], "attention_mask":[]}

        labels = [item['labels'] for item in batch]
        new_batch["labels"] = self.processor.tokenizer.batch_encode_plus(labels, return_tensors="pt", add_special_tokens=True, 
                                                                         max_length=20, truncation=True, padding="max_length").to(self.device)['input_ids']

        for item in batch:
            new_batch["flattened_patches"].append(item["flattened_patches"])
            new_batch["attention_mask"].append(item["attention_mask"])

        new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
        new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

        return new_batch
    

def prepare_dataloaders(cfg, processor, test_size=0.15):
    imgs_list = os.listdir(cfg.IMAGE_PATH)

    train_imgs, test_imgs = train_test_split(imgs_list, test_size=test_size)
    
    with open(cfg.QA_PATH, "r") as f:
        qa_json = json.load(f)
        
    train_ds = RealCQA(train_imgs, qa_json, cfg.IMAGE_PATH, processor, cfg.DEVICE)
    test_ds = RealCQA(test_imgs, qa_json, cfg.IMAGE_PATH, processor, cfg.DEVICE)
    
    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=train_ds.collator)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=test_ds.collator)
    
    return train_dl, test_dl