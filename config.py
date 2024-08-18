import torch
import random
import numpy as np
from dataclasses import dataclass

@dataclass
class Config:
    IMAGE_PATH = "data/images/"
    QA_PATH = "data/merged_qa.json"

    BATCH_SIZE = 4

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    RANDOM_SEED = 42
    

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False