import torch
import numpy as np
import random

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.device = next(model.parameters()).device
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].to(param.data.device)
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

@torch.no_grad()
def recall_at_k(top_k_items, true_item):
    if isinstance(top_k_items, torch.Tensor):
        top_k_items = top_k_items.cpu().numpy()
    if isinstance(true_item, torch.Tensor):
        true_item = true_item.item()
    return 1.0 if true_item in top_k_items else 0.0

@torch.no_grad()
def ndcg_at_k(top_k_items, true_item):
    if isinstance(top_k_items, torch.Tensor):
        top_k_items = top_k_items.cpu().numpy()
    if isinstance(true_item, torch.Tensor):
        true_item = true_item.item()
    if true_item not in top_k_items:
        return 0.0
    rank = np.where(top_k_items == true_item)[0][0] + 1
    return 1.0 / np.log2(rank + 1)