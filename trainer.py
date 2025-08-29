import torch
from torch.optim.lr_scheduler import LambdaLR
import math
from tqdm import tqdm
import numpy as np
from utils import recall_at_k, ndcg_at_k, EMA, to_device
import os
import json
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, config, model, train_loader, test_loader, pretrain_loader):
        self.config = config
        self.model = model.to(config.device)
        if hasattr(torch, 'compile') and torch.__version__.startswith('2'):
            self.model = torch.compile(self.model)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.pretrain_loader = pretrain_loader
        params = []
        params.append({'params': self.model.light_gcn.parameters(), 'lr': config.meta_gcn_lr})
        params.append({'params': self.model.gru.parameters(), 'lr': config.meta_lr})
        if hasattr(self.model, 'fusion_mlp') and self.model.fusion_mlp is not None:
            params.append({'params': self.model.fusion_mlp.parameters(), 'lr': config.meta_lr})
        if hasattr(self.model, 'predictor') and self.model.predictor is not None:
            params.append({'params': self.model.predictor.parameters(), 'lr': config.meta_lr})
        if hasattr(self.model, 'feature_projector'):
            params.append({'params': self.model.feature_projector.parameters(), 'lr': config.meta_lr})

        self.optimizer = torch.optim.AdamW(params, weight_decay=config.weight_decay)

        num_training_steps = self.config.num_epochs * len(self.train_loader)
        num_warmup_steps = self.config.warmup_epochs * len(self.train_loader)

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.use_ema = True
        if self.use_ema:
            self.ema = EMA(self.model, decay=config.ema_decay)

        self.history = {'epochs': [], 'train_loss': []}
        for k in self.config.eval_k_list:
            self.history[f'recall@{k}'] = []
            self.history[f'ndcg@{k}'] = []

        self.output_dir = f'results/{self.config.dataset_name}_full_{self.config.eval_scenario}'
        os.makedirs(self.output_dir, exist_ok=True)

    def _pretrain_gcn(self):
        if self.model.light_gcn is None:
            return
        gcn_optim = torch.optim.Adam(self.model.light_gcn.parameters(), lr=self.config.pretrain_lr)
        for epoch in tqdm(range(1, self.config.pretrain_epochs + 1), desc="Pre-training"):
            self.model.train()
            total_loss = 0.0
            for users, pos_items in self.pretrain_loader:
                users = users.to(self.config.device)
                pos_items = pos_items.to(self.config.device)
                neg_items = torch.randint(0, self.config.num_items, (len(users),), device=self.config.device)
                collisions = (neg_items == pos_items)
                while torch.any(collisions):
                    new_randoms = torch.randint(0, self.config.num_items, (collisions.sum(),),
                                                device=self.config.device)
                    neg_items.masked_scatter_(collisions, new_randoms)
                    collisions = (neg_items == pos_items)
                gcn_optim.zero_grad()
                loss = self.model.forward_pretrain(users, pos_items, neg_items)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.light_gcn.parameters(), 1.0)
                gcn_optim.step()
                total_loss += loss.item()

    def train(self):
        self._pretrain_gcn()
        for epoch in tqdm(range(1, self.config.num_epochs + 1), desc="Training"):
            avg_train_loss = self._train_epoch()
            if avg_train_loss is not None:
                self.history['train_loss'].append(avg_train_loss)
            self.history['epochs'].append(epoch)
            self._evaluate_test_set()
        
        final_model_path = os.path.join(self.output_dir, 'final_model.pth')
        torch.save(self.model.state_dict(), final_model_path)
        self._save_history_to_json()
        self._save_history_summary_txt()
        self._plot_history()

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        loader = self.train_loader
        for batch in loader:
            if batch is None: continue
            self.optimizer.zero_grad()
            batch = to_device(batch, self.config.device)
            loss = self.model.forward_meta(batch)

            if loss is None or torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.outer_grad_clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            if self.use_ema:
                self.ema.update()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches if num_batches > 0 else None

    def _evaluate_test_set(self):
        if self.use_ema: self.ema.apply_shadow()
        self.model.eval()
        k_list = self.config.eval_k_list
        all_recalls = {k: [] for k in k_list}
        all_ndcgs = {k: [] for k in k_list}

        gcn_user_emb, gcn_item_emb = None, None
        with torch.no_grad():
            gcn_user_emb, gcn_item_emb = self.model._get_base_embeddings()

        try:
            for batch in self.test_loader:
                if batch is None or batch['query_item'].numel() == 0: continue

                batch = to_device(batch, self.config.device)
                neg_items = batch['neg_items']
                scores = self.model.evaluate_batch(batch, gcn_user_emb, gcn_item_emb)
                eval_items = torch.cat([batch['query_item'], neg_items], dim=1)

                max_k = max(k_list)
                _, top_indices = torch.topk(scores, k=min(max_k, scores.size(1)))

                for i in range(scores.size(0)):
                    true_item = batch['query_item'][i].item()
                    top_items_for_user = eval_items[i][top_indices[i]].cpu()

                    for k in k_list:
                        top_k = top_items_for_user[:k]
                        all_recalls[k].append(recall_at_k(top_k, true_item))
                        all_ndcgs[k].append(ndcg_at_k(top_k, true_item))
        finally:
            if self.use_ema: self.ema.restore()

        for k in k_list:
            if all_recalls[k]:
                avg_recall = np.mean(all_recalls[k])
                avg_ndcg = np.mean(all_ndcgs[k])
                self.history[f'recall@{k}'].append(avg_recall)
                self.history[f'ndcg@{k}'].append(avg_ndcg)

    def _save_history_to_json(self):
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

    def _save_history_summary_txt(self):
        history_path = os.path.join(self.output_dir, 'training_history.txt')
        with open(history_path, 'w') as f:
            f.write(f"Training History Summary\n" + "=" * 50 + "\n")
            if self.history['epochs']:
                for k in self.config.eval_k_list:
                    if self.history[f'recall@{k}']:
                        best_recall_idx = np.argmax(self.history[f'recall@{k}'])
                        best_recall_epoch = self.history['epochs'][best_recall_idx]
                        best_recall = self.history[f'recall@{k}'][best_recall_idx]
                        f.write(f"Best Recall@{k}: {best_recall:.4f} (at epoch {best_recall_epoch})\n")
                    if self.history[f'ndcg@{k}']:
                        best_ndcg_idx = np.argmax(self.history[f'ndcg@{k}'])
                        best_ndcg_epoch = self.history['epochs'][best_ndcg_idx]
                        best_ndcg = self.history[f'ndcg@{k}'][best_ndcg_idx]
                        f.write(f"Best NDCG@{k}: {best_ndcg:.4f} (at epoch {best_ndcg_epoch})\n")
    
    def _plot_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        if self.history['train_loss']:
            ax1.plot(self.history['epochs'], self.history['train_loss'], label='Training Loss', color='tab:red')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Epochs')
        ax1.legend()
        ax1.grid(True)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.config.eval_k_list) * 2))
        color_idx = 0
        for k in self.config.eval_k_list:
            if self.history[f'recall@{k}']:
                ax2.plot(self.history['epochs'], self.history[f'recall@{k}'], label=f'Recall@{k}', linestyle='-', color=colors[color_idx])
                color_idx += 1
            if self.history[f'ndcg@{k}']:
                ax2.plot(self.history['epochs'], self.history[f'ndcg@{k}'], label=f'NDCG@{k}', linestyle='--', color=colors[color_idx])
                color_idx += 1
        
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Metric Value')
        ax2.set_title('Evaluation Metrics Over Epochs')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'training_curves.pdf')
        plt.savefig(plot_path, format='pdf')
        plt.close()