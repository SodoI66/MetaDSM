import torch
from torch.optim.lr_scheduler import LambdaLR
import math
from tqdm import tqdm
import numpy as np
from utils import recall_at_k, ndcg_at_k, EMA, to_device
import os
import json


class Trainer:
    def __init__(self, config, model, train_loader, test_loader, pretrain_loader, pop_sampler=None):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.pretrain_loader = pretrain_loader
        self.pop_sampler = pop_sampler
        params = []
        mode = self.config.ablation_mode

        if "baseline" in mode:
            lr = config.pretrain_lr
            if mode in ['baseline_MeLU', 'baseline_MAMO', 'baseline_TDAS', 'baseline_TaNP']:
                lr = config.meta_lr
            params.append({'params': self.model.parameters(), 'lr': lr})
        else:
            if self.model.use_gcn:
                params.append({'params': self.model.light_gcn.parameters(), 'lr': config.meta_gcn_lr})
            else:
                params.append({'params': self.model.user_embedding.parameters(), 'lr': config.meta_lr})
                params.append({'params': self.model.item_embedding.parameters(), 'lr': config.meta_lr})
            if self.model.use_gru:
                params.append({'params': self.model.gru.parameters(), 'lr': config.meta_lr})
            if hasattr(self.model, 'static_predictor'):
                params.append({'params': self.model.static_predictor.parameters(), 'lr': config.meta_lr})
            if hasattr(self.model, 'dynamic_predictor'):
                params.append({'params': self.model.dynamic_predictor.parameters(), 'lr': config.meta_lr})
            if hasattr(self.model, 'interaction_predictor'):
                params.append({'params': self.model.interaction_predictor.parameters(), 'lr': config.meta_lr})
            if hasattr(self.model, 'gating_network'):
                params.append({'params': self.model.gating_network.parameters(), 'lr': config.meta_lr})
            if hasattr(self.model, 'feature_projector'):
                params.append({'params': self.model.feature_projector.parameters(), 'lr': config.meta_lr})

        self.optimizer = torch.optim.AdamW(params, weight_decay=config.weight_decay)

        if mode == 'baseline_LightGCN':
            num_training_steps = self.config.pretrain_epochs * len(self.pretrain_loader)
            num_warmup_steps = self.config.warmup_epochs * len(self.pretrain_loader)
        else:
            num_training_steps = self.config.num_epochs * len(self.train_loader)
            num_warmup_steps = self.config.warmup_epochs * len(self.train_loader)

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.use_ema = "baseline" not in mode
        if self.use_ema:
            self.ema = EMA(self.model, decay=config.ema_decay)

        self.history = {'epochs': [], 'train_loss': []}
        for k in self.config.eval_k_list:
            self.history[f'recall@{k}'] = []
            self.history[f'ndcg@{k}'] = []

        run_name = f"{config.ablation_mode}"
        if config.ablation_mode == 'full':
            if config.fusion_ablation != 'full':
                run_name += f"_{config.fusion_ablation}"
            if config.inner_loop_mode != 'step_by_step':
                run_name += f"_{config.inner_loop_mode}"

        if config.support_size != 10:
            run_name += f"_k{config.support_size}"

        if config.seed != 2025:
            run_name += f"_seed{config.seed}"

        self.output_dir = os.path.join('results', config.dataset_name, config.eval_scenario, run_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def _pretrain_gcn(self):
        if not hasattr(self.model, 'light_gcn') or self.model.light_gcn is None:
            return
        gcn_optim = torch.optim.Adam(self.model.light_gcn.parameters(), lr=self.config.pretrain_lr)
        for epoch in tqdm(range(1, self.config.pretrain_epochs + 1), desc="Pre-training"):
            self.model.train()
            total_loss = 0.0
            for users, pos_items in self.pretrain_loader:
                users, pos_items = users.to(self.config.device), pos_items.to(self.config.device)
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
        if "baseline" not in self.config.ablation_mode:
            self._pretrain_gcn()

        for epoch in tqdm(range(1, self.config.num_epochs + 1), desc="Training"):
            avg_train_loss = self._train_epoch()
            if avg_train_loss is not None:
                self.history['train_loss'].append(avg_train_loss)
            self.history['epochs'].append(epoch)
            self._evaluate_test_set(epoch)
        if self.config.ablation_mode == 'full':
            final_model_path = os.path.join(self.output_dir, 'final_model.pth')
            torch.save(self.model.state_dict(), final_model_path)
        self._save_history_to_json()
        self._save_history_summary_txt()

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        mode = self.config.ablation_mode
        loader = self.pretrain_loader if mode == 'baseline_LightGCN' else self.train_loader
        for batch in loader:
            if batch is None: continue
            self.optimizer.zero_grad()
            loss = None

            if mode == 'baseline_LightGCN':
                users, pos_items = to_device(batch, self.config.device)
                neg_items = torch.randint(0, self.config.num_items, (len(users),), device=self.config.device)
                loss = self.model.forward(users, pos_items, neg_items)
            elif mode == 'baseline_GRU4Rec':
                batch = to_device(batch, self.config.device)
                if self.pop_sampler is None:
                    raise ValueError("Popularity sampler is required for GRU4Rec baseline.")
                pos_items = batch['query_item'].squeeze(1)
                batch_size = pos_items.size(0)
                random_values = torch.rand(batch_size, self.config.gru4rec_n_sample, device=self.config.device)
                neg_items = torch.searchsorted(torch.from_numpy(self.pop_sampler[1]).to(self.config.device),
                                               random_values)
                bpr_batch = {'support_seq': batch['support_seq'], 'pos_items': pos_items, 'neg_items': neg_items}
                loss = self.model.forward(bpr_batch)
            elif mode in ['baseline_MeLU', 'baseline_FMLPRec', 'baseline_SASRec', 'baseline_MAMO', 'baseline_TDAS',
                          'baseline_TaNP']:
                batch = to_device(batch, self.config.device)
                loss = self.model.forward(batch)
            else:
                batch = to_device(batch, self.config.device)
                if self.model.use_meta:
                    loss = self.model.forward_meta(batch)
                else:
                    loss = self.model.forward_no_meta(batch)

            if loss is None or torch.isnan(loss):
                continue
            loss.backward()
            if mode == 'baseline_LightGCN':
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.outer_grad_clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            if self.use_ema:
                self.ema.update()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches if num_batches > 0 else None

    def _evaluate_test_set(self, epoch=0):
        if self.use_ema: self.ema.apply_shadow()
        self.model.eval()
        k_list = self.config.eval_k_list
        all_recalls = {k: [] for k in k_list}
        all_ndcgs = {k: [] for k in k_list}
        mode = self.config.ablation_mode

        saved_weights = []
        save_weights_flag = (epoch == self.config.num_epochs) and ("baseline" not in mode)

        try:
            for batch_idx, batch in enumerate(self.test_loader):
                if batch is None or batch['query_item'].numel() == 0: continue
                batch = to_device(batch, self.config.device)

                scores = None
                weights = None

                if "baseline" in mode:
                    scores = self.model.evaluate_batch(batch)
                else:
                    need_weights = save_weights_flag and (batch_idx == 0)
                    if self.model.use_meta:
                        res = self.model.evaluate_batch(batch, return_weights=need_weights)
                    else:
                        res = self.model.evaluate_batch_no_meta(batch, return_weights=need_weights)

                    if need_weights:
                        scores, weights = res
                        if weights is not None:
                            pos_weights = weights[:, 0].squeeze().cpu().tolist()
                            user_ids = batch['user_id'].cpu().tolist()
                            for uid, w in zip(user_ids, pos_weights):
                                saved_weights.append({'user_id': uid, 'fusion_weight': w})
                    else:
                        scores = res

                eval_items = torch.cat([batch['query_item'], batch['neg_items']], dim=1)
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

        if saved_weights:
            weight_path = os.path.join(self.output_dir, 'case_study_weights.json')
            with open(weight_path, 'w') as f:
                json.dump(saved_weights, f, indent=4)

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
