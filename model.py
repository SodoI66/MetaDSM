import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from collections import OrderedDict
from modules import LightGCN


class MetaDSM(nn.Module):
    def __init__(self, config, interaction_matrix, item_genre_features=None):
        super(MetaDSM, self).__init__()
        self.config = config
        mode = self.config.ablation_mode
        self.use_gcn = mode not in ['no_gcn', 'gru_only', 'meta_only']
        self.use_gru = mode not in ['no_gru', 'gcn_only', 'meta_only']
        self.use_meta = mode not in ['no_meta', 'gcn_only', 'gru_only']

        if self.use_gcn:
            self.light_gcn = LightGCN(
                num_users=config.num_users,
                num_items=config.num_items,
                embedding_dim=config.embedding_dim,
                interaction_matrix=interaction_matrix,
                num_layers=config.gnn_layers,
            )
        else:
            self.user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
            self.item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
            nn.init.xavier_uniform_(self.user_embedding.weight)
            nn.init.xavier_uniform_(self.item_embedding.weight)
            self.light_gcn = None

        self.item_genre_features = None
        if item_genre_features is not None:
            if sp.issparse(item_genre_features):
                item_genre_features = item_genre_features.toarray()
            self.item_genre_features = torch.tensor(item_genre_features, dtype=torch.float)
            feature_dim = self.item_genre_features.shape[1]
            self.feature_projector = nn.Linear(feature_dim, config.embedding_dim)

        if self.use_gru:
            self.gru = nn.GRU(
                input_size=config.embedding_dim,
                hidden_size=config.embedding_dim,
                num_layers=self.config.gru_layers,
                batch_first=True,
                dropout=self.config.dropout if self.config.gru_layers > 1 else 0,
            )
            self.gru_dropout = nn.Dropout(self.config.dropout)
        else:
            self.gru = None

        self._initialize_meta_components()

    def _get_base_embeddings(self):
        if self.light_gcn:
            user_emb, item_emb = self.light_gcn()
        else:
            user_emb, item_emb = self.user_embedding.weight, self.item_embedding.weight

        if hasattr(self, 'feature_projector') and self.item_genre_features is not None:
            feature_emb = self.feature_projector(self.item_genre_features.to(item_emb.device))
            item_emb = item_emb + feature_emb

        return user_emb, item_emb

    def _initialize_meta_components(self):
        embedding_dim = self.config.embedding_dim
        predictor_head = lambda input_dim, output_dim: nn.Sequential(
            OrderedDict([
                ("layer1", nn.Linear(input_dim, embedding_dim)),
                ("ln1", nn.LayerNorm(embedding_dim)),
                ("relu1", nn.ReLU()),
                ("dropout1", nn.Dropout(self.config.predictor_dropout)),
                ("layer2", nn.Linear(embedding_dim, output_dim)),
            ])
        )
        self.static_predictor = predictor_head(embedding_dim, embedding_dim)

        if self.use_gru:
            self.dynamic_predictor = predictor_head(embedding_dim, embedding_dim)
            self.interaction_predictor = nn.Sequential(
                OrderedDict([
                    ("layer1", nn.Linear(embedding_dim * 2, embedding_dim * 2)),
                    ("ln1", nn.LayerNorm(embedding_dim * 2)),
                    ("relu1", nn.ReLU()),
                    ("dropout1", nn.Dropout(self.config.predictor_dropout)),
                    ("layer2", nn.Linear(embedding_dim * 2, embedding_dim)),
                ])
            )

            self.gating_network = nn.Sequential(OrderedDict([
                ('layer1', nn.Linear(embedding_dim * 2, embedding_dim)),
                ('relu1', nn.ReLU()),
                ('layer2', nn.Linear(embedding_dim, 1))
            ]))

    def _functional_linear(self, x, prefix, fast_weights):
        weight = fast_weights[f"{prefix}.layer1.weight"]
        bias = fast_weights[f"{prefix}.layer1.bias"]
        x = F.linear(x, weight, bias)

        ln_weight = fast_weights[f"{prefix}.ln1.weight"]
        ln_bias = fast_weights[f"{prefix}.ln1.bias"]
        x = F.layer_norm(x, (self.config.embedding_dim,), weight=ln_weight, bias=ln_bias)

        x = F.relu(x)
        x = F.dropout(x, p=self.config.predictor_dropout, training=self.training)

        weight = fast_weights[f"{prefix}.layer2.weight"]
        bias = fast_weights[f"{prefix}.layer2.bias"]
        x = F.linear(x, weight, bias)
        return x

    def _functional_forward(self, user_static_rep, user_dynamic_rep, target_item_emb, fast_weights):
        if fast_weights:
            final_static_rep = self._functional_linear(user_static_rep.unsqueeze(1), 'static', fast_weights)
        else:
            final_static_rep = self.static_predictor(user_static_rep.unsqueeze(1))

        if final_static_rep.size(1) != target_item_emb.size(1):
            final_static_rep = final_static_rep.expand(-1, target_item_emb.size(1), -1)

        static_scores = torch.sum(final_static_rep * target_item_emb, dim=-1)

        scores = static_scores
        alpha_weight = None

        if self.use_gru and user_dynamic_rep is not None:
            if fast_weights:
                final_dynamic_rep = self._functional_linear(user_dynamic_rep.unsqueeze(1), 'dynamic', fast_weights)
            else:
                final_dynamic_rep = self.dynamic_predictor(user_dynamic_rep.unsqueeze(1))

            if final_dynamic_rep.size(1) != target_item_emb.size(1):
                final_dynamic_rep = final_dynamic_rep.expand(-1, target_item_emb.size(1), -1)

            dynamic_scores = torch.sum(final_dynamic_rep * target_item_emb, dim=-1)

            interaction_scores = torch.zeros_like(static_scores)
            if self.config.fusion_ablation != 'no_residual':
                interaction_input = torch.cat([user_static_rep, user_dynamic_rep], dim=-1).unsqueeze(1)
                if interaction_input.size(1) != target_item_emb.size(1):
                    interaction_input = interaction_input.expand(-1, target_item_emb.size(1), -1)

                if fast_weights:
                    x = F.linear(interaction_input, fast_weights['interaction.layer1.weight'],
                                 fast_weights['interaction.layer1.bias'])
                    x = F.layer_norm(x, (self.config.embedding_dim * 2,), fast_weights['interaction.ln1.weight'],
                                     fast_weights['interaction.ln1.bias'])
                    x = F.relu(x)
                    x = F.dropout(x, p=self.config.predictor_dropout, training=self.training)
                    final_interaction_rep = F.linear(x, fast_weights['interaction.layer2.weight'],
                                                     fast_weights['interaction.layer2.bias'])
                else:
                    final_interaction_rep = self.interaction_predictor(interaction_input)
                interaction_scores = torch.sum(final_interaction_rep * target_item_emb, dim=-1)

            if self.config.fusion_ablation == 'no_gate':
                expert_scores = 0.0
                alpha_weight = None
            else:
                if hasattr(self, 'gating_network'):
                    gate_input = torch.cat([user_static_rep, user_dynamic_rep], dim=-1)
                    if fast_weights:
                        x = F.linear(gate_input, fast_weights['gate.layer1.weight'], fast_weights['gate.layer1.bias'])
                        x = F.relu(x)
                        gate_logits = F.linear(x, fast_weights['gate.layer2.weight'], fast_weights['gate.layer2.bias'])
                    else:
                        gate_logits = self.gating_network(gate_input)
                    gate = torch.sigmoid(gate_logits)
                    expert_scores = gate * static_scores + (1 - gate) * dynamic_scores
                    alpha_weight = gate
                else:
                    expert_scores = 0.5 * static_scores + 0.5 * dynamic_scores

            scores = interaction_scores + expert_scores

        return scores, alpha_weight

    def _maml_inner_update(self, fast_weights, user_static_rep_flat, user_dynamic_rep_flat, pos_emb_flat, neg_emb_adapt,
                           create_graph):
        pos_scores, _ = self._functional_forward(user_static_rep_flat, user_dynamic_rep_flat, pos_emb_flat,
                                                 fast_weights)
        neg_scores, _ = self._functional_forward(user_static_rep_flat, user_dynamic_rep_flat, neg_emb_adapt,
                                                 fast_weights)

        all_scores = torch.cat([pos_scores, neg_scores], dim=1)
        labels = torch.zeros(user_static_rep_flat.size(0), dtype=torch.long, device=all_scores.device)

        if self.config.inner_loop_mode == 'step_by_step':
            per_sample_loss = F.cross_entropy(all_scores / self.config.temperature, labels, reduction='none')
            seq_len = self.config.support_size - 1
            if per_sample_loss.size(0) % seq_len == 0:
                batch_size_curr = per_sample_loss.size(0) // seq_len
                loss_reshaped = per_sample_loss.view(batch_size_curr, seq_len)
                gamma = 0.9
                steps = torch.arange(seq_len, device=labels.device)
                time_weights = (gamma ** (seq_len - 1 - steps)).unsqueeze(0)
                avg_adapt_loss = (loss_reshaped * time_weights).sum() / (batch_size_curr * time_weights.sum())
            else:
                avg_adapt_loss = per_sample_loss.mean()
        else:
            avg_adapt_loss = F.cross_entropy(all_scores / self.config.temperature, labels)

        grads = torch.autograd.grad(avg_adapt_loss, fast_weights.values(), create_graph=create_graph, allow_unused=True)

        updated_fast_weights = OrderedDict()
        for (name, param), grad in zip(fast_weights.items(), grads):
            if grad is not None:
                clipped_grad = torch.clamp(grad, -self.config.inner_grad_clip, self.config.inner_grad_clip)
                updated_fast_weights[name] = param - self.config.local_lr * clipped_grad
            else:
                updated_fast_weights[name] = param
        return updated_fast_weights

    def get_user_representation_and_context(self, user_static_emb, support_seq_emb):
        user_dynamic_emb = None

        if self.use_gru and self.gru is not None and support_seq_emb.size(1) > 0:
            self.gru.flatten_parameters()
            gru_outputs, _ = self.gru(support_seq_emb)
            user_dynamic_emb = torch.mean(gru_outputs, dim=1)
            user_dynamic_emb = self.gru_dropout(user_dynamic_emb)
        else:
            if support_seq_emb.size(1) > 0:
                user_dynamic_emb = torch.mean(support_seq_emb, dim=1)
            else:
                emb_dim = user_static_emb.size(-1) if user_static_emb is not None else self.config.embedding_dim
                device = user_static_emb.device if user_static_emb is not None else 'cpu'
                batch_size = user_static_emb.size(0) if user_static_emb is not None else 1
                user_dynamic_emb = torch.zeros((batch_size, emb_dim), device=device)

        if user_static_emb is None:
            user_static_emb = torch.zeros_like(user_dynamic_emb)

        return user_static_emb, user_dynamic_emb

    def forward_pretrain(self, users, pos_items, neg_items):
        final_user_emb_all, final_item_emb_all = self.light_gcn()
        final_user_emb = final_user_emb_all[users]
        final_pos_item_emb = final_item_emb_all[pos_items]
        final_neg_item_emb = final_item_emb_all[neg_items]
        pos_scores = torch.sum(final_user_emb * final_pos_item_emb, dim=1)
        neg_scores = torch.sum(final_user_emb * final_neg_item_emb, dim=1)
        loss = F.softplus(neg_scores - pos_scores).mean()
        initial_user_emb = self.light_gcn.user_embedding(users)
        initial_pos_item_emb = self.light_gcn.item_embedding(pos_items)
        initial_neg_item_emb = self.light_gcn.item_embedding(neg_items)
        reg_loss = (initial_user_emb.norm(2).pow(2) + initial_pos_item_emb.norm(2).pow(2) + initial_neg_item_emb.norm(
            2).pow(2)) / len(users)
        return loss + self.config.gcn_reg_weight * reg_loss

    def forward_meta(self, batch):
        gcn_user_emb, gcn_item_emb = self._get_base_embeddings()
        user_ids, support_seqs, query_items = (batch["user_id"], batch["support_seq"], batch["query_item"])
        batch_size = support_seqs.size(0)
        if batch_size < 2 or support_seqs.size(1) <= 1: return None

        user_static_emb = gcn_user_emb[user_ids]

        support_size = support_seqs.size(1)
        support_seq_emb = gcn_item_emb[support_seqs]

        if support_size > 0:
            s = torch.cumsum(support_seq_emb, dim=1)
            n = torch.arange(1, support_size + 1, device=s.device).view(1, -1, 1)
            pseudo_static_seq = s / n
            pseudo_static_emb = pseudo_static_seq[:, -1, :]
            user_static_emb = user_static_emb + pseudo_static_emb

        fast_weights = OrderedDict()
        for name, param in self.static_predictor.named_parameters():
            fast_weights['static.' + name] = param
        if self.use_gru:
            for name, param in self.dynamic_predictor.named_parameters():
                fast_weights['dynamic.' + name] = param
            for name, param in self.interaction_predictor.named_parameters():
                fast_weights['interaction.' + name] = param
            for name, param in self.gating_network.named_parameters():
                fast_weights['gate.' + name] = param

        if self.use_gru and self.gru is not None:
            self.gru.flatten_parameters()
            gru_outputs, _ = self.gru(support_seq_emb)
            gru_seq_all = self.gru_dropout(gru_outputs)
            user_dynamic_emb_adapt_seq = gru_seq_all[:, :-1, :]
        else:
            s = torch.cumsum(support_seq_emb, dim=1)
            n = torch.arange(1, support_size + 1, device=s.device).view(1, -1, 1)
            user_dynamic_emb_adapt_seq = s / n
            user_dynamic_emb_adapt_seq = user_dynamic_emb_adapt_seq[:, :-1, :]

        user_static_emb_adapt_seq = user_static_emb.unsqueeze(1).expand(-1, support_size - 1, -1)
        target_items = support_seqs[:, 1:]
        pos_emb_adapt = gcn_item_emb[target_items]

        if self.config.inner_loop_mode == 'last_step_only':
            user_static_rep_flat = user_static_emb_adapt_seq[:, -1, :]
            user_dynamic_rep_flat = user_dynamic_emb_adapt_seq[:, -1, :]
            pos_emb_flat = pos_emb_adapt[:, -1, :].unsqueeze(1)
            num_adapt_instances = batch_size
        else:
            num_adapt_instances = batch_size * (support_size - 1)
            user_static_rep_flat = user_static_emb_adapt_seq.reshape(num_adapt_instances, -1)
            user_dynamic_rep_flat = user_dynamic_emb_adapt_seq.reshape(num_adapt_instances, -1)
            pos_emb_flat = pos_emb_adapt.reshape(num_adapt_instances, 1, -1)

        neg_items_adapt = torch.randint(
            0, self.config.num_items,
            (num_adapt_instances, self.config.local_neg_sample_size),
            device=user_static_emb.device
        )
        neg_emb_adapt = gcn_item_emb[neg_items_adapt]

        for _ in range(self.config.local_updates):
            fast_weights = self._maml_inner_update(fast_weights, user_static_rep_flat, user_dynamic_rep_flat,
                                                   pos_emb_flat, neg_emb_adapt,
                                                   create_graph=True)

        user_static_rep_final, user_dynamic_rep_final = self.get_user_representation_and_context(
            user_static_emb, gcn_item_emb[support_seqs])

        pos_item_emb = gcn_item_emb[query_items]
        neg_items_query = torch.randint(0, self.config.num_items, (batch_size, self.config.query_neg_sample_size),
                                        device=user_static_emb.device)
        collisions = (neg_items_query == query_items)
        if torch.any(collisions):
            neg_items_query[collisions] = (neg_items_query[collisions] + 1) % self.config.num_items
        neg_item_emb_query = gcn_item_emb[neg_items_query]
        candidate_items_emb = torch.cat([pos_item_emb, neg_item_emb_query], dim=1)

        all_scores, _ = self._functional_forward(user_static_rep_final, user_dynamic_rep_final, candidate_items_emb,
                                                 fast_weights)
        labels = torch.zeros(batch_size, dtype=torch.long, device=user_static_emb.device)
        query_loss = F.cross_entropy(all_scores / self.config.temperature, labels)

        return query_loss

    def forward_no_meta(self, batch):
        gcn_user_emb, gcn_item_emb = self._get_base_embeddings()
        user_ids, support_seqs, query_items = (batch["user_id"], batch["support_seq"], batch["query_item"])
        batch_size = support_seqs.size(0)
        if batch_size < 2 or support_seqs.size(1) <= 1: return None
        user_static_emb = gcn_user_emb[user_ids]
        support_seq_emb = gcn_item_emb[support_seqs]

        if support_seq_emb.size(1) > 0:
            pseudo_static_emb = torch.mean(support_seq_emb, dim=1)
            user_static_emb = user_static_emb + pseudo_static_emb

        user_static_rep_final, user_dynamic_rep_final = self.get_user_representation_and_context(
            user_static_emb, support_seq_emb)

        pos_item_emb = gcn_item_emb[query_items]
        neg_items_query = torch.randint(0, self.config.num_items, (batch_size, self.config.query_neg_sample_size),
                                        device=user_static_emb.device)
        collisions = (neg_items_query == query_items)
        if torch.any(collisions):
            neg_items_query[collisions] = (neg_items_query[collisions] + 1) % self.config.num_items
        neg_item_emb_query = gcn_item_emb[neg_items_query]
        candidate_items_emb = torch.cat([pos_item_emb, neg_item_emb_query], dim=1)

        all_scores, _ = self._functional_forward(user_static_rep_final, user_dynamic_rep_final, candidate_items_emb,
                                                 fast_weights=None)
        labels = torch.zeros(batch_size, dtype=torch.long, device=user_static_emb.device)
        query_loss = F.cross_entropy(all_scores / self.config.temperature, labels)
        return query_loss

    def evaluate_batch(self, batch, return_weights=False):
        gcn_user_emb, gcn_item_emb = self._get_base_embeddings()
        user_ids, support_seqs, query_items, neg_items = \
            batch['user_id'], batch['support_seq'], batch['query_item'], batch['neg_items']

        batch_size, support_size = support_seqs.size()
        user_static_emb = gcn_user_emb[user_ids]

        if support_size > 0:
            support_seq_emb_for_init = gcn_item_emb[support_seqs]
            pseudo_static_emb = torch.mean(support_seq_emb_for_init, dim=1)
            user_static_emb = user_static_emb + pseudo_static_emb

        fast_weights = OrderedDict()
        for name, param in self.static_predictor.named_parameters():
            fast_weights['static.' + name] = param
        if self.use_gru:
            for name, param in self.dynamic_predictor.named_parameters():
                fast_weights['dynamic.' + name] = param
            for name, param in self.interaction_predictor.named_parameters():
                fast_weights['interaction.' + name] = param
            for name, param in self.gating_network.named_parameters():
                fast_weights['gate.' + name] = param

        if support_size > 1:
            with torch.enable_grad():
                support_seq_emb = gcn_item_emb[support_seqs]

                if self.use_gru and self.gru is not None:
                    self.gru.flatten_parameters()
                    gru_outputs, _ = self.gru(support_seq_emb)
                    gru_seq_all = self.gru_dropout(gru_outputs)
                    user_dynamic_emb_adapt_seq = gru_seq_all[:, :-1, :]
                else:
                    s = torch.cumsum(support_seq_emb, dim=1)
                    n = torch.arange(1, support_size + 1, device=s.device).view(1, -1, 1)
                    user_dynamic_emb_adapt_seq = s / n
                    user_dynamic_emb_adapt_seq = user_dynamic_emb_adapt_seq[:, :-1, :]

                user_static_emb_adapt_seq = user_static_emb.unsqueeze(1).expand(-1, support_size - 1, -1)
                target_items = support_seqs[:, 1:]
                pos_emb_adapt = gcn_item_emb[target_items]

                if self.config.inner_loop_mode == 'last_step_only':
                    user_static_rep_flat = user_static_emb_adapt_seq[:, -1, :]
                    user_dynamic_rep_flat = user_dynamic_emb_adapt_seq[:, -1, :]
                    pos_emb_flat = pos_emb_adapt[:, -1, :].unsqueeze(1)
                    num_adapt_instances = batch_size
                else:
                    num_adapt_instances = batch_size * (support_size - 1)
                    user_static_rep_flat = user_static_emb_adapt_seq.reshape(num_adapt_instances, -1)
                    user_dynamic_rep_flat = user_dynamic_emb_adapt_seq.reshape(num_adapt_instances, -1)
                    pos_emb_flat = pos_emb_adapt.reshape(num_adapt_instances, 1, -1)

                neg_items_adapt = torch.randint(
                    0, self.config.num_items,
                    (num_adapt_instances, self.config.local_neg_sample_size),
                    device=user_static_emb.device)
                neg_emb_adapt = gcn_item_emb[neg_items_adapt]

                for _ in range(self.config.local_updates):
                    fast_weights = self._maml_inner_update(fast_weights, user_static_rep_flat, user_dynamic_rep_flat,
                                                           pos_emb_flat, neg_emb_adapt,
                                                           create_graph=False)

        with torch.no_grad():
            support_seq_emb = gcn_item_emb[support_seqs]
            user_static_rep_eval, user_dynamic_rep_eval = self.get_user_representation_and_context(
                user_static_emb, support_seq_emb)

            eval_items = torch.cat([query_items, neg_items], dim=1)
            scores, weights = self._functional_forward(user_static_rep_eval, user_dynamic_rep_eval,
                                                       gcn_item_emb[eval_items],
                                                       fast_weights)

        if return_weights:
            return scores, weights
        return scores

    def evaluate_batch_no_meta(self, batch, return_weights=False):
        with torch.no_grad():
            gcn_user_emb, gcn_item_emb = self._get_base_embeddings()
            user_ids, support_seqs, query_items, neg_items = \
                batch['user_id'], batch['support_seq'], batch['query_item'], batch['neg_items']
            user_static_emb = gcn_user_emb[user_ids]
            support_seq_emb = gcn_item_emb[support_seqs]

            if support_seq_emb.size(1) > 0:
                pseudo_static_emb = torch.mean(support_seq_emb, dim=1)
                user_static_emb = user_static_emb + pseudo_static_emb

            user_static_rep_eval, user_dynamic_rep_eval = self.get_user_representation_and_context(
                user_static_emb, support_seq_emb)

            eval_items = torch.cat([query_items, neg_items], dim=1)
            scores, weights = self._functional_forward(user_static_rep_eval, user_dynamic_rep_eval,
                                                       gcn_item_emb[eval_items],
                                                       fast_weights=None)
        if return_weights:
            return scores, weights


        return scores
