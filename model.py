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
        
        self.light_gcn = LightGCN(
            num_users=config.num_users,
            num_items=config.num_items,
            embedding_dim=config.embedding_dim,
            interaction_matrix=interaction_matrix,
            num_layers=config.gnn_layers,
        )

        self.item_genre_features = None
        if item_genre_features is not None:
            if sp.issparse(item_genre_features):
                item_genre_features = item_genre_features.toarray()
            self.item_genre_features = torch.tensor(item_genre_features, dtype=torch.float)
            feature_dim = self.item_genre_features.shape[1]
            self.feature_projector = nn.Linear(feature_dim, config.embedding_dim)

        self.gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.embedding_dim,
            num_layers=self.config.gru_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.gru_layers > 1 else 0,
        )
        self.gru_dropout = nn.Dropout(self.config.dropout)
        
        self._initialize_meta_components()

    def _get_base_embeddings(self):
        user_emb, item_emb = self.light_gcn()
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
            static_scores = torch.sum(final_static_rep * target_item_emb, dim=-1)
            final_dynamic_rep = self._functional_linear(user_dynamic_rep.unsqueeze(1), 'dynamic', fast_weights)
            dynamic_scores = torch.sum(final_dynamic_rep * target_item_emb, dim=-1)
            interaction_input = torch.cat([user_static_rep, user_dynamic_rep], dim=-1).unsqueeze(1)
            x = F.linear(interaction_input, fast_weights['interaction.layer1.weight'],
                         fast_weights['interaction.layer1.bias'])
            x = F.layer_norm(x, (self.config.embedding_dim * 2,), fast_weights['interaction.ln1.weight'],
                             fast_weights['interaction.ln1.bias'])
            x = F.relu(x)
            x = F.dropout(x, p=self.config.predictor_dropout, training=self.training)
            final_interaction_rep = F.linear(x, fast_weights['interaction.layer2.weight'],
                                             fast_weights['interaction.layer2.bias'])
            interaction_scores = torch.sum(final_interaction_rep * target_item_emb, dim=-1)
            gate_input = torch.cat([user_static_rep, user_dynamic_rep], dim=-1)
            x = F.linear(gate_input, fast_weights['gate.layer1.weight'], fast_weights['gate.layer1.bias'])
            x = F.relu(x)
            gate_logits = F.linear(x, fast_weights['gate.layer2.weight'], fast_weights['gate.layer2.bias'])
        else:
            final_static_rep = self.static_predictor(user_static_rep.unsqueeze(1))
            static_scores = torch.sum(final_static_rep * target_item_emb, dim=-1)
            final_dynamic_rep = self.dynamic_predictor(user_dynamic_rep.unsqueeze(1))
            dynamic_scores = torch.sum(final_dynamic_rep * target_item_emb, dim=-1)
            interaction_input = torch.cat([user_static_rep, user_dynamic_rep], dim=-1).unsqueeze(1)
            final_interaction_rep = self.interaction_predictor(interaction_input)
            interaction_scores = torch.sum(final_interaction_rep * target_item_emb, dim=-1)
            gate_input = torch.cat([user_static_rep, user_dynamic_rep], dim=-1)
            gate_logits = self.gating_network(gate_input)

        gate = torch.sigmoid(gate_logits)
        expert_scores = gate * static_scores + (1 - gate) * dynamic_scores
        scores = interaction_scores + expert_scores
        return scores

    def _maml_inner_update(self, fast_weights, user_static_rep_flat, user_dynamic_rep_flat, pos_emb_flat, neg_emb_adapt, create_graph):
        pos_scores = self._functional_forward(user_static_rep_flat, user_dynamic_rep_flat, pos_emb_flat,
                                              fast_weights)
        neg_scores = self._functional_forward(user_static_rep_flat, user_dynamic_rep_flat, neg_emb_adapt,
                                              fast_weights)

        all_scores = torch.cat([pos_scores, neg_scores], dim=1)
        labels = torch.zeros(user_static_rep_flat.size(0), dtype=torch.long, device=all_scores.device)
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

    def get_user_representation(self, user_static_emb, support_seq_emb):
        user_dynamic_emb = None
        if support_seq_emb.size(1) > 0:
            self.gru.flatten_parameters()
            gru_outputs, _ = self.gru(support_seq_emb)
            user_dynamic_emb = torch.mean(gru_outputs, dim=1)
            user_dynamic_emb = self.gru_dropout(user_dynamic_emb)
        else:
            emb_dim = user_static_emb.size(-1)
            device = user_static_emb.device
            batch_size = user_static_emb.size(0)
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

        fast_weights = OrderedDict()
        for name, param in self.static_predictor.named_parameters():
            fast_weights['static.' + name] = param
        for name, param in self.dynamic_predictor.named_parameters():
            fast_weights['dynamic.' + name] = param
        for name, param in self.interaction_predictor.named_parameters():
            fast_weights['interaction.' + name] = param
        for name, param in self.gating_network.named_parameters():
            fast_weights['gate.' + name] = param

        support_size = support_seqs.size(1)
        support_seq_emb = gcn_item_emb[support_seqs]

        self.gru.flatten_parameters()
        gru_outputs, _ = self.gru(support_seq_emb)
        user_dynamic_emb_adapt_seq = self.gru_dropout(gru_outputs[:, :-1, :])
        
        user_static_emb_adapt_seq = user_static_emb.unsqueeze(1).expand(-1, support_size - 1, -1)
        target_items = support_seqs[:, 1:]
        pos_emb_adapt = gcn_item_emb[target_items]

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

        user_static_rep_final, user_dynamic_rep_final = self.get_user_representation(user_static_emb,
                                                                                     gcn_item_emb[support_seqs])
        pos_item_emb = gcn_item_emb[query_items]
        neg_items_query = torch.randint(0, self.config.num_items, (batch_size, self.config.query_neg_sample_size),
                                        device=user_static_emb.device)
        collisions = (neg_items_query == query_items)
        if torch.any(collisions):
            neg_items_query[collisions] = (neg_items_query[collisions] + 1) % self.config.num_items
        neg_item_emb_query = gcn_item_emb[neg_items_query]
        candidate_items_emb = torch.cat([pos_item_emb, neg_item_emb_query], dim=1)
        all_scores = self._functional_forward(user_static_rep_final, user_dynamic_rep_final, candidate_items_emb,
                                              fast_weights)
        labels = torch.zeros(batch_size, dtype=torch.long, device=user_static_emb.device)
        query_loss = F.cross_entropy(all_scores / self.config.temperature, labels)

        return query_loss

    def evaluate_batch(self, batch, gcn_user_emb, gcn_item_emb):
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
        for name, param in self.dynamic_predictor.named_parameters():
            fast_weights['dynamic.' + name] = param
        for name, param in self.interaction_predictor.named_parameters():
            fast_weights['interaction.' + name] = param
        for name, param in self.gating_network.named_parameters():
            fast_weights['gate.' + name] = param

        if support_size > 1:
            with torch.enable_grad():
                support_seq_emb = gcn_item_emb[support_seqs]
                self.gru.flatten_parameters()
                gru_outputs, _ = self.gru(support_seq_emb)
                user_dynamic_emb_adapt_seq = self.gru_dropout(gru_outputs[:, :-1, :])
                
                user_static_emb_adapt_seq = user_static_emb.unsqueeze(1).expand(-1, support_size - 1, -1)
                target_items = support_seqs[:, 1:]
                pos_emb_adapt = gcn_item_emb[target_items]

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
            user_static_rep_eval, user_dynamic_rep_eval = self.get_user_representation(user_static_emb,
                                                                                       gcn_item_emb[support_seqs])
            eval_items = torch.cat([query_items, neg_items], dim=1)
            scores = self._functional_forward(user_static_rep_eval, user_dynamic_rep_eval, gcn_item_emb[eval_items],
                                              fast_weights)
        return scores