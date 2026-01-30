import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import copy
from modules import LightGCN


class SASRec(nn.Module):
    def __init__(self, config):
        super(SASRec, self).__init__()
        self.config = config
        self.item_embedding = nn.Embedding(config.num_items + 1, config.embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(config.support_size, config.embedding_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layernorm = nn.LayerNorm(config.embedding_dim, eps=1e-8)
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(config) for _ in range(config.sasrec_num_blocks)
        ])
        self.final_layernorm = nn.LayerNorm(config.embedding_dim, eps=1e-8)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        support_seq = batch['support_seq'] + 1

        if support_seq.size(1) > self.config.support_size:
            support_seq = support_seq[:, -self.config.support_size:]

        query_item = batch['query_item'] + 1
        batch_size = support_seq.size(0)
        device = support_seq.device
        num_neg_samples = self.config.local_neg_sample_size
        neg_items = torch.randint(0, self.config.num_items,
                                  (batch_size, num_neg_samples),
                                  device=device) + 1
        seq_output = self._forward_seq(support_seq)
        last_item_reps = seq_output[:, -1, :]
        pos_item_embs = self.item_embedding(query_item)
        neg_item_embs = self.item_embedding(neg_items)
        candidate_embs = torch.cat([pos_item_embs, neg_item_embs], dim=1)
        scores = torch.bmm(candidate_embs, last_item_reps.unsqueeze(-1)).squeeze(-1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        loss = self.criterion(scores, labels)
        return loss

    def _forward_seq(self, seq):
        seq_embedded = self.item_embedding(seq)
        seq_embedded *= self.config.embedding_dim ** 0.5
        positions = torch.arange(seq.size(1), dtype=torch.long, device=seq.device).unsqueeze(0)
        seq_embedded += self.position_embedding(positions)
        seq_embedded = self.embedding_dropout(seq_embedded)
        attention_mask = (seq == 0).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.triu(torch.ones((seq.size(1), seq.size(1)), dtype=torch.bool, device=seq.device),
                                 diagonal=1).unsqueeze(0).unsqueeze(0)
        attention_mask = torch.logical_or(attention_mask, causal_mask)
        for block in self.attention_blocks:
            seq_embedded = block(seq_embedded, attention_mask)
        seq_embedded = self.final_layernorm(seq_embedded)
        return seq_embedded

    def evaluate_batch(self, batch):
        support_seq = batch['support_seq'] + 1

        if support_seq.size(1) > self.config.support_size:
            support_seq = support_seq[:, -self.config.support_size:]

        eval_items = torch.cat([batch['query_item'], batch['neg_items']], dim=1) + 1
        seq_output = self._forward_seq(support_seq)
        last_item_reps = seq_output[:, -1, :]
        item_embs = self.item_embedding(eval_items)
        scores = torch.bmm(item_embs, last_item_reps.unsqueeze(-1)).squeeze(-1)
        return scores


class AttentionBlock(nn.Module):
    def __init__(self, config):
        super(AttentionBlock, self).__init__()
        self.attention = MultiHeadAttention(config)
        self.point_wise_feed_forward = PointWiseFeedForward(config)
        self.layernorm1 = nn.LayerNorm(config.embedding_dim, eps=1e-8)
        self.layernorm2 = nn.LayerNorm(config.embedding_dim, eps=1e-8)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask):
        x_norm = self.layernorm1(x)
        attn_output = self.attention(x_norm, mask)
        x = x + self.dropout(attn_output)
        x_norm = self.layernorm2(x)
        ff_output = self.point_wise_feed_forward(x_norm)
        x = x + ff_output
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = config.sasrec_num_heads
        self.hidden_size = config.embedding_dim
        self.head_size = self.hidden_size // self.num_heads
        self.q_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask):
        batch_size = x.size(0)
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size ** 0.5)
        scores.masked_fill_(mask, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        return self.output_linear(context)


class PointWiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(config.embedding_dim, config.embedding_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(config.embedding_dim, config.embedding_dim, kernel_size=1)
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.conv2(x))
        return x.transpose(1, 2)


class FMLPRec_LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(FMLPRec_LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class FMLPRec_FilterLayer(nn.Module):
    def __init__(self, config):
        super(FMLPRec_FilterLayer, self).__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(1, config.support_size // 2 + 1, config.embedding_dim, 2, dtype=torch.float32) * 0.01)
        self.out_dropout = nn.Dropout(config.dropout)
        self.LayerNorm = FMLPRec_LayerNorm(config.embedding_dim, eps=1e-12)

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FMLPRec_Intermediate(nn.Module):
    def __init__(self, config):
        super(FMLPRec_Intermediate, self).__init__()
        self.dense_1 = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.intermediate_act_fn = F.gelu
        self.dense_2 = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.LayerNorm = FMLPRec_LayerNorm(config.embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FMLPRec_Layer(nn.Module):
    def __init__(self, config):
        super(FMLPRec_Layer, self).__init__()
        self.filterlayer = FMLPRec_FilterLayer(config)
        self.intermediate = FMLPRec_Intermediate(config)

    def forward(self, hidden_states):
        hidden_states = self.filterlayer(hidden_states)
        intermediate_output = self.intermediate(hidden_states)
        return intermediate_output


class FMLPRec_Encoder(nn.Module):
    def __init__(self, config):
        super(FMLPRec_Encoder, self).__init__()
        layer = FMLPRec_Layer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.fmlprec_num_blocks)])

    def forward(self, hidden_states):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class FMLPRec(nn.Module):
    def __init__(self, config):
        super(FMLPRec, self).__init__()
        self.config = config
        self.item_embeddings = nn.Embedding(config.num_items + 1, config.embedding_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.support_size, config.embedding_dim)
        self.LayerNorm = FMLPRec_LayerNorm(config.embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)
        self.item_encoder = FMLPRec_Encoder(config)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, FMLPRec_LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _forward_encoder(self, support_seq):
        current_seq_len = support_seq.size(1)
        target_len = self.config.support_size
        if current_seq_len > target_len:
            support_seq = support_seq[:, -target_len:]
        elif current_seq_len < target_len:
            padding_size = target_len - current_seq_len
            padding = torch.zeros((support_seq.size(0), padding_size), dtype=torch.long, device=support_seq.device)
            support_seq = torch.cat([padding, support_seq], dim=1)
        seq_length = support_seq.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=support_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(support_seq)
        item_embeddings = self.item_embeddings(support_seq)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        sequence_output = self.item_encoder(sequence_emb)
        return sequence_output

    def forward(self, batch):
        support_seqs = batch["support_seq"] + 1
        query_items = batch["query_item"] + 1
        batch_size = support_seqs.size(0)
        device = support_seqs.device
        sequence_output = self._forward_encoder(support_seqs)
        seq_emb = sequence_output[:, -1, :]
        pos_item_emb = self.item_embeddings(query_items)
        neg_items = torch.randint(0, self.config.num_items, (batch_size, self.config.query_neg_sample_size),
                                  device=device) + 1
        collisions = (neg_items == query_items)
        while torch.any(collisions):
            new_randoms = torch.randint(0, self.config.num_items, (collisions.sum(),), device=device) + 1
            neg_items.masked_scatter_(collisions, new_randoms)
            collisions = (neg_items == query_items)
        neg_item_emb = self.item_embeddings(neg_items)
        candidate_items_emb = torch.cat([pos_item_emb, neg_item_emb], dim=1)
        seq_emb_expanded = seq_emb.unsqueeze(1).expand(-1, candidate_items_emb.size(1), -1)
        all_scores = torch.sum(seq_emb_expanded * candidate_items_emb, dim=-1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        loss = F.cross_entropy(all_scores / self.config.temperature, labels)
        return loss

    def evaluate_batch(self, batch):
        with torch.no_grad():
            support_seqs = batch['support_seq'] + 1
            query_items = batch['query_item'] + 1
            neg_items = batch['neg_items'] + 1
            eval_items = torch.cat([query_items, neg_items], dim=1)
            sequence_output = self._forward_encoder(support_seqs)
            seq_embs = sequence_output[:, -1, :]
            item_embs = self.item_embeddings(eval_items)
            scores = (seq_embs.unsqueeze(1) * item_embs).sum(dim=-1)
        return scores


class LightGCN_Baseline(nn.Module):
    def __init__(self, config, interaction_matrix):
        super(LightGCN_Baseline, self).__init__()
        self.config = config
        self.light_gcn = LightGCN(
            num_users=config.num_users,
            num_items=config.num_items,
            embedding_dim=config.embedding_dim,
            interaction_matrix=interaction_matrix,
            num_layers=config.gnn_layers
        )

    def forward(self, users, pos_items, neg_items):
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
        reg_loss = (initial_user_emb.norm(2).pow(2) +
                    initial_pos_item_emb.norm(2).pow(2) +
                    initial_neg_item_emb.norm(2).pow(2)) / len(users)
        return loss + self.config.gcn_reg_weight * reg_loss

    def evaluate_batch(self, batch):
        with torch.no_grad():
            user_emb_all, item_emb_all = self.light_gcn()
            user_ids, query_items, neg_items = batch['user_id'], batch['query_item'], batch['neg_items']
            user_embs = user_emb_all[user_ids]
            eval_items = torch.cat([query_items, neg_items], dim=1)
            item_embs = item_emb_all[eval_items]
            scores = (user_embs.unsqueeze(1) * item_embs).sum(dim=-1)
        return scores


class GRU4Rec(nn.Module):
    def __init__(self, config):
        super(GRU4Rec, self).__init__()
        self.config = config
        self.item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.item_bias = nn.Embedding(config.num_items, 1)
        nn.init.zeros_(self.item_bias.weight)
        self.gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.embedding_dim,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0,
        )
        self.emb_dropout = nn.Dropout(config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)

    def _get_user_representation(self, support_seq_emb):
        if support_seq_emb.size(1) > 0:
            self.gru.flatten_parameters()
            _, h_n = self.gru(support_seq_emb)
            return h_n.squeeze(0)
        else:
            return torch.zeros((support_seq_emb.size(0), self.config.embedding_dim), device=support_seq_emb.device)

    def forward(self, batch):
        item_emb_all = self.item_embedding.weight
        support_seqs = batch["support_seq"]
        pos_items = batch["pos_items"]
        neg_items = batch["neg_items"]

        seq_emb = item_emb_all[support_seqs]
        seq_emb = self.emb_dropout(seq_emb)

        user_dynamic_emb = self._get_user_representation(seq_emb)
        user_dynamic_emb = self.out_dropout(user_dynamic_emb)

        pos_item_emb = item_emb_all[pos_items]
        neg_item_emb = item_emb_all[neg_items]

        pos_bias = self.item_bias(pos_items).squeeze(-1)
        neg_bias = self.item_bias(neg_items).squeeze(-1)

        user_dynamic_emb_expanded = user_dynamic_emb.unsqueeze(1)

        pos_scores = torch.sum(user_dynamic_emb * pos_item_emb, dim=-1) + pos_bias
        neg_scores = torch.sum(user_dynamic_emb_expanded * neg_item_emb, dim=-1) + neg_bias

        softmax_scores = F.softmax(neg_scores, dim=-1)
        loss = -torch.log(
            torch.sum(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) * softmax_scores, dim=1) + 1e-9).mean()

        return loss

    def evaluate_batch(self, batch):
        with torch.no_grad():
            item_emb_all = self.item_embedding.weight
            support_seqs, query_items, neg_items = \
                batch['support_seq'], batch['query_item'], batch['neg_items']

            seq_emb = item_emb_all[support_seqs]
            user_pred_embs = self._get_user_representation(seq_emb)

            eval_items = torch.cat([query_items, neg_items], dim=1)
            item_embs = item_emb_all[eval_items]
            item_biases = self.item_bias(eval_items).squeeze(-1)

            scores = (user_pred_embs.unsqueeze(1) * item_embs).sum(dim=-1) + item_biases
        return scores


class MeLU(nn.Module):
    def __init__(self, config):
        super(MeLU, self).__init__()
        self.config = config
        self.user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        predictor_input_dim = config.embedding_dim * 2
        self.predictor = nn.Sequential(
            OrderedDict([
                ("layer1", nn.Linear(predictor_input_dim, config.embedding_dim * 2)),
                ("ln1", nn.LayerNorm(config.embedding_dim * 2)),
                ("relu1", nn.ReLU()),
                ("dropout1", nn.Dropout(config.predictor_dropout)),
                ("layer2", nn.Linear(config.embedding_dim * 2, 1)),
            ])
        )

    def _get_base_embeddings(self):
        return self.user_embedding.weight, self.item_embedding.weight

    def _functional_forward_melu(self, user_emb, item_emb, fast_weights):
        user_item_cat = torch.cat([user_emb, item_emb], dim=-1)
        x = user_item_cat
        if fast_weights is None:
            scores = self.predictor(x)
        else:
            x = F.linear(x, weight=fast_weights["layer1.weight"], bias=fast_weights["layer1.bias"])
            x = F.layer_norm(x, (self.config.embedding_dim * 2,), weight=fast_weights["ln1.weight"],
                             bias=fast_weights["ln1.bias"])
            x = F.relu(x)
            x = F.dropout(x, p=self.config.predictor_dropout, training=self.training)
            scores = F.linear(x, weight=fast_weights["layer2.weight"], bias=fast_weights["layer2.bias"])
        return scores.squeeze(-1)

    def _maml_inner_update(self, fast_weights, user_rep_flat, pos_emb_flat, neg_emb_adapt, create_graph):
        pos_scores = self._functional_forward_melu(user_rep_flat, pos_emb_flat, fast_weights)
        pos_scores = pos_scores.unsqueeze(1)
        user_rep_for_neg = user_rep_flat.unsqueeze(1).expand(-1, self.config.local_neg_sample_size, -1)
        neg_scores = self._functional_forward_melu(user_rep_for_neg, neg_emb_adapt, fast_weights)

        all_scores = torch.cat([pos_scores, neg_scores], dim=1)
        labels = torch.zeros(user_rep_flat.size(0), dtype=torch.long, device=all_scores.device)
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

    def forward(self, batch):
        user_ids, support_seqs, query_items = (batch["user_id"], batch["support_seq"], batch["query_item"])
        batch_size = support_seqs.size(0)
        user_emb, item_emb = self._get_base_embeddings()
        user_static_emb = user_emb[user_ids]
        fast_weights = OrderedDict(self.predictor.named_parameters())

        if support_seqs.size(1) > 1:
            support_size = support_seqs.size(1)
            user_rep_adapt = user_static_emb.unsqueeze(1).expand(-1, support_size - 1, -1)
            target_items = support_seqs[:, 1:]
            pos_emb_adapt = item_emb[target_items]
            num_adapt_instances = batch_size * (support_size - 1)
            user_rep_flat = user_rep_adapt.reshape(num_adapt_instances, -1)
            pos_emb_flat = pos_emb_adapt.reshape(num_adapt_instances, -1)
            neg_items_adapt = torch.randint(0, self.config.num_items,
                                            (num_adapt_instances, self.config.local_neg_sample_size),
                                            device=user_ids.device)
            neg_emb_adapt = item_emb[neg_items_adapt]
            for _ in range(self.config.local_updates):
                fast_weights = self._maml_inner_update(fast_weights, user_rep_flat, pos_emb_flat, neg_emb_adapt, create_graph=True)

        pos_item_emb_query = item_emb[query_items.squeeze(-1)].unsqueeze(1)
        neg_items_query = torch.randint(0, self.config.num_items, (batch_size, self.config.query_neg_sample_size),
                                        device=user_ids.device)
        neg_item_emb_query = item_emb[neg_items_query]
        candidate_items_emb = torch.cat([pos_item_emb_query, neg_item_emb_query], dim=1)
        user_emb_for_query = user_static_emb.unsqueeze(1).expand(-1, candidate_items_emb.size(1), -1)
        all_scores = self._functional_forward_melu(user_emb_for_query, candidate_items_emb, fast_weights)
        labels = torch.zeros(batch_size, dtype=torch.long, device=user_ids.device)
        query_loss = F.cross_entropy(all_scores / self.config.temperature, labels)
        return query_loss

    def evaluate_batch(self, batch):
        user_emb_all, item_emb_all = self._get_base_embeddings()
        user_ids, support_seqs, query_items, neg_items = \
            batch['user_id'], batch['support_seq'], batch['query_item'], batch['neg_items']
        user_embs = user_emb_all[user_ids]
        batch_size, support_size = support_seqs.size()
        fast_weights = OrderedDict(self.predictor.named_parameters())

        if support_size > 1:
            with torch.enable_grad():
                user_rep_adapt = user_embs.unsqueeze(1).expand(-1, support_size - 1, -1)
                target_items = support_seqs[:, 1:]
                pos_emb_adapt = item_emb_all[target_items]
                num_adapt_instances = batch_size * (support_size - 1)
                user_rep_flat = user_rep_adapt.reshape(num_adapt_instances, -1)
                pos_emb_flat = pos_emb_adapt.reshape(num_adapt_instances, -1)
                neg_items_adapt = torch.randint(0, self.config.num_items,
                                                (num_adapt_instances, self.config.local_neg_sample_size),
                                                device=user_ids.device)
                neg_emb_adapt = item_emb_all[neg_items_adapt]
                for _ in range(self.config.local_updates):
                    fast_weights = self._maml_inner_update(fast_weights, user_rep_flat, pos_emb_flat, neg_emb_adapt, create_graph=False)

        with torch.no_grad():
            eval_items = torch.cat([query_items, neg_items], dim=1)
            item_embs_eval = item_emb_all[eval_items]
            user_embs_eval = user_embs.unsqueeze(1).expand(-1, eval_items.size(1), -1)
            scores = self._functional_forward_melu(user_embs_eval, item_embs_eval, fast_weights)
        return scores


class MAMO(nn.Module):
    def __init__(self, config):
        super(MAMO, self).__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.n_k = 3
        self.tao = 0.01

        self.user_embedding = nn.Embedding(config.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, self.embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        self.feature_memory = nn.Parameter(torch.randn(self.n_k, self.embedding_dim))
        self.rec_in_dim = self.embedding_dim * 2
        self.task_memory = nn.Parameter(torch.randn(self.n_k, self.rec_in_dim, self.rec_in_dim))

        nn.init.normal_(self.feature_memory, std=0.01)
        nn.init.normal_(self.task_memory, std=0.01)

        self.att_layer = nn.Linear(self.n_k, self.n_k)
        self.att_act = nn.Softmax(dim=-1)

        self.fc_layers = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(self.rec_in_dim, self.embedding_dim)),
            ('relu1', nn.ReLU()),
            ('layer2', nn.Linear(self.embedding_dim, 1))
        ]))

    def get_cosine_attention(self, query, memory_keys):
        query_norm = F.normalize(query, p=2, dim=1)
        keys_norm = F.normalize(memory_keys, p=2, dim=1)
        sim = torch.mm(query_norm, keys_norm.t())
        att = self.att_layer(sim)
        return self.att_act(att)

    def functional_forward(self, user_emb, item_emb, mem_layer_weights, fast_weights_fc):
        x = torch.cat([user_emb, item_emb], dim=-1)

        if mem_layer_weights.dim() == 3:
            x = torch.bmm(x, mem_layer_weights)
        else:
            x = F.linear(x, mem_layer_weights)

        for name, param in fast_weights_fc.items():
            if 'weight' in name:
                bias_name = name.replace('weight', 'bias')
                bias = fast_weights_fc[bias_name]
                if 'layer1' in name:
                    x = F.linear(x, param, bias)
                    x = F.relu(x)
                elif 'layer2' in name:
                    x = F.linear(x, param, bias)

        return x.squeeze(-1)

    def forward(self, batch):
        user_ids = batch['user_id']
        support_seq = batch['support_seq']
        query_item = batch['query_item']

        batch_size = user_ids.size(0)

        p_u = self.user_embedding(user_ids)
        att_values = self.get_cosine_attention(p_u, self.feature_memory)
        bias_term = torch.mm(att_values, self.feature_memory)
        user_emb_initial = p_u - self.tao * bias_term

        att_values_expanded = att_values.view(batch_size, self.n_k, 1, 1)
        task_memory_expanded = self.task_memory.unsqueeze(0).expand(batch_size, -1, -1, -1)
        mem_layer_weights_initial = (att_values_expanded * task_memory_expanded).sum(dim=1)

        pos_items = support_seq
        neg_items = torch.randint(0, self.config.num_items, pos_items.shape, device=user_ids.device)
        support_items = torch.cat([pos_items, neg_items], dim=1)
        support_labels = torch.cat([torch.ones_like(pos_items), torch.zeros_like(neg_items)], dim=1)

        fast_user_emb = user_emb_initial
        fast_mem_weights = mem_layer_weights_initial
        fast_fc_weights = OrderedDict(self.fc_layers.named_parameters())
        item_emb_weight = self.item_embedding.weight

        for _ in range(self.config.local_updates):
            user_emb_exp = fast_user_emb.unsqueeze(1).expand(-1, support_items.size(1), -1)
            item_emb_supp = item_emb_weight[support_items]

            logits = self.functional_forward(user_emb_exp, item_emb_supp, fast_mem_weights, fast_fc_weights)
            loss = F.binary_cross_entropy_with_logits(logits, support_labels.float())

            grads = torch.autograd.grad(
                loss,
                [fast_user_emb, fast_mem_weights] + list(fast_fc_weights.values()),
                create_graph=True,
                allow_unused=True
            )

            lr = self.config.local_lr
            fast_user_emb = fast_user_emb - lr * grads[0]
            fast_mem_weights = fast_mem_weights - lr * grads[1]

            new_fc_weights = OrderedDict()
            for i, (name, param) in enumerate(fast_fc_weights.items()):
                if grads[i + 2] is not None:
                    new_fc_weights[name] = param - lr * grads[i + 2]
                else:
                    new_fc_weights[name] = param
            fast_fc_weights = new_fc_weights

        query_item_emb = item_emb_weight[query_item].squeeze(1)
        neg_query_items = torch.randint(
            0, self.config.num_items,
            (batch_size, self.config.query_neg_sample_size),
            device=user_ids.device
        )
        neg_query_emb = item_emb_weight[neg_query_items]

        candidate_embs = torch.cat([query_item_emb.unsqueeze(1), neg_query_emb], dim=1)
        user_emb_query = fast_user_emb.unsqueeze(1).expand(-1, candidate_embs.size(1), -1)

        logits = self.functional_forward(user_emb_query, candidate_embs, fast_mem_weights, fast_fc_weights)
        labels = torch.zeros(batch_size, dtype=torch.long, device=user_ids.device)
        loss = F.cross_entropy(logits / self.config.temperature, labels)

        return loss

    def evaluate_batch(self, batch):
        user_ids = batch['user_id']
        support_seq = batch['support_seq']
        query_item = batch['query_item']
        neg_items = batch['neg_items']
        batch_size = user_ids.size(0)

        p_u = self.user_embedding(user_ids)
        att_values = self.get_cosine_attention(p_u, self.feature_memory)
        bias_term = torch.mm(att_values, self.feature_memory)
        user_emb_initial = p_u - self.tao * bias_term

        att_values_expanded = att_values.view(batch_size, self.n_k, 1, 1)
        task_memory_expanded = self.task_memory.unsqueeze(0).expand(batch_size, -1, -1, -1)
        mem_layer_weights_initial = (att_values_expanded * task_memory_expanded).sum(dim=1)

        pos_items = support_seq
        neg_support = torch.randint(0, self.config.num_items, pos_items.shape, device=user_ids.device)
        support_items = torch.cat([pos_items, neg_support], dim=1)
        support_labels = torch.cat([torch.ones_like(pos_items), torch.zeros_like(neg_support)], dim=1)

        fast_user_emb = user_emb_initial
        fast_mem_weights = mem_layer_weights_initial
        fast_fc_weights = OrderedDict(self.fc_layers.named_parameters())
        item_emb_weight = self.item_embedding.weight

        with torch.enable_grad():
            for _ in range(self.config.local_updates):
                user_emb_exp = fast_user_emb.unsqueeze(1).expand(-1, support_items.size(1), -1)
                item_emb_supp = item_emb_weight[support_items]
                logits = self.functional_forward(user_emb_exp, item_emb_supp, fast_mem_weights, fast_fc_weights)
                loss = F.binary_cross_entropy_with_logits(logits, support_labels.float())

                grads = torch.autograd.grad(
                    loss,
                    [fast_user_emb, fast_mem_weights] + list(fast_fc_weights.values()),
                    allow_unused=True
                )

                fast_user_emb = fast_user_emb - self.config.local_lr * grads[0]
                fast_mem_weights = fast_mem_weights - self.config.local_lr * grads[1]

                new_fc_weights = OrderedDict()
                for i, (name, param) in enumerate(fast_fc_weights.items()):
                    if grads[i + 2] is not None:
                        new_fc_weights[name] = param - self.config.local_lr * grads[i + 2]
                    else:
                        new_fc_weights[name] = param
                fast_fc_weights = new_fc_weights

        eval_items = torch.cat([query_item, neg_items], dim=1)
        eval_embs = item_emb_weight[eval_items]
        user_emb_eval = fast_user_emb.unsqueeze(1).expand(-1, eval_embs.size(1), -1)

        scores = self.functional_forward(user_emb_eval, eval_embs, fast_mem_weights, fast_fc_weights)
        return scores


class TDAS(nn.Module):
    def __init__(self, config):
        super(TDAS, self).__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim

        self.user_embedding = nn.Embedding(config.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.predictor = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(self.embedding_dim * 2, self.embedding_dim * 2)),
            ('relu1', nn.ReLU()),
            ('layer2', nn.Linear(self.embedding_dim * 2, 1))
        ]))

        self.task_dim = 32
        self.feat_num = 2
        self.input_stat_dim = self.feat_num * self.embedding_dim * 2

        self.task_encoder = nn.Sequential(
            nn.Linear(self.input_stat_dim, self.task_dim),
            nn.ReLU(),
            nn.Linear(self.task_dim, self.task_dim),
            nn.ReLU()
        )

        self.layer_params = list(self.predictor.named_parameters())
        self.num_params = len(self.layer_params)
        self.hp_output_dim = 2 * self.num_params

        self.hp_generator = nn.Sequential(
            nn.Linear(self.task_dim, self.task_dim),
            nn.ReLU(),
            nn.Linear(self.task_dim, self.hp_output_dim),
            nn.Sigmoid()
        )

        self.init_lr = nn.Parameter(torch.tensor(config.local_lr))
        self.init_wd = nn.Parameter(torch.tensor(1e-4))

    def get_task_embedding(self, user_emb, support_item_embs):
        u_mean = user_emb
        u_var = torch.zeros_like(user_emb)

        i_mean = torch.mean(support_item_embs, dim=1)
        i_var = torch.var(support_item_embs, dim=1)
        if torch.isnan(i_var).any(): i_var = torch.zeros_like(i_mean)

        stats = torch.cat([u_mean, u_var, i_mean, i_var], dim=-1)

        task_emb = self.task_encoder(stats)
        return task_emb

    def gen_hyper_params(self, task_emb):
        raw_hps = self.hp_generator(task_emb)
        alphas_scale, betas_scale = torch.chunk(raw_hps, 2, dim=-1)
        alphas = alphas_scale * self.init_lr
        betas = betas_scale * self.init_wd
        return alphas, betas

    def functional_forward(self, user_emb, item_emb, weights):
        x = torch.cat([user_emb, item_emb], dim=-1)

        w1 = weights['layer1.weight']
        b1 = weights['layer1.bias']

        if w1.dim() == 3:
            x = torch.bmm(x, w1.transpose(1, 2)) + b1.unsqueeze(1)
        else:
            x = F.linear(x, w1, b1)

        x = F.relu(x)

        w2 = weights['layer2.weight']
        b2 = weights['layer2.bias']

        if w2.dim() == 3:
            x = torch.bmm(x, w2.transpose(1, 2)) + b2.unsqueeze(1)
        else:
            x = F.linear(x, w2, b2)

        return x.squeeze(-1)

    def forward(self, batch):
        user_ids = batch['user_id']
        support_seq = batch['support_seq']
        query_item = batch['query_item']
        batch_size = user_ids.size(0)

        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding.weight

        pos_items = support_seq
        neg_items = torch.randint(0, self.config.num_items, pos_items.shape, device=user_ids.device)

        support_items = torch.cat([pos_items, neg_items], dim=1)
        support_labels = torch.cat([torch.ones_like(pos_items), torch.zeros_like(neg_items)], dim=1)

        support_item_embs = item_emb[support_items]

        task_emb = self.get_task_embedding(user_emb, support_item_embs)
        alphas, betas = self.gen_hyper_params(task_emb)

        fast_weights = OrderedDict(self.predictor.named_parameters())

        user_emb_exp = user_emb.unsqueeze(1).expand(-1, support_items.size(1), -1)

        for _ in range(self.config.local_updates):
            logits = self.functional_forward(user_emb_exp, support_item_embs, fast_weights)
            loss = F.binary_cross_entropy_with_logits(logits, support_labels.float())

            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)

            updated_weights = OrderedDict()
            for idx, (name, param) in enumerate(fast_weights.items()):
                if 'weight' in name:
                    alpha = alphas[:, idx].view(-1, 1, 1)
                    beta = betas[:, idx].view(-1, 1, 1)
                else:
                    alpha = alphas[:, idx].view(-1, 1)
                    beta = betas[:, idx].view(-1, 1)

                if param.dim() == len(grads[idx].shape) - 1:
                    param_expanded = param.unsqueeze(0).expand(batch_size, *param.shape)
                else:
                    param_expanded = param

                updated_param = (1 - beta) * param_expanded - alpha * grads[idx]
                updated_weights[name] = updated_param

            fast_weights = updated_weights

        query_emb = item_emb[query_item].squeeze(1)
        neg_query_items = torch.randint(0, self.config.num_items, (batch_size, self.config.query_neg_sample_size),
                                        device=user_ids.device)
        neg_query_emb = item_emb[neg_query_items]
        candidate_embs = torch.cat([query_emb.unsqueeze(1), neg_query_emb], dim=1)

        user_emb_query = user_emb.unsqueeze(1).expand(-1, candidate_embs.size(1), -1)

        logits = self.functional_forward(user_emb_query, candidate_embs, fast_weights)

        labels = torch.zeros(batch_size, dtype=torch.long, device=user_ids.device)
        return F.cross_entropy(logits / self.config.temperature, labels)

    def evaluate_batch(self, batch):
        user_ids = batch['user_id']
        support_seq = batch['support_seq']
        query_item = batch['query_item']
        neg_items = batch['neg_items']
        batch_size = user_ids.size(0)

        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding.weight

        pos_items = support_seq
        neg_support = torch.randint(0, self.config.num_items, pos_items.shape, device=user_ids.device)
        support_items = torch.cat([pos_items, neg_support], dim=1)
        support_labels = torch.cat([torch.ones_like(pos_items), torch.zeros_like(neg_support)], dim=1)
        support_item_embs = item_emb[support_items]

        task_emb = self.get_task_embedding(user_emb, support_item_embs)
        alphas, betas = self.gen_hyper_params(task_emb)

        fast_weights = OrderedDict(self.predictor.named_parameters())
        user_emb_exp = user_emb.unsqueeze(1).expand(-1, support_items.size(1), -1)

        with torch.enable_grad():
            for _ in range(self.config.local_updates):
                logits = self.functional_forward(user_emb_exp, support_item_embs, fast_weights)
                loss = F.binary_cross_entropy_with_logits(logits, support_labels.float())
                grads = torch.autograd.grad(loss, fast_weights.values(), allow_unused=True)

                updated_weights = OrderedDict()
                for idx, (name, param) in enumerate(fast_weights.items()):
                    if 'weight' in name:
                        alpha = alphas[:, idx].view(-1, 1, 1)
                        beta = betas[:, idx].view(-1, 1, 1)
                    else:
                        alpha = alphas[:, idx].view(-1, 1)
                        beta = betas[:, idx].view(-1, 1)

                    if param.dim() == len(grads[idx].shape) - 1:
                        param_expanded = param.unsqueeze(0).expand(batch_size, *param.shape)
                    else:
                        param_expanded = param

                    updated_param = (1 - beta) * param_expanded - alpha * grads[idx]
                    updated_weights[name] = updated_param
                fast_weights = updated_weights

        eval_items = torch.cat([query_item, neg_items], dim=1)
        eval_embs = item_emb[eval_items]
        user_emb_eval = user_emb.unsqueeze(1).expand(-1, eval_embs.size(1), -1)

        scores = self.functional_forward(user_emb_eval, eval_embs, fast_weights)
        return scores


class TaNP(nn.Module):
    def __init__(self, config):
        super(TaNP, self).__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim

        self.user_embedding = nn.Embedding(config.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.xy_dim = self.embedding_dim * 2 + 1
        self.latent_dim = self.embedding_dim
        self.task_dim = self.embedding_dim
        self.hidden_dim = self.embedding_dim * 2

        self.xy_to_z = nn.Sequential(
            nn.Linear(self.xy_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.z_to_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.z_to_logsigma = nn.Linear(self.hidden_dim, self.latent_dim)

        self.xy_to_r = nn.Sequential(
            nn.Linear(self.xy_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.task_dim)
        )

        self.num_clusters = 10
        self.temperature = 1.0
        self.memory_keys = nn.Parameter(torch.randn(self.num_clusters, self.task_dim))
        nn.init.xavier_uniform_(self.memory_keys)

        self.decoder_h1 = nn.Linear(self.embedding_dim * 2 + self.latent_dim, self.hidden_dim)
        self.decoder_h2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decoder_out = nn.Linear(self.hidden_dim, 1)

        self.film_layer1_gamma = nn.Linear(self.task_dim, self.hidden_dim, bias=False)
        self.film_layer1_beta = nn.Linear(self.task_dim, self.hidden_dim, bias=False)
        self.film_layer2_gamma = nn.Linear(self.task_dim, self.hidden_dim, bias=False)
        self.film_layer2_beta = nn.Linear(self.task_dim, self.hidden_dim, bias=False)

    def aggregate(self, z_i):
        return torch.mean(z_i, dim=1)

    def encode_latent(self, user_emb, item_emb, r):
        batch_size, seq_len, _ = user_emb.size()
        xy = torch.cat([user_emb, item_emb, r], dim=-1)
        h = self.xy_to_z(xy)
        h_agg = self.aggregate(h)
        mu = self.z_to_mu(h_agg)
        logsigma = self.z_to_logsigma(h_agg)
        return mu, logsigma

    def encode_task(self, user_emb, item_emb, r):
        xy = torch.cat([user_emb, item_emb, r], dim=-1)
        r_i = self.xy_to_r(xy)
        r_agg = self.aggregate(r_i)

        diff = r_agg.unsqueeze(1) - self.memory_keys.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        scores = torch.pow((dist_sq / self.temperature) + 1, -(self.temperature + 1) / 2)
        probs = scores / scores.sum(dim=1, keepdim=True)

        mem_val = torch.mm(probs, self.memory_keys)
        new_task_emb = r_agg + mem_val
        return new_task_emb, probs

    def reparameterize(self, mu, logsigma):
        std = torch.exp(0.5 * logsigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, user_emb, item_emb, z, task_emb):
        batch_size, seq_len, _ = user_emb.size()
        z_exp = z.unsqueeze(1).expand(-1, seq_len, -1)
        inp = torch.cat([user_emb, item_emb, z_exp], dim=-1)

        h1 = self.decoder_h1(inp)
        gamma1 = self.film_layer1_gamma(task_emb).unsqueeze(1)
        beta1 = self.film_layer1_beta(task_emb).unsqueeze(1)
        h1 = h1 * gamma1 + beta1
        h1 = F.relu(h1)

        h2 = self.decoder_h2(h1)
        gamma2 = self.film_layer2_gamma(task_emb).unsqueeze(1)
        beta2 = self.film_layer2_beta(task_emb).unsqueeze(1)
        h2 = h2 * gamma2 + beta2
        h2 = F.relu(h2)

        out = self.decoder_out(h2)
        return out.squeeze(-1)

    def forward(self, batch):
        user_ids = batch['user_id']
        support_seq = batch['support_seq']
        query_item = batch['query_item']
        batch_size = user_ids.size(0)

        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding.weight

        pos_items = support_seq
        neg_items = torch.randint(0, self.config.num_items, pos_items.shape, device=user_ids.device)
        support_items = torch.cat([pos_items, neg_items], dim=1)
        support_labels = torch.cat([torch.ones_like(pos_items), torch.zeros_like(neg_items)], dim=1).unsqueeze(-1)

        support_user_emb = user_emb.unsqueeze(1).expand(-1, support_items.size(1), -1)
        support_item_emb = item_emb[support_items]

        mu, logsigma = self.encode_latent(support_user_emb, support_item_emb, support_labels)
        z = self.reparameterize(mu, logsigma)
        task_emb, cluster_probs = self.encode_task(support_user_emb, support_item_emb, support_labels)

        query_item_emb = item_emb[query_item].squeeze(1)
        neg_query_items = torch.randint(0, self.config.num_items, (batch_size, self.config.query_neg_sample_size), device=user_ids.device)
        neg_query_emb = item_emb[neg_query_items]

        target_items_emb = torch.cat([query_item_emb.unsqueeze(1), neg_query_emb], dim=1)
        target_user_emb = user_emb.unsqueeze(1).expand(-1, target_items_emb.size(1), -1)

        logits = self.decode(target_user_emb, target_items_emb, z, task_emb)

        labels = torch.zeros(batch_size, dtype=torch.long, device=user_ids.device)
        rec_loss = F.cross_entropy(logits / self.config.temperature, labels)
        kl_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp(), dim=1).mean()

        q = cluster_probs
        p = q ** 2 / q.sum(dim=0)
        p = p / p.sum(dim=1, keepdim=True)
        cluster_loss = F.kl_div(q.log(), p, reduction='batchmean')

        total_loss = rec_loss + 0.1 * kl_loss + 0.1 * cluster_loss
        return total_loss

    def evaluate_batch(self, batch):
        user_ids = batch['user_id']
        support_seq = batch['support_seq']
        query_item = batch['query_item']
        neg_items = batch['neg_items']
        batch_size = user_ids.size(0)

        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding.weight

        pos_items = support_seq
        neg_support = torch.randint(0, self.config.num_items, pos_items.shape, device=user_ids.device)
        support_items = torch.cat([pos_items, neg_support], dim=1)
        support_labels = torch.cat([torch.ones_like(pos_items), torch.zeros_like(neg_support)], dim=1).unsqueeze(-1)

        support_user_emb = user_emb.unsqueeze(1).expand(-1, support_items.size(1), -1)
        support_item_emb = item_emb[support_items]

        mu, logsigma = self.encode_latent(support_user_emb, support_item_emb, support_labels)
        z = mu
        task_emb, _ = self.encode_task(support_user_emb, support_item_emb, support_labels)

        eval_items = torch.cat([query_item, neg_items], dim=1)
        eval_embs = item_emb[eval_items]
        eval_user_emb = user_emb.unsqueeze(1).expand(-1, eval_embs.size(1), -1)

        scores = self.decode(eval_user_emb, eval_embs, z, task_emb)
        return scores
