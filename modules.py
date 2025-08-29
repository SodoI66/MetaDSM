import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, interaction_matrix, num_layers=2):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        adj_matrix = self._create_normalized_adj_matrix(interaction_matrix)
        self.register_buffer('adj_matrix', adj_matrix)

    def _create_normalized_adj_matrix(self, interaction_matrix):
        R = interaction_matrix.tocoo()
        num_nodes = self.num_users + self.num_items
        adj_shape = (num_nodes, num_nodes)
        A_data = np.ones(R.nnz)
        A_rows = R.row
        A_cols = R.col + self.num_users
        rows = np.concatenate((A_rows, A_cols))
        cols = np.concatenate((A_cols, A_rows))
        data = np.concatenate((A_data, A_data))
        A = sp.csr_matrix((data, (rows, cols)), shape=adj_shape)
        row_sum = np.array(A.sum(axis=1)).flatten()
        row_sum = np.where(row_sum == 0, 1e-12, row_sum)
        d_inv_sqrt = np.power(row_sum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = D_inv_sqrt.dot(A).dot(D_inv_sqrt).tocoo()
        indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col))).long()
        values = torch.from_numpy(norm_adj.data).float()
        sparse_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj.shape))
        return sparse_tensor

    def forward(self):
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            embeddings_list = [all_embeddings]
            for _ in range(self.num_layers):
                all_embeddings = torch.sparse.mm(self.adj_matrix, all_embeddings)
                embeddings_list.append(all_embeddings)
            final_embeddings = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)

        final_user_emb, final_item_emb = torch.split(final_embeddings, [self.num_users, self.num_items])
        return final_user_emb, final_item_emb