import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
from tqdm import tqdm
import pickle
from sklearn.preprocessing import normalize

def preprocess_data(config):
    item_genre_features = None
    if config.dataset_name == 'ml-100k':
        df = pd.read_csv(os.path.join(config.raw_data_path, 'u.data'), sep='\t', header=None,
                         names=['user_id', 'item_id', 'rating', 'timestamp'])
        df['user_id'] -= 1
        df['item_id'] -= 1
        num_items = df['item_id'].max() + 1
        item_genre_features = np.zeros((num_items, 19), dtype=np.float32)
        item_meta_path = os.path.join(config.raw_data_path, 'u.item')
        if os.path.exists(item_meta_path):
            meta_df = pd.read_csv(item_meta_path, sep='|', header=None, index_col=0, encoding='latin-1')
            genre_df = meta_df.iloc[:, 4:23]
            for item_id, row in genre_df.iterrows():
                if (item_id - 1) < num_items:
                    item_genre_features[item_id - 1] = row.values

    elif config.dataset_name == 'last.fm':
        main_df = pd.read_csv(os.path.join(config.raw_data_path, 'user_artists.dat'), sep='\t', header=0,
                              names=['user_id', 'item_id', 'weight'])
        user_mapping = {old_id: new_id for new_id, old_id in enumerate(main_df['user_id'].unique())}
        item_mapping = {old_id: new_id for new_id, old_id in enumerate(main_df['item_id'].unique())}

        timestamps_path = os.path.join(config.raw_data_path, 'user_taggedartists-timestamps.dat')
        df = pd.read_csv(timestamps_path, sep='\t', header=0, names=['user_id', 'item_id', 'tag_id', 'timestamp'])

        df = df[df['user_id'].isin(user_mapping.keys()) & df['item_id'].isin(item_mapping.keys())]
        df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)

        df.drop_duplicates(subset=['user_id', 'item_id', 'timestamp'], keep='first', inplace=True)

        df['prev_item_id'] = df.groupby('user_id')['item_id'].shift(1)
        df = df[df['item_id'] != df['prev_item_id']]
        df = df[['user_id', 'item_id', 'timestamp']]

        df['user_id'] = df['user_id'].map(user_mapping)
        df['item_id'] = df['item_id'].map(item_mapping)

        num_items = len(item_mapping)
        tags_path = os.path.join(config.raw_data_path, 'tags.dat')
        user_tagged_path = os.path.join(config.raw_data_path, 'user_taggedartists.dat')

        if os.path.exists(tags_path) and os.path.exists(user_tagged_path):
            tags_df = pd.read_csv(tags_path, sep='\t', header=0, names=['tagID', 'tagValue'], encoding='latin-1')
            tag_mapping = {tag_id: i for i, tag_id in enumerate(tags_df['tagID'].unique())}
            num_tags = len(tag_mapping)
            tagged_artists_df = pd.read_csv(user_tagged_path, sep='\t', header=0,
                                            names=['userID', 'artistID', 'tagID'], encoding='latin-1')
            tagged_artists_df = tagged_artists_df[
                tagged_artists_df['artistID'].isin(item_mapping) & tagged_artists_df['tagID'].isin(tag_mapping)]
            tagged_artists_df['mapped_item_id'] = tagged_artists_df['artistID'].map(item_mapping)
            tagged_artists_df['mapped_tag_idx'] = tagged_artists_df['tagID'].map(tag_mapping)

            tag_counts = tagged_artists_df.groupby(['mapped_item_id', 'mapped_tag_idx']).size().reset_index(
                name='counts')

            item_indices = tag_counts['mapped_item_id'].values
            tag_indices = tag_counts['mapped_tag_idx'].values
            counts = tag_counts['counts'].values

            item_genre_features = sp.csr_matrix((counts, (item_indices, tag_indices)),
                                                shape=(num_items, num_tags),
                                                dtype=np.float32)
            item_genre_features = normalize(item_genre_features, norm='l2', axis=1)

    else:
        raise ValueError(f"Unsupported dataset: {config.dataset_name}")

    df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    num_users = df['user_id'].max() + 1

    all_items_arr = np.arange(num_items)

    output_dir = os.path.join(config.processed_data_path, config.dataset_name, config.eval_scenario)
    os.makedirs(output_dir, exist_ok=True)
    min_interactions = config.support_size + 1

    task_keys_train = ['user_id', 'support_seq', 'query_item', 'history_items']
    task_keys_test = ['user_id', 'support_seq', 'query_item', 'history_items', 'neg_items']
    train_tasks = {key: [] for key in task_keys_train}
    test_tasks = {key: [] for key in task_keys_test}

    grouped_by_user = df.groupby('user_id')
    interaction_matrix = None

    if config.eval_scenario == 'warm_start':
        train_interactions_df_list = []
        for user_id, group in tqdm(grouped_by_user, desc="Creating warm-start tasks"):
            interactions = group['item_id'].values
            if len(interactions) < min_interactions + 1:
                continue
            split_point = int(len(interactions) * 0.8)
            train_interactions = interactions[:split_point]
            test_interactions = interactions[split_point:]

            train_interactions_df_list.append(group.iloc[:split_point])

            for i in range(config.support_size, len(train_interactions)):
                task = {
                    'user_id': user_id,
                    'support_seq': train_interactions[i - config.support_size: i],
                    'query_item': [train_interactions[i]],
                    'history_items': train_interactions[:i + 1]
                }
                for key in task_keys_train:
                    train_tasks[key].append(task[key])

            for i in range(len(test_interactions)):
                history_up_to_query = interactions[:split_point + i]
                if len(history_up_to_query) < config.support_size:
                    continue

                seen_items = history_up_to_query
                candidate_neg_pool = np.setdiff1d(all_items_arr, seen_items, assume_unique=True)
                num_neg_to_sample = config.eval_neg_sample_size
                if len(candidate_neg_pool) < num_neg_to_sample:
                    neg_items_list = np.random.choice(candidate_neg_pool, size=num_neg_to_sample, replace=True)
                else:
                    neg_items_list = np.random.choice(candidate_neg_pool, size=num_neg_to_sample, replace=False)

                task = {
                    'user_id': user_id,
                    'support_seq': history_up_to_query[-config.support_size:],
                    'query_item': [test_interactions[i]],
                    'history_items': interactions[:split_point + i + 1],
                    'neg_items': neg_items_list
                }
                for key in task_keys_test:
                    test_tasks[key].append(task[key])
        train_df = pd.concat(train_interactions_df_list)
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        data = np.ones(len(rows))
        interaction_matrix = sp.csr_matrix((data, (rows, cols)), shape=(num_users, num_items))

    elif config.eval_scenario == 'cold_start':
        unique_users = df['user_id'].unique()
        np.random.shuffle(unique_users)
        train_user_ratio = 0.8
        num_train_users = int(len(unique_users) * train_user_ratio)
        train_user_ids = set(unique_users[:num_train_users])
        test_user_ids = set(unique_users[num_train_users:])

        for user_id, group in tqdm(grouped_by_user, desc="Creating cold-start tasks"):
            interactions = group['item_id'].values
            if len(interactions) < min_interactions:
                continue

            target_dict = None
            is_test_user = False
            if user_id in train_user_ids:
                target_dict = train_tasks
            elif user_id in test_user_ids:
                target_dict = test_tasks
                is_test_user = True
            else:
                continue

            num_tasks_per_user = min(5, len(interactions) - min_interactions + 1)
            start_index = len(interactions) - num_tasks_per_user - config.support_size
            start_index = max(config.support_size, start_index)

            for i in range(start_index, len(interactions) - 1):
                task = {
                    'user_id': user_id,
                    'support_seq': interactions[i - config.support_size: i],
                    'query_item': [interactions[i]],
                    'history_items': interactions[:i + 1]
                }

                if is_test_user:
                    seen_items = interactions[:i + 1]
                    candidate_neg_pool = np.setdiff1d(all_items_arr, seen_items, assume_unique=True)
                    num_neg_to_sample = config.eval_neg_sample_size
                    if len(candidate_neg_pool) < num_neg_to_sample:
                        neg_items_list = np.random.choice(candidate_neg_pool, size=num_neg_to_sample, replace=True)
                    else:
                        neg_items_list = np.random.choice(candidate_neg_pool, size=num_neg_to_sample, replace=False)
                    task['neg_items'] = neg_items_list
                    for key in task_keys_test:
                        target_dict[key].append(task[key])
                else:
                    for key in task_keys_train:
                        target_dict[key].append(task[key])

        train_df = df[df['user_id'].isin(train_user_ids)]
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        data = np.ones(len(rows))
        interaction_matrix = sp.csr_matrix((data, (rows, cols)), shape=(num_users, num_items))

    torch.save(train_tasks, os.path.join(output_dir, "train_tasks.pt"))
    torch.save(test_tasks, os.path.join(output_dir, "test_tasks.pt"))
    assert interaction_matrix is not None, "Interaction matrix was not created!"
    matrix_path = os.path.join(output_dir, 'interaction_matrix.npz')
    sp.save_npz(matrix_path, interaction_matrix)
    meta_path = os.path.join(output_dir, 'meta.pkl')
    meta_info = {'num_users': num_users, 'num_items': num_items}
    if item_genre_features is not None:
        meta_info['item_genre_features'] = item_genre_features
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_info, f)

class MetaTaskDataset(Dataset):
    def __init__(self, tasks_data, config):
        self.config = config
        self.tasks = tasks_data
        self.task_keys = list(tasks_data.keys())
        self.num_tasks = len(tasks_data['user_id'])
        self.is_test = 'neg_items' in self.task_keys

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, idx):
        task_data = {key: self.tasks[key][idx] for key in self.task_keys}

        item = {
            'user_id': torch.tensor(task_data['user_id'], dtype=torch.long),
            'support_seq': torch.LongTensor(task_data['support_seq']),
            'query_item': torch.LongTensor(task_data['query_item']),
            'history_items': torch.LongTensor(task_data['history_items'])
        }
        if self.is_test:
            item['neg_items'] = torch.LongTensor(task_data['neg_items'])
        return item

class PretrainDataset(Dataset):
    def __init__(self, interaction_matrix):
        self.users = interaction_matrix.row
        self.pos_items = interaction_matrix.col

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.pos_items[idx]

def meta_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    collated_batch = {}
    keys = batch[0].keys()
    for key in keys:
        if key == 'history_items':
            collated_batch[key] = [d[key] for d in batch]
        else:
            tensors = [d[key] for d in batch]
            collated_batch[key] = torch.stack(tensors, 0)
    return collated_batch

def load_data(config):
    processed_base_path = os.path.join(config.processed_data_path, config.dataset_name, config.eval_scenario)
    processed_meta_path = os.path.join(processed_base_path, 'meta.pkl')
    interaction_matrix_path = os.path.join(processed_base_path, 'interaction_matrix.npz')
    if not os.path.exists(processed_meta_path) or not os.path.exists(interaction_matrix_path):
        raise FileNotFoundError(
            f"Processed data not found at {processed_base_path}. Please run with --preprocess flag first.")

    with open(processed_meta_path, 'rb') as f:
        meta_info = pickle.load(f)
    config.num_users = meta_info['num_users']
    config.num_items = meta_info['num_items']
    item_genre_features = meta_info.get('item_genre_features', None)

    train_task_path = os.path.join(processed_base_path, 'train_tasks.pt')
    test_task_path = os.path.join(processed_base_path, 'test_tasks.pt')
    train_tasks_data = torch.load(train_task_path, weights_only=False)
    test_tasks_data = torch.load(test_task_path, weights_only=False)
    interaction_matrix = sp.load_npz(interaction_matrix_path)
    
    train_dataset = MetaTaskDataset(train_tasks_data, config)
    test_dataset = MetaTaskDataset(test_tasks_data, config)
    pretrain_dataset = PretrainDataset(interaction_matrix.tocoo())

    loader_common_kwargs = {}
    if config.num_workers > 0:
        loader_common_kwargs['persistent_workers'] = True
        loader_common_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=config.meta_batch_size,
        shuffle=True, num_workers=config.num_workers,
        collate_fn=meta_collate_fn, pin_memory=True,
        drop_last=True, **loader_common_kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.meta_batch_size,
        shuffle=False, num_workers=config.num_workers,
        collate_fn=meta_collate_fn, pin_memory=True,
        **loader_common_kwargs
    )
    pretrain_loader = DataLoader(
        pretrain_dataset, batch_size=config.pretrain_batch_size,
        shuffle=True, num_workers=config.num_workers,
        **loader_common_kwargs
    )

    print(f"Data loading complete for '{config.dataset_name}'")
    print(f"Users: {config.num_users}, Items: {config.num_items}")
    print(f"Train tasks: {len(train_dataset)}, Test tasks: {len(test_dataset)}")
    print("-" * 50)
    return train_loader, test_loader, pretrain_loader, interaction_matrix, item_genre_features