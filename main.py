from parse import parse_args
from utils import seed_everything
from dataset import load_data, preprocess_data
from model import MetaDSM
from baselines import LightGCN_Baseline, GRU4Rec, MeLU, FMLPRec, SASRec, MAMO, TDAS, TaNP
from trainer import Trainer
import os

def get_model(config, interaction_matrix, item_genre_features):
    mode = config.ablation_mode
    if mode == 'baseline_LightGCN':
        return LightGCN_Baseline(config, interaction_matrix)
    elif mode == 'baseline_GRU4Rec':
        return GRU4Rec(config)
    elif mode == 'baseline_MeLU':
        return MeLU(config)
    elif mode == 'baseline_FMLPRec':
        return FMLPRec(config)
    elif mode == 'baseline_SASRec':
        return SASRec(config)
    elif mode == 'baseline_MAMO':
        return MAMO(config)
    elif mode == 'baseline_TDAS':
        return TDAS(config)
    elif mode == 'baseline_TaNP':
        return TaNP(config)
    else:
        return MetaDSM(config, interaction_matrix, item_genre_features)

def run():
    config = parse_args()
    config.processed_data_path = f"{config.processed_data_path}-{config.support_size}"
    seed_everything(config.seed)
    processed_dir = os.path.join(config.processed_data_path, config.dataset_name, config.eval_scenario)
    if config.preprocess or not os.path.exists(processed_dir):
        if not os.path.exists(config.raw_data_path):
            raise FileNotFoundError(f"Raw data not found at {config.raw_data_path}.")
        preprocess_data(config)

    train_loader, test_loader, pretrain_loader, interaction_matrix, pop_sampler, item_genre_features = load_data(config)
    model = get_model(config, interaction_matrix, item_genre_features)
    trainer = Trainer(config, model, train_loader, test_loader, pretrain_loader, pop_sampler=pop_sampler)
    trainer.train()

if __name__ == '__main__':
    run()
