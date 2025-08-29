from parse import parse_args
from utils import seed_everything
from dataset import load_data, preprocess_data
from model import MetaDSM
from trainer import Trainer
import os

def run():
    config = parse_args()
    seed_everything(config.seed)
    processed_dir = os.path.join(config.processed_data_path, config.dataset_name, config.eval_scenario)
    if config.preprocess or not os.path.exists(processed_dir):
        if not os.path.exists(config.raw_data_path):
            raise FileNotFoundError(f"Raw data not found at {config.raw_data_path}.")
        preprocess_data(config)

    train_loader, test_loader, pretrain_loader, interaction_matrix, item_genre_features = load_data(config)
    model = MetaDSM(config, interaction_matrix, item_genre_features)
    trainer = Trainer(config, model, train_loader, test_loader, pretrain_loader)
    trainer.train()

if __name__ == '__main__':
    run()