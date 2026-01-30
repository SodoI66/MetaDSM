import argparse
import torch
import os

def parse_args():
    parser = argparse.ArgumentParser(description="MetaDSM Experiment Configuration")

    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=default_device)
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--raw_data_path', type=str, default='../data')
    parser.add_argument('--processed_data_path', type=str, default='../data/processed')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--dataset_name', type=str, required=True, choices=['ML-1M', 'BX'])
    parser.add_argument('--support_size', type=int, default=10, choices=[5, 10, 20])
    parser.add_argument('--eval_k_list', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--eval_neg_sample_size', type=int, default=99)
    parser.add_argument('--eval_scenario', type=str, default='cold_start', choices=['warm_start', 'cold_start'])

    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--gnn_layers', type=int, default=2)
    parser.add_argument('--gru_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--predictor_dropout', type=float, default=0.5)

    parser.add_argument('--fusion_ablation', type=str, default='full', choices=['full', 'no_residual', 'no_gate'])
    parser.add_argument('--inner_loop_mode', type=str, default='step_by_step', choices=['step_by_step', 'last_step_only'])

    parser.add_argument('--meta_lr', type=float, default=1e-5)
    parser.add_argument('--meta_gcn_lr', type=float, default=1e-5)
    parser.add_argument('--local_lr', type=float, default=5e-3)
    parser.add_argument('--local_updates', type=int, default=3)
    parser.add_argument('--local_neg_sample_size', type=int, default=64)
    parser.add_argument('--query_neg_sample_size', type=int, default=128)
    parser.add_argument('--inner_grad_clip', type=float, default=0.25)
    parser.add_argument('--temperature', type=float, default=1.0)

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--meta_batch_size', type=int, default=64)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--outer_grad_clip', type=float, default=1.0)
    parser.add_argument('--ema_decay', type=float, default=0.9995)

    parser.add_argument('--pretrain_epochs', type=int, default=150)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)
    parser.add_argument('--pretrain_batch_size', type=int, default=1024)
    parser.add_argument('--gcn_reg_weight', type=float, default=1e-4)

    parser.add_argument('--fmlprec_num_blocks', type=int, default=2)
    parser.add_argument('--sasrec_num_blocks', type=int, default=2)
    parser.add_argument('--sasrec_num_heads', type=int, default=4)
    parser.add_argument('--initializer_range', type=float, default=0.02)
    parser.add_argument('--gru4rec_n_sample', type=int, default=2048)
    parser.add_argument('--gru4rec_sample_alpha', type=float, default=0.75)

    parser.add_argument('--ablation_mode', type=str, default='full',
                        choices=['full', 'no_gcn', 'no_gru', 'no_meta',
                                 'gcn_only', 'gru_only', 'meta_only',
                                 'baseline_MeLU', 'baseline_LightGCN',
                                 'baseline_GRU4Rec', 'baseline_FMLPRec', 'baseline_SASRec',
                                 'baseline_MAMO', 'baseline_TDAS', 'baseline_TaNP'])

    args = parser.parse_args()
    args.raw_data_path = os.path.join(args.raw_data_path, str(args.dataset_name))
    return args
