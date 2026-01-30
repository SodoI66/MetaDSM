# MetaDSM Recommendation Model

This is the source code for the paper "Integrating Static and Dynamic Preferences via Meta-Learning for User Cold Start Recommendation".

## Key Features

- **Meta-Learning Framework**: Utilizes MAML to quickly adapt to new users using limited interaction support sets.
- **Hybrid Preference Modeling**:
  - **Static**: Uses **LightGCN** to capture long-term user/item representations from the global interaction graph.
  - **Dynamic**: Uses **GRU** to capture short-term sequential patterns from recent interactions.
- **Adaptive Fusion**: A gating mechanism to dynamically weigh static and dynamic signals based on the user's context.
- **Comprehensive Baselines**: Includes implementations of various state-of-the-art recommendation models for comparison.

## Dependencies

All required packages and their specific versions are listed in the `requirements.txt` file. You can install them all with a single command:

```
pip install -r requirements.txt
```

The contents of `requirements.txt` are as follows:

```text
torch==2.7.1+cu126
pandas==2.3.0
numpy==2.1.2
scipy==1.16.0
tqdm==4.67.1
```

## Supported Datasets

The code currently supports the following datasets:

1.  [ML-1M](https://grouplens.org/datasets/movielens/)
2.  [BX](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

## Quick Start

### 1. Prepare Data

1. Create a directory structure: `../data/[dataset_name]`.

2. Download the raw data files:

   *   **ML-1M**: Place `ratings.dat` and `movies.dat` in `../data/ML-1M/`.
   *   **BX**: Place `Ratings.csv` in `../data/BX/`.

3. Run the preprocessing script:

   ```bash
   python main.py --preprocess --dataset_name [dataset_name] --eval_scenario [scenario]
   ```

   *   This will generate processed files in `../data/processed/`.
   *   `[scenario]` can be `warm_start` or `cold_start`.

### 2. Train MetaDSM

To train the proposed MetaDSM model:

```bash
python main.py --dataset_name [dataset_name] --eval_scenario [scenario] --ablation_mode full
```

### 3. Train Baselines

The framework includes several baselines. You can run them by changing the `--ablation_mode` argument:

```bash
python main.py --dataset_name [dataset_name] --ablation_mode [baseline_name]
```

- `[baseline_name]` can be `baseline_LightGCN`, `baseline_GRU4Rec`, `baseline_SASRec`, `baseline_FMLPRec`, `baseline_MeLU`, `baseline_MAMO`, `baseline_TDAS`, `baseline_TaNP`.

### 4. Ablation Studies

You can also run ablation studies on the MetaDSM components to verify the contribution of each module. The available modes allow you to disable specific components (`no_*`) or run isolated components (`*_only`):

*   `no_gcn`: Remove the Graph component (uses standard embeddings instead of LightGCN).
*   `no_gru`: Remove the Sequential component (uses static context only).
*   `no_meta`: Remove the Meta-learning adaptation (standard training without inner loop).
*   `gcn_only`: Use only the Graph component (LightGCN) without GRU or Meta-learning.
*   `gru_only`: Use only the Sequential component (GRU) without LightGCN or Meta-learning.
*   `meta_only`: Use only Meta-learning with standard embeddings (no LightGCN or GRU).

## Key Command-Line Arguments

Configuration is handled in `parse.py`. Common arguments include:

| Argument          | Description                                               | Default      |
| :---------------- | :-------------------------------------------------------- | :----------- |
| `--dataset_name`  | `ML-1M` or `BX`                                           | Required     |
| `--eval_scenario` | `cold_start` (new users) or `warm_start` (existing users) | `cold_start` |
| `--support_size`  | Number of interactions used for adaptation (K-shot)       | `10`         |
| `--ablation_mode` | Selects the model variant or baseline to run              | `full`       |
| `--meta_lr`       | Learning rate for meta-parameters                         | `1e-5`       |
| `--local_lr`      | Learning rate for inner-loop adaptation                   | `5e-3`       |
| `--local_updates` | Number of gradient steps in the inner loop                | `3`          |
| `--gnn_layers`    | Number of LightGCN layers                                 | `2`          |

## File Descriptions

| File           | Description                                                  |
| :------------- | :----------------------------------------------------------- |
| `main.py`      | Entry point for training and evaluation.                     |
| `model.py`     | Implementation of the core **MetaDSM** model.                |
| `baselines.py` | Implementations of baseline models (SASRec, MeLU, TaNP, etc.). |
| `dataset.py`   | Data loading, preprocessing, and task generation (Support/Query sets). |
| `trainer.py`   | Manages the training loops (Pre-training, Meta-training) and evaluation. |
| `modules.py`   | Contains the `LightGCN` implementation.                      |
| `parse.py`     | Argument parsing and configuration.                          |
| `utils.py`     | Utility functions (metrics, seeding, EMA).                   |

## Results

Training logs, model checkpoints, and evaluation metrics (Recall@K, NDCG@K) are saved in the `results/` directory, organized by dataset and experiment configuration.
