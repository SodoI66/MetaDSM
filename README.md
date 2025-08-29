# MetaDSM Recommendation Model

This is the source code for the paper "Integrating Static and Dynamic Preferences via Meta-Learning for User Cold Start Recommendation".

## Key Features

- **Meta-Learning Framework**: Quickly adapts to new users with limited interaction data (cold-start scenario).
- **Hybrid Preference Modeling**: Combines users' long-term (static) and short-term (dynamic) interests.
- **Core Components**:
  - **LightGCN**: Learns static user and item embeddings from the interaction graph.
  - **GRU**: Captures dynamic patterns from user interaction sequences.

## Dependencies

All required packages and their specific versions are listed in the `requirements.txt` file. You can install them all with a single command:

```
pip install -r requirements.txt
```

The contents of `requirements.txt` are as follows:

```
torch==2.7.1+cu126
pandas==2.3.0
numpy==2.1.2
scipy==1.16.0
matplotlib==3.10.3
tqdm==4.67.1
scikit-learn==1.7.1
seaborn==0.13.2
```

## Quick Start

1. **Prepare Data**

   - Download `ml-100k` or `last.fm` datasets into a `../data/` directory.

   - Preprocess the data:

     ```
     python main.py --preprocess --dataset_name [dataset_name]
     ```

     Replace `[dataset_name]` with `ml-100k` or `last.fm`.

2. **Train the Model**

   - Run the main script to start training:

     ```
     python main.py --dataset_name [dataset_name] --eval_scenario [scenario]
     ```

     - `[scenario]` can be `warm_start` or `cold_start`.

3. **View Results**

   - Training logs, performance charts, and model weights will be saved in the `results/` directory.

## Command-Line Arguments

Key arguments are shown in the examples above. For a full list of configurable options and their descriptions, please see the `parse.py` file.

## File Descriptions

| File         | Description                                       |
| ------------ | ------------------------------------------------- |
| `main.py`    | Main script to run experiments.                   |
| `parse.py`   | Defines all command-line arguments.               |
| `dataset.py` | Handles data preprocessing and loading.           |
| `model.py`   | Defines the `MetaDSM` model architecture.         |
| `modules.py` | Implements the `LightGCN` component.              |
| `trainer.py` | Manages the training and evaluation loop.         |
| `utils.py`   | Contains helper functions and evaluation metrics. |
