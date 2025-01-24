# Project: SEO Ranker Optimization

## Overview
This repository contains scripts and tools for training, evaluating, and selecting ranking models for SEO-related tasks. It uses Java-based RankLib along with deep learning models for ranking optimization.

## File Breakdown
### Core Scripts
#### 1. `baseline_model_choice.py`
- **Purpose**: Selects the best ranking model based on evaluation metrics.
- **Inputs**: Model directory, embeddings, summary files.
- **Outputs**: CSV files with ranking results.

#### 2. `choose_seo_texts_new_java.py`
- **Purpose**: Processes ranking models for SEO text selection.
- **Inputs**: Feature files, trained models, and working set data.
- **Outputs**: Ranked documents in CSV format.

#### 3. `create_baseline_training_set.py`
- **Purpose**: Generates a baseline dataset for model training.
- **Inputs**: Data from previous ranking rounds.
- **Outputs**: Baseline dataset for training and validation.

#### 4. `create_student_file.py`
- **Purpose**: Creates test files for ranking evaluations.
- **Inputs**: Previous round data and query mappings.
- **Outputs**: Student ranking evaluation CSV files.

#### 5. `run_ranking_E5.py`
- **Purpose**: Uses E5 embeddings for ranking.
- **Inputs**: Queries and document embeddings.
- **Outputs**: Ranked documents.

### Utility Scripts
#### 6. `gen_utils.py`
- **Purpose**: Provides general utility functions.
- **Includes**: Parallel execution, command execution.

#### 7. `utils.py`
- **Purpose**: Contains helper functions for file handling, indexing, and evaluation.

### Training & Validation
#### 8. `model_train_test.py`
- **Purpose**: Trains and evaluates ranking models.
- **Inputs**: Training and test data files.
- **Outputs**: Trained models and evaluation metrics.

#### 9. `model_train_val_test.py`
- **Purpose**: Trains, validates, and tests models.
- **Inputs**: Training data, validation sets.
- **Outputs**: Trained models and validation results.

#### 10. `train_val_split.py`
- **Purpose**: Splits datasets into training and validation sets.
- **Inputs**: Ranking dataset.
- **Outputs**: Training and validation files.

## Installation & Setup
### Prerequisites
Ensure you have the following installed:
- **Python 3.8+**
- **Java (JDK 21.0.1)**
- **RankLib 2.18**
- **Required Python Packages:**
  ```bash
  pip install pandas tqdm transformers torch lxml
  ```

### Configuration
Hyperparameters and training settings are stored in `config.py`:
```python
is_train = True  # Set to False for testing
metrics = ['MAP', 'NDCG@1', 'DCG@1', 'P@1', 'RR@1', 'ERR@1']
tree_vals = [50, 100, 200, 500, 1000]
leaf_vals = [5, 10, 20, 50]
shrinkage_vals = [0.005, 0.01, 0.1]
```
Modify these settings as needed before running experiments.

## Running the Pipeline
### Training a Model
To train a new ranking model, run:
```bash
python model_train_test.py
```
This script trains models based on the hyperparameters defined in `config.py`.

### Running Ranking with E5 Embeddings
To rank documents using E5 embeddings:
```bash
python run_ranking_E5.py
```
This will generate ranked output based on the input queries.

### Selecting the Best Model
To find the best-performing ranking model:
```bash
python baseline_model_choice.py
```
This script evaluates multiple models and saves ranking scores.

## Dataset Format
### `t_data.csv`
This dataset contains historical ranking data. The format includes:
```
docno, query_id, round_no, username, position, current_document
```
### `feature_data_asrc_t.csv`
Contains ranking features and labels for training. Sample columns:
```
query_id, docno, rank, score, rank_promotion
```

## Notes
- Ensure all necessary dataset files are available before running scripts.
- Adjust hyperparameters in `config.py` based on experimentation.

---
For any issues or further improvements, feel free to contribute or raise an issue!
