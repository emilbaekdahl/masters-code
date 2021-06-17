This repository contains code for my maters's theisis.

`kgdata/model.py` contains a PyTorch Lightning `Module` and `DataModule` which together encpasulates the entire model.


### Dependencies
The project's dependencies are listed in Pipfile and can be installed using pipenv by running `pipenv install` in the project root.


### Downloading Data
The `kgdata` module provides a simple CLI that can help with downloading the FB15K237, WN18RR, YAGO3-10, and OpenBioLink knowledge graphs.
This is triggere by the command `pipenv run python -m kgdata download <kg> <folder>`
For instance, the following command downloads WN18RR (`wn`) into a folder called `wordnet`.
```
pipevn run python -m kgdata download wn wordnet
```


### Training
`train.py` provides a script for training, validating, and testing the model. Example:
```
pipenv run python train.py --path wn --batch_size 256 --emb_dim 1000 --optimiser adam --max_epochs 100 --early_stopping val_mrr
```
This trains a model on a knowledge graph located in the `wn` folder with batches of size 256.
The relation embeddings learned by the model have 1000 dimensions and are optimised using the Adam algorithm
We train for at most 100 epochs by stop early if the validation MRR score does not improve.

Besides standard PyTorch parameters, the CLI accpets the following:
 * `--path`: Location of data. The program expects this folder to contain the files `train.csv`, `valid.csv`, and `test.csv`. Each CSV file must have "head", "relation", and "tail" as columns.
 * `--emb_dim`: Number of dimensions for relation embeddings.
 * `--pooling`: Aggregation function for summarising similarity scores.
 * `--optimiser`: Optimisation algorithm for learning.
 * `--learning_rate`: (Initial) learning rate for the optmisation algorithm
 * `--subgraph_sampling`: Type of neighbourhood sampling to use.
 * `--neg_rate`: Number of negative triples to generate per positive.
 * `--max_paths`: Maximum number of relation sequences to extract per entity pair.
 * `--min_path_length`: Only extract relations sequences of at least this length.
 * `--max_path_length`: Only extract relation sequences of at most this length.
 * `--shuffle_train`: Shuffle the training dataset.
 * `--early_stopping`: Metric used to determine whether to stop training early.
