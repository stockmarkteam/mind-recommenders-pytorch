# Perceiver IO Recommender
This repository includes a sample implementation of Perceiver-IO recommender for a news recommendation task on MIND dataset, along with [NAML](https://www.ijcai.org/proceedings/2019/536) and [NRMS](https://aclanthology.org/D19-1671/) baseline implementations.

## System Requirements
This code can be run from Docker on Linux environment. We confirmed that it runs under below environment.

* Linux (Ubuntu 20.04 LTS)
* Docker: version 20.10.6, build 370c289
* docker-compose: 1.29.1, build c34c88b2
* nvidia-container-toolkit: 1.5.1-1 amd64
* GPU: NVIDIA GeForce RTX 2080 Ti 


## Setting up
### 1. Clone the repository branch
```
git clone --branch perceiver_io_recommender --recursive https://github.com/stockmarkteam/mind-recommenders-pytorch.git
```
**IMPORTANT NOTE:** Please be sure that you are cloning and working on perceiver_io_recommender branch. Main branch does not contain perceiver-io implementation
### 2. Setup environment parameters

Environment parameters need to be written in `.env` file. You can simply copy it from `.env.sample` to work with default parameters.

```
cd mind-recommenders-pytorch
cp .env.sample .env
```

These are the necessary parameters written in `.env` file, you can edit it if necessary.

* `COMPOSE_PROJECT_NAME`:
    * Needed for docker-compose. For details please [refer](https://docs.docker.com/compose/reference/envvars/#compose_project_name).
* `DEVICE`（Default: `gpu`）: 
    * Device setting for docker. Parameters can be set to `gpu` or `cpu`, but we tested the code only with `gpu` parameter.
* `DATASET_PATH`Default: `$(PWD)/dataset`）:
    * Host directory for MIND dataset. It is mounted to `dataset/` directory from the container.
* `MODEL_PATH`（Default: `$(PWD)/models`）:
    * Host directory for pretrained models for GloVe and Transformer. It is mounted to `models/` directory from the container.
* `LOG_PATH` （Default: `$(PWD)/logs`）:
    * Host directory for training logs. It is mounted to `logs/` folder from the container.
* `VENV_PATH`:
    * Host directory for python virtual environment. It is mounted `.venv/` folder from the container.
* `JUPYTER_PORT`:（Default: `8888`）
    * The port number binded for the host OS access to the jupyter notebook that is launched in the container.
* `TENSORBOARD_PORT`: （Default: `6006`）
    * The port number binded for the host OS access to the tensorboard that is launched in the container.

### 3. Setup Docker Environment
```bash
make setup
```

### 4. Download dataset

Download [MIND dataset](https://msnews.github.io/) and put the zip file to a directory which is visible from the container. You can put it to the same folder with README.

### 5. Enter the container
```bash
make sh
```

## Preprocessing
Run below command in container to do all necessary preprocessing.
```bash
pipenv run preprocess-all data_path.train_zip=<path/to/MINDxxx_train.zip> data_path.valid_zip=<path/to/MINDxxx_dev.zip>
```
If you are working with the large dataset, please add this parameter to above command:

`params.dataset_type=large`


## Training
Run below command in container for training the perceiver-io model.
```bash
pipenv run train
```

Some of the optional parameters are listed below.

* `model`: 
    * `naml` or `nrms` (default: `nrms`)
* `embedding_layer`:
    *  `word_embedding` or `transformer` (default: word_embedding)
* `hparams.article_attributes`:
    *  Can be selected from [title,body,category,subcategory]（default: `[title,body,category,subcategory]`)
* `hparams.n_epochs`: 
    * default: 3
* `hparams.max_title_length`:
    *  Max. number of tokens from article titles（default: `30`)
* `hparams.max_body_length`:
    *  Max. number of tokens from article bodies（default: `128`）
* `hparams.batch_size.train`:
* `hparams.batch_size.valid`:
    * (Default batch sizes are different depending on the selected embedding layer）
* `hparams.accumulate_grad_batches`: 
    * Training batch size becomes `hparams.batch_size.train` * `hparams.accumulate_grad_batches`
* `dataset`:
    * If it is set to `precomputed`, it reads from serialized article text data hence fetching data during training can be speeded up.
* `num_workers`:
    * For dataLoader（default: `4`）

This library uses [hydra](https://github.com/facebookresearch/hydra) as config manager and  everything in [config](mind_recommenders_pytorch/train/config) can be overwritten from the command line.

## Check training results
```
pipenv run tensorboard
```
You can browse results from this link `localhost:${TENSORBOARD_PORT}` in the host.
