# mind-recommenders-pytorch
MINDデータセットを利用して、ニュース推薦モデル（[NAML](https://www.ijcai.org/proceedings/2019/536), [NRMS](https://aclanthology.org/D19-1671/)）をカスタマイズしつつ学習するためのコード群が置かれています。


## 動作環境
linux上のdockerで動作させる想定です。以下の環境で動作確認済みです。

* linux (Ubuntu 20.04 LTS)
* docker: version 20.10.6, build 370c289
* docker-compose: 1.29.1, build c34c88b2
* nvidia-container-toolkit: 1.5.1-1 amd64
* GPU: NVIDIA GeForce RTX 2080 Ti 


## セットアップ
### 1. リポジトリのclone
```
git clone --recursive https://github.com/stockmarkteam/mind-recommenders-pytorch
```
### 2. `.env`の作成
```
mv mind-recommenders-pytorch
cp .env.sample .env
```
`.env`に定義された環境変数は以下のとおりです。必要に応じて変更可能ですが、以降の説明はデフォルト設定を前提として行われます。
* `COMPOSE_PROJECT_NAME`:
    * docker-composeの環境変数。詳細は[こちら](https://docs.docker.com/compose/reference/envvars/#compose_project_name)。
* `DEVICE`（デフォルト値：`gpu`）: 
    * dockerで利用するデバイスを指定します。`gpu, cpu`のうちいずれかを選択してください。一応切り替えができるようになっていますが、`cpu`設定での前処理/学習スクリプトの動作は未確認です。
* `DATASET_PATH`（デフォルト値：`$(PWD)/dataset`）:
    * mind datasetを保存するhostディレクトリ。container上では`dataset/`にmountされます。
* `MODEL_PATH`（デフォルト値：`$(PWD)/models`）:
    * GloVe, Transformerのpretrained modelを保存するhostディレクトリ。container上では`models/`にmountされます。
* `LOG_PATH` （デフォルト値：`$(PWD)/logs`）:: 
    * 学習のログを保存するhostディレクトリ。container上では`logs/`にmountされます。
* `VENV_PATH`:
    * pythonの仮想環境をinstallするhostディレクトリ。container上では`.venv/`にmountされます。
* `JUPYTER_PORT`:
    * container上で立ち上げたjupyter notebookにhostOS上のブラウザからアクセスするためbindするportを指定します。（default:`8888`）
* `TENSORBOARD_PORT`:
    * container上で立ち上げたtensorboardにhostOS上のブラウザからアクセスするためbindするportを指定します。（default: `6006`）
### 3. docker環境のsetup
```bash
make setup
```
### 4. データセットのDL
[公式サイト](https://msnews.github.io/)から訓練データセット・開発データセットのzipファイルをDLして、containerから見える場所に配置してください。
迷ったらこのREADMEと同じディレクトリに配置すれば問題ありません。

### 5. containerに入る
```bash
make sh
```

## 前処理
コンテナ内で以下のコマンドを実行することにより、必要な前処理が全て行われます。
```bash
pipenv run preprocess-all data_path.train_zip=<MINDxxx_train.zipのpath> data_path.valid_zip=<MINDxxx_dev.zipへのpath>
```
通常版データセットを利用する場合は、上記コマンドの引数に`params.dataset_type=large`を追加してください。

ここで行われる各処理の概要ついては、[こちら](doc/preprocess.md)をご確認ください。

## 学習
コンテナ内で以下のコマンドを実行することにより、モデルが学習できます。
```bash
pipenv run train
```

[当社のブログ記事](https://tech.stockmark.co.jp/blog/20211120_mind_discovery/)で言及した12通りのモデルをすべて学習したい場合は、以下のコマンドを実行してください。
```bash
pipenv run train-all
```


指定できるオプションの一例は以下のとおりです。
* `model`: 
    * `naml` or `nrms` (default: `nrms`)
* `embedding_layer`:
    *  `word_embedding` or `transformer` (default: word_embedding)
* `hparams.article_attributes`:
    *  利用する記事属性を`[title,body,category,subcategory]`から指定（default: `[title,body,category,subcategory]`)
* `hparams.n_epochs`: 
    * 訓練のエポック数
* `hparams.max_title_length`:
    *  最大タイトルトークン長（default: `30`)
* `hparams.max_body_length`:
    *  最大本文トークン長（default: `128`）
* `hparams.batch_size.train`:
    * train datasetのbatch size（default: 利用するembedding layerに応じて変化）
* `hparams.batch_size.valid`:
    * validation datasetのbatch size（default: 利用するembedding layerに応じて変化）
* `hparams.accumulate_grad_batches`: 
    * この数値と同じstep数が経過するたびに勾配を更新します。これにより実質的な訓練バッチサイズは`hparams.batch_size.train` * `hparams.accumulate_grad_batches`になります。
* `dataset`:
    * `precomputed`を指定すると、事前にシリアライズされた記事を学習に用います。毎stepごとに行われる記事データのシリアライズ処理をスキップできるので、学習が高速化されます。
* `num_workers`: 
    * DataLoaderのworker数（default: `4`）

本ライブラリではコマンドラインパーサとして[hydra](https://github.com/facebookresearch/hydra)を用いているため、[config](mind_recommenders_pytorch/train/config)で定義されている値は全てコマンドラインから書き換え可能になっています。

## 学習結果の確認
```
pipenv run tensorboard
```
host OS上のブラウザで`localhost:${TENSORBOARD_PORT}`にアクセスするとログが確認できます。
