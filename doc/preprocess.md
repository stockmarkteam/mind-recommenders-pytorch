# 各前処理の概要
ここでは前処理の実施内容を個別に記載します。
## 1. データセットの解凍＆ディレクトリへの配置

```bash
pipenv run extract-dataset data_path.train_zip=<MINDxxx_train.zipのpath> data_path.valid_zip=<MINDxxx_dev.zipへのpath>
```

通常版データセットを利用する場合は、上記コマンドの引数に`params.dataset_type=large`を追加してください。
これにより、`dataset/`に以下のようにファイルが配置されます。
```
dataset
└── mind
    └── <small or large>
        └── raw
            ├── train
            │   ├── behaviors.tsv
            │   ├── entity_embedding.vec
            │   ├── news.tsv
            │   └── relation_embedding.vec
            └── valid
                ├── behaviors.tsv
                ├── entity_embedding.vec
                ├── news.tsv
                └── relation_embedding.vec

```

## 2. 記事本文のDL
```bash
pipenv run download-article-body 
```
公開されているクローラを用いて記事本文をDLします。通常版データセットを利用する場合、引数に `params.dataset_type=large`を追加してください。


## 3. 各種モデルのDL
```bash
pipenv run download-models
```

学習済みGloVeモデル、学習済みtransformerモデル（bert-base-uncased）、Stanford CoreNLPをDLし、`models/`に以下のように配置します。

```
models/
├── glove
├── stanford-corenlp-4.3.1
└── transformers
```

## 4. Stanford CoreNLP Serverの起動
```bash
pipenv run launch-corenlp-server
```
Stanford Tokenizerを利用するため、CoreNLPサーバを起動します。


## 5. 記事の前処理
```bash
pipenv run preprocess-dataset
```
行動データ・記事データをモデルのフォーマットを扱いやすい形に変換します。
また、行動データの中からクリック履歴が存在しないサンプルを除外します。  
通常版データセットを利用する場合、引数に `params.dataset_type=large`を追加してください。


## 6. 学習済みGloVeモデルの前処理
```bash
pipenv run preprocess-glove
```
text形式で保存されたモデルから、語彙とEmbedding Matrixを分離します。


## 7. 記事データのシリアライズ（事前計算）
```bash
pipenv run precompute-article-input
```
記事データ中の各属性は、モデルへ入力する際に数値へマッピングする（シリアライズする）必要があります。
特にテキストデータを扱う場合は、マッピングの前にTokenizeの処理が必要となります。
これらの処理は学習中に行うこともできますが、このスクリプトによって事前に計算しておくこともできます。

Stanford TokenizerはtransformerモデルのTokenizerに比して処理が重いため、学習には事前計算したものを利用するのが推奨されます。


引数`hparams.max_title_length, hparams.max_body_length`によって最大タイトルトークン長、最大本文トークン長を変更できます。ただし、**ここで設定した長さは、学習時に変更できない**点には注意してください。

通常版データセットを利用する場合、引数に `params.dataset_type=large`を追加してください。

