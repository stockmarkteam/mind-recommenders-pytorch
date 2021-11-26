import os
from itertools import chain
from logging import getLogger
from typing import List, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = getLogger(__name__)


class GlovePreprocessor:
    @classmethod
    def run(
        cls,
        glove_path,
        processed_data_dir,
        filter_vocab_by_articles=True,
        data_categories=None,
        text_attributes=None,
        tokenizer=None,
    ):
        """
        .txt形式のglove vectorをvocebとvector matrixに分けて保存する
        """
        logger.info("--- convert glove file ---")

        if filter_vocab_by_articles:
            vocab, vectors = cls._build_vocab_with_articles(
                glove_path,
                processed_data_dir,
                data_categories,
                text_attributes,
                tokenizer,
            )
        else:
            vocab, vectors = cls._build_vocab(glove_path)

        logger.info(f"final vocab size: {len(vocab)}")
        logger.info("start writing vocab...")

        vocab_path = os.path.join(processed_data_dir, "vocab.txt")
        embedding_path = os.path.join(processed_data_dir, "embeddings.npy")

        with open(vocab_path, "w") as file:
            for word in vocab:
                file.write(f"{word}\n")

        embeddings = np.vstack(vectors)
        logger.info(f"matrix size: {embeddings.shape}")

        logger.info("start writing embedding matrix...")
        np.save(embedding_path, embeddings)

    @classmethod
    def _build_vocab_with_articles(
        cls,
        glove_path: str,
        processed_data_dir: str,
        data_categories: List[str],
        text_attributes: List[str],
        tokenizer: object,
    ) -> Set[str]:
        """
        gloveの語彙のうち、記事に含まれるもののみを取り出す
        embedding vectorのサイズが小さくなるのでメモリ消費量が抑えられるが、未知語が増える
        """
        vocab = list()
        vectors = list()

        article_vocab = ArticleVocabularyBuilder.run(
            processed_data_dir,
            data_categories,
            text_attributes,
            tokenizer,
        )
        for line in tqdm(open(glove_path)):
            word, vector = cls._parse_line(line)
            if word in article_vocab:
                vocab.append(word)
                vectors.append(np.array(vector).astype(np.float32))

        return vocab, vectors

    @classmethod
    def _build_vocab(
        cls,
        glove_path: str,
    ) -> Set[str]:
        vocab = list()
        vectors = list()

        for line in tqdm(open(glove_path)):
            word, vector = cls._parse_line(line)
            vocab.append(word)
            vectors.append(np.array(vector).astype(np.float32))
        return vocab, vectors

    @staticmethod
    def _parse_line(line):
        elements = line.rstrip().split(" ")
        word = elements[0]
        vector = elements[1:]
        return word, vector


class ArticleVocabularyBuilder:
    @classmethod
    def run(
        cls,
        processed_data_dir: str,
        data_categories: List[str],
        text_attributes: List[str],
        tokenizer,
    ) -> Set[str]:
        """
        記事に使われている単語からvocaburaryを作成する
        """
        logger.info("--- build vocabulary from articles ---")
        vocab = set()
        for data_category in data_categories:
            logger.info(f"calculate vocab of {data_category} data")
            articles = pd.read_pickle(os.path.join(processed_data_dir, data_category, "articles.pkl"))

            vocab |= cls._get_article_vocab(articles, text_attributes, tokenizer)

        return vocab

    @staticmethod
    def _get_article_vocab(articles, text_attributes, tokenizer):
        vocabs = articles.parallel_apply(
            lambda article: set(
                chain.from_iterable([set(tokenizer.tokenize(article[col])) for col in text_attributes])
            ),
            axis=1,
        )
        return set(chain.from_iterable(vocabs.values))
