import os
from itertools import chain
from logging import getLogger
from typing import List

import pandas as pd

logger = getLogger(__name__)


def build_vocab_from_articles(
    processed_data_dir: str,
    data_categories: List[str],
    text_attributes: List[str],
    tokenizer,
) -> None:
    logger.info("--- build vocabulary from articles ---")
    vocab = set()
    for data_category in data_categories:
        logger.info(f"calculate vocab of {data_category} data")
        articles = pd.read_pickle(os.path.join(processed_data_dir, data_category, "articles.pkl"))

        vocab |= _calc_vocab(articles, text_attributes, tokenizer)

    logger.info(f"vocab size: {len(vocab)}")
    logger.info("writing vocabulary...")
    with open(os.path.join(processed_data_dir, "vocab.txt"), "w") as file:
        for word in vocab:
            file.write(f"{word}\n")


def _calc_vocab(articles, text_attributes, tokenizer):

    vocabs =  articles.parallel_apply(
        lambda article: 
            set(
                chain.from_iterable(
                    [set(tokenizer.tokenize(article[col])) for col in text_attributes]
                )
            )
        ,axis=1
    )
    return set(chain.from_iterable(vocabs.values))
