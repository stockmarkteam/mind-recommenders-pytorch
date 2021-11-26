import os
from logging import getLogger

import pandas as pd

logger = getLogger(__name__)


def precompute_article_input(
    processed_data_dir,
    article_input_converter,
    embedding_type,
    precompute_attributes,
    data_categories,
    parallel=True,
):
    for data_category in data_categories:
        logger.info(f"start preconputing {embedding_type} based article inputs for {data_category} data...")
        article_path = os.path.join(processed_data_dir, data_category, "articles.pkl")
        save_path = os.path.join(processed_data_dir, data_category, f"article_inputs_{embedding_type}.pkl")
        _preprocess_each_data(article_path, save_path, article_input_converter, precompute_attributes, parallel)

def _preprocess_each_data(article_path, save_path, article_input_converter, precompute_attributes, parallel):
    logger.info("loading articles...")

    articles = pd.read_pickle(article_path)
    series_of_article_df = _to_series_of_df(articles)

    logger.info("calculating article inputs...")
    if parallel:
        article_inputs = series_of_article_df.parallel_map(
            lambda x: article_input_converter.run(x, attributes=precompute_attributes)
        )
    else:
        article_inputs = series_of_article_df.progress_map(
            lambda x: article_input_converter.run(x, attributes=precompute_attributes)
        )

    logger.info("saving article inputs...")
    article_inputs = article_inputs.rename("article_input").to_frame()
    article_inputs.index = articles.index
    article_inputs.to_pickle(save_path)


def _to_series_of_df(df):
    # NOTE: input するdata formatをArticleInputConverterに合わせるため、Series of DataFrameに変換
    # faster than np.array_split
    return pd.Series([df[i : i + 1] for i in range(len(df))])
