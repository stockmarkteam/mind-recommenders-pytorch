import os
from logging import getLogger
from pathlib import Path
from typing import List

import pandas as pd

logger = getLogger(__name__)


class ArticlePreprocessor:
    @classmethod
    def run(
        cls,
        raw_data_dir: str,
        processed_data_dir: str,
        data_categories: List[str],
    ) -> None:
        for data_category in data_categories:
            logger.info(f"start preprocessing articles for {data_category} data")
            article_path = os.path.join(raw_data_dir, data_category, "news.tsv")
            article_body_path = os.path.join(raw_data_dir, data_category, "article_body.json")
            save_path = os.path.join(processed_data_dir, data_category, "articles.pkl")
            cls._run_each_data(article_path, article_body_path, save_path)

    @classmethod
    def _run_each_data(cls, article_path, article_body_path, output_path):
        articles = cls._read_articles(article_path)
        article_bodies = cls._read_article_bodies(article_body_path)

        articles = articles.join(article_bodies, on="nid").drop("nid", axis=1)
        articles["body"] = articles.body.fillna("").astype(str)

        cls._show_article_stats(articles)
        articles.to_pickle(output_path)

    def _show_article_stats(articles):
        logger.info(f"total articles: {len(articles)}")
        is_title_empty = articles.title.map(len) == 0
        is_body_empty = articles.body.map(len) == 0
        logger.info(f"{is_title_empty.sum()} articles have no title.")
        logger.info(f"{is_body_empty.sum()} articles have no body.")
        logger.info(f"{ (is_body_empty & is_title_empty).sum()} articles have no title or body.")

    @staticmethod
    def _read_articles(path):
        articles = pd.read_csv(path, delimiter="\t", header=None)
        articles.columns = [
            "id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities",
        ]

        articles["nid"] = articles.url.map(lambda url: Path(url).stem)
        articles["title"] = articles.title.fillna("")
        articles["url"] = articles.url.fillna("")
        articles["abstract"] = articles.abstract.fillna("")

        logger.info(f"{len(articles)} articles loaded.")
        return articles

    @staticmethod
    def _read_article_bodies(path):
        article_bodies = pd.read_json(path, orient="records")
        article_bodies["body"] = article_bodies.body.map(
            lambda body: " ".join([sent.strip() for sent in body]) if isinstance(body, list) else ""
        )
        logger.info(f"{len(article_bodies)} article bodies loaded.")
        return article_bodies.set_index("nid")
