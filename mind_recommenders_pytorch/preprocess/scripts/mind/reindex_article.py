import os
import pandas as pd
from logging import getLogger

logger = getLogger(__name__)

class ArticleReindexer:
    @classmethod
    def run(cls, processed_data_dir, data_categories):
        for data_category in data_categories:
            logger.info(f"start reindexing articles for {data_category} data")
            article_path = os.path.join(processed_data_dir, data_category, "articles.pkl")
            behavior_path = os.path.join(processed_data_dir, data_category, "behaviors.pkl")
            cls._run_each_data(article_path, behavior_path)
    
    @classmethod
    def _run_each_data(cls, article_path, behavior_path):
        articles = pd.read_pickle(article_path)
        behaviors = pd.read_pickle(behavior_path)
        logger.info("files loaded.")

        articles = cls._reindex_articles(articles)
        id_mapper = dict(zip(articles.original_id, articles.id))
        behaviors = cls._reindex_behaviors(behaviors, id_mapper)

        logger.info("reindex finished.")

        articles.to_pickle(article_path)
        behaviors.to_pickle(behavior_path)
        logger.info("files saved.")

    @staticmethod
    def _reindex_articles(articles):
        articles = articles.rename(columns={"id": "original_id"})
        articles["id"] = articles.index
        return articles

    @staticmethod
    def _reindex_behaviors(behaviors, id_mapper):
        behaviors["history"] = behaviors.history.map(lambda history: [id_mapper[article_id] for article_id in history])
        behaviors["impressions"] = behaviors.impressions.map(
            lambda impressions: [(id_mapper[impression[0]], impression[1]) for impression in impressions]
        )
        return behaviors
