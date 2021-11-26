import os
from logging import getLogger

import pandas as pd

logger = getLogger(__name__)


def preprocess_article_category(processed_data_dir, data_categories):
    category2article = dict()

    for data_category in data_categories:
        article_path = os.path.join(processed_data_dir, data_category, "articles.pkl")
        category2article[data_category] = pd.read_pickle(article_path)

    total_articles = pd.concat(list(category2article.values()))
    categories = pd.Categorical(total_articles.category).categories
    subcategories = pd.Categorical(total_articles.subcategory).categories

    logger.info(f"categories: {len(categories)}")
    logger.info(f"subcategories: {len(subcategories)}")

    with open(os.path.join(processed_data_dir, "categories.txt"), "w") as file:
        for category in categories:
            file.write(f"{category}\n")
    
    with open(os.path.join(processed_data_dir, "subcategories.txt"), "w") as file:
        for subcategory in subcategories:
            file.write(f"{subcategory}\n")
