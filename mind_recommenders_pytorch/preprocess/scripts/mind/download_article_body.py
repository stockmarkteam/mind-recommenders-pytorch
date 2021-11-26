import os
from logging import getLogger

logger = getLogger(__name__)


def download_article_body(raw_data_dir, data_categories):
    """
    {raw_data_dir}/{data_category}/news.tsvを参照し、{raw_data_dir}/{data_category}/artile_body.jsonにスクレイピングしたbodyを保存
    """
    logger.info("--- download article body ---")
    for data_category in data_categories:
        logger.info(f"start: run scraping for {data_category} data.")
        news_path = os.path.join(raw_data_dir, data_category, "news.tsv")
        body_path = os.path.join(raw_data_dir, data_category, "article_body.json")
        _scrape_each_data(news_path, body_path)


def _scrape_each_data(news_path, body_path):
    # NOTE: 既存scrapingアルゴリズムは出力先をMIND/crawler/から変更できない模様
    tmp_output_path = os.path.join(os.environ["WORKDIR"], "MIND/crawler/msn.json")

    # ファイルが存在していると上書きではなく追記になってしまうのでremove
    if os.path.exists(tmp_output_path):
        os.remove(tmp_output_path)

    os.system(
        f"chdir {os.path.dirname(tmp_output_path)}; "
        f"MIND_NEWS_PATH={news_path} scrapy crawl msn -o msn.json --loglevel INFO; "
        f"mv msn.json {body_path}"
    )
