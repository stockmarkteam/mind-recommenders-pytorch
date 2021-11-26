import os
from logging import getLogger

import pandas as pd

logger = getLogger(__name__)


class BehaviorPreprocessor:
    @classmethod
    def run(cls, raw_data_dir, processed_data_dir, data_categories):
        for data_category in data_categories:
            logger.info(f"start preprocessing behaviors for {data_category} data")
            behavior_path = os.path.join(raw_data_dir, data_category, "behaviors.tsv")
            save_path = os.path.join(processed_data_dir, data_category, "behaviors.pkl")
            cls._run_each_data(behavior_path, save_path)

    @classmethod
    def _run_each_data(cls, input_path, output_path):
        behaviors = cls._read_behaviors(input_path)
        behaviors = cls._exclude_no_history_sample(behaviors)
        behaviors = cls._exclude_no_impression_sample(behaviors)
        behaviors["impressions"] = behaviors.impressions.progress_map(cls._parse_impressions)
        behaviors["history"] = behaviors.history.progress_map(cls._parse_history)

        logger.info(f"{len(behaviors)} behaviors remained.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        behaviors.to_pickle(output_path)

    @staticmethod
    def _read_behaviors(history_path):
        history_cols = ["impression_id", "user_id", "time", "history", "impressions"]
        histories = pd.read_csv(history_path, delimiter="\t", header=None)
        histories.columns = history_cols
        logger.info(f"{len(histories)} behaviors loaded.")
        return histories

    @classmethod
    def _exclude_no_history_sample(cls, histories):
        return cls._exclude_null_data(histories, "history")

    @classmethod
    def _exclude_no_impression_sample(cls, histories):
        return cls._exclude_null_data(histories, "impressions")

    @staticmethod
    def _exclude_null_data(histories, col_name):
        is_null = histories[col_name].isnull()
        logger.info(f"{is_null.sum()} behaviors has no {col_name}.")
        return histories[~is_null].reset_index(drop=True)

    @staticmethod
    def _parse_history(history: str):
        return history.strip().split(" ")

    @classmethod
    def _parse_impressions(cls, impressions: str):
        return [cls._parse_impression(impression) for impression in impressions.strip().split(" ")]

    @staticmethod
    def _parse_impression(impression: str):
        impression = impression.strip().split("-")
        impression[1] = int(impression[1])
        return impression
