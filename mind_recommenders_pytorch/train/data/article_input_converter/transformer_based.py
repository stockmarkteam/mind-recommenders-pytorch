from transformers import AutoTokenizer

from ..objects import ArticleInput, SequenceInput
from .category_converter import CategoryConverter


class ArticleInputConverter:
    def __init__(
        self,
        max_title_length,
        max_body_length,
        tokenizer_path,
        category_file_path=None,
        subcategory_file_path=None,
    ):

        self.max_title_length = max_title_length
        self.max_body_length = max_body_length

        self.category_converter = CategoryConverter(category_file_path, subcategory_file_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def run(self, articles, attributes):
        attribute_dics = {attr: getattr(self, f"_convert_{attr}")(articles) for attr in attributes}
        return ArticleInput(**attribute_dics, n_articles=len(articles))

    def _convert_category(self, articles):
        return self.category_converter.convert_category(articles)

    def _convert_subcategory(self, articles):
        return self.category_converter.convert_subcategory(articles)

    def _convert_title(self, articles):
        return self._convert_sequence(articles, "title")

    def _convert_body(self, articles):
        return self._convert_sequence(articles, "body")

    def _convert_sequence(self, articles, attribute):
        encoded_dic = self.tokenizer.batch_encode_plus(
            articles[attribute].tolist(),
            padding="max_length",
            max_length=self.max_body_length,
            return_tensors="pt",
            truncation=True,
        )
        return SequenceInput(**encoded_dic)
