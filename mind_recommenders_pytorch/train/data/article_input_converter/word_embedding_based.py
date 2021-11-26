import numpy as np
import torch
from nltk.parse import CoreNLPParser

from ..objects import SequenceInput, ArticleInput
from .category_converter import CategoryConverter


class ArticleInputConverter:
    def __init__(self, 
        max_title_length, 
        max_body_length,
        sentence_serializer,
        category_file_path=None,
        subcategory_file_path=None,
        ):
        self.max_title_length = max_title_length
        self.max_body_length = max_body_length
  
        self.sentence_serializer = sentence_serializer
        self.category_converter = CategoryConverter(category_file_path, subcategory_file_path)

        self.pad_idx = 0


    def run(self, articles, attributes):
        attribute_dics = {attr: getattr(self, f"_convert_{attr}")(articles) for attr in attributes}
        return ArticleInput(**attribute_dics,n_articles=len(articles))

    def _convert_category(self, articles):
        return self.category_converter.convert_category(articles)

    def _convert_subcategory(self, articles):
        return self.category_converter.convert_subcategory(articles)

    def _convert_title(self, articles):
        return self._convert_sequence(articles, "title")

    def _convert_body(self, articles):
        return self._convert_sequence(articles, "body")

    def _convert_sequence(self, articles, attribute):
        sequences = torch.IntTensor(
            np.stack(
                [
                    self._pad_sequence(self.sentence_serializer.run(attr), getattr(self, f"max_{attribute}_length"))
                    for attr in articles[attribute]
                ]
            )
        )

        masks = (sequences != self.pad_idx).type(torch.uint8)

        return SequenceInput(
            **{
                f"input_ids": sequences,
                f"attention_mask": masks,
            }
        )

    @staticmethod
    def _pad_sequence(array, max_seq_length):
        if len(array) >= max_seq_length:
            return np.array(array[:max_seq_length])
        else:
            return np.pad(array, [0, max_seq_length - len(array)])


class StanfordSentenceSerializer:
    def __init__(self, vocab_path):
        self.vocab = [x.strip() for x in open(vocab_path)]
        # pad tokenにindex 0 を利用したいため、1からindexを振る
        self.token2idx = dict(zip(self.vocab, range(1, len(self.vocab) + 1)))
        self.tokenizer = StanfordTokenizer()

    def run(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        ids = [self.token2idx[x] for x in tokens if x in self.token2idx.keys()]
        return ids


class StanfordTokenizer:
    def __init__(self):
        self.parser = CoreNLPParser()

    def tokenize(self, sentence):
        return list(self.parser.tokenize(sentence))
