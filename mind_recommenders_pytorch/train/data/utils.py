import torch

from .objects import ArticleInput, SequenceInput

SEQUENCE_STACK_RULES = {
    "input_ids": torch.vstack,
    "token_type_ids": torch.vstack,
    "attention_mask": torch.vstack,
}


def _stack_sequence_inputs(sequence_inputs):
    return SequenceInput(
        **{key: SEQUENCE_STACK_RULES[key]([getattr(x, key) for x in sequence_inputs]) for key in SequenceInput._fields}
    )


ARTICLE_STACK_RULES = {
    "title": _stack_sequence_inputs,
    "body": _stack_sequence_inputs,
    "category": torch.hstack,
    "subcategory": torch.hstack,
    "n_articles": sum,
}


def stack_article_inputs(article_inputs):
    return ArticleInput(
        **{
            key: ARTICLE_STACK_RULES[key]([getattr(x, key) for x in article_inputs])
            for key in ArticleInput._fields
        }
    )
