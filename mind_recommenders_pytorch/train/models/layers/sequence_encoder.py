import inspect

import torch


class SequenceEncoder(torch.nn.Module):
    def __init__(self, token_embedding_layer, sentence_embedding_layer):
        super().__init__()
        self.token_embedding_layer = token_embedding_layer
        self.sentence_embedding_layer = sentence_embedding_layer
        
        self._word_embedding_args = self._inspect_input_args(self.token_embedding_layer)

    def forward(self, input_ids, attention_mask, **additional_args):
        token_embeddings = self.token_embedding_layer(
            **self._constract_arguments_for_token_embedding_layer(
                input_ids,
                attention_mask,
                additional_args,
            )
        )
        sequence_embeddings = self.sentence_embedding_layer(
            token_embeddings,
            attention_mask,
        )
        return sequence_embeddings

    @staticmethod
    def _inspect_input_args(module):
        return tuple([arg for arg in inspect.signature(module.forward).parameters.keys() if arg != "self"])

    def _constract_arguments_for_token_embedding_layer(self, input_ids, attention_mask, additional_args):
        args = {key: value for key, value in additional_args.items() if key in self._word_embedding_args}
        args["input_ids"] = input_ids
        if "attention_mask" in self._word_embedding_args:
            args["attention_mask"] = attention_mask
        return args
