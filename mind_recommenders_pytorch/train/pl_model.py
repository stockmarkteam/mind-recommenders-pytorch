from collections.abc import Sequence
from itertools import chain

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate

from .evaluation.evaluation_result_writer import EvaluationResultWriter
from .evaluation.evaluator import Evaluator
from .evaluation.inference_result_aggregator import InferenceResultAggregator
from .evaluation.inferencer import Inferencer
from .models.layers.multi_triplet_loss import MultiTripletLoss

class PlModel(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.model = instantiate(conf.model)
        self.loss_func = MultiTripletLoss()
        self.save_hyperparameters(dict(conf.hparams))
        self.automatic_optimization = False

    def forward(self, candidates, histories, n_candidates_per_sample):
        return self.model(candidates, histories, n_candidates_per_sample)

    def training_step(self, batch, batch_idx):
        candidates, n_candidates_per_sample, targets, histories = batch
        out = self.forward(candidates, histories, n_candidates_per_sample)
        if out == None: return

        loss = self.loss_func(out, targets, n_candidates_per_sample)

        self.log("train_loss", loss, prog_bar=True)
        self.manual_backward(loss/self.hparams.accumulate_grad_batches)

        
        # deal with multiple optimizers
        optimizers = self.optimizers()
        if not isinstance(optimizers, Sequence):
            optimizers = [optimizers]
        
        # accumulate gradients
        if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0:
            for opt in optimizers:
                opt.step()
                opt.zero_grad()
    
    def test_step(self, batch, batch_idx):
        return self._evaluation_step(batch)

    def validation_step(self, batch, batch_idx):
        return self._evaluation_step(batch)

    def test_epoch_end(self, outputs):
        self._evaluation_epoch_end(outputs, "test")

    def validation_epoch_end(self, outputs):
        self._evaluation_epoch_end(outputs, "val")

    def _evaluation_step(self, batch):
        return Inferencer.run(self, batch)

    def _evaluation_epoch_end(self, outputs, phase_name):
        outputs = InferenceResultAggregator.run(outputs)
        result = Evaluator.run(outputs)
        EvaluationResultWriter(self, phase_name).run(result)

    def configure_optimizers(self):
        if isinstance(self.hparams.optim, Sequence):
            return [hydra.utils.instantiate(conf, params=getattr(self, conf.params)) for conf in self.hparams.optim]
        else:
            return hydra.utils.instantiate(self.hparams.optim, params=getattr(self, self.hparams.optim.params))

    @property
    def trainable_params(self):
        return [param for param in self.parameters() if param.requires_grad == True]

    @property
    def sparse_params(self):
        return self._separate_sparse_and_dense_params()[0]

    @property
    def dense_params(self):
        return self._separate_sparse_and_dense_params()[1]

    def _separate_sparse_and_dense_params(self):
        param_to_name = {param: name for name, param in self.named_parameters()}
        sparse_params = list(
            chain.from_iterable(
                [
                    module.parameters()
                    for module in self.modules()
                    if isinstance(module, torch.nn.Embedding) and module.sparse is True
                ]
            )
        )
        sparse_param_names = set([param_to_name[param] for param in sparse_params])
        dense_params = [param for name, param in self.named_parameters() if name not in sparse_param_names]
        return sparse_params, dense_params
