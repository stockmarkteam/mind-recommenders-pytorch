from logging import getLogger
import warnings


import hydra
import pandas as pd
from hydra.utils import instantiate

from .utils import init_random_seed, register_omegaconf_resolvers

logger = getLogger(__name__)

# custom resolver is used for logging and count categories
register_omegaconf_resolvers()
# supress warnings from pytorch-lightning
warnings.filterwarnings("ignore")


def read_data(path_conf, choices):
    samples = pd.read_pickle(path_conf.samples)
    if choices.dataset == "precomputed":
        articles = pd.read_pickle(path_conf.precomputed_articles)
    else:
        articles = pd.read_pickle(path_conf.articles)

    return samples, articles

@hydra.main(config_path="config", config_name="config")
def main(config):
    train_samples, train_articles = read_data(config.data_path.train, config.default_choices)
    logger.info("train data loaded")
    
    valid_samples, valid_articles = read_data(config.data_path.valid, config.default_choices)
    logger.info("valid data loaded")

    init_random_seed(config.hparams.data_shuffle_seed)

    train_dataset = instantiate(config.dataset.train, train_samples, train_articles)
    valid_dataset = instantiate(config.dataset.valid, valid_samples, valid_articles)

    logger.info("datasets initialized")

    train_loader = instantiate(config.data_loader.train, train_dataset)
    valid_loader = instantiate(config.data_loader.valid, valid_dataset)

    logger.info("dataloader initialized")

    init_random_seed(config.hparams.train_seed)

    pl_model = instantiate(
        config.pl_model,
        conf=config,
        _recursive_=False,
    )

    trainer = instantiate(config.trainer)
    trainer.fit(pl_model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
