import hydra
from hydra.utils import call
from pandarallel import pandarallel
from tqdm import tqdm

@hydra.main(config_path="config", config_name="config")
def main(config):
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)

    for job in config.target_jobs:
        call(config.job_definitions[job])


if __name__ == "__main__":
    main()
