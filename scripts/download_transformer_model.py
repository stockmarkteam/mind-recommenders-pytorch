from transformers import AutoTokenizer, AutoModel
import os
import argparse

def download_model(model_name):
    AutoTokenizer.from_pretrained(model_name).save_pretrained(
        os.path.join(
            os.environ["WORKDIR"],
            "models",
            "transformers",
            model_name,
        )
    )
    AutoModel.from_pretrained(model_name).save_pretrained(
        os.path.join(
            os.environ["WORKDIR"],
            "models",
            "transformers",
            model_name,
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='pretrained transformer model name listed here: https://huggingface.co/transformers/pretrained_models.html')
    args = parser.parse_args()
    download_model(args.model_name)
