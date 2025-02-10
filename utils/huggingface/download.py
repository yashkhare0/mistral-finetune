#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging
import subprocess
import yaml
import webbrowser
import time
from pandas import DataFrame
from huggingface_hub import snapshot_download, login, HfApi
from huggingface_hub.utils import GatedRepoError
from tqdm import tqdm
from ruamel.yaml import YAML


logger = logging.getLogger("download")
logging.basicConfig(level=logging.INFO)


def load_config(config_path: str = "config/7B.yaml") -> dict:
    """
    Loads the configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: The configuration dictionary.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration file {config_path}: {e}")
        raise


def _attempt_download(repo_id: str, token: str, models_path: Path):
    """
    Attempts to download the model from Hugging Face Hub.

    Args:
        repo_id (str): The Hugging Face model identifier.
        token (str): The Hugging Face authentication token.
        models_path (Path): Path where the model should be downloaded.

    Returns:
        None
    """
    try:
        logger.info(f"Attempting to download {repo_id} with token {'provided' if token else 'not provided'}")
        if token:
            login(token=token)
            logger.info("Successfully logged in to Hugging Face")
        
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"],
            local_dir=models_path,
            token=token,
        )
        logger.info("Model downloaded successfully!")
    except Exception as e:
        error_msg = str(e)
        if "Access to model" in error_msg and "is restricted" in error_msg:
            model_url = f"https://huggingface.co/{repo_id}"
            logger.info(f"""
                    Opening license agreement page in your browser: {model_url}
                    Please:
                    1. Log in with your Hugging Face account
                    2. Click on "Agree and access repository" button
                    3. Wait a few seconds for the agreement to be processed
            """)
            webbrowser.open(model_url)
            
            logger.info("Waiting 15 seconds for license acceptance...")
            time.sleep(15)
            
            logger.info("Retrying download...")
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"],
                local_dir=models_path,
                token=token,
            )
            logger.info("Model downloaded successfully!")
            return
        elif "Please enable access to public gated repositories" in error_msg:
            logger.error("""
                    Token permissions error. Please follow these steps:
                    1. Go to https://huggingface.co/settings/tokens
                    2. Either create a new token or edit your existing token
                    3. Enable the 'read' access under 'Access public gated models'
                    4. Copy the token and try again
            """)
            raise ValueError("Token needs 'read' permission for gated models") from e
        logger.error(f"Download error details: {error_msg}")
        raise


def _download_model(repo_id: str, token: str) -> Path:
    """
    Downloads the model from the Hugging Face Hub to a workspace subdirectory.

    Args:
        repo_id (str): The Hugging Face model identifier.
        token (str, optional): The Hugging Face authentication token.
        
    Returns:
        Path: Path to the downloaded model directory
    """
    try:
        models_path = Path("./models") / repo_id.split("/")[-1]
        
        if models_path.exists() and any(models_path.iterdir()):
            logger.info(f"Model already exists at {models_path}")
            return models_path
            
        models_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading model into {models_path}")
        _attempt_download(repo_id, token, models_path)
        return models_path
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise


def download_model(model_id: str, token: str) -> Path:
    """
    Loads the configuration and downloads the specified model using the Hugging Face token.

    Args:
        model_id (str): The model identifier
        token (str): The Hugging Face token
        
    Returns:
        Path: Path to the downloaded model directory
    """
    if token:
        logger.info("Using provided Hugging Face token.")
    else:
        logger.warning("No Hugging Face token provided in config; proceeding anonymously.")
    model_path = _download_model(model_id, token)
    logger.info("Download process complete!")
    return model_path


def _split_and_save_dataset(data_set: DataFrame, dataset_name: str, split_percentage: float = 0.95, random_state: int = 200) -> tuple[Path, Path]:
    """
    Splits the dataset into training and evaluation sets and saves them to disk.
    
    Args:
        data_set (DataFrame): The dataset to split
        dataset_name (str): Name to use for the dataset directory
        split_percentage (float): Percentage of data for training (0.0 to 1.0)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple[Path, Path]: Paths to the training and evaluation datasets
        
    Raises:
        ValueError: If split_percentage is not between 0 and 1
    """
    if not 0 < split_percentage < 1:
        raise ValueError("split_percentage must be between 0 and 1")
        
    try:
        output_dir = Path('dataset') / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = output_dir / 'train.jsonl'
        eval_path = output_dir / 'eval.jsonl'
        

        if train_path.exists() and eval_path.exists():
            logger.info(f"Dataset already exists at {output_dir}")
            return train_path, eval_path
        
        training_data_set = data_set.sample(frac=split_percentage, random_state=random_state)
        evaluation_data_set = data_set.drop(training_data_set.index)
        training_data_set.to_json(train_path, orient="records", lines=True)
        evaluation_data_set.to_json(eval_path, orient="records", lines=True)


        logger.info(f"Dataset split and saved: {len(training_data_set)} training samples, "
                   f"{len(evaluation_data_set)} evaluation samples")
        return train_path, eval_path
    except Exception as e:
        logger.error(f"Error splitting and saving dataset: {e}")
        raise


def download_dataset(data_set_url: str = "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/resolve/main/data/test_gen-00000-of-00001-3d4cd8309148a71f.parquet") -> tuple[Path, Path]:
    """
    Downloads the dataset from the Hugging Face Hub using the provided configuration.

    Args:
        data_set_url (str): URL to the dataset

    Returns:
        tuple[Path, Path]: Paths to the training and evaluation datasets
    """
    import pandas as pd
    try:
        dataset_name = data_set_url.split('/')[-1].split('.')[0]
        output_dir = Path('dataset') / dataset_name
        train_path = output_dir / 'train.parquet'
        eval_path = output_dir / 'eval.parquet'
        if train_path.exists() and eval_path.exists():
            logger.info(f"Dataset already exists at {output_dir}")
            return train_path, eval_path
        df = pd.read_parquet(data_set_url)
        return _split_and_save_dataset(df, dataset_name)
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise


def reformat_data(path: list[str]):
    """
    Reformat the data from the Hugging Face Hub using the provided configuration.
    """
    try:
        for p in path:
            logger.info(f"Reformatting data from {p}")
            command = ["python", "-m", "utils.reformat_data", p]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logger.info(f"Reformatting output for {p}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Reformatting failed for {p}: {e.stderr}")
        raise

def validate_data(path: str):
    """
    Reformat the data from the Hugging Face Hub using the provided configuration.

    Args:
        data_path (list[str]): List of paths to data files to validate
    """
    logger.info(f"Validating data from {path}")
    try:
        command = ["python", "-m", "utils.validate_data", "--train_yaml", path]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Validation output for {path}:\n{result.stdout}")
        if "incorrectly formatted" in result.stdout:
            logger.error("Data is incorrectly formatted, please run utils.reformat_data")
            return False
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Validation failed for {path}: {e.stderr}")
        raise


def update_config(config_path: str, results: dict, new_name: str):
    import os

    yaml = YAML()
    try:
        logger.info(f"Creating updated copy of config at {config_path}")
        directory = os.path.dirname(config_path)
        filename = os.path.basename(config_path)  # get just "example.yaml"
        base, ext = os.path.splitext(filename)      # split into "example" and ".yaml"
        new_config_path = f"{directory}/{new_name}{ext}" if new_name else f"{directory}/{base}_updated{ext}"
        with open(config_path, 'r') as f:
            config = yaml.load(f)
        model_path = str(results["model"])
        if "/app/" in config.get("model_id_or_path", ""):
            model_path = os.path.join("/app", model_path)
        config["model_id_or_path"] = model_path
        train_path = str(results["dataset"][0])
        eval_path = str(results["dataset"][1])
        if "/app/" in config["data"].get("instruct_data", ""):
            train_path = os.path.join("/app", train_path)
            eval_path = os.path.join("/app", eval_path)
        config["data"]["instruct_data"] = train_path
        config["data"]["eval_instruct_data"] = eval_path
        with open(new_config_path, 'w') as f:
            yaml.dump(config, f)
        logger.info(f"Updated config saved to: {new_config_path}")
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise


def process_handler(args: dict) -> dict:
    """
    Processes downloads in parallel and returns paths to downloaded files.
    
    Args:
        args (dict): Configuration dictionary with hugging_face settings
        
    Returns:
        dict: Paths to downloaded files {'model': Path, 'dataset': (train_path, eval_path)}
        
    Raises:
        ValueError: If token is missing or invalid
        Exception: If any download fails
    """
    logger.info(f"Processing downloads with args")
    results = {}
    token = args.get("hugging_face", {}).get("token", None)
    model_id = args.get("hugging_face", {}).get("model_id", None)
    dataset_url = args.get("hugging_face", {}).get("dataset_url", None)
    
    if not token:
        raise ValueError("Token is required for downloading this model")
    
    tasks = []
    if model_id:
        tasks.append(('model', lambda: download_model(model_id, token)))
    if dataset_url and dataset_url.endswith('.parquet'):
        tasks.append(('dataset', lambda: download_dataset(dataset_url)))
    
    if not tasks:
        logger.warning("No downloads requested - check your configuration")
        return results
        
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {executor.submit(task[1]): task[0] for task in tasks}
        with tqdm(total=len(tasks), desc="Downloads") as pbar:
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                    pbar.update(1)
                    logger.info(f"Successfully completed {key} download")
                except Exception as e:
                    logger.error(f"Error in {key} download: {str(e)}")
                    for f in futures:
                        f.cancel()
                    raise
    return results

def get_new_name(config: dict):
    model_name = config["hugging_face"]["model_id"].split('/')[-1]
    dataset_name = config["hugging_face"]["dataset_url"].split('/')[-1].split('.')[0]
    return f"c_{model_name}_{dataset_name}"

def main():

    """
    Entry point for command-line usage.
    Allows specifying an alternative configuration file.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Download a model from the Hugging Face Hub using a YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/local/7B.yaml",
        help="Path to the configuration YAML file (default: config/example/7B.yaml)",
    )
    parser.add_argument(
        "--validate",
        default=True,
        action="store_true",
        help="Validate the data after downloading (default: True)",
    )
    parser.add_argument(
        "--reformat",
        default=False,
        action="store_true",
        help="Reformat the data after downloading (default: False)",

    )
    parser.add_argument(
        "--update_config",
        default=False,
        action="store_true",
        help="Update the config after downloading (default: False)",
    )
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        results = process_handler(config)

        if 'dataset' in results:
            train_path, eval_path = results['dataset']
            if args.validate or args.reformat:
                data_is_valid = validate_data(args.config)
                if not data_is_valid:
                    logger.info("Data is incorrectly formatted, proceeding to reformat")
                    if args.reformat:
                        reformat_data([train_path, eval_path])
                        data_is_valid = validate_data(args.config)
                        if not data_is_valid:
                            logger.error("Data is still incorrectly formatted, please check the data")
                            raise ValueError("Data is incorrectly formatted, please check the data")
                        logger.info(f"Dataset downloaded and split:")
            logger.info(f"  - Training data: {train_path}")
            logger.info(f"  - Evaluation data: {eval_path}")
        if 'model' in results:
            logger.info(f"Model downloaded to: {results['model']}")
        if args.update_config:
            update_config(args.config, results, get_new_name(config))
    except Exception as e:
        logger.error(f"Download process failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
