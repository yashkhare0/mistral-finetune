#!/usr/bin/env python3

from pathlib import Path
import logging
import yaml
import webbrowser
import time

from huggingface_hub import snapshot_download, login, HfApi
from huggingface_hub.utils import GatedRepoError

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


def _download_model(repo_id: str, token: str):
    """
    Downloads the model from the Hugging Face Hub to a workspace subdirectory.

    Args:
        repo_id (str): The Hugging Face model identifier.
        token (str, optional): The Hugging Face authentication token.
    """
    try:
        models_path = Path("./models") / repo_id.split("/")[-1]
        models_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading model into {models_path}")
        _attempt_download(repo_id, token, models_path)
    except GatedRepoError:
        logger.warning("Access to the model is restricted. Attempting to login...")
        try:
            if not token:
                logger.info("""
                    To access this model, you need a Hugging Face token with proper permissions.
                    1. Go to https://huggingface.co/settings/tokens
                    2. Create a new token with 'read' access to public gated models
                    3. Enter the token below
                """)
                token = input("Please enter your Hugging Face token: ").strip()
            if token:
                login(token=token)
                logger.info("Login successful, retrying download...")
                _attempt_download(repo_id, token, models_path)
            else:
                logger.error("No token provided. Cannot download gated model.")
                raise ValueError("Token is required for downloading this model")
        except Exception as e:
            logger.error(f"Login failed or download error: {e}")
            raise
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise


def download(config_file: str = "config/7B.yaml"):
    """
    Loads the configuration and downloads the specified model using the Hugging Face token.

    Args:
        config_file (str): Path to the YAML configuration file.
    """
    config = load_config(config_file)
    model_id = config.get("model_id_or_path", "mistralai/Mistral-7B-v0.3")
    token = config.get("hugging_face", {}).get("token", None)
    logger.info(f"Preparing to download model: {model_id}")
    if token:
        logger.info("Using provided Hugging Face token.")
    else:
        logger.warning("No Hugging Face token provided in config; proceeding anonymously.")
    _download_model(model_id, token)
    logger.info("Download process complete!")


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
        default="config/7B.yaml",
        help="Path to the configuration YAML file (default: config/7B.yaml)",
    )
    args = parser.parse_args()
    download(args.config)

if __name__ == "__main__":
    main()
