#!/usr/bin/env python3
"""
Fireworks.ai Supervised Fine-Tuning Helper

Steps:
1. Create a dataset record
2. Upload a local JSONL file
3. Launch a fine-tuning job
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import requests
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
load_dotenv()  # look for .env in cwd

API_TOKEN = os.getenv("FIREWORKS_API_TOKEN")
ACCOUNT_ID = os.getenv("FIREWORKS_ACCOUNT_ID")

if not API_TOKEN or not ACCOUNT_ID:
    logging.error("Environment variables FIREWORKS_API_TOKEN and FIREWORKS_ACCOUNT_ID must be set.")
    sys.exit(1)

BASE_URL = f"https://api.fireworks.ai/v1/accounts/{ACCOUNT_ID}"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
HEADERS_JSON = {**HEADERS, "Content-Type": "application/json"}

DATA_DIR = Path("./fireworks_datasets")


# -----------------------------------------------------------------------------
# API helper functions
# -----------------------------------------------------------------------------
def create_dataset(dataset_id: str) -> dict:
    """Create a new dataset record."""
    url = f"{BASE_URL}/datasets"
    payload = {"datasetId": dataset_id, "dataset": {"userUploaded": {}}}
    resp = requests.post(url, headers=HEADERS_JSON, json=payload)
    resp.raise_for_status()
    return resp.json()


def upload_dataset_file(dataset_id: str, file_path: Path) -> dict:
    """Upload a local file to an existing dataset."""
    url = f"{BASE_URL}/datasets/{dataset_id}:upload"
    with file_path.open("rb") as f:
        resp = requests.post(url, headers=HEADERS, files={"file": f})
    resp.raise_for_status()
    return resp.json()


def create_fine_tuning_job(
    dataset_id: str,
    display_name: str,
    base_model: str,
    output_model: str,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    max_context_length: int = 2048,
    lora_rank: int = 8,
    early_stop: bool = False,
    is_turbo: bool = True,
) -> dict:
    """Submit a supervised fine-tuning job."""
    url = f"{BASE_URL}/supervisedFineTuningJobs"
    payload = {
        "displayName": display_name,
        "dataset": f"accounts/{ACCOUNT_ID}/datasets/{dataset_id}",
        "baseModel": base_model,
        "outputModel": output_model,
        "epochs": epochs,
        "learningRate": learning_rate,
        "maxContextLength": max_context_length,
        "loraRank": lora_rank,
        "earlyStop": early_stop,
        "isTurbo": is_turbo,
        "wandbConfig": {"enabled": False},
    }
    resp = requests.post(url, headers=HEADERS_JSON, json=payload)
    resp.raise_for_status()
    return resp.json()


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fireworks.ai: create dataset, upload file, and fine-tune"
    )
    parser.add_argument(
        "--dataset-id", required=True, help="Unique identifier for your dataset"
    )
    parser.add_argument(
        "--local-file",
        type=Path,
        required=True,
        help="Path to the .jsonl file you want to upload",
    )
    parser.add_argument(
        "--display-name", required=True, help="Name for the fine-tuning job"
    )
    parser.add_argument(
        "--base-model", required=True, help="URI of the base model"
    )
    parser.add_argument(
        "--output-model", required=True, help="URI for the resulting fine-tuned model"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Creating dataset record '%s'…", args.dataset_id)
    ds = create_dataset(args.dataset_id)
    logging.info("✅ Dataset created: %s", ds)

    logging.info("Uploading file '%s'…", args.local_file)
    up = upload_dataset_file(args.dataset_id, args.local_file)
    logging.info("✅ Upload response: %s", up)

    logging.info("Launching fine-tuning job '%s'…", args.display_name)
    ft = create_fine_tuning_job(
        args.dataset_id,
        args.display_name,
        args.base_model,
        args.output_model,
    )
    logging.info("✅ Fine-tuning job created: %s", ft)


if __name__ == "__main__":
    main()
