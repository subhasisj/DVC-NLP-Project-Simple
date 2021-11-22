import argparse
import logging
import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from src.utils.common import create_directories, read_yaml

STAGE = "Two - Feature Engineering"  ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    # Read Train and Test Data from Path
    processed_data_dir_path = os.path.join(artifacts_dir, artifacts["PROCESSED_DATA"])
    train_data_path = os.path.join(processed_data_dir_path, artifacts["TRAIN_DATA"])
    test_data_path = os.path.join(processed_data_dir_path, artifacts["TEST_DATA"])

    # Get Feature Engineering Data Path and create directory
    feature_dir = os.path.join(artifacts_dir, artifacts["FEATURES"]["FEATURE_DIR"])
    create_directories([feature_dir])

    # get features_train amd features_test
    features_train_path = os.path.join(
        feature_dir, artifacts["FEATURES"]["FEATURE_TRAIN_DATA"]
    )
    features_test_path = os.path.join(
        feature_dir, artifacts["FEATURES"]["FEATURE_TEST_DATA"]
    )

    # get max features and ngram_range from params
    max_features = params["FEATURES"]["MAX_FEATURES"]
    ngram_range = params["FEATURES"]["NGRAM_RANGE"]

    # read the processed train and test data in TSV format from artifacts directory
    train_data = pd.read_csv(train_data_path, sep="\t", encoding="utf-8",names=['id','text','label'])
    logging.info(
        f"Training data read successfully from {train_data_path} , Shape: {train_data.shape}"
    )
    test_data = pd.read_csv(test_data_path, sep="\t", encoding="utf-8",names=['id','text','label'])
    logging.info(
        f"Testing data read successfully from {test_data_path} , Shape: {test_data.shape}"
    )

    # convert the data to lower case
    train_data["text"] = train_data["text"].str.lower()
    test_data["text"] = test_data["text"].str.lower()

    print(train_data["text"][:10].values)




if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
