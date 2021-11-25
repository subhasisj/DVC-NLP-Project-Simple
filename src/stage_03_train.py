import argparse
import logging
import os
import random
import shutil
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from src.utils.common import create_directories, read_yaml

STAGE = "Train" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    # get artifacts path from config
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    # Get Feature Engineering Data Path and create directory
    feature_dir = os.path.join(artifacts_dir, artifacts["FEATURES"]["FEATURE_DIR"])
    # get features_train amd features_test paths
    features_train_path = os.path.join(
        feature_dir, artifacts["FEATURES"]["FEATURE_TRAIN_DATA"]
    )

    # Get MODEL path from config
    model_dir = os.path.join(artifacts_dir, artifacts["MODEL"]["MODEL_DIR"])
    create_directories([model_dir])
    model_path = os.path.join(model_dir, artifacts["MODEL"]["MODEL_FILE"])  

    # Load Pickle format features from features_train_path 
    logging.info("Loading features from {}".format(features_train_path))
    features_train = joblib.load(features_train_path)
    labels_train = np.squeeze(features_train[:,1].toarray())
    X_train = features_train[:,2:]

    # Get Train parameters from PARAMS
    n_estimators = params["TRAIN"]["N_ESTIMATORS"]
    max_depth = params["TRAIN"]["MAX_DEPTH"]
    min_samples_split = params["TRAIN"]["MIN_SAMPLES_SPLIT"]
    seed = params["TRAIN"]["SEED"]

    # Train Random Forest Classifier
    logging.info("Training Random Forest Classifier")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=seed,
    )
    start_time = time.time()
    clf.fit(X_train, labels_train)
    end_time = time.time()
    logging.info("Training took {} seconds".format(end_time - start_time))

    # Save trained model to model_path
    logging.info("Saving trained model to {}".format(model_path))
    joblib.dump(clf, model_path)







if __name__ == '__main__':
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
