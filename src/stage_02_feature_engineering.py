import argparse
import html
import logging
import os
from re import T, U
import shutil

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm

from src.utils.common import create_directories, read_yaml
from src.utils.featurize import save_features_in_matrix

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

    # get features_train amd features_test paths
    features_train_path = os.path.join(
        feature_dir, artifacts["FEATURES"]["FEATURE_TRAIN_DATA"]
    )
    features_test_path = os.path.join(
        feature_dir, artifacts["FEATURES"]["FEATURE_TEST_DATA"]
    )

    # get max features and ngram_range from params
    max_features = params["FEATURES"]["MAX_FEATURES"]
    ngrams = params["FEATURES"]["NGRAMS"]

    # read the processed train and test data in TSV format from artifacts directory
    train_data = pd.read_csv(
        train_data_path, sep="\t", encoding="utf-8", names=["id", "text", "label"]
    )
    logging.info(
        f"Training data read successfully from {train_data_path} , Shape: {train_data.shape}"
    )
    test_data = pd.read_csv(
        test_data_path, sep="\t", encoding="utf-8", names=["id", "text", "label"]
    )
    logging.info(
        f"Testing data read successfully from {test_data_path} , Shape: {test_data.shape}"
    )

    # Clean the text data
    clean_text_data(train_data, test_data)

    # Create Count and TFIDF Features
    ngram_range = (1, ngrams)
    X_train_tfidf, X_test_tfidf = create_features(
        max_features, ngram_range, train_data, test_data
    )

    # Save the features in CSR Matrix format
    save_features_in_matrix(train_data, X_train_tfidf, features_train_path)
    save_features_in_matrix(test_data, X_test_tfidf, features_test_path)


def create_features(max_features, ngram_range, train_data, test_data):
    # create a count vectorizer object
    count_vect = CountVectorizer(
        max_features=max_features, ngram_range=ngram_range, stop_words="english"
    )
    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of strings.
    X_train_counts = count_vect.fit_transform(train_data["text"])
    logging.info(
        f"Training data transformed successfully, Shape: {X_train_counts.shape}"
    )
    X_test_counts = count_vect.transform(np.array(test_data["text"]))
    logging.info(f"Testing data transformed successfully, Shape: {X_test_counts.shape}")

    # get the feature names
    feature_names = count_vect.get_feature_names()
    logging.info(f"Total number of Features: {len(feature_names)}")

    # create a tfidf vectorizer object
    tfidf_vect = TfidfTransformer(
        smooth_idf=False,
    )
    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of strings.
    X_train_tfidf = tfidf_vect.fit_transform(X_train_counts)
    logging.info(
        f"Training data transformed successfully, Shape: {X_train_tfidf.shape}"
    )
    X_test_tfidf = tfidf_vect.transform(X_test_counts)
    logging.info(f"Testing data transformed successfully, Shape: {X_test_tfidf.shape}")


    return X_train_tfidf, X_test_tfidf

def clean_text_data(train_data, test_data):
    # convert the data to lower case
    train_data["text"] = train_data["text"].str.lower().values.astype("U")
    test_data["text"] = test_data["text"].str.lower().values.astype("U")

    # unescape all HTML tags
    train_data["text"] = train_data["text"].apply(html.unescape)
    test_data["text"] = test_data["text"].apply(html.unescape)

    # remove all html tags
    train_data["text"] = train_data["text"].str.replace("<.*?>", " ")
    test_data["text"] = test_data["text"].str.replace("<.*?>", " ")


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
