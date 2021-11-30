import argparse
import os
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, save_json
import joblib
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
)


STAGE = "Evaluate"  ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)


def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    # get artifacts path from config
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    # Get Feature Engineering Data Path and create directory
    feature_dir = os.path.join(artifacts_dir, artifacts["FEATURES"]["FEATURE_DIR"])

    # get features_test paths
    features_test_path = os.path.join(
        feature_dir, artifacts["FEATURES"]["FEATURE_TEST_DATA"]
    )

    # Get MODEL path from config
    model_dir = os.path.join(artifacts_dir, artifacts["MODEL"]["MODEL_DIR"])
    model_path = os.path.join(model_dir, artifacts["MODEL"]["MODEL_FILE"])

    # load Model from path
    logging.info("Loading Model from {}".format(model_path))
    model = joblib.load(model_path)

    # Load Pickle format features from features_test_path
    logging.info("Loading features from {}".format(features_test_path))
    features_test = joblib.load(features_test_path)
    labels_test = np.squeeze(features_test[:, 1].toarray())
    X_test = features_test[:, 2:]

    # Predict
    logging.info("Predicting")
    y_probs = model.predict_proba(X_test)
    y_preds = np.argmax(y_probs, axis=1)

    # Evaluate
    logging.info("Evaluating")
    prc_json_path = config["PLOTS"]["PRC_FILE"]
    roc_json_path = config["PLOTS"]["ROC_FILE"]
    metrics_json_path = config["METRICS"]["METRICS_FILE"]

    # precision recall curve
    precision, recall, thresholds = precision_recall_curve(labels_test, y_probs[:, 1])
    prc_dict = {
        "PRC": [
            {"precision": p, "recall": r, "threshold": t}
            for p, r, t in zip(precision, recall, thresholds)
        ]
    }

    save_json(prc_json_path, prc_dict)

    # roc curve
    fpr, tpr, thresholds = roc_curve(labels_test, y_probs[:, 1])
    roc_dict = {"ROC": [{"fpr": f, "tpr": t, "threshold": t} for f, t in zip(fpr, tpr)]}

    save_json(roc_json_path, roc_dict)

    # average precision score
    average_precision = average_precision_score(labels_test, y_probs[:, 1])
    # roc auc score
    roc_auc = roc_auc_score(labels_test, y_probs[:, 1])

    # metrics
    metrics_dict = {
        "roc_auc": roc_auc,
        "average_precision": average_precision,
    }
    save_json(metrics_json_path, metrics_dict)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
