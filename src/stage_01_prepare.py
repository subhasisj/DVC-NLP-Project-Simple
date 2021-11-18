import argparse
import os
import shutil
from tqdm import tqdm
import logging
import os
from src.utils.common import read_yaml, create_directories
import random


STAGE = "One - Prepare" ## <<< change stage name 

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

    source_data = config['source_data']
    source_data_file = os.path.join(source_data["source_data_dir"],source_data["source_data_file"])

    # get params
    split_ratio = params["prepare"]['split_ratio']
    random_seed = params["prepare"]['random_seed']

    random.seed(random_seed)

    artifacts = config['artifacts']
    artifacts_dir = artifacts['ARTIFACTS_DIR']
    processed_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"],artifacts['PROCESSED_DATA'])
    create_directories([processed_data_dir_path])


    logging.info("Data split and written to disk")

    


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