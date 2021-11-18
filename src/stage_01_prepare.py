import argparse
import os
import shutil
from tqdm import tqdm
import logging
import os
from src.utils.common import read_yaml, create_directories
from src.utils.data_mgmt import process_posts
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

    # Create Processed Data Directory
    artifacts = config['artifacts']
    artifacts_dir = artifacts['ARTIFACTS_DIR']
    processed_data_dir_path = os.path.join(artifacts_dir,artifacts['PROCESSED_DATA'])
    create_directories([processed_data_dir_path])

    # Read Train and Test Data from Path
    train_data_path = os.path.join(processed_data_dir_path,artifacts['TRAIN_DATA'])
    test_data_path = os.path.join(processed_data_dir_path, artifacts['TEST_DATA'])

    # Read the source data file and create Train and Test Data
    target_tag = "<python>"
    encoding = "utf-8"
    with open(source_data_file, 'r',encoding=encoding) as f_in:
        with open(train_data_path, 'w',encoding=encoding) as f_train:
            with open(test_data_path, 'w',encoding=encoding) as f_test:
                process_posts(f_in, f_train, f_test,target_tag, split_ratio)


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