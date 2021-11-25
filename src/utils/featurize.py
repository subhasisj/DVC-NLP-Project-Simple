import os
import scipy.sparse as sp
import numpy as np
import joblib
import logging

def save_features_in_matrix(data, matrix,features_path):
    # save the features in CSR Matrix format
    id_matrix = sp.csr_matrix(data.id.astype(np.int64)).T
    label_matrix = sp.csr_matrix(data.label.astype(np.int64)).T

    result = sp.hstack([id_matrix, label_matrix, matrix], format='csr')
    logging.info(f"Saving features in matrix format in {features_path}")
    logging.info(f"Features in matrix shape {result.shape} and data type {result.dtype}")
    joblib.dump(result, features_path)
