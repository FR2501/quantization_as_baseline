import pathlib

import numpy as np
import os
import shutil

from utils import replace_path_component
from scipy.stats import zscore

DATASETS_ROOT = ""

def clean_datasets(collection_name, cleaning_function):
    raw_dataset_root = os.path.join(DATASETS_ROOT, "raw")

    print(collection_name)
    raw_root = os.path.join(raw_dataset_root, collection_name)
    cleaned_root = replace_path_component(raw_root, "raw", "cleaned")
    normalized_root = replace_path_component(raw_root, "raw", "normalized")

    os.makedirs(cleaned_root, exist_ok=True)
    os.makedirs(normalized_root, exist_ok=True)

    for dataset in sorted(os.listdir(raw_root)):
        print(dataset)
        dataset_name, _ = os.path.splitext(dataset)
        print(f"Cleaning and normalizing {dataset_name} ...")

        raw_dataset_path = os.path.join(raw_root, dataset)
        cleaned_dataset_path = os.path.join(cleaned_root, f"{dataset_name}.npy")
        normalized_dataset_path = os.path.join(normalized_root, f"{dataset_name}.npy")

        if not os.path.exists(cleaned_dataset_path) or not os.path.exists(normalized_dataset_path):
            if os.path.exists(cleaned_dataset_path):
                os.remove(cleaned_dataset_path)

            if os.path.exists(normalized_dataset_path):
                os.remove(normalized_dataset_path)

            cleaned_data = cleaning_function(raw_dataset_path)
            np.save(cleaned_dataset_path, cleaned_data.reshape(-1))

            if cleaned_data.ndim == 1:
                cleaned_data = cleaned_data.reshape(1, -1)

            np.save(normalized_dataset_path, zscore(cleaned_data, axis=1).astype(cleaned_data.dtype).reshape(-1)) # type: ignore
        else:
            print("\tSkipped, exists already.")


## LFZip
def _lfzip_cleaning_function(raw_dataset_path):
    data = np.load(raw_dataset_path)
    return data

# UCR TSCA
def _ucr_tsca_cleaning_function(raw_dataset_path):
    dataset_name = pathlib.Path(raw_dataset_path).parts[-1]

    data_test = np.genfromtxt(os.path.join(raw_dataset_path, f"{dataset_name}_TEST.tsv"), dtype=np.float32)[:,1:]
    data_train = np.genfromtxt(os.path.join(raw_dataset_path, f"{dataset_name}_TRAIN.tsv"), dtype=np.float32)[:,1:]
    data = np.concatenate([data_test, data_train])

    return data

# Synthetic
def _synth_cleaning_function(raw_dataset_path):
    return np.load(raw_dataset_path)


def prepare_datasets(datasets_root):
    global DATASETS_ROOT
    DATASETS_ROOT = datasets_root

    clean_datasets("lfzip", _lfzip_cleaning_function)
    clean_datasets("ucr_tsca", _ucr_tsca_cleaning_function)
    clean_datasets("synthetic_datasets", _synth_cleaning_function)