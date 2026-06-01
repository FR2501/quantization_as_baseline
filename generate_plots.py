from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import os
import pandas as pd

from compressors import GenericPQECompressor
import data_preparation
from entropy_coders import HuffmanCompressor, GZip, Zstd, BSC
from plot import make_plots
from predictors import ZeroPredictor, SimpleLorenzoPredictor, NLMSPredictor
from quantizers import LinearQuantizer
from utils import process_dataset

DATASETS_ROOT = "datasets"

data_preparation.prepare_datasets(DATASETS_ROOT)
normalized_datasets_root = os.path.join(DATASETS_ROOT, "normalized")

datasets_ucr = [f"ucr_tsca/{dataset}" for dataset in sorted(os.listdir(os.path.join(normalized_datasets_root, 'ucr_tsca')))]
datasets_lfzip = [f"lfzip/{dataset}" for dataset in sorted(os.listdir(os.path.join(normalized_datasets_root, 'lfzip')))]
datasets_synth = [f"synthetic_datasets/{dataset}" for dataset in sorted(os.listdir(os.path.join(normalized_datasets_root, 'synthetic_datasets')))]

experimental_settings = set()

## Figures 2, 3

quantizer_0_1 = LinearQuantizer(np.float32().dtype, np.int32().dtype, 0.1)
compressors_0_1_all = [GenericPQECompressor(predictor, entropy_coder, name=f"{predictor.get_unique_identifier()}_e0.1_{entropy_coder.get_unique_identifier()}") 
                       for predictor in [ZeroPredictor(quantizer_0_1), SimpleLorenzoPredictor(quantizer_0_1), NLMSPredictor(quantizer_0_1)] 
                       for entropy_coder in [HuffmanCompressor(), GZip(), Zstd(), BSC()]]

for dataset in datasets_ucr:
    for compressor in compressors_0_1_all:
        experimental_settings.add((dataset, compressor))

## Figures 4, 6, 7, 9
error_bounds = [1e-5, 1e-3, 1e-1]
for error_bound in error_bounds:
    quantizer = LinearQuantizer(np.float32().dtype, np.int32().dtype, error_bound)
    
    compressors = [GenericPQECompressor(predictor, BSC(), name=f"{predictor.get_unique_identifier()}_e{str(error_bound)}_{BSC().get_unique_identifier()}") 
                   for predictor in [ZeroPredictor(quantizer), SimpleLorenzoPredictor(quantizer), NLMSPredictor(quantizer)]]
    
    for dataset in datasets_ucr:
        for compressor in compressors:
            experimental_settings.add((dataset, compressor))
    for dataset in datasets_synth:
        for compressor in compressors:
            experimental_settings.add((dataset, compressor))
               
results = []
with ProcessPoolExecutor(max_workers=12) as executor:
    futures = {executor.submit(process_dataset, dataset, compressor, normalized_datasets_root): (dataset, compressor) for dataset, compressor in sorted(experimental_settings)}
    for future in as_completed(futures):
        results.extend(future.result())

result_df = pd.DataFrame(results, columns=["compressor", "dataset", "max_data_val", "stddev", "error_bound", "orig_size", "comp_size", "comp_ratio", "entropy"])
result_df["compressor_short"] = result_df["compressor"].str.split("_").str[0]
result_df["entropy_coder"] = result_df["compressor"].str.split("_").str[-1]
result_df["entropy_b"] = result_df["entropy"].apply(lambda x: x.get("entropy"))
result_df["cont_entr_1"] = result_df["entropy"].apply(lambda x: x.get("cont_entr_1"))
result_df["cont_entr_2"] = result_df["entropy"].apply(lambda x: x.get("cont_entr_2"))
result_df["cont_entr_4"] = result_df["entropy"].apply(lambda x: x.get("cont_entr_4"))
result_df["avg_rl"] = result_df["entropy"].apply(lambda x: x.get("avg_rl"))
result_df["int_entropy"] = result_df["entropy"].apply(lambda x: x.get("int_entropy"))
result_df["variance"] = result_df["entropy"].apply(lambda x: x.get("variance"))

make_plots(result_df, dataset_root=normalized_datasets_root)