from dataclasses import dataclass
import os
import pathlib
import pickle
import numpy as np

from compressors import GenericPQECompressor
from stats import stat_report

def replace_path_component(old_path, old_component, new_component):
    # Source - https://stackoverflow.com/a/27258962
    # Posted by abarnert, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-03-20, License - CC BY-SA 4.0

    path = pathlib.Path(old_path)
    index = path.parts.index(old_component)

    new_path = pathlib.Path().joinpath(*path.parts[:index]).joinpath(new_component).joinpath(*path.parts[index+1:])

    return new_path


@dataclass
class ExperimentResult:
    original_size: int
    compressed_size: int
    measured_max_error: float
    delta_entropy: float

    def compression_ratio(self):
        return self.original_size / self.compressed_size


def run_experiment(compressor, dataset_path):
    dataset_name, _ = os.path.splitext(pathlib.Path(dataset_path).parts[-1])

    original_size = os.path.getsize(dataset_path)

    result_path = replace_path_component(dataset_path, "quantization_datasets", "quantization_results")
    result_path = pathlib.Path().joinpath(*result_path.parts[:-1]).joinpath(f"{dataset_name}_{compressor.get_unique_identifier()}")
    recon_path = str(result_path) + "_recon.npy"
    deltas_path = pathlib.Path().joinpath(*result_path.parts[:-1]).joinpath(
        f"{dataset_name}_{compressor.get_predictor().get_unique_identifier()}_{compressor.get_error_bound()}_deltas.npy")

    stats_path = str(result_path) + "_stats.pkl"

    if not os.path.exists(result_path) or not os.path.exists(deltas_path):
        if isinstance(compressor, GenericPQECompressor):
            compressor.set_prediction_output_path(deltas_path)

        compressor.compress(dataset_path, result_path)

        if os.path.exists(recon_path):
            os.remove(recon_path)
        if os.path.exists(stats_path):
            os.remove(stats_path)

    if not os.path.exists(result_path):
        return None
    
    if not os.path.exists(recon_path):
        compressor.decompress(result_path, recon_path)

        if os.path.exists(stats_path):
            os.remove(stats_path)

    if not os.path.exists(recon_path):
        return None
    
    if not os.path.exists(stats_path):
        with open(stats_path, "wb+") as f:
            pickle.dump(stat_report(np.load(deltas_path)), f)
    else:
        retries = 0
        while True:
            try:
                f = open(stats_path, "rb+")
                report = pickle.load(f)
                stat_report(np.load(deltas_path), f, report)
                f.close()

                with open(stats_path, "wb+") as f:
                    pickle.dump(report, f)
                    break
            except EOFError as e:
                if retries > 2:
                    raise RuntimeError(f"pkl file {stats_path} is broken")
                else:
                    os.remove(stats_path)
                    retries += 1

    if not os.path.exists(stats_path):
        return None
    
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    compressed_size = os.path.getsize(result_path)
    measured_max_error = np.max(np.abs(np.load(dataset_path) - np.load(recon_path)))

    return ExperimentResult(original_size, compressed_size, measured_max_error, stats)


def process_dataset(dataset, compressor, cleaned_datasets_root):
    full_dataset_path = os.path.join(cleaned_datasets_root, dataset)
    
    try:
        result = run_experiment(compressor, full_dataset_path)
        if result:
            data = np.load(full_dataset_path)
            full_result = (
                compressor.get_unique_identifier(),
                dataset,
                np.max(np.abs(data)),
                np.std(data),
                compressor.get_error_bound(),
                result.original_size,
                result.compressed_size,
                result.compression_ratio(),
                result.delta_entropy,
            )
            if result.measured_max_error > 1.1 * compressor.get_error_bound():
                print(f"Numerical problems on {dataset} with {compressor}, error bound violated by "
                    f"{result.measured_max_error / compressor.get_error_bound()}")
                
            return [full_result]
    except Exception as e:
        print(f"Error on {dataset} with {compressor}: {e}")
        raise e
    


