# Quantization as baseline

This repository accompanies our paper "The undervalued role of quantization as a baseline in lossy compression of univariate floating-point time series".

## Prerequisites

All Python dependencies are listed in the *requirements.txt*, install those as usual. We use Python 3.14.3, other versions might work as well.

To fully reproduce all experiments we report, the following programs must be available (either via PATH or by placing the binaries into this directory):
- BSC [https://github.com/IlyaGrebnov/libbsc](https://github.com/IlyaGrebnov/libbsc)
- Gzip [https://www.gzip.org/](https://www.gzip.org/)
- Zstd [https://github.com/facebook/zstd](https://github.com/facebook/zstd)

The datasets should be placed into *./datasets/ucr_tsca* and *./datasets/lfzip*, respectively, and can be obtained from
- [https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/) and
- [https://github.com/shubhamchandak94/LFZip/tree/master/data/evaluation_datasets](https://github.com/shubhamchandak94/LFZip/tree/master/data/evaluation_datasets).

If you want to include the our synthetic datasets as well, run *./synth_datasets.py*.

Executing **all** experiments takes several hours. If you place only some datasets into their respective folders, total runtime will decrease accordingly.

## Running the experiments

Simply run *generate_plots.py*, it will automatically convert the datasets to univariate 32-bit floating-point arrays, perform the required experiments, and generate all plots.

## Support/Sharing of intermediate results
If you can do so anonymously, please feel free to reach out ([fabian.richter@kit.edu](mailto:fabian.richter@kit.edu)) for technical support/to obtain intermediate results.
