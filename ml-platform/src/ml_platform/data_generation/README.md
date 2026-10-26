# Data generation

The [generate_datasets.py](./generate_datasets.py) script takes a data configuration file and performs subsequent processing steps to preprocess data and upload it to the S3 storage though the lakeFS control plane.

See [test_data_config.json](../../../../example/test_data_config.json) for a simple example.

The functions that the processing can use are in [data_processing](../shared/data_processing). The functions use the previously returned object as the first argument to the next function. Along the first argument, keyword arguments can be provided (see example).

The generation script tries to obtain a git hash to generate a hash for the lakeFS branch. If the files are not part of a git repository, the hash will be generated based on pipeline functions, their keyworsd arguments, and the initial argument.