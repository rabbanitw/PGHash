# PGHash
On-device training of large networks via Locality-Sensitive Hashing (LSH) and Federated Learning (FL).

## Datasets
Extreme classification datasets, overview, and results can be found here: http://manikvarma.org/downloads/XC/XMLRepository.html.
Furthermore, to run our code, an extreme dataset of your choice must be downloaded (Amazon670K, for example) here: http://manikvarma.org/downloads/XC/XMLRepository.html
1. You will want the dataset with BoW features.
2. To parse out the training/test files you will also need pyxclib: https://github.com/kunaldahiya/pyxclib

## Code Dependencies

Our code was constructed and tested using the following Python packages:
1. tensorflow 2.10.0
2. numpy 1.23.4
3. mpi4py 3.1.4
4. sklearn 0.0
5. pyxclib (as detailed above)

Furthermore, we utilize MPI with Open MPI 4.1.4 (slightly older versions should work as well).

## Running the Code

Under construction... Examples & tutorials will be provided soon.
