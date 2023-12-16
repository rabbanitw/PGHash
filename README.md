# PGHash
On-device training of large networks via Locality-Sensitive Hashing (LSH) and Federated Learning (FL).

## Datasets
Extreme classification datasets, overview, and results can be found here: http://manikvarma.org/downloads/XC/XMLRepository.html.
Furthermore, to run our code and reproduce our experiments, Delicious-200K, Amazon-670K, and Wiki-325K must be downloaded.
Some notes about this process:
1. You will want the dataset with BoW features
2. To parse out the training/test files you will also need pyxclib: https://github.com/kunaldahiya/pyxclib
3. Store the data in the data folder under the corresponding dataset name, naming files train.txt and test.txt

## Code Dependencies

Our code was constructed and tested using the following Python packages:
1. tensorflow 2.10.0
2. numpy 1.23.4
3. mpi4py 3.1.4
4. pyxclib (which requires the following packages):
   1. sklearn 0.0 (downloading scikit-learn works)
   2. Cython

There are instructions at https://github.com/kunaldahiya/pyxclib for how to download pyxclib (this can be a slightly buggy process).
Furthermore, we utilize MPI with Open MPI 4.1.4 (slightly older versions should work as well).

## Running the Code

To run our code, first test for a single process:
```
mpirun -np 1 python run_pg.py --hash_type pghash --dataset Delicious200K --name test-run-single-worker
```

If this works, feel free to run the scripts that are present in the codebase!

## Citation

```
@inproceedings{
    rabbani2023pghash,
    title={Large-Scale Distributed Learning via Private On-Device LSH},
    author={Tahseen Rabbani and Marco Bornstein and Furong Huang},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=dpdbbN7AKr},
}
```
