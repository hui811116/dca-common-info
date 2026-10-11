# dca-common-info
Difference of Convex Algorithm for Discrete Wyner Common Information

Accepted on IEEE Transactions on Information Theory ( Volume: 71, Issue: 3, March 2025)


[Paper](https://ieeexplore.ieee.org/abstract/document/10848133)
##

## Citation
Please cite the following if you find it useful
```
@ARTICLE{wynerDCA,
  author={Huang, Teng-Hui and El Gamal, Hesham},
  journal={IEEE Transactions on Information Theory}, 
  title={Efficient Solvers for Wyner Common Information With Application to Multi-Modal Clustering}, 
  year={2025},
  volume={71},
  number={3},
  pages={2054-2074},
  keywords={Optimization;Vectors;Indexes;Convergence;Mutual information;Feature extraction;Clustering algorithms;Unsupervised learning;Inference algorithms;Entropy;Wyner common information;algorithms;optimization;variational inference;conditional independence;multi-modal learning;clustering;unsupervised},
  doi={10.1109/TIT.2025.3532280}}

```

## Author


Teng-Hui Huang


School of Electrical and Computer Engineering


University of Sydney


tenghui.huang@sydneyDOTeduDOTau

## Installation

Install the package from the repository root:

```sh
python -m pip install -e .
```

The package source is under `src/dcacinfo`, organized into `dataset`,
`metrics`, `solver`, and `utils` modules. Use `-e` for an editable
installation during development; omit it for a regular installation.

## Examples

The scripts in `examples/scripts` are command-line entry points and should be
run from the repository root after installation. Each accepts `--help` to
display its available options. For example, use a small run to check the
multiview solver:

```sh
python examples/scripts/main_nview.py wynerdca --nrun 1 --numbeta 1 --maxiter 10
```

The scripts provide different workflows:

- `main_nview.py` runs a synthetic multiview experiment and writes tabular
  results (CSV) and a configuration file.
- `main_threshold_nview.py` runs the synthetic multiview experiment over a
  range of latent cardinalities and writes tabular results (CSV) and a
  configuration file.
- `main_clustering.py` runs a clustering experiment, evaluates predicted
  labels, and saves the detailed results and configuration as pickle files.
- `read_result.py <results-directory>` reads clustering result pickle files
  and prints aggregated records; it also writes a MATLAB `.mat` file in the
  current working directory.

For example, run a small clustering experiment and aggregate its results:

```sh
python examples/scripts/main_clustering.py wynerdca --nrun 1 --numbeta 1 --maxiter 10 --output_dir results
python examples/scripts/read_result.py results
```

The `method` positional argument selects the algorithm; see each script's
`--help` output for supported methods and options such as `--ny`, `--nview`,
and `--output_dir`. Output paths are relative to the current working
directory unless an absolute path is provided.

## Tests

Install the test extra and run the basic tests with:

```sh
python -m pip install -e ".[test]"
python -m pytest
```
