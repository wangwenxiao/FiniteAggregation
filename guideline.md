# Example scripts

## assigning samples to training sets of different base learners
```
python3 FiniteAggregation_data_norm_hash.py --dataset=cifar --k=50 --d=3
```
Here `--dataset` can be `mnist`, `cifar` and `gtsrb`, which are benchmarks evaluated in our paper; `--k` and `--d` corresponds to the hyper-parameters for our Finite Aggregation. For details, please refer to Section 3 of our [Finite Aggregation paper](https://proceedings.mlr.press/v162/wang22m.html).

## training the base learners
```
python3 FiniteAggregation_train_cifar_nin_baseline.py --k=50 --d=3 --start=0 --range=150
```
Here `--k` and `--d` are the same as above, and a total of $k\cdot d$ base learners will be trained independently. `--start` and `--range` specify which base learners are trained with this script: For instance, when one use `--k=50` and `--d=3`, one can use `--start=0` and `--range=150` to train all base learners sequentially, or one can use two separate runs with repsectively `--start=0` and `--start=75` (both with `--range=75`) to train in parallel the first 75 and the last 75 base learners.  
To train on MNIST and GTSRB, run `FiniteAggregation_train_mnist_nin_baseline.py` and `FiniteAggregation_train_gtsrb_nin_baseline.py` instead.  
To use augmented learners as showcased in our [Lethal Dose Conjecture paper](https://openreview.net/forum?id=PYnSpt3jAz), run `FiniteAggregation_train_cifar_nin_autoaug_cifar.py`, `FiniteAggregation_train_cifar_nin_autoaug_imagenet.py` and `FiniteAggregation_train_gtsrb_nin_autoaug_svhn.py`.  
To use nearest neighbor as base learners, run `FiniteAggregation_train_cifar_nearest_neighbor.py`.


## collecting predictions of base learners on test sets
```
python3 FiniteAggregation_evaluate_cifar_nin_baseline.py --models=cifar_nin_baseline_FiniteAggregation_k50_d3
```
For MNIST and GTSRB, run `FiniteAggregation_evaluate_mnist_nin_baseline.py` and `FiniteAggregation_evaluate_gtsrb_nin_baseline.py` instead.

## computing the certified radius using the collected predictions
```
python3 FiniteAggregation_cerfity.py --evaluations=cifar_nin_baseline_FiniteAggregation_k50_d5.pth --k=50 --d=3 --num_classes=10
```
Here `--num_classes` is the size of the label set on the evalauted dataset (i.e. `--num_classes=10` for MNIST and CIFAR-10 and `--num_classes=43` for GTSRB).  
When using nearest neighbor as base learners, run `FiniteAggregation_certify_with_outlier.py` instead.