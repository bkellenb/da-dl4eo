# Deep Learning for Earth Observation: Chapter Domain Adaptation

Code base for reproducing the experiments in the book "[Deep Learning for the Earth Sciences: A Comprehensive Approach to Remote Sensing, Climate Science, and Geosciences](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119646181)", Chapter 7: "[Deep Domain Adaptation in Earth Observation](https://onlinelibrary.wiley.com/doi/10.1002/9781119646181.ch7)", Section 7.3.1: "Adapting the inner representation."


## Overview

This repository contains code to train, adapt, and evaluate [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)-based models on and to the following two datasets:

* [UC-Merced](http://weegee.vision.ucmerced.edu/datasets/landuse.html)
* [WHU-RS19](http://www.xinhua-fluid.com/people/yangwen/WHU-RS19.html)


The following domain adaptation methodologies are supported:
* [MMD](https://raw.githubusercontent.com/jindongwang/transferlearning/master/code/deep/DDC_DeepCoral/mmd.py)
* [DeepCORAL](https://raw.githubusercontent.com/jindongwang/transferlearning/master/code/deep/DDC_DeepCoral/Coral.py)
* [DeepJDOT](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bharath_Bhushan_Damodaran_DeepJDOT_Deep_Joint_ECCV_2018_paper.pdf)



## Installation

### Setup code base
The following instructions apply to [Conda](https://conda.io).

```console
    conda create -y -n da4dl4eo python=3.7
    conda activate da4dl4eo
    conda install -c anaconda scikit-learn
    conda install -c conda-forge tqdm matplotlib pot
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

### Download datasets

1. Create folders in the repository root: `mkdir -p datasets/UCMerced; mkdir -p datasets/WHU-RS19`
2. Download the UC-Merced dataset from [here](http://weegee.vision.ucmerced.edu/datasets/landuse.html) and extract it into the folder `datasets/UCMerced`
3. Repeat the same for the [WHU-RS19](http://www.xinhua-fluid.com/people/yangwen/WHU-RS19.html) dataset and extract it into the folder `datasets/WHU-RS19` (note: you might need a .rar extraction utility to do so).
4. Create train/val/test splits: `conda activate da4dl4eo; python datasets.py`




## Run

The code snippets below can be used to train, adapt, and test a model, respectively.

Make sure to `cd` to the repository root and `conda activate da4dl4eo` before running one of the commands below.


### Train

Example: train ResNet-18 on UC-Merced:

```console
    python 1_train.py --dataset UCMerced --backbone resnet18
```


### Perform domain adaptation

Example: adapt a ResNet-18 from UC-Merced to WHU-RS19 using DeepCORAL:

```console
    python 2_adapt.py --dataset_source UCMerced --dataset_target WHU-RS19 --daMethod DeepCORAL --backbone resnet18
```


### Test

_Note:_ Models that have undergone domain adaptation will be saved in the sub-folder of the target dataset.


Example 1: test a ResNet-18, adapted from UC-Merced to WHU-RS19 using DeepCORAL, on WHU-RS19:

```console
    python 3_test.py --dataset_target WHU-RS19 --dataset_model UCMerced --daMethod DeepCORAL --backbone resnet18
```

Example 2: test a ResNet-18, trained on UC-Merced without domain adaptation, on WHU-RS19. Also save the resulting statistics to LaTeX files:

```console
    python 3_test.py --dataset_target WHU-RS19 --dataset_model UCMerced --backbone resnet18 --saveResults 1
```


## Replicate book chapter results

To replicate the results, the following experiments need to be conducted:

```console

    # train base models
    python 1_train.py --dataset UCMerced
    python 1_train.py --dataset WHU-RS19
    python 1_train.py --dataset both


    # adapt
    python 2_adapt.py --dataset_source UCMerced --dataset_target WHU-RS19 --daMethod MMD
    python 2_adapt.py --dataset_source UCMerced --dataset_target WHU-RS19 --daMethod DeepCORAL
    python 2_adapt.py --dataset_source UCMerced --dataset_target WHU-RS19 --daMethod DeepJDOT

    python 2_adapt.py --dataset_source WHU-RS19 --dataset_target UCMerced --daMethod MMD
    python 2_adapt.py --dataset_source WHU-RS19 --dataset_target UCMerced --daMethod DeepCORAL
    python 2_adapt.py --dataset_source WHU-RS19 --dataset_target UCMerced --daMethod DeepJDOT


    # test (order: source only, MMD, DeepCORAL, DeepJDOT, target only; for both UCMerced and WHU-RS19 target datasets)
    python 3_test.py --dataset_target UCMerced --dataset_model WHU-RS19
    python 3_test.py --dataset_target UCMerced --dataset_model WHU-RS19 --daMethod MMD
    python 3_test.py --dataset_target UCMerced --dataset_model WHU-RS19 --daMethod DeepCORAL
    python 3_test.py --dataset_target UCMerced --dataset_model WHU-RS19 --daMethod DeepJDOT
    python 3_test.py --dataset_target UCMerced --dataset_model UCMerced
    python 3_test.py --dataset_target UCMerced --dataset_model all

    python 3_test.py --dataset_target WHU-RS19 --dataset_model UCMerced
    python 3_test.py --dataset_target WHU-RS19 --dataset_model UCMerced --daMethod MMD
    python 3_test.py --dataset_target WHU-RS19 --dataset_model UCMerced --daMethod DeepCORAL
    python 3_test.py --dataset_target WHU-RS19 --dataset_model UCMerced --daMethod DeepJDOT
    python 3_test.py --dataset_target WHU-RS19 --dataset_model WHU-RS19
    python 3_test.py --dataset_target WHU-RS19 --dataset_model all
```
