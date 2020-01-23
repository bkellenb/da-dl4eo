# DA4DL4EO

## Overview

This repository contains code to train, adapt, and evaluate ResNet-based models on and to the following two datasets:

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
    conda install -c conda-forge tqdm pot
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
    python 2_adapt.py --dataset_source UCMerced --dataset_target WHU-RS19 --daMethod DeepCORAL --backbone ResNet-18
```


### Test

_Note:_ Models that have undergone domain adaptation will be saved in the sub-folder of the target dataset.


Example 1: test a ResNet-18, adapted from UC-Merced to WHU-RS19 using DeepCORAL, on WHU-RS19:

```console
    python 3_test.py --dataset_target WHU-RS19 --dataset_model WHU-RS19 --daMethod DeepCORAL --backbone resnet18
```

Example 2: test a ResNet-18, trained on UC-Merced without domain adaptation, on WHU-RS19. Also save the resulting statistics to LaTeX files:

```console
    python 3_test.py --dataset_target WHU-RS19 --dataset_model UCMerced --backbone resnet18 --saveResults 1
```