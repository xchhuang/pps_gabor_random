# Point-Pattern Synthesis using Gabor and Random Filters

This repository contains the code for our paper:

> [Point-Pattern Synthesis using Gabor and Random Filters](https://xchhuang.github.io/pps_gabor_random/paper.pdf)
>
> Xingchang Huang, Pooran Memari, Hans-Peter Seidel, Gurprit Singh
> 
> Computer Graphics Forum (Proceedings of EGSR), 2022

For more details, please refer to our [project page](https://xchhuang.github.io/pps_gabor_random/index.html).

### Updates:
* 27 June 2022: code released

### Prerequisites
* Python 3.7.9
* Pytorch 1.6.0
* CUDA 10.1

You can follow the installation in Anaconda (tested in Windows 10):
```
conda create -n pps python=3.7
conda activate pps
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install matplotlib scipy tqdm
pip install -U scikit-learn
```

### Structure
* `test_data/init` : initialized poisson disk distributions for different patterns.
* `test_data/testset_point` : exemplar single-class point patterns
* `src` : code

### Run

You can simply run a demo by: 
```
cd src
python main.py --logs=run --kernel_sigma1=1.0 --kernel_sigma2=2.6 --test_data=../test_data/testset_point --scene_name=lines
```

The `results` folder will be automatically created and the outputs will be saved in `run` folder. Please find more commands in `src/scripts/run.sh`. `kernel_sigma1, kernel_sigma2` are two hyper-parameters `c1, c2` explained in the paper.

### Citation
