# Point-Pattern Synthesis using Gabor and Random Filters

This repository contains the code for our paper:

> [Point-Pattern Synthesis using Gabor and Random Filters](https://xchhuang.github.io/pps_gabor_random/paper.pdf)
>
> Xingchang Huang, Pooran Memari, Hans-Peter Seidel, Gurprit Singh
> 
> Computer Graphics Forum (Proceedings of EGSR), 2022

![teaser](teaser.png)

For more details, please refer to our [project page](https://xchhuang.github.io/pps_gabor_random/index.html).

### Updates:
* 31 August 2022: updated project page
* 10 July 2022: added installation guide with CPU
* 27 June 2022: code released

### Prerequisites
* Python 3.7.9
* Pytorch 1.6.0
* matplotlib
* scipy 
* tqdm
* scikit-learn

##### Installation with GPU (tested on Windows 10 with an NVIDIA GPU)
```
conda create -n pps python=3.7
conda activate pps
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install matplotlib scipy tqdm
pip install -U scikit-learn
```

##### Installation with CPU (tested on MacOS, but much slower)
```
conda create -n pps_cpu python=3.7
conda activate pps_cpu
conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch
pip install matplotlib scipy tqdm
pip install -U scikit-learn
```


### Structure
* `test_data/init` : initialized poisson disk distributions for different patterns.
* `test_data/testset_point` : exemplar single-class point patterns
* `test_data/testset_disk` : exemplar disk patterns
* `test_data/testset_multiattributes` : exemplar multi-attribute patterns
* `src` : code

### Run

You can simply run a demo by: 
```
cd src
python main.py --logs=run --kernel_sigma1=1.0 --kernel_sigma2=2.6 --test_data=../test_data/testset_point --scene_name=lines
```

The `results` folder will be automatically created and the outputs will be saved in `run` folder. Please find more commands in `src/scripts/run.sh`. `kernel_sigma1, kernel_sigma2` are two hyper-parameters `c1, c2` explained in the paper.

### Results
Note that the generated results might be close to the ones presented in the paper but not exactly the same, due to the differences between machines.

### Citation
If you find this code useful please consider citing:

```
@article {huang22point,
    journal = {Computer Graphics Forum},
    title = {{Point-Pattern Synthesis using Gabor and Random Filters}},
    author = {Huang, Xingchang and Memari, Pooran and Seidel, Hans-Peter and Singh, Gurprit},
    year = {2022},
    publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
    ISSN = {1467-8659},
    DOI = {10.1111/cgf.14596}
}
```

### Acknowledgement
This work builds upon [Point-Synthesis](https://github.com/phtu-cs/Point-Synthesis) and [DiffCompositing](https://github.com/preddy5/DiffCompositing). We thank the authors for releasing their code.

