# Empirical Privacy Risk Analysis on Differentially-private Machine Learning
This repo includes implementation of an empirical measurement of privacy protection of differentially private machine learning (DPML) algorithms. DPML algorithms provide a theoretical lower-bound guarantee for data privacy. However, it is unknown if DPML has desired protection of data privacy in practice. We empirically measured the actual effects of DPML in protecting privacy by imposing privacy attacks (e.g. Membership Inference Attack). For details, see [project summary](./project_summary.pdf) 

## Set up
Install the requirements in `requirements.txt` in a conda environment. Note that we require tensorflow==1.14.0; for tensorflow 2.x, you may have to tune the code a little bit.

## Training
### Normal training
Run nodp training on [mnist_model](./utils.py#L364) with CIFAR10, use gpu 2, and load the randomly split train/test indices from `logs/split_idx-cifar10_0.txt`:

`python dptrain.py --dpsgd False --model mnist_model --dataset cifar10 --dataset_split_path logs/split_idx-cifar10_0.txt --gpu 2`
The model will be saved at `logs/nodp/mnist_model-cifar10/lr0.0005-epochs100-splitratio0.5/`
### DP training
Run dp training on [mnist_model](./utils.py#L364) with CIFAR100, use gpu 1, and load the randomly split train/test indices from `logs/split_idx-cifar100_0.txt`:

`python dptrain.py --dpsgd True --model mnist_model --dataset cifar100 --noise_multiplier 1.0 --dataset_split_path logs/split_idx-cifar100_0.txt --gpu 0`
The model will be saved at `logs/dp/mnist_model-cifar100/lr0.0005-noisem1.0-C1.0-cclip-nodecay-splitratio0.5/`

## Tensorflow MIA
Run TF Attack (**including Meta-attack**) against 2 trained models: normal mnist_model on cifar10, DP mnist_model on cifar10 with noise_multiplier 1.0. You can also attack 1 model or more models at one run: simply add the path to the models to be attacked after `python tfattack.py`
```
python tfattack.py logs/nodp/mnist_model-cifar10/lr0.0005-epochs100-splitratio0.5/04250933/epochs100/nodp-final logs/dp/mnist_model-cifar10/lr0.0005-noisem1.0-C1.0-cclip-nodecay-splitratio0.5/04250943/epochs100/eps6.205-delta1e-06 --model mnist_model --dataset cifar10 --dataset_split_path logs/split_idx-cifar10_0.txt --gpu 1
```
Attack results will be writen to `logs/attack/tf_results.txt`

**meta attack improvement**
- add more TF attackers or tune hyperparameters at [here](./privacy/tensorflow_privacy/privacy/membership_inference_attack/models.py)
- change meta attacker model at [here](./privacy/tensorflow_privacy/privacy/membership_inference_attack/membership_inference_attack.py#L409)
## Shadow model MIA
Run Shadwo model MIA against 2 trained models: normal mnist_model on cifar10, DP mnist_model on cifar10 with noise_multiplier 1.0. You can also attack 1 model or more models at one run: simply add the path to the models to be attacked after `python tfattack.py`

`python miaattack.py logs/nodp/mnist_model-cifar10/lr0.0005-epochs100-splitratio0.5/04250933/epochs100/nodp-final logs/dp/mnist_model-cifar10/lr0.0005-noisem1.0-C1.0-cclip-nodecay-splitratio0.5/04250943/epochs100/eps6.205-delta1e-06 --model mnist_model --dataset cifar10 --dataset_split_path logs/split_idx-cifar10_0.txt --gpu 2`

Attack results will be writen to `logs/attack/mia_results.txt`


## Code structure
```
.
 ├── dptrain.py: DP training/ normal training script, uses the Huawei privai library
 ├── tfattack.py: Tensorflow attack script, uses the [modified tensorflow privacy library](privacy/tensorflow_privacy/privacy), [original repo](https://github.com/tensorflow/privacy)
 ├── miaattack.py: Shadow Model MIA script, uses the [modified MIA library](mia/), [original repo](https://github.com/spring-epfl/mia)
 ├── utils.py: helper functions
 ├── draw.ipynb: draw barplot of the attack result
 ├── mia/mia: modified MIA library
    ├── estimators.py
    ...
 ├── privacy/tensorflow_privacy/privacy/membership_inference_attack
    ├── membership_inference_attack.py  # run tf attacks and meta attack
    ├── models.py: # define tf attackers
    ...
    └── data_structures.py # define tf attack result data classes
```
