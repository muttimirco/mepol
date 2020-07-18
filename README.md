# MEPOL
This repository contains the implementation of the **MEPOL** algorithm, presented in [A Policy Gradient Method for Task-Agnostic Exploration](https://arxiv.org/pdf/2007.04640.pdf).

## Installation
In order to use this codebase you need to work with a Python version >= 3.6. Moreover, you need to have a working setup of Mujoco with a valid Mujco license. To setup Mujoco, have a look [here](http://www.mujoco.org/). To use MEPOL, just clone this repository and install the required libraries:
```bash
git clone https://github.com/muttimirco/mepol.git && \
cd mepol/ && \
python -m pip install -r requirements.txt
```

## Usage
Before launching any script, add to the PYTHONPATH the root folder (mepol/):
```bash
export PYTHONPATH=$(pwd)
```

### Task-Agnostic Exploration Learning
To reproduce the maximum entropy experiments in the paper, run:
```bash
./scripts/tae/[mountain_car.sh | grid_world.sh | ant.sh | humanoid.sh | hand_reach.sh | higher_lvl_ant.sh | higher_lvl_humanoid.sh]
```
It should be straightforward to run MEPOL on your custom gym-like environments. For this purpose, you can have a look at the [main training script](src/experiments/mepol.py).

### Results visualization
Once launched, MEPOL will log statistics in the [results](results) folder. You can visualize everything by launching tensorboard targeting that directory:
```bash
tensorboard --logdir=./results --port 8080
```
and visiting the board at [http://localhost:8080](results).

### Goal-Based Reinforcement Learning
**Under construction**

## Citing
To cite the MEPOL paper:
```
@misc{mutti2020policy,
    title={A Policy Gradient Method for Task-Agnostic Exploration},
    author={Mirco Mutti and Lorenzo Pratissoli and Marcello Restelli},
    year={2020},
    eprint={2007.04640},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
