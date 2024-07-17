# Offline Reinforcement Learning for Railway Optimal Maintenance and Comparison with Online Reinforcement Learning Solutions
Code for the research project "Offline Reinforcement Learning for Railway Optimal Maintenance and Comparison with Online Reinforcement Learning Solutions" by David Streuli. 

## Abstract
This paper explores the application of offline reinforcement learning (RL) for
optimising railway maintenance planning. By utilising historical data, we develop
decision-making policies without further environmental interaction, addressing
the impracticality of direct experimentation. We compare the performance of
offline RL algorithms, such as Deep Q-Networks (DQN), Batch-Constrained Q-
learning (BCQ), and Conservative Q-Learning (CQL), against traditional online
RL methods. Our results demonstrate the potential of offline RL to enhance
maintenance decisions and highlight the importance of a balanced dataset for
training.

## Setup
For the setup please run the following commands.

```
conda create -n rllib python=3.8.13
conda activate rllib
pip install -r requirements.txt
```

## Environment
The environment is implemented in ``env.py``.

## Data Sampling
Running ``data_sampling.py`` generates a dataset of trajectories using a specified environment and reward matrix. It allows for customisable parameters such as randomness in action selection, number of trajectories, and random seed for reproducibility. The dataset is saved in a compressed pickle file.

### Usage
```
python data_sampling.py --seed <seed> --randomness <randomness> --num_trajectories <num_trajectories>
```

## Training of Offline RL Algorithms
The algorithms can be trained using ``training.py``. The models will be saved in ``d3rlpy_logs/``. It assumes the availability of the datasets in ``datasets/``. 

### Usage
```
python training.py --optimalities <optimalities> --seed <seed> --num_trajectories_list <num_trajectories> --algos <algos>
```

## Evaluating Algorithms
The algorithms can be evaluated using ``evaluation_env.py``, ``evaluation_fqe.py`` and ``evaluation_magic.py`` to evaluate the algorithms using the respective methods. The results will be saved in ``d3rlpy_logs/`` in the directory of the corresponding algorithm.

### Usage
```
python evaluation_env.py --optimalities <optimalities> --seed <seed> --num_episodes <num_episodes>

python evaluation_fqe.py --optimalities <optimalities> --seed <seed>

python evaluation_magic.py --optimalities <optimalities> --seed <seed>
```
