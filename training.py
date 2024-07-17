import d3rlpy
from d3rlpy.metrics import EnvironmentEvaluator
from d3rlpy.metrics import TDErrorEvaluator
import numpy as np
import gzip
import pickle
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Set parameters for the experiment.')
    parser.add_argument('--optimalities', nargs='*', type=int, default=[0, 25, 50, 75, 100],
                        help='List of optimalities to use. Default is [0, 25, 50, 75, 100].')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility. Default is 1.')
    parser.add_argument('--num_trajectories_list', nargs='*', type=int,
                        default=[100, 1000, 10000, 50000, 100000],
                        help='List of trajectory counts. Default is [100, 1000, 10000, 50000, 100000].')
    parser.add_argument('--algos', nargs='*', type=str, default=['DQN', 'BCQ', 'CQL'],
                        help='List of algorithms to use. Default is ["DQN", "BCQ", "CQL"].')
    args = parser.parse_args()
    return args

args = parse_arguments()
optimalities = args.optimalities
seed = args.seed
num_trajectories_list = args.num_trajectories_list
algos = args.algos

# set the seed for reproducibility
np.random.seed(seed)

for optimality in optimalities:
    gz_path = f'datasets/fractal_env_dataset_opt_{optimality}_nr_tra_100000_seed_{seed}.pkl.gz'
    
    # Unzip and load the pickle file
    with gzip.open(gz_path, 'rb') as f:
        trajectories = pickle.load(f)

    for num_trajectories in num_trajectories_list:

        # Flatten the list of trajectories
        flattened_data = [record for trajectory in trajectories[:num_trajectories] for record in trajectory]

        # Separate the components
        observations = np.array([record[0] for record in flattened_data])
        rewards = np.array([record[1] for record in flattened_data])
        actions = np.array([record[3] for record in flattened_data]).reshape(-1, 1)  
        terminals = np.zeros(len(flattened_data)) # No terminal states in this environment
        timeouts = np.zeros(len(flattened_data), dtype=float)

        episode_length = 50
        for i in range(1, len(terminals)):
            if i % episode_length == 0:
                terminals[i-1] = 1  # Mark the end of each episode

        # Create the MDPDataset
        dataset = d3rlpy.dataset.MDPDataset(
            observations=observations,
            actions=actions, 
            rewards=rewards,
            terminals=terminals,
            timeouts=timeouts
        )
        
        if ('DQN' in algos):
            dqn_config = d3rlpy.algos.DQNConfig(
                n_critics=2
            )
            dqn = dqn_config.create(device=None)
            dqn.fit(
                dataset,
                experiment_name = f'DQN_nr_traj_{num_trajectories}_opt_{optimality}_seed_{seed}',
                with_timestamp = False,
                n_steps=100000,
                n_steps_per_epoch=10000,
            )
        
        if ('CQL' in algos):
            cql = d3rlpy.algos.DiscreteCQLConfig().create(device=None)
            cql.fit(
                dataset,
                experiment_name = f'CQL_nr_traj_{num_trajectories}_opt_{optimality}_seed_{seed}',
                with_timestamp = False,
                n_steps=100000,
                n_steps_per_epoch=10000,
            )

        if ('BCQ' in algos):
            bcq_config = d3rlpy.algos.DiscreteBCQConfig(
                n_critics=2,
                action_flexibility=0.6,
                beta=0.3,
                target_update_interval=5000,
            )
            bcq = bcq_config.create(device=None)
            bcq.fit(
                dataset=dataset,
                experiment_name = f'BCQ_nr_traj_{num_trajectories}_opt_{optimality}_seed_{seed}',
                with_timestamp = False,
                n_steps=100000,
                n_steps_per_epoch=10000,
            )


