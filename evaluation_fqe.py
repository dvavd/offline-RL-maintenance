import d3rlpy
import numpy as np
import gzip
import pickle
import argparse
import csv
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description='Set parameters of models for the evaluation.')
    parser.add_argument('--optimalities', nargs='*', type=int, default=[0, 25, 50, 75, 100],
                        help='List of optimalities to use. Default is [0, 25, 50, 75, 100].')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility. Default is 1.')
    parser.add_argument('--num_trajectories_list', nargs='*', type=int,
                        default=[100, 1000, 10000, 50000, 100000],
                        help='List of trajectory counts. Default is [100, 1000, 10000, 50000, 100000].')
    parser.add_argument('--algos', nargs='*', type=str, default=['DQN', 'BCQ', 'CQL'],
                        help='List of algorithms to use. Default is ["DQN", "BCQ", "CQL"].')
    return parser.parse_args()

def evaluate_policy(policy):

    # setup FQE algorithm
    config = d3rlpy.ope.FQEConfig(learning_rate = 1e-3, n_critics=2, target_update_interval=10)
    fqe = d3rlpy.ope.DiscreteFQE(algo=policy, config=config)

    # start FQE training
    fqe.fit(
        dataset,
        n_steps=10000,
        n_steps_per_epoch=1000,
        experiment_name = f'FQE_algo_{algo}_nr_traj_{num_trajectories}_opt_{optimality}_seed_{seed}_model_{step}',
        with_timestamp=False,
        save_interval=20
    )
    # predict for initial state
    initial_state = np.array([0.31492019, 0.40100431, 0.19309788, 0.09097762]).reshape(1, -1)
    initial_action = policy.predict(initial_state)
    value = fqe.predict_value(initial_state, initial_action)
    print(f"Predicted value for initial state: {value}")
    return value

args = parse_arguments()
seed = args.seed
np.random.seed(seed)

for optimality in args.optimalities:
    gz_path = f'datasets/fractal_env_dataset_opt_{optimality}_nr_tra_100000_seed_{seed}.pkl.gz'

    with gzip.open(gz_path, 'rb') as f:
        trajectories = pickle.load(f)

    for num_trajectories in args.num_trajectories_list:
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
        
        for algo in args.algos:
            dir = f'd3rlpy_logs/{algo}_nr_traj_{num_trajectories}_opt_{optimality}_seed_{seed}'
            results_path = os.path.join(dir, 'evaluation_fqe.csv')
            with open(results_path, 'w', newline='') as csvfile:
                result_writer = csv.writer(csvfile)
                for index, step in enumerate([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000], 1):
                    policy = d3rlpy.load_learnable(os.path.join(dir, f'model_{step}.d3'))
                    value = evaluate_policy(policy)
                    result_writer.writerow([index, step, value])
                   



