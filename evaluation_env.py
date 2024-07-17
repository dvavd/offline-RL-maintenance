import d3rlpy
from env import FractalEnv
from hmm_AR_k_Tstud import HMMStates, TruncatedNormalEmissionsAR_k
import numpy as np
import pickle
import math
import argparse
import csv
import os

trace_file = 'trace.pickle'
with open(trace_file, "rb") as fp:
    trace = pickle.load(fp)

reward_matrix = np.asarray([
    [0 - 100, 0 - 200, 0 - 1000, 0 - 8000],
    [-50 - 100, -50 - 200, -50 - 1000, -50 - 8000],
    [-2000 - 50 - 100, -2660 - 50 - 200, -3330 - 50 - 1000, -4000 - 50 - 8000]
])

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
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes to evaluate. Default is 100.')
    return parser.parse_args()

def evaluate_policy(policy, env, num_episodes=100):
    rewards = []
    lengths = []

    for episode in range(num_episodes):
        observation = env.reset()
        observation = observation.reshape(1, -1)
        done = False
        total_reward = 0
        length = 0

        while not done and length < 50:
            action = policy.predict(observation)[0]
            observation, reward, done, info = env.step(action)
            observation = observation.reshape(1, -1)
            total_reward += reward
            length += 1

        rewards.append(total_reward)
        lengths.append(length)

    return np.mean(rewards), np.var(rewards), math.sqrt(np.var(rewards) / len(rewards))

args = parse_arguments()

seed = args.seed

np.random.seed(seed)

env_config = {}
env = FractalEnv(trace=trace, reward_matrix=reward_matrix, env_config=env_config)

for optimality in args.optimalities:
    for num_trajectories in args.num_trajectories_list:
        for algo in args.algos:
            dir = f'd3rlpy_logs/{algo}_nr_traj_{num_trajectories}_opt_{optimality}_seed_{seed}'
            results_path = os.path.join(dir, 'evaluation_env.csv')
            with open(results_path, 'w', newline='') as csvfile:
                result_writer = csv.writer(csvfile)
                for index, step in enumerate([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000], 1):
                    policy = d3rlpy.load_learnable(os.path.join(dir, f'model_{step}.d3'))
                    avg_return, variance, sem = evaluate_policy(policy, env, args.num_episodes)
                    result_writer.writerow([index, step, avg_return, variance, sem])
env.close()


