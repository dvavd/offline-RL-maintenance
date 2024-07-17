import pickle
import gzip
import argparse
import numpy as np
from env import FractalEnv
from hmm_AR_k_Tstud import HMMStates, TruncatedNormalEmissionsAR_k

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Generate a dataset based on the specified parameters.')
parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
parser.add_argument('--randomness', type=float, default=0.0, help='Level of randomness in action selection')
parser.add_argument('--num_trajectories', type=int, default=100, help='Number of trajectories to generate')

args = parser.parse_args()

# set the seed for reproducibility
np.random.seed(args.seed)

# load the trace file
trace_file = 'trace.pickle'
with open(trace_file, "rb") as fp:
    trace = pickle.load(fp)

# define the reward matrix
reward_a_0, reward_a_R2, reward_a_A1 = 0, -50, -2000
reward_s_0, reward_s_1, reward_s_2, reward_s_3 = -100, -200, -1000, -8000

reward_matrix = np.asarray([
    [reward_a_0 + reward_s_0, reward_a_0 + reward_s_1, reward_a_0 + reward_s_2, reward_a_0 + reward_s_3],
    [reward_a_R2 + reward_s_0, reward_a_R2 + reward_s_1, reward_a_R2 + reward_s_2, reward_a_R2 + reward_s_3],
    [1*reward_a_A1 + reward_a_R2 + reward_s_0, 1.33*reward_a_A1 + reward_a_R2 + reward_s_1, 1.66*reward_a_A1 + reward_a_R2 + reward_s_2, 2*reward_a_A1 + reward_a_R2 + reward_s_3]
])

# time-dependent optimal policy considering only the mean parameters
actions_lookup = {
    0: [0] * 50,  
    1: [1] * 46 + [0] * 4,
    2: [2] * 42 + [1] * 7 + [0], 
    3: [2] * 45 + [1] * 4 + [0],
}

# Function to run a single trajectory
def run_trajectory(env, actions_lookup, randomness=0.0):
    trajectory = []
    done = False
    observation = env.reset()
    while not done and env.t < 50:
        current_state = env.state
        if np.random.rand() < randomness:
            # Choose a random action
            action = np.random.choice([0, 1, 2])
        else:
            # Choose the optimal action otherwise
            action = actions_lookup[current_state][env.t]
        next_observation, reward, done, info = env.step(action)
        trajectory.append((observation, reward, env.t, action, current_state))
        observation = next_observation
    return trajectory

# Main function to create the dataset
def create_dataset(num_trajectories, randomness):
    dataset = []
    env_config = {}
    for i in range(num_trajectories):
        env = FractalEnv(trace=trace, reward_matrix=reward_matrix, env_config=env_config)
        trajectory = run_trajectory(env, actions_lookup, randomness=randomness)
        dataset.append(trajectory)
        env.close()

        if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{num_trajectories} trajectories generated. {(i+1)/num_trajectories*100:.2f}% complete.")
        
    return dataset

def save_with_pickle_compression(data, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data, f)

num_trajectories = args.num_trajectories
randomness = args.randomness
optimality_percentage = 100 - (randomness * 100)
dataset = create_dataset(num_trajectories, randomness)
filename = f'datasets/fractal_env_dataset_opt_{optimality_percentage:.0f}_nr_tra_{num_trajectories}_seed_{args.seed}.pkl.gz'
save_with_pickle_compression(dataset, filename)

