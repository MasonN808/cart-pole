import json
import pickle

from cartpole_env import CartPoleEnv
import numpy as np
import math
import matplotlib.pyplot as plt

class Parameters():
    def __init__(self, theta, nPerturbations, sigma, n_episodes, alpha, M, max_steps):
        self.theta = theta
        self.nPerturbations = nPerturbations
        self.sigma = sigma
        self.n_episodes = n_episodes
        self.alpha = alpha
        self.M = M
        self.max_steps = max_steps

    
def policy_search(theta: np.array, nPerturbations: int, sigma: float, n_episodes: int, alpha: float, M: int, max_steps: int):
    """
    Evolution Strategies
    """
    n = theta.size
    mean = np.zeros(n)  # set mean to 0
    idty = np.eye(n)    # identity matrix
    J_values = []
    for _ in range(0, max_steps):
        J_epsilon_prod = []
        J_sum = 0
        for _ in range(1, nPerturbations+1):
            epsilon = np.random.multivariate_normal(mean, idty)
            J = estimate_J(theta + sigma * epsilon, n_episodes, M)
            J_sum += J
            J_epsilon_prod.append(J * epsilon)
        J_average = J_sum / nPerturbations
        J_values.append(J_average)
        # Update theta wrt the objective funciton 
        theta = theta + alpha * sum(J_epsilon_prod)/(sigma * nPerturbations)
    return J_values

def estimate_J(theta: np.array, n_episodes: int, M: int):
    # Initialize environment
    env = CartPoleEnv()
    returns = []
    for _ in range(0, n_episodes):
        env.reset()
        G = 0
        done = False
        while not done:
            _, reward, done = env.step(compute_policy(theta, M, env))
            G += reward
        returns.append(G)

    return sum(returns) / len(returns)

def compute_policy(theta: np.array, M: int, env: CartPoleEnv) -> float: 
    features = [1]
    # Normalize state features
    for normalized_feature in normalize_features(env):
        # Use Fourier basis as feature function
        for m in range(1, M + 1):
            features.append(math.cos(m * math.pi * normalized_feature))
    features = np.array(features)
    threshold = np.dot(theta, features)

    if threshold <= 0:
        # Left
        return -10.0
    else:
        # Right
        return 10.0
        
def normalize_features(env: CartPoleEnv) -> list:
    x_normalized = normalize(env.obs["x"], -2.4, 2.4)
    w_normalized = normalize(env.obs["w"], -math.pi/15, math.pi/15)
    # The limits of these features were found via simulations and clipping was used if it goes beyond it
    vx_normalized = normalize(env.obs["vx"], -3.8, 3.8)
    vw_normalized = normalize(env.obs["vw"], -3.9, 3.9)
    return [x_normalized, vx_normalized, w_normalized, vw_normalized]

def normalize(x, x_min, x_max):
    # Clip if x is not in interval
    x = max(x_min, min(x, x_max))
    return (x - x_min) / (x_max - x_min)


def find_intervals():
    # Initialize the environment
    env = CartPoleEnv()

    num_episodes = 1000000
    cart_velocities = []
    pole_angular_velocities = []

    for _ in range(num_episodes):
        done = False
        env.reset()
        while not done:
            # Take a random action
            action = np.random.choice([-10, 10])
            obs, reward, done = env.step(action)
            
            vx = obs["vx"]
            vw = obs["vw"]
            
            cart_velocities.append(vx)
            pole_angular_velocities.append(vw)

    # Determine the empirical intervals
    min_vx, max_vx = min(cart_velocities), max(cart_velocities)
    min_vw, max_vw = min(pole_angular_velocities), max(pole_angular_velocities)

    print(f"Cart velocities (vx) range: [{min_vx}, {max_vx}]")
    print(f"Pole angular velocities (vw) range: [{min_vw}, {max_vw}]")

def test_random_policy():
    # Initialize the environment
    env = CartPoleEnv()
    num_episodes = 10

    G_values = []
    for _ in range(num_episodes):
        done = False
        G = 0
        env.reset()
        while not done:
            # Take a random action
            action = np.random.choice([0, 1])  # For gym's CartPole, actions are 0 or 1
            _, reward, done = env.step(action)
            G += reward
        G_values.append(G)
    print(f"Average G Value: {sum(G_values)/len(G_values)}")

def plot_line_graph(data_vector: list, std_dev: list, n_simulations: int, misc: str):
    """
    Plots a line graph given a data vector.

    Parameters:
    - data_vector: A list or array containing data points.
    """
    # Convert lists to numpy arrays for element-wise operations
    data_vector = np.array(data_vector)
    std_dev = np.array(std_dev)

    x = np.arange(len(data_vector))
    plt.plot(x, data_vector)
    # Filling between the mean + std_dev and mean - std_dev
    plt.fill_between(x, data_vector - std_dev, data_vector + std_dev, color='blue', alpha=0.2, label='Standard Deviation')
    plt.xlabel('Iteration')
    plt.ylabel('Average J Value')
    plt.title(f'{misc} performance of policy over {n_simulations} simulations')
    plt.grid(True)
    plt.show()


def random_tune(iterations: int):
    J_values_dict = {}
    best_avg_return = 0
    for i in range(0, iterations):
        print(f'iteration: {i}')
        # Hyperparamter tune via Guassian values
        nPerturbations = int(np.random.uniform(10, 15))
        sigma = np.random.uniform(.03, .05)
        n_episodes = int(np.random.uniform(7, 15))
        alpha = np.random.uniform(0, .0002)
        M = int(np.random.uniform(4, 6))
        max_steps = 300
        # Pull theta weights from standard normal distribution
        theta = np.random.uniform(-0.5, 0.5, M * 4 + 1)
        J_values = policy_search(theta, nPerturbations, sigma, n_episodes, alpha, M, max_steps)

        average_return = sum(J_values) / len(J_values)
        if average_return > best_avg_return:
            best_avg_return = average_return

        print(f"average return: {average_return}; best return: {best_avg_return}")
        J_values_dict[Parameters(theta, nPerturbations, sigma, n_episodes, alpha, M, max_steps)] = J_values

    return J_values_dict

def plot(n_simulations: int, theta: np.array, nPerturbations: int, sigma: float, n_episodes: int, alpha: float, M: int, max_steps: int):
    J_list_values = []
    for _ in range(0, n_simulations):
        J_value = policy_search(theta=theta, nPerturbations=nPerturbations, sigma=sigma, n_episodes=n_episodes, alpha=alpha, M=M, max_steps=max_steps)
        J_list_values.append(J_value)
        print(J_value)

    J_values = average_lists(J_list_values)
    J_std_values = std_dev_lists(J_list_values)
    # J_values = max_lists(J_values)
    plot_line_graph(J_values, J_std_values, n_simulations=n_simulations, misc="Average")

def std_dev_lists(list_of_lists):
    """Returns a list where each element is the average of the elements at the corresponding indices in the input lists."""
    return [np.std(values) for values in zip(*list_of_lists)]

def average_lists(list_of_lists):
    """Returns a list where each element is the average of the elements at the corresponding indices in the input lists."""
    return [sum(values)/len(values) for values in zip(*list_of_lists)]

def max_lists(list_of_lists):
    """Returns a list where each element is the average of the elements at the corresponding indices in the input lists."""
    return [max(values) for values in zip(*list_of_lists)]


if __name__ == "__main__":
    # test_random_policy()
    DUMP = False
    filename = 'data300steps.pkl'
    if DUMP:
        # DUMPING
        # Load the existing dictionary from the pickle file (or start with an empty dictionary if the file doesn't exist)
        try:
            with open(filename, 'rb') as fp:
                data = pickle.load(fp)
        except (FileNotFoundError, EOFError):
            data = {}

        # Your new data
        J_values_dict = random_tune(iterations=100)

        # Update the dictionary with the new data
        data.update(J_values_dict)

        # Save the updated dictionary back to the pickle file
        with open(filename, 'wb') as fp:
            pickle.dump(data, fp)

    # READING
    # Read the pickled dictionary
    with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)

    # Calculate the average for each list and sort by it
    sorted_data = dict(sorted(loaded_data.items(), key=lambda item: sum(item[1])/len(item[1]), reverse=True))
    top_5_items = list(sorted_data.items())[3:5]
    for key, value in top_5_items:
        print(f"Key: {key.theta, key.nPerturbations, key.sigma, key.n_episodes, key.alpha, key.M, key.max_steps}, Value: {value}")
        plot(n_simulations=5,theta=key.theta, nPerturbations=key.nPerturbations, sigma=key.sigma, n_episodes=key.n_episodes, alpha=key.alpha, M=key.M, max_steps=key.max_steps)


    plot_line_graph(data)