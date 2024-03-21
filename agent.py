"""
Custom riscRL implementation on Tensorflow 2.0
Current Implementations:
1. Q-Learning
2. Q-Learning with Experience Replay
3. Q-Learning with Target Network
4. Q-Learning with Experience Replay and Target Network
5. Double Q-Learning
6. Double Q-Learning with Experience Replay
7. Double Q-Learning with Target Network
8. Double Q-Learning with Experience Replay and Target Network
9. DQN
10. DQN with Experience Replay
11. DQN with Target Network
12. DQN with Experience Replay and Target Network
13. Double DQN
14. Double DQN with Experience Replay
15. DDPG
16. MADDPG
"""

# Imports
import os
import time
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import keras.backend as K
from collections import deque
import matplotlib.pyplot as plt

# Set seed for reproducibility 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Control GPU growth
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# 1. Q-Learning
class QLearning:
    """
    Q-Learning
    """
    def __init__(self, env=None, episodes=int(), decay_style="sigmoid", alpha=0.1, gamma=0.9, epsilon=1, policy_path=None, test_episodes=100, testing=False, verbose = 0):
        """
        Initialize Q-Learning
        """
        # Initialize environment
        self.env = env
        # Initialize number of episodes to train on
        self.episodes = episodes
        # Initialize epsilon decay style
        self.decay_style = decay_style
        # Initialize decay rate for epsilon decay in exponential and sigmoid decay styles
        self.decay_rate = 10/self.episodes
        # Initialize learning rate
        self.alpha = alpha
        # Initialize discount factor
        self.gamma = gamma
        # Initialize exploration rate
        self.epsilon = epsilon
        # Initialize number of episodes to test the trained agent on
        self.test_episodes = test_episodes
        # Initialize the episode number to test the trained agent
        self.test_every = self.episodes/100
        # flag to decide to create folder or not
        self.testing = testing
        # Initialize verbosity
        self.verbose = verbose
        # Initialize the Q-Table
        if policy_path is not None:
            # Load Q-Table
            with open(policy_path, "rb") as f:
                self.Q = pickle.load(f)
        else:
            # Initialize Q-Table
            self.Q = {}
        # Data Log Pad
        self.pad = 25
        # env name
        self.env_name = self.env.name if hasattr(self.env, 'name') else 'custom env'
        curr_time = int(time.time())
        # create plots folder
        if not self.testing:
            self.plots_path = f"results/{self.env_name}/{self.__class__.__name__}/{self.decay_style}/{curr_time}/plots"
            if not os.path.exists(self.plots_path):
                os.makedirs(self.plots_path)
            # create q-tables folder
            self.policy_path = f"results/{self.env_name}/{self.__class__.__name__}/{self.decay_style}/{curr_time}/q_policies"
            if not os.path.exists(self.policy_path):
                os.makedirs(self.policy_path)
            # create logs folder
            self.logs_path = f"results/{self.env_name}/{self.__class__.__name__}/{self.decay_style}/{curr_time}/logs"
            if not os.path.exists(self.logs_path):
                os.makedirs(self.logs_path)
            # create episodes folder
            self.episodes_path = f"results/{self.env_name}/{self.__class__.__name__}/{self.decay_style}/{curr_time}/episodes_figures"
            if not os.path.exists(self.episodes_path):
                os.makedirs(self.episodes_path)
        
    def add_table_keys(self, q_table, state):
        """
        Add observation to Q-Table
        """
        # Check if observation is in Q-Table
        if state not in q_table.keys():
            # Initialize observation in Q-Table
            q_table[state] = {}
            # Initialize actions for observation in Q-Table
            for a in range(len(self.env.action_space)):
                q_table[state][a] = random.uniform(-1, 1)
        # Return Q-Table
        return q_table

    def get_q_action(self, observation):
        """
        Get action from Q-Table
        """
        state = tuple(observation)
        # Add observation to Q-Table if needed
        self.Q = self.add_table_keys(self.Q, state)
        # Get action from Q-Table
        action = max(self.Q[state], key=self.Q[state].get)
        # Return action
        return action
    
    def get_epsilon_greedy_action(self, observation):
        """
        Get epsilon-greedy action
        """
        # Get random number
        rand = random.uniform(0, 1)
        # Check if random number is less than epsilon
        if rand < self.epsilon:
            # Get random action
            action = random.choice(self.env.action_space)
        else:
            # Get action from Q-Table
            action = self.get_q_action(observation)
        # Return action
        return action
    
    def get_greedy_action(self, observation): 
        """
        Get greedy action (no exploration)
        """
        # Get action from Q-Table
        action = self.get_q_action(observation)
        # Return action
        return action
    
    def epsilon_decay(self, episode):
        """
        Decay epsilon
        """
        # Check if decay style is linear
        if self.decay_style == "linear":
            # Decay epsilon linearly
            self.epsilon = self.epsilon - (1/self.episodes)
        # Check if decay style is exponential
        elif self.decay_style == "exponential":
            # Decay epsilon exponentially
            self.epsilon = np.exp(-self.decay_rate*0.5*episode)
        # Check if decay style is sigmoid
        elif self.decay_style == "sigmoid":
            # Decay epsilon sigmoidally
            self.epsilon = 1/(1 + np.exp(self.decay_rate*(episode - self.episodes/2)))
        # Check if decay style is constant
        elif self.decay_style == "constant":
            # Decay epsilon constantly
            self.epsilon = self.epsilon
    
    def update_q_table(self, q, q_target, data):
        """
        Update Q-Table
        """
        for _, (observation, action, reward, new_observation, done) in enumerate(data):
            state = tuple(observation)
            new_state = tuple(new_observation)
            # Add new observation to Q-Tables if needed
            q = self.add_table_keys(q, state)
            q = self.add_table_keys(q, new_state)
            q_target = self.add_table_keys(q_target, state)
            q_target = self.add_table_keys(q_target, new_state)
            # Update Q-Table
            if not done:
                q[state][action] = q[state][action] + self.alpha*(reward + self.gamma*max(q_target[new_state].values()) - q[state][action])
            else:
                q[state][action] = reward
            
        # return updated Q-Table
        return q

    def log_data(self, episode_data):
        """
        Log data
        """
        print("Logging Data...")
        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(episode_data)
        # Pad column headings
        df.columns = [col.rjust(self.pad) for col in df.columns]
        # Save DataFrame to CSV file with padded headers
        df.to_csv(f"{self.logs_path}/{self.__class__.__name__}_training_data.csv", index=False, header=True)
        print("Done Logging Data")
    
    def generate_plots(self):
        """
        Generate plots
        """
        print("Generating Plots...")
        # Load training data
        df = pd.read_csv(f"{self.logs_path}/{self.__class__.__name__}_training_data.csv")
        # Remove leading/trailing whitespaces from column names
        df.columns = df.columns.str.strip()
        # Convert columns to numeric (in case they were saved as strings)
        df = df.apply(pd.to_numeric, errors='coerce')
        # Plot figures and save them
        first_column = df.columns[0]
        for _, column in enumerate(df.columns[1:], start=1):
            plt.figure(figsize=(10, 6))
            plt.plot(df[first_column], df[column])
            plt.title(f'{column} vs {first_column}')
            plt.xlabel(first_column)
            plt.ylabel(column)
            plt.tight_layout()
            # Save the plot to the plots folder
            plt.savefig(f'{self.plots_path}/{column}_plot.png')
            plt.close()  # Close the current figure to start a new one for the next iteration
        print("Done Generating Plots")
    
    def test(self):
        """
        Test agent
        """
        print("\nTesting Agent...")
        # Success Count
        success_count = 0
        for _ in range(1, self.test_episodes+1):
            # Reset Environment
            observation, done = self.env.reset()
            while not done:
                # take a step
                action = self.get_greedy_action(observation)
                new_observation, _, done, episode_status = self.env.step(action, observation)
                # update observation
                observation = new_observation
            # update success count
            if episode_status:
                success_count += 1
        print(f"Done Testing Agent with {success_count/self.test_episodes*100}% success rate")
        # return success rate
        return success_count/self.test_episodes
    
    def test_policy(self, path, type):
        """
        Test agent with a given policy
        """
        print("Testing Agent...")
        # Load Q-Table
        if type == "pickle":
            with open(path, "rb") as f:
                Q = pickle.load(f)
        elif type == "numpy":
            Q = np.load(path, allow_pickle=True).flat[0]
        # Success Count
        success_count = 0
        for _ in range(1, self.test_episodes+1):
            # Reset Environment
            observation, done = self.env.reset()
            while not done:
                # take a step
                action = max(Q[tuple(observation)], key=Q[tuple(observation)].get)
                new_observation, _, done, episode_status = self.env.step(action, observation)
                # update observation
                observation = new_observation
            # update success count
            if episode_status:
                success_count += 1
        print(f"Done Testing Agent with {success_count/self.test_episodes*100}% success rate")

    def train(self):
        """
        Train agent
        """
        # Initialize data collection
        episode_rewards = []
        episode_data = []

        # Initialize success rate
        prev_best_rate = 0
        prev_success_rate = 0

        for episode in tqdm(range(1, self.episodes+1), ascii=True, unit="episode"):
            # Reset Environment
            observation, done = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            # update epsilon
            self.epsilon_decay(episode)
            while not done:
                # take a step
                action = self.get_epsilon_greedy_action(observation)
                new_observation, reward, done, _ = self.env.step(action, observation)

                # update data collection
                episode_reward += reward
                episode_steps += 1

                # update main Q-Table and target Q-Table
                data = [[observation,
                         action,
                         reward,
                         new_observation,
                         done]]
                self.Q = self.update_q_table(self.Q, self.Q, data)
                # update observation
                observation = new_observation

            # update data collection
            episode_rewards.append(episode_reward)

            # update success rate every 100 episodes and save Q-Table
            if episode % self.test_every == 0:
                # save Q-Table
                with open(f"{self.policy_path}/q_table.pkl", "wb") as f:
                    pickle.dump(self.Q, f)
                success_rate = self.test()
                prev_success_rate = success_rate

                if success_rate >= prev_best_rate:
                    prev_best_rate = success_rate
                    # save best Q-Table
                    with open(f"{self.policy_path}/best_q_table.pkl", "wb") as f:
                        pickle.dump(self.Q, f)
            
            # log data
            episode_data.append({
                "episode number": str(episode).rjust(self.pad),
                "episode reward": str(episode_reward).rjust(self.pad),
                "average reward": str(np.mean(episode_rewards[-100:])).rjust(self.pad),
                "num of steps": str(episode_steps).rjust(self.pad),
                "epsilon": str(self.epsilon).rjust(self.pad),
                "test success_rate": str(prev_success_rate).rjust(self.pad),
                "best success_rate": str(prev_best_rate).rjust(self.pad)
            })
        
        self.log_data(episode_data)
        self.generate_plots()
        # save Q-Table
        with open(f"{self.policy_path}/q_table.pkl", "wb") as f:
            pickle.dump(self.Q, f)

# 2. Q-Learning with Experience Replay
class QLearningER(QLearning):
    """
    Q-Learning with Experience Replay
    """
    def __init__(self, env=None, episodes=int(), decay_style="sigmoid", alpha=0.1, gamma=0.9, epsilon=1, policy_path=None, test_episodes=100, testing=False, verbose=0):
        super().__init__(env, episodes, decay_style, alpha, gamma, epsilon, policy_path, test_episodes, testing, verbose)
        # Initialize memory size
        self.memory_size = int(self.episodes/100)
        # Initialize replay memory
        self.replay_memory = deque(maxlen=self.memory_size)
        # Initialize batch size
        self.batch_size = 64
        # Initialize minimum memory size to start training
        self.min_memory_size = int(self.memory_size/100)
    
    def record_experience(self, data):
        """
        Record experience
        """
        # Add experience to replay memory
        self.replay_memory.append(data)
    
    def sample_experience(self):
        """
        Sample experience
        """
        # Sample experience from replay memory
        return random.sample(self.replay_memory, self.batch_size)
    
    def train(self):
        """
        Train agent
        """
        # Initialize data collection
        episode_rewards = []
        episode_data = []

        # Initialize success rate
        prev_success_rate = 0
        prev_best_rate = 0

        for episode in tqdm(range(1, self.episodes+1), ascii=True, unit="episode"):
            # Reset Environment
            observation, done = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            # update epsilon
            self.epsilon_decay(episode)
            while not done:
                # take a step
                action = self.get_epsilon_greedy_action(observation)
                new_observation, reward, done, _ = self.env.step(action, observation)

                # update data collection
                episode_reward += reward
                episode_steps += 1

                # update experience replay
                data = [observation,
                         action,
                         reward,
                         new_observation,
                         done]
                # record experience
                self.record_experience(data)
                if len(self.replay_memory) >= self.min_memory_size:
                    # sample experience
                    data_exp = self.sample_experience()
                    # update main Q-Table
                    self.Q = self.update_q_table(self.Q, self.Q, data_exp)
                # update observation
                observation = new_observation
            
            # update data collection
            episode_rewards.append(episode_reward)

            # update success rate every 100 episodes and save Q-Table
            if episode % self.test_every == 0:
                # save Q-Table
                with open(f"{self.policy_path}/q_table.pkl", "wb") as f:
                    pickle.dump(self.Q, f)
                success_rate = self.test()
                prev_success_rate = success_rate

                if success_rate >= prev_best_rate:
                    prev_best_rate = success_rate
                    # save best Q-Table
                    with open(f"{self.policy_path}/best_q_table.pkl", "wb") as f:
                        pickle.dump(self.Q, f)
            
            # log data
            episode_data.append({
                "episode number": str(episode).rjust(self.pad),
                "episode reward": str(episode_reward).rjust(self.pad),
                "average reward": str(np.mean(episode_rewards[-100:])).rjust(self.pad),
                "num of steps": str(episode_steps).rjust(self.pad),
                "epsilon": str(self.epsilon).rjust(self.pad),
                "test success_rate": str(prev_success_rate).rjust(self.pad),
                "best success_rate": str(prev_best_rate).rjust(self.pad)
            })
        
        self.log_data(episode_data)
        self.generate_plots()
        # save Q-Table
        with open(f"{self.policy_path}/q_table.pkl", "wb") as f:
            pickle.dump(self.Q, f)
        

# 3. Q-Learning with Target Network
class QLearningT(QLearning):
    """
    Q-Learning with Target Network
    """
    def __init__(self, env=None, episodes=int(), decay_style="sigmoid", alpha=0.1, gamma=0.9, epsilon=1, policy_path=None, test_episodes=100, testing=False, verbose=0):
        super().__init__(env, episodes, decay_style, alpha, gamma, epsilon, policy_path, test_episodes, testing, verbose)
        # Initialize target network
        self.target_Q = {}
        # Initialize target network update frequency
        self.target_update_freq = 10
        # Initialize target network update counter
        self.target_update_counter = 0

    def train(self):
        """
        Train agent
        """
        # Initialize data collection
        episode_rewards = []
        episode_data = []

        # Initialize success rate
        prev_success_rate = 0
        prev_best_rate = 0

        for episode in tqdm(range(1, self.episodes+1), ascii=True, unit="episode"):
            # Reset Environment
            observation, done = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            # update epsilon
            self.epsilon_decay(episode)
            while not done:
                # take a step
                action = self.get_epsilon_greedy_action(observation)
                new_observation, reward, done, _ = self.env.step(action, observation)

                # update data collection
                episode_reward += reward
                episode_steps += 1

                # update main Q-Table and target Q-Table
                data = [[observation,
                         action,
                         reward,
                         new_observation,
                         done]]
                self.Q = self.update_q_table(self.Q, self.target_Q, data)
                # update observation
                observation = new_observation

            # update data collection
            episode_rewards.append(episode_reward)

            # update success rate every 100 episodes and save Q-Table
            if episode % self.test_every == 0:
                # save Q-Table
                with open(f"{self.policy_path}/q_table.pkl", "wb") as f:
                    pickle.dump(self.Q, f)
                success_rate = self.test()
                prev_success_rate = success_rate

                if success_rate >= prev_best_rate:
                    prev_best_rate = success_rate
                    # save best Q-Table
                    with open(f"{self.policy_path}/best_q_table.pkl", "wb") as f:
                        pickle.dump(self.Q, f)
            
            # log data
            episode_data.append({
                "episode number": str(episode).rjust(self.pad),
                "episode reward": str(episode_reward).rjust(self.pad),
                "average reward": str(np.mean(episode_rewards[-100:])).rjust(self.pad),
                "num of steps": str(episode_steps).rjust(self.pad),
                "epsilon": str(self.epsilon).rjust(self.pad),
                "test success_rate": str(prev_success_rate).rjust(self.pad),
                "best success_rate": str(prev_best_rate).rjust(self.pad)
            })

            # update target network
            self.target_update_counter += 1
            if self.target_update_counter == self.target_update_freq:
                self.target_Q = self.Q
                self.target_update_counter = 0

        self.log_data(episode_data)
        self.generate_plots()
        # save Q-Table
        with open(f"{self.policy_path}/q_table.pkl", "wb") as f:
            pickle.dump(self.Q, f)


# 4. Q-Learning with Experience Replay and Target Network
class QLearningERT(QLearningER,QLearningT):
    """
    Q-Learning with Experience Replay and Target Network
    """
    def __init__(self, env=None, episodes=int(), decay_style="sigmoid", alpha=0.1, gamma=0.9, epsilon=1, policy_path=None, test_episodes=100, testing=False, verbose=0):
        super().__init__(env, episodes, decay_style, alpha, gamma, epsilon, policy_path, test_episodes, testing, verbose)

    def train(self):
        """
        Train agent
        """
        # Initialize data collection
        episode_rewards = []
        episode_data = []

        # Initialize success rate
        prev_success_rate = 0
        prev_best_rate = 0

        for episode in tqdm(range(1, self.episodes+1), ascii=True, unit="episode"):
            # Reset Environment
            observation, done = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            # update epsilon
            self.epsilon_decay(episode)
            while not done:
                # take a step
                action = self.get_epsilon_greedy_action(observation)
                new_observation, reward, done, _ = self.env.step(action, observation)

                # update data collection
                episode_reward += reward
                episode_steps += 1

                # update experience replay
                data = [observation,
                         action,
                         reward,
                         new_observation,
                         done]
                # record experience
                self.record_experience(data)
                if len(self.replay_memory) >= self.min_memory_size:
                    # sample experience
                    data_exp = self.sample_experience()
                    # update main Q-Table
                    self.Q = self.update_q_table(self.Q, self.target_Q, data_exp)
                # update observation
                observation = new_observation
            
            # update data collection
            episode_rewards.append(episode_reward)

            # update success rate every 100 episodes and save Q-Table
            if episode % self.test_every == 0:
                # save Q-Table
                with open(f"{self.policy_path}/q_table.pkl", "wb") as f:
                    pickle.dump(self.Q, f)
                success_rate = self.test()
                prev_success_rate = success_rate

                if success_rate >= prev_best_rate:
                    prev_best_rate = success_rate
                    # save best Q-Table
                    with open(f"{self.policy_path}/best_q_table.pkl", "wb") as f:
                        pickle.dump(self.Q, f)
            
            # log data
            episode_data.append({
                "episode number": str(episode).rjust(self.pad),
                "episode reward": str(episode_reward).rjust(self.pad),
                "average reward": str(np.mean(episode_rewards[-100:])).rjust(self.pad),
                "num of steps": str(episode_steps).rjust(self.pad),
                "epsilon": str(self.epsilon).rjust(self.pad),
                "test success_rate": str(prev_success_rate).rjust(self.pad),
                "best success_rate": str(prev_best_rate).rjust(self.pad)
            })

            # update target network
            self.target_update_counter += 1
            if self.target_update_counter == self.target_update_freq:
                self.target_Q = self.Q
                self.target_update_counter = 0
            
        self.log_data(episode_data)
        self.generate_plots()
        # save Q-Table
        with open(f"{self.policy_path}/q_table.pkl", "wb") as f:
            pickle.dump(self.Q, f)

# 5. Double Q-Learning
class DQLearning(QLearning):
    def __init__(self, env=None, episodes=int(), decay_style="sigmoid", alpha=0.1, gamma=0.9, epsilon=1, policy_path=None, test_episodes=100, testing=False, verbose=0, path1=None, path2=None):
        super().__init__(env, episodes, decay_style, alpha, gamma, epsilon, policy_path, test_episodes, testing, verbose)
        # Initialize Q-Tables
        if path1 is not None:
            self.Q1 = pickle.load(open(path1, "rb"))
        else:
            self.Q1 = {}
        if path2 is not None:
            self.Q2 = pickle.load(open(path2, "rb"))
        else:
            self.Q2 = {}

    def get_q_action(self, observation):
        """
        Get action from Q-Table
        """
        state = tuple(observation)
        # Add observation to Q-Table if needed
        self.Q1 = self.add_table_keys(self.Q1, state)
        self.Q2 = self.add_table_keys(self.Q2, state)
        # Get action from average of Q-Tables
        table = {}
        table[state] = {}
        for key in self.Q1[state].keys():
            table[state][key] = (self.Q1[state][key] + self.Q2[state][key])/2
        action = max(table[state], key=table[state].get)
        # Return action
        return action
    
    def update_q_table(self, q, q_target, data):
        """
        Update both the Q-Tables
        """
        for _, (observation, action, reward, new_observation, done) in enumerate(data):
            state = tuple(observation)
            new_state = tuple(new_observation)
            # Add new observation to Q-Tables if needed
            q = self.add_table_keys(q, state)
            q = self.add_table_keys(q, new_state)
            q_target = self.add_table_keys(q_target, state)
            q_target = self.add_table_keys(q_target, new_state)
            # Update Q-Table
            if not done:
                opt_action = max(q[new_state], key=q[new_state].get)
                q[state][action] = q[state][action] + self.alpha*(reward + self.gamma*q_target[new_state][opt_action] - q[state][action])
            else:
                q[state][action] = reward
            
        # return updated Q-Table
        return q
                    
    def test_policy(self, path1, path2):
        """
        Test agent with a given policy
        """
        print("Testing Agent...")
        # Load Q-Table
        with open(path1, "rb") as f:
            Q1 = pickle.load(f)
        with open(path2, "rb") as f:
            Q2 = pickle.load(f)
        # Success Count
        success_count = 0
        for _ in range(1, self.test_episodes+1):
            # Reset Environment
            observation, done = self.env.reset()
            while not done:
                # take a step
                # Get action from average of Q-Tables
                table = {}
                table[tuple(observation)] = {}
                for key in Q1[tuple(observation)].keys():
                    table[tuple(observation)][key] = (Q1[tuple(observation)][key] + Q2[tuple(observation)][key])/2
                action = max(table[tuple(observation)], key=table[tuple(observation)].get)
                new_observation, _, done, episode_status = self.env.step(action, observation)
                # update observation
                observation = new_observation
            # update success count
            if episode_status:
                success_count += 1
        print(f"Done Testing Agent with {success_count/self.test_episodes*100}% success rate")

    def display_policy(self, path1, path2, save_path, create_gif, epsiodes):
        """
        Test agent with a given policy
        """
        print("Testing Agent...")
        # Load Q-Table
        with open(path1, "rb") as f:
            Q1 = pickle.load(f)
        with open(path2, "rb") as f:
            Q2 = pickle.load(f)
        # Success Count
        success_count = 0
        for episode in tqdm(range(1, epsiodes+1), ascii=True, unit="episode"):
            # Reset Environment
            observation, done = self.env.reset()
            # step count
            step = 0
            prev_action = None
            if create_gif:
                step_path = f"{save_path}/episode_{episode}"
                if not os.path.exists(step_path):
                    os.makedirs(step_path)
            while not done:
                # take a step
                if (observation[0],observation[1]) in self.env.ugs_loc:
                    # Get action from average of Q-Tables
                    table = {}
                    table[tuple(observation)] = {}
                    for key in Q1[tuple(observation)].keys():
                        table[tuple(observation)][key] = (Q1[tuple(observation)][key] + Q2[tuple(observation)][key])/2
                    action = max(table[tuple(observation)], key=table[tuple(observation)].get)
                else:
                    action = prev_action
                new_observation, _, done, episode_status = self.env.step(action, observation)
                if self.env.render_sim:
                    self.env.render()
                # update observation
                observation = new_observation
                prev_action = action
                step += 1
                if create_gif:
                    # save frames
                    self.env.save_frame(step_path, episode, step)
            # update success count
            if self.env.render_sim:
                self.env.close(save_path, episode)
            if episode_status:
                success_count += 1
        print(f"Done Testing Agent with {success_count*100/epsiodes}% success rate")

    def train(self):
        """
        Train agent
        """
        # Initialize data collection
        episode_rewards = []
        episode_data = []

        # Initialize success rate
        prev_success_rate = 0
        prev_best_rate = 0

        for episode in tqdm(range(1, self.episodes+1), ascii=True, unit="episode"):
            # Reset Environment
            observation, done = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            # update epsilon    
            self.epsilon_decay(episode)
            while not done:
                # take a step
                action = self.get_epsilon_greedy_action(observation)
                new_observation, reward, done, _ = self.env.step(action, observation)

                # update data collection
                episode_reward += reward
                episode_steps += 1

                # update experience replay
                data = [[observation,
                         action,
                         reward,
                         new_observation,
                         done]]  
                # update Q-Tables
                rand = random.uniform(0, 1)
                if rand < 0.5:
                    self.Q1 = self.update_q_table(self.Q1, self.Q2, data)
                else:
                    self.Q2 = self.update_q_table(self.Q2, self.Q1, data)
                # update observation
                observation = new_observation

            # update data collection
            episode_rewards.append(episode_reward)

            # update success rate every 100 episodes and save Q-Table
            if episode % self.test_every == 0:
                # save Q-Tables
                with open(f"{self.policy_path}/q_table_1.pkl", "wb") as f:
                    pickle.dump(self.Q1, f)
                with open(f"{self.policy_path}/q_table_2.pkl", "wb") as f:
                    pickle.dump(self.Q2, f)
                success_rate = self.test()
                prev_success_rate = success_rate

                if success_rate >= prev_best_rate:
                    prev_best_rate = success_rate
                    # save best Q-Tables
                    with open(f"{self.policy_path}/best_q_table_1.pkl", "wb") as f:
                        pickle.dump(self.Q1, f)
                    with open(f"{self.policy_path}/best_q_table_2.pkl", "wb") as f:
                        pickle.dump(self.Q2, f)

            # log data
            episode_data.append({
                "episode number": str(episode).rjust(self.pad),
                "episode reward": str(episode_reward).rjust(self.pad),
                "average reward": str(np.mean(episode_rewards[-100:])).rjust(self.pad),
                "num of steps": str(episode_steps).rjust(self.pad),
                "epsilon": str(self.epsilon).rjust(self.pad),
                "test success_rate": str(prev_success_rate).rjust(self.pad),
                "best success_rate": str(prev_best_rate).rjust(self.pad)
            })

        self.log_data(episode_data)
        self.generate_plots()
        # save Q-Table
        with open(f"{self.policy_path}/q_table_1.pkl", "wb") as f:
            pickle.dump(self.Q1, f)
        with open(f"{self.policy_path}/q_table_2.pkl", "wb") as f:
            pickle.dump(self.Q2, f)


# 6. Double Q-Learning with Experience Replay
class DQLearningER(DQLearning, QLearningER):
    """
    Double Q-Learning with Experience Replay
    """
    def __init__(self, env=None, episodes=int(), decay_style="sigmoid", alpha=0.1, gamma=0.9, epsilon=1, policy_path=None, test_episodes=100, testing=False, verbose=0, path1=None, path2=None):
        super().__init__(env, episodes, decay_style, alpha, gamma, policy_path, epsilon, test_episodes, testing, verbose, path1, path2)
        
    def train(self):
        """
        Train agent
        """
        # Initialize data collection
        episode_rewards = []
        episode_data = []

        # Initialize success rate
        prev_success_rate = 0
        prev_best_rate = 0

        for episode in tqdm(range(1, self.episodes+1), ascii=True, unit="episode"):
            # Reset Environment
            observation, done = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            # update epsilon    
            self.epsilon_decay(episode)
            while not done:
                # take a step
                action = self.get_epsilon_greedy_action(observation)
                new_observation, reward, done, _ = self.env.step(action, observation)

                # update data collection
                episode_reward += reward
                episode_steps += 1

                # update experience replay
                data = [observation,
                         action,
                         reward,
                         new_observation,
                         done]
                # record experience
                self.record_experience(data)
                if len(self.replay_memory) >= self.min_memory_size:    
                    # sample experience
                    data = self.sample_experience()
                    # update Q-Tables
                    rand = random.uniform(0, 1)
                    if rand < 0.5:
                        self.Q1 = self.update_q_table(self.Q1, self.Q2, data)
                    else:
                        self.Q2 = self.update_q_table(self.Q2, self.Q1, data)
                # update observation
                observation = new_observation

            # update data collection
            episode_rewards.append(episode_reward)

            # update success rate every 100 episodes and save Q-Table
            if episode % self.test_every == 0:
                # save Q-Tables
                with open(f"{self.policy_path}/q_table_1.pkl", "wb") as f:
                    pickle.dump(self.Q1, f)
                with open(f"{self.policy_path}/q_table_2.pkl", "wb") as f:
                    pickle.dump(self.Q2, f)
                success_rate = self.test()
                prev_success_rate = success_rate

                if success_rate >= prev_best_rate:
                    prev_best_rate = success_rate
                    # save best Q-Tables
                    with open(f"{self.policy_path}/best_q_table_1.pkl", "wb") as f:
                        pickle.dump(self.Q1, f)
                    with open(f"{self.policy_path}/best_q_table_2.pkl", "wb") as f:
                        pickle.dump(self.Q2, f)

            # log data
            episode_data.append({
                "episode number": str(episode).rjust(self.pad),
                "episode reward": str(episode_reward).rjust(self.pad),
                "average reward": str(np.mean(episode_rewards[-100:])).rjust(self.pad),
                "num of steps": str(episode_steps).rjust(self.pad),
                "epsilon": str(self.epsilon).rjust(self.pad),
                "test success_rate": str(prev_success_rate).rjust(self.pad),
                "best success_rate": str(prev_best_rate).rjust(self.pad)
            })

        self.log_data(episode_data)
        self.generate_plots()
        # save Q-Table
        with open(f"{self.policy_path}/q_table_1.pkl", "wb") as f:
            pickle.dump(self.Q1, f)
        with open(f"{self.policy_path}/q_table_2.pkl", "wb") as f:
            pickle.dump(self.Q2, f)
            

# 7. Double Q-Learning with Target Network
class DQLearningT(DQLearning):
    """
    Double Q-Learning with Target Network
    """
    def __init__(self, env=None, episodes=int(), decay_style="sigmoid", alpha=0.1, gamma=0.9, epsilon=1, policy_path=None, test_episodes=100, testing=False, verbose=0, path1=None, path2=None):
        super().__init__(env, episodes, decay_style, alpha, gamma, epsilon, policy_path, test_episodes, testing, verbose, path1, path2)
        # Initialize target network
        self.target_Q1 = {}
        self.target_Q2 = {}
        # Initialize target network update frequency
        self.target_update_freq = 10
        # Initialize target network update counter
        self.target_update_counter = 0

    def update_q_table(self, q1, q2, q_target, data):
        """
        Update both the Q-Tables
        """
        for _, (observation, action, reward, new_observation, done) in enumerate(data):
            state = tuple(observation)
            new_state = tuple(new_observation)
            # Add new observation to Q-Tables if needed
            q1 = self.add_table_keys(q1, state)
            q1 = self.add_table_keys(q1, new_state)
            q2 = self.add_table_keys(q2, state)
            q2 = self.add_table_keys(q2, new_state)
            q_target = self.add_table_keys(q_target, state)
            q_target = self.add_table_keys(q_target, new_state)
            # Update Q-Table
            if not done:
                opt_action = max(q_target[new_state], key=q_target[new_state].get)
                q1[state][action] = q1[state][action] + self.alpha*(reward + self.gamma*q2[new_state][opt_action] - q1[state][action])
            else:
                q1[state][action] = reward
            
        # return updated Q-Table
        return q1
    
    def train(self):
        """
        Train agent
        """
        # Initialize data collection
        episode_rewards = []
        episode_data = []

        # Initialize success rate
        prev_success_rate = 0
        prev_best_rate = 0

        for episode in tqdm(range(1, self.episodes+1), ascii=True, unit="episode"):
            # Reset Environment
            observation, done = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            # update epsilon    
            self.epsilon_decay(episode)
            while not done:
                # take a step
                action = self.get_epsilon_greedy_action(observation)
                new_observation, reward, done, _ = self.env.step(action, observation)

                # update data collection
                episode_reward += reward
                episode_steps += 1

                # update main Q-Table and target Q-Table
                data = [[observation,
                         action,
                         reward,
                         new_observation,
                         done]]
                self.Q1 = self.update_q_table(self.Q1, self.Q2, self.target_Q1, data)
                # update observation
                observation = new_observation

            # update data collection
            episode_rewards.append(episode_reward)

            # update success rate every 100 episodes and save Q-Table
            if episode % self.test_every == 0:
                # save Q-Tables
                with open(f"{self.policy_path}/q_table_1.pkl", "wb") as f:
                    pickle.dump(self.Q1, f)
                with open(f"{self.policy_path}/q_table_2.pkl", "wb") as f:
                    pickle.dump(self.Q2, f)
                success_rate = self.test()
                prev_success_rate = success_rate

                if success_rate >= prev_best_rate:
                    prev_best_rate = success_rate
                    # save best Q-Tables
                    with open(f"{self.policy_path}/best_q_table_1.pkl", "wb") as f:
                        pickle.dump(self.Q1, f)
                    with open(f"{self.policy_path}/best_q_table_2.pkl", "wb") as f:
                        pickle.dump(self.Q2, f)

            # log data
            episode_data.append({
                "episode number": str(episode).rjust(self.pad),
                "episode reward": str(episode_reward).rjust(self.pad),
                "average reward": str(np.mean(episode_rewards[-100:])).rjust(self.pad),
                "num of steps": str(episode_steps).rjust(self.pad),
                "epsilon": str(self.epsilon).rjust(self.pad),
                "test success_rate": str(prev_success_rate).rjust(self.pad),
                "best success_rate": str(prev_best_rate).rjust(self.pad)
            })

            # update target network
            self.target_update_counter += 1
            if self.target_update_counter == self.target_update_freq:
                self.target_Q1 = self.Q1
                self.target_Q2 = self.Q2
                self.target_update_counter = 0

        self.log_data(episode_data)
        self.generate_plots()
        # save Q-Table
        with open(f"{self.policy_path}/q_table_1.pkl", "wb") as f:
            pickle.dump(self.Q1, f)
        with open(f"{self.policy_path}/q_table_2.pkl", "wb") as f:
            pickle.dump(self.Q2, f)


# 8. Double Q-Learning with Experience Replay and Target Network
class DQLearningERT(DQLearningER, DQLearningT):
    """
    Double Q-Learning with Experience Replay and Target Network
    """
    def __init__(self, env=None, episodes=int(), decay_style="sigmoid", alpha=0.1, gamma=0.9, epsilon=1, policy_path=None, test_episodes=100, testing=False, verbose=0, path1=None, path2=None):
        super().__init__(env, episodes, decay_style, alpha, gamma, epsilon, policy_path, test_episodes, testing, verbose, path1, path2)
    
    def train(self):
        """
        Train agent
        """
        # Initialize data collection
        episode_rewards = []
        episode_data = []

        # Initialize success rate
        prev_success_rate = 0
        prev_best_rate = 0

        for episode in tqdm(range(1, self.episodes+1), ascii=True, unit="episode"):
            # Reset Environment
            observation, done = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            # update epsilon    
            self.epsilon_decay(episode)
            while not done:
                # take a step
                action = self.get_epsilon_greedy_action(observation)
                new_observation, reward, done, _ = self.env.step(action, observation)

                # update data collection
                episode_reward += reward
                episode_steps += 1

                # update experience replay
                data = [observation,
                         action,
                         reward,
                         new_observation,
                         done]
                # record experience
                self.record_experience(data)
                if len(self.replay_memory) >= self.min_memory_size:    
                    # sample experience
                    data = self.sample_experience()
                    # update Q-Tables
                    rand = random.uniform(0, 1)
                    if rand < 0.5:
                        self.Q1 = self.update_q_table(self.Q1, self.Q2, self.target_Q1, data)
                    else:
                        self.Q2 = self.update_q_table(self.Q2, self.Q1, self.target_Q2, data)
                # update observation
                observation = new_observation

            # update data collection
            episode_rewards.append(episode_reward)

            # update success rate every 100 episodes and save Q-Table
            if episode % self.test_every == 0:
                # save Q-Tables
                with open(f"{self.policy_path}/q_table_1.pkl", "wb") as f:
                    pickle.dump(self.Q1, f)
                with open(f"{self.policy_path}/q_table_2.pkl", "wb") as f:
                    pickle.dump(self.Q2, f)
                success_rate = self.test()
                prev_success_rate = success_rate

                if success_rate >= prev_best_rate:
                    prev_best_rate = success_rate
                    # save best Q-Tables
                    with open(f"{self.policy_path}/best_q_table_1.pkl", "wb") as f:
                        pickle.dump(self.Q1, f)
                    with open(f"{self.policy_path}/best_q_table_2.pkl", "wb") as f:
                        pickle.dump(self.Q2, f)

            # log data
            episode_data.append({
                "episode number": str(episode).rjust(self.pad),
                "episode reward": str(episode_reward).rjust(self.pad),
                "average reward": str(np.mean(episode_rewards[-100:])).rjust(self.pad),
                "num of steps": str(episode_steps).rjust(self.pad),
                "epsilon": str(self.epsilon).rjust(self.pad),
                "test success_rate": str(prev_success_rate).rjust(self.pad),
                "best success_rate": str(prev_best_rate).rjust(self.pad)
            })

            # update target network
            self.target_update_counter += 1
            if self.target_update_counter == self.target_update_freq:
                self.target_Q1 = self.Q1
                self.target_Q2 = self.Q2
                self.target_update_counter = 0

        self.log_data(episode_data)
        self.generate_plots()
        # save Q-Table
        with open(f"{self.policy_path}/q_table_1.pkl", "wb") as f:
            pickle.dump(self.Q1, f)
        with open(f"{self.policy_path}/q_table_2.pkl", "wb") as f:
            pickle.dump(self.Q2, f)
            

# 9. DQN
class DQN(QLearning):
    """
    Deep Q-Learning Network
    """
    def __init__(self, env=None, episodes=int(), decay_style="sigmoid", alpha=0.001, gamma=0.9, epsilon=1, policy_path=None, test_episodes=100, batch_size=64, testing=False, use_conv=False, verbose = 0):
        super().__init__(env, episodes, decay_style, alpha, gamma, epsilon, policy_path, test_episodes, testing, verbose)
        # Initialize Q-Network
        if policy_path is not None:
            self.Q = tf.keras.models.load_model(f"{policy_path}")
        else:
            self.Q = self.build_network(use_conv)
        # Initialize batch size
        self.batch_size = batch_size

    def build_network(self, use_conv):
        """
        Build Q-Network
        """
        # Initialize Q-Network
        model = tf.keras.Sequential()
        # Check if using convolutional layers
        if use_conv:
            # Add convolutional layers
            model.add(tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.env.observation_space))
            model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
            model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(512, activation='relu'))
            model.add(tf.keras.layers.Dense(len(self.env.action_space), activation='linear'))
        else:
            # Add dense layers
            model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=self.env.observation_space))
            model.add(tf.keras.layers.Dense(64, activation='relu'))
            model.add(tf.keras.layers.Dense(len(self.env.action_space), activation='linear'))
        # Compile Q-Network
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), metrics=['accuracy'])
        # Return Q-Network
        return model
    
    def get_q_action(self, observation):
        """
        Get action from Q-Network
        """
        # Get action from Q-Network
        action = np.argmax(self.Q.predict(np.array([observation]), verbose=self.verbose)[0])
        # Return action
        return action
    
    def update_q_network(self, data):
        """
        Update Q-Network
        """
        # Initialize training data
        # Get states and their corresponding Q-Values
        state = np.array([state[0] for state in data])
        q_state = self.Q.predict(state, verbose=self.verbose)
        
        # Get new states and their corresponding Q-Values
        new_state = np.array([new_state[3] for new_state in data])
        q_new_state = self.Q.predict(new_state, verbose=self.verbose)

        # Initialize training labels
        X = []
        y = []

        # Update training labels
        for index, (observation, action, reward, _, done) in enumerate(data):
            if not done:
                # Update Q-Value for action
                q_state[index][action] = reward + self.gamma*np.max(q_new_state[index])
            else:
                # Update Q-Value for action in terminal state
                q_state[index][action] = reward
            # Update training labels
            X.append(observation)
            y.append(q_state[index])

        # Train Q-Network
        history = self.Q.fit(np.array(X), np.array(y), epochs=1, batch_size=self.batch_size, verbose=self.verbose, shuffle=False)
        # return training history
        return history
    
    def train(self):
        """
        Train agent
        """
        # Initialize data collection
        episode_rewards = []
        episode_data = []

        # Initialize success rate
        prev_success_rate = 0
        prev_best_rate = 0

        for episode in tqdm(range(1, self.episodes+1), ascii=True, unit="episode"):
            # Reset Environment
            observation, done = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            # update epsilon    
            self.epsilon_decay(episode)
            while not done:
                # take a step
                action = self.get_epsilon_greedy_action(observation)
                new_observation, reward, done, _ = self.env.step(action, observation)

                # update data collection
                episode_reward += reward
                episode_steps += 1

                # update experience replay
                data = [observation,
                         action,
                         reward,
                         new_observation,
                         done]
                # update Q-Network
                history = self.update_q_network(data)
                # update observation
                observation = new_observation

            # update data collection
            episode_rewards.append(episode_reward)

            # update success rate every 100 episodes and save Q-Network
            if episode % self.test_every == 0:
                # save Q-Network
                self.Q.save(f"{self.policy_path}/q_network.h5")
                success_rate = self.test()
                prev_success_rate = success_rate

                if success_rate >= prev_best_rate:
                    prev_best_rate = success_rate
                    # save best Q-Network
                    self.Q.save(f"{self.policy_path}/best_q_network.h5")
                    # Clear Session and unload model to avoid memory leaks
                    K.clear_session()
                    del self.Q
                    # reload best Q-Network
                    self.Q = tf.keras.models.load_model(f"{self.policy_path}/best_q_network.h5")
                    self.Q.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), metrics=['accuracy'])

            # log data
            episode_data.append({
                "episode number": str(episode).rjust(self.pad),
                "episode reward": str(episode_reward).rjust(self.pad),
                "average reward": str(np.mean(episode_rewards[-100:])).rjust(self.pad),
                "num of steps": str(episode_steps).rjust(self.pad),
                "epsilon": str(self.epsilon).rjust(self.pad),
                "model accuracy": str(history.history['accuracy'][-1]).rjust(self.pad),
                "model loss": str(history.history['loss'][-1]).rjust(self.pad),
                "test success_rate": str(prev_success_rate).rjust(self.pad),
                "best success_rate": str(prev_best_rate).rjust(self.pad)
            })

        self.log_data(episode_data)
        self.generate_plots()
        # save Q-Table
        self.Q.save(f"{self.policy_path}/q_network.h5")


# 10. DQN with Experience Replay
class DQN_ER(DQN,QLearningER):
    """
    Deep Q-Learning Network with Experience Replay
    """
    def __init__(self, env=None, episodes=int(), decay_style="sigmoid", alpha=0.001, gamma=0.9, epsilon=1, policy_path=None, test_episodes=100, batch_size=64, testing=False, use_conv=False, verbose=0):
        super().__init__(env, episodes, decay_style, alpha, gamma, epsilon, policy_path, test_episodes, batch_size, testing, use_conv, verbose)

    def train(self):
        """
        Train agent
        """
        # Initialize data collection
        episode_rewards = []
        episode_data = []

        # Initialize success rate
        prev_success_rate = 0
        prev_best_rate = 0

        for episode in tqdm(range(1, self.episodes+1), ascii=True, unit="episode"):
            # Reset Environment
            observation, done = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            # update epsilon    
            self.epsilon_decay(episode)
            while not done:
                # take a step
                action = self.get_epsilon_greedy_action(observation)
                new_observation, reward, done, _ = self.env.step(action, observation)

                # update data collection
                episode_reward += reward
                episode_steps += 1

                # update experience replay
                data = [observation,
                         action,
                         reward,
                         new_observation,
                         done]
                # record experience
                self.record_experience(data)
                if len(self.replay_memory) >= self.min_memory_size:    
                    # sample experience
                    data = self.sample_experience()
                    # update Q-Network
                    history = self.update_q_network(data)
                # update observation
                observation = new_observation

            # update data collection
            episode_rewards.append(episode_reward)

            # update success rate every 100 episodes and save Q-Network
            if episode % self.test_every == 0:
                # save Q-Network
                self.Q.save(f"{self.policy_path}/q_network.h5")
                success_rate = self.test()
                prev_success_rate = success_rate

                if success_rate >= prev_best_rate:
                    prev_best_rate = success_rate
                    # save best Q-Network
                    self.Q.save(f"{self.policy_path}/best_q_network.h5")
                    # Clear Session and unload model to avoid memory leaks
                    K.clear_session()
                    del self.Q
                    # reload best Q-Network
                    self.Q = tf.keras.models.load_model(f"{self.policy_path}/best_q_network.h5")
                    self.Q.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), metrics=['accuracy'])

            # log data
            episode_data.append({
                "episode number": str(episode).rjust(self.pad),
                "episode reward": str(episode_reward).rjust(self.pad),
                "average reward": str(np.mean(episode_rewards[-100:])).rjust(self.pad),
                "num of steps": str(episode_steps).rjust(self.pad),
                "epsilon": str(self.epsilon).rjust(self.pad),
                "model accuracy": str(history.history['accuracy'][-1]).rjust(self.pad),
                "model loss": str(history.history['loss'][-1]).rjust(self.pad),
                "test success_rate": str(prev_success_rate).rjust(self.pad),
                "best success_rate": str(prev_best_rate).rjust(self.pad)
            })

        self.log_data(episode_data)
        self.generate_plots()
        # save Q-Table
        self.Q.save(f"{self.policy_path}/q_network.h5")

    
# 11. DQN with Target Network
class DQN_T(DQN):
    """
    Deep Q-Learning Network with Target Network
    """
    def __init__(self, env=None, episodes=int(), decay_style="sigmoid", alpha=0.001, gamma=0.9, epsilon=1, policy_path=None, test_episodes=100, batch_size=64, testing=False, use_conv=False, verbose=0):
        super().__init__(env, episodes, decay_style, alpha, gamma, epsilon, policy_path, test_episodes, batch_size, testing, use_conv, verbose)
        # Initialize target network
        self.target_Q = self.build_network(use_conv)
        self.target_Q.set_weights(self.Q.get_weights())
        # Initialize target network update frequency
        self.target_update_freq = 10
        # Initialize target network update counter
        self.target_update_counter = 0

    def update_q_network(self, data):
        """
        Update Q-Network
        """
        # Initialize training data
        # Get states and their corresponding Q-Values
        state = np.array([state[0] for state in data])
        q_state = self.Q.predict(state, verbose=self.verbose)
        
        # Get new states and their corresponding Q-Values from target network
        new_state = np.array([new_state[3] for new_state in data])
        q_new_state = self.target_Q.predict(new_state, verbose=self.verbose)

        # Initialize training labels
        X = []
        y = []

        # Update training labels
        for index, (observation, action, reward, _, done) in enumerate(data):
            if not done:
                # Update Q-Value for action
                q_state[index][action] = reward + self.gamma*np.max(q_new_state[index])
            else:
                # Update Q-Value for action in terminal state
                q_state[index][action] = reward
            # Update training labels
            X.append(observation)
            y.append(q_state[index])

        # Train Q-Network
        history = self.Q.fit(np.array(X), np.array(y), epochs=1, batch_size=self.batch_size, verbose=self.verbose, shuffle=False)
        # return training history
        return history
        
    def train(self):
        """
        Train agent
        """
        # Initialize data collection
        episode_rewards = []
        episode_data = []

        # Initialize success rate
        prev_success_rate = 0
        prev_best_rate = 0

        for episode in tqdm(range(1, self.episodes+1), ascii=True, unit="episode"):
            # Reset Environment
            observation, done = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            # update epsilon    
            self.epsilon_decay(episode)
            while not done:
                # take a step
                action = self.get_epsilon_greedy_action(observation)
                new_observation, reward, done, _ = self.env.step(action, observation)

                # update data collection
                episode_reward += reward
                episode_steps += 1

                # update experience replay
                data = [observation,
                         action,
                         reward,
                         new_observation,
                         done]
                # update Q-Network
                history = self.update_q_network(data)
                # update observation
                observation = new_observation

            # update data collection
            episode_rewards.append(episode_reward)

            # update success rate every 100 episodes and save Q-Network
            if episode % self.test_every == 0:
                # save Q-Network
                self.Q.save(f"{self.policy_path}/q_network.h5")
                success_rate = self.test()
                prev_success_rate = success_rate

                if success_rate >= prev_best_rate:
                    prev_best_rate = success_rate
                    # save best Q-Network
                    self.Q.save(f"{self.policy_path}/best_q_network.h5")
                    # Clear Session and unload model to avoid memory leaks
                    K.clear_session()
                    del self.Q
                    # reload best Q-Network
                    self.Q = tf.keras.models.load_model(f"{self.policy_path}/best_q_network.h5")
                    self.Q.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), metrics=['accuracy'])

            # log data
            episode_data.append({
                "episode number": str(episode).rjust(self.pad),
                "episode reward": str(episode_reward).rjust(self.pad),
                "average reward": str(np.mean(episode_rewards[-100:])).rjust(self.pad),
                "num of steps": str(episode_steps).rjust(self.pad),
                "epsilon": str(self.epsilon).rjust(self.pad),
                "model accuracy": str(history.history['accuracy'][-1]).rjust(self.pad),
                "model loss": str(history.history['loss'][-1]).rjust(self.pad),
                "test success_rate": str(prev_success_rate).rjust(self.pad),
                "best success_rate": str(prev_best_rate).r
            })

            # update target network
            self.target_update_counter += 1
            if self.target_update_counter == self.target_update_freq:
                self.target_Q.set_weights(self.Q.get_weights())
                self.target_update_counter = 0

        self.log_data(episode_data)
        self.generate_plots()
        # save Q-Table
        self.Q.save(f"{self.policy_path}/q_network.h5")


# 12. DQN with Experience Replay and Target Network
class DQN_ERT(DQN_T,DQN_ER):
    """
    Deep Q-Learning Network with Experience Replay and Target Network
    """
    def __init__(self, env=None, episodes=int(), decay_style="sigmoid", alpha=0.001, gamma=0.9, epsilon=1, policy_path=None, test_episodes=100, batch_size=64, testing=False, use_conv=False, verbose=0):
        super().__init__(env, episodes, decay_style, alpha, gamma, epsilon, policy_path, test_episodes, batch_size, testing, use_conv, verbose)

    def train(self):
        """
        Train agent
        """
        # Initialize data collection
        episode_rewards = []
        episode_data = []

        # Initialize success rate
        prev_success_rate = 0
        prev_best_rate = 0

        for episode in tqdm(range(1, self.episodes+1), ascii=True, unit="episode"):
            # Reset Environment
            observation, done = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            # update epsilon    
            self.epsilon_decay(episode)
            while not done:
                # take a step
                action = self.get_epsilon_greedy_action(observation)
                new_observation, reward, done, _ = self.env.step(action, observation)

                # update data collection
                episode_reward += reward
                episode_steps += 1

                # update experience replay
                data = [observation,
                         action,
                         reward,
                         new_observation,
                         done]
                # record experience
                self.record_experience(data)
                if len(self.replay_memory) >= self.min_memory_size:    
                    # sample experience
                    data = self.sample_experience()
                    # update Q-Network
                    history = self.update_q_network(data)
                # update observation
                observation = new_observation

            # update data collection
            episode_rewards.append(episode_reward)

            # update success rate every 100 episodes and save Q-Network
            if episode % self.test_every == 0:
                # save Q-Network
                self.Q.save(f"{self.policy_path}/q_network.h5")
                success_rate = self.test()
                prev_success_rate = success_rate

                if success_rate >= prev_best_rate:
                    prev_best_rate = success_rate
                    # save best Q-Network
                    self.Q.save(f"{self.policy_path}/best_q_network.h5")
                    # Clear Session and unload model to avoid memory leaks
                    K.clear_session()
                    del self.Q
                    # reload best Q-Network
                    self.Q = tf.keras.models.load_model(f"{self.policy_path}/best_q_network.h5")
                    self.Q.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), metrics=['accuracy'])

            # log data
            episode_data.append({
                "episode number": str(episode).rjust(self.pad),
                "episode reward": str(episode_reward).rjust(self.pad),
                "average reward": str(np.mean(episode_rewards[-100:])).rjust(self.pad),
                "num of steps": str(episode_steps).rjust(self.pad),
                "epsilon": str(self.epsilon).rjust(self.pad),
                "model accuracy": str(history.history['accuracy'][-1]).rjust(self.pad),
                "model loss": str(history.history['loss'][-1]).rjust(self.pad),
                "test success_rate": str(prev_success_rate).rjust(self.pad),
                "best success_rate": str(prev_best_rate).rjust(self.pad)
            })

            # update target network
            self.target_update_counter += 1
            if self.target_update_counter == self.target_update_freq:
                self.target_Q.set_weights(self.Q.get_weights())
                self.target_update_counter = 0

        self.log_data(episode_data)
        self.generate_plots()
        # save Q-Table
        self.Q.save(f"{self.policy_path}/q_network.h5")


# 13. Double DQN
class DDQN(DQN_T):
    """
    Double Deep Q-Learning Network
    ''' Its a combination of DQN with Target Network. Use the Q-network to select the best action and the target Network to evaluate the Q-Value of the best action'''
    """
    def __init__(self, env=None, episodes=int(), decay_style="sigmoid", alpha=0.001, gamma=0.9, epsilon=1, policy_path=None, test_episodes=100, batch_size=64, testing=False, use_conv=False, verbose=0):
        super().__init__(env, episodes, decay_style, alpha, gamma, epsilon, policy_path, test_episodes, batch_size, testing, use_conv, verbose)

    def get_q_action(self, observation):
        """
        Get action from Q-Network
        """
        # Get action from Q-Network
        action = np.argmax(self.Q.predict(np.array([observation]), verbose=self.verbose)[0])
        # Return action
        return action
    
    def update_q_network(self, q1, q2, data):
        """
        Update Q-Network
        """
        # Initialize training data
        # Get states and their corresponding Q-Values
        state = np.array([state[0] for state in data])
        q_state = q1.predict(state, verbose=self.verbose)
        
        # Get new states and their corresponding Q-Values
        new_state = np.array([new_state[3] for new_state in data])
        q_new_state_q1 = q1.predict(new_state, verbose=self.verbose)
        q_new_state_q2 = q2.predict(new_state, verbose=self.verbose)

        # Initialize training labels
        X = []
        y = []

        # Update training labels
        for index, (observation, action, reward, _, done) in enumerate(data):
            if not done:
                # Update Q-Value for action
                q_state[index][action] = reward + self.gamma*q_new_state_q2[index][np.argmax(q_new_state_q1[index])]
            else:
                # Update Q-Value for action in terminal state
                q_state[index][action] = reward
            # Update training labels
            X.append(observation)
            y.append(q_state[index])

        # Train Q-Network
        history = q1.fit(np.array(X), np.array(y), epochs=1, batch_size=self.batch_size, verbose=self.verbose, shuffle=False)
        # return training history
        return history
    
    def train(self):
        """
        Train agent
        """
        # Initialize data collection
        episode_rewards = []
        episode_data = []

        # Initialize success rate
        prev_success_rate = 0
        prev_best_rate = 0

        for episode in tqdm(range(1, self.episodes+1), ascii=True, unit="episode"):
            # Reset Environment
            observation, done = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            # update epsilon    
            self.epsilon_decay(episode)
            while not done:
                # take a step
                action = self.get_epsilon_greedy_action(observation)
                new_observation, reward, done, _ = self.env.step(action, observation)

                # update data collection
                episode_reward += reward
                episode_steps += 1

                # update experience replay
                data = [[observation,
                         action,
                         reward,
                         new_observation,
                         done]]
                # update Q-Network
                history = self.update_q_network(self.Q, self.target_Q, data)
                # update observation
                observation = new_observation

            # update data collection
            episode_rewards.append(episode_reward)

            # update target network
            self.target_update_counter += 1
            if self.target_update_counter == self.target_update_freq:
                self.target_Q.set_weights(self.Q.get_weights())
                self.target_update_counter = 0

            # update success rate every 100 episodes and save Q-Network
            if episode % self.test_every == 0:
                # save Q-Network
                self.Q.save(f"{self.policy_path}/q_network.h5")
                success_rate = self.test()
                prev_success_rate = success_rate

                if success_rate >= prev_best_rate:
                    prev_best_rate = success_rate
                    # save best Q-Network
                    self.Q.save(f"{self.policy_path}/best_q_network.h5")
                    # Clear Session and unload model to avoid memory leaks
                    K.clear_session()
                    del self.Q
                    # reload best Q-Network
                    self.Q = tf.keras.models.load_model(f"{self.policy_path}/best_q_network.h5")
                    self.Q.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), metrics=['accuracy'])
        
            # log data
            episode_data.append({
                "episode number": str(episode).rjust(self.pad),
                "episode reward": str(episode_reward).rjust(self.pad),
                "average reward": str(np.mean(episode_rewards[-100:])).rjust(self.pad),
                "num of steps": str(episode_steps).rjust(self.pad),
                "epsilon": str(self.epsilon).rjust(self.pad),
                "model accuracy": str(history.history['accuracy'][-1]).rjust(self.pad),
                "model loss": str(history.history['loss'][-1]).rjust(self.pad),
                "test success_rate": str(prev_success_rate).rjust(self.pad),
                "best success_rate": str(prev_best_rate).rjust(self.pad)
            })

        self.log_data(episode_data)
        self.generate_plots()
        # save Q-Table
        self.Q.save(f"{self.policy_path}/q_network.h5")


# 14. Double DQN with Experience Replay
class DDQN_ER(DDQN):
    """
    Double Deep Q-Learning Network with Experience Replay
    """
    def __init__(self, env=None, episodes=int(), decay_style="sigmoid", alpha=0.001, gamma=0.9, epsilon=1, policy_path=None, test_episodes=100, batch_size=64, testing=False, use_conv=False, verbose=0):
        super().__init__(env, episodes, decay_style, alpha, gamma, epsilon, policy_path, test_episodes, batch_size, testing, use_conv, verbose)
        # Initialize memory size
        self.memory_size = int(self.episodes/100)
        # Initialize replay memory
        self.replay_memory = deque(maxlen=self.memory_size)
        # Initialize batch size
        self.batch_size = batch_size
        # Initialize minimum memory size to start training
        self.min_memory_size = int(self.memory_size/100)
    
    def record_experience(self, data):
        """
        Record experience
        """
        # Add experience to replay memory
        self.replay_memory.append(data)
    
    def sample_experience(self):
        """
        Sample experience
        """
        # Sample experience from replay memory
        return random.sample(self.replay_memory, self.batch_size)

    def train(self):
        """
        Train agent
        """
        # Initialize data collection
        episode_rewards = []
        episode_data = []

        # Initialize success rate
        prev_success_rate = 0
        prev_best_rate = 0

        for episode in tqdm(range(1, self.episodes+1), ascii=True, unit="episode"):
            # Reset Environment
            observation, done = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            # update epsilon    
            self.epsilon_decay(episode)
            while not done:
                # take a step
                action = self.get_epsilon_greedy_action(observation)
                new_observation, reward, done, _ = self.env.step(action, observation)

                # update data collection
                episode_reward += reward
                episode_steps += 1

                # update experience replay
                data = [observation,
                         action,
                         reward,
                         new_observation,
                         done]
                # record experience
                self.record_experience(data)
                if len(self.replay_memory) >= self.min_memory_size:    
                    # sample experience
                    data = self.sample_experience()
                    # update Q-Network
                    history = self.update_q_network(self.Q, self.target_Q, data)
                # update observation
                observation = new_observation

            # update data collection
            episode_rewards.append(episode_reward)

            # update target network
            self.target_update_counter += 1
            if self.target_update_counter == self.target_update_freq:
                self.target_Q.set_weights(self.Q.get_weights())
                self.target_update_counter = 0

            # update success rate every 100 episodes and save Q-Network
            if episode % self.test_every == 0:
                # save Q-Network
                self.Q.save(f"{self.policy_path}/q_network.h5")
                success_rate = self.test()
                prev_success_rate = success_rate

                if success_rate >= prev_best_rate:
                    prev_best_rate = success_rate
                    # save best Q-Network
                    self.Q.save(f"{self.policy_path}/best_q_network.h5")
                    # Clear Session and unload model to avoid memory leaks
                    K.clear_session()
                    del self.Q
                    # reload best Q-Network
                    self.Q = tf.keras.models.load_model(f"{self.policy_path}/best_q_network.h5")
                    self.Q.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), metrics=['accuracy'])

            # log data
            episode_data.append({
                "episode number": str(episode).rjust(self.pad),
                "episode reward": str(episode_reward).rjust(self.pad),
                "average reward": str(np.mean(episode_rewards[-100:])).rjust(self.pad),
                "num of steps": str(episode_steps).rjust(self.pad),
                "epsilon": str(self.epsilon).rjust(self.pad),
                "model accuracy": str(history.history['accuracy'][-1]).rjust(self.pad),
                "model loss": str(history.history['loss'][-1]).rjust(self.pad),
                "test success_rate": str(prev_success_rate).rjust(self.pad),
                "best success_rate": str(prev_best_rate).rjust(self.pad)
            })

        self.log_data(episode_data)
        self.generate_plots()
        # save Q-Table
        self.Q.save(f"{self.policy_path}/q_network.h5")