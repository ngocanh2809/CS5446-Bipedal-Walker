from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

# Load a pre-trained policy (change the path to your model file)
weight_path = 'out/ddpg/ez/weights/model_100_steps' #no .zip 
model = DDPG.load(weight_path)

# Create an evaluation environment (change to match your environment)
env = gym.make('BipedalWalker-v3', hardcore = False, render_mode='human')#'human')

# Evaluate the policy for a specific number of episodes
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=2)

print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
