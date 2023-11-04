'''
Hyper params: ddpg_BipedalWalker-v3, these params were not automatically tuned
'''

import os
import gymnasium as gym
import numpy as np
import torch
# from callback_utils import VideoRecorderCallback

from stable_baselines3 import DDPG
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure


SEED = 4260429117
outfolder = 'out/ddpg/ez'
os.makedirs(outfolder, exist_ok = True)
save_freq = 100 #int num timesteps, saves video and log model

#Build env
record = False
hardcore = False
if record:
    env = gym.make('BipedalWalker-v3', hardcore = hardcore, render_mode='rgb_array')#'human')
    env = gym.wrappers.RecordVideo(env, video_folder=f'{outfolder}/video', episode_trigger = lambda x: x % 50 == 0) #Saving every n = 1 episode
else:
    env = gym.make('BipedalWalker-v3', hardcore = hardcore, render_mode='human')#'human')

_, _ = env.reset(seed = SEED)
print(torch.cuda.is_available())
#Build model
model = DDPG("MlpPolicy", env, 
             verbose=1, 
             buffer_size = 200000,
             train_freq=(1, 'episode'),
             gamma=0.98,           
            #  nb_eval_steps = 10000,
             policy_kwargs=dict(net_arch=[400, 300]),
             learning_rate=0.001,  
             learning_starts=10000,             
             tensorboard_log=f"./{outfolder}/tensorboard/", device="cuda")

n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

#Logging
new_logger = configure(f'{outfolder}/log', ["csv", 'json', 'stdout'])
model.set_logger(new_logger)

# Create a CheckpointCallback to save the model at specified intervals
checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=f'{outfolder}/weights', name_prefix="model")
#10000

#Train
model.learn(total_timesteps=1000000, callback=[checkpoint_callback], progress_bar=True)

env.close()