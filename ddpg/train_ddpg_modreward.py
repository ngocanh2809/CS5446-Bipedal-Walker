'''
Hyper params: ddpg_BipedalWalker-v3, these params were not automatically tuned
'''

import os
import gymnasium as gym
import numpy as np

from mod_reward import RunFasterWrapper, NoIdleWrapper
from stable_baselines3 import DDPG
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure


SEED = 4260429117
outfolder = 'out/ddpg/ez_noidle_runfaster'
os.makedirs(outfolder, exist_ok = True)

#Build env
record = True
hardcore = False
if record:
    env = gym.make('BipedalWalker-v3', hardcore = hardcore, render_mode='rgb_array')#'human')
    env = gym.wrappers.RecordVideo(env, video_folder=f'{outfolder}/video', episode_trigger = lambda x: x % 50 == 0) #Saving every n = 1 episode
    eval_env = gym.make('BipedalWalker-v3', hardcore = hardcore, render_mode='rgb_array')#'human')
else:
    env = gym.make('BipedalWalker-v3', hardcore = hardcore, render_mode='human')#'human')
    eval_env = gym.make('BipedalWalker-v3', hardcore = hardcore, render_mode='human')#'human')

#Wrapping with modified reward environment
wrappers = ['no_idle', 'run_faster']
if 'no_idle' in wrappers:
    env = NoIdleWrapper(env=env)
if 'run_faster' in wrappers:
    env = RunFasterWrapper(env=env)

_, _ = env.reset(seed = SEED)

#Build model
model = DDPG("MlpPolicy", env, 
             verbose=1, 
             buffer_size = 200000,
             train_freq=(1, 'episode'),
             gamma=0.98,           
            #  nb_eval_steps = 10000,
             policy_kwargs=dict(net_arch=[400, 300]),
             learning_rate=0.0001,  
             learning_starts=10000,             
             tensorboard_log=f"./{outfolder}/tensorboard/")

n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

#Logging
new_logger = configure(f'{outfolder}/log', ["csv", 'json', 'stdout', 'tensorboard'])
model.set_logger(new_logger)


# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path=f'{outfolder}/best_weights',
                             log_path="./logs/", eval_freq=25000,
                             deterministic=True, render=False)

# Create a CheckpointCallback to save the model at specified intervals
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=f'{outfolder}/weights', name_prefix="model")
#10000

#Train
model.learn(total_timesteps=1000000, callback=[eval_callback, checkpoint_callback], progress_bar=True)

env.close()
eval_env.close()