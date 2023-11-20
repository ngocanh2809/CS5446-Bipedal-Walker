'''
Hyper params: ddpg_BipedalWalker-v3, these params were not automatically tuned
'''

import os
import gymnasium as gym
import numpy as np
import argparse

from mod_reward import *
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure


parser = argparse.ArgumentParser(description="Train DDPG")
parser.add_argument("outfolder", default= 'out/ddpg/test', help="Path to the output weights folder.")
parser.add_argument("--hardcore", action="store_true", help="Hardcore level.")
parser.add_argument("--record", action="store_true", help="Record video")
parser.add_argument('--mod', nargs='+', default=['no_idle', 'run_faster'], type=str, help='Input list of modified reward wrappers, has to be a combination of no_idle, run_faster, jump_higher, no_leg_contact')

args = parser.parse_args()
print(args)
outfolder = args.outfolder
record = args.record
hardcore = args.hardcore
wrappers = args.mod

SEED = 4260429117
os.makedirs(outfolder, exist_ok = True)


if record:
    env = gym.make('BipedalWalker-v3', hardcore = hardcore, render_mode='rgb_array')#'human')
    env = gym.wrappers.RecordVideo(env, video_folder=f'{outfolder}/video', episode_trigger = lambda x: x % 50 == 0) #Saving every n = 1 episode
    eval_env = gym.make('BipedalWalker-v3', hardcore = hardcore, render_mode='rgb_array')#'human')
else:
    env = gym.make('BipedalWalker-v3', hardcore = hardcore, render_mode='human')#'human')
    eval_env = gym.make('BipedalWalker-v3', hardcore = hardcore, render_mode='human')#'human')

#Wrapping with modified reward environment
# wrappers = ['no_idle', 'run_faster']

#Wrapping with modified reward environment
if 'no_idle' in wrappers:
    env = NoIdleWrapper(env=env)
if 'run_faster' in wrappers:
    env = RunFasterWrapper(env=env)
if 'jump_higher' in wrappers:
    env = JumpHigherWrapper(env=env)
if 'no_leg_contact' in wrappers:
    env = NoLeg0ContactWrapper(env=env)

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