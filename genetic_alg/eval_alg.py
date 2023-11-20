from evolved_nn import Agent, load_evolved_Agent, run_trial
import matplotlib.pyplot as plt
import os
import pickle
import json
import argparse
from statistics import stdev
import gymnasium as gym
import glob

def run_trial4eval(env, agent, n_episodes = 3):
    totals = []

    for _ in range(n_episodes):
        state, info = env.reset()
        total = 0
        done = False
        #while not done:
        # for timestep in range(timesteps):
        timestep = 0
        cum_reward = 0
        
        while not done:
            # print(timestep)
            
            # start_time = time.time()
            state, reward, terminated, truncated, _ = env.step(agent.act(state))
            done = terminated or truncated

            # end_time = time.time()
            # print('time_1-trial: ', end_time-start_time)
            total += reward
            if done:
                break

            cum_reward += reward
            # print(timestep, cum_reward, reward)
            timestep += 1
            

        totals.append(total)

    mean_reward = sum(totals)/len(totals)
    std_reward = stdev(totals)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    
    eval_dict = {
        'model': 'evo_alg',
        'weight_path': weight_path,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'hardcore': hardcore
    }

    return eval_dict

def eval_evoalg(weight_path, save_video, hardcore, output_path,  noviz_noshow = False):
    agent = load_evolved_Agent(weight_path)

    os.makedirs(output_path, exist_ok=True)

    if save_video:
        env = gym.make("BipedalWalker-v3", hardcore = hardcore, render_mode="rgb_array")
        #Record video    
        env = gym.wrappers.RecordVideo(env, video_folder=output_path, name_prefix='best', episode_trigger = lambda x: x % 1 == 0) #Saving every n = 1 episode
    elif noviz_noshow:
        env = gym.make("BipedalWalker-v3", hardcore = hardcore, render_mode="rgb_array")
    else:
        env = gym.make("BipedalWalker-v3", hardcore = hardcore, render_mode="human")
    
    eval_dict = run_trial4eval(env, agent)

    with open(os.path.join(output_path, 'metrics_eval.json'), 'w') as json_file:
        json.dump(eval_dict, json_file, indent=4)
    
    return eval_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process a folder.")
    parser.add_argument("weight_path", help="Path to the input weight.")
    parser.add_argument('--mode', metavar=None, help="Either eval (eval 1 model), viz (output video for 1 model), if left empty: mass_eval (eval a sorted list of models in a directory) or mass_viz (mass visualization) for a folder full of ddpg weights")
    parser.add_argument("--hardcore", action="store_true", help="Hardcore level.")
    parser.add_argument("--output", help="Path to output metrics.", metavar = 'out')
    
    args = parser.parse_args()
    hardcore = args.hardcore
    outfolder = args.output
    mode = args.mode 
    weight_path = args.weight_path
    
    if mode == 'eval':
        eval_dict = eval_evoalg(
            weight_path=weight_path,
            hardcore=hardcore,
            save_video=False,
            output_path=outfolder
            )
        
    elif mode == 'viz':
        eval_dict = eval_evoalg(
            weight_path=weight_path,
            hardcore=hardcore,
            save_video=True,
            output_path=outfolder
            )

    else:
        eval_dict = eval_evoalg(
            weight_path=weight_path,
            hardcore=hardcore,
            save_video=False,
            output_path=outfolder,
            noviz_noshow=True
            )