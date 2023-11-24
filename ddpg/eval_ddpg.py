from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import argparse
from pprint import pprint
import json
import os

# import sys
# sys.path.append('GymnasiumMod')
# import GymnasiumMod.gymnasium as gym

def eval_ddpg(weight_path, save_video, hardcore, output_path):
    model = DDPG.load(weight_path)
    os.makedirs(output_path, exist_ok=True)

    if save_video:
        env = gym.make("BipedalWalker-v3", hardcore = hardcore, render_mode="rgb_array")
        #Record video    
        env = gym.wrappers.RecordVideo(env, video_folder=output_path, name_prefix='best', episode_trigger = lambda x: x % 1 == 0) #Saving every n = 1 episode

    else:
        env = gym.make("BipedalWalker-v3", hardcore = hardcore, render_mode="human")
    
    # Evaluate the policy for a specific number of episodes
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=2)
    env.close()
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    
    eval_dict = {
        'model': 'DDPG',
        'weight_path': weight_path,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'hardcore': hardcore
    }
    
    pprint(eval_dict)

    with open(os.path.join(output_path, 'metrics_eval.json'), 'w') as json_file:
        json.dump(eval_dict, json_file, indent=4)
    
    return eval_dict

if __name__ == '__main__':

    # Load a pre-trained policy (change the path to your model file)
    weight_path = 'out/ddpg/ez/weights/best_model' #no .zip 


    parser = argparse.ArgumentParser(description="Process a folder.")
    parser.add_argument("weight_path", help="Path to the input weight.")
    parser.add_argument('--mode', help="Either eval (eval 1 model), viz (output video for 1 model), mass_eval (eval a sorted list of models in a directory) or mass_viz (mass visualization) for a folder full of ddpg weights")
    parser.add_argument("--hardcore", action="store_true", help="Hardcore level.")
    parser.add_argument("--output", help="Path to output metrics.", metavar = 'out')
    
    args = parser.parse_args()
    hardcore = args.hardcore
    outfolder = args.output
    mode = args.mode 
    weight_path = args.weight_path
    
    if mode == 'eval':
        eval_dict = eval_ddpg(
            weight_path=weight_path,
            hardcore=hardcore,
            save_video=False,
            output_path=outfolder
            )
        
    elif mode == 'viz':
        eval_dict = eval_ddpg(
            weight_path=weight_path,
            hardcore=hardcore,
            save_video=True,
            output_path=outfolder
            )
    else:
        raise NotImplementedError(f"Mode {mode} not yet implemented")
        