import gymnasium as gym
from pprint import pprint
import moviepy

def speed_yield(env):
    action = env.action_space.sample()
    # action[0], action[1] = random.randint(-1, 1), random.randint(-1, 1)
    return action

def run(
        harcore = True,
        timesteps = 1000, 
        success_reward = 300, 
        max_time_steps_per_trial = 200, 
        save_video = False
        
        ):
    #Keep track of success rate
    fails = 0
    wins = 0

    #Keep track of trials
    records = {}
    trial_idx = 0
    records[trial_idx] = {
        'acc_rewards': 0,
        'timesteps': 0
    }

    if save_video:
        env = gym.make("BipedalWalker-v3", hardcore = harcore, render_mode="rgb_array")
        #Record video    
        env = gym.wrappers.RecordVideo(env, video_folder='out', name_prefix=trial_idx, episode_trigger = lambda x: x % 1 == 0) #Saving every n = 1 episode
    else:
        env = gym.make("BipedalWalker-v3", hardcore = harcore, render_mode="human")

    observation, info = env.reset(seed = 0)

    for _ in range(timesteps): #Time step
        
        #Randomly sample [a, b, c, d] rotational speed values for joints, within the range [-1, 1]    
        action = speed_yield(env=env) #Plug policy here

        observation, reward, terminated, truncated, info = env.step(action)
        # print(reward)
        # State (observation) consists of 
        # hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints 
        # and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements. 
        # There are no coordinates in the state vector.
        #https://www.gymlibrary.dev/environments/box2d/bipedal_walker/

        
        #Reward is given for moving forward, totaling 300+ points up to the far end.
        # If the robot falls, it gets -100. 
        # Applying motor torque costs a small amount of points.
        # A more optimal agent will get a better score.
        
        records[trial_idx]['acc_rewards'] += reward
        records[trial_idx]['timesteps'] += 1

        if terminated or truncated or records[trial_idx]['timesteps'] == max_time_steps_per_trial: #Terminated: Fall down, Truncated?
            observation, info = env.reset()
            
            #Conclude old failed trial
            fails += 1
            records[trial_idx]['pass'] = 0
            if terminated:
                records[trial_idx]['cause_of_death'] = 'fell'
            elif records[trial_idx]['timesteps'] == max_time_steps_per_trial:
                records[trial_idx]['cause_of_death'] = 'timeout'
            elif truncated: 
                #"whether a truncation condition outside the scope of the MDP is satisfied. 
                # Typically a timelimit, but could also be used to indicate agent physically going out of bounds. 
                # Can be used to end the episode prematurely before a terminal state is reached."
                records[trial_idx]['cause_of_death'] = ' ¯\_(ツ)_/¯'

            #Init new trial
            trial_idx += 1
            records[trial_idx] = {
                'acc_rewards': 0,
                'timesteps': 0
            }


        if reward >= success_reward: #Successfully gets to the other side
            observation, info = env.reset()
            wins += 1
            
            records[trial_idx]['cause_of_death'] = ''
            records[trial_idx]['pass'] = 1
            
            #New trial
            trial_idx += 1
            records[trial_idx] = {
                'acc_rewards': 0,
                'timesteps': 0
            }

    env.close()
    
    success_rate = wins/(wins+ fails + 0.0000001)

    return wins, success_rate, records

if __name__ == '__main__':
    wins, succ_rate, records = run(save_video=False)

    print('wins: ', wins)
    print('success_rates: ', succ_rate)
    pprint(records)