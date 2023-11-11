from pprint import pprint
import gymnasium as gym
from stable_baselines3 import DDPG
from mod_reward import NoIdleWrapper, RunFasterWrapper

SEED = 0

def random_speed_yield(env):
    action = env.action_space.sample()
    # action[0], action[1] = random.randint(-1, 1), random.randint(-1, 1)
    return action

def run(
        alg = 'ddpg',
        weight_path= 'out/ddpg/ez_lowerLR/best_weights/best_model', #without the .zip
        hardcore = False,
        n_trials = 1, 
        success_reward = 200, 
        save_video = False,  
        wrappers = ["no_idle", 'run_faster']      
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
    
    if alg == 'ddpg':
        model = DDPG.load(weight_path)
    else:
        raise NotImplementedError

    if save_video:
        env = gym.make("BipedalWalker-v3", hardcore = hardcore, render_mode="rgb_array")
        #Record video    
        env = gym.wrappers.RecordVideo(env, video_folder='out', name_prefix=trial_idx, episode_trigger = lambda x: x % 1 == 0) #Saving every n = 1 episode

    else:
        env = gym.make("BipedalWalker-v3", hardcore = hardcore, render_mode="human")

    #Wrapping with modified reward environment
    if 'no_idle' in wrappers:
        env = NoIdleWrapper(env=env)
    if 'run_faster' in wrappers:
        env = RunFasterWrapper(env=env)

    env.action_space.seed(SEED)
    observations, info = env.reset(seed = SEED)
    states = None
    step = 0
    for episode in range(n_trials):
        done  = False
        speed = []
        while not done: #Time step
            
            #Randomly sample [a, b, c, d] rotational speed values for joints, within the range [-1, 1]    
            if alg == 'random':
                action = random_speed_yield(env=env) #Plug policy here
            elif alg in ['ddpg']:
                action, states = model.predict(
                    observations,  # type: ignore[arg-type]
                    state=states,
                    deterministic=True,
                )

            observations, reward, terminated, truncated, info = env.step(action)
            
            if step % 20 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in action]))
                print(f"step {step} total_reward {records[trial_idx]['acc_rewards']:+0.2f}")
                print("hull " + str([f"{x:+0.2f}" for x in observations[0:4]]))
                print("leg0 " + str([f"{x:+0.2f}" for x in observations[4:9]]))
                print("leg1 " + str([f"{x:+0.2f}" for x in observations[9:14]]))
                print('shaping', env.prev_shaping)
                print('reward ', reward)
                print('speed  ', env.hull.linearVelocity.x)
                # print(env.hull.linearVelocity.x, abs(env.hull.linearVelocity.x) < 1e-05)

                speed.append(env.hull.linearVelocity.x)
                

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
            step += 1

            if terminated or truncated: #Terminated: Fall down, Truncated?
                observations, info  = env.reset()
                records[trial_idx]['avg_speed'] = sum(speed)/len(speed)

                if records[trial_idx]['acc_rewards']>success_reward:
                    records[trial_idx]['pass'] = True
                    wins += 1
                else: 
                    records[trial_idx]['pass'] = False
                    fails += 1
                    
                #Init new trial
                trial_idx += 1
                records[trial_idx] = {
                    'acc_rewards': 0,
                    'timesteps': 0
                }

                done = True

    env.close()
    
    success_rate = wins/(wins+ fails + 0.0000001)

    return wins, success_rate, records

if __name__ == '__main__':
    wins, succ_rate, records = run(
        save_video=False,
        weight_path='out/ddpg/ez_lowerLR/weights/model_250000_steps'
        )

    print('wins: ', wins)
    print('success_rates: ', succ_rate)
    pprint(records)