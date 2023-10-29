import gymnasium as gym

def speed_yield(env):
    action = env.action_space.sample()
    # action[0], action[1] = random.randint(-1, 1), random.randint(-1, 1)
    return action

max_time_steps = 1000
if __name__ == '__main__':
    env = gym.make("BipedalWalker-v3", hardcore = True, render_mode="human")

    observation, info = env.reset(seed = 0)

    for _ in range(max_time_steps): #Time step
        
        #Randomly sample [a, b, c, d] rotational speed values for joints, within the range [-1, 1]    
        action = speed_yield(env=env) #Plug policy here

        observation, reward, terminated, truncated, info = env.step(action)

        # State (observation) consists of 
        # hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints 
        # and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements. 
        # There are no coordinates in the state vector.
        #https://www.gymlibrary.dev/environments/box2d/bipedal_walker/

        
        
        #Reward is given for moving forward, totaling 300+ points up to the far end.
        # If the robot falls, it gets -100. 
        # Applying motor torque costs a small amount of points.
        #  A more optimal agent will get a better score.

        if terminated or truncated: #Terminated: Fall down, Truncated?
            observation, info = env.reset()
            print('ENDGAME:',terminated, truncated)
            raise KeyboardInterrupt

    env.close()