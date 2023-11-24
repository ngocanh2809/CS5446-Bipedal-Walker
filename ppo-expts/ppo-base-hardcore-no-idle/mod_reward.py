import gymnasium as gym
import numpy as np
from bipedal_walker import MOTORS_TORQUE

#If walker idle for more than 50 timesteps, it be and get negative -100 points, not including moving backwards
class NoIdleWrapper(gym.Wrapper):
    def __init__(self, env, timestep_idle = 50, vel_threshold = 1e-5):
        super().__init__(env)   
        self.hull_vel_horizontal_record = []
        self.timestep_idle = timestep_idle
        self.vel_threshold = vel_threshold

    def step(self, action):
        observations, reward, terminated, truncated, info = self.env.step(action)
        self.hull_vel_horizontal_record.append(self.env.hull.linearVelocity.x)
        
        if len(self.hull_vel_horizontal_record) > self.timestep_idle:
            if all(abs(x) < self.vel_threshold for x in self.hull_vel_horizontal_record[-self.timestep_idle:]):
                reward = -100
                terminated = True
                # print("WE QUIT")

        # print('im here',self.hull_vel_horizontal_record[-self.timestep_idle:])
        return observations, reward, terminated, truncated, info    
    

#If walker runs slower compared to speed_threshold, it gets punished, and likewise for faster than speed threshold
class RunFasterWrapper(gym.Wrapper):
    def __init__(self, env, 
                speed_threshold = 4.2, #Calculated with best weights
                ratio = 1.0,
                 ):
        super().__init__(env)   
        self.speed_threshold = speed_threshold
        self.ratio_to_torque_punishment = ratio #Ratio compared to punishment for torque movement

    def step(self, action):
        observations, reward, terminated, truncated, info = self.env.step(action)

        motor_punishment = 0 
        for a in action:
            motor_punishment += 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1) #See bipedal_walker implementation in gymnasium
        
        # print('motor_punishment: ', motor_punishment, reward, motor_punishment/reward )        
        if self.env.hull.linearVelocity.x < self.speed_threshold:
            scaling = (self.env.hull.linearVelocity.x - self.speed_threshold)/self.speed_threshold 
        else:
            scaling = self.env.hull.linearVelocity.x / self.speed_threshold

        reward += scaling * self.ratio_to_torque_punishment * motor_punishment

        # print('motor_punishment: ', motor_punishment, reward, scaling * self.ratio_to_torque_punishment * motor_punishment)
        # print(self.env.hull.linearVelocity.x, self.speed_max, scaling)
        return observations, reward, terminated, truncated, info    
    

#If walker don't keep their height higher compared to height threshold, it gets punished, and likewise for shorter than height threshold
class JumpHigherWrapper(gym.Wrapper):
    def __init__(self, env, 
                height_threshold = 5.5, #Calculated with best weights
                ratio = 1.0,
                height_scaling = 10,
                 ):
        super().__init__(env)   
        self.height_threshold = height_threshold
        self.ratio_to_torque_punishment = ratio #Ratio compared to punishment for torque movement
        self.height_scaling = height_scaling #SCALING when height is less then height threshold, must be tuned with height_threshold

    def step(self, action):
        observations, reward, terminated, truncated, info = self.env.step(action)

        motor_punishment = 0 
        for a in action:
            motor_punishment += 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1) #See bipedal_walker implementation in gymnasium
        
        if self.env.hull.position.y < self.height_threshold:
            scaling = (self.env.hull.position.y - self.height_threshold)/self.height_threshold  * self.height_scaling
        else:
            scaling = self.env.hull.position.y / self.height_threshold 
        # print(scaling, self.env.hull.position.y)

        reward += scaling * self.ratio_to_torque_punishment * motor_punishment

        # print('motor_punishment: ', motor_punishment, reward, scaling * self.ratio_to_torque_punishment * motor_punishment)
        # print(self.env.hull.linearVelocity.x, self.speed_max, scaling)
        return observations, reward, terminated, truncated, info    


#Keep 1 leg up for some reward. No threshold
class NoLeg0ContactWrapper(gym.Wrapper):
    def __init__(self, env, 
                ratio = 1.25,
                contact_scaling = .25,
                 ):
        super().__init__(env)   
        self.ratio_to_torque_punishment = ratio #Ratio compared to punishment for torque movement
        self.contact_scaling = contact_scaling

    def step(self, action):
        observations, reward, terminated, truncated, info = self.env.step(action)

        motor_punishment = 0 
        for a in action:
            motor_punishment += 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1) #See bipedal_walker implementation in gymnasium

        contact0 = observations[8] #Boolean Contact of leg 0 (the lighter color one)

        abs_cur_reward = self.ratio_to_torque_punishment * motor_punishment
        if contact0: #Touching ground
            reward -= abs_cur_reward
        elif not contact0: #Not touching ground, reward by 1/4 of abs_cur_reward
            reward += abs_cur_reward * self.contact_scaling

        # print('motor_punishment: ', motor_punishment, reward, abs_cur_reward)

        return observations, reward, terminated, truncated, info    
    