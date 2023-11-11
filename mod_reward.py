import gym
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
    

#If walker idle for more than 50 timesteps, it stops and get negative -100 points
class RunFasterWrapper(gym.Wrapper):
    def __init__(self, env, 
                speed_max = 4.2, #Calculated with best weights
                ratio = 1.0,
                 ):
        super().__init__(env)   
        self.speed_max = speed_max
        self.ratio_to_torque_punishment = ratio #Ratio compared to punishment for torque movement

    def step(self, action):
        observations, reward, terminated, truncated, info = self.env.step(action)

        motor_punishment = 0 
        for a in action:
            motor_punishment += 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1) #See bipedal_walker implementation in gymnasium
        
        # print('motor_punishment: ', motor_punishment, reward, motor_punishment/reward )        
        if self.env.hull.linearVelocity.x < self.speed_max:
            scaling = (self.env.hull.linearVelocity.x - self.speed_max)/self.speed_max 
        else:
            scaling = self.env.hull.linearVelocity.x / self.speed_max

        reward += scaling * self.ratio_to_torque_punishment * motor_punishment

        # print('motor_punishment: ', motor_punishment, reward, scaling * self.ratio_to_torque_punishment * motor_punishment)
        # print(self.env.hull.linearVelocity.x, self.speed_max, scaling)
        return observations, reward, terminated, truncated, info    