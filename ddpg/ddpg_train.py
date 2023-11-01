import os
import gymnasium as gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import Agent

#TODO fix random noise

SEED = 0
env = gym.make('BipedalWalker-v3', render_mode='rgb_array')#'human')
_, _ = env.reset(seed = SEED)
agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=SEED)

os.makedirs('out/ddpg', exist_ok = True)
ACTOR_PATH = 'out/ddpg/checkpoint_actor.pth'
CRITIC_PATH = 'out/ddpg/checkpoint_critic.pth'

def ddpg(n_episodes=2000, max_t=700):
    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        state, info = env.reset(seed = SEED)
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated #EDIT for later version of gym
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), ACTOR_PATH)
            torch.save(agent.critic_local.state_dict(), CRITIC_PATH)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
    return scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('out/ddpg/ddpg_train.png')  


# agent.actor_local.load_state_dict(torch.load(ACTOR_PATH))
# agent.critic_local.load_state_dict(torch.load(CRITIC_PATH))


#See the agent interact
# state = env.reset()
# agent.reset()   
# while True:
#     action = agent.act(state)
#     env.render()
#     next_state, reward, terminated, truncated, _ = env.step(action)
#     done = terminated or truncated
#     state = next_state
#     if done:
#         break
        
env.close()