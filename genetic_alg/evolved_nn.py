#Inspired by: https://github.com/wfleshman/Evolving_To_Walk/blob/master/walk.py#L57

# Bro still used python2
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 14:10:37 2017 and Oct 30 2023

@author: bill (and me)
"""
import os
import pickle
import gymnasium as gym
import numpy as np
from copy import deepcopy
import tqdm
import time
import pprint
import json
import argparse

SEED = 0
def glorot_uniform(n_inputs,n_outputs,multiplier=1.0):
    ''' Glorot uniform initialization '''
    glorot = multiplier*np.sqrt(6.0/(n_inputs+n_outputs))
    return np.random.uniform(-glorot,glorot,size=(n_inputs,n_outputs))

def softmax(scores,temp=5.0):
    ''' transforms scores to probabilites '''
    exp = np.exp(np.array(scores)/temp)
    return exp/exp.sum()

class Agent(object):
    ''' A Neural Network '''
    
    def __init__(self, n_inputs, n_hidden, n_outputs, mutate_rate=.05, init_multiplier=1.0):
        ''' Create agent's brain '''
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.mutate_rate = mutate_rate
        self.init_multiplier = init_multiplier
        self.network = {'Layer 1' : glorot_uniform(n_inputs, n_hidden,init_multiplier),
                        'Bias 1'  : np.zeros((1,n_hidden)),
                        'Layer 2' : glorot_uniform(n_hidden, n_outputs,init_multiplier),
                        'Bias 2'  : np.zeros((1,n_outputs))}
                        
    def act(self, state):
        ''' Use the network to decide on an action '''   
        if(state.shape[0] != 1):
            state = state.reshape(1,-1)
        net = self.network
        layer_one = np.tanh(np.matmul(state,net['Layer 1']) + net['Bias 1'])
        layer_two = np.tanh(np.matmul(layer_one, net['Layer 2']) + net['Bias 2'])
        return layer_two[0]
    
    def __add__(self, another):
        ''' overloads the + operator for breeding '''
        child = Agent(self.n_inputs, self.n_hidden, self.n_outputs, self.mutate_rate, self.init_multiplier)
        for key in child.network:
            n_inputs,n_outputs = child.network[key].shape
            mask = np.random.choice([0,1],size=child.network[key].shape,p=[.5,.5])
            random = glorot_uniform(mask.shape[0],mask.shape[1])
            child.network[key] = np.where(mask==1,self.network[key],another.network[key])
            mask = np.random.choice([0,1],size=child.network[key].shape,p=[1-self.mutate_rate,self.mutate_rate])
            child.network[key] = np.where(mask==1,child.network[key]+random,child.network[key])
        return child
    
    def save_weight(self, outfile):
        with open(outfile, 'wb') as pickle_file:
            pickle.dump(self.network, pickle_file)

    def load_weight(self, loaded_weight):
        self.network = loaded_weight

def load_evolved_Agent(
        infile, 
        mutate_rate=.05, 
        init_multiplier=1.0):
    
    with open(infile, 'rb') as pickle_file:
        loaded_data = pickle.load(pickle_file)
    
    agent =  Agent(
        n_inputs=loaded_data['Layer 1'].shape[0],
        n_hidden=loaded_data['Layer 1'].shape[1],
        n_outputs=loaded_data['Layer 2'].shape[1],
        mutate_rate=mutate_rate,
        init_multiplier=init_multiplier
    )

    agent.network = loaded_data
    return agent

def run_trial(env,agent,verbose=True, timesteps = 500):
    ''' an agent performs 3 episodes of the env '''
    totals = []
    for _ in range(3):
        state, info = env.reset(seed = SEED)
        if verbose: env.render()
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
            if verbose: env.render()
            total += reward
            if done:
                break

            cum_reward += reward
            # print(timestep, cum_reward, reward)
            timestep += 1
            

        totals.append(total)
    return sum(totals)/3.0

def next_generation(env,population,scores,temperature):
    ''' breeds a new generation of agents '''
    scores, population =  zip(*sorted(zip(scores,population),reverse=True)) #Sort population by score, descending
    children = list(population[:int(len(population)/4)]) #Keep 1/4* pop_size best agents
    parents = list(np.random.choice(population,size=2*(len(population)-len(children)),p=softmax(scores,temperature))) #Choose randomly from 75% of the population, more weight to the ones with higher score
    children = children + [parents[i]+parents[i+1] for i in range(0,len(parents)-1,2)] #Populate the rest as the next generation
    scores = [run_trial(env,agent) for agent in children]

    return children,scores

def main():


    parser = argparse.ArgumentParser(description="Draw graph of mean reward across generations ")
    parser.add_argument("outfolder", default= 'out/genetic_alg/evolved_nn/weights', help="Path to the output weights folder.")
    parser.add_argument("--hardcore", action="store_true", help="Hardcore level.")
    parser.add_argument("--gen", default= 40, help="Number of generations")
    parser.add_argument("--pop_size", default= 48, help="Population size each generation")

    args = parser.parse_args()
    print(args)
    weight_path = args.outfolder
    n_generations = args.gen
    pop_size = args.pop_size

    ''' main function '''
    # Setup environment
    env = gym.make('BipedalWalker-v3', hardcore = args.hardcore, render_mode='rgb_array')#'human')
    observation, info = env.reset(seed = SEED)

    np.random.seed(0)
    
    # network params
    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    n_hidden = 512
    multiplier = 5 # For weight initialization

    # Population params
    pop_size = 48 #50 #50 different robots with different neural networks for brains
    mutate_rate = .1
    softmax_temp = 5.0

    # Training
    n_generations = 60 #40 for non-hardcore
    # Initiate all pop_size number of brains
    population = [Agent(n_inputs,n_hidden,n_actions,mutate_rate,multiplier) for i in range(pop_size)]
    

    # Run trials and gather scores for all agents
    print('First trial runs:')
    scores = []
    for index, agent in enumerate(population):
        # print(index)
        cur_score = run_trial(env,agent)
        print(f"Trail run: 0, agent {index}| avg_score: {cur_score} ")
        scores.append(cur_score)
        # end_time = time.time()
        # print('time_1-agent: ', end_time-start_time)
        
    # Get best weights that produce the best scores 
    best = [deepcopy(population[np.argmax(scores)])]


    os.makedirs(weight_path, exist_ok=True)
    best[0].save_weight(os.path.join(weight_path, 'gen_0_' + str(round(max(scores), 2)) + '.pkl'))
    
    print('Breeding and mutating across generations:')
    for generation in tqdm.tqdm(range(n_generations)):
        population,scores = next_generation(env,population, scores,softmax_temp)
        best_agent = deepcopy(population[np.argmax(scores)]) 
        best_agent.save_weight(os.path.join(weight_path, 'gen_'+ str(generation) + '_' + str(round(max(scores), 2)) +'.pkl'))
    
        best.append(best_agent)
        print("Generation:",generation,"Score:",np.max(scores))

    for index, agent in enumerate(best):
        run_trial(env,agent)

    env.close()
    
if __name__ == '__main__':
    main()
